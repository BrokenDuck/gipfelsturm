#!/usr/bin/env python3
"""Generate and optionally submit SLURM jobs from ablation_plan.csv.

Usage:
    uv run experiments/run_ablation.py [--csv PATH] [--run NAME...] [--dry-run] [--output-dir PATH] [--mode MODE]
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).parent.parent

# Model architecture configs — mirrors launch.sh case statement
MODEL_CONFIGS = {
    "125m": dict(
        num_layers=12, hidden=768, ffn=2048, heads=12, kv_heads=4, default_mbs=16
    ),
    "350m": dict(
        num_layers=24, hidden=1024, ffn=2816, heads=16, kv_heads=4, default_mbs=8
    ),
    "760m": dict(
        num_layers=24, hidden=1536, ffn=4096, heads=16, kv_heads=4, default_mbs=4
    ),
    "1.5b": dict(
        num_layers=48, hidden=1600, ffn=4352, heads=20, kv_heads=4, default_mbs=4
    ),
    "3b": dict(
        num_layers=32, hidden=3072, ffn=8192, heads=24, kv_heads=8, default_mbs=4
    ),
    "8b": dict(
        num_layers=32, hidden=4096, ffn=14336, heads=32, kv_heads=8, default_mbs=2
    ),
    "13b": dict(
        num_layers=40,
        hidden=5120,
        ffn=17920,
        heads=40,
        kv_heads=8,
        default_mbs=1,
    ),
}

# cuda_graph column values -> extra Megatron CLI flags
# "full" requires --no-check-for-nan-in-loss-and-grad; render_sbatch handles this automatically.
CUDA_GRAPH_PRESETS = {
    "none": [],
    # Iteration graph captured by MCore's local backend
    "full": [
        "--cuda-graph-impl",
        "local",
        "--te-rng-tracker",
        "--cuda-graph-warmup-steps",
        "3",
    ],
    # Full-iteration graph captured by MCore's local backend
    "full_iter": [
        "--cuda-graph-impl",
        "local",
        "--te-rng-tracker",
        "--cuda-graph-scope",
        "full_iteration",
        "--cuda-graph-warmup-steps",
        "3",
    ],
    # TE-scoped graphs capturing attention + MLP blocks
    "te": [
        "--cuda-graph-impl",
        "transformer_engine",
        "--te-rng-tracker",
        "--cuda-graph-scope",
        "attn",
        "mlp",
    ],
}

# fusion_opts column: comma-separated "+name" / "-name" tokens.
# Each entry: (enable_flags, disable_flags). disable_flags may be None if Megatron lacks a negation.
FUSION_FLAGS: dict[str, tuple[list[str], list[str] | None]] = {
    "attention": (["--attention-backend", "fused"], ["--attention-backend", "unfused"]),
    "gelu": (["--bias-gelu-fusion"], ["--no-bias-gelu-fusion"]),
    "swiglu": (["--bias-swiglu-fusion"], ["--no-bias-swiglu-fusion"]),
    "dropout": (["--bias-dropout-fusion"], ["--no-bias-dropout-fusion"]),
    "softmax": (["--masked-softmax-fusion"], ["--no-masked-softmax-fusion"]),
    "rope": (["--apply-rope-fusion"], ["--no-rope-fusion"]),
    "cross_entropy": (
        ["--cross-entropy-loss-fusion"],
        ["--no-cross-entropy-loss-fusion"],
    ),
    "grad_accum": (
        ["--gradient-accumulation-fusion"],
        ["--no-gradient-accumulation-fusion"],
    ),
}


def load_ablation_plan(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        # skipinitialspace so quoted cells work even when preceded by "key , value"
        reader = csv.DictReader(f, skipinitialspace=True)
        return [
            {k.strip(): v.strip() for k, v in row.items() if k is not None}
            for row in reader
        ]


def resolve_model_config(model_size: str) -> dict:
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size '{model_size}'. Choose: {', '.join(MODEL_CONFIGS)}"
        )
    return MODEL_CONFIGS[model_size]


def _tbd_or(value: str, default):
    """Return default if value is TBD/empty, else return value."""
    return default if value.upper() in ("TBD", "") else value


_FP8_BASE_FLAGS = ["--fp8-format", "hybrid"]

_FP8_RECIPE_FLAGS = {
    "delayed": [
        "--fp8-recipe", "delayed",
        "--fp8-amax-history-len", "1024",
        "--fp8-amax-compute-algo", "max",
        "--fp8-margin", "0",
        "--fp8-param-gather",
    ],
    "current": [
        "--fp8-recipe", "tensorwise",
        "--first-last-layers-bf16",
        "--num-layers-at-start-in-bf16", "1",
        "--num-layers-at-end-in-bf16", "1",
        "--fp8-param-gather",
    ],
    "subchannel": [
        "--fp8-recipe", "blockwise",
    ],
}


def build_precision_args(precision: str) -> list[str]:
    p = precision.lower()
    if p == "fp32":
        # Try to reduce memory footprint more aggressively
        return [
            "--recompute-granularity",
            "full",
            "--recompute-method",
            "block",
            "--recompute-num-layers",
            "4",
        ]
    elif p == "bf16":
        return ["--bf16"]
    elif p == "fp16":
        return ["--fp16"]
    elif p.endswith(("_fp8_delayed", "_fp8_current", "_fp8_subchannel")):
        base, _, recipe = p.rpartition("_fp8_")
        if base not in ("bf16", "fp16"):
            raise ValueError(
                f"Unknown precision '{precision}'. "
                f"Choose: fp32, fp16, bf16, "
                f"bf16_fp8_delayed, bf16_fp8_current, bf16_fp8_subchannel, "
                f"fp16_fp8_delayed, fp16_fp8_current, fp16_fp8_subchannel"
            )
        prec_flag = f"--{base}"
        return [prec_flag] + _FP8_BASE_FLAGS + _FP8_RECIPE_FLAGS[recipe]
    else:
        raise ValueError(
            f"Unknown precision '{precision}'. "
            f"Choose: fp32, fp16, bf16, "
            f"bf16_fp8_delayed, bf16_fp8_current, bf16_fp8_subchannel, "
            f"fp16_fp8_delayed, fp16_fp8_current, fp16_fp8_subchannel"
        )


def build_attention_args(attention_backend: str) -> list[str]:
    b = attention_backend.lower()
    # cuDNN fused attention is the default — no flag needed
    if b in ("default", "cudnn", "tbd", ""):
        return []
    elif b == "fa3" or b == "fa2":
        # Fa2 is activated by changing venvs
        return ["--attention-backend", "flash"]
    elif b == "local":
        return [
            "--attention-backend",
            "local",
            "--spec",
            "local",
            # Persist layer norm crashed on local backend
            "--no-persist-layer-norm",
        ]
    else:
        raise ValueError(
            f"Unknown attention_backend '{attention_backend}'. "
            "Choose: default, cuDNN, fa3, fa2, local"
        )


def build_fusion_args(fusion_opts: str) -> list[str]:
    """Parse '+name,-name,...' tokens into Megatron fusion flags.

    Unknown names raise ValueError. Empty / 'none' / 'default' returns [].
    """
    val = fusion_opts.strip().lower()
    if val in ("", "none", "default", "tbd"):
        return []
    args: list[str] = []
    for token in val.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("+"):
            name = token[1:]
            on = True
        elif token.startswith("-"):
            name = token[1:]
            on = False
        else:
            raise ValueError(
                f"fusion_opts token '{token}' must start with '+' (enable) or '-' (disable)"
            )
        if name not in FUSION_FLAGS:
            raise ValueError(
                f"Unknown fusion name '{name}'. Choose: {', '.join(FUSION_FLAGS)}"
            )
        enable_flags, disable_flags = FUSION_FLAGS[name]
        if on:
            args.extend(enable_flags)
        else:
            if disable_flags is None:
                raise ValueError(f"Fusion '{name}' has no disable flag in Megatron")
            args.extend(disable_flags)
    return args


def build_cuda_graph_args(cuda_graph: str) -> list[str]:
    c = cuda_graph.lower()
    if c in ("tbd", ""):
        c = "none"
    if c not in CUDA_GRAPH_PRESETS:
        raise ValueError(
            f"Unknown cuda_graph '{cuda_graph}'. Choose: {', '.join(CUDA_GRAPH_PRESETS)}"
        )
    return CUDA_GRAPH_PRESETS[c]


_PREC_AWARE_OPT_COMBINATIONS = {
    "fp32_fp32": ("fp32", "fp32"),
    "bf16_fp32": ("bf16", "fp32"),
    "fp32_fp16": ("fp32", "fp16"),
    "bf16_fp16": ("bf16", "fp16"),
}


def build_prec_aware_opt_args(prec_aware_opt: str, dp_strat: str = "distopt") -> list[str]:
    """Return precision-aware-optimizer flags, or [] if none/off.

    When dp_strat starts with 'mega', the dtype flags are renamed to the
    Megatron-FSDP variants (--megatron-fsdp-main-{grads,params}-dtype).
    When active, always appends --enable-experimental.
    """
    val = prec_aware_opt.strip().lower()
    if val in ("", "none", "off", "tbd"):
        return []
    if val not in _PREC_AWARE_OPT_COMBINATIONS:
        raise ValueError(
            f"Unknown prec_aware_opt '{prec_aware_opt}'. "
            f"Choose: none, {', '.join(_PREC_AWARE_OPT_COMBINATIONS)}"
        )
    grads_dtype, params_dtype = _PREC_AWARE_OPT_COMBINATIONS[val]
    dp_strat_val = dp_strat.strip().lower()
    if dp_strat_val.startswith("mega"):
        return [
            "--use-precision-aware-optimizer",
            "--megatron-fsdp-main-grads-dtype",
            grads_dtype,
            "--megatron-fsdp-main-params-dtype",
            params_dtype,
            "--enable-experimental",
        ]
    return [
        "--use-precision-aware-optimizer",
        "--main-grads-dtype",
        grads_dtype,
        "--main-params-dtype",
        params_dtype,
        "--enable-experimental",
    ]


_FSDP_SHARDING_STRATEGIES = {
    "z1": "optim",
    "z2": "optim_grads",
    "z3": "optim_grads_params",
}

_MEGA_FSDP_EXTRA_FLAGS = [
    "--calculate-per-token-loss",
    "--init-model-with-meta-device",
    "--grad-reduce-in-bf16",
    "--fsdp-double-buffer",
    "--use-nccl-ub",
]


def build_dp_strat_args(dp_strat: str, pp: str = "1") -> list[str]:
    """Return data-parallelism flags for the given dp_strat column value.

    Values:
      'ddp'      — plain DDP, no distributed optimizer
      'distopt'  — distributed optimizer (--use-distributed-optimizer)
      '<prefix>_<suffix>' — FSDP: prefix 'torch' or 'mega', suffix 'z1'/'z2'/'z3'

    Constraints:
      - torch requires PP=1
      - distopt/ddp emit --data-parallel-sharding-strategy no_shard (explicit default)
      - mega FSDP appends extra required flags
    """
    val = dp_strat.strip().lower()
    if val in ("", "none", "tbd", "no_shard", "distopt"):
        return ["--data-parallel-sharding-strategy", "no_shard"]
    if val == "ddp":
        return ["--data-parallel-sharding-strategy", "no_shard"]

    parts = val.split("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid dp_strat value '{dp_strat}'. Expected 'ddp', 'distopt', or "
            f"'<prefix>_<suffix>' (e.g. torch_z1, mega_z2)."
        )
    prefix, suffix = parts
    if prefix not in ("torch", "mega"):
        raise ValueError(f"Unknown dp_strat prefix '{prefix}'. Choose: torch, mega")
    if suffix not in _FSDP_SHARDING_STRATEGIES:
        raise ValueError(
            f"Unknown dp_strat suffix '{suffix}'. Choose: {', '.join(_FSDP_SHARDING_STRATEGIES)}"
        )
    if prefix == "torch" and int(_tbd_or(pp, "1")) != 1:
        raise ValueError(f"torch FSDP2 requires pipeline_parallel_size=1, got pp={pp}")

    strategy = _FSDP_SHARDING_STRATEGIES[suffix]
    impl_flag = "--use-torch-fsdp2" if prefix == "torch" else "--use-megatron-fsdp"
    args = [impl_flag, "--data-parallel-sharding-strategy", strategy]
    if prefix == "mega":
        args.extend(_MEGA_FSDP_EXTRA_FLAGS)
    return args


def build_distributed_args(tp: str, pp: str, vpp: str = "none") -> list[str]:
    tp_val = int(_tbd_or(tp, "1"))
    pp_val = int(_tbd_or(pp, "1"))
    args = [
        "--tensor-model-parallel-size",
        str(tp_val),
        "--pipeline-model-parallel-size",
        str(pp_val),
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
    ]
    # Enable sequence parallelism when TP > 1
    if tp_val > 1:
        args.append("--sequence-parallel")
    vpp_norm = _tbd_or(vpp, "none").strip().lower()
    if vpp_norm not in ("", "none"):
        args += ["--num-layers-per-virtual-pipeline-stage", vpp_norm]
    return args


def render_sbatch(run: dict, model: dict) -> str:
    run_name = run["run_name"]
    mode = _tbd_or(run.get("mode", "throughput"), "throughput").lower()
    nodes = int(_tbd_or(run["nodes"], "4"))

    mbs = int(_tbd_or(run["micro_batch"], str(model["default_mbs"])))
    gbs = int(_tbd_or(run["global_batch"], "256"))
    seq_len = int(_tbd_or(run["seq_len"], "4096"))
    target_steps = _tbd_or(run["target_steps"], "50")
    target_minutes = _tbd_or(run["target_minutes"], "30")

    if mode == "throughput":
        training_steps = int(target_steps) if target_steps != "TBD" else 100
        slurm_time = "00:30:00"
        eval_interval = 100000
        eval_iters = 5
        lr_warmup_iters = training_steps // 2
        logging_extra = ""
        wandb_block = "export WANDB_MODE=disabled"
        no_nan_check = "\n    --no-check-for-nan-in-loss-and-grad"
    else:  # train
        no_nan_check = ""
        training_steps = int(target_steps) if target_steps != "TBD" else 1000
        minutes = float(target_minutes)
        # Add buffer for SLURM overhead
        h = int(minutes // 60) + 1
        m = int(minutes % 60) + 30
        if m >= 60:
            h += 1
            m -= 60
        slurm_time = f"{h:02d}:{m:02d}:00"
        eval_interval = 25
        eval_iters = 5
        lr_warmup_iters = 50
        logging_extra = (
            "\n    --tensorboard-dir $TENSORBOARD_DIR"
            "\n    --log-timers-to-tensorboard"
            "\n    --log-memory-to-tensorboard"
            "\n    --timing-log-level 1"
            "\n    --timing-log-option minmax"
        )
        wandb_block = dedent("""\
            # WANDB
            if [ -n "$WANDB_API_KEY" ]; then
                echo "[$(date)] WANDB enabled."
                SCRIPT_ARGS="$SCRIPT_ARGS \\
                    --wandb-save-dir $LOG_DIR \\
                    --wandb-project $PROJECT_NAME \\
                    --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
            else
                export WANDB_MODE=disabled
                echo "[$(date)] WANDB disabled."
            fi""")

    dp_strat_val = _tbd_or(run.get("dp_strat", "distopt"), "distopt")
    dp_strat_norm = dp_strat_val.strip().lower()
    fsdp_active = dp_strat_norm not in ("", "none", "tbd", "no_shard", "distopt", "ddp")
    if dp_strat_norm.startswith("torch") and str(
        run.get("prec_aware_opt", "none")
    ).strip().lower() not in ("", "none", "off", "tbd"):
        raise ValueError(
            f"run '{run['run_name']}': torch FSDP2 is incompatible with prec_aware_opt "
            f"(got prec_aware_opt={run.get('prec_aware_opt')})"
        )
    prec_aware_opt_args = build_prec_aware_opt_args(
        run.get("prec_aware_opt", "none"), dp_strat=dp_strat_val
    )
    dp_strat_args = build_dp_strat_args(dp_strat_val, pp=run.get("pp", "1"))
    precision_args = build_precision_args(_tbd_or(run["precision"], "bf16"))
    precision_val = _tbd_or(run["precision"], "bf16").lower()
    attention_backend_val = _tbd_or(run["attention_backend"], "default").lower()
    attention_args = build_attention_args(attention_backend_val)
    cuda_graph_val = _tbd_or(run.get("cuda_graph", "none"), "none").lower()
    cuda_graph_args = build_cuda_graph_args(cuda_graph_val)
    # full-iteration CUDA graphs require NaN check to be disabled
    if cuda_graph_val == "full":
        no_nan_check = "\n    --no-check-for-nan-in-loss-and-grad"
    fusion_args = build_fusion_args(run.get("fusion_opts", ""))
    # cross_entropy: hardcoded default on; disabled only if explicitly negated via fusion_opts
    if "--no-cross-entropy-loss-fusion" in fusion_args:
        ce_fusion_line = ""
        fusion_args = [
            a for a in fusion_args if a not in ("--no-cross-entropy-loss-fusion",)
        ]
    else:
        ce_fusion_line = "\n    --cross-entropy-loss-fusion"
        fusion_args = [
            a for a in fusion_args if a not in ("--cross-entropy-loss-fusion",)
        ]
    distributed_args = build_distributed_args(run["tp"], run["pp"], run.get("vpp", "none")) + dp_strat_args
    if dp_strat_norm == "ddp":
        _distopt_flags = {
            "--use-distributed-optimizer",
            "--overlap-grad-reduce",
            "--overlap-param-gather",
        }
        distributed_args = [
            a for a in distributed_args if a not in _distopt_flags
        ]
    if dp_strat_norm.startswith("torch"):
        _torch_fsdp_incompatible = {
            "--use-distributed-optimizer",
            "--overlap-grad-reduce",
            "--overlap-param-gather",
        }
        distributed_args = [
            a for a in distributed_args if a not in _torch_fsdp_incompatible
        ]
    if dp_strat_norm.startswith("mega"):
        distributed_args += ["--ckpt-format", "fsdp_dtensor"]

    venv_name = (
        ".venv-gipfelturm-fa2" if attention_backend_val == "fa2" else ".venv-gipfelturm"
    )

    nvte_env_block = ""
    if "_fp8_" in precision_val:
        # FP8 requires TE auto-select; cuDNN is the default so no flag needed
        attention_args = []

    def fmt_args(args: list[str]) -> str:
        if not args:
            return ""
        lines = []
        i = 0
        while i < len(args):
            if not args[i].startswith("--") and i > 0:
                lines[-1] = lines[-1] + " " + args[i]
            else:
                lines.append("    " + args[i])
            i += 1
        return "\n".join(lines)

    prec_aware_opt_inline = (
        "\n" + fmt_args(prec_aware_opt_args) if prec_aware_opt_args else ""
    )

    # Render MIXED_PRECISION_ARGS
    mixed_precision_block = (
        "MIXED_PRECISION_ARGS=(\n" + fmt_args(precision_args) + "\n)"
    )

    # Render extra attention/kernel/fusion args block
    extra_args = attention_args + cuda_graph_args + fusion_args
    extra_block = ""
    if extra_args:
        extra_block = "\nEXTRA_ARGS=(\n" + fmt_args(extra_args) + "\n)\n"

    # DISTRIBUTED_ARGS
    distributed_block = "DISTRIBUTED_ARGS=(\n" + fmt_args(distributed_args) + "\n)"

    # TRAINING_CMD extra args reference
    extra_cmd_ref = "\n    ${EXTRA_ARGS[@]} \\" if extra_args else ""

    job_name = f"gipfel-{run_name}"

    script = f"""\
#!/bin/bash
#SBATCH --account=g34
#SBATCH --time={slurm_time}
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{mode}/%x-%j.out
#SBATCH --error=logs/{mode}/%x-%j.err
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=850000
#SBATCH --no-requeue

echo "START TIME: $(date)"

################ Configs ################
WORKDIR=/users/$USER/gipfelsturm
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/iopsstor/scratch/cscs/$USER/dataset/climbmix_small/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache

# Training config
MBS={mbs}
GBS={gbs}
SEQ_LEN={seq_len}
TRAINING_STEPS={training_steps}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME={run_name}-${{SLURM_NNODES}}n
LOG_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/$PROJECT_NAME/$EXP_NAME
TENSORBOARD_DIR=$LOG_DIR/tensorboard
#########################################

mkdir -p $WORKDIR/logs/{mode} $LOG_DIR $TENSORBOARD_DIR

# Required for TP; must be unset with FSDP (conflicts with FSDP's async comms)
{"# " if fsdp_active else ""}export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE ))
# Try to reduce our memory footprint a bit: https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html#techniques-for-reducing-memory-to-avoid-memory-overflow-and-enhance-training-efficiency
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# export NCCL_NVLS_ENABLE=0
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000)){nvte_env_block}

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine{prec_aware_opt_inline}
)

NETWORK_SIZE_ARGS=(
    --num-layers {model["num_layers"]}
    --hidden-size {model["hidden"]}
    --ffn-hidden-size {model["ffn"]}
    --num-attention-heads {model["heads"]}
    --group-query-attention
    --num-query-groups {model["kv_heads"]}
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length $SEQ_LEN
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $TRAINING_STEPS
    --log-interval 1
    --eval-interval {eval_interval}
    --eval-iters {eval_iters}{ce_fusion_line}
    --optimizer adam
    --dataloader-type single{no_nan_check}
    --manual-gc
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --lr-warmup-iters {lr_warmup_iters}
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

{mixed_precision_block}

{distributed_block}
{extra_block}
LOGGING_ARGS=(
    --log-throughput
    --log-progress{logging_extra}
)

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 2
)

SCRIPT_ARGS="$MEGATRON_LM_DIR/pretrain_gpt.py \\
    ${{TRANSFORMER_ENGINE_ARGS[@]}} \\
    ${{NETWORK_SIZE_ARGS[@]}} \\
    ${{TRAINING_ARGS[@]}} \\
    ${{REGULARIZATION_ARGS[@]}} \\
    ${{LEARNING_RATE_ARGS[@]}} \\
    ${{INITIALIZATION_ARGS[@]}} \\
    ${{MIXED_PRECISION_ARGS[@]}} \\
    ${{DISTRIBUTED_ARGS[@]}} \\{extra_cmd_ref}
    ${{LOGGING_ARGS[@]}} \\
    ${{TOKENIZER_ARGS[@]}} \\
    ${{DATA_ARGS[@]}}"

{wandb_block}

# Build TRAINING_CMD as a plain string so it survives bash -c expansion safely.
# Bash arrays cannot be exported into srun's bash -c subshell.
TRAINING_CMD="python -m torch.distributed.run \\
    --nproc-per-node $SLURM_GPUS_PER_NODE \\
    --nnodes $SLURM_NNODES \\
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \\
    --rdzv_backend c10d \\
    --max_restarts 0 \\
    --tee 3 \\
    --node-rank \\$SLURM_NODEID \\
    $SCRIPT_ARGS"

echo "TRAINING_CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "
    export JOB_CACHE=/tmp/gipfel-\\$SLURM_JOB_ID-\\$SLURM_NODEID
    export TRITON_CACHE_DIR=\\$JOB_CACHE/triton
    export TORCHINDUCTOR_CACHE_DIR=\\$JOB_CACHE/inductor
    export TORCH_EXTENSIONS_DIR=\\$JOB_CACHE/torch_extensions
    export CUDA_CACHE_PATH=\\$JOB_CACHE/cuda
    mkdir -p \"\\$TRITON_CACHE_DIR\" \"\\$TORCHINDUCTOR_CACHE_DIR\" \"\\$TORCH_EXTENSIONS_DIR\" \"\\$CUDA_CACHE_PATH\"

    source /iopsstor/scratch/cscs/$USER/{venv_name}/bin/activate
    numactl --membind=0-3 $TRAINING_CMD
"

echo "END TIME: $(date)"
"""
    return script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default=str(REPO_ROOT / "experiments" / "ablation_plan.csv")
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        metavar="NAME",
        help="Run only this named ablation (can repeat). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts but do not submit via sbatch.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write .sbatch files (default: logs/<mode>/ per run)",
    )
    args = parser.parse_args()

    plan = load_ablation_plan(args.csv)
    if args.runs:
        names = set(args.runs)
        plan = [r for r in plan if r["run_name"] in names]
        if not plan:
            print(f"No runs matched: {args.runs}", file=sys.stderr)
            sys.exit(1)

    # Write all scripts first
    script_paths = []
    for run in plan:
        name = run["run_name"]
        mode = _tbd_or(run.get("mode", "throughput"), "throughput").lower()
        model = resolve_model_config(run["model_size"])
        script = render_sbatch(run, model)

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = REPO_ROOT / "logs" / mode
        output_dir.mkdir(parents=True, exist_ok=True)

        script_path = output_dir / f"gipfel-{name}.sbatch"
        script_path.write_text(script)
        script_path.chmod(0o755)
        script_paths.append((name, script_path))

    if args.dry_run:
        for name, script_path in script_paths:
            print(f"Generated (dry-run): {script_path}")
        return

    # Submit in batches of 8; job i*BATCH_SIZE+j depends only on job (i-1)*BATCH_SIZE+j
    BATCH_SIZE = 8
    prev_batch_job_ids: list[str] = []
    for i in range(0, len(script_paths), BATCH_SIZE):
        batch = script_paths[i : i + BATCH_SIZE]
        batch_job_ids = []
        for j, (name, script_path) in enumerate(batch):
            sbatch_cmd = ["sbatch"]
            if prev_batch_job_ids and j < len(prev_batch_job_ids):
                sbatch_cmd += [f"--dependency=afterany:{prev_batch_job_ids[j]}"]
            result = subprocess.run(
                sbatch_cmd + [str(script_path)], capture_output=True, text=True
            )
            if result.returncode != 0:
                print(
                    f"Failed to submit {name}: {result.stderr.strip()}", file=sys.stderr
                )
                sys.exit(1)
            job_id = result.stdout.strip().split()[-1]
            dep_str = (
                f" (depends on {prev_batch_job_ids[j]})"
                if prev_batch_job_ids and j < len(prev_batch_job_ids)
                else ""
            )
            print(f"Submitted {name}: {job_id}{dep_str}")
            batch_job_ids.append(job_id)
        prev_batch_job_ids = batch_job_ids


if __name__ == "__main__":
    main()
