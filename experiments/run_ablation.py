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
}

# kernel_opts column values -> extra Megatron CLI flags
KERNEL_PRESETS = {
    "none": [],
    "all_fused": [
        "--bias-activation-fusion",
        "--masked-softmax-fusion",
        "--bias-dropout-fusion",
    ],
    "no_fusion": [
        "--no-rope-fusion",
        "--no-gradient-accumulation-fusion",
        "--no-persist-layer-norm",
    ],
    # transformer_engine is already the default backend — no extra flags needed
    "transformer_engine": [],
    # CUDA graphs require --use-te-rng-tracker; TE-scoped graphs capture attn+mlp
    "cuda_graphs_attn": [
        "--use-te-rng-tracker",
        "--cuda-graph-impl",
        "transformer_engine",
        "--cuda-graph-scope",
        "attn",
    ],
    "cuda_graphs_attn_mlp": [
        "--use-te-rng-tracker",
        "--cuda-graph-impl",
        "transformer_engine",
        "--cuda-graph-scope",
        "attn,mlp",
    ],
    # Full-iteration graph via local MCore capture; requires --no-check-for-nan-in-loss-and-grad
    # (already set in TRAINING_ARGS) and --use-te-rng-tracker
    "cuda_graphs_local": [
        "--use-te-rng-tracker",
        "--cuda-graph-impl",
        "local",
        "--cuda-graph-scope",
        "full_iteration",
    ],
    # Profiling preset: enable NVTE NVTX markers for NSYS traces; no extra Megatron flags
    # Actual env vars (NVTE_NVTX_ENABLED, NVTE_DEBUG) are injected in the SLURM script body
    "profiling": [],
}

# cuda_graph column values -> extra Megatron CLI flags
# "full" requires --no-check-for-nan-in-loss-and-grad; render_sbatch handles this automatically.
CUDA_GRAPH_PRESETS = {
    "none": [],
    # Full-iteration graph captured by MCore's local backend
    "full": [
        "--cuda-graph-impl",
        "local",
        "--te-rng-tracker",
        # "--cuda-graph-scope",
        # "full_iteration",
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
# Each entry: (enable_flag, disable_flag). disable_flag may be None if Megatron lacks a negation.
FUSION_FLAGS: dict[str, tuple[str, str | None]] = {
    "gelu": ("--bias-gelu-fusion", "--no-bias-gelu-fusion"),
    "swiglu": ("--bias-swiglu-fusion", "--no-bias-swiglu-fusion"),
    "dropout": ("--bias-dropout-fusion", "--no-bias-dropout-fusion"),
    "softmax": ("--masked-softmax-fusion", "--no-masked-softmax-fusion"),
    "rope": ("--apply-rope-fusion", "--no-rope-fusion"),
    "cross_entropy": ("--cross-entropy-loss-fusion", "--no-cross-entropy-loss-fusion"),
    "grad_accum": (
        "--gradient-accumulation-fusion",
        "--no-gradient-accumulation-fusion",
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


def build_precision_args(precision: str) -> list[str]:
    p = precision.lower()
    if p == "bf16":
        return ["--bf16"]
    elif p == "fp16":
        return ["--fp16"]
    elif p == "fp8":
        return ["--bf16", "--fp8-format", "hybrid", "--fp8-recipe", "delayed"]
    else:
        raise ValueError(f"Unknown precision '{precision}'. Choose: bf16, fp16, fp8")


def build_attention_args(attention_backend: str) -> list[str]:
    b = attention_backend.lower()
    if b in ("default", "tbd", ""):
        return []
    elif b in ("auto", "fused", "unfused"):
        return ["--attention-backend", b]
    elif b == "local":
        return [
            "--attention-backend",
            "local",
            "--spec",
            "local",
            "--no-persist-layer-norm",
        ]
    elif b in ("flash", "flash_fa3"):
        # FA3 via default venv
        return ["--attention-backend", "flash"]
    elif b == "flash_fa2":
        # FA2 via .venv-gipfelturm-fa2; same Megatron flag as FA3
        return ["--attention-backend", "flash"]
    else:
        raise ValueError(
            f"Unknown attention_backend '{attention_backend}'. "
            "Choose: default, auto, fused, flash, flash_fa3, flash_fa2, unfused, local"
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
        enable_flag, disable_flag = FUSION_FLAGS[name]
        if on:
            args.append(enable_flag)
        else:
            if disable_flag is None:
                raise ValueError(f"Fusion '{name}' has no disable flag in Megatron")
            args.append(disable_flag)
    return args


def build_kernel_args(kernel_opts: str) -> list[str]:
    k = kernel_opts.lower()
    if k in ("tbd", ""):
        k = "none"
    if k not in KERNEL_PRESETS:
        raise ValueError(
            f"Unknown kernel_opts '{kernel_opts}'. Choose: {', '.join(KERNEL_PRESETS)}"
        )
    return KERNEL_PRESETS[k]


def build_cuda_graph_args(cuda_graph: str) -> list[str]:
    c = cuda_graph.lower()
    if c in ("tbd", ""):
        c = "none"
    if c not in CUDA_GRAPH_PRESETS:
        raise ValueError(
            f"Unknown cuda_graph '{cuda_graph}'. Choose: {', '.join(CUDA_GRAPH_PRESETS)}"
        )
    return CUDA_GRAPH_PRESETS[c]


def build_distributed_args(tp: str, pp: str) -> list[str]:
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
    return args


def render_sbatch(run: dict, model: dict, mode: str = "throughput") -> str:
    run_name = run["run_name"]
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

    precision_args = build_precision_args(_tbd_or(run["precision"], "bf16"))
    precision_val = _tbd_or(run["precision"], "bf16").lower()
    attention_backend_val = _tbd_or(run["attention_backend"], "default").lower()
    attention_args = build_attention_args(attention_backend_val)
    kernel_opts_val = _tbd_or(run["kernel_opts"], "none")
    kernel_args = build_kernel_args(kernel_opts_val)
    cuda_graph_val = _tbd_or(run.get("cuda_graph", "none"), "none").lower()
    cuda_graph_args = build_cuda_graph_args(cuda_graph_val)
    # full-iteration CUDA graphs require NaN check to be disabled
    if cuda_graph_val == "full":
        no_nan_check = "\n    --no-check-for-nan-in-loss-and-grad"
    fusion_args = build_fusion_args(run.get("fusion_opts", ""))
    # cross_entropy: hardcoded default on; disabled only if explicitly negated via fusion_opts
    if "--no-cross-entropy-loss-fusion" in fusion_args:
        ce_fusion_line = ""
        fusion_args = [a for a in fusion_args if a != "--no-cross-entropy-loss-fusion"]
    else:
        ce_fusion_line = "\n    --cross-entropy-loss-fusion"
        fusion_args = [a for a in fusion_args if a != "--cross-entropy-loss-fusion"]
    distributed_args = build_distributed_args(run["tp"], run["pp"])

    venv_name = (
        ".venv-gipfelturm-fa2"
        if attention_backend_val == "flash_fa2"
        else ".venv-gipfelturm"
    )

    nvte_env_block = ""
    if precision_val == "fp8":
        attention_args = ["--attention-backend", "auto"]
    # Profiling preset: inject NVTE env vars for NSYS traces
    if kernel_opts_val == "profiling":
        nvte_env_block += (
            "\nexport NVTE_NVTX_ENABLED=1"
            "\nexport NVTE_DEBUG=1"
            "\nexport NVTE_DEBUG_LEVEL=1"
        )

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

    # Render MIXED_PRECISION_ARGS
    mixed_precision_block = (
        "MIXED_PRECISION_ARGS=(\n" + fmt_args(precision_args) + "\n)"
    )

    # Render extra attention/kernel/fusion args block
    extra_args = attention_args + kernel_args + cuda_graph_args + fusion_args
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
#SBATCH --mem=460000
#SBATCH --no-requeue

echo "START TIME: $(date)"

################ Configs ################
WORKDIR=/users/$USER/gipfelsturm
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
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

mkdir -p logs/{mode} $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE ))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000)){nvte_env_block}

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
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
    --disable-bias-linear
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
    --num-workers 8
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
        help="Where to write .sbatch files (default: logs/<mode>/)",
    )
    parser.add_argument("--mode", default="throughput", choices=["throughput", "train"])
    args = parser.parse_args()

    plan = load_ablation_plan(args.csv)
    if args.runs:
        names = set(args.runs)
        plan = [r for r in plan if r["run_name"] in names]
        if not plan:
            print(f"No runs matched: {args.runs}", file=sys.stderr)
            sys.exit(1)

    output_dir = (
        Path(args.output_dir) if args.output_dir else REPO_ROOT / "logs" / args.mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for run in plan:
        name = run["run_name"]
        model = resolve_model_config(run["model_size"])
        script = render_sbatch(run, model, mode=args.mode)

        script_path = output_dir / f"gipfel-{name}.sbatch"
        script_path.write_text(script)
        script_path.chmod(0o755)

        if args.dry_run:
            print(f"Generated (dry-run): {script_path}")
        else:
            result = subprocess.run(
                ["sbatch", str(script_path)], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"Submitted {name}: {result.stdout.strip()}")
            else:
                print(
                    f"Failed to submit {name}: {result.stderr.strip()}", file=sys.stderr
                )
                sys.exit(1)


if __name__ == "__main__":
    main()
