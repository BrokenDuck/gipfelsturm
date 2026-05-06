"""Tests for run_ablation.py — all runnable offline."""

import pytest
from pathlib import Path
from experiments.run_ablation import (
    load_ablation_plan,
    resolve_model_config,
    build_precision_args,
    build_attention_args,
    build_distributed_args,
    build_prec_aware_opt_args,
    render_sbatch,
    MODEL_CONFIGS,
)

REPO_ROOT = Path(__file__).parent.parent
ABLATION_CSV = REPO_ROOT / "experiments" / "ablation_plan.csv"


# ── CSV loading ──────────────────────────────────────────────────────────────

def test_load_ablation_plan():
    plan = load_ablation_plan(str(ABLATION_CSV))
    assert len(plan) >= 1
    first = plan[0]
    assert "run_name" in first
    assert "model_size" in first
    assert "precision" in first


def test_load_ablation_plan_strips_whitespace():
    plan = load_ablation_plan(str(ABLATION_CSV))
    for row in plan:
        for k, v in row.items():
            assert k == k.strip(), f"Key has whitespace: {repr(k)}"
            assert v == v.strip(), f"Value has whitespace in {k}: {repr(v)}"


# ── Model config ─────────────────────────────────────────────────────────────

def test_resolve_model_config_all_sizes():
    expected = {
        "125m": dict(num_layers=12,  hidden=768,   ffn=2048,  heads=12, kv_heads=4,  default_mbs=16),
        "350m": dict(num_layers=24,  hidden=1024,  ffn=2816,  heads=16, kv_heads=4,  default_mbs=8),
        "760m": dict(num_layers=24,  hidden=1536,  ffn=4096,  heads=16, kv_heads=4,  default_mbs=4),
        "1.5b": dict(num_layers=48,  hidden=1600,  ffn=4352,  heads=20, kv_heads=4,  default_mbs=4),
        "3b":   dict(num_layers=32,  hidden=3072,  ffn=8192,  heads=24, kv_heads=8,  default_mbs=4),
        "8b":   dict(num_layers=32,  hidden=4096,  ffn=14336, heads=32, kv_heads=8,  default_mbs=2),
    }
    for size, cfg in expected.items():
        assert resolve_model_config(size) == cfg, f"Mismatch for {size}"


def test_resolve_model_config_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model size"):
        resolve_model_config("999b")


# ── Precision args ────────────────────────────────────────────────────────────

def test_build_precision_args_bf16():
    assert build_precision_args("bf16") == ["--bf16"]


def test_build_precision_args_fp16():
    assert build_precision_args("fp16") == ["--fp16"]


def test_build_precision_args_fp32():
    args = build_precision_args("fp32")
    assert "--recompute-granularity" in args
    assert "--bf16" not in args
    assert "--fp16" not in args


def test_build_precision_args_fp8():
    args = build_precision_args("fp8")
    assert "--fp8-format" in args
    assert "hybrid" in args
    assert "--fp8-recipe" in args
    assert "delayed" in args


def test_build_precision_args_fp8_hybrid():
    args = build_precision_args("fp8_hybrid")
    assert "--fp8-format" in args
    assert "hybrid" in args
    assert "--fp8-recipe" in args
    assert "delayed" in args


def test_build_precision_args_fp8_e4m3():
    args = build_precision_args("fp8_e4m3")
    assert "--fp8-format" in args
    assert "e4m3" in args
    assert "--fp8-recipe" in args
    assert "delayed" in args


def test_build_precision_args_unknown_raises():
    with pytest.raises(ValueError, match="Unknown precision"):
        build_precision_args("int8")


# ── Precision-aware optimizer args ───────────────────────────────────────────

def test_build_prec_aware_opt_args_none():
    assert build_prec_aware_opt_args("none") == []


def test_build_prec_aware_opt_args_empty():
    assert build_prec_aware_opt_args("") == []


def test_build_prec_aware_opt_args_bf16_fp32():
    args = build_prec_aware_opt_args("bf16_fp32")
    assert "--use-precision-aware-optimizer" in args
    assert "--main-grads-dtype" in args
    assert "bf16" in args
    assert "--main-params-dtype" in args
    assert "fp32" in args


def test_build_prec_aware_opt_args_fp32_fp16():
    args = build_prec_aware_opt_args("fp32_fp16")
    assert "--use-precision-aware-optimizer" in args
    assert "fp32" in args
    assert "fp16" in args


def test_build_prec_aware_opt_args_all_combinations():
    for combo in ("fp32_fp32", "bf16_fp32", "fp32_fp16", "bf16_fp16"):
        args = build_prec_aware_opt_args(combo)
        assert "--use-precision-aware-optimizer" in args


def test_build_prec_aware_opt_args_unknown_raises():
    with pytest.raises(ValueError, match="Unknown prec_aware_opt"):
        build_prec_aware_opt_args("fp16_bf16")


# ── Attention args ────────────────────────────────────────────────────────────

def test_build_attention_args_default():
    assert build_attention_args("default") == []


def test_build_attention_args_cudnn():
    assert build_attention_args("cuDNN") == []


def test_build_attention_args_fa3():
    args = build_attention_args("fa3")
    assert args == ["--attention-backend", "flash"]


def test_build_attention_args_fa2():
    args = build_attention_args("fa2")
    assert args == ["--attention-backend", "flash"]


def test_build_attention_args_local():
    args = build_attention_args("local")
    assert "--attention-backend" in args
    assert "local" in args


def test_build_attention_args_tbd():
    assert build_attention_args("TBD") == []


def test_build_attention_args_unknown_raises():
    with pytest.raises(ValueError, match="Unknown attention_backend"):
        build_attention_args("xformers")


# ── Distributed args ──────────────────────────────────────────────────────────

def test_build_distributed_args_defaults():
    args = build_distributed_args("1", "1")
    assert "--tensor-model-parallel-size" in args
    idx = args.index("--tensor-model-parallel-size")
    assert args[idx + 1] == "1"
    assert "--pipeline-model-parallel-size" in args
    assert "--sequence-parallel" not in args


def test_build_distributed_args_tp4():
    args = build_distributed_args("4", "1")
    idx = args.index("--tensor-model-parallel-size")
    assert args[idx + 1] == "4"
    assert "--sequence-parallel" in args


def test_build_distributed_args_tp4_pp4():
    args = build_distributed_args("4", "4")
    assert args[args.index("--tensor-model-parallel-size") + 1] == "4"
    assert args[args.index("--pipeline-model-parallel-size") + 1] == "4"


def test_build_distributed_args_tbd():
    # TBD values should fall back to defaults (TP=1, PP=1)
    args = build_distributed_args("TBD", "TBD")
    assert args[args.index("--tensor-model-parallel-size") + 1] == "1"
    assert "--sequence-parallel" not in args


# ── render_sbatch ─────────────────────────────────────────────────────────────

def _make_run(**overrides):
    base = {
        "run_name": "test_760m",
        "model_size": "760m",
        "nodes": "1",
        "gpus": "4",
        "tp": "1",
        "pp": "1",
        "dp": "4",
        "seq_len": "4096",
        "micro_batch": "4",
        "global_batch": "256",
        "precision": "bf16",
        "prec_aware_opt": "none",
        "attention_backend": "default",
        "mode": "throughput",
        "target_minutes": "30",
        "target_steps": "50",
        "expected_tok_s_gpu": "TBD",
        "status": "planned",
        "notes": "test",
    }
    base.update(overrides)
    return base


def test_render_sbatch_has_sbatch_directives():
    run = _make_run()
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node=4" in script
    assert "#SBATCH --cpus-per-task=288" in script
    assert "#SBATCH --mem=850000" in script
    assert "#SBATCH --no-requeue" in script


def test_render_sbatch_has_env_setup():
    run = _make_run()
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "WORKDIR=/users/$USER/gipfelsturm" in script
    assert "CUDA_DEVICE_MAX_CONNECTIONS=1" in script


def test_render_sbatch_bf16_precision():
    run = _make_run(precision="bf16")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--bf16" in script
    assert "--fp16" not in script
    assert "--fp8-format" not in script


def test_render_sbatch_fp8_precision():
    run = _make_run(precision="fp8")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--fp8-format" in script
    assert "hybrid" in script  # e4m3 fwd, e5m2 bwd
    assert "--bf16" in script
    assert "--attention-backend" not in script


def test_render_sbatch_fa3_attention():
    run = _make_run(attention_backend="fa3")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--attention-backend flash" in script


def test_render_sbatch_default_attention_no_flag():
    run = _make_run(attention_backend="default")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--attention-backend" not in script


def test_render_sbatch_unfused_attention_via_fusion_opts():
    run = _make_run(fusion_opts="-attention")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--attention-backend unfused" in script


def test_render_sbatch_tp4_pp1():
    run = _make_run(nodes="1", tp="4", pp="1")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--tensor-model-parallel-size 4" in script
    assert "--pipeline-model-parallel-size 1" in script
    assert "--sequence-parallel" in script


def test_render_sbatch_network_size():
    run = _make_run()
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--num-layers 24" in script
    assert "--hidden-size 1536" in script
    assert "--ffn-hidden-size 4096" in script


def test_render_sbatch_has_srun():
    run = _make_run()
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "srun" in script
    assert "pretrain_gpt.py" in script


def test_render_sbatch_throughput_disables_wandb():
    run = _make_run(mode="throughput")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "WANDB_MODE=disabled" in script


def test_render_sbatch_train_mode_enables_wandb_block():
    run = _make_run(mode="train", target_steps="1000", target_minutes="30")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "WANDB_API_KEY" in script
    assert "--tensorboard-dir" in script


def test_render_sbatch_custom_mbs_gbs():
    run = _make_run(micro_batch="8", global_batch="512")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "MBS=8" in script
    assert "GBS=512" in script


def test_render_sbatch_job_name():
    run = _make_run(run_name="my_ablation")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "#SBATCH --job-name=gipfel-my_ablation" in script


def test_render_sbatch_train_mode_timing_flags():
    run = _make_run(mode="train", target_steps="1000", target_minutes="30")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--timing-log-level 1" in script
    assert "--timing-log-option minmax" in script


def test_render_sbatch_no_disable_bias_linear():
    run = _make_run()
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--disable-bias-linear" not in script


def test_render_sbatch_prec_aware_opt_off_by_default():
    run = _make_run()
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--use-precision-aware-optimizer" not in script


def test_render_sbatch_prec_aware_opt_bf16_fp32():
    run = _make_run(prec_aware_opt="bf16_fp32")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--use-precision-aware-optimizer" in script
    assert "--main-grads-dtype bf16" in script
    assert "--main-params-dtype fp32" in script


# ── dry-run end-to-end ────────────────────────────────────────────────────────

def test_dry_run_generates_file(tmp_path):
    """End-to-end: run_ablation --dry-run should write .sbatch files."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "run_ablation.py"),
         "--dry-run",
         "--output-dir", str(tmp_path),
         "--run", "baseline_760m_4n_bf16"],
        capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    files = list(tmp_path.glob("*.sbatch"))
    assert len(files) == 1
    content = files[0].read_text()
    assert "#SBATCH" in content
    assert "pretrain_gpt.py" in content
