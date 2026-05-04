"""Tests for run_ablation.py — all runnable offline."""

import pytest
from pathlib import Path
from experiments.run_ablation import (
    load_ablation_plan,
    resolve_model_config,
    build_precision_args,
    build_attention_args,
    build_kernel_args,
    build_distributed_args,
    render_sbatch,
    MODEL_CONFIGS,
    KERNEL_PRESETS,
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


def test_build_precision_args_fp8():
    args = build_precision_args("fp8")
    assert "--fp8-format" in args
    assert "hybrid" in args  # e4m3 fwd, e5m2 bwd — recommended for training
    assert "--fp8-recipe" in args
    assert "delayed" in args


def test_build_precision_args_unknown_raises():
    with pytest.raises(ValueError, match="Unknown precision"):
        build_precision_args("int8")


# ── Attention args ────────────────────────────────────────────────────────────

def test_build_attention_args_default():
    assert build_attention_args("default") == []


def test_build_attention_args_flash():
    args = build_attention_args("flash")
    assert "--attention-backend" in args
    assert "flash" in args


def test_build_attention_args_tbd():
    assert build_attention_args("TBD") == []


def test_build_attention_args_unknown_raises():
    with pytest.raises(ValueError, match="Unknown attention_backend"):
        build_attention_args("xformers")


# ── Kernel args ───────────────────────────────────────────────────────────────

def test_build_kernel_args_none():
    assert build_kernel_args("none") == []


def test_build_kernel_args_tbd():
    assert build_kernel_args("TBD") == []


def test_build_kernel_args_all_fused():
    args = build_kernel_args("all_fused")
    assert "--bias-activation-fusion" in args


def test_build_kernel_args_cuda_graphs_attn():
    args = build_kernel_args("cuda_graphs_attn")
    assert "--use-te-rng-tracker" in args
    assert "--cuda-graph-impl" in args
    assert "transformer_engine" in args
    assert "--cuda-graph-scope" in args
    assert "attn" in args


def test_build_kernel_args_cuda_graphs_local():
    args = build_kernel_args("cuda_graphs_local")
    assert "--use-te-rng-tracker" in args
    assert "--cuda-graph-impl" in args
    assert "local" in args


def test_build_kernel_args_unknown_raises():
    with pytest.raises(ValueError, match="Unknown kernel_opts"):
        build_kernel_args("magic_kernel")


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
        "attention_backend": "default",
        "kernel_opts": "none",
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
    assert "#SBATCH --mem=460000" in script
    assert "#SBATCH --no-requeue" in script


def test_render_sbatch_has_env_setup():
    run = _make_run()
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "WORKDIR=/users/$USER/gipfelsturm" in script
    assert "git apply" in script
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
    assert "--attention-backend auto" in script
    assert "NVTE_FLASH_ATTN" not in script
    assert "NVTE_UNFUSED_ATTN" not in script


def test_render_sbatch_flash_attention():
    run = _make_run(attention_backend="flash")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--attention-backend" in script
    assert "flash" in script


def test_render_sbatch_default_attention_no_flag():
    run = _make_run(attention_backend="default")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--attention-backend" not in script


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
    run = _make_run()
    model = resolve_model_config("760m")
    script = render_sbatch(run, model, mode="throughput")
    assert "WANDB_MODE=disabled" in script


def test_render_sbatch_train_mode_enables_wandb_block():
    run = _make_run(target_steps="1000", target_minutes="30")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model, mode="train")
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


def test_render_sbatch_profiling_injects_nvte_env():
    run = _make_run(kernel_opts="profiling")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "NVTE_NVTX_ENABLED=1" in script
    assert "NVTE_DEBUG=1" in script


def test_render_sbatch_non_profiling_no_nvte_env():
    run = _make_run(kernel_opts="none")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "NVTE_NVTX_ENABLED" not in script


def test_render_sbatch_train_mode_timing_flags():
    run = _make_run(target_steps="1000", target_minutes="30")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model, mode="train")
    assert "--timing-log-level 1" in script
    assert "--timing-log-option minmax" in script


def test_render_sbatch_cuda_graphs_attn_has_rng_tracker():
    run = _make_run(kernel_opts="cuda_graphs_attn")
    model = resolve_model_config("760m")
    script = render_sbatch(run, model)
    assert "--use-te-rng-tracker" in script
    assert "--cuda-graph-impl transformer_engine" in script


# ── dry-run end-to-end ────────────────────────────────────────────────────────

def test_dry_run_generates_file(tmp_path):
    """End-to-end: run_ablation --dry-run should write .sbatch files."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "experiments" / "run_ablation.py"),
         "--dry-run",
         "--output-dir", str(tmp_path),
         "--run", "baseline_125m_1n_bf16"],
        capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    files = list(tmp_path.glob("*.sbatch"))
    assert len(files) == 1
    content = files[0].read_text()
    assert "#SBATCH" in content
    assert "pretrain_gpt.py" in content
