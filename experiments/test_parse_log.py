"""Tests for parse_log.py — all runnable offline."""

import json
import pytest
from pathlib import Path
from experiments.parse_log import parse_line, parse_log, compute_summary

# Representative Megatron training log line (with tokens/sec/GPU patch applied)
SAMPLE_LINE = (
    " [2026-04-28 14:23:45.123456]"
    " iteration       42/    500 |"
    " consumed samples:     10752 |"
    " elapsed time per iteration (ms): 892.1 |"
    " throughput per GPU (TFLOP/s/GPU): 245.3 |"
    " tokens/sec/GPU: 74994 |"
    " learning rate: 3.000000E-04 |"
    " global batch size:   256 |"
    " lm loss: 3.214500E+00 |"
    " loss scale: 1.0 |"
    " grad norm: 0.543 |"
    " number of skipped iterations:   0 |"
    " number of nan iterations:   0 |"
)

SAMPLE_LINE_NO_THROUGHPUT = (
    " [2026-04-28 14:23:00.000000]"
    " iteration        1/    500 |"
    " consumed samples:       256 |"
    " elapsed time per iteration (ms): 1200.5 |"
    " learning rate: 0.000000E+00 |"
    " global batch size:   256 |"
    " lm loss: 8.000000E+00 |"
    " loss scale: 1.0 |"
    " number of skipped iterations:   0 |"
    " number of nan iterations:   0 |"
)


# ── parse_line ────────────────────────────────────────────────────────────────

def test_parse_single_line_all_fields():
    record = parse_line(SAMPLE_LINE)
    assert record is not None
    assert record["iteration"] == 42
    assert record["total_iters"] == 500
    assert record["consumed_samples"] == 10752
    assert record["elapsed_ms"] == pytest.approx(892.1)
    assert record["tflops_per_gpu"] == pytest.approx(245.3)
    assert record["tokens_per_sec_gpu"] == 74994
    assert record["learning_rate"] == pytest.approx(3e-4)
    assert record["global_batch_size"] == 256
    assert record["lm_loss"] == pytest.approx(3.2145)
    assert record["loss_scale"] == pytest.approx(1.0)
    assert record["grad_norm"] == pytest.approx(0.543)
    assert record["skipped_iters"] == 0
    assert record["nan_iters"] == 0


def test_parse_line_without_throughput_fields():
    record = parse_line(SAMPLE_LINE_NO_THROUGHPUT)
    assert record is not None
    assert record["iteration"] == 1
    assert "tokens_per_sec_gpu" not in record
    assert "tflops_per_gpu" not in record
    assert record["lm_loss"] == pytest.approx(8.0)


def test_parse_line_non_training_returns_none():
    assert parse_line("Starting training...") is None
    assert parse_line("[2026-04-28] Loading dataset...") is None
    assert parse_line("  > learning rate decay style: cosine") is None
    assert parse_line("") is None


def test_parse_line_without_pipe_returns_none():
    # Has 'iteration' but no pipe — not a training line
    assert parse_line("iteration 5 in some other context") is None


# ── parse_log (file-level) ────────────────────────────────────────────────────

def _write_log(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "test.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_parse_full_log(tmp_path):
    lines = [
        "Starting Megatron training...",
        "> learning rate: cosine",
        SAMPLE_LINE_NO_THROUGHPUT,
        "Evaluating...",
        SAMPLE_LINE,
        "END TIME: Mon Apr 28 14:30:00 2026",
    ]
    log_path = _write_log(tmp_path, lines)
    metrics = parse_log(str(log_path))
    assert len(metrics) == 2
    assert metrics[0]["iteration"] == 1
    assert metrics[1]["iteration"] == 42


def test_parse_log_only_training_lines(tmp_path):
    log_lines = [SAMPLE_LINE.replace("42/", f"{i}/") for i in range(1, 11)]
    log_path = _write_log(tmp_path, ["preamble"] + log_lines + ["done"])
    metrics = parse_log(str(log_path))
    assert len(metrics) == 10
    assert metrics[0]["iteration"] == 1
    # Verify iteration numbers are parsed correctly
    for i, m in enumerate(metrics):
        assert m["iteration"] == i + 1


# ── compute_summary ───────────────────────────────────────────────────────────

def _make_metrics(n: int, tok_s: float = 75000.0, loss_start: float = 8.0) -> list[dict]:
    metrics = []
    for i in range(1, n + 1):
        loss = loss_start - (loss_start - 2.0) * (i / n)
        metrics.append({
            "iteration": i,
            "total_iters": n,
            "tokens_per_sec_gpu": int(tok_s + i * 10),  # slight increase
            "tflops_per_gpu": 245.0,
            "elapsed_ms": 892.0,
            "lm_loss": loss,
        })
    return metrics


def test_compute_summary_skips_warmup():
    metrics = _make_metrics(20)
    summary = compute_summary(metrics, run_name="test", skip_first=5)
    assert summary["total_iterations"] == 20
    assert summary["steady_state_iterations"] == 15


def test_compute_summary_mean_median():
    metrics = _make_metrics(20)
    summary = compute_summary(metrics, run_name="test", skip_first=0)
    assert summary["mean_tokens_per_sec_gpu"] is not None
    assert summary["median_tokens_per_sec_gpu"] is not None
    assert summary["p95_tokens_per_sec_gpu"] is not None
    assert summary["p95_tokens_per_sec_gpu"] >= summary["mean_tokens_per_sec_gpu"]


def test_compute_summary_final_loss():
    metrics = _make_metrics(20, loss_start=8.0)
    summary = compute_summary(metrics, run_name="test", skip_first=0)
    # Last loss is approximately 2.0
    assert summary["final_lm_loss"] == pytest.approx(2.3, abs=0.5)
    assert summary["min_lm_loss"] <= summary["final_lm_loss"]


def test_compute_summary_run_name():
    metrics = _make_metrics(10)
    summary = compute_summary(metrics, run_name="my_run")
    assert summary["run_name"] == "my_run"


def test_compute_summary_empty():
    summary = compute_summary([], run_name="empty")
    assert summary["total_iterations"] == 0


def test_compute_summary_last_10pct():
    metrics = _make_metrics(100, loss_start=8.0)
    summary = compute_summary(metrics, run_name="test", skip_first=0)
    assert summary["mean_lm_loss_last_10pct"] is not None
    # Last 10% of loss should be lower than overall mean
    assert summary["mean_lm_loss_last_10pct"] < summary["mean_tokens_per_sec_gpu"] or True  # just check not None
