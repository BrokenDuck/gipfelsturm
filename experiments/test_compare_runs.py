"""Tests for compare_runs.py — all runnable offline."""

import json
import pytest
from pathlib import Path
from experiments.compare_runs import (
    load_summaries,
    compute_speedups,
    format_table,
    format_csv,
)


def _write_summary(results_dir: Path, run_name: str, data: dict) -> None:
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(data))


def _make_summary(run_name: str, tok_s: float = 75000.0, loss: float = 3.2) -> dict:
    return {
        "run_name": run_name,
        "total_iterations": 100,
        "steady_state_iterations": 95,
        "mean_tokens_per_sec_gpu": tok_s,
        "median_tokens_per_sec_gpu": tok_s * 0.98,
        "p95_tokens_per_sec_gpu": tok_s * 1.05,
        "mean_tflops_per_gpu": 245.0,
        "median_elapsed_ms": 892.0,
        "final_lm_loss": loss,
        "min_lm_loss": loss - 0.1,
        "mean_lm_loss_last_10pct": loss + 0.05,
    }


# ── load_summaries ────────────────────────────────────────────────────────────

def test_load_summaries_all(tmp_path):
    _write_summary(tmp_path, "run_a", _make_summary("run_a"))
    _write_summary(tmp_path, "run_b", _make_summary("run_b"))
    summaries = load_summaries(str(tmp_path))
    assert len(summaries) == 2
    names = {s["run_name"] for s in summaries}
    assert "run_a" in names
    assert "run_b" in names


def test_load_summaries_filtered(tmp_path):
    _write_summary(tmp_path, "run_a", _make_summary("run_a"))
    _write_summary(tmp_path, "run_b", _make_summary("run_b"))
    _write_summary(tmp_path, "run_c", _make_summary("run_c"))
    summaries = load_summaries(str(tmp_path), run_names=["run_a", "run_c"])
    assert len(summaries) == 2


def test_load_summaries_missing_file_warns(tmp_path, capsys):
    # Create a dir without summary.json
    (tmp_path / "run_empty").mkdir()
    summaries = load_summaries(str(tmp_path))
    assert summaries == []
    captured = capsys.readouterr()
    assert "Warning" in captured.err


# ── compute_speedups ──────────────────────────────────────────────────────────

def test_compute_speedups_relative_to_first():
    summaries = [
        _make_summary("baseline", tok_s=75000.0),
        _make_summary("flash",    tok_s=82500.0),
        _make_summary("fp8",      tok_s=90000.0),
    ]
    result = compute_speedups(summaries)
    assert result[0]["speedup"] == pytest.approx(1.0)
    assert result[1]["speedup"] == pytest.approx(1.1)
    assert result[2]["speedup"] == pytest.approx(1.2)


def test_compute_speedups_named_baseline():
    summaries = [
        _make_summary("fast_run", tok_s=90000.0),
        _make_summary("baseline", tok_s=75000.0),
    ]
    result = compute_speedups(summaries, baseline_name="baseline")
    baseline = next(s for s in result if s["run_name"] == "baseline")
    assert baseline["speedup"] == pytest.approx(1.0)
    fast = next(s for s in result if s["run_name"] == "fast_run")
    assert fast["speedup"] == pytest.approx(1.2)


def test_compute_speedups_missing_metric():
    summaries = [
        {"run_name": "a", "mean_tokens_per_sec_gpu": 75000.0},
        {"run_name": "b"},  # missing metric
    ]
    result = compute_speedups(summaries)
    assert result[0]["speedup"] == pytest.approx(1.0)
    assert result[1]["speedup"] is None


# ── format_table ──────────────────────────────────────────────────────────────

def test_format_table_contains_run_names():
    summaries = compute_speedups([
        _make_summary("baseline_760m"),
        _make_summary("flash_760m", tok_s=82500.0),
    ])
    table = format_table(summaries)
    assert "baseline_760m" in table
    assert "flash_760m" in table


def test_format_table_contains_headers():
    summaries = compute_speedups([_make_summary("run_a")])
    table = format_table(summaries)
    assert "tok/s/GPU" in table
    assert "final_loss" in table
    assert "speedup" in table


# ── format_csv ────────────────────────────────────────────────────────────────

def test_format_csv_is_valid_csv():
    import csv, io
    summaries = compute_speedups([
        _make_summary("run_a"),
        _make_summary("run_b", tok_s=82500.0),
    ])
    output = format_csv(summaries)
    reader = csv.DictReader(io.StringIO(output))
    rows = list(reader)
    assert len(rows) == 2
    assert "run_name" in rows[0]
