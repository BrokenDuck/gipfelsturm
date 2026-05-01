#!/usr/bin/env python3
"""Parse Megatron-LM training logs into structured metrics.

Usage:
    uv run experiments/parse_log.py <out_file> [err_file] [--output PATH] [--summary PATH] [--skip-first N]
    uv run experiments/parse_log.py logs/gipfel-baseline_760m-12345.out logs/gipfel-baseline_760m-12345.err --summary results/baseline/summary.json
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# Field-by-field extractors: (key, regex, type)
# Each regex is applied independently — robust to field reordering or new fields.
_EXTRACTORS = [
    ("iteration",          re.compile(r"iteration\s+(\d+)/\s*(\d+)"),            None),  # special case: two groups
    ("consumed_samples",   re.compile(r"consumed samples:\s+(\d+)"),              int),
    ("elapsed_ms",         re.compile(r"elapsed time per iteration \(ms\):\s+([\d.]+)"), float),
    ("tflops_per_gpu",     re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s+([\d.]+)"), float),
    ("tokens_per_sec_gpu", re.compile(r"tokens/sec/GPU:\s+(\d+)"),                int),
    ("learning_rate",      re.compile(r"learning rate:\s+([\d.eE+-]+)"),          float),
    ("global_batch_size",  re.compile(r"global batch size:\s+(\d+)"),             int),
    ("lm_loss",            re.compile(r"lm loss:\s+([\d.eE+-]+)"),                float),
    ("loss_scale",         re.compile(r"loss scale:\s+([\d.]+)"),                 float),
    ("grad_norm",          re.compile(r"grad norm:\s+([\d.]+)"),                  float),
    ("skipped_iters",      re.compile(r"number of skipped iterations:\s+(\d+)"),  int),
    ("nan_iters",          re.compile(r"number of nan iterations:\s+(\d+)"),      int),
]


def parse_line(line: str) -> dict | None:
    """Parse a single Megatron training log line. Returns None for non-training lines."""
    if "iteration" not in line or "|" not in line:
        return None

    fields = {}
    for key, pattern, cast in _EXTRACTORS:
        m = pattern.search(line)
        if not m:
            continue
        if key == "iteration":
            fields["iteration"] = int(m.group(1))
            fields["total_iters"] = int(m.group(2))
        else:
            fields[key] = cast(m.group(1))  # type: ignore[misc]

    if "iteration" not in fields:
        return None
    return fields


def parse_log(*log_paths: str) -> list[dict]:
    """Parse all training iteration lines from one or more log files."""
    metrics = []
    for log_path in log_paths:
        with open(log_path) as f:
            for line in f:
                record = parse_line(line)
                if record is not None:
                    metrics.append(record)
    metrics.sort(key=lambda r: r["iteration"])
    return metrics


def compute_summary(metrics: list[dict], run_name: str = "", skip_first: int = 5) -> dict:
    """Compute aggregate statistics over steady-state iterations."""
    if not metrics:
        return {"run_name": run_name, "total_iterations": 0}

    steady = metrics[skip_first:]
    if not steady:
        steady = metrics

    def collect(key: str) -> list[float]:
        return [r[key] for r in steady if key in r]

    def safe_mean(vals):
        return statistics.mean(vals) if vals else None

    def safe_median(vals):
        return statistics.median(vals) if vals else None

    def safe_p95(vals):
        if not vals:
            return None
        s = sorted(vals)
        idx = int(0.95 * len(s))
        return s[min(idx, len(s) - 1)]

    tok_s = collect("tokens_per_sec_gpu")
    tflops = collect("tflops_per_gpu")
    elapsed = collect("elapsed_ms")
    losses = collect("lm_loss")

    last_10pct = losses[max(0, len(losses) - max(1, len(losses) // 10)):]

    return {
        "run_name": run_name,
        "total_iterations": len(metrics),
        "steady_state_iterations": len(steady),
        "mean_tokens_per_sec_gpu": safe_mean(tok_s),
        "median_tokens_per_sec_gpu": safe_median(tok_s),
        "p95_tokens_per_sec_gpu": safe_p95(tok_s),
        "mean_tflops_per_gpu": safe_mean(tflops),
        "median_elapsed_ms": safe_median(elapsed),
        "final_lm_loss": losses[-1] if losses else None,
        "min_lm_loss": min(losses) if losses else None,
        "mean_lm_loss_last_10pct": safe_mean(last_10pct),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_files", nargs="+", metavar="FILE",
                        help="Path to .out file (and optionally .err file) from a Megatron training run")
    parser.add_argument("--output", metavar="PATH",
                        help="Write metrics.jsonl to PATH (default: stdout)")
    parser.add_argument("--summary", metavar="PATH",
                        help="Write summary.json to PATH")
    parser.add_argument("--run-name", metavar="NAME",
                        help="Run name to embed in summary (default: inferred from log filename)")
    parser.add_argument("--skip-first", type=int, default=5, metavar="N",
                        help="Skip first N iterations for summary stats (default: 5)")
    args = parser.parse_args()

    run_name = args.run_name or Path(args.log_files[0]).stem

    metrics = parse_log(*args.log_files)

    if not metrics:
        print(f"No training iterations found in {', '.join(args.log_files)}", file=sys.stderr)
        sys.exit(1)

    # Write JSONL
    jsonl_lines = [json.dumps(r) for r in metrics]
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(jsonl_lines) + "\n")
        print(f"Wrote {len(metrics)} iterations to {args.output}")
    else:
        print("\n".join(jsonl_lines))

    # Write summary
    if args.summary:
        summary = compute_summary(metrics, run_name=run_name, skip_first=args.skip_first)
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote summary to {args.summary}")
        # Print a quick overview to stdout
        print(f"\nSummary for {run_name}:")
        if summary.get("mean_tokens_per_sec_gpu"):
            print(f"  tok/s/GPU (mean):   {summary['mean_tokens_per_sec_gpu']:,.0f}")
        if summary.get("mean_tflops_per_gpu"):
            print(f"  TFLOP/s/GPU (mean): {summary['mean_tflops_per_gpu']:.1f}")
        if summary.get("final_lm_loss"):
            print(f"  final lm_loss:      {summary['final_lm_loss']:.4f}")
        print(f"  iterations:         {summary['total_iterations']}")


if __name__ == "__main__":
    main()
