#!/usr/bin/env python3
"""Aggregate and compare results across ablation runs.

Usage:
    uv run experiments/compare_runs.py [--results-dir PATH] [--runs NAME...] [--sort METRIC] [--format table|csv] [--plot PATH]
"""

import argparse
import csv as csv_mod
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

COLUMNS = [
    ("run_name",                  "Run",           "{:s}",    str),
    ("mean_tokens_per_sec_gpu",   "tok/s/GPU",     "{:,.0f}", float),
    ("mean_tflops_per_gpu",       "TFLOP/s",       "{:.1f}",  float),
    ("median_elapsed_ms",         "iter_ms",       "{:.1f}",  float),
    ("final_lm_loss",             "final_loss",    "{:.4f}",  float),
    ("mean_lm_loss_last_10pct",   "loss_10%",      "{:.4f}",  float),
    ("total_iterations",          "iters",         "{:d}",    int),
    ("speedup",                   "speedup",       "{:.2f}x", float),
]


def load_summaries(results_dir: str, run_names: list[str] | None = None) -> list[dict]:
    base = Path(results_dir)
    summaries = []

    if run_names:
        dirs = [base / name for name in run_names]
    else:
        dirs = sorted(p for p in base.iterdir() if p.is_dir())

    for d in dirs:
        summary_path = d / "summary.json"
        if not summary_path.exists():
            print(f"Warning: no summary.json in {d}", file=sys.stderr)
            continue
        with open(summary_path) as f:
            summaries.append(json.load(f))

    return summaries


def compute_speedups(summaries: list[dict], baseline_name: str | None = None) -> list[dict]:
    if not summaries:
        return summaries

    if baseline_name:
        baselines = [s for s in summaries if s.get("run_name") == baseline_name]
        baseline = baselines[0] if baselines else summaries[0]
    else:
        baseline = summaries[0]

    base_tok = baseline.get("mean_tokens_per_sec_gpu") or 1.0

    for s in summaries:
        tok = s.get("mean_tokens_per_sec_gpu")
        s["speedup"] = (tok / base_tok) if tok else None

    return summaries


def format_table(summaries: list[dict]) -> str:
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback to simple formatting without tabulate
        return _format_table_simple(summaries)

    headers = [col_header for _, col_header, _, _ in COLUMNS]
    rows = []

    for s in summaries:
        row = []
        for col_key, _, fmt, cast in COLUMNS:
            val = s.get(col_key)
            if val is None:
                row.append("—")
            else:
                try:
                    row.append(fmt.format(cast(val)))
                except (ValueError, TypeError):
                    row.append(str(val))
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="simple")


def _format_table_simple(summaries: list[dict]) -> str:
    headers = [col_header for _, col_header, _, _ in COLUMNS]
    rows = []
    for s in summaries:
        row = []
        for col_key, _, fmt, cast in COLUMNS:
            val = s.get(col_key)
            if val is None:
                row.append("—")
            else:
                try:
                    row.append(fmt.format(cast(val)))
                except (ValueError, TypeError):
                    row.append(str(val))
        rows.append(row)

    # Compute column widths
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines = [header_line, sep]
    for row in rows:
        lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    return "\n".join(lines)


def format_csv(summaries: list[dict]) -> str:
    import io
    out = io.StringIO()
    writer = csv_mod.DictWriter(out, fieldnames=[k for k, _, _, _ in COLUMNS],
                                extrasaction="ignore")
    writer.writeheader()
    for s in summaries:
        writer.writerow({k: s.get(k, "") for k, _, _, _ in COLUMNS})
    return out.getvalue()


def plot_comparison(summaries: list[dict], metric: str, output_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot", file=sys.stderr)
        return

    names = [s.get("run_name", "?") for s in summaries]
    values = [s.get(metric) for s in summaries]

    # Drop runs missing the metric
    pairs = [(n, v) for n, v in zip(names, values) if v is not None]
    if not pairs:
        print(f"No data for metric '{metric}' — skipping plot", file=sys.stderr)
        return

    names, values = zip(*pairs)
    _, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    bars = ax.bar(range(len(names)), values)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"Ablation comparison: {metric}")

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,.0f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--runs", nargs="+", metavar="NAME")
    parser.add_argument("--baseline", metavar="NAME",
                        help="Name of baseline run for speedup calculation (default: first run)")
    parser.add_argument("--sort", default="mean_tokens_per_sec_gpu", metavar="METRIC",
                        help="Sort by this metric (descending)")
    parser.add_argument("--format", choices=["table", "csv"], default="table")
    parser.add_argument("--plot", metavar="PATH",
                        help="Save a bar chart PNG to this path")
    args = parser.parse_args()

    summaries = load_summaries(args.results_dir, args.runs)
    if not summaries:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Sort
    summaries.sort(key=lambda s: s.get(args.sort) or 0, reverse=True)

    summaries = compute_speedups(summaries, baseline_name=args.baseline)

    if args.format == "csv":
        print(format_csv(summaries))
    else:
        print(format_table(summaries))

    if args.plot:
        plot_comparison(summaries, metric=args.sort, output_path=args.plot)


if __name__ == "__main__":
    main()
