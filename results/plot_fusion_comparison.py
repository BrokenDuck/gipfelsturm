"""Report-quality kernel-fusion Pareto plots (throughput + TFLOPs).

Each bar = individual contribution of one fusion kernel
            (measured as: baseline_value - value_without_that_kernel).
The cumulative line shows running-total % of all attributed gains.
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

DATA_FILE = Path(__file__).parent / "fusion_comparison.csv"

with open(DATA_FILE) as f:
    rows = {r["run_name"]: r for r in csv.DictReader(f)}

def tp(key):  return float(rows[key]["mean_tokens_per_sec_gpu"])
def tf(key):  return float(rows[key]["mean_tflops_per_gpu"])

BASELINE = "attn_cudnn"

# Individual contribution of each fusion = baseline − ablation_without_it.
# no_softmax_fusion excluded (attention already fused via cuDNN).
ABLATIONS = [
    ("no_rope_fusion",          "RoPE"),
    ("no_swiglu_fusion",        "SwiGLU"),
    ("no_cross_entropy_fusion", "Cross-\nentropy"),
    ("no_grad_accum_fusion",    "Grad-\naccum"),
    ("no_dropout_fusion",       "Dropout"),
]

ABLATIONS_COLORS = ["#E53935", "#FB8C00", "#F9A825", "#43A047", "#1E88E5"]
ALL_KERNELS_COLOR = "#5E35B1"

gains_tp = np.array([tp(BASELINE) - tp(key) for key, _ in ABLATIONS])
gains_tf = np.array([tf(BASELINE) - tf(key) for key, _ in ABLATIONS])
labels   = [label for _, label in ABLATIONS]

# Sort individual ablations by throughput gain descending
order = np.argsort(gains_tp)[::-1]
gains_tp = gains_tp[order]
gains_tf = gains_tf[order]
labels   = [labels[i] for i in order]
colors   = [ABLATIONS_COLORS[i] for i in order]

# Prepend no_fusion absolute value as a reference bar
gains_tp = np.concatenate([[tp("no_fusion")], gains_tp])
gains_tf = np.concatenate([[tf("no_fusion")], gains_tf])
labels   = ["No\nFusion"] + labels
colors   = [ALL_KERNELS_COLOR] + colors

x = np.arange(len(labels))


def draw_pareto(ax, values, ylabel, fmt):
    # values[0] is the no_fusion absolute reference; values[1:] are gains
    gains = values[1:]
    total = gains.sum()
    cumulative_pct = np.cumsum(gains) / total * 100

    # Bars
    ax.bar(x, values, color=colors, width=0.6,
           linewidth=0.4, edgecolor="white", zorder=3)

    # Value annotation above each bar
    pad = values.max() * 0.015
    for i, v in enumerate(values):
        ax.text(i, v + pad, fmt(v),
                ha="center", va="bottom", fontsize=9, color="#111111")

    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, ha="center", linespacing=1.3)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0, values.max() * 1.22)
    ax.yaxis.grid(True, linestyle=":", color="#d0d0d0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Cumulative % line over gain bars only (x[1:]), starting from x[0]+0.3
    ax2 = ax.twinx()
    cx = np.concatenate([[x[0] + 0.3], x[1:] + 0.3])
    cy = np.concatenate([[0.0], cumulative_pct])
    ax2.plot(cx, cy, color="#333333", marker="o", markersize=5,
             linewidth=1.6, linestyle="--", zorder=4)
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Cumulative gain  (%)", fontsize=9.5, color="#333333")
    ax2.tick_params(axis="y", labelcolor="#444444", labelsize=8.5)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.spines["top"].set_visible(False)

    return ax2


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
fig.patch.set_facecolor("white")
fig.subplots_adjust(wspace=0.55, bottom=0.22)

draw_pareto(axL, gains_tp, "Throughput gain  (tok / sec / GPU)",
            lambda v: f"{v/1e3:.1f}k")
draw_pareto(axR, gains_tf, "TFLOPs / GPU gain",
            lambda v: f"{v:.1f}")

axL.set_title("Throughput", fontsize=12, fontweight="bold", pad=8)
axR.set_title("TFLOPs / GPU", fontsize=12, fontweight="bold", pad=8)


# --------------------------------------------------------------------------- #
# Shared legend (centred below both panels)
# --------------------------------------------------------------------------- #

patch_handles = [
    mpatches.Patch(color=ALL_KERNELS_COLOR, label="No fusion (reference)"),
] + [
    mpatches.Patch(color=colors[i], label=labels[i].replace("\n", "") + " fusion")
    for i in range(1, len(labels))
]
line_handle = mlines.Line2D(
    [0], [0], color="#333333", linestyle="--", marker="o",
    markersize=5, linewidth=1.6, label="Cumulative gain (%)",
)
fig.legend(
    handles=patch_handles + [line_handle],
    loc="lower center",
    ncol=6,
    fontsize=9.5,
    framealpha=0.95,
    edgecolor="#cccccc",
    bbox_to_anchor=(0.5, 0.0),
)

# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #

base = Path(__file__).parent / "fusion_comparison"
fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print(f"Saved: {base}.png / .pdf")
