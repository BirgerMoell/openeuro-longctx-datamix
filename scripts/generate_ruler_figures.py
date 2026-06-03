"""Generate RULER eval figures for README."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = os.path.join(os.path.dirname(__file__), "../docs/figures")
os.makedirs(OUT_DIR, exist_ok=True)

CTX = [4096, 8192, 16384, 32768]
CTX_LABELS = ["4K", "8K", "16K", "32K"]

RESULTS = {
    "niah_single_1":   [1.000, 1.000, 1.000, 0.620],
    "niah_single_2":   [0.660, 0.350, 0.020, 0.020],
    "niah_single_3":   [0.630, 0.200, 0.110, 0.000],
    "niah_multikey_1": [0.340, 0.330, 0.310, 0.040],
    "niah_multikey_2": [0.140, 0.070, 0.010, 0.000],
    "niah_multikey_3": [0.060, 0.010, 0.010, 0.000],
    "niah_multivalue": [0.290, 0.045, 0.003, 0.000],
    "niah_multiquery": [0.263, 0.160, 0.010, 0.003],
    "ruler_cwe":       [0.544, 0.312, 0.183, 0.047],
    "ruler_fwe":       [0.360, 0.273, 0.333, 0.330],
}

AVG = [np.mean([v[i] for v in RESULTS.values()]) for i in range(4)]

COLORS = {
    "niah_single":   "#2196F3",
    "niah_multikey": "#FF9800",
    "niah_multi":    "#9C27B0",
    "ruler":         "#4CAF50",
    "avg":           "#F44336",
}

CMAP = LinearSegmentedColormap.from_list(
    "ruler", [(0.85, 0.18, 0.18), (0.98, 0.75, 0.15), (0.18, 0.65, 0.28)], N=256
)

# ── Figure 1: Line chart — all tasks + average ───────────────────────────────

fig1, ax1 = plt.subplots(figsize=(10, 6))

style = {
    "niah_single_1":   dict(color=COLORS["niah_single"],   lw=2.5, ls="-",  marker="o"),
    "niah_single_2":   dict(color=COLORS["niah_single"],   lw=1.5, ls="--", marker="s"),
    "niah_single_3":   dict(color=COLORS["niah_single"],   lw=1.5, ls=":",  marker="^"),
    "niah_multikey_1": dict(color=COLORS["niah_multikey"], lw=1.5, ls="-",  marker="o"),
    "niah_multikey_2": dict(color=COLORS["niah_multikey"], lw=1.5, ls="--", marker="s"),
    "niah_multikey_3": dict(color=COLORS["niah_multikey"], lw=1.5, ls=":",  marker="^"),
    "niah_multivalue": dict(color=COLORS["niah_multi"],    lw=1.5, ls="-",  marker="D"),
    "niah_multiquery": dict(color=COLORS["niah_multi"],    lw=1.5, ls="--", marker="P"),
    "ruler_cwe":       dict(color=COLORS["ruler"],         lw=1.5, ls="-",  marker="o"),
    "ruler_fwe":       dict(color=COLORS["ruler"],         lw=1.5, ls="--", marker="s"),
}

x = np.arange(4)
for task, scores in RESULTS.items():
    ax1.plot(x, scores, label=task, markersize=6, **style[task])

ax1.plot(x, AVG, color=COLORS["avg"], lw=3, ls="-", marker="*",
         markersize=12, label="AVG", zorder=10)

ax1.set_xticks(x)
ax1.set_xticklabels(CTX_LABELS, fontsize=12)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_ylim(-0.02, 1.08)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax1.set_title("RULER Eval — YaRN v2 Multilingual 9B · All tasks by context length",
              fontsize=13, fontweight="bold")
ax1.legend(ncol=2, fontsize=9, loc="upper right")
ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
ax1.set_axisbelow(True)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "ruler_all_tasks.png"), dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved figure 1 (all tasks line chart)")


# ── Figure 2: Heatmap — tasks × context lengths ──────────────────────────────

tasks = list(RESULTS.keys())
grid = np.array([RESULTS[t] for t in tasks])

fig2, ax2 = plt.subplots(figsize=(8, 7))
im = ax2.imshow(grid, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(4))
ax2.set_yticks(range(len(tasks)))
ax2.set_xticklabels(CTX_LABELS, fontsize=12)
ax2.set_yticklabels(tasks, fontsize=11)
ax2.set_xlabel("Context length", fontsize=12)
ax2.set_title("RULER Score Heatmap — YaRN v2 Multilingual 9B",
              fontsize=13, fontweight="bold", pad=10)
for r in range(len(tasks)):
    for c in range(4):
        v = grid[r, c]
        color = "white" if v < 0.5 else "black"
        ax2.text(c, r, f"{v:.2f}", ha="center", va="center",
                 fontsize=10, color=color, fontweight="bold")
cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)
cbar.set_label("Score", fontsize=11)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "ruler_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved figure 2 (heatmap)")


# ── Figure 3: Grouped bar chart by task category ─────────────────────────────

categories = {
    "Single needle\n(avg 1/2/3)": [np.mean([RESULTS[f"niah_single_{i}"][c] for i in range(1,4)]) for c in range(4)],
    "Multi-key\n(avg 1/2/3)":     [np.mean([RESULTS[f"niah_multikey_{i}"][c] for i in range(1,4)]) for c in range(4)],
    "Multi-value /\nMulti-query":  [np.mean([RESULTS["niah_multivalue"][c], RESULTS["niah_multiquery"][c]]) for c in range(4)],
    "Aggregation\n(CWE + FWE)":   [np.mean([RESULTS["ruler_cwe"][c], RESULTS["ruler_fwe"][c]]) for c in range(4)],
}

cat_colors = [COLORS["niah_single"], COLORS["niah_multikey"], COLORS["niah_multi"], COLORS["ruler"]]
n_cats = len(categories)
n_ctx = 4
width = 0.18
x3 = np.arange(n_ctx)

fig3, ax3 = plt.subplots(figsize=(11, 6))
for i, (cat, scores) in enumerate(categories.items()):
    offset = (i - n_cats / 2 + 0.5) * width
    bars = ax3.bar(x3 + offset, scores, width, label=cat,
                   color=cat_colors[i], edgecolor="white", linewidth=0.8)
    for bar, score in zip(bars, scores):
        if score > 0.03:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{score:.2f}", ha="center", va="bottom", fontsize=8)

ax3.set_xticks(x3)
ax3.set_xticklabels(CTX_LABELS, fontsize=12)
ax3.set_ylabel("Score", fontsize=12)
ax3.set_ylim(0, 1.1)
ax3.set_title("RULER Eval by Task Category — YaRN v2 Multilingual 9B",
              fontsize=13, fontweight="bold")
ax3.legend(fontsize=10, loc="upper right")
ax3.yaxis.grid(True, linestyle=":", alpha=0.5)
ax3.set_axisbelow(True)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "ruler_by_category.png"), dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved figure 3 (grouped bar by category)")

print(f"\nAll figures saved to {OUT_DIR}/")
