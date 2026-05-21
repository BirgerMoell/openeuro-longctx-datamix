"""
Generate NIAH evaluation figures for the YaRN v2 multilingual model card.
Produces four PNG files saved to lumi/slurm/figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = os.path.join(os.path.dirname(__file__), "../lumi/slurm/figures")
os.makedirs(OUT_DIR, exist_ok=True)

CTX_LABELS = ["2K", "4K", "8K", "16K", "32K"]
DEPTH_LABELS = ["0%", "25%", "50%", "75%", "100%"]

CMAP = LinearSegmentedColormap.from_list(
    "niah", [(0.85, 0.18, 0.18), (0.98, 0.75, 0.15), (0.18, 0.65, 0.28)], N=256
)


def make_grid(d0, d25=1.0, d50=1.0, d75=1.0, d100=1.0,
              c2k=None, c4k=None, c8k=None, c16k=None, c32k=None):
    def row(overrides):
        return [1.0, 1.0, 1.0, 1.0, 1.0] if overrides is None else list(overrides)
    return np.array([
        row(c2k), row(c4k), row(c8k), row(c16k),
        [d0, d25, d50, d75, d100] if c32k is None else row(c32k),
    ])


# ── Data ────────────────────────────────────────────────────────────────────

COMPARISON = {
    "fr": make_grid(d0=0.20, c4k=[1.00, 0.90, 1.00, 1.00, 1.00]),
    "fi": make_grid(d0=0.50),
    "cs": make_grid(d0=0.20, c2k=[1.00, 1.00, 0.90, 1.00, 1.00]),
    "nl": make_grid(d0=0.40, c4k=[1.00, 0.90, 1.00, 1.00, 1.00]),
}

EXTENDED = {
    "en": make_grid(d0=0.80),
    "bg": make_grid(d0=0.20),
    "da": make_grid(d0=0.20, c4k=[1.00, 0.80, 1.00, 1.00, 1.00]),
    "de": make_grid(d0=0.20),
    "el": make_grid(d0=0.20),
    "es": make_grid(d0=0.20),
    "et": make_grid(d0=0.40),
    "ga": make_grid(d0=0.80),
    "hr": make_grid(d0=0.00),
    "hu": make_grid(d0=0.20),
    "it": make_grid(d0=0.40, c2k=[1.00, 1.00, 0.80, 1.00, 1.00]),
    "lt": make_grid(d0=1.00),
    "lv": make_grid(d0=0.00),
}

LANG_LABELS = {
    "fr": "French (fr)", "fi": "Finnish (fi)", "cs": "Czech (cs)", "nl": "Dutch (nl)",
    "en": "English (en)", "bg": "Bulgarian (bg)", "da": "Danish (da)",
    "de": "German (de)", "el": "Greek (el)", "es": "Spanish (es)",
    "et": "Estonian (et)", "ga": "Irish (ga)", "hr": "Croatian (hr)",
    "hu": "Hungarian (hu)", "it": "Italian (it)", "lt": "Lithuanian (lt)",
    "lv": "Latvian (lv)",
}


def plot_heatmap(ax, grid, title, fs_title=13, fs_tick=11, fs_val=11):
    im = ax.imshow(grid, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(DEPTH_LABELS, fontsize=fs_tick)
    ax.set_yticklabels(CTX_LABELS, fontsize=fs_tick)
    ax.set_xlabel("Needle depth", fontsize=fs_tick)
    ax.set_ylabel("Context length", fontsize=fs_tick)
    ax.set_title(title, fontsize=fs_title, fontweight="bold", pad=8)
    for r in range(5):
        for c in range(5):
            v = grid[r, c]
            color = "white" if v < 0.6 else "black"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=fs_val, color=color, fontweight="bold")
    return im


# ── Figure 1: Comparison languages — 2×2 grid ───────────────────────────────

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
fig1.suptitle(
    "NIAH Accuracy — Comparison Languages (10 trials/cell)\n"
    "YaRN v2 Multilingual 9B · 32 768-token context",
    fontsize=15, fontweight="bold", y=1.01,
)

ims = []
for ax, lang in zip(axes1.flat, ["fr", "fi", "cs", "nl"]):
    im = plot_heatmap(ax, COMPARISON[lang], LANG_LABELS[lang])
    ims.append(im)

fig1.tight_layout(h_pad=3, w_pad=3)
cbar = fig1.colorbar(ims[-1], ax=axes1.ravel().tolist(), fraction=0.025, pad=0.04)
cbar.set_label("Accuracy", fontsize=12)
cbar.ax.tick_params(labelsize=11)
fig1.savefig(os.path.join(OUT_DIR, "niah_comparison_languages.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved figure 1")


# ── Figure 2: Extended batch 1 — 2×4 grid ───────────────────────────────────

BATCH1 = ["en", "bg", "da", "de", "el", "es", "et", "ga"]

fig2, axes2 = plt.subplots(2, 4, figsize=(22, 12))
fig2.suptitle(
    "NIAH Accuracy — Extended Languages Batch 1 (5 trials/cell)\n"
    "YaRN v2 Multilingual 9B · 32 768-token context",
    fontsize=15, fontweight="bold",
)

ims = []
for ax, lang in zip(axes2.flat, BATCH1):
    im = plot_heatmap(ax, EXTENDED[lang], LANG_LABELS[lang])
    ims.append(im)

fig2.tight_layout(h_pad=3, w_pad=2)
cbar = fig2.colorbar(ims[-1], ax=axes2.ravel().tolist(), fraction=0.015, pad=0.04)
cbar.set_label("Accuracy", fontsize=12)
cbar.ax.tick_params(labelsize=11)
fig2.savefig(os.path.join(OUT_DIR, "niah_extended_batch1.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved figure 2")


# ── Figure 3: Extended batch 2 — 2+3 layout ─────────────────────────────────
# 5 languages: top row 3, bottom row 2 (centred)

BATCH2 = ["hr", "hu", "it", "lt", "lv"]

fig3 = plt.figure(figsize=(22, 12))
fig3.suptitle(
    "NIAH Accuracy — Extended Languages Batch 2 (5 trials/cell)\n"
    "YaRN v2 Multilingual 9B · 32 768-token context",
    fontsize=15, fontweight="bold",
)

# Top row: 3 panels
top_axes = [fig3.add_subplot(2, 3, i + 1) for i in range(3)]
# Bottom row: 2 panels centred using gridspec offset
import matplotlib.gridspec as gridspec
gs_bottom = gridspec.GridSpec(2, 6, figure=fig3)
bot_ax1 = fig3.add_subplot(gs_bottom[1, 1:3])
bot_ax2 = fig3.add_subplot(gs_bottom[1, 3:5])
all_axes = top_axes + [bot_ax1, bot_ax2]

ims = []
for ax, lang in zip(all_axes, BATCH2):
    im = plot_heatmap(ax, EXTENDED[lang], LANG_LABELS[lang])
    ims.append(im)

fig3.tight_layout(h_pad=3, w_pad=2)
cbar = fig3.colorbar(ims[-1], ax=all_axes, fraction=0.015, pad=0.04)
cbar.set_label("Accuracy", fontsize=12)
cbar.ax.tick_params(labelsize=11)
fig3.savefig(os.path.join(OUT_DIR, "niah_extended_batch2.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved figure 3")


# ── Figure 4: 32K depth=0% bar chart ────────────────────────────────────────

ALL_LANGS = list(COMPARISON.keys()) + list(EXTENDED.keys())
ALL_GRIDS = {**COMPARISON, **EXTENDED}

scores_32k_d0 = {lang: ALL_GRIDS[lang][4, 0] for lang in ALL_LANGS}
sorted_langs = sorted(scores_32k_d0, key=scores_32k_d0.get)
sorted_scores = [scores_32k_d0[l] for l in sorted_langs]
bar_colors = [CMAP(s) for s in sorted_scores]

fig4, ax4 = plt.subplots(figsize=(18, 7))
ax4.bar(range(len(sorted_langs)), sorted_scores, color=bar_colors,
        edgecolor="white", linewidth=0.8, width=0.7)

ax4.set_xticks(range(len(sorted_langs)))
ax4.set_xticklabels(
    [LANG_LABELS[l].split("(")[0].strip() for l in sorted_langs],
    rotation=35, ha="right", fontsize=13,
)
ax4.set_ylabel("Accuracy", fontsize=13)
ax4.set_ylim(0, 1.2)
ax4.set_title(
    "32K context, needle at depth=0% — accuracy by language\n"
    "YaRN v2 Multilingual 9B  ·  all other depths and context lengths score 1.00",
    fontsize=14, fontweight="bold",
)
ax4.axhline(0.25, color="gray", linestyle="--", linewidth=1.5, label="Chance (0.25)")
ax4.legend(fontsize=12)
ax4.set_xlim(-0.6, len(sorted_langs) - 0.4)
ax4.yaxis.grid(True, linestyle=":", alpha=0.5)
ax4.set_axisbelow(True)
ax4.tick_params(axis="y", labelsize=12)

for i, (lang, score) in enumerate(zip(sorted_langs, sorted_scores)):
    ax4.text(i, score + 0.04, f"{score:.2f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold")

fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, "niah_32k_depth0_summary.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig4)
print("Saved figure 4")

print(f"\nAll figures saved to {OUT_DIR}/")
