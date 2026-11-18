#!/usr/bin/env python3
"""Reproduce every number, table and figure in the depth-0 / RoPE-base paper.

Inputs: the raw per-trial NIAH result files copied from LUMI into
``paper/raw/lumi/eval_niah_128k_<job>/`` and the launch scripts next to them.
Outputs: LaTeX macros/tables in ``paper/generated/`` and figures in
``paper/figures/``. Dependencies: numpy, matplotlib.

    python3 paper/analysis.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "lumi"
OUT = ROOT / "generated"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# ----------------------------------------------------------------- model geometry
HEAD_DIM = 128            # per-head dimension d (config.json)
THETA_PRE = 100_000       # RoPE base during 4K pretraining (openeurollm/prelude config)
T_PRE = 4096              # pretraining context length


def d_extra(theta_pre=THETA_PRE, t_pre=T_PRE, d=HEAD_DIM):
    """Critical dimension (Liu et al., 2024): dimensions whose period exceeds the
    pretraining length are not trained over a full rotation."""
    return 2 * math.ceil((d / 2) * math.log(t_pre / (2 * math.pi)) / math.log(theta_pre))


DEX = d_extra()


def t_extra(theta, d=HEAD_DIM, dex=DEX):
    """Extrapolation bound for base theta: period of the critical dimension."""
    return 2 * math.pi * theta ** (dex / d)


def theta_crit(L, d=HEAD_DIM, dex=DEX):
    """Smallest base whose critical-dimension period covers L tokens."""
    return (L / (2 * math.pi)) ** (d / dex)


def lam_max(theta, d=HEAD_DIM):
    """Longest RoPE wavelength (dimension pair d/2-1)."""
    return 2 * math.pi * theta ** ((d - 2) / d)


# ----------------------------------------------------------------- runs
# Each evaluated HF checkpoint, the eval jobs that scored it, and its recipe.
# Recipes are read from the launch scripts in raw/lumi (see table_runs).
RUNS = {
    "v1":       dict(label="v1 (catalogue data)", theta=5e6, L=131072, base="0.6T", tokens=1.0e9,
                     start="own 64K (θ=2M)", data="catalogue mix", jobs=[19408317, 19415715, 19418402, 19419241]),
    "v2j64":    dict(label="64K, length-biased A", theta=2e6, L=65536, base="0.6T", tokens=3.0e9,
                     start="32K (θ=1M)", data="length-biased A", jobs=[19471359]),
    "v2o64":    dict(label="64K, length-biased B", theta=2e6, L=65536, base="0.6T", tokens=3.0e9,
                     start="32K (θ=1M)", data="length-biased B", jobs=[19472175]),
    "th8":      dict(label="sweep θ=8M", theta=8e6, L=131072, base="0.6T", tokens=0.294e9,
                     start="64K A (θ=2M)", data="length-biased A", jobs=[19485815]),
    "th16":     dict(label="sweep θ=16M", theta=16e6, L=131072, base="0.6T", tokens=0.294e9,
                     start="64K A (θ=2M)", data="length-biased A", jobs=[19493083]),
    "th32":     dict(label="sweep θ=32M", theta=32e6, L=131072, base="0.6T", tokens=0.294e9,
                     start="64K A (θ=2M)", data="length-biased A", jobs=[19495677, 19506820, 19506821, 19506822]),
    "v3":       dict(label="v3 θ=32M (0.6T base)", theta=32e6, L=131072, base="0.6T", tokens=2.0e9,
                     start="64K A (θ=2M)", data="length-biased A", jobs=[19517071, 19517072, 19517073]),
    "prelude":  dict(label="Prelude θ=32M (1T base)", theta=32e6, L=131072, base="1T", tokens=2.0e9,
                     start="Prelude 64K (θ=2M)", data="length-biased A", jobs=[19630747, 19630748, 19630749]),
    "prelude2": dict(label="Prelude θ=32M, 1.26× tokens", theta=32e6, L=131072, base="1T", tokens=2.52e9,
                     start="Prelude 64K (θ=2M)", data="length-biased A", jobs=[19717361, 19717362]),
    "p256":     dict(label="Prelude θ=64M, 256K", theta=64e6, L=262144, base="1T", tokens=0.99e9,
                     start="Prelude 128K (θ=32M)", data="256K blend", jobs=[19668631, 19679941, 19713329, 19713330,
                                                                             19713331, 20285729, 20285730, 20285731,
                                                                             20285732, 20285733]),
}


def load_run(jobs):
    seen = {}
    for j in jobs:
        d = RAW / f"eval_niah_128k_{j}"
        for f in sorted(d.glob("*_results.jsonl")):
            if f.name == "all_results.jsonl":
                continue
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r["lang"], r.get("condition", "main"), r["target_tokens"], round(r["depth"], 4), r["trial"])
                seen.setdefault(key, r)  # identical seeded case => keep first
    return list(seen.values())


DATA = {k: load_run(v["jobs"]) for k, v in RUNS.items()}
MNAME = {"v1": "VOne", "v2j64": "DataA", "v2o64": "DataB", "th8": "ThEight", "th16": "ThSixteen",
         "th32": "ThThirtyTwo", "v3": "VThree", "prelude": "Prelude", "prelude2": "PreludeTwo", "p256": "TwoFiftySix"}


def cell(rows, ctx=None, depth=None, cond="main", langs=None):
    sel = [r for r in rows if r.get("condition", "main") == cond
           and (ctx is None or r["target_tokens"] == ctx)
           and (depth is None or abs(r["depth"] - depth) < 1e-6)
           and (langs is None or r["lang"] in langs)]
    return sum(1 for r in sel if r["correct"]), len(sel)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def binom_cdf_le(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def fisher_two_sided(a, b, c, d):
    """Fisher exact test for [[a,b],[c,d]]."""
    n1, n2, m1 = a + b, c + d, a + c
    N = n1 + n2

    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, m1 - x) / math.comb(N, m1)
    p0 = pmf(a)
    lo, hi = max(0, m1 - n2), min(n1, m1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p0 * (1 + 1e-9)))


macros: dict[str, str] = {}


def pfmt(p):
    return "<0.001" if p < 0.001 else f"={p:.3f}"


def mac(name, v, fmt="{}"):
    macros[name] = fmt.format(v) if not isinstance(v, str) else v


def pct(k, n):
    return f"{100 * k / n:.0f}" if n else "--"


def frac(k, n):
    return f"{k}/{n}"


def sci(x):
    """32e6 -> 32M"""
    if x >= 1e9:
        return f"{x / 1e9:.2f}B"
    if x >= 1e6:
        v = x / 1e6
        if abs(v - round(v)) < 1e-9 or v >= 100:
            return f"{v:.0f}M"
        return f"{v:.1f}M"
    return f"{x / 1e3:.0f}K"


DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
# Men et al. (2024), Table 1: lower bound on the base (head dim 128)
MEN = {65536: 2.1e6, 131072: 7.8e6, 262144: 3.6e7, 524288: 6.4e7, 1048576: 5.1e8}


# ----------------------------------------------------------------- geometry macros
mac("dExtra", DEX)
mac("dExtraPairs", (HEAD_DIM - DEX) // 2)
mac("dOverDex", f"{HEAD_DIM / DEX:.2f}")
for L, nm in [(65536, "SixtyFour"), (131072, "OneTwentyEight"), (262144, "TwoFiftySix"),
              (524288, "FiveTwelve"), (1048576, "OneM"), (2097152, "TwoM")]:
    mac(f"thetaCrit{nm}", sci(theta_crit(L)))
for th, nm in [(2e6, "TwoM"), (5e6, "FiveM"), (8e6, "EightM"), (16e6, "SixteenM"), (32e6, "ThirtyTwoM"), (64e6, "SixtyFourM")]:
    mac(f"tExtra{nm}", f"{t_extra(th) / 1024:.0f}K")
mac("lamMaxEightM", f"{lam_max(8e6) / 1e6:.1f}M")
mac("lamRatioEightM", f"{lam_max(8e6) / 131072:.0f}")

# ----------------------------------------------------------------- per-run summaries
summ_rows = []
for k, spec in RUNS.items():
    rows = DATA[k]
    L = spec["L"]
    langs = sorted({r["lang"] for r in rows if r["target_tokens"] == L and r.get("condition", "main") == "main"})
    ok, n = cell(rows, ctx=L)
    d0k, d0n = cell(rows, ctx=L, depth=0.0)
    d5k, d5n = cell(rows, ctx=L, depth=0.5)
    spec.update(n_langs=len(langs), ok=ok, n=n, d0=(d0k, d0n), d5=(d5k, d5n))
    M = MNAME[k]
    spec["M"] = M
    mac(f"acc{M}", pct(ok, n))
    mac(f"accFrac{M}", frac(ok, n))
    mac(f"dz{M}", pct(d0k, d0n))
    mac(f"dzFrac{M}", frac(d0k, d0n))
    mac(f"dh{M}", pct(d5k, d5n))
    mac(f"dhFrac{M}", frac(d5k, d5n))
    mac(f"nLangs{M}", len(langs))
    by_depth = " & ".join(
        (lambda kn: f"{pct(*kn)} ({kn[0]}/{kn[1]})" if kn[1] else "--")(cell(rows, ctx=L, depth=dp)) for dp in DEPTHS)
    summ_rows.append(f"{spec['label']} & {sci(spec['theta'])} & {L // 1024}K & {spec['base']} & {spec['tokens'] / 1e9:.2f}B & {len(langs)} & {by_depth} \\\\")

(OUT / "table_runs.tex").write_text(
    "\\begin{tabular}{lrrrrr" + "r" * len(DEPTHS) + "}\n\\toprule\n"
    "Run & $\\theta$ & Eval $L$ & Base & Tokens & Lang. & " + " & ".join(f"$d={dp}$" for dp in DEPTHS) + " \\\\\n\\midrule\n"
    + "\n".join(summ_rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------------- theta sweep (controlled)
sweep = ["th8", "th16", "th32"]
SWEEP_LANGS = {"fr", "fi", "cs"}
sw_rows = []
for k in sweep:
    rows = DATA[k]
    cells = []
    for L in (65536, 131072):
        for dp in DEPTHS:
            kk, nn = cell(rows, ctx=L, depth=dp, langs=SWEEP_LANGS)
            cells.append(f"{pct(kk, nn)}")
    k0, n0 = cell(rows, ctx=131072, depth=0.0, langs=SWEEP_LANGS)
    lo, hi = wilson(k0, n0)
    sw_rows.append(f"{sci(RUNS[k]['theta'])} & {t_extra(RUNS[k]['theta']) / 1024:.0f}K & " + " & ".join(cells)
                   + f" & [{100 * lo:.0f}, {100 * hi:.0f}] \\\\")
    M = RUNS[k]["M"]
    mac(f"swDz{M}", pct(k0, n0))
    mac(f"swDzFrac{M}", frac(k0, n0))
    mac(f"swDzLo{M}", f"{100 * lo:.0f}")
    mac(f"swDzHi{M}", f"{100 * hi:.0f}")
    k64, n64 = cell(rows, ctx=65536, depth=0.0, langs=SWEEP_LANGS)
    mac(f"swSixtyFourDz{M}", frac(k64, n64))
(OUT / "table_sweep.tex").write_text(
    "\\begin{tabular}{rr" + "r" * 10 + "r}\n\\toprule\n"
    " & & \\multicolumn{5}{c}{Evaluated at 64K} & \\multicolumn{5}{c}{Evaluated at 128K} & \\\\\n"
    "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\n"
    "$\\theta$ & $T_{\\mathrm{extra}}$ & " + " & ".join(f"{dp}" for dp in DEPTHS) * 1 + " & "
    + " & ".join(f"{dp}" for dp in DEPTHS) + " & 95\\% CI $d=0$ \\\\\n\\midrule\n"
    + "\n".join(sw_rows) + "\n\\bottomrule\n\\end{tabular}\n")
k8, n8 = cell(DATA["th8"], ctx=131072, depth=0.0, langs=SWEEP_LANGS)
k16, n16 = cell(DATA["th16"], ctx=131072, depth=0.0, langs=SWEEP_LANGS)
k32, n32 = cell(DATA["th32"], ctx=131072, depth=0.0, langs=SWEEP_LANGS)
mac("pEightSixteen", pfmt(fisher_two_sided(k8, n8 - k8, k16, n16 - k16)))
mac("pSixteenThirtyTwo", pfmt(fisher_two_sided(k16, n16 - k16, k32, n32 - k32)))
mac("nSweepTrials", n8)
mac("sweepTokens", f"{RUNS['th8']['tokens'] / 1e6:.0f}M")

# below chance
for k in ["v1", "v2j64", "v2o64"]:
    rows = DATA[k]
    L = RUNS[k]["L"]
    kk, nn = cell(rows, ctx=L, depth=0.0)
    M = RUNS[k]["M"]
    mac(f"belowChanceP{M}", f"{binom_cdf_le(kk, nn, 0.25):.1g}")
# 64K data comparison
mac("dataA", frac(*cell(DATA["v2j64"], ctx=65536, depth=0.0)))
mac("dataB", frac(*cell(DATA["v2o64"], ctx=65536, depth=0.0)))
mac("dataAquarter", frac(*cell(DATA["v2j64"], ctx=65536, depth=0.25)))
mac("dataBquarter", frac(*cell(DATA["v2o64"], ctx=65536, depth=0.25)))
mac("vOneSixtyFour", frac(*cell(DATA["v1"], ctx=65536, depth=0.0)))
mac("vOneQuarter", frac(*cell(DATA["v1"], ctx=131072, depth=0.25)))
mac("vOneHalf", frac(*cell(DATA["v1"], ctx=131072, depth=0.5)))

# ----------------------------------------------------------------- controls
ctrl_rows = []
for cond, nm in [("no_context", "No context"), ("shuffled", "Shuffled bindings"), ("short_ctx", "256-token context")]:
    tot_k = tot_n = 0
    for k in RUNS:
        kk, nn = cell(DATA[k], cond=cond)
        tot_k += kk
        tot_n += nn
    ctrl_rows.append((nm, tot_k, tot_n))
    mac({"no_context": "ctrlNoCtx", "shuffled": "ctrlShuffled", "short_ctx": "ctrlShort"}[cond], f"{pct(tot_k, tot_n)}\\% ({tot_k}/{tot_n})")

# ----------------------------------------------------------------- log-prob margins
def margins(rows, L, dp, langs=None):
    out = []
    for r in rows:
        if r.get("condition", "main") != "main" or r["target_tokens"] != L or abs(r["depth"] - dp) > 1e-6:
            continue
        if langs and r["lang"] not in langs:
            continue
        s = r.get("scores")
        if not s:
            continue
        t = s[r["true_idx"]]
        best_other = max(v for i, v in enumerate(s) if i != r["true_idx"])
        out.append(t - best_other)
    return np.array(out)


for k in sweep + ["v1"]:
    m = margins(DATA[k], 131072, 0.0, SWEEP_LANGS if k in sweep else None)
    if len(m):
        mac(f"margin{RUNS[k]['M']}", f"{np.median(m):+.1f}".replace("-", "\\textminus "))

fig, ax = plt.subplots(figsize=(3.4, 2.5))
data = [margins(DATA[k], 131072, 0.0, SWEEP_LANGS if k in sweep else None) for k in ["v1"] + sweep]
ax.boxplot(data, widths=0.5, showfliers=True, flierprops=dict(markersize=2))
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(range(1, 5))
ax.set_xticklabels(["5M\n(v1)", "8M", "16M", "32M"], fontsize=7)
ax.set_xlabel("RoPE base θ (128K, depth 0)", fontsize=7)
ax.set_ylabel("log p(true) − max log p(distractor)", fontsize=7)
ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "margins.pdf")
plt.close(fig)

# ----------------------------------------------------------------- depth profiles at 128K
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for k, col in [("v1", "#999999"), ("th8", "#e08a2c"), ("th16", "#c94f7c"), ("th32", "#1f4e9c")]:
    rows = DATA[k]
    langs = None if k == "v1" else SWEEP_LANGS
    ys, lo, hi = [], [], []
    for dp in DEPTHS:
        kk, nn = cell(rows, ctx=131072, depth=dp, langs=langs)
        ys.append(kk / nn if nn else np.nan)
        a, b = wilson(kk, nn)
        lo.append(a)
        hi.append(b)
    ax.plot(DEPTHS, ys, marker="o", ms=3, color=col, label=f"θ={sci(RUNS[k]['theta'])}" + (" (v1)" if k == "v1" else ""))
    ax.fill_between(DEPTHS, lo, hi, color=col, alpha=0.12)
ax.axhline(0.25, color="black", lw=0.5, ls=":")
ax.text(0.02, 0.27, "chance", fontsize=6, ha="left")
ax.set_xlabel("Needle depth (0 = start of window)", fontsize=7)
ax.set_ylabel("Accuracy at 128K", fontsize=7)
ax.set_ylim(-0.03, 1.05)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, frameon=False, loc="center right")
fig.tight_layout()
fig.savefig(FIG / "depth_profiles.pdf")
plt.close(fig)

# ----------------------------------------------------------------- unified distance curve
pts = []
for k, spec in RUNS.items():
    for r in DATA[k]:
        if r.get("condition", "main") != "main" or r["target_tokens"] < 65536:
            continue
        dist = (1 - r["depth"]) * r["actual_ctx_tokens"]
        if dist < 1:
            continue
        pts.append((k, dist / t_extra(spec["theta"]), r["correct"]))
bins = [0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.5]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
centers, accs, los, his, ns = [], [], [], [], []
for a, b in zip(bins[:-1], bins[1:]):
    sel = [c for _, x, c in pts if a <= x < b]
    if not sel:
        continue
    kk, nn = sum(sel), len(sel)
    lo, hi = wilson(kk, nn)
    centers.append((a + b) / 2)
    accs.append(kk / nn)
    los.append(lo)
    his.append(hi)
    ns.append(nn)
ax.errorbar(centers, accs, yerr=[np.array(accs) - np.array(los), np.array(his) - np.array(accs)],
            fmt="o-", ms=3, color="#1f4e9c", capsize=2, lw=1)
for c, a, n in zip(centers, accs, ns):
    ax.annotate(f"n={n}", (c, a), fontsize=5.5, xytext=(2, -9), textcoords="offset points")
ax.axvline(1.0, color="black", lw=0.5, ls="--")
ax.axhline(0.25, color="black", lw=0.5, ls=":")
ax.set_xlabel("Needle distance / $T_{\\mathrm{extra}}(\\theta)$", fontsize=7)
ax.set_ylabel("Accuracy (all runs, ≥64K)", fontsize=7)
ax.set_ylim(-0.03, 1.05)
ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "distance_ratio.pdf")
plt.close(fig)
binned = list(zip(bins[:-1], bins[1:]))
def acc_range(a, b):
    sel = [c for _, x, c in pts if a <= x < b]
    return sum(sel), len(sel)
k1, n1 = acc_range(0, 1.0)
k2, n2 = acc_range(1.0, 1.5)
k3, n3 = acc_range(1.5, 2.0)
k4, n4 = acc_range(2.0, 10)
mac("nRatioTrials", f"{len(pts):,}".replace(",", "{,}"))
mac("ratioBelowOne", f"{pct(k1, n1)}\\% ({k1}/{n1})")
mac("ratioOneToOneHalf", f"{pct(k2, n2)}\\% ({k2}/{n2})")
mac("ratioOneHalfToTwo", f"{pct(k3, n3)}\\% ({k3}/{n3})")
mac("ratioAboveTwo", f"{pct(k4, n4)}\\% ({k4}/{n4})")

# ----------------------------------------------------------------- critical theta vs L
fig, ax = plt.subplots(figsize=(3.4, 2.6))
Ls = np.logspace(np.log10(16384), np.log10(2.2e6), 100)
ax.plot(Ls, [theta_crit(x) for x in Ls], color="#1f4e9c", lw=1.2, label="critical-dimension bound")
ax.plot(Ls, [32e6 * (x / 131072) for x in Ls], color="#e08a2c", lw=1, ls="--", label="doubling per octave")
ax.plot(list(MEN), list(MEN.values()), color="#777777", lw=0.8, ls=":", marker="x", ms=3, label="Xu et al. (2024) bound")
obs = []
for k in ["v1", "v2j64", "v2o64", "th8", "th16", "th32", "v3", "prelude", "p256"]:
    spec = RUNS[k]
    kk, nn = spec["d0"]
    obs.append((spec["L"], spec["theta"], kk / nn))
for k in ["th8", "th16", "th32"]:
    kk, nn = cell(DATA[k], ctx=65536, depth=0.0)
    obs.append((65536, RUNS[k]["theta"], kk / nn))
for L, th, a in obs:
    col = "#2a9d4b" if a >= 0.9 else ("#e0b12c" if a >= 0.5 else "#c0392b")
    ax.scatter([L], [th], s=22, color=col, edgecolor="black", lw=0.4, zorder=3)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([65536, 131072, 262144, 524288, 1048576, 2097152])
ax.set_xticklabels(["64K", "128K", "256K", "512K", "1M", "2M"], fontsize=7)
ax.set_xlabel("Context length L", fontsize=7)
ax.set_ylabel("RoPE base θ", fontsize=7)
ax.tick_params(labelsize=7)
from matplotlib.lines import Line2D  # noqa: E402
handles, labels = ax.get_legend_handles_labels()
handles += [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="black", markersize=4) for c in ["#2a9d4b", "#e0b12c", "#c0392b"]]
labels += ["depth-0 ≥ 90%", "50–90%", "< 50%"]
ax.legend(handles, labels, fontsize=5.8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "theta_vs_length.pdf")
plt.close(fig)

# predicted vs doubling table
pred_rows = []
for L in [65536, 131072, 262144, 524288, 1048576, 2097152]:
    doubling = 32e6 * (L / 131072)
    men = sci(MEN[L]) if L in MEN else "--"
    pred_rows.append(f"{L // 1024 if L < 1048576 else L // 1048576}{'K' if L < 1048576 else 'M'} & {sci(theta_crit(L))} & {men} & {sci(doubling)} & {theta_crit(L) / doubling:.1f}$\\times$ \\\\")
(OUT / "table_predictions.tex").write_text(
    "\\begin{tabular}{lrrrr}\n\\toprule\n$L$ & $\\theta_{\\mathrm{crit}}$ (Eq.~\\ref{eq:thetacrit}) & Bound of \\citet{men2024base} & Doubling rule & Ratio to doubling \\\\\n\\midrule\n"
    + "\n".join(pred_rows) + "\n\\bottomrule\n\\end{tabular}\n")
mac("ratioFiveTwelve", f"{theta_crit(524288) / (32e6 * 4):.1f}")
mac("ratioOneM", f"{theta_crit(1048576) / (32e6 * 8):.1f}")
mac("lenExponent", f"{HEAD_DIM / DEX:.2f}")
mac("octaveFactor", f"{2 ** (HEAD_DIM / DEX):.2f}")

# ----------------------------------------------------------------- base comparison (v3 vs prelude)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for k, col in [("v3", "#1f4e9c"), ("prelude", "#e08a2c"), ("prelude2", "#c94f7c"), ("p256", "#2a9d4b")]:
    rows = DATA[k]
    L = RUNS[k]["L"]
    ys = []
    for dp in DEPTHS:
        kk, nn = cell(rows, ctx=L, depth=dp)
        ys.append(kk / nn if nn else np.nan)
    ax.plot(DEPTHS, ys, marker="o", ms=3, color=col, label=RUNS[k]["label"])
ax.set_ylim(0.6, 1.02)
ax.set_xlabel("Needle depth", fontsize=7)
ax.set_ylabel("Accuracy", fontsize=7)
ax.tick_params(labelsize=7)
ax.legend(fontsize=5.8, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "base_comparison.pdf")
plt.close(fig)

# v3 vs prelude at depth 0.5 (same 15 languages, same seeded cases)
a, n_a = cell(DATA["v3"], ctx=131072, depth=0.5)
b, n_b = cell(DATA["prelude"], ctx=131072, depth=0.5)
mac("pMiddle", pfmt(fisher_two_sided(a, n_a - a, b, n_b - b)))
# mid-window (0.25-0.75) pooled
mid = lambda k: [cell(DATA[k], ctx=131072, depth=dp) for dp in (0.25, 0.5, 0.75)]
va = mid("v3"); pa = mid("prelude")
vk, vn = sum(x[0] for x in va), sum(x[1] for x in va)
pk, pn = sum(x[0] for x in pa), sum(x[1] for x in pa)
mac("midVthree", f"{pct(vk, vn)}\\% ({vk}/{vn})")
mac("midPrelude", f"{pct(pk, pn)}\\% ({pk}/{pn})")
mac("midPreludePct", pct(pk, pn))
mac("midVthreePct", pct(vk, vn))
mac("pMid", pfmt(fisher_two_sided(vk, vn - vk, pk, pn - pk)))
# 2x vs prelude on the languages both cover
langs2 = sorted({r["lang"] for r in DATA["prelude2"] if r.get("condition", "main") == "main"})
c1 = cell(DATA["prelude"], ctx=131072, depth=0.5, langs=set(langs2))
c2 = cell(DATA["prelude2"], ctx=131072, depth=0.5, langs=set(langs2))
mac("halfPreludeSame", frac(*c1))
mac("halfPreludeTwo", frac(*c2))
mac("nLangsPreludeTwo", len(langs2))
mac("prelTwoIters", "300")
mac("prelTwoPlanned", "476")
# per-language prelude depth-0.5
prel_lang = defaultdict(lambda: [0, 0])
for r in DATA["prelude"]:
    if r.get("condition", "main") == "main" and abs(r["depth"] - 0.5) < 1e-6:
        prel_lang[r["lang"]][0] += r["correct"]
        prel_lang[r["lang"]][1] += 1
mac("prelMidLangsImperfect", sum(1 for v in prel_lang.values() if v[0] < v[1]))
mac("prelMidLangs", len(prel_lang))

# 256K detail
for dp, nm in [(0.0, "Zero"), (0.25, "Quarter"), (0.5, "Half")]:
    kk, nn = cell(DATA["p256"], ctx=262144, depth=dp)
    mac(f"tfs{nm}", f"{pct(kk, nn)}\\% ({kk}/{nn})")
kk, nn = cell(DATA["p256"], ctx=262144)
mac("tfsAll", f"{pct(kk, nn)}\\% ({kk}/{nn})")
mac("tfsLangs", len({r['lang'] for r in DATA['p256'] if r['target_tokens'] == 262144}))
kk, nn = cell(DATA["p256"], ctx=131072)
mac("tfsAtOneTwentyEight", f"{pct(kk, nn)}\\% ({kk}/{nn})")
mac("tfsRatioZero", f"{262144 / t_extra(64e6):.2f}")

# ----------------------------------------------------------------- write
lines = ["% Auto-generated by paper/analysis.py — do not edit."]
for k, v in macros.items():
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(OUT / "macros.tex").write_text("\n".join(lines) + "\n")
if __name__ == "__main__":
    for k, v in macros.items():
        print(f"{k:28s} {v}")
