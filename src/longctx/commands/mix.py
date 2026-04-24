"""Emit a weighted --data-path block for Megatron training.

Mixing policy — tempered (α-) sampling:
    p_i = tokens_i / Σ tokens_j
    w_i ∝ p_i^alpha
    then normalize to sum to 1 and enforce a --floor (renormalize).

α = 1.0  → natural frequency (English dominates)
α = 0.3  → strong upsampling of low-resource langs (default, common for
            multilingual continued pretraining)
α = 0.0  → uniform across languages

Megatron accepts `--data-path <w1> <prefix1> <w2> <prefix2> …` as a single
space-separated argument. We emit it three ways:

  data_mix.json   machine-readable record of weights, token counts, paths
  data_mix.txt    human-readable table
  data_path.args  exactly the tokens to append after `--data-path`
"""

from __future__ import annotations

import json
from pathlib import Path


def _collect_sizes(bin_dir: Path, suffix: str, wanted: list[str] | None) -> dict[str, int]:
    """Return {lc: bin_size_bytes} for every tokenized language we find."""
    sizes: dict[str, int] = {}
    for bin_path in sorted(bin_dir.glob(f"*{suffix}.bin")):
        lc = bin_path.name.removesuffix(f"{suffix}.bin")
        if wanted and lc not in wanted:
            continue
        idx_path = bin_path.with_suffix(".idx")
        if not idx_path.exists():
            print(f"[warn] {lc}: {idx_path.name} missing, skipping")
            continue
        sizes[lc] = bin_path.stat().st_size
    return sizes


def _tempered_weights(sizes: dict[str, int], alpha: float, floor: float) -> dict[str, float]:
    total = sum(sizes.values())
    if total <= 0:
        return {lc: 0.0 for lc in sizes}
    probs = {lc: s / total for lc, s in sizes.items()}
    raw = {lc: (p ** alpha) if p > 0 else 0.0 for lc, p in probs.items()}
    raw_sum = sum(raw.values()) or 1.0
    w = {lc: v / raw_sum for lc, v in raw.items()}

    if floor > 0:
        w = {lc: max(v, floor) for lc, v in w.items()}
        s = sum(w.values())
        w = {lc: v / s for lc, v in w.items()}
    return w


def cmd_mix(args) -> None:
    bin_dir = Path(args.bin_dir).resolve()
    mix_dir = Path(args.mix_dir)
    mix_dir.mkdir(parents=True, exist_ok=True)

    wanted = None
    if args.languages:
        wanted = [l.strip() for l in args.languages.split(",") if l.strip()]

    sizes = _collect_sizes(bin_dir, args.suffix, wanted)
    if not sizes:
        print(f"[error] No tokenized datasets found under {bin_dir}")
        print(f"        Expected files like <lc>{args.suffix}.{{bin,idx}}")
        return

    weights = _tempered_weights(sizes, args.alpha, args.floor)
    total_bytes = sum(sizes.values())

    # Machine-readable manifest
    manifest = {
        "bin_dir": str(bin_dir),
        "suffix": args.suffix,
        "alpha": args.alpha,
        "floor": args.floor,
        "total_bin_bytes": total_bytes,
        "languages": {
            lc: {
                "prefix": str(bin_dir / f"{lc}{args.suffix}"),
                "bin_bytes": sizes[lc],
                "natural_p": sizes[lc] / total_bytes if total_bytes else 0.0,
                "weight": weights[lc],
            }
            for lc in sorted(sizes)
        },
    }
    with open(mix_dir / "data_mix.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Megatron --data-path tokens (space-joined)
    tokens: list[str] = []
    for lc in sorted(sizes.keys(), key=lambda x: -weights[x]):
        prefix = str(bin_dir / f"{lc}{args.suffix}")
        tokens.extend([f"{weights[lc]:.6f}", prefix])
    data_path_str = " ".join(tokens)
    (mix_dir / "data_path.args").write_text(data_path_str + "\n", encoding="utf-8")

    # Human-readable table
    lines = [
        f"# data mix for Megatron --data-path",
        f"# bin_dir = {bin_dir}",
        f"# alpha   = {args.alpha}",
        f"# floor   = {args.floor}",
        f"# total_bin_bytes = {total_bytes:,}",
        "",
        f"{'Lang':<6} {'Bin MB':>10} {'Natural%':>9} {'Weight%':>9} {'Prefix'}",
        "-" * 80,
    ]
    for lc in sorted(sizes.keys(), key=lambda x: -weights[x]):
        mb = sizes[lc] / 1e6
        nat = 100 * sizes[lc] / total_bytes if total_bytes else 0
        w = 100 * weights[lc]
        lines.append(f"{lc:<6} {mb:>10.1f} {nat:>8.2f}% {w:>8.2f}% "
                     f"{bin_dir / f'{lc}{args.suffix}'}")
    lines.append("")
    lines.append("# Paste into your Megatron launcher:")
    lines.append(f"DATA_PATH=\"{data_path_str}\"")
    lines.append("# then: --data-path $DATA_PATH")
    mix_txt = "\n".join(lines)
    (mix_dir / "data_mix.txt").write_text(mix_txt + "\n", encoding="utf-8")

    print(mix_txt)
    print()
    print(f"Wrote: {mix_dir / 'data_mix.json'}")
    print(f"Wrote: {mix_dir / 'data_mix.txt'}")
    print(f"Wrote: {mix_dir / 'data_path.args'}")
