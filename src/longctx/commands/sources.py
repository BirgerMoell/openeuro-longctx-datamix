"""`longctx sources {list,fetch}` — pluggable adapters for alternative corpora.

Used to fill in the three OpenEuroLLM target languages missing from
FinePDFs-Edu (ga, sq, lb) and to supplement any of the 35 covered
languages with additional long-context data.
"""

from __future__ import annotations

import json
from pathlib import Path

from longctx.languages import MISSING_LANGS, parse_languages
from longctx.sources import SOURCES


def cmd_sources_list(_args) -> None:
    print(f"{'Source':<10} {'Gated':<6} {'Langs':>6}  Dataset")
    print("-" * 72)
    for name, mod in SOURCES.items():
        print(f"{name:<10} {'yes' if mod.GATED else 'no':<6} "
              f"{len(mod.SUPPORTED):>6}  {mod.DATASET_ID}")
    print()
    print("Coverage of the 3 languages missing from FinePDFs-Edu:")
    for lc in MISSING_LANGS:
        cov = [n for n, m in SOURCES.items() if lc in m.SUPPORTED]
        print(f"  {lc}: {', '.join(cov) if cov else '(none)'}")


def cmd_sources_fetch(args) -> None:
    if args.source not in SOURCES:
        print(f"[error] Unknown source '{args.source}'. "
              f"Known: {', '.join(SOURCES)}")
        raise SystemExit(2)

    mod = SOURCES[args.source]
    langs = parse_languages(args.languages)
    megatron_dir = Path(args.megatron_dir)
    megatron_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    for lc in langs:
        if lc not in mod.SUPPORTED:
            print(f"[skip] {lc}: {args.source} has no config for this language")
            continue

        out_path = megatron_dir / f"{lc}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {lc}: {out_path} exists (use --overwrite to replace)")
            continue

        print(f"\n[{lc}] {args.source} → {out_path}")
        try:
            stats = mod.fetch(
                lc, out_path,
                sample=args.sample,
                max_docs=args.max_docs,
            )
        except Exception as e:
            print(f"  [error] {e}")
            report[lc] = {"error": str(e)}
            continue

        size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
        stats["jsonl_mb"] = size_mb
        report[lc] = stats
        print(f"  → {stats['docs']:,} docs, ~{stats['tokens_approx']/1e6:.1f}M "
              f"approx-tokens, {size_mb:.0f} MB")

    report_path = megatron_dir / f"sources_{args.source}_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSource fetch summary: {report_path}")
