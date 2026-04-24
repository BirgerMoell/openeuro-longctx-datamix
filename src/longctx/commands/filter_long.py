"""Filter JSONL to long documents only — the core context-extension step.

If each JSONL record has a `token_count` field (written by `convert`), we use
that directly and avoid re-tokenization. Otherwise we fall back to a crude
character heuristic: ~4 chars per token. For a proper token-based filter on
JSONL that lacks token_count, re-run `convert` with `--keep-token-count` (on
by default) or run `longctx tokenize` first and use the Megatron .bin files.
"""

from __future__ import annotations

import json
from pathlib import Path

from longctx.utils import write_json


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _doc_tokens(rec: dict) -> int:
    """Best effort token count for a doc. Uses token_count if present, else ~4 chars/token."""
    tc = rec.get("token_count")
    if isinstance(tc, int) and tc > 0:
        return tc
    text = rec.get("text", "") or ""
    return max(1, len(text) // 4)


def cmd_filter_long(args) -> None:
    src_dir = Path(args.megatron_dir)
    dst_dir = Path(args.long_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if args.languages:
        wanted = [l.strip() for l in args.languages.split(",") if l.strip()]
        jsonl_files = [src_dir / f"{lc}.jsonl" for lc in wanted]
    else:
        jsonl_files = sorted(src_dir.glob("*.jsonl"))

    min_t = args.min_tokens
    max_t = args.max_tokens
    print(f"Filtering docs with {min_t} <= tokens"
          + (f" <= {max_t}" if max_t else "")
          + f"  →  {dst_dir}\n")

    summary: dict[str, dict] = {}
    print(f"{'Lang':<6} {'In docs':>10} {'Out docs':>10} {'Kept%':>7} "
          f"{'In Mtok':>9} {'Out Mtok':>9}")
    print("-" * 60)

    for jf in jsonl_files:
        lc = jf.stem
        if not jf.exists():
            print(f"{lc:<6} (missing)")
            continue

        out_path = dst_dir / jf.name
        in_docs = out_docs = 0
        in_tok = out_tok = 0

        with open(out_path, "w", encoding="utf-8") as out:
            for rec in _iter_jsonl(jf):
                n = _doc_tokens(rec)
                in_docs += 1
                in_tok += n
                if n < min_t:
                    continue
                if max_t is not None and n > max_t:
                    continue
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_docs += 1
                out_tok += n

        kept = 100 * out_docs / in_docs if in_docs else 0.0
        print(f"{lc:<6} {in_docs:>10,} {out_docs:>10,} {kept:>6.1f}% "
              f"{in_tok/1e6:>9.1f} {out_tok/1e6:>9.1f}")
        summary[lc] = {
            "in_docs": in_docs,
            "out_docs": out_docs,
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "min_tokens": min_t,
            "max_tokens": max_t,
        }

    write_json(dst_dir / "filter_summary.json", summary)
    print(f"\nFilter summary: {dst_dir / 'filter_summary.json'}")
