"""Convert downloaded parquets into Megatron-style JSONL.

Streams row groups from pyarrow so peak RAM is one row group (~64K rows), not
the whole parquet. A 3 GB English shard stays under ~300 MB RAM.

Each output line is a JSON object with at least a 'text' field. If
--keep-token-count is set (default), we also emit 'token_count' from the
parquet — `longctx filter-long` uses it to skip the retokenize pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from longctx.utils import read_json, write_json


def _print_distribution(summary: dict) -> None:
    if not summary:
        return
    total_tokens = sum(v["tokens"] for v in summary.values())
    print(f"\n{'Lang':<6} {'Docs':>10} {'M-tokens':>10} {'Share%':>8} {'JSONL MB':>10}")
    print("-" * 50)
    for lc, v in sorted(summary.items(), key=lambda x: -x[1]["tokens"]):
        pct = 100 * v["tokens"] / total_tokens if total_tokens else 0
        print(f"{lc:<6} {v['docs']:>10,} {v['tokens']/1e6:>10.1f} "
              f"{pct:>8.2f} {v['jsonl_mb']:>10.0f}")
    print(f"{'TOTAL':<6} {sum(v['docs'] for v in summary.values()):>10,} "
          f"{total_tokens/1e6:>10.1f}")


def cmd_convert(args) -> None:
    import pyarrow.parquet as pq

    output_dir = Path(args.output_dir)
    megatron_dir = Path(args.megatron_dir)
    megatron_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[error] No manifest at {manifest_path}. Run `longctx download` first.")
        sys.exit(1)

    manifest = read_json(manifest_path)
    summary: dict[str, dict] = {}
    keep_tc = getattr(args, "keep_token_count", True)

    for lc, meta in manifest.items():
        files = meta["files"]
        if not files:
            continue

        jsonl_path = megatron_dir / f"{lc}.jsonl"
        n_docs = 0
        n_tokens = 0
        print(f"[{lc}] converting {len(files)} parquet(s) → {jsonl_path.name}")

        with open(jsonl_path, "w", encoding="utf-8") as out:
            for parquet_file in files:
                try:
                    pf = pq.ParquetFile(parquet_file)
                    for rg_idx in range(pf.metadata.num_row_groups):
                        batch = pf.read_row_group(
                            rg_idx, columns=["text", "token_count"],
                        )
                        texts = batch.column("text").to_pylist()
                        tokens = batch.column("token_count").to_pylist()

                        if args.max_docs_per_shard:
                            remaining = args.max_docs_per_shard - n_docs
                            if remaining <= 0:
                                break
                            if len(texts) > remaining:
                                texts = texts[:remaining]
                                tokens = tokens[:remaining]

                        for text, tok in zip(texts, tokens):
                            text = (text or "").strip()
                            if not text:
                                continue
                            rec = {"text": text}
                            if keep_tc:
                                rec["token_count"] = int(tok or 0)
                            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            n_docs += 1
                            n_tokens += int(tok or 0)
                except Exception as e:
                    print(f"  [error] {parquet_file}: {e}")

        size_mb = jsonl_path.stat().st_size / 1e6
        summary[lc] = {"docs": n_docs, "tokens": n_tokens, "jsonl_mb": size_mb}
        print(f"  → {n_docs:,} docs, ~{n_tokens/1e6:.1f}M tokens, {size_mb:.0f} MB")

    summary_path = megatron_dir / "conversion_summary.json"
    write_json(summary_path, summary)

    print("\nConversion summary:")
    _print_distribution(summary)
    print(f"\nSummary saved to {summary_path}")
