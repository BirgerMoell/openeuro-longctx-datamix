#!/usr/bin/env python3
"""Construct long-context documents by bundling related source documents.

The output is raw JSONL with a `text` field and bundle metadata. It is intended
for the 256K natural-ish part of the 2M experiment data mix:

* stream long source documents from local JSONL/parquet or FinePDFs-Edu on HF
* keep bundles single-language
* insert explicit document boundaries
* stop once the requested examples per language and target length are built
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from longctx.languages import DATASET_ID, LANG_MAP, parse_languages  # noqa: E402


@dataclass
class SourceDoc:
    text: str
    lang: str
    token_count: int
    source: str
    local_id: str

    @property
    def doc_id(self) -> str:
        payload = f"{self.source}\n{self.local_id}\n{self.text[:256]}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]


@dataclass
class BundleState:
    lang: str
    target_tokens: int
    parts: list[str] = field(default_factory=list)
    docs: list[dict[str, Any]] = field(default_factory=list)
    estimated_tokens: int = 0

    def add(self, doc: SourceDoc) -> None:
        boundary = (
            "\n\n===== OELLM DOCUMENT BOUNDARY =====\n"
            f"language: {doc.lang}\n"
            f"source: {doc.source}\n"
            f"doc_id: {doc.doc_id}\n"
            f"source_tokens: {doc.token_count}\n"
            "===== CONTENT =====\n"
        )
        self.parts.append(boundary + doc.text.strip())
        self.estimated_tokens += doc.token_count
        self.docs.append(
            {
                "doc_id": doc.doc_id,
                "source": doc.source,
                "local_id": doc.local_id,
                "token_count": doc.token_count,
            }
        )

    def to_row(self, pack_id: str, split: str, min_fill_ratio: float) -> dict[str, Any]:
        return {
            "text": "\n".join(self.parts).strip() + "\n",
            "lang": self.lang,
            "split": split,
            "task": "constructed_long_document_clm",
            "pack_id": pack_id,
            "target_tokens": self.target_tokens,
            "estimated_tokens": self.estimated_tokens,
            "fill_ratio": self.estimated_tokens / self.target_tokens,
            "min_fill_ratio": min_fill_ratio,
            "source_doc_count": len(self.docs),
            "source_docs": self.docs,
            "boundary": "===== OELLM DOCUMENT BOUNDARY =====",
        }

    def reset(self) -> None:
        self.parts.clear()
        self.docs.clear()
        self.estimated_tokens = 0


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def infer_lang_from_path(path: Path) -> str:
    candidates = [path.stem.split("_")[0], path.parent.name.split("_")[0]]
    for item in candidates:
        if item in LANG_MAP:
            return item
    return "unknown"


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return max(1, int(len(text) / chars_per_token))


def row_text_and_tokens(row: dict[str, Any], chars_per_token: float) -> tuple[str, int]:
    text = str(row.get("text") or row.get("content") or "").strip()
    token_count = row.get("token_count")
    if token_count is None:
        token_count = row.get("tokens")
    if token_count is None:
        token_count = estimate_tokens(text, chars_per_token)
    return text, int(token_count or 0)


def iter_jsonl(paths: list[Path], chars_per_token: float) -> Iterable[SourceDoc]:
    for path in paths:
        fallback_lang = infer_lang_from_path(path)
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                text, token_count = row_text_and_tokens(row, chars_per_token)
                if not text:
                    continue
                lang = str(row.get("lang") or row.get("language") or fallback_lang)
                yield SourceDoc(
                    text=text,
                    lang=lang,
                    token_count=token_count,
                    source=str(path),
                    local_id=str(row.get("id") or row.get("doc_id") or idx),
                )


def iter_parquet(paths: list[Path], chars_per_token: float, batch_rows: int) -> Iterable[SourceDoc]:
    import pyarrow.parquet as pq

    for path in paths:
        fallback_lang = infer_lang_from_path(path)
        pf = pq.ParquetFile(path)
        names = set(pf.schema.names)
        columns = [name for name in ["text", "content", "token_count", "tokens", "lang", "language", "id"] if name in names]
        if "text" not in names and "content" not in names:
            print(f"[warn] skip parquet without text/content column: {path}", file=sys.stderr)
            continue
        row_offset = 0
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            records = batch.to_pylist()
            for idx, row in enumerate(records):
                text, token_count = row_text_and_tokens(row, chars_per_token)
                if not text:
                    continue
                lang = str(row.get("lang") or row.get("language") or fallback_lang)
                yield SourceDoc(
                    text=text,
                    lang=lang,
                    token_count=token_count,
                    source=str(path),
                    local_id=str(row.get("id") or row_offset + idx),
                )
            row_offset += len(records)


def iter_hf_finepdfs(
    languages: list[str],
    shards_per_language: int,
    cache_dir: Path,
    chars_per_token: float,
    batch_rows: int,
) -> Iterable[SourceDoc]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for lang in languages:
        hf_dir = LANG_MAP[lang]
        files = list(
            api.list_repo_tree(
                DATASET_ID,
                repo_type="dataset",
                path_in_repo=f"data/{hf_dir}/train",
            )
        )
        parquet_files = sorted(
            getattr(item, "path", "")
            for item in files
            if getattr(item, "path", "").endswith(".parquet")
        )
        if shards_per_language > 0:
            parquet_files = parquet_files[:shards_per_language]
        print(f"[hf] {lang}: {len(parquet_files)} shard(s)", flush=True)
        for path_in_repo in parquet_files:
            local = hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=path_in_repo,
                local_dir=str(cache_dir),
                token=os.environ.get("HF_TOKEN"),
            )
            for doc in iter_parquet([Path(local)], chars_per_token, batch_rows):
                yield SourceDoc(
                    text=doc.text,
                    lang=lang,
                    token_count=doc.token_count,
                    source=path_in_repo,
                    local_id=doc.local_id,
                )


def discover_input_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"[warn] no matches for input glob: {pattern}", file=sys.stderr)
        paths.extend(Path(match) for match in matches)
    return paths


def should_keep(doc: SourceDoc, selected_langs: set[str], min_tokens: int, max_tokens: int) -> bool:
    if selected_langs and doc.lang not in selected_langs:
        return False
    if doc.lang == "unknown":
        return False
    if doc.token_count < min_tokens:
        return False
    if max_tokens > 0 and doc.token_count > max_tokens:
        return False
    return True


def make_doc_iterator(args) -> Iterable[SourceDoc]:
    if args.source == "jsonl":
        paths = discover_input_paths(args.input_glob)
        return iter_jsonl(paths, args.approx_chars_per_token)
    if args.source == "parquet":
        paths = discover_input_paths(args.input_glob)
        return iter_parquet(paths, args.approx_chars_per_token, args.batch_rows)
    if args.source == "hf-finepdfs":
        languages = parse_languages(args.languages)
        languages = [lang for lang in languages if lang in LANG_MAP]
        return iter_hf_finepdfs(
            languages=languages,
            shards_per_language=args.hf_shards_per_language,
            cache_dir=Path(args.hf_cache_dir),
            chars_per_token=args.approx_chars_per_token,
            batch_rows=args.batch_rows,
        )
    raise ValueError(f"unsupported source: {args.source}")


def all_done(
    counts: dict[tuple[str, int], int],
    languages: list[str],
    targets: list[int],
    examples_per_language_per_length: int,
) -> bool:
    return all(
        counts[(lang, target)] >= examples_per_language_per_length
        for lang in languages
        for target in targets
    )


def choose_target(
    lang: str,
    targets: list[int],
    counts: dict[tuple[str, int], int],
    round_robin: dict[str, int],
    examples_per_language_per_length: int,
) -> int | None:
    pending = [
        target
        for target in targets
        if counts[(lang, target)] < examples_per_language_per_length
    ]
    if not pending:
        return None
    idx = round_robin[lang] % len(pending)
    round_robin[lang] += 1
    return pending[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["jsonl", "parquet", "hf-finepdfs"], required=True)
    parser.add_argument("--input-glob", nargs="*", default=[],
                        help="Input glob(s) for --source jsonl/parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="oellm_constructed_longdocs")
    parser.add_argument("--split", default="train")
    parser.add_argument("--languages", default=None,
                        help="Comma-separated language codes. Default: all known FinePDFs languages.")
    parser.add_argument("--target-tokens", default="262144")
    parser.add_argument("--examples-per-language-per-length", type=int, default=8)
    parser.add_argument("--min-doc-tokens", type=int, default=4096)
    parser.add_argument("--max-doc-tokens", type=int, default=0,
                        help="Skip source docs above this length. 0 disables the cap.")
    parser.add_argument("--min-fill-ratio", type=float, default=0.92)
    parser.add_argument("--flush-partial", action="store_true")
    parser.add_argument("--hf-shards-per-language", type=int, default=1,
                        help="For --source hf-finepdfs. Use 0 for all shards.")
    parser.add_argument("--hf-cache-dir", default="/tmp/oellm_finepdfs_cache")
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument("--approx-chars-per-token", type=float, default=3.8)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()

    random.seed(args.seed)
    targets = sorted(parse_csv_ints(args.target_tokens))
    languages = parse_languages(args.languages)
    languages = [lang for lang in languages if lang in LANG_MAP]
    selected_langs = set(languages)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_files = {
        target: (output_dir / f"{args.prefix}_{args.split}_{target}.jsonl").open("w", encoding="utf-8")
        for target in targets
    }
    states: dict[tuple[str, int], BundleState] = {
        (lang, target): BundleState(lang=lang, target_tokens=target)
        for lang in languages
        for target in targets
    }
    counts: dict[tuple[str, int], int] = defaultdict(int)
    round_robin: dict[str, int] = defaultdict(int)
    scanned_docs = 0
    kept_docs = 0
    skipped_docs = 0
    started = time.time()

    def flush(lang: str, target: int, partial: bool = False) -> None:
        state = states[(lang, target)]
        if not state.docs:
            return
        fill_ratio = state.estimated_tokens / target
        if fill_ratio < args.min_fill_ratio and not partial:
            return
        if counts[(lang, target)] >= args.examples_per_language_per_length:
            state.reset()
            return
        pack_id = f"{args.prefix}_{args.split}_{lang}_{target}_{counts[(lang, target)]:06d}"
        row = state.to_row(pack_id=pack_id, split=args.split, min_fill_ratio=args.min_fill_ratio)
        row["partial"] = partial
        out_files[target].write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[(lang, target)] += 1
        print(
            f"[write] {pack_id}: docs={len(state.docs)} "
            f"tokens~{state.estimated_tokens:,} fill={fill_ratio:.2f}",
            flush=True,
        )
        state.reset()

    try:
        for doc in make_doc_iterator(args):
            scanned_docs += 1
            if not should_keep(doc, selected_langs, args.min_doc_tokens, args.max_doc_tokens):
                skipped_docs += 1
                continue
            target = choose_target(
                doc.lang,
                targets,
                counts,
                round_robin,
                args.examples_per_language_per_length,
            )
            if target is None:
                continue
            kept_docs += 1
            state = states[(doc.lang, target)]
            state.add(doc)
            if state.estimated_tokens >= target:
                flush(doc.lang, target)
            if all_done(counts, languages, targets, args.examples_per_language_per_length):
                break

        if args.flush_partial:
            for lang in languages:
                for target in targets:
                    flush(lang, target, partial=True)
    finally:
        for handle in out_files.values():
            handle.close()

    manifest = {
        "created_at_unix": int(time.time()),
        "source": args.source,
        "prefix": args.prefix,
        "split": args.split,
        "languages": languages,
        "target_tokens": targets,
        "examples_per_language_per_length": args.examples_per_language_per_length,
        "min_doc_tokens": args.min_doc_tokens,
        "max_doc_tokens": args.max_doc_tokens,
        "min_fill_ratio": args.min_fill_ratio,
        "scanned_docs": scanned_docs,
        "kept_docs": kept_docs,
        "skipped_docs": skipped_docs,
        "elapsed_seconds": time.time() - started,
        "counts": {
            f"{lang}_{target}": counts[(lang, target)]
            for lang in languages
            for target in targets
        },
        "shards": [
            {
                "path": str(output_dir / f"{args.prefix}_{args.split}_{target}.jsonl"),
                "target_tokens": target,
                "bytes": (output_dir / f"{args.prefix}_{args.split}_{target}.jsonl").stat().st_size,
            }
            for target in targets
        ],
    }
    manifest_path = output_dir / f"{args.prefix}_{args.split}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
