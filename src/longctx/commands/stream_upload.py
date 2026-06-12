"""Stream FinePDFs-Edu through Megatron tokenization and upload each shard.

This command is meant for small NVIDIA boxes with tight local disks. It avoids
building one giant JSONL or one giant tokenized artifact by:

1. downloading one parquet shard from Hugging Face,
2. splitting rows into sequence-length tiers in bounded JSONL chunks,
3. tokenizing each chunk with Megatron's preprocess_data.py,
4. uploading the resulting .bin/.idx pair immediately, and
5. deleting local temporaries unless --keep-local is set.

The uploaded dataset is still plain Megatron indexed data. The only difference
from the three-file artifact is that mix/data_path.args contains many prefixes.
Megatron accepts that directly, and the LUMI scripts in this repo can read it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

from longctx.languages import DATASET_ID, LANG_MAP, MISSING_LANGS, parse_languages

DEFAULT_TIER_SPECS = {
    "16k_plus": {"min": 16384, "weight": 0.5},
    "4_16k": {"min": 4096, "max": 16384, "weight": 0.3},
    "under4k": {"max": 4096, "weight": 0.2},
}

TIER_PRESETS = {
    "lc16k": DEFAULT_TIER_SPECS,
    "lc128k": {
        "128k_plus": {"min": 131072, "weight": 0.25},
        "64_128k": {"min": 65536, "max": 131072, "weight": 0.25},
        "32_64k": {"min": 32768, "max": 65536, "weight": 0.20},
        "16_32k": {"min": 16384, "max": 32768, "weight": 0.15},
        "4_16k": {"min": 4096, "max": 16384, "weight": 0.10},
        "under4k": {"max": 4096, "weight": 0.05},
    },
    "lc256k": {
        "256k_plus": {"min": 262144, "weight": 0.20},
        "128_256k": {"min": 131072, "max": 262144, "weight": 0.25},
        "64_128k": {"min": 65536, "max": 131072, "weight": 0.20},
        "32_64k": {"min": 32768, "max": 65536, "weight": 0.15},
        "16_32k": {"min": 16384, "max": 32768, "weight": 0.10},
        "4_16k": {"min": 4096, "max": 16384, "weight": 0.07},
        "under4k": {"max": 4096, "weight": 0.03},
    },
}


def _normalise_tier_specs(raw_specs: dict) -> dict[str, dict[str, float | int]]:
    if not isinstance(raw_specs, dict) or not raw_specs:
        raise ValueError("Tier specs must be a non-empty JSON object")

    specs: dict[str, dict[str, float | int]] = {}
    total_weight = 0.0
    for tier, raw_spec in raw_specs.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Tier {tier!r} must be an object")
        min_tokens = int(raw_spec.get("min", 0) or 0)
        max_raw = raw_spec.get("max")
        max_tokens = int(max_raw) if max_raw is not None else None
        weight = float(raw_spec.get("weight", 0.0))
        if min_tokens < 0:
            raise ValueError(f"Tier {tier!r} has negative min tokens")
        if max_tokens is not None and max_tokens <= min_tokens:
            raise ValueError(f"Tier {tier!r} must have max > min")
        if weight < 0:
            raise ValueError(f"Tier {tier!r} has negative weight")

        spec: dict[str, float | int] = {"min": min_tokens, "weight": weight}
        if max_tokens is not None:
            spec["max"] = max_tokens
        specs[tier] = spec
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("At least one tier weight must be positive")

    for spec in specs.values():
        spec["weight"] = float(spec["weight"]) / total_weight
    return specs


def _load_tier_specs(args) -> tuple[dict[str, dict[str, float | int]], str]:
    if getattr(args, "tier_spec_json", None):
        path = Path(args.tier_spec_json).expanduser()
        try:
            raw_specs = json.loads(path.read_text(encoding="utf-8"))
        except OSError as e:
            raise ValueError(f"Could not read --tier-spec-json {path}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --tier-spec-json {path}: {e}") from e
        return _normalise_tier_specs(raw_specs), str(path)

    preset = getattr(args, "tier_preset", "lc16k")
    if preset not in TIER_PRESETS:
        raise ValueError(
            f"Unknown tier preset {preset!r}; expected one of {', '.join(TIER_PRESETS)}"
        )
    return _normalise_tier_specs(TIER_PRESETS[preset]), f"preset:{preset}"


def _tier_sort_key(item: tuple[str, dict[str, float | int]]) -> tuple[int, str]:
    tier, spec = item
    return (int(spec.get("min", 0)), tier)


def _tier_summary(tier_specs: dict[str, dict[str, float | int]]) -> str:
    parts = []
    for tier, spec in sorted(tier_specs.items(), key=_tier_sort_key, reverse=True):
        min_tokens = int(spec.get("min", 0))
        max_tokens = spec.get("max")
        if max_tokens is None:
            span = f">={min_tokens}"
        elif min_tokens == 0:
            span = f"<{int(max_tokens)}"
        else:
            span = f"{min_tokens}-{int(max_tokens)}"
        parts.append(f"{tier}({span}, w={float(spec['weight']):.3f})")
    return ", ".join(parts)


def _parse_tier_chunk_docs(
    spec: str | None,
    tier_specs: dict[str, dict[str, float | int]],
) -> dict[str, int]:
    if not spec:
        return {}
    out: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid --chunk-docs-by-tier item {item!r}; expected tier=docs"
            )
        tier, value = (part.strip() for part in item.split("=", 1))
        if tier not in tier_specs:
            raise ValueError(
                f"Invalid tier {tier!r}; expected one of {', '.join(tier_specs)}"
            )
        docs = int(value)
        if docs <= 0:
            raise ValueError(f"Chunk docs for {tier} must be positive, got {docs}")
        out[tier] = docs
    return out


def _parse_skip_shards(spec: str | None) -> dict[str, set[int]]:
    """Parse 'fr:24,en:20-22' into per-language shard indexes to skip."""
    out: dict[str, set[int]] = {}
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid --skip-shards item {item!r}; expected lang:index or lang:start-end"
            )
        lang, shard_spec = (part.strip() for part in item.split(":", 1))
        if not lang or not shard_spec:
            raise ValueError(
                f"Invalid --skip-shards item {item!r}; expected lang:index or lang:start-end"
            )
        if shard_spec.startswith("s") and shard_spec[1:].isdigit():
            shard_spec = shard_spec[1:]
        if "-" in shard_spec:
            start_s, end_s = (part.strip() for part in shard_spec.split("-", 1))
            if start_s.startswith("s"):
                start_s = start_s[1:]
            if end_s.startswith("s"):
                end_s = end_s[1:]
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid --skip-shards range {item!r}; end < start")
            indexes = range(start, end + 1)
        else:
            indexes = [int(shard_spec)]
        out.setdefault(lang, set()).update(indexes)
    return out


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _tier_for_tokens(
    token_count: int,
    tier_specs: dict[str, dict[str, float | int]],
) -> str:
    for tier, spec in sorted(tier_specs.items(), key=_tier_sort_key, reverse=True):
        min_tokens = int(spec.get("min", 0))
        max_tokens = spec.get("max")
        if token_count >= min_tokens and (max_tokens is None or token_count < int(max_tokens)):
            return tier
    raise ValueError(f"No tier matched token_count={token_count:,}; check tier specs")


def _list_parquet_files(api, lang_code: str) -> list[dict]:
    hf_dir = LANG_MAP[lang_code]
    files = list(api.list_repo_tree(
        DATASET_ID,
        repo_type="dataset",
        path_in_repo=f"data/{hf_dir}/train",
    ))
    out = []
    for item in files:
        path = getattr(item, "path", "")
        if path.endswith(".parquet"):
            out.append({"path": path, "size": int(getattr(item, "size", 0) or 0)})
    return sorted(out, key=lambda x: x["path"])


def _flush_jsonl_chunk(path: Path, rows: list[dict]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    docs = 0
    tokens = 0
    with path.open("w", encoding="utf-8") as out:
        for rec in rows:
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            tc = int(rec.get("token_count", 0) or 0)
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            docs += 1
            tokens += tc
    return docs, tokens


def _run_preprocess(args, jsonl_path: Path, prefix: Path) -> None:
    cmd = [
        args.python,
        str(Path(args.megatron_path) / "tools" / "preprocess_data.py"),
        "--input", str(jsonl_path),
        "--output-prefix", str(prefix),
        "--tokenizer-type", args.tokenizer_type,
        "--tokenizer-model", args.tokenizer_model,
        "--json-keys", "text",
        "--workers", str(args.workers),
        "--append-eod",
    ]
    if args.vocab_size is not None:
        cmd.extend(["--vocab-size", str(args.vocab_size)])
    subprocess.run(cmd, check=True)


def _upload_file(api, args, local_path: Path, path_in_repo: str) -> None:
    if args.dry_run:
        print(f"  [dry-run] upload {local_path} -> {args.repo_id}/{path_in_repo}")
        return
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"Upload {path_in_repo}",
    )


def _stage_file(stage_dir: Path, local_path: Path, path_in_repo: str) -> None:
    target = stage_dir / path_in_repo
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, target)


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _flush_upload_batch(api, args, stage_dir: Path, chunk_count: int) -> int:
    if args.dry_run or chunk_count == 0:
        return 0
    if not stage_dir.exists() or not any(stage_dir.rglob("*")):
        return 0
    stage_bytes = _dir_size(stage_dir)
    _upload_folder_with_retry(
        api,
        args,
        folder_path=str(stage_dir),
        commit_message=f"Upload {chunk_count} streamed Megatron chunk(s)",
    )
    print(
        f"    [cleanup] uploaded {stage_bytes/1e9:.2f} GB; deleting committed upload stage",
        flush=True,
    )
    shutil.rmtree(stage_dir, ignore_errors=True)
    return stage_bytes


def _upload_folder_with_retry(api, args, folder_path: str, commit_message: str) -> None:
    for attempt in range(1, args.upload_retries + 1):
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
            return
        except Exception as e:
            message = str(e)
            is_rate_limit = "429" in message or "rate limit" in message.lower()
            is_transient_server_error = any(
                marker in message
                for marker in (
                    "500 Internal Server Error",
                    "502 Bad Gateway",
                    "503 Service Unavailable",
                    "504 Gateway Timeout",
                )
            )
            if not (is_rate_limit or is_transient_server_error) or attempt >= args.upload_retries:
                raise
            match = re.search(r"Retry after (\d+) seconds", message)
            if match:
                sleep_seconds = int(match.group(1)) + 30
            elif "about 1 hour" in message:
                sleep_seconds = 3700
            elif is_transient_server_error:
                sleep_seconds = min(900, 120 * attempt)
            else:
                sleep_seconds = min(600, 60 * attempt)
            print(
                f"  [upload-retry] HF upload attempt {attempt}/{args.upload_retries} "
                f"failed; sleeping {sleep_seconds}s before retry.",
                flush=True,
            )
            time.sleep(sleep_seconds)


def _repo_file_set(api, args) -> set[str]:
    if args.dry_run:
        return set()
    try:
        return set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    except Exception:
        return set()


def _load_remote_chunk_manifests(api, args, work_root: Path, files: set[str]) -> list[dict]:
    if args.dry_run:
        return []
    from huggingface_hub import hf_hub_download

    entries = []
    manifest_prefixes = ["manifests/chunks/"]
    if args.run_id:
        run_id = args.run_id.strip().strip("/")
        manifest_prefixes.insert(0, f"runs/{run_id}/manifests/chunks/")
    manifest_files = sorted(
        path for path in files
        if any(path.startswith(prefix) for prefix in manifest_prefixes) and path.endswith(".json")
    )
    if not manifest_files:
        return entries
    cache_dir = work_root / "remote_manifest_cache"
    for path in manifest_files:
        try:
            local = hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=path,
                local_dir=str(cache_dir),
                token=os.environ.get("HF_TOKEN"),
            )
            entries.append(json.loads(Path(local).read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [warn] could not read remote chunk manifest {path}: {e}")
    print(f"[resume] loaded {len(entries):,} existing remote chunk manifest(s)")
    return entries


def _load_local_chunk_manifests(local_dir: Path) -> list[dict]:
    entries = []
    chunk_dir = local_dir / "manifests" / "chunks"
    if not chunk_dir.exists():
        return entries
    for path in sorted(chunk_dir.glob("*.json")):
        try:
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [warn] could not read local chunk manifest {path}: {e}")
    print(f"[resume] loaded {len(entries):,} existing local chunk manifest(s)")
    return entries


def _completed_chunk_names_from_files(files: set[str], run_id: str) -> set[str]:
    manifest_prefixes = ["manifests/chunks/"]
    if run_id:
        manifest_prefixes.insert(0, f"runs/{run_id}/manifests/chunks/")
    names = set()
    for path in files:
        if not path.endswith(".json"):
            continue
        if any(path.startswith(prefix) for prefix in manifest_prefixes):
            names.add(Path(path).stem)
    return names


def _remove_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def _build_mix(
    entries: list[dict],
    tier_specs: dict[str, dict[str, float | int]],
    local_bin_dir: str = "bin",
) -> dict:
    totals_by_tier = {}
    for entry in entries:
        totals_by_tier[entry["tier"]] = totals_by_tier.get(entry["tier"], 0) + entry["bin_bytes"]

    languages = {}
    tokens: list[str] = []
    total_bin_bytes = sum(e["bin_bytes"] for e in entries)
    for entry in entries:
        tier_total = totals_by_tier.get(entry["tier"], 0)
        tier_weight = float(tier_specs[entry["tier"]]["weight"])
        weight = tier_weight * entry["bin_bytes"] / tier_total if tier_total else 0.0
        name = entry["name"]
        prefix = f"{local_bin_dir}/{name}_text_document"
        entry["weight"] = weight
        entry["prefix"] = prefix
        languages[name] = {
            "weight": weight,
            "bin_bytes": entry["bin_bytes"],
            "natural_p": entry["bin_bytes"] / total_bin_bytes if total_bin_bytes else 0.0,
            "prefix": prefix,
            "tier": entry["tier"],
            "language": entry["language"],
            "source": entry["source"],
            "docs": entry["docs"],
            "source_token_count": entry["source_token_count"],
        }
        tokens.extend([f"{weight:.8f}", prefix])

    return {
        "alpha": None,
        "floor": None,
        "suffix": "_text_document",
        "bin_dir": local_bin_dir,
        "total_bin_bytes": total_bin_bytes,
        "tiers": tier_specs,
        "languages": languages,
        "data_path_args": " ".join(tokens),
    }


def _write_readme(path: Path, args, entries: list[dict]) -> None:
    total_bytes = sum(e["bin_bytes"] + e["idx_bytes"] for e in entries)
    total_docs = sum(e["docs"] for e in entries)
    languages = sorted({e["language"] for e in entries})
    text = f"""---
license: other
task_categories:
- text-generation
language:
{chr(10).join(f'- {language}' for language in languages)}
pretty_name: OpenEuroLLM long-context Megatron streamed tokenized artifacts
---

# OpenEuroLLM long-context Megatron streamed tokenized artifacts

This dataset contains Megatron-LM indexed data (`.bin`/`.idx`) uploaded shard by
shard from `longctx stream-upload`. It is intended as a transport format for
training on LUMI or another machine; it is not raw text.

- Source dataset: `{DATASET_ID}`
- Tokenizer: `{args.tokenizer_model}`
- Tokenizer type: `{args.tokenizer_type}`
- Hub repo: `{args.repo_id or "local-only"}`
- Files uploaded: `{len(entries) * 2}` Megatron data files
- Documents: `{total_docs:,}`
- Uploaded binary/index bytes: `{total_bytes:,}`

## Use on LUMI

```bash
python -m longctx.cli artifacts download \\
  --repo-id {args.repo_id or "birgermoell/<uploaded-dataset>"} \\
  --output-dir /scratch/project_462000353/$USER/oellm-longctx-tokenized

export MULTILINGUAL_DIR=/scratch/project_462000353/$USER/oellm-longctx-tokenized
export DATA_PATH="$(cat "$MULTILINGUAL_DIR/mix/data_path.args")"
```

The `mix/data_path.args` file contains many weighted Megatron prefixes. This is
expected: the data was uploaded incrementally to keep the source machine's disk
usage low.
"""
    path.write_text(text, encoding="utf-8")


def cmd_stream_upload(args) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, create_repo, hf_hub_download

    if not args.megatron_path:
        args.megatron_path = os.environ.get("MEGATRON_LM")
    if not args.megatron_path:
        print("[error] Set --megatron-path or MEGATRON_LM.", file=sys.stderr)
        sys.exit(2)
    try:
        tier_specs, tier_spec_source = _load_tier_specs(args)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)

    langs = parse_languages(args.languages)
    excluded_langs = set(parse_languages(args.exclude_languages)) if args.exclude_languages else set()
    langs = [lc for lc in langs if lc not in excluded_langs]
    langs = [lc for lc in langs if lc not in MISSING_LANGS]
    if not langs:
        print("[error] No FinePDFs-Edu languages selected.", file=sys.stderr)
        sys.exit(2)
    try:
        chunk_docs_by_tier = _parse_tier_chunk_docs(args.chunk_docs_by_tier, tier_specs)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)

    work_root = Path(args.work_dir).expanduser().resolve() if args.work_dir else Path(
        tempfile.mkdtemp(prefix="longctx-stream-upload-")
    )
    work_root.mkdir(parents=True, exist_ok=True)
    print(f"Work dir: {work_root}")
    print(f"Tier spec: {tier_spec_source}")
    print(f"Tiers: {_tier_summary(tier_specs)}")
    if excluded_langs:
        print(f"Excluded languages: {','.join(sorted(excluded_langs))}")
    if chunk_docs_by_tier:
        print(
            "Chunk docs by tier: "
            + ", ".join(f"{tier}={docs}" for tier, docs in sorted(chunk_docs_by_tier.items()))
        )

    api = HfApi()
    local_dir = Path(args.local_dir).expanduser().resolve() if args.local_dir else None
    if args.local_only and local_dir is None:
        print("[error] --local-only requires --local-dir.", file=sys.stderr)
        sys.exit(2)
    if not args.local_only and not args.repo_id:
        print("[error] --repo-id is required unless --local-only is set.", file=sys.stderr)
        sys.exit(2)
    if not args.dry_run and not args.local_only:
        create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    run_id = args.run_id.strip().strip("/") if args.run_id else ""

    if args.resume and args.local_only and local_dir:
        entries: list[dict] = _load_local_chunk_manifests(local_dir)
        remote_files = set()
    else:
        remote_files = _repo_file_set(api, args) if args.resume else set()
        if args.resume and args.resume_names_only:
            entries = []
            completed_chunks = _completed_chunk_names_from_files(remote_files, run_id)
            print(
                f"[resume] loaded {len(completed_chunks):,} completed chunk name(s) "
                "from remote manifest filenames",
                flush=True,
            )
        else:
            entries = (
                _load_remote_chunk_manifests(api, args, work_root, remote_files)
                if args.resume else []
            )
            completed_chunks = {entry["name"] for entry in entries}
    if args.resume and args.local_only and local_dir:
        completed_chunks = {entry["name"] for entry in entries}
    unknown_entry_tiers = sorted({entry.get("tier", "") for entry in entries} - set(tier_specs))
    if unknown_entry_tiers:
        print(
            "[error] Resume manifests contain tiers not present in the active tier spec: "
            + ", ".join(unknown_entry_tiers)
            + ". Use a matching --tier-preset/--tier-spec-json or a fresh --run-id.",
            file=sys.stderr,
        )
        sys.exit(2)
    upload_stage_dir = work_root / "upload_stage"
    staged_chunk_count = 0
    staged_bytes = 0
    shard_limit = None if args.all_shards else args.shards
    skip_shards = _parse_skip_shards(getattr(args, "skip_shards", None))
    total_seen_docs = 0
    repo_bin_dir = f"runs/{run_id}/bin" if run_id else "bin"
    repo_chunk_manifest_dir = f"runs/{run_id}/manifests/chunks" if run_id else "manifests/chunks"

    try:
        for lc in langs:
            files = _list_parquet_files(api, lc)
            if shard_limit is not None:
                files = files[:shard_limit]
            print(f"\n[{lc}] {len(files)} parquet shard(s)")

            for shard_idx, item in enumerate(files):
                if shard_idx in skip_shards.get(lc, set()):
                    print(f"  [skip-shard] {lc}_s{shard_idx:04d} by --skip-shards")
                    continue
                source_path = item["path"]
                print(f"  [download] {source_path} ({item['size']/1e9:.2f} GB)")
                parquet_path = Path(hf_hub_download(
                    repo_id=DATASET_ID,
                    repo_type="dataset",
                    filename=source_path,
                    local_dir=str(work_root / "hf_cache"),
                    token=os.environ.get("HF_TOKEN"),
                ))

                tier_rows = {tier: [] for tier in tier_specs}
                tier_chunk_index = {tier: 0 for tier in tier_specs}
                pf = pq.ParquetFile(parquet_path)

                def flush_tier(tier: str) -> None:
                    nonlocal staged_chunk_count, staged_bytes
                    rows = tier_rows[tier]
                    if not rows:
                        return
                    chunk_idx = tier_chunk_index[tier]
                    tier_chunk_index[tier] += 1
                    chunk_basename = f"{lc}_s{shard_idx:04d}_c{chunk_idx:05d}_{tier}"
                    name = f"{run_id}_{chunk_basename}" if run_id else chunk_basename
                    if name in completed_chunks:
                        print(f"    [resume] skip uploaded chunk {name}")
                        tier_rows[tier] = []
                        return
                    jsonl_path = work_root / "jsonl" / f"{name}.jsonl"
                    docs, source_tokens = _flush_jsonl_chunk(jsonl_path, rows)
                    tier_rows[tier] = []
                    if docs == 0:
                        _remove_paths([jsonl_path])
                        return

                    out_prefix = work_root / "bin" / name
                    out_prefix.parent.mkdir(parents=True, exist_ok=True)
                    print(f"    [tokenize] {name}: {docs:,} docs")
                    if not args.dry_run:
                        _run_preprocess(args, jsonl_path, out_prefix)

                    bin_path = out_prefix.with_name(out_prefix.name + "_text_document.bin")
                    idx_path = out_prefix.with_name(out_prefix.name + "_text_document.idx")
                    if args.dry_run:
                        bin_bytes = idx_bytes = 0
                        bin_sha = idx_sha = None
                    else:
                        bin_bytes = bin_path.stat().st_size
                        idx_bytes = idx_path.stat().st_size
                        bin_sha = _sha256(bin_path)
                        idx_sha = _sha256(idx_path)

                    entry = {
                        "name": name,
                        "language": lc,
                        "tier": tier,
                        "source": source_path,
                        "docs": docs,
                        "source_token_count": source_tokens,
                        "bin_bytes": bin_bytes,
                        "idx_bytes": idx_bytes,
                        "bin_sha256": bin_sha,
                        "idx_sha256": idx_sha,
                    }
                    entries.append(entry)
                    completed_chunks.add(name)
                    if not args.dry_run:
                        chunk_manifest = work_root / "chunk_manifests" / f"{name}.json"
                        _write_json(chunk_manifest, entry)
                        if args.local_only and local_dir:
                            _stage_file(local_dir, bin_path, f"bin/{bin_path.name}")
                            _stage_file(local_dir, idx_path, f"bin/{idx_path.name}")
                            _stage_file(local_dir, chunk_manifest, f"manifests/chunks/{name}.json")
                        else:
                            _stage_file(upload_stage_dir, bin_path, f"{repo_bin_dir}/{bin_path.name}")
                            _stage_file(upload_stage_dir, idx_path, f"{repo_bin_dir}/{idx_path.name}")
                            _stage_file(upload_stage_dir, chunk_manifest, f"{repo_chunk_manifest_dir}/{name}.json")
                            staged_chunk_count += 1
                            staged_bytes += bin_bytes + idx_bytes + chunk_manifest.stat().st_size
                            hit_chunk_cap = staged_chunk_count >= args.upload_batch_chunks
                            hit_byte_cap = bool(args.upload_batch_bytes and staged_bytes >= args.upload_batch_bytes)
                            if hit_chunk_cap or hit_byte_cap:
                                print(
                                    f"    [upload] flushing {staged_chunk_count} chunk(s), "
                                    f"{staged_bytes/1e9:.2f} GB staged",
                                    flush=True,
                                )
                                _flush_upload_batch(api, args, upload_stage_dir, staged_chunk_count)
                                staged_chunk_count = 0
                                staged_bytes = 0
                    if not args.keep_local:
                        _remove_paths([jsonl_path, bin_path, idx_path])

                for batch in pf.iter_batches(batch_size=args.batch_rows, columns=["text", "token_count"]):
                    texts = batch.column("text").to_pylist()
                    token_counts = batch.column("token_count").to_pylist()
                    for text, tok in zip(texts, token_counts):
                        if args.max_docs and total_seen_docs >= args.max_docs:
                            break
                        tc = int(tok or 0)
                        try:
                            tier = _tier_for_tokens(tc, tier_specs)
                        except ValueError as e:
                            print(f"[error] {e}", file=sys.stderr)
                            sys.exit(2)
                        tier_rows[tier].append({"text": text, "token_count": tc})
                        total_seen_docs += 1
                        tier_chunk_docs = chunk_docs_by_tier.get(tier, args.chunk_docs)
                        if len(tier_rows[tier]) >= tier_chunk_docs:
                            flush_tier(tier)
                    if args.max_docs and total_seen_docs >= args.max_docs:
                        break

                for tier in tier_specs:
                    flush_tier(tier)

                if not args.keep_local:
                    _remove_paths([parquet_path, work_root / "hf_cache" / ".cache"])
                if args.max_docs and total_seen_docs >= args.max_docs:
                    break
            if args.max_docs and total_seen_docs >= args.max_docs:
                break

        if not entries:
            print("[error] No tokenized chunks were produced.", file=sys.stderr)
            sys.exit(1)

        if not args.local_only:
            if staged_chunk_count:
                print(
                    f"    [upload] flushing final {staged_chunk_count} chunk(s), "
                    f"{staged_bytes/1e9:.2f} GB staged",
                    flush=True,
                )
            _flush_upload_batch(api, args, upload_stage_dir, staged_chunk_count)

        if args.skip_final_metadata:
            print(
                f"\nUploaded streamed chunks for this continuation run: {len(entries)} chunk(s).",
                flush=True,
            )
            print("[metadata] skipped final mix/manifests upload by request.", flush=True)
            return

        manifest_dir = local_dir if args.local_only and local_dir else work_root / "final"
        mix_dir = manifest_dir / "mix"
        manifests_dir = manifest_dir / "manifests"
        local_bin_dir = str((manifest_dir / "bin").resolve()) if args.local_only else repo_bin_dir
        mix = _build_mix(entries, tier_specs=tier_specs, local_bin_dir=local_bin_dir)
        _write_json(mix_dir / "data_mix.json", mix)
        (mix_dir / "data_path.args").write_text(mix["data_path_args"] + "\n", encoding="utf-8")
        _write_json(manifests_dir / "stream_manifest.json", {
            "created_at_unix": int(time.time()),
            "artifact_format": "openeuro-longctx-megatron-stream-v1",
            "source_dataset": DATASET_ID,
            "tokenizer_type": args.tokenizer_type,
            "tokenizer_model": args.tokenizer_model,
            "vocab_size": args.vocab_size,
            "tier_spec_source": tier_spec_source,
            "tiers": tier_specs,
            "entries": entries,
        })
        _write_readme(manifest_dir / "README.md", args, entries)

        if args.local_only:
            print(f"[local-only] wrote artifact to {manifest_dir}")
        elif args.dry_run:
            print(f"[dry-run] would upload mix/manifests/README from {manifest_dir}")
        else:
            _upload_folder_with_retry(
                api,
                args,
                folder_path=str(manifest_dir),
                commit_message="Upload streamed Megatron long-context manifests",
            )

        print(f"\nUploaded streamed artifact metadata for {len(entries)} chunks.")
        print(f"DATA_PATH entries: {len(entries)}")
        print(f"Manifest work dir: {manifest_dir}")
    finally:
        if not args.keep_work_dir and not args.keep_local:
            _remove_paths([work_root])
