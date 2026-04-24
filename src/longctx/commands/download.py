"""Download FinePDFs-Edu parquet shards for target languages."""

from __future__ import annotations

import os
import shutil as _sh
from pathlib import Path

from longctx.languages import DATASET_ID, LANG_MAP, MISSING_LANGS, parse_languages
from longctx.utils import check_disk_space, write_json


def _get_lang_file_list(api, lang_code: str) -> list[dict]:
    hf_dir = LANG_MAP[lang_code]
    try:
        files = list(api.list_repo_tree(
            DATASET_ID, repo_type="dataset",
            path_in_repo=f"data/{hf_dir}/train",
        ))
        return [{"path": f.path, "size": getattr(f, "size", 0)} for f in files]
    except Exception:
        return []


def cmd_download(args) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    langs = parse_languages(args.languages)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    disk = check_disk_space(str(output_dir))
    print(f"Free disk: {disk['free_gb']:.1f} GB")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[info] HF_TOKEN not set — using unauthenticated (rate-limited) downloads.")

    stats: dict[str, dict] = {}
    for lc in langs:
        if lc in MISSING_LANGS:
            print(f"[skip] {lc}: {MISSING_LANGS[lc]}")
            continue

        hf_code = LANG_MAP[lc]
        files = _get_lang_file_list(api, lc)
        if not files:
            print(f"[skip] {lc} ({hf_code}): no files found")
            continue

        to_download = files[: args.shards] if args.sample else files
        lang_dir = output_dir / lc
        lang_dir.mkdir(exist_ok=True)

        total_bytes = sum(f["size"] for f in to_download)
        print(f"\n[{lc}] {hf_code}: {len(to_download)}/{len(files)} shards "
              f"({total_bytes/1e6:.0f} MB)")

        downloaded: list[str] = []
        for f in to_download:
            dest = lang_dir / Path(f["path"]).name
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  [cache] {dest.name}")
                downloaded.append(str(dest))
                continue
            print(f"  [dl]    {f['path']} ({f['size']/1e6:.0f} MB) ...",
                  end=" ", flush=True)
            try:
                local = hf_hub_download(
                    repo_id=DATASET_ID,
                    filename=f["path"],
                    repo_type="dataset",
                    local_dir=str(output_dir / lc / "hf_cache"),
                    token=hf_token,
                )
                _sh.move(local, str(dest))
                print(f"ok ({dest.stat().st_size/1e6:.0f} MB)")
                downloaded.append(str(dest))
            except Exception as e:
                print(f"ERROR: {e}")

        stats[lc] = {"hf_code": hf_code, "files": downloaded}

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, stats)
    print(f"\nManifest written to {manifest_path}")
