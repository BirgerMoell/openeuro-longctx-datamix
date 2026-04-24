"""Estimate dataset sizes and disk-space requirements."""

from __future__ import annotations

import os

from longctx.languages import DATASET_ID, LANG_MAP, MISSING_LANGS, parse_languages
from longctx.utils import check_disk_space


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


def cmd_estimate(args) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    langs = parse_languages(args.languages)
    probe = args.output_dir if os.path.exists(args.output_dir) else "/"
    disk = check_disk_space(probe)
    print(f"Disk:  {disk['free_gb']:.1f} GB free  /  {disk['total_gb']:.1f} GB total")
    print()

    print(f"{'Lang':<6} {'HF code':<12} {'Files':>6} {'Full GB':>9} {'1-shard MB':>11}")
    print("-" * 50)

    total_full = 0.0
    total_sample = 0.0
    missing: list[str] = []

    for lc in langs:
        if lc in MISSING_LANGS:
            missing.append(lc)
            continue
        files = _get_lang_file_list(api, lc)
        if not files:
            print(f"{lc:<6} {'NOT FOUND':<12}")
            continue
        full_gb = sum(f["size"] for f in files) / 1e9
        first_mb = files[0]["size"] / 1e6
        total_full += full_gb
        total_sample += files[0]["size"] / 1e9
        hf_code = LANG_MAP[lc]
        print(f"{lc:<6} {hf_code:<12} {len(files):>6} {full_gb:>9.2f} {first_mb:>11.0f}")

    print("-" * 50)
    print(f"{'TOTAL':<6} {'':<12} {'':<6} {total_full:>9.2f} {total_sample*1000:>11.0f}")
    print()
    print(f"Full download:   {total_full:.1f} GB  (need ~{total_full*1.1:.1f} GB with overhead)")
    print(f"Sample download: {total_sample:.1f} GB  (1 shard per language)")

    if total_full * 1.1 > disk["free_gb"]:
        print(f"\n[warn] Full download ({total_full:.1f} GB) exceeds free space "
              f"({disk['free_gb']:.1f} GB). Use `--sample` or `--shards N`.")
    elif total_sample * 1.1 > disk["free_gb"]:
        print(f"\n[error] Even sample download exceeds free space.")
    else:
        print(f"\n[ok] Free space is sufficient.")

    if missing:
        print(f"\nLanguages not in {DATASET_ID}: {', '.join(missing)}")
        for lc in missing:
            print(f"  {lc}: {MISSING_LANGS[lc]}")
