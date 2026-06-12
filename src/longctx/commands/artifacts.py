"""Package Megatron tokenized artifacts for Hugging Face Hub handoff.

The artifact format is intentionally plain:

  bin/                 Megatron *.bin/*.idx files
  mix/                 data_mix.json, data_mix.txt, data_path.args
  manifests/           checksums and build metadata

Megatron training still uses local files. The HF Hub is only the transport and
artifact store; `download` rewrites `data_path.args` for the machine that will
train, for example LUMI.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_artifact_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.iterdir()):
        target = dst / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        elif path.is_file():
            shutil.copy2(path, target)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_checksums(root: Path) -> dict:
    checksums = {}
    for path in _iter_artifact_files(root):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".cache/"):
            continue
        if rel in {"manifests/checksums.json", "manifests/checksums.sha256"}:
            continue
        checksums[rel] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    _write_json(root / "manifests" / "checksums.json", checksums)
    with (root / "manifests" / "checksums.sha256").open("w", encoding="utf-8") as f:
        for rel, item in sorted(checksums.items()):
            f.write(f"{item['sha256']}  {rel}\n")
    return checksums


def _rewrite_mix_paths(mix_json: Path, bin_dir: Path, mix_dir: Path) -> dict:
    manifest = _read_json(mix_json)
    suffix = manifest.get("suffix", "_text_document")
    languages = manifest.get("languages", {})

    for lc, item in languages.items():
        item["prefix"] = str((bin_dir / f"{lc}{suffix}").resolve())
    manifest["bin_dir"] = str(bin_dir.resolve())

    tokens: list[str] = []
    for lc, item in sorted(languages.items(), key=lambda kv: -float(kv[1]["weight"])):
        tokens.extend([f"{float(item['weight']):.6f}", item["prefix"]])

    mix_dir.mkdir(parents=True, exist_ok=True)
    (mix_dir / "data_path.args").write_text(" ".join(tokens) + "\n", encoding="utf-8")
    _write_json(mix_dir / "data_mix.json", manifest)

    lines = [
        "# data mix for Megatron --data-path",
        f"# bin_dir = {bin_dir.resolve()}",
        f"# alpha   = {manifest.get('alpha')}",
        f"# floor   = {manifest.get('floor')}",
        f"# total_bin_bytes = {int(manifest.get('total_bin_bytes', 0)):,}",
        "",
        f"{'Lang':<12} {'Bin MB':>10} {'Natural%':>9} {'Weight%':>9} {'Prefix'}",
        "-" * 96,
    ]
    for lc, item in sorted(languages.items(), key=lambda kv: -float(kv[1]["weight"])):
        mb = int(item.get("bin_bytes", 0)) / 1e6
        nat = 100 * float(item.get("natural_p", 0.0))
        w = 100 * float(item.get("weight", 0.0))
        lines.append(f"{lc:<12} {mb:>10.1f} {nat:>8.2f}% {w:>8.2f}% {item['prefix']}")
    lines += ["", "# Paste into your Megatron launcher:", f"DATA_PATH=\"{' '.join(tokens)}\""]
    (mix_dir / "data_mix.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def cmd_artifacts_pack(args) -> None:
    bin_dir = Path(args.bin_dir).expanduser().resolve()
    mix_dir = Path(args.mix_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not bin_dir.exists():
        print(f"[error] bin dir does not exist: {bin_dir}")
        sys.exit(2)
    if not (mix_dir / "data_mix.json").exists():
        print(f"[error] mix manifest missing: {mix_dir / 'data_mix.json'}")
        print("        Run `longctx mix` before packaging artifacts.")
        sys.exit(2)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"[error] output dir is not empty: {out_dir}")
        print("        Pass --force to merge/overwrite files in it.")
        sys.exit(2)

    (out_dir / "bin").mkdir(parents=True, exist_ok=True)
    (out_dir / "mix").mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)

    for pattern in ("*.bin", "*.idx", "*.json"):
        for path in sorted(bin_dir.glob(pattern)):
            shutil.copy2(path, out_dir / "bin" / path.name)
    _copy_tree_contents(mix_dir, out_dir / "mix")

    mix_manifest = _read_json(mix_dir / "data_mix.json")
    build_info = {
        "created_at_unix": int(time.time()),
        "artifact_format": "openeuro-longctx-megatron-v1",
        "source_bin_dir": str(bin_dir),
        "source_mix_dir": str(mix_dir),
        "tokenizer_type": args.tokenizer_type,
        "tokenizer_model": args.tokenizer_model,
        "tokenizer_revision": args.tokenizer_revision,
        "vocab_size": args.vocab_size,
        "megatron_path": args.megatron_path,
        "megatron_git_commit": None,
        "mix": mix_manifest,
        "notes": args.notes or "",
    }
    if args.megatron_path:
        try:
            build_info["megatron_git_commit"] = subprocess.check_output(
                ["git", "-C", args.megatron_path, "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            build_info["megatron_git_commit"] = None
    _write_json(out_dir / "manifests" / "build_info.json", build_info)

    checksums = _write_checksums(out_dir)

    total = sum(item["bytes"] for item in checksums.values())
    print(f"Packed artifact: {out_dir}")
    print(f"Files: {len(checksums)}")
    print(f"Total bytes: {total:,} ({total / 1e9:.2f} GB)")


def cmd_artifacts_upload(args) -> None:
    from huggingface_hub import HfApi, create_repo

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"[error] artifact folder does not exist: {folder}")
        sys.exit(2)

    create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api = HfApi()
    print(f"Uploading {folder} → dataset repo {args.repo_id}")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        ignore_patterns=args.ignore_patterns or None,
    )
    print("Upload complete.")


def cmd_artifacts_download(args) -> None:
    from huggingface_hub import snapshot_download

    out_dir = Path(args.output_dir).expanduser().resolve()
    local = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(out_dir),
        allow_patterns=args.allow_patterns or None,
    )
    root = Path(local)
    if args.path_in_repo:
        root = root / args.path_in_repo

    bin_dir = (root / "bin").resolve()
    mix_dir = (root / "mix").resolve()
    mix_json = mix_dir / "data_mix.json"
    if not mix_json.exists():
        print(f"[error] downloaded artifact has no mix/data_mix.json under {root}")
        sys.exit(2)

    _rewrite_mix_paths(mix_json, bin_dir, mix_dir)
    _write_checksums(root)
    print(f"Downloaded artifact: {root}")
    print(f"Rewrote local data path: {mix_dir / 'data_path.args'}")
    print((mix_dir / "data_path.args").read_text(encoding="utf-8").strip())
