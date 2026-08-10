#!/usr/bin/env python3
"""Fail-closed, read-only preflight for a LUMI sparse DSA round trip."""

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess


def complete_checkpoint(root: Path, expected_iteration: int):
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise RuntimeError(f"missing checkpoint marker: {marker}")
    actual = int(marker.read_text().strip())
    if actual != expected_iteration:
        raise RuntimeError(
            f"checkpoint marker is {actual}, expected {expected_iteration}: {root}"
        )
    iteration = root / f"iter_{actual:07d}"
    if not iteration.is_dir() or not any(iteration.glob("*.distcp")):
        raise RuntimeError(f"checkpoint has no distributed shards: {iteration}")
    if not (iteration / ".metadata").is_file() or not (iteration / "common.pt").is_file():
        raise RuntimeError(f"checkpoint metadata/common state is incomplete: {iteration}")
    return iteration


def validate_data_blend(path: Path):
    fields = path.read_text().split()
    if not fields or len(fields) % 2:
        raise RuntimeError(f"data blend must contain weight/prefix pairs: {path}")
    missing = []
    for offset in range(0, len(fields), 2):
        float(fields[offset])
        prefix = Path(fields[offset + 1])
        for suffix in (".bin", ".idx"):
            candidate = Path(f"{prefix}{suffix}")
            if not candidate.is_file():
                missing.append(str(candidate))
    if missing:
        raise RuntimeError(f"data blend has {len(missing)} missing files, e.g. {missing[:3]}")
    return len(fields) // 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsa-dir", type=Path, required=True)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--warm-checkpoint", type=Path, required=True)
    parser.add_argument("--data-blend", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--cp-size", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--routed-blocks", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    args = parser.parse_args()

    expected_revision = (args.dsa_dir / "MEGATRON_REVISION").read_text().strip()
    actual_revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={args.megatron_root.resolve()}", "-C",
         str(args.megatron_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"Megatron revision mismatch: expected {expected_revision}, got {actual_revision}"
        )

    module = importlib.import_module("gpt_builders")
    module_path = Path(module.__file__).resolve()
    expected_module = (args.dsa_dir / "gpt_builders.py").resolve()
    if module_path != expected_module:
        raise RuntimeError(
            f"wrong gpt_builders import: expected {expected_module}, got {module_path}"
        )
    pretrain = importlib.util.find_spec("pretrain_gpt")
    expected_pretrain = (args.megatron_root / "pretrain_gpt.py").resolve()
    if pretrain is None or Path(pretrain.origin).resolve() != expected_pretrain:
        raise RuntimeError(
            f"wrong pretrain_gpt module: expected {expected_pretrain}, "
            f"got {None if pretrain is None else pretrain.origin}"
        )

    complete_checkpoint(args.warm_checkpoint, 300)
    pairs = validate_data_blend(args.data_blend)
    if args.seq_length % (2 * args.cp_size * args.block_size):
        raise RuntimeError("sequence length is not aligned to Megatron CP halves and DSA blocks")
    expected_topk = args.block_size * (1 + args.routed_blocks)
    if args.topk != expected_topk:
        raise RuntimeError(f"topk {args.topk} does not match routed candidate count {expected_topk}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "overlay": str(module_path),
                "pretrain_module": str(expected_pretrain),
                "megatron_revision": actual_revision,
                "source_checkpoint": str(args.warm_checkpoint),
                "data_pairs": pairs,
                "seq_length": args.seq_length,
                "cp_size": args.cp_size,
                "cp_local_length": args.seq_length // args.cp_size,
                "block_size": args.block_size,
                "routed_blocks": args.routed_blocks,
                "topk": args.topk,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
