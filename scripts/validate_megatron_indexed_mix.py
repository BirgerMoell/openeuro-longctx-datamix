#!/usr/bin/env python3
"""Validate a weighted Megatron ``--data-path`` mix by reading every dataset.

Run this with the same Megatron checkout/container used for training.  File
existence and checksums are necessary but not sufficient: constructing
``IndexedDataset`` and reading records from both ends also checks that the
published ``.idx`` metadata agrees with its paired ``.bin`` payload.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np

from megatron.core.datasets.indexed_dataset import IndexedDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path-args",
        type=Path,
        required=True,
        help="File containing alternating Megatron weights and dataset prefixes.",
    )
    parser.add_argument("--expected-prefixes", type=int, default=None)
    parser.add_argument("--weight-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--gpt-sequence-length",
        type=int,
        default=None,
        help="Also construct a real Megatron GPT blend at this sequence length.",
    )
    parser.add_argument("--gpt-train-samples", type=int, default=302)
    parser.add_argument("--gpt-cache-dir", type=Path, default=None)
    parser.add_argument("--mid-level-dataset-surplus", type=float, default=0.5)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable validation report.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def validate_gpt_builder(
    prefixes: list[Path], weights: list[float], args: argparse.Namespace
) -> dict[str, object]:
    """Build and sample the same GPT dataset objects used by training."""
    if args.gpt_cache_dir is None:
        fail("--gpt-cache-dir is required with --gpt-sequence-length")
    if args.gpt_sequence_length <= 0 or args.gpt_train_samples <= 0:
        fail("GPT sequence length and train sample count must be positive")

    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.training.tokenizer.tokenizer import _NullTokenizer
    import torch

    args.gpt_cache_dir.mkdir(parents=True, exist_ok=True)
    initialized_process_group = False
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        rendezvous = args.gpt_cache_dir / ".single_rank_rendezvous"
        rendezvous.unlink(missing_ok=True)
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{rendezvous.resolve()}",
            rank=0,
            world_size=1,
        )
        initialized_process_group = True
    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=args.gpt_sequence_length,
        blend=([str(prefix) for prefix in prefixes], weights),
        blend_per_split=None,
        split="100,0,0",
        path_to_cache=str(args.gpt_cache_dir.resolve()),
        tokenizer=_NullTokenizer(vocab_size=262144),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        num_dataset_builder_threads=1,
    )
    train, valid, test = BlendedMegatronDatasetBuilder(
        GPTDataset,
        [args.gpt_train_samples, 0, 0],
        lambda: True,
        config,
    ).build()
    if train is None or len(train) < args.gpt_train_samples:
        fail(
            "GPT builder returned too few train samples: "
            f"requested={args.gpt_train_samples}, built={0 if train is None else len(train)}"
        )
    if valid is not None or test is not None:
        fail("GPT builder unexpectedly produced validation/test data for split 100,0,0")

    first = train[0]
    last = train[args.gpt_train_samples - 1]
    for name, sample in (("first", first), ("last", last)):
        for key in ("tokens", "labels", "loss_mask", "position_ids"):
            if key not in sample:
                fail(f"{name} GPT sample is missing {key!r}")
            if sample[key].numel() != args.gpt_sequence_length:
                fail(
                    f"{name} GPT sample {key!r} has {sample[key].numel()} values; "
                    f"expected {args.gpt_sequence_length}"
                )

    result = {
        "sequence_length": args.gpt_sequence_length,
        "requested_train_samples": args.gpt_train_samples,
        "built_train_samples": len(train),
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "first_token_min": int(first["tokens"].min()),
        "first_token_max": int(first["tokens"].max()),
        "last_token_min": int(last["tokens"].min()),
        "last_token_max": int(last["tokens"].max()),
    }
    if initialized_process_group:
        torch.distributed.destroy_process_group()
    print(
        "MEGATRON_GPT_BLEND_PASS "
        f"sequence_length={args.gpt_sequence_length} "
        f"samples={len(train)} surplus={args.mid_level_dataset_surplus}"
    )
    return result


def main() -> None:
    args = parse_args()
    tokens = shlex.split(args.data_path_args.read_text(encoding="utf-8"))
    if not tokens or len(tokens) % 2:
        fail("data-path file must contain alternating weight/prefix tokens")

    weights: list[float] = []
    prefixes: list[Path] = []
    for offset in range(0, len(tokens), 2):
        try:
            weight = float(tokens[offset])
        except ValueError as exc:
            fail(f"invalid weight {tokens[offset]!r}: {exc}")
        if not np.isfinite(weight) or weight <= 0:
            fail(f"weight must be finite and positive, got {weight}")
        prefix = Path(tokens[offset + 1])
        if not prefix.is_absolute():
            fail(f"prefix is not absolute: {prefix}")
        weights.append(weight)
        prefixes.append(prefix)

    if args.expected_prefixes is not None and len(prefixes) != args.expected_prefixes:
        fail(f"expected {args.expected_prefixes} prefixes, found {len(prefixes)}")
    if len(set(prefixes)) != len(prefixes):
        fail("duplicate dataset prefixes found")

    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > args.weight_tolerance:
        fail(f"weights sum to {weight_sum:.12f}, expected 1.0")

    results: list[dict[str, object]] = []
    for number, (weight, prefix) in enumerate(zip(weights, prefixes), start=1):
        bin_path = Path(f"{prefix}.bin")
        idx_path = Path(f"{prefix}.idx")
        for path in (bin_path, idx_path):
            if not path.is_file():
                fail(f"missing dataset file: {path}")

        try:
            dataset = IndexedDataset(str(prefix))
            sequence_lengths = dataset.sequence_lengths
            sequence_count = len(dataset)
            if sequence_count <= 0:
                fail(f"empty dataset: {prefix}")
            if len(sequence_lengths) != sequence_count:
                fail(
                    f"index length mismatch for {prefix}: "
                    f"len(dataset)={sequence_count}, lengths={len(sequence_lengths)}"
                )
            first = np.asarray(dataset[0])
            last = np.asarray(dataset[sequence_count - 1])
        except SystemExit:
            raise
        except Exception as exc:
            fail(f"Megatron could not read {prefix}: {type(exc).__name__}: {exc}")

        if first.size != int(sequence_lengths[0]):
            fail(f"first record length disagrees with index for {prefix}")
        if last.size != int(sequence_lengths[-1]):
            fail(f"last record length disagrees with index for {prefix}")

        result = {
            "prefix": str(prefix),
            "weight": weight,
            "sequences": sequence_count,
            "tokens": int(sequence_lengths.sum(dtype=np.int64)),
            "min_sequence_length": int(sequence_lengths.min()),
            "max_sequence_length": int(sequence_lengths.max()),
            "first_sequence_length": int(first.size),
            "last_sequence_length": int(last.size),
            "boundary_token_min": int(min(first.min(), last.min())),
            "boundary_token_max": int(max(first.max(), last.max())),
        }
        results.append(result)
        print(
            f"[{number:02d}/{len(prefixes):02d}] OK {prefix.name} "
            f"sequences={sequence_count} tokens={result['tokens']} "
            f"lengths={result['min_sequence_length']}..{result['max_sequence_length']}"
        )

    report = {
        "status": "PASS",
        "data_path_args": str(args.data_path_args.resolve()),
        "prefix_count": len(prefixes),
        "weight_sum": weight_sum,
        "total_sequences": sum(int(item["sequences"]) for item in results),
        "total_tokens": sum(int(item["tokens"]) for item in results),
        "datasets": results,
    }
    if args.gpt_sequence_length is not None:
        report["gpt_builder"] = validate_gpt_builder(prefixes, weights, args)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "MEGATRON_INDEXED_MIX_PASS "
        f"prefixes={report['prefix_count']} weights={weight_sum:.6f} "
        f"sequences={report['total_sequences']} tokens={report['total_tokens']}"
    )


if __name__ == "__main__":
    main()
