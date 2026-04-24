"""CulturaX (`uonlp/CulturaX`) adapter.

Fallback / supplementary source. Config names are ISO 639-1 so `lang` maps
1:1. Covers all 38 OpenEuroLLM targets.

⚠ CulturaX is **gated** on the Hugging Face Hub — you must visit
https://huggingface.co/datasets/uonlp/CulturaX, accept the terms, and export
HF_TOKEN=hf_... before fetching.

Record schema: {text, url, timestamp, source}.
"""

from __future__ import annotations

import os
from pathlib import Path

from longctx.sources.base import write_jsonl_from_iter

NAME = "culturax"
DATASET_ID = "uonlp/CulturaX"
GATED = True

# ISO 639-1 codes passed through unchanged; all 38 targets are present.
SUPPORTED: dict[str, str] = {
    lc: lc for lc in [
        # EU official (present in FinePDFs-Edu)
        "bg", "hr", "cs", "da", "nl", "et", "fi", "fr", "de", "el", "hu",
        "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "en",
        # Additional European
        "eu", "bs", "ca", "gl", "is", "mk", "no", "ru", "sr", "tr", "uk", "cy",
        # Three missing from FinePDFs-Edu — CulturaX has them all.
        "ga", "sq", "lb",
    ]
}


def fetch(
    lang: str,
    output_path: Path,
    *,
    sample: bool = False,
    max_docs: int | None = None,
) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "CulturaX adapter needs `datasets`. Install with: pip install datasets"
        ) from e

    if lang not in SUPPORTED:
        raise ValueError(f"CulturaX adapter has no mapping for lang '{lang}'.")
    config = SUPPORTED[lang]

    if not os.environ.get("HF_TOKEN"):
        print("[warn] CulturaX is gated. If fetch fails with 401, accept the "
              "terms at https://huggingface.co/datasets/uonlp/CulturaX and "
              "`export HF_TOKEN=hf_...`.")

    if sample and max_docs is None:
        max_docs = 1000

    ds = load_dataset(
        DATASET_ID, config, split="train", streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )
    return write_jsonl_from_iter(ds, output_path, max_docs=max_docs)
