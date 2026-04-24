"""Helpers shared by all source adapters."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def approx_tokens(text: str) -> int:
    """~4 chars per token heuristic. Close enough for filtering; the tokenize
    step produces exact counts via the Megatron tokenizer."""
    return max(1, len(text) // 4)


def write_jsonl_from_iter(
    records: Iterable[dict[str, Any]],
    output_path: Path,
    *,
    max_docs: int | None = None,
    min_chars: int = 50,
) -> dict[str, int]:
    """Stream records → JSONL with canonical schema {text, token_count}.

    Accepts records with a 'text' field (required). If the record already has
    an integer 'token_count', we keep it; otherwise we approximate.

    Returns a stats dict with docs / chars / approx_tokens counts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_chars = 0
    n_tokens_approx = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            if max_docs is not None and n_docs >= max_docs:
                break
            text = (rec.get("text") or "").strip()
            if len(text) < min_chars:
                continue
            tc = rec.get("token_count") or rec.get("tokens")
            if not isinstance(tc, int) or tc <= 0:
                tc = approx_tokens(text)
            out = {"text": text, "token_count": int(tc)}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_docs += 1
            n_chars += len(text)
            n_tokens_approx += int(tc)

    return {
        "docs": n_docs,
        "chars": n_chars,
        "tokens_approx": n_tokens_approx,
    }
