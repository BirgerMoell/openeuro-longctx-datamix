"""Pluggable data-source adapters.

Each adapter exposes:
    NAME:  str
    SUPPORTED: dict[iso_639_1, adapter_specific_config]
    fetch(lang, output_path, *, sample=False, max_docs=None) -> dict

Adapters write `{lc}.jsonl` with one JSON record per line:
    {"text": "<document>", "token_count": int}
Downstream commands (filter-long, tokenize, mix) are source-agnostic.

To register a new adapter: add it here and point the SOURCES dict at it.
"""

from __future__ import annotations

from longctx.sources import culturax, hplt

SOURCES = {
    "hplt":     hplt,
    "culturax": culturax,
}

__all__ = ["SOURCES", "hplt", "culturax"]
