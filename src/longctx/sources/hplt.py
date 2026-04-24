"""HPLT v2.0 cleaned (`HPLT/HPLT2.0_cleaned`) adapter.

Covers all 38 OpenEuroLLM target languages. This is the **primary** source
for the three languages missing from FinePDFs-Edu (ga, sq, lb).

HPLT v2 uses ISO 639-3 + ISO 15924 script suffixes (e.g. `gle_Latn`).
Note Albanian: HPLT uses `als_Latn` (Tosk Albanian), not `sqi_Latn`.
Note Norwegian: HPLT has both `nob_Latn` (Bokmål) and `nno_Latn` (Nynorsk);
we default to Bokmål to match OpenEuroLLM conventions.

Record schema includes many metadata fields; we only read `text`.
"""

from __future__ import annotations

from pathlib import Path

from longctx.sources.base import write_jsonl_from_iter

NAME = "hplt"
DATASET_ID = "HPLT/HPLT2.0_cleaned"
GATED = False

# ISO 639-1 → HPLT subset (ISO 639-3 + script)
SUPPORTED: dict[str, str] = {
    # EU official (present in FinePDFs-Edu too)
    "bg": "bul_Cyrl", "hr": "hrv_Latn", "cs": "ces_Latn", "da": "dan_Latn",
    "nl": "nld_Latn", "et": "est_Latn", "fi": "fin_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "hu": "hun_Latn", "it": "ita_Latn",
    "lv": "lav_Latn", "lt": "lit_Latn", "mt": "mlt_Latn", "pl": "pol_Latn",
    "pt": "por_Latn", "ro": "ron_Latn", "sk": "slk_Latn", "sl": "slv_Latn",
    "es": "spa_Latn", "sv": "swe_Latn", "en": "eng_Latn",
    # Additional European
    "eu": "eus_Latn", "bs": "bos_Latn", "ca": "cat_Latn", "gl": "glg_Latn",
    "is": "isl_Latn", "mk": "mkd_Cyrl", "no": "nob_Latn", "ru": "rus_Cyrl",
    "sr": "srp_Cyrl", "tr": "tur_Latn", "uk": "ukr_Cyrl", "cy": "cym_Latn",
    # Three missing from FinePDFs-Edu — this is the main reason to use HPLT.
    "ga": "gle_Latn",   # Irish
    "sq": "als_Latn",   # Albanian (Tosk)
    "lb": "ltz_Latn",   # Luxembourgish
    # EU official still missing everywhere: none for HPLT v2.
}


def fetch(
    lang: str,
    output_path: Path,
    *,
    sample: bool = False,
    max_docs: int | None = None,
) -> dict:
    """Stream HPLT for `lang` into `output_path` as JSONL.

    `sample=True` with no explicit max_docs caps at 1000 docs (fast probe).
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "HPLT adapter needs `datasets`. Install with: pip install datasets"
        ) from e

    if lang not in SUPPORTED:
        raise ValueError(
            f"HPLT adapter has no mapping for lang '{lang}'. "
            f"Add it to longctx.sources.hplt.SUPPORTED."
        )
    config = SUPPORTED[lang]

    if sample and max_docs is None:
        max_docs = 1000

    ds = load_dataset(DATASET_ID, config, split="train", streaming=True)
    # HPLT records carry extra metadata (doc_scores, lang probs, …). We only
    # keep `text` — write_jsonl_from_iter drops the rest.
    return write_jsonl_from_iter(ds, output_path, max_docs=max_docs)
