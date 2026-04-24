"""Language coverage for OpenEuroLLM context-extension data mix.

LANG_MAP    — ISO 639-1 → FinePDFs-Edu folder (ISO 639-3 + script).
MISSING_LANGS — OpenEuroLLM targets that are NOT in FinePDFs-Edu.

If you add a new source adapter (HPLT, CulturaX, Wikipedia, …), extend
LANG_MAP or add a parallel mapping in the adapter module — do not
silently change the folder names here; downstream code keys on them.
"""

from __future__ import annotations

# EU official (24) + additional European coverage = 35 languages present in FinePDFs-Edu.
LANG_MAP: dict[str, str] = {
    # --- 22 EU official languages present in FinePDFs-Edu (ga + lb missing) ---
    "bg": "bul_Cyrl",   # Bulgarian
    "hr": "hrv_Latn",   # Croatian
    "cs": "ces_Latn",   # Czech
    "da": "dan_Latn",   # Danish
    "nl": "nld_Latn",   # Dutch
    "et": "ekk_Latn",   # Estonian
    "fi": "fin_Latn",   # Finnish
    "fr": "fra_Latn",   # French
    "de": "deu_Latn",   # German
    "el": "ell_Grek",   # Greek
    "hu": "hun_Latn",   # Hungarian
    "it": "ita_Latn",   # Italian
    "lv": "lvs_Latn",   # Latvian
    "lt": "lit_Latn",   # Lithuanian
    "mt": "mlt_Latn",   # Maltese
    "pl": "pol_Latn",   # Polish
    "pt": "por_Latn",   # Portuguese
    "ro": "ron_Latn",   # Romanian
    "sk": "slk_Latn",   # Slovak
    "sl": "slv_Latn",   # Slovenian
    "es": "spa_Latn",   # Spanish
    "sv": "swe_Latn",   # Swedish
    "en": "eng_Latn",   # English (anchor / replay)
    # --- Additional European coverage ---
    "eu": "eus_Latn",   # Basque
    "bs": "bos_Latn",   # Bosnian
    "ca": "cat_Latn",   # Catalan
    "gl": "glg_Latn",   # Galician
    "is": "isl_Latn",   # Icelandic
    "mk": "mkd_Cyrl",   # Macedonian
    "no": "nob_Latn",   # Norwegian Bokmål (nno_Latn = Nynorsk also exists upstream)
    "ru": "rus_Cyrl",   # Russian
    "sr": "srp_Cyrl",   # Serbian
    "tr": "tur_Latn",   # Turkish
    "uk": "ukr_Cyrl",   # Ukrainian
    "cy": "cym_Latn",   # Welsh
}

# OpenEuroLLM targets without FinePDFs-Edu coverage (need a different source).
MISSING_LANGS: dict[str, str] = {
    "ga": "Irish Gaelic (gle) — not in finepdfs-edu; use HPLT or CulturaX",
    "sq": "Albanian (sqi) — not in finepdfs-edu; use HPLT or CulturaX",
    "lb": "Luxembourgish (ltz) — not in finepdfs-edu; use HPLT or Wikipedia",
}

DATASET_ID = "HuggingFaceFW/finepdfs-edu"

# NLTK punkt language names for Megatron's --lang flag (only used when
# --split-sentences is on, which we do NOT recommend for LLM pretraining).
# Kept for completeness; safe to ignore.
NLTK_LANG = {
    "bg": "bulgarian", "cs": "czech", "da": "danish", "nl": "dutch",
    "en": "english", "et": "estonian", "fi": "finnish", "fr": "french",
    "de": "german", "el": "greek", "it": "italian", "no": "norwegian",
    "pl": "polish", "pt": "portuguese", "ru": "russian", "sl": "slovene",
    "es": "spanish", "sv": "swedish", "tr": "turkish",
}


def parse_languages(lang_str: str | None) -> list[str]:
    """Parse 'bg,fr,de' → ['bg','fr','de']. None → all supported langs."""
    if not lang_str:
        return list(LANG_MAP.keys())
    langs = [l.strip() for l in lang_str.split(",") if l.strip()]
    unknown = [l for l in langs if l not in LANG_MAP and l not in MISSING_LANGS]
    if unknown:
        print(f"[warn] unknown language codes: {unknown}")
    return langs
