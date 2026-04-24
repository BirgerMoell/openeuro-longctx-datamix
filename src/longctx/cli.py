"""Top-level CLI dispatcher for `longctx`.

Subcommands:
  estimate     — show dataset sizes and disk-space check
  download     — pull FinePDFs-Edu parquets for the target languages
  convert      — parquets → Megatron JSONL (streaming, preserves token_count)
  filter-long  — keep only docs >= min tokens (context-extension data)
  tokenize     — tokenize JSONL → Megatron .bin/.idx via preprocess_data.py
  mix          — emit weighted --data-path block for Megatron training
  run          — download + convert (+ optional filter) end-to-end
"""

from __future__ import annotations

import argparse

from longctx import __version__
from longctx.commands import (
    cmd_convert,
    cmd_download,
    cmd_estimate,
    cmd_filter_long,
    cmd_mix,
    cmd_run,
    cmd_tokenize,
)

DEFAULT_OUTPUT_DIR = "./data/raw"
DEFAULT_MEGATRON_DIR = "./data/megatron"
DEFAULT_LONG_DIR = "./data/long"
DEFAULT_BIN_DIR = "./data/bin"
DEFAULT_MIX_DIR = "./data/mix"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="longctx",
        description=(
            "Build a long-context data mix from FinePDFs-Edu for OpenEuroLLM / "
            "NVIDIA-Megatron-LM (https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"longctx {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # ── estimate ─────────────────────────────────────────────────────────────
    pe = sub.add_parser("estimate", help="Show dataset sizes and disk-space check")
    pe.add_argument("--languages", default=None,
                    help="Comma-separated ISO 639-1 codes (default: all 35 supported)")
    pe.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    pe.set_defaults(func=cmd_estimate)

    # ── download ─────────────────────────────────────────────────────────────
    pd = sub.add_parser("download", help="Download parquet shards for target languages")
    pd.add_argument("--languages", default=None)
    pd.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    pd.add_argument("--sample", action="store_true",
                    help="Download only --shards shards per language (default 1)")
    pd.add_argument("--shards", type=int, default=1)
    pd.set_defaults(func=cmd_download)

    # ── convert ──────────────────────────────────────────────────────────────
    pc = sub.add_parser("convert", help="Convert parquets → Megatron JSONL")
    pc.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help="Dir with parquets + manifest.json from `download`")
    pc.add_argument("--megatron-dir", default=DEFAULT_MEGATRON_DIR,
                    help="Where to write the per-language {lc}.jsonl")
    pc.add_argument("--max-docs-per-shard", type=int, default=None,
                    help="Subsample N docs per parquet shard (testing)")
    pc.add_argument("--keep-token-count", action="store_true", default=True,
                    help="Preserve parquet token_count in JSONL (needed for filter-long)")
    pc.set_defaults(func=cmd_convert)

    # ── filter-long ──────────────────────────────────────────────────────────
    pf = sub.add_parser("filter-long",
                        help="Keep only docs >= min tokens (context-extension data)")
    pf.add_argument("--megatron-dir", default=DEFAULT_MEGATRON_DIR,
                    help="Input dir with {lc}.jsonl from `convert`")
    pf.add_argument("--long-dir", default=DEFAULT_LONG_DIR,
                    help="Output dir for filtered {lc}.jsonl")
    pf.add_argument("--min-tokens", type=int, default=8192,
                    help="Drop docs shorter than this (default 8192; try 4096/16384/32768)")
    pf.add_argument("--max-tokens", type=int, default=None,
                    help="Optional upper bound (e.g. 131072) to exclude pathological outliers")
    pf.add_argument("--languages", default=None,
                    help="Only filter these langs (default: every jsonl in megatron-dir)")
    pf.set_defaults(func=cmd_filter_long)

    # ── tokenize ─────────────────────────────────────────────────────────────
    pt = sub.add_parser("tokenize",
                        help="Tokenize JSONL → Megatron .bin/.idx via preprocess_data.py")
    pt.add_argument("--input-dir", default=DEFAULT_LONG_DIR,
                    help="Dir of {lc}.jsonl to tokenize (long-dir or megatron-dir)")
    pt.add_argument("--output-dir", default=DEFAULT_BIN_DIR,
                    help="Where to write {lc}_text_document.bin/.idx")
    pt.add_argument("--megatron-path", default=None,
                    help="Path to NVIDIA-Megatron-LM checkout "
                         "(default: $MEGATRON_LM env var)")
    pt.add_argument("--tokenizer-type", default="HuggingFaceTokenizer",
                    help="One of HuggingFaceTokenizer | SentencePieceTokenizer | "
                         "GPTSentencePieceTokenizer | Llama2Tokenizer | TikTokenizer")
    pt.add_argument("--tokenizer-model", required=True,
                    help="HF repo id or local path to tokenizer (e.g. meta-llama/Llama-3.1-8B)")
    pt.add_argument("--vocab-size", type=int, default=None,
                    help="Required for some tokenizers; e.g. 128256 for Llama-3 HF tokenizer")
    pt.add_argument("--append-eod", action="store_true", default=True,
                    help="Append <eod> between docs (default: on)")
    pt.add_argument("--workers", type=int, default=8)
    pt.add_argument("--partitions", type=int, default=1)
    pt.add_argument("--languages", default=None)
    pt.add_argument("--dry-run", action="store_true",
                    help="Print preprocess_data.py invocations without running")
    pt.set_defaults(func=cmd_tokenize)

    # ── mix ──────────────────────────────────────────────────────────────────
    pm = sub.add_parser("mix",
                        help="Emit weighted --data-path block for Megatron training")
    pm.add_argument("--bin-dir", default=DEFAULT_BIN_DIR,
                    help="Dir of tokenized {lc}_text_document.{bin,idx}")
    pm.add_argument("--mix-dir", default=DEFAULT_MIX_DIR,
                    help="Where to write data_mix.json / data_mix.txt / data_path.args")
    pm.add_argument("--alpha", type=float, default=0.3,
                    help="Temperature for upsampling low-resource langs. "
                         "w_i ∝ p_i^alpha. 1.0 = natural, 0.0 = uniform. Default 0.3.")
    pm.add_argument("--floor", type=float, default=0.005,
                    help="Minimum weight per language before renormalization (default 0.005)")
    pm.add_argument("--languages", default=None,
                    help="Restrict mix to these langs (default: all tokenized langs)")
    pm.add_argument("--suffix", default="_text_document",
                    help="Megatron prefix suffix (default '_text_document')")
    pm.set_defaults(func=cmd_mix)

    # ── run (download + convert, optionally filter) ──────────────────────────
    pr = sub.add_parser("run",
                        help="End-to-end convenience: download + convert (+ filter)")
    pr.add_argument("--languages", default=None)
    pr.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    pr.add_argument("--megatron-dir", default=DEFAULT_MEGATRON_DIR)
    pr.add_argument("--long-dir", default=DEFAULT_LONG_DIR)
    pr.add_argument("--sample", action="store_true")
    pr.add_argument("--shards", type=int, default=1)
    pr.add_argument("--max-docs-per-shard", type=int, default=None)
    pr.add_argument("--keep-token-count", action="store_true", default=True)
    pr.add_argument("--filter-long", action="store_true",
                    help="Also emit a long-docs-only copy under --long-dir")
    pr.add_argument("--min-tokens", type=int, default=8192)
    pr.add_argument("--max-tokens", type=int, default=None)
    pr.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
