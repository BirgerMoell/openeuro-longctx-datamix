"""Top-level CLI dispatcher for `longctx`.

Subcommands:
  estimate     — show dataset sizes and disk-space check
  download     — pull FinePDFs-Edu parquets for the target languages
  convert      — parquets → Megatron JSONL (streaming, preserves token_count)
  filter-long  — keep only docs >= min tokens (context-extension data)
  tokenize     — tokenize JSONL → Megatron .bin/.idx via preprocess_data.py
  mix          — emit weighted --data-path block for Megatron training
  stream-upload — tokenize/upload parquet shards incrementally for low disk
  artifacts    — package/upload/download Megatron tokenized artifact folders
  run          — download + convert (+ optional filter) end-to-end
"""

from __future__ import annotations

import argparse
import sys

from longctx import __version__
from longctx.commands import (
    cmd_convert,
    cmd_artifacts_download,
    cmd_artifacts_pack,
    cmd_artifacts_upload,
    cmd_download,
    cmd_estimate,
    cmd_filter_long,
    cmd_mix,
    cmd_run,
    cmd_sources_fetch,
    cmd_sources_list,
    cmd_stream_upload,
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
    pm.add_argument("--weights", default=None,
                    help="Explicit comma-separated weights, e.g. "
                         "multilingual_16k_plus=0.5,multilingual_4_16k=0.3,"
                         "multilingual_under4k=0.2. Overrides --alpha/--floor.")
    pm.add_argument("--languages", default=None,
                    help="Restrict mix to these langs (default: all tokenized langs)")
    pm.add_argument("--suffix", default="_text_document",
                    help="Megatron prefix suffix (default '_text_document')")
    pm.set_defaults(func=cmd_mix)

    # ── stream-upload ───────────────────────────────────────────────────────
    psu = sub.add_parser(
        "stream-upload",
        help="Download, tier, tokenize, upload, and delete chunks to keep disk usage low",
    )
    psu.add_argument("--repo-id", default=None,
                     help="HF dataset repo id, e.g. birgermoell/oellm-longctx-tokenized")
    psu.add_argument("--languages", default=None,
                     help="Comma-separated ISO 639-1 codes (default: all supported FinePDFs-Edu langs)")
    psu.add_argument("--exclude-languages", default=None,
                     help="Comma-separated ISO 639-1 codes to skip from the selected languages")
    psu.add_argument("--shards", type=int, default=1,
                     help="Parquet shards per language unless --all-shards is set")
    psu.add_argument("--all-shards", action="store_true",
                     help="Process every parquet shard for each selected language")
    psu.add_argument("--skip-shards", default=None,
                     help="Comma-separated shard indexes to skip, e.g. fr:24,en:20-22. "
                          "Use to route around known-bad source shards while continuing.")
    psu.add_argument("--max-docs", type=int, default=None,
                     help="Global doc cap for smoke tests")
    psu.add_argument("--chunk-docs", type=int, default=2000,
                     help="Docs per tier chunk before tokenization/upload")
    psu.add_argument("--tier-preset", default="lc16k",
                     choices=["lc16k", "lc128k", "lc256k"],
                     help="Length-tier preset for streamed chunks. "
                          "lc16k preserves the original 16k_plus/4_16k/under4k mix; "
                          "lc128k and lc256k expose higher-length buckets.")
    psu.add_argument("--tier-spec-json", default=None,
                     help="Optional JSON file overriding --tier-preset. Format: "
                          "{\"tier\": {\"min\": 32768, \"max\": 65536, \"weight\": 0.2}}")
    psu.add_argument("--chunk-docs-by-tier", default=None,
                     help="Optional tier-specific chunk sizes, e.g. "
                          "128k_plus=64,64_128k=128,32_64k=256,16_32k=512")
    psu.add_argument("--upload-batch-chunks", type=int, default=8,
                     help="Upload this many tokenized chunks per HF commit")
    psu.add_argument("--upload-batch-bytes", type=int, default=0,
                     help="Also flush an HF commit once this many bytes are staged "
                          "(0 disables byte-based flushing)")
    psu.add_argument("--run-id", default=None,
                     help="Namespace uploaded chunks under runs/<run-id>/... and prefix "
                          "chunk names. Use this when changing chunking/resume settings.")
    psu.add_argument("--upload-retries", type=int, default=24,
                     help="Retry batched HF uploads this many times on rate limits")
    psu.add_argument("--batch-rows", type=int, default=4096,
                     help="Parquet rows per read batch")
    psu.add_argument("--work-dir", default=None,
                     help="Temporary work dir (default: system temp)")
    psu.add_argument("--local-only", action="store_true",
                     help="Write a local artifact layout and do not upload to HF")
    psu.add_argument("--local-dir", default=None,
                     help="Local artifact dir for --local-only")
    psu.add_argument("--keep-local", action="store_true",
                     help="Keep temporary parquet/jsonl/bin/idx files after upload")
    psu.add_argument("--keep-work-dir", action="store_true",
                     help="Keep the temp work root even when --keep-local is off")
    psu.add_argument("--megatron-path", default=None,
                     help="Path to NVIDIA-Megatron-LM checkout (default: $MEGATRON_LM)")
    psu.add_argument("--python", default=sys.executable,
                     help="Python executable for Megatron preprocess_data.py")
    psu.add_argument("--tokenizer-type", default="HuggingFaceTokenizer")
    psu.add_argument("--tokenizer-model", required=True)
    psu.add_argument("--vocab-size", type=int, default=None)
    psu.add_argument("--workers", type=int, default=4)
    psu.add_argument("--private", action="store_true", default=True)
    psu.add_argument("--public", dest="private", action="store_false",
                     help="Create/upload to a public HF dataset repo")
    psu.add_argument("--resume", action="store_true", default=True,
                     help="Load uploaded chunk manifests and skip completed chunks (default)")
    psu.add_argument("--no-resume", dest="resume", action="store_false",
                     help="Do not inspect existing HF chunk manifests before running")
    psu.add_argument("--resume-names-only", action="store_true",
                     help="For HF resume, derive completed chunk names from remote manifest filenames "
                          "instead of downloading every manifest. Use with --skip-final-metadata "
                          "for continuation runs that only need to push more chunks.")
    psu.add_argument("--skip-final-metadata", action="store_true",
                     help="Upload tokenized chunks but skip rebuilding/uploading mix/ and stream manifests")
    psu.add_argument("--dry-run", action="store_true",
                     help="Download/read and print operations without tokenizing/uploading")
    psu.set_defaults(func=cmd_stream_upload)

    # ── artifacts ───────────────────────────────────────────────────────────
    pa = sub.add_parser(
        "artifacts",
        help="Package/upload/download Megatron tokenized artifacts via Hugging Face Hub",
    )
    pa_sub = pa.add_subparsers(dest="artifacts_command", required=True, metavar="ACTION")

    pa_pack = pa_sub.add_parser("pack", help="Package bin/mix files with checksums")
    pa_pack.add_argument("--bin-dir", default=DEFAULT_BIN_DIR)
    pa_pack.add_argument("--mix-dir", default=DEFAULT_MIX_DIR)
    pa_pack.add_argument("--output-dir", required=True)
    pa_pack.add_argument("--tokenizer-type", default=None)
    pa_pack.add_argument("--tokenizer-model", default=None)
    pa_pack.add_argument("--tokenizer-revision", default=None)
    pa_pack.add_argument("--vocab-size", type=int, default=None)
    pa_pack.add_argument("--megatron-path", default=None)
    pa_pack.add_argument("--notes", default=None)
    pa_pack.add_argument("--force", action="store_true")
    pa_pack.set_defaults(func=cmd_artifacts_pack)

    pa_upload = pa_sub.add_parser("upload", help="Upload packaged artifact folder to HF")
    pa_upload.add_argument("--folder", required=True)
    pa_upload.add_argument("--repo-id", required=True,
                           help="HF dataset repo id, e.g. birgermoell/oellm-longctx-tokenized")
    pa_upload.add_argument("--path-in-repo", default="")
    pa_upload.add_argument("--private", action="store_true")
    pa_upload.add_argument("--commit-message", default="Upload Megatron tokenized artifacts")
    pa_upload.add_argument("--ignore-patterns", nargs="*", default=None)
    pa_upload.set_defaults(func=cmd_artifacts_upload)

    pa_download = pa_sub.add_parser("download", help="Download HF artifact and rewrite paths")
    pa_download.add_argument("--repo-id", required=True)
    pa_download.add_argument("--revision", default=None)
    pa_download.add_argument("--output-dir", required=True)
    pa_download.add_argument("--path-in-repo", default="")
    pa_download.add_argument("--allow-patterns", nargs="*", default=None)
    pa_download.set_defaults(func=cmd_artifacts_download)

    # ── sources (alt corpora for missing/supplementary langs) ────────────────
    ps = sub.add_parser(
        "sources",
        help="Alternative sources for languages missing from FinePDFs-Edu",
    )
    ps_sub = ps.add_subparsers(dest="sources_command", required=True, metavar="ACTION")

    ps_list = ps_sub.add_parser("list", help="List registered source adapters")
    ps_list.set_defaults(func=cmd_sources_list)

    ps_fetch = ps_sub.add_parser("fetch",
                                 help="Fetch a source into Megatron JSONL format")
    ps_fetch.add_argument("--source", required=True,
                          help="Adapter name (see `longctx sources list`)")
    ps_fetch.add_argument("--languages", required=True,
                          help="Comma-separated ISO 639-1 codes (e.g. ga,sq,lb)")
    ps_fetch.add_argument("--megatron-dir", default=DEFAULT_MEGATRON_DIR,
                          help="Where to write {lc}.jsonl")
    ps_fetch.add_argument("--sample", action="store_true",
                          help="Quick probe: cap at 1000 docs per language")
    ps_fetch.add_argument("--max-docs", type=int, default=None,
                          help="Cap docs per language (overrides --sample cap)")
    ps_fetch.add_argument("--overwrite", action="store_true",
                          help="Overwrite existing {lc}.jsonl files")
    ps_fetch.set_defaults(func=cmd_sources_fetch)

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
