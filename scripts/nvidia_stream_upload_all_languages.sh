#!/usr/bin/env bash
# Stream-tokenize all FinePDFs-Edu languages and upload Megatron shards to HF.
#
# This is the low-disk path for the small NVIDIA machine. It never builds one
# giant JSONL or artifact folder. Each tokenized chunk is uploaded immediately,
# and local temporaries are removed after upload.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEGATRON_DIR="${MEGATRON_DIR:-${MEGATRON_LM:-/home/ubuntu/birger/NVIDIA-Megatron-LM-context-extension}}"
RUN_PYTHON="${RUN_PYTHON:-/home/ubuntu/birger/swedish-medical-benchmark/.venv-gemma4/bin/python}"
PREPROCESS_PYTHON="${PREPROCESS_PYTHON:-$RUN_PYTHON}"

HF_REPO_ID="${HF_REPO_ID:-birgermoell/oellm-longctx-tokenized-streamed-all}"
WORK_DIR="${WORK_DIR:-$REPO_DIR/data/stream_upload_all_tmp}"
TOKENIZER="${TOKENIZER:-/home/ubuntu/birger/Megatron-Bridge-utils/tokenizers/openeurollm/tokenizer-256k}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
VOCAB_SIZE="${VOCAB_SIZE:-262144}"
WORKERS="${WORKERS:-4}"
CHUNK_DOCS="${CHUNK_DOCS:-2000}"
TIER_PRESET="${TIER_PRESET:-lc16k}"
TIER_SPEC_JSON="${TIER_SPEC_JSON:-}"
BATCH_ROWS="${BATCH_ROWS:-4096}"
UPLOAD_BATCH_CHUNKS="${UPLOAD_BATCH_CHUNKS:-16}"
UPLOAD_BATCH_BYTES="${UPLOAD_BATCH_BYTES:-0}"
UPLOAD_RETRIES="${UPLOAD_RETRIES:-24}"
RUN_ID="${RUN_ID:-}"
LANGUAGES="${LANGUAGES:-}"
EXCLUDE_LANGUAGES="${EXCLUDE_LANGUAGES:-}"
SHARDS="${SHARDS:-}"
SKIP_SHARDS="${SKIP_SHARDS:-}"
MAX_DOCS="${MAX_DOCS:-}"
CHUNK_DOCS_BY_TIER="${CHUNK_DOCS_BY_TIER:-}"
RESUME_NAMES_ONLY="${RESUME_NAMES_ONLY:-0}"
SKIP_FINAL_METADATA="${SKIP_FINAL_METADATA:-0}"
PRIVATE="${PRIVATE:-1}"

if [[ -z "${HF_TOKEN:-}" && -f "${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}" ]]; then
  HF_TOKEN="$(<"${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}")"
  export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[error] HF_TOKEN must be set in the environment or configured with huggingface-cli login." >&2
  exit 2
fi

if [[ ! -x "$RUN_PYTHON" ]]; then
  echo "[error] RUN_PYTHON is not executable: $RUN_PYTHON" >&2
  exit 2
fi

if [[ ! -d "$MEGATRON_DIR" ]]; then
  echo "[error] MEGATRON_DIR does not exist: $MEGATRON_DIR" >&2
  exit 2
fi

mkdir -p "$WORK_DIR"

export PYTHONPATH="$REPO_DIR/src:$MEGATRON_DIR:${PYTHONPATH:-}"
export MEGATRON_LM="$MEGATRON_DIR"
export TOKENIZERS_PARALLELISM=false

args=(
  -m longctx.cli stream-upload
  --repo-id "$HF_REPO_ID"
  --work-dir "$WORK_DIR"
  --python "$PREPROCESS_PYTHON"
  --megatron-path "$MEGATRON_DIR"
  --tokenizer-type "$TOKENIZER_TYPE"
  --tokenizer-model "$TOKENIZER"
  --vocab-size "$VOCAB_SIZE"
  --workers "$WORKERS"
  --chunk-docs "$CHUNK_DOCS"
  --tier-preset "$TIER_PRESET"
  --batch-rows "$BATCH_ROWS"
  --upload-batch-chunks "$UPLOAD_BATCH_CHUNKS"
  --upload-batch-bytes "$UPLOAD_BATCH_BYTES"
  --upload-retries "$UPLOAD_RETRIES"
  --resume
)

if [[ -n "$RUN_ID" ]]; then
  args+=(--run-id "$RUN_ID")
fi

if [[ -n "$LANGUAGES" ]]; then
  args+=(--languages "$LANGUAGES")
fi

if [[ -n "$EXCLUDE_LANGUAGES" ]]; then
  args+=(--exclude-languages "$EXCLUDE_LANGUAGES")
fi

if [[ -n "$SKIP_SHARDS" ]]; then
  args+=(--skip-shards "$SKIP_SHARDS")
fi

if [[ -n "$CHUNK_DOCS_BY_TIER" ]]; then
  args+=(--chunk-docs-by-tier "$CHUNK_DOCS_BY_TIER")
fi

if [[ -n "$TIER_SPEC_JSON" ]]; then
  args+=(--tier-spec-json "$TIER_SPEC_JSON")
fi

if [[ "$RESUME_NAMES_ONLY" == "1" ]]; then
  args+=(--resume-names-only)
fi

if [[ "$SKIP_FINAL_METADATA" == "1" ]]; then
  args+=(--skip-final-metadata)
fi

if [[ -n "$SHARDS" ]]; then
  args+=(--shards "$SHARDS")
else
  args+=(--all-shards)
fi

if [[ -n "$MAX_DOCS" ]]; then
  args+=(--max-docs "$MAX_DOCS")
fi

if [[ "$PRIVATE" == "1" ]]; then
  args+=(--private)
else
  args+=(--public)
fi

echo "=== Stream upload all languages ==="
echo "Repo:       $REPO_DIR"
echo "HF repo:    $HF_REPO_ID"
echo "Megatron:   $MEGATRON_DIR"
echo "Python:     $RUN_PYTHON"
echo "Tokenizer:  $TOKENIZER"
echo "Work dir:   $WORK_DIR"
echo "Chunk docs: $CHUNK_DOCS"
echo "Tier preset: $TIER_PRESET"
if [[ -n "$TIER_SPEC_JSON" ]]; then
  echo "Tier spec JSON: $TIER_SPEC_JSON"
fi
if [[ -n "$CHUNK_DOCS_BY_TIER" ]]; then
  echo "Chunk docs by tier: $CHUNK_DOCS_BY_TIER"
fi
echo "Resume names only: $RESUME_NAMES_ONLY"
echo "Skip final metadata: $SKIP_FINAL_METADATA"
echo "Upload batch chunks: $UPLOAD_BATCH_CHUNKS"
echo "Upload batch bytes: $UPLOAD_BATCH_BYTES"
echo "Upload retries: $UPLOAD_RETRIES"
if [[ -n "$RUN_ID" ]]; then
  echo "Run id:     $RUN_ID"
fi
if [[ -n "$LANGUAGES" ]]; then
  echo "Languages:  $LANGUAGES"
else
  echo "Languages:  all FinePDFs-Edu languages in longctx.languages"
fi
if [[ -n "$EXCLUDE_LANGUAGES" ]]; then
  echo "Excluding:  $EXCLUDE_LANGUAGES"
fi
if [[ -n "$SKIP_SHARDS" ]]; then
  echo "Skip shards: $SKIP_SHARDS"
fi
if [[ -n "$SHARDS" ]]; then
  echo "Shards:     first $SHARDS per language"
else
  echo "Shards:     all"
fi
echo ""

exec "$RUN_PYTHON" "${args[@]}"
