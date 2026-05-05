#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: run-inside-lumi.sh setup|hello}"

LUMI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${LONGCTX_REPO_DIR:-$LUMI_DIR/openeuro-longctx-datamix}"
WORKDIR="${LONGCTX_LUMI_WORKDIR:-$LUMI_DIR/work}"

load_python() {
  if ! type module >/dev/null 2>&1; then
    source /etc/profile
  fi
  module purge
  module load cray-python
}

prepare_environment() {
  load_python
  mkdir -p "$WORKDIR"

  if [[ ! -d "$REPO_DIR" ]]; then
    echo "Cloning openeuro-longctx-datamix..."
    git clone https://github.com/BirgerMoell/openeuro-longctx-datamix "$REPO_DIR"
  fi

  cd "$REPO_DIR"

  if [[ ! -d ".venv" ]]; then
    python -m venv .venv
  fi
  source .venv/bin/activate
  pip install -q -U pip
  pip install -q -e .

  python -c "import longctx; print('longctx installed OK')"
}

run_hello() {
  prepare_environment
  cd "$REPO_DIR"
  source .venv/bin/activate

  export HF_HOME="$WORKDIR/hf"
  export HF_HUB_CACHE="$WORKDIR/hf"
  mkdir -p "$HF_HOME"

  echo ""
  echo "=== Step 1: estimate disk usage (mt = Maltese, small language) ==="
  longctx estimate --languages mt

  echo ""
  echo "=== Step 2: download 1 shard for Maltese ==="
  longctx download --sample --shards 1 --languages mt

  echo ""
  echo "=== Step 3: convert parquet → JSONL ==="
  longctx convert

  echo ""
  echo "=== Step 4: filter for long documents (≥4096 tokens) ==="
  longctx filter-long --min-tokens 4096

  echo ""
  echo "=== Step 5: tokenize dry-run (no Megatron needed) ==="
  longctx tokenize --dry-run \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.1-8B

  echo ""
  echo "=== Hello world complete! ==="
  echo "JSONL data is in: $REPO_DIR/data/long/"
  echo "To run actual tokenization, clone Megatron and re-run without --dry-run."
}

case "$MODE" in
  setup)
    prepare_environment
    ;;
  hello)
    run_hello
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
