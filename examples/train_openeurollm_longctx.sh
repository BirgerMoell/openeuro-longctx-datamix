#!/usr/bin/env bash
# End-to-end long-context continued pretraining launcher.
#
# Produces a weighted multilingual data mix from FinePDFs-Edu and hands it
# to the OpenEuroLLM NVIDIA-Megatron-LM fork's training script.
#
# Prerequisites:
#   pip install -e ..                              # this repo
#   git clone https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM
#   export MEGATRON_LM=$PWD/NVIDIA-Megatron-LM
#   export HF_TOKEN=hf_...                         # optional but recommended
#
# Usage:
#   bash examples/train_openeurollm_longctx.sh <checkpoint_dir> <tb_dir> <tokenizer_path>

set -euo pipefail

CHECKPOINT_PATH=${1:-"checkpoints/openeurollm_longctx_32k_yarn"}
TB_PATH=${2:-"tensorboard/openeurollm_longctx_32k_yarn"}
TOKENIZER_PATH=${3:-"meta-llama/Llama-3.1-8B"}

LANGUAGES=${LANGUAGES:-"bg,cs,da,de,el,en,es,et,fi,fr,hr,hu,it,lt,lv,mt,nl,pl,pt,ro,sk,sl,sv,ca,cy,eu,gl,is,mk,no,sr,tr,uk"}
MIN_TOKENS=${MIN_TOKENS:-32768}       # match TARGET_CTX for 32K context extension
ALPHA=${ALPHA:-0.3}

# ── 1. Download + convert ────────────────────────────────────────────────────
longctx run \
    --languages "$LANGUAGES" \
    --output-dir data/raw \
    --megatron-dir data/megatron \
    --long-dir data/long \
    --filter-long --min-tokens "$MIN_TOKENS" \
    --sample --shards 1            # drop --sample for full download

# ── 2. Tokenize with the Megatron fork ───────────────────────────────────────
if [[ -z "${MEGATRON_LM:-}" ]]; then
    echo "[error] Set MEGATRON_LM to your NVIDIA-Megatron-LM checkout."
    exit 1
fi

longctx tokenize \
    --input-dir data/long \
    --output-dir data/bin \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_PATH" \
    --vocab-size 128256 \
    --workers 16

# ── 3. Emit weighted --data-path block ───────────────────────────────────────
longctx mix --bin-dir data/bin --mix-dir data/mix --alpha "$ALPHA"
DATA_PATH="$(cat data/mix/data_path.args)"

# ── 4. Kick off context-extension training (YaRN) ────────────────────────────
bash "$MEGATRON_LM/examples/llama/train_llama3_8b_context_extension.sh" \
    "$CHECKPOINT_PATH" \
    "$TB_PATH" \
    "$TOKENIZER_PATH" \
    "$DATA_PATH" \
    yarn
