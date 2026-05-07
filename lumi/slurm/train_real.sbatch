#!/bin/bash
#SBATCH -p dev-g
#SBATCH -t 00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH -J longctx-real
#SBATCH -A project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/longctx-real-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longctx-real-%j.err

# Train on the full real Maltese corpus (5,463 filtered docs, min 4096 tokens).
# Assumes data pipeline has already been run (data/long/mt.jsonl exists).
# Submit with: sbatch lumi/slurm/train_real.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix
MEGATRON_DIR=$WORKDIR/NVIDIA-Megatron-LM
TRAIN_DIR=$WORKDIR/data_tokenized

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export MIOPEN_USER_DB_PATH=$WORKDIR/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$WORKDIR/miopen_cache
mkdir -p $MIOPEN_USER_DB_PATH
export MIOPEN_FIND_ENFORCE=NONE
export ROCR_VISIBLE_DEVICES=0
export HF_HOME=$WORKDIR/hf
export HF_HUB_OFFLINE=1          # tokenizer already cached — no network needed

mkdir -p $TRAIN_DIR

# ── 1. Tokenize full corpus ────────────────────────────────────────────────────
# Skip if already done (bin/idx present and newer than the source JSONL).
# NOTE: Do NOT set PYTHONPATH here. preprocess_data.py picks up megatron from the
# container's site-packages (via the tools/ script dir path). Setting PYTHONPATH to
# the local fork causes a namespace package conflict (local training imports
# AdamOptimizerConfig from local core, but container core resolves first).
BIN=$TRAIN_DIR/mt_full_text_document.bin
if [ -f "$BIN" ] && [ "$BIN" -nt "$REPO/data/long/mt.jsonl" ]; then
  echo "=== Tokenized data already up-to-date, skipping ==="
else
  echo "=== Tokenizing 5,463 docs → $TRAIN_DIR ==="
  singularity exec "$SIF" python $MEGATRON_DIR/tools/preprocess_data.py \
    --input  $REPO/data/long/mt.jsonl \
    --output-prefix $TRAIN_DIR/mt_full \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model openeurollm/tokenizer-256k \
    --json-keys text \
    --workers 1 \
    --append-eod
  echo "Tokenization done. Files:"
  ls -lh $TRAIN_DIR/mt_full_text_document.*
fi

# ── 2. Train on real data ──────────────────────────────────────────────────────
# Running pretrain_gpt.py directly causes Python to add $MEGATRON_DIR to sys.path,
# which ensures the local fork's megatron package (with ROCm fixes) is used.
echo ""
echo "=== Training on real data ==="

export MASTER_ADDR=localhost
export MASTER_PORT=29501
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

singularity exec "$SIF" python $MEGATRON_DIR/pretrain_gpt.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 4 \
  --hidden-size 256 \
  --num-attention-heads 4 \
  --seq-length 512 \
  --max-position-embeddings 512 \
  --micro-batch-size 1 \
  --global-batch-size 1 \
  --train-iters 20 \
  --eval-iters 5 \
  --eval-interval 10 \
  --data-path $TRAIN_DIR/mt_full_text_document \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model openeurollm/tokenizer-256k \
  --vocab-size 262144 \
  --split 950,30,20 \
  --lr 0.0001 \
  --min-lr 0.00001 \
  --lr-decay-style cosine \
  --log-interval 1 \
  --no-save-optim \
  --no-save-rng \
  --distributed-backend gloo \
  --transformer-impl local \
  --no-gradient-accumulation-fusion \
  --no-bias-dropout-fusion \
  --no-masked-softmax-fusion \
  --num-workers 0

echo ""
echo "=== Done ==="
