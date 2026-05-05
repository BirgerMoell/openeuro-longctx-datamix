#!/bin/bash
#SBATCH -p dev-g
#SBATCH -t 00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH -J longctx-hello
#SBATCH -A project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/longctx-hello-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longctx-hello-%j.err

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix
MEGATRON_DIR=$WORKDIR/NVIDIA-Megatron-LM

# Pass --quick to skip full tokenization (just verify the command runs on 1 doc)
QUICK=${QUICK:-1}

# Initialise Lmod if not already available
if ! type module &>/dev/null; then
  source /etc/profile
fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

echo "=== 1. GPU check ==="
rocm-smi

singularity exec "$SIF" python -c "
import torch
print('torch:', torch.__version__)
print('ROCm available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU 0:', torch.cuda.get_device_name(0))
    x = torch.ones(4, 4, device='cuda')
    print('Tensor on GPU OK:', x.sum().item())
"

echo ""
echo "=== 2. Clone Megatron (if needed) ==="
if [[ ! -d "$MEGATRON_DIR" ]]; then
  git clone --depth=1 https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM "$MEGATRON_DIR"
  echo "Megatron cloned."
else
  echo "Megatron already present at $MEGATRON_DIR"
fi

echo ""
echo "=== 3. Data pipeline ==="

singularity exec "$SIF" bash -c "
  set -euo pipefail
  cd $REPO
  python -m venv .venv --system-site-packages 2>/dev/null || true
  source .venv/bin/activate
  pip install -q -e .
  export HF_HOME=$WORKDIR/hf
  export HF_HUB_CACHE=$WORKDIR/hf
  export MEGATRON_LM=$MEGATRON_DIR
  export HF_TOKEN=\$(cat ~/.hf_token 2>/dev/null || true)
  mkdir -p \$HF_HOME

  echo '--- estimate ---'
  longctx estimate --languages mt

  echo '--- download 1 shard ---'
  longctx download --sample --shards 1 --languages mt

  echo '--- convert ---'
  longctx convert

  echo '--- filter-long ---'
  longctx filter-long --min-tokens 4096

  if [[ '$QUICK' == '1' ]]; then
    echo '--- tokenize (quick: 1 doc only) ---'
    # Write a single-doc JSONL to a temp file for a fast end-to-end tokenize check
    TMPDIR=$WORKDIR/tmp_quick
    mkdir -p \$TMPDIR
    head -n 1 data/long/mt.jsonl > \$TMPDIR/mt_quick.jsonl
    # Prepend the cloned fork so it shadows the container's megatron-core pip package
    export PYTHONPATH=\$MEGATRON_LM:\${PYTHONPATH:-}
    python \$MEGATRON_LM/tools/preprocess_data.py \
      --input \$TMPDIR/mt_quick.jsonl \
      --output-prefix \$TMPDIR/mt_quick \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model openeurollm/tokenizer-256k \
      --json-keys text \
      --workers 1 \
      --append-eod
    echo 'Quick tokenize OK — bin/idx written to:' \$TMPDIR/
    ls -lh \$TMPDIR/
  else
    echo '--- tokenize (full) ---'
    longctx tokenize \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model openeurollm/tokenizer-256k
    echo 'Full tokenize OK'
    ls -lh data/bin/
  fi
"

echo ""
echo "=== 4. Tiny training run ==="

singularity exec "$SIF" bash -c "
  set -euo pipefail
  export PYTHONPATH=$MEGATRON_DIR:\${PYTHONPATH:-}
  export HF_HOME=$WORKDIR/hf
  export HF_TOKEN=\$(cat ~/.hf_token 2>/dev/null || true)

  # ROCm settings for LUMI MI250X
  # Persistent MIOpen cache so autotuning survives across jobs
  export MIOPEN_USER_DB_PATH=$WORKDIR/miopen_cache
  export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH
  mkdir -p \$MIOPEN_USER_DB_PATH
  # Disable MIOpen autotuning to avoid first-run hang
  export MIOPEN_FIND_ENFORCE=NONE
  export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
  export NCCL_NET_GDR_LEVEL=3
  export ROCR_VISIBLE_DEVICES=0

  # Tokenize 50 docs for training data
  TRAIN_DIR=$WORKDIR/tmp_train
  mkdir -p \$TRAIN_DIR
  head -n 50 $REPO/data/long/mt.jsonl > \$TRAIN_DIR/mt_train.jsonl

  python $MEGATRON_DIR/tools/preprocess_data.py \
    --input \$TRAIN_DIR/mt_train.jsonl \
    --output-prefix \$TRAIN_DIR/mt_train \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model openeurollm/tokenizer-256k \
    --json-keys text \
    --workers 1 \
    --append-eod
  echo Training data tokenized.

  # Tiny GPT from scratch: 4 layers, hidden=256, 5 iterations
  # vocab-size=262144 matches openeurollm/tokenizer-256k (2^18 tokens)
  # Use direct Python invocation (avoids torchrun process spawning issues on LUMI)
  export MASTER_ADDR=localhost
  export MASTER_PORT=29500
  export WORLD_SIZE=1
  export RANK=0
  export LOCAL_RANK=0
  python $MEGATRON_DIR/pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 4 \
    --hidden-size 256 \
    --num-attention-heads 4 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 5 \
    --eval-iters 2 \
    --eval-interval 5 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 262144 \
    --split 900,50,50 \
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
  echo Training OK!
"

echo ""
echo "=== Done ==="
