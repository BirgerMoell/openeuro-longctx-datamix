#!/bin/bash
#SBATCH -p small-g
#SBATCH -t 00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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
    python \$MEGATRON_LM/tools/preprocess_data.py \
      --input \$TMPDIR/mt_quick.jsonl \
      --output-prefix \$TMPDIR/mt_quick \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model meta-llama/Llama-3.1-8B \
      --json-keys text \
      --workers 1 \
      --append-eod
    echo 'Quick tokenize OK — bin/idx written to:' \$TMPDIR/
    ls -lh \$TMPDIR/
  else
    echo '--- tokenize (full) ---'
    longctx tokenize \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model meta-llama/Llama-3.1-8B
    echo 'Full tokenize OK'
    ls -lh data/bin/
  fi
"

echo ""
echo "=== Done ==="
