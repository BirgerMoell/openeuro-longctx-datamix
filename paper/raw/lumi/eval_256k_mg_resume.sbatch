#!/bin/bash
#SBATCH --job-name=niah256mg
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-12:00:00
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

# Multilingual 256K forced-choice NIAH. Results are case-resumable: set NIAH_OUT to an
# existing output directory after a timeout or node failure and completed JSONL rows are kept.
if [ -z ${SLURM_JOB_ID+x} ]; then
  mkdir -p logs
  sbatch "$0" "$@"
  exit
fi

set -euo pipefail
module purge

CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
REPO=/scratch/project_465002530/users/bmoell/openeuro-longctx-datamix
MODEL=/scratch/project_465002530/users/bmoell/longctx-extend/prelude_256k_hf
OUT=${NIAH_OUT:-/scratch/project_465002530/users/bmoell/longctx-extend/eval_niah_256k_$SLURM_JOB_ID}
mkdir -p "$OUT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/project_462000963/cache/

LANGS=${LANGS:-"fr fi cs nl"}
CTX=${CTX:-262144}
DEPTHS=${DEPTHS:-"0.0 0.25 0.5 0.75 1.0"}
TRIALS=${TRIALS:-4}

echo "=== NIAH eval: model=$MODEL ==="
echo "output=$OUT langs=$LANGS ctx=$CTX depths=$DEPTHS trials=$TRIALS $(date)"
srun singularity exec -B /pfs -B /scratch -B /flash "$CONTAINER" \
  bash -lc "cd $REPO && python3 -u scripts/eval_base_lm_niah.py \
    --model $MODEL --output $OUT --languages $LANGS --context-lengths $CTX \
    --depths $DEPTHS --trials $TRIALS --resume"

echo "=== done $(date) ==="
if [ -f "$OUT/summary.json" ]; then
  cat "$OUT/summary.json"
fi
