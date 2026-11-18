#!/bin/bash
#SBATCH --job-name=niah-v3
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=0-12:00:00
#SBATCH --output logs/%j.out
#SBATCH --error  logs/%j.err
# Base-LM multilingual NIAH (forced-choice log-likelihood) on the EXTENDED 128K model.
if [ -z ${SLURM_JOB_ID+x} ]; then mkdir -p logs; sbatch "$0" "$@"; exit; fi
set -euo pipefail; module purge
C=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
R=/scratch/project_465002530/users/bmoell/openeuro-longctx-datamix
MODEL=/scratch/project_465002530/users/bmoell/longctx-extend/v3_th32m_hf
OUT=/scratch/project_465002530/users/bmoell/longctx-extend/eval_niah_128k_$SLURM_JOB_ID
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/project_462000963/cache/
# Languages: spread across the 37 (script has templates for a subset — start with those it supports).
LANGS="${LANGS:-fr fi cs nl}"
CTX="${CTX:-4096 16384 65536 131072}"
DEPTHS="${DEPTHS:-0.0 0.25 0.5 0.75 1.0}"
TRIALS="${TRIALS:-10}"
echo "=== NIAH eval: model=$MODEL ==="
echo "langs=$LANGS ctx=$CTX depths=$DEPTHS trials=$TRIALS  $(date)"
srun singularity exec -B /pfs -B /scratch -B /flash "$C" bash -lc "cd $R && python3 -u scripts/eval_base_lm_niah.py --model $MODEL --output $OUT --languages $LANGS --context-lengths $CTX --depths $DEPTHS --trials $TRIALS 2>&1 | tail -80"
echo "=== done $(date) ==="
cat "$OUT/summary.json" 2>/dev/null || true
