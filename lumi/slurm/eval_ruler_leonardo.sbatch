#!/bin/bash
#SBATCH --job-name=ruler-yarn-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=boost_usr_prod
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=OELLM_prod2026
#SBATCH -o $SCRATCH/ruler-yarn-v2-%j.out
#SBATCH -e $SCRATCH/ruler-yarn-v2-%j.err

# RULER eval for YaRN v2 multilingual 9B model on Leonardo (CINECA).
# Uses NVIDIA A100 64GB — CUDA, not ROCm.
# Loads model directly from HuggingFace.
#
# Storage layout (per Jonas Lindh's best practices):
#   $WORK  — venv and lm_eval install (persistent, 1TB limit)
#   $SCRATCH — job outputs (no size limit, auto-deleted after 40 days)
#
# Tasks: NIAH single (x3), multi-key (x3), multi-query, multi-value, CWE, FWE
#        QA tasks skipped — require instruction following.
# Context lengths: 4K, 8K, 16K, 32K
#
# One-time setup before first run:
#   huggingface-cli login   (paste a read token)
#
# Submit: sbatch lumi/slurm/eval_ruler_leonardo.sbatch

set -euo pipefail

# $WORK and $SCRATCH are set automatically by Leonardo's environment
VENV=$WORK/ruler_venv        # persistent across jobs
OUT_DIR=$SCRATCH/ruler-yarn-v2-$SLURM_JOB_ID

# Load modules
module purge
module load python/3.11.7--gcc--8.5.0
module load cuda/12.1

mkdir -p "$OUT_DIR"

# Create venv in $WORK on first run, reuse after
if [ ! -d "$VENV" ]; then
    echo "Creating venv in $WORK..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --quiet --upgrade pip
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
    pip install --quiet transformers accelerate "lm_eval[ruler]>=0.4.4" nltk rouge_score
    echo "Venv ready"
else
    source "$VENV/bin/activate"
    echo "Venv loaded from $WORK"
fi

echo "=== RULER eval — YaRN v2 Multilingual 9B — $(date) ==="
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Model:   birgermoell/oellm-9b-yarn-multilingual-v2-32k"
echo "Output:  $OUT_DIR"
echo ""

MODEL="birgermoell/oellm-9b-yarn-multilingual-v2-32k"

# NIAH + aggregation tasks — QA skipped (requires instruction following)
TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,ruler_cwe,ruler_fwe"

for CTX in 4096 8192 16384 32768; do
    echo "--- Context length: ${CTX} tokens ---"
    python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=${MODEL},max_length=${CTX},dtype=bfloat16,trust_remote_code=True" \
        --tasks "$TASKS" \
        --metadata "{\"max_seq_lengths\":[${CTX}]}" \
        --output_path "$OUT_DIR/ctx_${CTX}" \
        --log_samples \
        --num_fewshot 0 \
        --batch_size 1 2>&1
    echo ""
done

echo "=== Done — $(date) ==="
echo "Results: $OUT_DIR/"
