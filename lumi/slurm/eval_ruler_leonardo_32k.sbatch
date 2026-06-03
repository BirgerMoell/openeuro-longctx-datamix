#!/bin/bash
#SBATCH --job-name=ruler-32k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=OELLM_prod2026
#SBATCH -o /leonardo_scratch/large/userexternal/pmoell00/ruler-32k-%j.out
#SBATCH -e /leonardo_scratch/large/userexternal/pmoell00/ruler-32k-%j.err

# 32K-only RULER pass — completes the table from job 42990662 which timed out.

set -euo pipefail

VENV=$WORK/ruler_venv
OUT_DIR=/leonardo_scratch/large/userexternal/pmoell00/ruler-yarn-v2-$SLURM_JOB_ID

export HF_HOME=$WORK/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA=$WORK/nltk_data

module purge
module load python/3.11.7
module load cuda/12.1

mkdir -p "$OUT_DIR"
source "$VENV/bin/activate"
echo "Venv loaded from $WORK"

echo "=== RULER eval — YaRN v2 Multilingual 9B — 32K only — $(date) ==="
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Output:  $OUT_DIR"
echo ""

MODEL=/leonardo_work/OELLM_prod2026/models/oellm-9b-yarn-v2-32k
TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,ruler_cwe,ruler_fwe"

echo "--- Context length: 32768 tokens ---"
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL},max_length=32768,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --metadata "{\"max_seq_lengths\":[32768]}" \
    --output_path "$OUT_DIR/ctx_32768" \
    --log_samples \
    --num_fewshot 0 \
    --limit 100 \
    --batch_size 1 2>&1

echo ""
echo "=== Done — $(date) ==="
echo "Results: $OUT_DIR/"
