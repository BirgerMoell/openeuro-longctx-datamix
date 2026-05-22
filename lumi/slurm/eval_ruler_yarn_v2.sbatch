#!/bin/bash
#SBATCH --job-name=ruler-yarn-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=small-g
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000963
#SBATCH -o /flash/project_462000963/bmoell/ruler-yarn-v2-%j.out
#SBATCH -e /flash/project_462000963/bmoell/ruler-yarn-v2-%j.err

# RULER eval for YaRN v2 multilingual 9B model using lm-evaluation-harness.
#
# Tasks: NIAH single (x3), multi-key (x3), multi-query, multi-value, CWE, FWE
#        QA tasks are skipped — they require instruction following.
# Context lengths: 4K, 8K, 16K, 32K
#
# lm_eval is installed into flash on first run (~2 min).
#
# Submit: sbatch lumi/slurm/eval_ruler_yarn_v2.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
FLASH=/flash/project_462000963/bmoell
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix

MODEL_PATH=$FLASH/yarn-multilingual-v2-1k/hf
LM_EVAL_DIR=$FLASH/lm_eval_install
OUT_DIR=$FLASH/ruler-yarn-v2-$SLURM_JOB_ID

export HF_HOME="$WORKDIR/hf"
export PYTHONPATH="$LM_EVAL_DIR:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR" "$LM_EVAL_DIR"

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

git -C "$REPO" pull --quiet 2>/dev/null || true

echo "=== RULER eval — YaRN v2 Multilingual 9B — $(date) ==="
echo "Model:   $MODEL_PATH"
echo "Output:  $OUT_DIR"
echo ""

# Install lm_eval into flash if not already there
singularity exec \
    --bind /pfs,/scratch,/projappl,/project,/flash,/appl \
    --env PYTHONPATH="$LM_EVAL_DIR" \
    "$SIF" \
    python3 -c "import lm_eval" 2>/dev/null || \
singularity exec \
    --bind /pfs,/scratch,/projappl,/project,/flash,/appl \
    "$SIF" \
    pip install --quiet --target "$LM_EVAL_DIR" \
        "lm_eval[ruler]>=0.4.4" \
        "nltk" "rouge_score" "pycountry" 2>&1 | tail -5

echo "lm_eval ready"
echo ""

# NIAH + aggregation tasks — skip QA which requires instruction following
TASKS="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,ruler_cwe,ruler_fwe"

# Run at 4K, 8K, 16K, 32K
for CTX in 4096 8192 16384 32768; do
    echo "--- Context length: ${CTX} tokens ---"
    singularity exec \
        --bind /pfs,/scratch,/projappl,/project,/flash,/appl \
        --env PYTHONPATH="$LM_EVAL_DIR" \
        "$SIF" \
        python3 -m lm_eval \
            --model hf \
            --model_args "pretrained=${MODEL_PATH},max_length=${CTX},dtype=bfloat16,trust_remote_code=True" \
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
