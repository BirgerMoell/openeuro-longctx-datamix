#!/bin/bash
#SBATCH --job-name=eval-base-lm-niah
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=dev-g
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_465002530
#SBATCH -o /scratch/project_462000963/bmoell/eval-base-lm-niah-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/eval-base-lm-niah-%j.err

# Base-LM NIAH via forced-choice log-likelihood scoring.
#
# No instruction following required. Tests whether the model can retrieve
# and bind a key→value pair across a long context by comparing log-probabilities
# of candidate completions (all candidate values appear in the context, attached
# to different keys, so this tests retrieval + binding not memorisation).
#
# Grid:
#   context lengths : 2048  4096  8192  16384  32768  tokens
#   needle depths   : 0%  25%  50%  75%  100%
#   languages       : fr  fi  cs  nl  (subset of 8 training langs)
#   trials per cell : 10
#
# Controls:
#   no_context    — prefix only, should score ~25% (4-way random)
#   shuffled      — key/value bindings rotated, correct answer changes
#   short_ctx     — 256-token context, confirms scoring code works
#
# Results:
#   $OUT_DIR/<lang>_results.jsonl   — per-trial JSONL
#   $OUT_DIR/summary.json           — accuracy table by lang × ctx_length
#   $OUT_DIR/all_results.jsonl      — everything combined
#
# Submit: sbatch lumi/slurm/eval_base_lm_niah.sbatch
# Extend languages/lengths:
#   sbatch --export=ALL,LANGUAGES="bg cs da et fi fr hr nl",CTX_LENGTHS="4096 8192 16384 32768" \
#          lumi/slurm/eval_base_lm_niah.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix

MODEL_PATH=/flash/project_462000963/bmoell/yarn-multilingual/converted/checkpoint_0001000
OUT_DIR=${OUT_DIR:-$WORKDIR/eval-base-lm-niah-$SLURM_JOB_ID}

LANGUAGES=${LANGUAGES:-"fr fi cs nl"}
CTX_LENGTHS=${CTX_LENGTHS:-"2048 4096 8192 16384 32768"}
DEPTHS=${DEPTHS:-"0.0 0.25 0.5 0.75 1.0"}
TRIALS=${TRIALS:-10}

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="$WORKDIR/hf"
mkdir -p "$OUT_DIR"

# Pull latest eval script
git -C "$REPO" pull --quiet 2>/dev/null || true

echo "=== Base-LM NIAH eval — $(date) ==="
echo "Model:    $MODEL_PATH"
echo "Langs:    $LANGUAGES"
echo "Ctx:      $CTX_LENGTHS"
echo "Depths:   $DEPTHS"
echo "Trials:   $TRIALS"
echo "Output:   $OUT_DIR"
echo ""

singularity exec \
    --bind /pfs,/scratch,/projappl,/project,/flash,/appl \
    "$SIF" \
    python3 -u "$REPO/scripts/eval_base_lm_niah.py" \
        --model "$MODEL_PATH" \
        --output "$OUT_DIR" \
        --languages $LANGUAGES \
        --context-lengths $CTX_LENGTHS \
        --depths $DEPTHS \
        --trials "$TRIALS"

echo ""
echo "=== Done — $(date) ==="
echo "Summary: $OUT_DIR/summary.json"
cat "$OUT_DIR/summary.json" 2>/dev/null || true
