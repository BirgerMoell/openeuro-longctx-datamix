#!/bin/bash
#SBATCH --job-name=niah-v2-ext
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=small-g
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_465002530
#SBATCH -o /scratch/project_462000963/bmoell/niah-v2-ext-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/niah-v2-ext-%j.err

# Extended Base-LM NIAH eval for YaRN v2 multilingual model.
#
# Covers the 31 languages NOT in the comparison job (eval_base_lm_niah_yarn_v2.sbatch).
# That job runs fr/fi/cs/nl (matching v1 eval for direct comparison).
# This job runs the remaining 31 languages across the full 35-language model.
#
# Grid:
#   context lengths : 2048  4096  8192  16384  32768  tokens
#   needle depths   : 0%  25%  50%  75%  100%
#   languages       : all 35 model languages minus fr/fi/cs/nl
#   trials per cell : 5  (reduced from 10 to fit 12h budget)
#
# Approximate runtime: ~10h on a single MI250X (5 trials × 31 langs × 25 cells)
#
# Submit: sbatch lumi/slurm/eval_base_lm_niah_yarn_v2_extended.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix

MODEL_PATH=/flash/project_462000963/bmoell/yarn-multilingual-v2-1k/hf
OUT_DIR=$WORKDIR/niah-v2-ext-$SLURM_JOB_ID

# All 35 model languages minus the 4 comparison languages (fr fi cs nl)
# which are handled by eval_base_lm_niah_yarn_v2.sbatch for v1 comparability.
LANGUAGES=${LANGUAGES:-"en bg da de el es et ga hr hu it lt lv mt pl pt ro sk sl sv ca eu gl is lb mk no oc sq sr uk"}
CTX_LENGTHS=${CTX_LENGTHS:-"2048 4096 8192 16384 32768"}
DEPTHS=${DEPTHS:-"0.0 0.25 0.5 0.75 1.0"}
TRIALS=${TRIALS:-5}

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="$WORKDIR/hf"
mkdir -p "$OUT_DIR"

git -C "$REPO" pull --quiet 2>/dev/null || true

echo "=== YaRN v2 Extended Base-LM NIAH eval — $(date) ==="
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
