#!/bin/bash
#SBATCH --job-name=niah-v2-ext2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=small-g
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000963
#SBATCH -o /flash/project_462000963/bmoell/niah-v2-ext2-%j.out
#SBATCH -e /flash/project_462000963/bmoell/niah-v2-ext2-%j.err

# Follow-up to eval_base_lm_niah_yarn_v2_extended.sbatch (job 18746959).
# That job crashed with "Disk quota exceeded" on scratch after completing
# EN, BG, DA. This job covers the remaining 28 languages (DE through UK),
# writing outputs to /flash which has ~84TB free.
#
# Grid:
#   context lengths : 2048  4096  8192  16384  32768  tokens
#   needle depths   : 0%  25%  50%  75%  100%
#   languages       : 28 remaining (de el es et ga hr hu it lt lv mt pl pt ro sk sl sv ca eu gl is lb mk no oc sq sr uk)
#   trials per cell : 5
#
# Submit: sbatch lumi/slurm/eval_base_lm_niah_yarn_v2_ext2.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
FLASH=/flash/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix

MODEL_PATH=$FLASH/yarn-multilingual-v2-1k/hf
# Write outputs to flash (84 TB free) to avoid scratch quota issues
OUT_DIR=$FLASH/niah-v2-ext2-$SLURM_JOB_ID

# Remaining 28 languages: DE through UK (EN, BG, DA already done in job 18746959)
LANGUAGES=${LANGUAGES:-"de el es et ga hr hu it lt lv mt pl pt ro sk sl sv ca eu gl is lb mk no oc sq sr uk"}
CTX_LENGTHS=${CTX_LENGTHS:-"2048 4096 8192 16384 32768"}
DEPTHS=${DEPTHS:-"0.0 0.25 0.5 0.75 1.0"}
TRIALS=${TRIALS:-5}

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="$WORKDIR/hf"
mkdir -p "$OUT_DIR"

git -C "$REPO" pull --quiet 2>/dev/null || true

echo "=== YaRN v2 Extended Base-LM NIAH eval (part 2) — $(date) ==="
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
