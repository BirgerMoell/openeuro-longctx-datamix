#!/bin/bash
#SBATCH --job-name=niah-yarn-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=dev-g
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/niah-yarn-v2-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/niah-yarn-v2-%j.err

# Base-LM NIAH eval for YaRN v2 multilingual model (iter 1000).
#
# v2 key fix: mscale=1.277 (= 0.1*ln(16)+1.0), which should eliminate the
# depth=0% failure at 32K seen in v1 (mscale was missing / defaulted to 0).
#
# Grid:
#   context lengths : 2048  4096  8192  16384  32768  tokens
#   needle depths   : 0%  25%  50%  75%  100%
#   languages       : fr  fi  cs  nl
#   trials per cell : 10
#
# Submit: sbatch lumi/slurm/eval_base_lm_niah_yarn_v2.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix

MODEL_PATH=/flash/project_462000963/bmoell/yarn-multilingual-v2-1k/hf
OUT_DIR=$WORKDIR/niah-yarn-v2-$SLURM_JOB_ID

LANGUAGES=${LANGUAGES:-"fr fi cs nl"}
CTX_LENGTHS=${CTX_LENGTHS:-"2048 4096 8192 16384 32768"}
DEPTHS=${DEPTHS:-"0.0 0.25 0.5 0.75 1.0"}
TRIALS=${TRIALS:-10}

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="$WORKDIR/hf"
mkdir -p "$OUT_DIR"

git -C "$REPO" pull --quiet 2>/dev/null || true

echo "=== YaRN v2 Base-LM NIAH eval — $(date) ==="
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
