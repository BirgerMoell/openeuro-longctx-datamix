#!/bin/bash
#SBATCH --job-name=eval-oneruler-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=dev-g
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/eval-oneruler-smoke-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/eval-oneruler-smoke-%j.err

# Smoke eval: OneRuler-OELLM NIAH on our YaRN multilingual checkpoint.
#
# Runs niah_single across 3 context lengths (4K / 16K / 32K tokens) on
# a subset of our 8 training languages (fr, fi, cs) to confirm that the
# 32K context extension is working on downstream retrieval tasks.
#
# Model: iter_0001000 HuggingFace checkpoint (converted from Megatron,
#        rope_scaling patched to yarn factor=16.0)
# Backend: HuggingFace (loads model directly, no vLLM server needed)
#
# Context-word targets approximate token counts via the ~1.3x word/token ratio:
#   --context-words  3000 ≈  4K tokens
#   --context-words 12000 ≈ 16K tokens
#   --context-words 24000 ≈ 32K tokens
#
# Results written to: $OUT_DIR/{context_length}/{lang}_results.jsonl
#
# Submit: sbatch lumi/slurm/eval_oneruler_smoke.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
ONERULER_DIR=$WORKDIR/OneRuler-OELLM

MODEL_PATH=/flash/project_462000963/bmoell/yarn-multilingual/converted/checkpoint_0001000
OUT_DIR=/scratch/project_462000963/bmoell/eval-oneruler-smoke

LANGUAGES="fr fi cs"
QUESTIONS=5

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="$WORKDIR/hf"
mkdir -p "$OUT_DIR"

# Clone or update OneRuler-OELLM
if [ ! -d "$ONERULER_DIR" ]; then
    echo "=== Cloning OneRuler-OELLM ==="
    git clone https://github.com/BirgerMoell/OneRuler-OELLM.git "$ONERULER_DIR"
else
    echo "=== Updating OneRuler-OELLM ==="
    git -C "$ONERULER_DIR" pull --quiet
fi

echo ""
echo "=== OneRuler NIAH smoke eval — $(date) ==="
echo "Model:     $MODEL_PATH"
echo "Languages: $LANGUAGES"
echo "Questions: $QUESTIONS per language per context length"
echo "Output:    $OUT_DIR"
echo ""

run_eval() {
    local ctx_words=$1
    local ctx_label=$2
    local out_subdir="$OUT_DIR/${ctx_label}"
    mkdir -p "$out_subdir"

    echo "--- Context: $ctx_label (~${ctx_words} words) — $(date) ---"
    singularity exec \
        --bind /pfs,/scratch,/projappl,/project,/flash,/appl \
        "$SIF" \
        python3 -u "$ONERULER_DIR/scripts/run_oellm_mini_eval.py" \
            --backend huggingface \
            --model "$MODEL_PATH" \
            --languages $LANGUAGES \
            --questions "$QUESTIONS" \
            --context-words "$ctx_words" \
            --num-predict 64 \
            --output-dir "$out_subdir"
    echo "--- Done: $ctx_label — $(date) ---"
    echo ""
}

# Install deps inside the SIF once
singularity exec --bind /pfs,/scratch,/projappl,/project,/flash,/appl "$SIF" \
    pip install --quiet --user accelerate 2>/dev/null || true

run_eval  3000  "4k"
run_eval 12000 "16k"
run_eval 24000 "32k"

echo "=== All context lengths complete — $(date) ==="
echo ""
echo "Results:"
find "$OUT_DIR" -name "*.jsonl" | sort | while read f; do
    echo "  $f"
    python3 -c "
import json, sys
lines = [json.loads(l) for l in open('$f') if l.strip()]
if lines:
    correct = sum(1 for l in lines if l.get('correct', False))
    print(f'    {correct}/{len(lines)} correct ({100*correct/len(lines):.0f}%)')
" 2>/dev/null || true
done
