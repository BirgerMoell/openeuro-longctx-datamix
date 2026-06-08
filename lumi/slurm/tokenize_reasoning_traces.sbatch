#!/bin/bash
#SBATCH --job-name=tokenize-reasoning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --account=project_465002530
#SBATCH -o /scratch/project_462000963/bmoell/tokenize-reasoning-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/tokenize-reasoning-%j.err

# Tokenize am-deepseek-r1-think reasoning traces into Megatron bin/idx format.
#
# Source: /scratch/project_462000963/datasets/posttraining_data/Megatron_format/
#         am-deepseek-r1-think/{math,code,science,if,multiturn,other}.jsonl
#
# Each category is tokenized into its own bin/idx pair so subsets can be
# included independently in a training data mix.
#
# Output: /scratch/project_462000963/preprocessed/oellm-v1-256k/reasoning-traces/
#         am-deepseek-r1-think/{category}_text_document.{bin,idx}
#
# Tokenizer: oellm-datamix-9b-80-20 (same as all other oellm-v1-256k datasets)
#
# To include in a training mix, point at the prefix:
#   /scratch/project_462000963/preprocessed/oellm-v1-256k/reasoning-traces/am-deepseek-r1-think/{category}_text_document
#
# Note: the JSONL "text" field contains Llama-3 chat-formatted traces with
# <think>...</think> blocks. These are tokenized as-is (full trace including
# system/user/assistant headers). Strip the chat template first if you want
# plain text only.
#
# Submit: sbatch lumi/slurm/tokenize_reasoning_traces.sbatch

set -euo pipefail

CONTAINER=/scratch/project_462000963/containers/staging/rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0.sif
BIND_DIRS="/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd"
MEGATRON_DIR=/flash/project_462000963/jouni/test/NVIDIA-Megatron-LM
TOKENIZER=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20

SRC_DIR=/scratch/project_462000963/datasets/posttraining_data/Megatron_format/am-deepseek-r1-think
OUT_DIR=/scratch/project_462000963/preprocessed/oellm-v1-256k/reasoning-traces/am-deepseek-r1-think

mkdir -p "$OUT_DIR"

echo "=== Tokenize am-deepseek-r1-think reasoning traces — $(date) ==="
echo "Source: $SRC_DIR"
echo "Output: $OUT_DIR"
echo ""

# Tokenize each category independently (skip combined_shuffled — it's a duplicate)
CATEGORIES="math code science if multiturn other"

singularity exec \
  -B "$BIND_DIRS" \
  "$CONTAINER" \
  bash -s "$SRC_DIR" "$MEGATRON_DIR" "$TOKENIZER" "$OUT_DIR" "$CATEGORIES" << 'INNER'
set -euo pipefail

SRC_DIR=$1
MEGATRON_DIR=$2
TOKENIZER=$3
OUT_DIR=$4
CATEGORIES=$5

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"

for category in $CATEGORIES; do
    INPUT="$SRC_DIR/${category}.jsonl"
    OUTPUT_PREFIX="$OUT_DIR/${category}"

    if [ ! -f "$INPUT" ]; then
        echo "  [$category] WARNING: $INPUT not found — skipping"
        continue
    fi

    n_lines=$(wc -l < "$INPUT")
    size=$(du -sh "$INPUT" | cut -f1)
    echo ""
    echo "  ── $category ── ($n_lines docs, $size)"
    echo "     input:  $INPUT"
    echo "     output: ${OUTPUT_PREFIX}_text_document.{bin,idx}"

    python3 -u "$MEGATRON_DIR/tools/preprocess_data.py" \
        --input         "$INPUT" \
        --output-prefix "$OUTPUT_PREFIX" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "$TOKENIZER" \
        --json-keys     text \
        --workers       8 \
        --append-eod

    bin_size=$(du -sh "${OUTPUT_PREFIX}_text_document.bin" 2>/dev/null | cut -f1 || echo "?")
    echo "     done → ${bin_size} bin"
done

echo ""
echo "=== Summary ==="
echo "Output files:"
ls -lh "$OUT_DIR"/*.bin 2>/dev/null | awk '{print $5, $9}' || echo "(none)"
echo ""

# Emit a manifest with per-category sizes for use in data mix planning
python3 -u - "$OUT_DIR" "$CATEGORIES" << 'PYEOF'
import sys, pathlib, json

out_dir    = pathlib.Path(sys.argv[1])
categories = sys.argv[2].split()

manifest = {}
for cat in categories:
    bin_path = out_dir / f"{cat}_text_document.bin"
    idx_path = out_dir / f"{cat}_text_document.idx"
    if bin_path.exists():
        manifest[cat] = {
            "prefix": str(out_dir / f"{cat}_text_document"),
            "bin_bytes": bin_path.stat().st_size,
            "bin_gb": round(bin_path.stat().st_size / 1e9, 2),
        }

total_gb = sum(v["bin_gb"] for v in manifest.values())

print("\nCategory sizes (tokenized):")
for cat, info in sorted(manifest.items(), key=lambda x: -x[1]["bin_bytes"]):
    pct = 100 * info["bin_bytes"] / (total_gb * 1e9) if total_gb else 0
    print(f"  {cat:<12} {info['bin_gb']:>6.1f} GB  ({pct:4.1f}%)")
print(f"  {'TOTAL':<12} {total_gb:>6.1f} GB")

manifest_path = out_dir / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump({"categories": manifest, "total_gb": total_gb}, f, indent=2)
print(f"\nManifest written to {manifest_path}")
PYEOF

INNER

echo ""
echo "=== Done — $(date) ==="
echo "Tokenized output: $OUT_DIR"
