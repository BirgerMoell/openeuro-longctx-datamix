#!/bin/bash
#SBATCH --job-name=tokenize-tiers-mini
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --time=01:00:00
#SBATCH --account=project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/tokenize-tiers-mini-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/tokenize-tiers-mini-%j.err

# Fast mini tokenization for smoke-test purposes only.
# Reads only the first MAX_DOCS_PER_LANG lines from each language JSONL,
# so it completes in minutes rather than hours.
# Writes to the same output paths as tokenize_tiers.sbatch so
# yarn_multilingual_test.sbatch works unchanged.
#
# DO NOT use the output for a real training run — use tokenize_tiers.sbatch
# for the full balanced dataset.
#
# Submit: sbatch lumi/slurm/tokenize_tiers_mini.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix
MEGATRON_DIR=/flash/project_462000963/jouni/test/NVIDIA-Megatron-LM
TOKENIZER=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20
OUT_DIR=/flash/project_462000963/bmoell/data_tokenized_multilingual
TIER_TMP=$WORKDIR/tier_jsonl_tmp_mini

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

mkdir -p "$OUT_DIR" "$TIER_TMP"

echo "=== Tokenize multilingual tiers (MINI) — $(date) ==="

singularity exec \
  "$SIF" \
  bash -s "$REPO" "$MEGATRON_DIR" "$TOKENIZER" "$OUT_DIR" "$TIER_TMP" << 'INNER'
set -euo pipefail
REPO=$1
MEGATRON_DIR=$2
TOKENIZER=$3
OUT_DIR=$4
TIER_TMP=$5

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false   # prevent HF fast tokenizer from forking unsafely
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

echo ""
echo "=== Step 1: Sampling and splitting JSONL files ==="

LONG_JSONL=$TIER_TMP/multilingual_16k_plus.jsonl
MED_JSONL=$TIER_TMP/multilingual_4_16k.jsonl
SHORT_JSONL=$TIER_TMP/multilingual_under4k.jsonl

python3 -u - "$REPO/data/megatron" "$LONG_JSONL" "$MED_JSONL" "$SHORT_JSONL" << 'PYEOF'
import json, sys, os, glob

data_dir  = sys.argv[1]
out_long  = sys.argv[2]
out_med   = sys.argv[3]
out_short = sys.argv[4]

# Read at most this many lines per language file — low enough to stay under
# 1000 docs per tier and avoid multiprocessing pool worker respawn crash
MAX_DOCS_PER_LANG = 50

jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
print(f"Found {len(jsonl_files)} language JSONL files (sampling first {MAX_DOCS_PER_LANG} docs each)")

counts = {"long": 0, "med": 0, "short": 0}

with open(out_long,  "w") as fl, \
     open(out_med,   "w") as fm, \
     open(out_short, "w") as fs:
    for path in jsonl_files:
        lang = os.path.basename(path).replace(".jsonl", "")
        n = 0
        with open(path) as f:
            for line in f:
                if n >= MAX_DOCS_PER_LANG:
                    break
                d = json.loads(line)
                tc = d.get("token_count", 0)
                if tc >= 16384:
                    fl.write(line)
                    counts["long"] += 1
                elif tc >= 4096:
                    fm.write(line)
                    counts["med"] += 1
                else:
                    fs.write(line)
                    counts["short"] += 1
                n += 1
        print(f"  {lang}: {n} docs sampled")

total = sum(counts.values())
print(f"\nTier totals (mini sample):")
print(f"  >= 16K  (long):  {counts['long']:>6,}")
print(f"  4K–16K  (med):   {counts['med']:>6,}")
print(f"  < 4K    (short): {counts['short']:>6,}")
print(f"  Total:           {total:>6,}")
PYEOF

echo "Tier JSONL sizes:"
du -sh $LONG_JSONL $MED_JSONL $SHORT_JSONL

echo ""
echo "=== Step 2: Tokenizing tiers ==="

for TIER in "long:$LONG_JSONL:multilingual_16k_plus" \
            "med:$MED_JSONL:multilingual_4_16k" \
            "short:$SHORT_JSONL:multilingual_under4k"; do
  TIER_NAME=$(echo $TIER | cut -d: -f1)
  TIER_FILE=$(echo $TIER | cut -d: -f2)
  OUT_PREFIX="$OUT_DIR/$(echo $TIER | cut -d: -f3)"

  if [ ! -f "$TIER_FILE" ] || [ ! -s "$TIER_FILE" ]; then
    echo "  Skipping empty tier: $TIER_NAME"
    continue
  fi

  echo "  Tokenizing $TIER_NAME → $OUT_PREFIX"
  python3 -u $MEGATRON_DIR/tools/preprocess_data.py \
    --input         "$TIER_FILE" \
    --output-prefix "$OUT_PREFIX" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER" \
    --json-keys     text \
    --workers       1 \
    --append-eod
  echo "  Done: $(ls -lh ${OUT_PREFIX}_text_document.bin 2>/dev/null || echo 'no bin')"
done

echo ""
echo "=== Output files ==="
ls -lh $OUT_DIR/

echo ""
echo "=== Done (mini) — $(date) ==="
INNER
