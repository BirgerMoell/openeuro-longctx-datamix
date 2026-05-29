#!/bin/bash
#SBATCH --job-name=longctx-tokenize-tiers
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --partition=standard-g
#SBATCH --time=08:00:00
#SBATCH --account=project_465002530
#SBATCH -o /scratch/project_462000963/bmoell/longctx-tokenize-tiers-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longctx-tokenize-tiers-%j.err

# Tokenize all 35 FinePDFs-Edu languages into three length-tier Megatron bin/idx files:
#   - multilingual_16k_plus    : docs with token_count >= 16384
#   - multilingual_4_16k       : docs with 4096 <= token_count < 16384
#   - multilingual_under4k     : docs with token_count < 4096
#
# Input:  /scratch/.../openeuro-longctx-datamix/data/megatron/{lang}.jsonl
# Output: /flash/project_462000963/bmoell/data_tokenized_multilingual/
#
# Submit: sbatch lumi/slurm/tokenize_tiers.sbatch

set -euo pipefail

CONTAINER=/scratch/project_462000963/containers/staging/rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0.sif
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix
MEGATRON_DIR=/flash/project_462000963/jouni/test/NVIDIA-Megatron-LM
TOKENIZER=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20
OUT_DIR=/flash/project_462000963/bmoell/data_tokenized_multilingual
TIER_TMP=$WORKDIR/tier_jsonl_tmp

mkdir -p "$OUT_DIR" "$TIER_TMP"

echo "=== Tokenize multilingual tiers — $(date) ==="
echo "Input JSONL dir: $REPO/data/megatron/"
echo "Output dir:      $OUT_DIR"
echo "Temp tier dir:   $TIER_TMP"

BIND_DIRS="/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd"

singularity exec \
  -B "$BIND_DIRS" \
  "$CONTAINER" \
  bash -s "$REPO" "$MEGATRON_DIR" "$TOKENIZER" "$OUT_DIR" "$TIER_TMP" << 'INNER'
set -euo pipefail
REPO=$1
MEGATRON_DIR=$2
TOKENIZER=$3
OUT_DIR=$4
TIER_TMP=$5

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# ── Step 1: Split all language JSONLs into 3 tier files ──────────────────────
echo ""
echo "=== Step 1: Splitting JSONL files into length tiers ==="

LONG_JSONL=$TIER_TMP/multilingual_16k_plus.jsonl
MED_JSONL=$TIER_TMP/multilingual_4_16k.jsonl
SHORT_JSONL=$TIER_TMP/multilingual_under4k.jsonl

python3 -u - "$REPO/data/megatron" "$LONG_JSONL" "$MED_JSONL" "$SHORT_JSONL" << 'PYEOF'
import json, sys, os, glob

data_dir  = sys.argv[1]
out_long  = sys.argv[2]   # >= 16384 tokens
out_med   = sys.argv[3]   # 4096 <= x < 16384
out_short = sys.argv[4]   # < 4096 tokens

jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
print(f"Found {len(jsonl_files)} language JSONL files")

counts = {"long": 0, "med": 0, "short": 0}

with open(out_long,  "w") as fl, \
     open(out_med,   "w") as fm, \
     open(out_short, "w") as fs:
    for path in jsonl_files:
        lang = os.path.basename(path).replace(".jsonl", "")
        n = 0
        with open(path) as f:
            for line in f:
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
        print(f"  {lang}: {n:,} docs processed")

total = sum(counts.values())
print(f"\nTier totals:")
print(f"  >= 16K  (long):  {counts['long']:>8,}")
print(f"  4K–16K  (med):   {counts['med']:>8,}")
print(f"  < 4K    (short): {counts['short']:>8,}")
print(f"  Total:           {total:>8,}")
PYEOF

echo "Tier JSONL files written:"
du -sh $LONG_JSONL $MED_JSONL $SHORT_JSONL

# ── Step 2: Tokenize each tier ───────────────────────────────────────────────
echo ""
echo "=== Step 2: Tokenizing tiers with Megatron preprocess_data.py ==="

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

  echo ""
  echo "  Tokenizing $TIER_NAME tier → $OUT_PREFIX"
  python3 -u $MEGATRON_DIR/tools/preprocess_data.py \
    --input        "$TIER_FILE" \
    --output-prefix "$OUT_PREFIX" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER" \
    --json-keys    text \
    --workers      4 \
    --append-eod
  echo "  Done: $(ls -lh ${OUT_PREFIX}_text_document.bin 2>/dev/null || echo 'no bin file')"
done

echo ""
echo "=== Tokenized output files ==="
ls -lh $OUT_DIR/

echo ""
echo "=== Done — $(date) ==="
INNER
