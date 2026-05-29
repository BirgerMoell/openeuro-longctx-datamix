#!/bin/bash
#SBATCH --job-name=tokenize-tiers-64k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=256G
#SBATCH --partition=standard-g
#SBATCH --time=12:00:00
#SBATCH --account=project_465002530
#SBATCH -o /scratch/project_462000963/bmoell/tokenize-tiers-64k-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/tokenize-tiers-64k-%j.err

# Tokenize 8 multilingual languages into 4 length-tier Megatron bin/idx files,
# with a 32K token threshold as the top tier.  Designed for 64K context training.
#
# Tiers:
#   32k_plus    : token_count >= 32768  (primary data for 64K training)
#   16_32k      : 16384 <= token_count < 32768
#   4_16k       : 4096  <= token_count < 16384
#   under4k     : token_count < 4096
#
# Input:  per-language JSONL files with a "text" and "token_count" field.
#         Default: $HF_DATA_RAW_DIR/{lang}.jsonl
#         Override: RAW_DATA_DIR env var
#
# Output: /flash/project_462000963/bmoell/data_tokenized_64k/
#         One bin/idx pair per language per tier.
#         data_path.args file ready for yarn_multilingual.sbatch at 64K.
#
# Doc-count report is printed at the end — use this to judge whether
# the 32k_plus tier has enough documents for a 64K training run.
#
# Submit: sbatch lumi/slurm/tokenize_tiers_64k.sbatch

set -euo pipefail

CONTAINER=/scratch/project_462000963/containers/staging/rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0.sif
WORKDIR=/scratch/project_462000963/bmoell
MEGATRON_DIR=/flash/project_462000963/jouni/test/NVIDIA-Megatron-LM
TOKENIZER=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20

# Raw per-language JSONL — must have "text" and "token_count" fields.
# Adjust RAW_DATA_DIR if your files are elsewhere.
RAW_DATA_DIR=${RAW_DATA_DIR:-/flash/project_462000963/bmoell/raw_jsonl_multilingual}

OUT_DIR=${OUT_DIR:-/flash/project_462000963/bmoell/data_tokenized_64k}
TIER_TMP=$WORKDIR/tier_jsonl_tmp_64k

LANGUAGES=${LANGUAGES:-"bg cs da et fi fr hr nl"}

mkdir -p "$OUT_DIR" "$TIER_TMP"

echo "=== Tokenize tiers for 64K training — $(date) ==="
echo "Raw JSONL: $RAW_DATA_DIR"
echo "Output:    $OUT_DIR"
echo "Languages: $LANGUAGES"
echo ""

BIND_DIRS="/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd"

singularity exec \
  -B "$BIND_DIRS" \
  "$CONTAINER" \
  bash -s "$RAW_DATA_DIR" "$MEGATRON_DIR" "$TOKENIZER" "$OUT_DIR" "$TIER_TMP" "$LANGUAGES" << 'INNER'
set -euo pipefail

RAW_DATA_DIR=$1
MEGATRON_DIR=$2
TOKENIZER=$3
OUT_DIR=$4
TIER_TMP=$5
LANGUAGES=$6

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"

# ── Step 1: Split each language JSONL into 4 tiers ───────────────────────────
echo "=== Step 1: Splitting per-language JSONLs into 4 tiers ==="
echo ""

python3 -u - "$RAW_DATA_DIR" "$TIER_TMP" "$LANGUAGES" << 'PYEOF'
import json, sys, os, pathlib

raw_dir   = sys.argv[1]
tier_tmp  = sys.argv[2]
languages = sys.argv[3].split()

# Thresholds
T32K = 32768
T16K = 16384
T4K  = 4096

total_counts = {"32k_plus": 0, "16_32k": 0, "4_16k": 0, "under4k": 0}
lang_stats   = {}

for lang in languages:
    src = pathlib.Path(raw_dir) / f"{lang}.jsonl"
    if not src.exists():
        print(f"  [{lang}] WARNING: {src} not found — skipping")
        continue

    out = {
        "32k_plus": open(pathlib.Path(tier_tmp) / f"{lang}_32k_plus.jsonl",  "w"),
        "16_32k":   open(pathlib.Path(tier_tmp) / f"{lang}_16_32k.jsonl",    "w"),
        "4_16k":    open(pathlib.Path(tier_tmp) / f"{lang}_4_16k.jsonl",     "w"),
        "under4k":  open(pathlib.Path(tier_tmp) / f"{lang}_under4k.jsonl",   "w"),
    }
    counts = {k: 0 for k in out}

    with open(src) as f:
        for line in f:
            d  = json.loads(line)
            tc = d.get("token_count", 0)
            if tc >= T32K:
                out["32k_plus"].write(line);  counts["32k_plus"] += 1
            elif tc >= T16K:
                out["16_32k"].write(line);    counts["16_32k"]   += 1
            elif tc >= T4K:
                out["4_16k"].write(line);     counts["4_16k"]    += 1
            else:
                out["under4k"].write(line);   counts["under4k"]  += 1

    for f in out.values():
        f.close()

    total = sum(counts.values())
    lang_stats[lang] = counts
    for k, v in counts.items():
        total_counts[k] += v
    pct_32k = 100 * counts["32k_plus"] / total if total else 0
    print(f"  [{lang}]  total={total:>8,}  "
          f"32k+={counts['32k_plus']:>7,} ({pct_32k:4.1f}%)  "
          f"16-32k={counts['16_32k']:>7,}  "
          f"4-16k={counts['4_16k']:>7,}  "
          f"<4k={counts['under4k']:>7,}")

print()
print("  ── Totals ──────────────────────────────────────────────────────────")
grand = sum(total_counts.values())
for tier, n in total_counts.items():
    print(f"  {tier:>10}: {n:>9,}  ({100*n/grand:.1f}%)")
print(f"  {'TOTAL':>10}: {grand:>9,}")
PYEOF

echo ""
echo "Tier JSONL sizes:"
du -sh "$TIER_TMP"/*.jsonl 2>/dev/null | sort -h

# ── Step 2: Tokenize per-language per-tier ────────────────────────────────────
echo ""
echo "=== Step 2: Tokenizing with Megatron preprocess_data.py ==="

for lang in $LANGUAGES; do
  echo ""
  echo "  ── Language: $lang ──"
  for tier in "32k_plus" "16_32k" "4_16k" "under4k"; do
    TIER_FILE="$TIER_TMP/${lang}_${tier}.jsonl"
    OUT_PREFIX="$OUT_DIR/${lang}_${tier}"

    if [ ! -f "$TIER_FILE" ] || [ ! -s "$TIER_FILE" ]; then
      echo "    [$lang/$tier] empty — skipping"
      continue
    fi

    n_lines=$(wc -l < "$TIER_FILE")
    echo "    [$lang/$tier] $n_lines docs → $OUT_PREFIX"
    python3 -u "$MEGATRON_DIR/tools/preprocess_data.py" \
      --input         "$TIER_FILE" \
      --output-prefix "$OUT_PREFIX" \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model "$TOKENIZER" \
      --json-keys     text \
      --workers       4 \
      --append-eod
  done
done

# ── Step 3: Generate data_path.args for 64K training ─────────────────────────
echo ""
echo "=== Step 3: Generating data_path.args ==="

python3 -u - "$OUT_DIR" "$LANGUAGES" << 'PYEOF'
import os, sys, pathlib

out_dir   = pathlib.Path(sys.argv[1])
languages = sys.argv[2].split()

# Tier weights for 64K context training.
# 32k_plus dominates since these are the only docs that span the full window.
# Weights sum to 1.0 across all languages × tiers.
n_langs = len(languages)
tier_weights = {
    "32k_plus": 0.50,
    "16_32k":   0.30,
    "4_16k":    0.15,
    "under4k":  0.05,
}

entries = []
for lang in languages:
    for tier, tw in tier_weights.items():
        prefix = out_dir / f"{lang}_{tier}_text_document"
        if not (prefix.parent / (prefix.name + ".bin")).exists():
            print(f"  WARNING: {prefix}.bin not found — omitting from data_path.args")
            continue
        weight = tw / n_langs
        entries.append((weight, str(prefix)))

if not entries:
    print("ERROR: no bin files found — check output directory")
    sys.exit(1)

# Normalise weights to sum to 1.0
total_w = sum(w for w, _ in entries)
entries = [(w / total_w, p) for w, p in entries]

args_line = " ".join(f"{w:.6f} {p}" for w, p in entries)
args_file = out_dir / "data_path.args"
args_file.write_text(args_line + "\n")

print(f"Written {len(entries)} entries to {args_file}")
print()
print("Weight breakdown by tier (across all languages):")
for tier, tw in tier_weights.items():
    per_lang = tw / n_langs
    print(f"  {tier:>10}:  {tw:.2f} total  →  {per_lang:.4f} per language")
print()
print("Use in yarn_multilingual_64k.sbatch:")
print('  DATA_PATH="$(cat ' + str(args_file) + ')"')
PYEOF

echo ""
echo "=== Final output ==="
ls -lh "$OUT_DIR"/*.bin 2>/dev/null | awk '{print $5, $9}' | sort -h || echo "(no .bin files yet)"
echo ""
echo "=== Done — $(date) ==="
INNER
