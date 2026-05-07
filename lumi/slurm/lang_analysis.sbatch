#!/bin/bash
#SBATCH -p dev-g
#SBATCH -t 02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH -J longctx-langstats
#SBATCH -A project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/longctx-langstats-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longctx-langstats-%j.err

# Download 1 sample shard of each target language, convert, and report
# token-length statistics per language. Results go to lang_stats.json
# and a human-readable table.
# Submit: sbatch lumi/slurm/lang_analysis.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME=$WORKDIR/hf

# Target languages: Nordic, Baltic, Central European
LANGUAGES="mt sv no da fi et lv lt pl cs sk hu ro bg hr sl"

echo "=== Language data analysis — $(date) ==="
echo "Languages: $LANGUAGES"

# Run everything inside the container so longctx and Python are available
singularity exec "$SIF" bash -s "$REPO" "$LANGUAGES" << 'INNER'
set -euo pipefail
REPO=$1
LANGUAGES=$2
cd "$REPO"

# Set up venv with longctx installed (matches hello.sbatch pattern)
python -m venv .venv --system-site-packages 2>/dev/null || true
source .venv/bin/activate
pip install -q -e . 2>/dev/null

STATS_FILE="$REPO/lang_stats.json"
echo "{}" > "$STATS_FILE"

for LANG in $LANGUAGES; do
  echo ""
  echo "══════ $LANG ══════════════════════════════════════════════"

  # 1. Estimate what's available
  echo "  [estimate]"
  longctx estimate --languages "$LANG" 2>/dev/null || echo "  (estimate unavailable)"

  # 2. Download 1 sample shard (skips if already cached)
  echo "  [download]"
  longctx download --languages "$LANG" --sample --shards 1 2>/dev/null \
    && echo "  download OK" \
    || { echo "  download FAILED — skipping $LANG"; continue; }

  # 3. Convert parquet → JSONL
  echo "  [convert]"
  longctx convert 2>/dev/null \
    && echo "  convert OK" \
    || { echo "  convert FAILED — skipping $LANG"; continue; }

  # 4. Analyse converted file (before filtering)
  JSONL="$REPO/data/megatron/${LANG}.jsonl"
  if [ ! -f "$JSONL" ]; then
    echo "  No JSONL at $JSONL — skipping"
    continue
  fi

  python3 -u - "$LANG" "$JSONL" "$STATS_FILE" << 'PYEOF'
import json, sys, os

lang   = sys.argv[1]
jsonl  = sys.argv[2]
stats_path = sys.argv[3]

counts = []
with open(jsonl) as f:
    for line in f:
        d = json.loads(line)
        counts.append(d.get("token_count", 0))

if not counts:
    print("  Empty file")
    sys.exit(0)

counts.sort()
n = len(counts)
total = sum(counts)

def pct(lst, p):
    return lst[min(int(len(lst)*p/100), len(lst)-1)]

thresholds = [4096, 8192, 16384, 32768, 65536, 131072]

print(f"  docs={n:,}  total_tokens={total:,}")
print(f"  min={counts[0]:,}  median={pct(counts,50):,}  p90={pct(counts,90):,}  max={counts[-1]:,}")
print(f"  Context coverage:")
for t in thresholds:
    d = sum(1 for c in counts if c >= t)
    tok = sum(c for c in counts if c >= t)
    pct_d = d/n*100
    print(f"    >= {t:>7,}: {d:>6,} docs ({pct_d:5.1f}%)  {tok:>12,} tokens")

# Update JSON stats file
with open(stats_path) as f:
    stats = json.load(f)

stats[lang] = {
    "docs": n,
    "total_tokens": total,
    "min": counts[0],
    "median": pct(counts, 50),
    "p90": pct(counts, 90),
    "max": counts[-1],
    "coverage": {
        str(t): {
            "docs": sum(1 for c in counts if c >= t),
            "tokens": sum(c for c in counts if c >= t)
        }
        for t in thresholds
    }
}

with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
PYEOF

done

# ── Aggregate table ───────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "AGGREGATE TABLE (1 sample shard per language)"
echo "══════════════════════════════════════════════════════════════"

python3 -u - "$STATS_FILE" << 'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

if not data:
    print("No data collected.")
    sys.exit(0)

hdr = f"{'Lang':>5} | {'Docs':>8} | {'MTok':>8} | {'Median':>8} | {'p90':>8} | {'>=4K':>7} | {'>=8K':>7} | {'>=32K':>7} | {'>=128K':>8}"
print(hdr)
print("-" * len(hdr))

for lang, s in sorted(data.items()):
    cov = s["coverage"]
    print(
        f"{lang:>5} | {s['docs']:>8,} | {s['total_tokens']/1e6:>8.1f} | "
        f"{s['median']:>8,} | {s['p90']:>8,} | "
        f"{cov.get('4096',{}).get('docs',0):>7,} | "
        f"{cov.get('8192',{}).get('docs',0):>7,} | "
        f"{cov.get('32768',{}).get('docs',0):>7,} | "
        f"{cov.get('131072',{}).get('docs',0):>8,}"
    )

print()
print("MTok = millions of tokens in the sample shard.")
print("Coverage columns = number of docs >= that token threshold.")
PYEOF

echo ""
echo "Stats saved to: $REPO/lang_stats.json"
echo "=== Done — $(date) ==="
INNER
