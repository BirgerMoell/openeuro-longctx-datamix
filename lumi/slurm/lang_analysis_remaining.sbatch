#!/bin/bash
#SBATCH -p dev-g
#SBATCH -t 02:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH -J longctx-langstats2
#SBATCH -A project_462000963
#SBATCH -o /scratch/project_462000963/bmoell/longctx-langstats2-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longctx-langstats2-%j.err

# Analyze the remaining 19 OpenEuroLLM languages not yet in lang_stats.json.
# Preserves existing lang_stats.json entries — only adds missing languages.
# Languages already done (job 18479845): bg cs da et fi hr hu lt lv mt no pl ro sk sl sv
#
# Remaining from FinePDFs-Edu (19):
#   EU official:         de el en es fr it nl pt
#   Additional European: bs ca cy eu gl is mk ru sr tr uk
#
# The 3 languages missing from FinePDFs-Edu (ga sq lb) need --source hplt
# and are handled in a separate section at the end.
#
# Submit with: sbatch lumi/slurm/lang_analysis_remaining.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME=$WORKDIR/hf

# All remaining FinePDFs-Edu languages (19 missing from previous run)
LANGUAGES="de el en es fr it nl pt bs ca cy eu gl is mk ru sr tr uk"

echo "=== Language data analysis (remaining) — $(date) ==="
echo "Languages: $LANGUAGES"

singularity exec "$SIF" bash -s "$REPO" "$LANGUAGES" << 'INNER'
set -euo pipefail
REPO=$1
LANGUAGES=$2
cd "$REPO"

# Set up venv with longctx installed
python -m venv .venv --system-site-packages 2>/dev/null || true
source .venv/bin/activate
pip install -q -e . 2>/dev/null

STATS_FILE="$REPO/lang_stats.json"
# Preserve existing stats — only create empty file if missing
[ -f "$STATS_FILE" ] || echo "{}" > "$STATS_FILE"

for LANG in $LANGUAGES; do
  # Skip if already analyzed
  if python3 -c "import json,sys; d=json.load(open('$STATS_FILE')); sys.exit(0 if '$LANG' in d else 1)" 2>/dev/null; then
    echo ""
    echo "══════ $LANG — already in stats, skipping ══════"
    continue
  fi

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
import json, sys

lang       = sys.argv[1]
jsonl      = sys.argv[2]
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
n     = len(counts)
total = sum(counts)

def pct(lst, p):
    return lst[min(int(len(lst)*p/100), len(lst)-1)]

thresholds = [4096, 8192, 16384, 32768, 65536, 131072]

print(f"  docs={n:,}  total_tokens={total:,}")
print(f"  min={counts[0]:,}  median={pct(counts,50):,}  p90={pct(counts,90):,}  max={counts[-1]:,}")
print(f"  Context coverage:")
for t in thresholds:
    d   = sum(1 for c in counts if c >= t)
    tok = sum(c for c in counts if c >= t)
    print(f"    >= {t:>7,}: {d:>6,} docs ({d/n*100:5.1f}%)  {tok:>12,} tokens")

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

# ── Handle ga/sq/lb via HPLT (not in FinePDFs-Edu) ───────────────────────────
echo ""
echo "══════ ga/sq/lb — attempting via HPLT source ══════════════════"
for LANG in ga sq lb; do
  if python3 -c "import json,sys; d=json.load(open('$STATS_FILE')); sys.exit(0 if '$LANG' in d else 1)" 2>/dev/null; then
    echo "  $LANG already done"
    continue
  fi
  echo "  [$LANG] trying longctx sources fetch --source hplt ..."
  longctx sources fetch --source hplt --languages "$LANG" --megatron-dir data/megatron 2>/dev/null \
    && echo "  $LANG fetch OK" \
    || { echo "  $LANG HPLT fetch FAILED (may need HF_TOKEN or is unsupported) — skipping"; continue; }

  JSONL="$REPO/data/megatron/${LANG}.jsonl"
  [ -f "$JSONL" ] || { echo "  No JSONL for $LANG — skipping"; continue; }

  python3 -u - "$LANG" "$JSONL" "$STATS_FILE" << 'PYEOF'
import json, sys
lang, jsonl, stats_path = sys.argv[1], sys.argv[2], sys.argv[3]
counts = []
with open(jsonl) as f:
    for line in f:
        counts.append(json.loads(line).get("token_count", 0))
if not counts:
    sys.exit(0)
counts.sort()
n, total = len(counts), sum(counts)
pct = lambda lst, p: lst[min(int(len(lst)*p/100), len(lst)-1)]
thresholds = [4096, 8192, 16384, 32768, 65536, 131072]
print(f"  docs={n:,}  total_tokens={total:,}")
with open(stats_path) as f:
    stats = json.load(f)
stats[lang] = {"docs": n, "total_tokens": total, "min": counts[0],
    "median": pct(counts,50), "p90": pct(counts,90), "max": counts[-1],
    "coverage": {str(t): {"docs": sum(1 for c in counts if c>=t),
        "tokens": sum(c for c in counts if c>=t)} for t in thresholds}}
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
PYEOF
done

# ── Aggregate table ───────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "AGGREGATE TABLE — all languages in lang_stats.json"
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

totals = {"docs": 0, "mtok": 0, "ge4k": 0, "ge8k": 0, "ge32k": 0, "ge128k": 0}
for lang, s in sorted(data.items()):
    cov = s["coverage"]
    ge4k   = cov.get("4096",  {}).get("docs", 0)
    ge8k   = cov.get("8192",  {}).get("docs", 0)
    ge32k  = cov.get("32768", {}).get("docs", 0)
    ge128k = cov.get("131072",{}).get("docs", 0)
    print(
        f"{lang:>5} | {s['docs']:>8,} | {s['total_tokens']/1e6:>8.1f} | "
        f"{s['median']:>8,} | {s['p90']:>8,} | "
        f"{ge4k:>7,} | {ge8k:>7,} | {ge32k:>7,} | {ge128k:>8,}"
    )
    totals["docs"]  += s["docs"]
    totals["mtok"]  += s["total_tokens"] / 1e6
    totals["ge4k"]  += ge4k
    totals["ge8k"]  += ge8k
    totals["ge32k"] += ge32k
    totals["ge128k"]+= ge128k

print("-" * len(hdr))
print(
    f"{'TOT':>5} | {totals['docs']:>8,} | {totals['mtok']:>8.1f} | "
    f"{'':>8} | {'':>8} | "
    f"{totals['ge4k']:>7,} | {totals['ge8k']:>7,} | "
    f"{totals['ge32k']:>7,} | {totals['ge128k']:>8,}"
)
print()
print("MTok = millions of tokens in the sample shard.")
print("Coverage columns = number of docs >= that token threshold.")
PYEOF

echo ""
echo "Stats saved to: $REPO/lang_stats.json"
echo "=== Done — $(date) ==="
INNER
