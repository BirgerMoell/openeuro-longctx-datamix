#!/bin/bash
#SBATCH --job-name=download-tokenized
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --account=project_465002530
#SBATCH -o /scratch/project_462000963/bmoell/download-tokenized-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/download-tokenized-%j.err

# Download pre-tokenized Megatron bin/idx files from:
#   birgermoell/oellm-longctx-tokenized-streamed-all-v2
#
# This dataset contains 312 bin/idx pairs already tokenized with the
# openeurollm tokenizer-256k (vocab 262144), organized as:
#   8 languages × 3 tiers × multiple shards/chunks
#   Estimated total: ~114 GB, ~35B tokens, ~3M docs
#
# Languages: bg cs da et fi fr hr nl
# Tiers: 16k_plus (≥16384 tokens), 4_16k (4096–16383), under4k (<4096)
#
# After download this job:
#   1. Organises chunks into per-language-per-tier directories
#   2. Merges each with merge_datasets.py → 24 merged files
#   3. Writes data_path.args with uniform per-language weighting
#
# Prerequisites:
#   HF_TOKEN env var set (or dataset is public — currently public)
#
# Output:
#   $OUT_DIR/per_lang_tier/<lang>_<tier>/   — raw downloaded chunks
#   $OUT_DIR/merged/<lang>_<tier>_text_document.{bin,idx}  — merged per lang/tier
#   $OUT_DIR/data_path.args                 — ready for yarn_multilingual.sbatch
#
# Submit: sbatch lumi/slurm/download_tokenized.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
MEGATRON_DIR=/flash/project_462000963/jouni/test/NVIDIA-Megatron-LM
OUT_DIR=/flash/project_462000963/bmoell/data_tokenized_hf_multilingual

HF_DATASET=birgermoell/oellm-longctx-tokenized-streamed-all-v2
HF_RUN=lc16k_full_20260507

# Optional: restrict to one language for smoke testing.
# Pass as: sbatch --export=ALL,LANGUAGES_FILTER=et download_tokenized.sbatch
# Leave empty to download all 8 languages.
LANGUAGES_FILTER=${LANGUAGES_FILTER:-""}

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export HF_HOME="$WORKDIR/hf"
mkdir -p "$OUT_DIR/per_lang_tier" "$OUT_DIR/merged"

echo "=== Download pre-tokenized multilingual Megatron data — $(date) ==="
echo "Dataset: $HF_DATASET"
echo "Run:     $HF_RUN"
echo "Output:  $OUT_DIR"

singularity exec "$SIF" python3 -u - \
    "$HF_DATASET" "$HF_RUN" "$OUT_DIR" "$MEGATRON_DIR" "$HF_HOME" "${LANGUAGES_FILTER}" \
<< 'PYEOF'
import sys, os, re, json, shutil
from pathlib import Path

dataset_id       = sys.argv[1]
run_name         = sys.argv[2]
out_dir          = sys.argv[3]
megatron         = sys.argv[4]
hf_home          = sys.argv[5]
languages_filter = [l.strip() for l in sys.argv[6].split(",") if l.strip()] if len(sys.argv) > 6 and sys.argv[6] else []

os.environ["HF_HOME"] = hf_home
os.environ["PYTHONPATH"] = f"{megatron}:{os.environ.get('PYTHONPATH','')}"

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub import constants as hf_const

per_lang_dir = Path(out_dir) / "per_lang_tier"
merged_dir   = Path(out_dir) / "merged"

# ── Step 1: list all bin/idx files for this run ──────────────────────────────
print(f"\n=== Step 1: listing files in {dataset_id} ===")
all_files = list(list_repo_files(dataset_id, repo_type="dataset"))
run_prefix = f"runs/{run_name}/bin/"
target_files = [f for f in all_files if f.startswith(run_prefix) and
                (f.endswith(".bin") or f.endswith(".idx"))]

if languages_filter:
    target_files = [f for f in target_files
                    if any(f"_{lang}_" in Path(f).name for lang in languages_filter)]
    print(f"Language filter active: {languages_filter}")

print(f"Found {len(target_files)} bin/idx files to download")

# ── Step 2: download and sort into per-lang-tier directories ─────────────────
print(f"\n=== Step 2: downloading to {per_lang_dir} ===")
pattern = re.compile(
    rf"runs/{run_name}/bin/{run_name}_([a-z]+)_s\d+_c\d+_(16k_plus|4_16k|under4k)_text_document\.(bin|idx)"
)

downloaded = 0
for remote_path in sorted(target_files):
    m = pattern.match(remote_path)
    if not m:
        print(f"  SKIP (no match): {remote_path}")
        continue
    lang, tier, ext = m.group(1), m.group(2), m.group(3)
    dest_dir = per_lang_dir / f"{lang}_{tier}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(remote_path).name
    dest_file = dest_dir / fname

    if dest_file.exists():
        print(f"  already exists: {fname}")
        downloaded += 1
        continue

    print(f"  downloading: {remote_path} → {dest_dir}/")
    local = hf_hub_download(
        repo_id=dataset_id,
        filename=remote_path,
        repo_type="dataset",
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    # hf_hub_download may create subdirs; move file to dest_dir flat
    downloaded_path = Path(local)
    flat_dest = dest_dir / downloaded_path.name
    if downloaded_path != flat_dest:
        shutil.move(str(downloaded_path), str(flat_dest))
    downloaded += 1

print(f"Downloaded/verified {downloaded} files")

# ── Step 3: merge per language per tier using merge_datasets.py ──────────────
print(f"\n=== Step 3: merging into {merged_dir} ===")
import subprocess

lang_tier_dirs = sorted(per_lang_dir.iterdir())
for lt_dir in lang_tier_dirs:
    if not lt_dir.is_dir():
        continue
    lt_name = lt_dir.name          # e.g. "fr_16k_plus"
    out_prefix = str(merged_dir / lt_name)
    bin_out = out_prefix + "_text_document.bin"

    if Path(bin_out).exists():
        print(f"  already merged: {lt_name}")
        continue

    # merge_datasets.py expects all bin/idx files in the input directory
    # but files are named *_text_document.bin — strip suffix for the output
    print(f"  merging {lt_name} ...")
    result = subprocess.run(
        [sys.executable, f"{megatron}/tools/merge_datasets.py",
         "--input", str(lt_dir),
         "--output-prefix", out_prefix + "_text_document"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
    else:
        size = Path(bin_out).stat().st_size if Path(bin_out).exists() else 0
        print(f"  done: {lt_name} ({size/1e6:.0f} MB)")

# ── Step 4: generate data_path.args with uniform per-language weighting ───────
print(f"\n=== Step 4: generating data_path.args ===")

TIER_WEIGHT = {"16k_plus": 0.5, "4_16k": 0.3, "under4k": 0.2}
TIERS = ["16k_plus", "4_16k", "under4k"]

# Discover which languages were merged successfully
langs = sorted(set(
    d.name.rsplit("_", 1)[0]
    for d in merged_dir.iterdir()
    if d.name.endswith(".bin")
    and any(d.name.endswith(f"_{t}_text_document.bin") for t in TIERS)
))

# Strip _text_document.bin to get lang_tier names
def merged_langs_for_tier(tier):
    return sorted(set(
        f.stem.replace(f"_{tier}_text_document", "")
        for f in merged_dir.glob(f"*_{tier}_text_document.bin")
    ))

# Build per-lang entries: weight = tier_weight / num_langs
entries = []
for tier in TIERS:
    tier_langs = merged_langs_for_tier(tier)
    if not tier_langs:
        continue
    per_lang_weight = TIER_WEIGHT[tier] / len(tier_langs)
    for lt in tier_langs:
        prefix = str(merged_dir / f"{lt}_{tier}_text_document")
        entries.append(f"{per_lang_weight:.8f} {prefix}")

data_path_args = " ".join(entries)
args_file = Path(out_dir) / "data_path.args"
args_file.write_text(data_path_args)

print(f"Languages: {langs}")
print(f"Written {args_file}")
print(f"\nDATA_PATH preview (first 3 entries):")
for e in entries[:3]:
    print(f"  {e}")
print(f"  ... ({len(entries)} total entries)")

# ── Step 5: document-length report (from .idx files) ─────────────────────────
# Each merged .idx stores per-document token counts as int32 arrays.
# This confirms rebinning is possible at any future tier boundary.
print(f"\n=== Step 5: document length report ===")
from megatron.core.datasets.indexed_dataset import IndexedDataset
import numpy as np

lang_tier_stats = {}
for bin_f in sorted(merged_dir.glob("*_text_document.bin")):
    prefix = str(bin_f).replace(".bin", "")   # keep _text_document in prefix
    name = bin_f.stem.replace("_text_document", "")
    try:
        ds = IndexedDataset(prefix)
        lengths = ds.sequence_lengths  # numpy int32 array, one entry per doc
        lang_tier_stats[name] = {
            "docs": len(lengths),
            "tokens": int(lengths.sum()),
            "min": int(lengths.min()),
            "median": int(np.median(lengths)),
            "p90": int(np.percentile(lengths, 90)),
            "max": int(lengths.max()),
        }
        s = lang_tier_stats[name]
        print(f"  {name}: {s['docs']:,} docs | {s['tokens']/1e6:.0f}M tokens | "
              f"min={s['min']} med={s['median']} p90={s['p90']} max={s['max']}")
    except Exception as e:
        print(f"  {name}: ERROR reading idx — {e}")

# Write length stats as JSON for future rebinning
stats_file = Path(out_dir) / "length_stats.json"
stats_file.write_text(json.dumps(lang_tier_stats, indent=2))
print(f"\nLength stats written to: {stats_file}")

# ── Step 6: summary ────────────────────────────────────────────────────────
print(f"\n=== Summary ===")
bin_files = list(merged_dir.glob("*.bin"))
total_size = sum(f.stat().st_size for f in bin_files)
print(f"Merged files: {len(bin_files)}")
print(f"Total size:   {total_size/1e9:.1f} GB")
print(f"\n=== Done ===")

print(f"\nTo rebin at different boundaries, use length_stats.json as input.")
print(f"Each merged .idx stores per-doc token counts via dataset.sequence_lengths.")
PYEOF

echo ""
echo "=== Download and merge complete — $(date) ==="
echo "Merged data: $OUT_DIR/merged/"
echo "DATA_PATH:   $OUT_DIR/data_path.args"
echo ""
echo "To use in yarn_multilingual.sbatch:"
echo "  DATA_PATH=\"\$(cat $OUT_DIR/data_path.args)\""
