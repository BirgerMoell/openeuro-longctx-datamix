#!/bin/bash
#SBATCH --job-name=longrope-search-tokenize
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --partition=standard-g
#SBATCH --time=04:00:00
#SBATCH --account=project_465002530
#SBATCH -o /scratch/project_462000963/bmoell/longrope-search-tokenize-%j.out
#SBATCH -e /scratch/project_462000963/bmoell/longrope-search-tokenize-%j.err

# Build the multilingual proxy dataset for the LongRoPE rescaling-factor search.
#
# Strategy: sample from the 12 highest-resource languages (by ≥32K doc count),
# covering 5 language families. Each language contributes docs proportionally
# to its ≥32K token count — this produces a linguistically diverse proxy
# representative of the full multilingual training distribution.
#
# Languages selected (all have ≥3000 docs ≥32K tokens per shard):
#   Germanic:   en
#   Romance:    ro, es, pt
#   Slavic:     pl, hr, sr, cs, sk, uk, ru
#   Finno-Ugric: hu, fi
#   Turkic:     tr
#   Hellenic:   el
#
# Output: HuggingFace Arrow dataset at $OUT_DIR (load_from_disk compatible)
#         ready for longrope_search.sbatch
#
# Prerequisites: data/megatron/*.jsonl must exist (longctx convert already done)
#
# Submit: sbatch lumi/slurm/longrope_search_tokenize.sbatch

set -euo pipefail

SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
WORKDIR=/scratch/project_462000963/bmoell
REPO=$WORKDIR/openeuro-longctx-datamix
TOKENIZER=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20
OUT_DIR=/flash/project_462000963/bmoell/longrope-search/datasets/multilingual-proxy-tokenized

if ! type module &>/dev/null; then source /etc/profile; fi
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

mkdir -p "$OUT_DIR" "$(dirname $OUT_DIR)"
export HF_HOME="$WORKDIR/hf"

echo "=== LongRoPE search: build multilingual proxy dataset — $(date) ==="
echo "Source JSONL: $REPO/data/megatron/"
echo "Tokenizer:    $TOKENIZER"
echo "Output:       $OUT_DIR"

singularity exec "$SIF" python3 -u - \
    "$REPO/data/megatron" \
    "$TOKENIZER" \
    "$OUT_DIR" \
    "$HF_HOME" \
<< 'PYEOF'
import json, sys, os, glob, random
from collections import defaultdict

data_dir   = sys.argv[1]
tok_path   = sys.argv[2]
out_dir    = sys.argv[3]
hf_home    = sys.argv[4]

os.environ["HF_HOME"] = hf_home
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from datasets import Dataset
from transformers import AutoTokenizer

MIN_TOKENS = 32768   # must match --dataset-min-tokens in the search script
DOCS_PER_LANG = 80  # sample up to this many ≥32K docs per language
RANDOM_SEED = 42

# Languages to include — 14 high-resource languages across 6 families
TARGET_LANGS = [
    "en",          # Germanic
    "ro", "es", "pt",           # Romance
    "pl", "hr", "sr", "cs", "sk", "uk", "ru",  # Slavic
    "hu", "fi",    # Finno-Ugric
    "el",          # Hellenic
]

random.seed(RANDOM_SEED)

print(f"Sampling up to {DOCS_PER_LANG} docs ≥{MIN_TOKENS} tokens from {len(TARGET_LANGS)} languages")

all_docs = []
lang_counts = {}

for lang in TARGET_LANGS:
    path = os.path.join(data_dir, f"{lang}.jsonl")
    if not os.path.exists(path):
        print(f"  {lang}: MISSING — skipping")
        continue

    long_docs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("token_count", 0) >= MIN_TOKENS:
                long_docs.append({"text": d["text"], "lang": lang})

    if not long_docs:
        print(f"  {lang}: 0 docs ≥{MIN_TOKENS} tokens — skipping")
        continue

    # Sample proportionally (cap at DOCS_PER_LANG)
    sample = random.sample(long_docs, min(DOCS_PER_LANG, len(long_docs)))
    lang_counts[lang] = len(sample)
    all_docs.extend(sample)
    print(f"  {lang}: {len(long_docs):,} eligible → sampled {len(sample)}")

print(f"\nTotal proxy docs: {len(all_docs):,}")
for lang, n in sorted(lang_counts.items(), key=lambda x: -x[1]):
    print(f"  {lang}: {n}")

# Tokenize
print(f"\nLoading tokenizer from {tok_path} ...")
tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_batch(examples):
    results = []
    for text in examples["text"]:
        enc = tokenizer(text, add_special_tokens=False, truncation=False)
        results.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "tokenized_len": len(enc["input_ids"]),
        })
    return {
        "input_ids":      [r["input_ids"]      for r in results],
        "attention_mask": [r["attention_mask"]  for r in results],
        "tokenized_len":  [r["tokenized_len"]   for r in results],
    }

print("Building HuggingFace dataset ...")
ds = Dataset.from_list(all_docs)

print("Tokenizing (this may take a while for long documents) ...")
ds = ds.map(tokenize_batch, batched=True, batch_size=8, num_proc=4)

# Confirm all docs are still ≥ MIN_TOKENS after tokenization
before = len(ds)
ds = ds.filter(lambda x: x["tokenized_len"] >= MIN_TOKENS)
print(f"After token-length filter: {len(ds):,} / {before:,} docs kept")

print(f"\nSaving to {out_dir} ...")
ds.save_to_disk(out_dir, num_proc=4)

print(f"\nDataset info:")
print(f"  Rows:    {len(ds):,}")
print(f"  Columns: {ds.column_names}")
print(f"  Min tokenized_len: {min(ds['tokenized_len']):,}")
print(f"  Max tokenized_len: {max(ds['tokenized_len']):,}")
print(f"\n=== Done — proxy dataset ready for longrope_search.sbatch ===")
PYEOF

echo ""
echo "=== Proxy dataset written to: $OUT_DIR ==="
echo "=== Done — $(date) ==="
