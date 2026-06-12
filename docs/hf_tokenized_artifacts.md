# HF Tokenized Artifact Handoff

This workflow lets an NVIDIA box do data prep and Megatron tokenization, then
stores the final `.bin/.idx` artifacts in a Hugging Face dataset repo for LUMI.

The Hub is used as artifact storage only. Megatron still trains from local
memory-mapped files after download.

## Low-Disk Streaming Upload

On a shared or nearly full NVIDIA box, prefer streaming upload over the
monolithic `nvidia_tokenize_tiers_pack.sh` path. It downloads one parquet shard,
splits it into tier chunks, tokenizes one chunk, uploads the `.bin/.idx` pair,
and removes local temporaries before moving on.

```bash
cd openeuro-longctx-datamix
source .venv/bin/activate

export HF_TOKEN=hf_...
export MEGATRON_LM=/home/ubuntu/birger/NVIDIA-Megatron-LM-context-extension

python -m longctx.cli stream-upload \
  --repo-id birgermoell/oellm-longctx-tokenized-streamed \
  --languages mt \
  --shards 1 \
  --chunk-docs 2000 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20 \
  --vocab-size 262144 \
  --workers 4 \
  --private
```

For a smoke test, add `--max-docs 100`. For the real run, either increase
`--languages` incrementally or use `--all-shards` once the first language has
uploaded cleanly.

The resulting HF dataset contains many Megatron prefixes instead of the three
large `multilingual_*` prefixes. That is deliberate: `mix/data_path.args`
contains the weighted list that Megatron needs, while peak disk on the NVIDIA
box stays close to one parquet shard plus one JSONL/tokenized chunk.

For 64K/128K experiments, use the higher-resolution tier preset rather than the
default coarse 16K mix:

```bash
TIER_PRESET=lc128k \
CHUNK_DOCS_BY_TIER=128k_plus=32,64_128k=64,32_64k=128,16_32k=256,4_16k=1024,under4k=4096 \
HF_REPO_ID=birgermoell/oellm-longctx-tokenized-finepdfs-lc128k-v1 \
RUN_ID=lc128k_full_20260612 \
scripts/nvidia_stream_upload_all_languages.sh
```

## NVIDIA Build

```bash
cd openeuro-longctx-datamix
python -m venv .venv
source .venv/bin/activate
pip install -e . torch transformers sentencepiece protobuf tiktoken pybind11 psutil

export MEGATRON_DIR=/home/ubuntu/birger/NVIDIA-Megatron-LM-context-extension
export PYTHON=.venv/bin/python
export TOKENIZER=/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20

# Assumes data/megatron/{lang}.jsonl already exists.
scripts/nvidia_tokenize_tiers_pack.sh
```

For a small test:

```bash
LANGUAGES=mt MAX_DOCS_PER_LANG=100 scripts/nvidia_tokenize_tiers_pack.sh
```

To upload:

```bash
export HF_TOKEN=hf_...
UPLOAD=1 HF_REPO_ID=birgermoell/oellm-longctx-tokenized PRIVATE=1 \
  scripts/nvidia_tokenize_tiers_pack.sh
```

The produced files match Jouni's LUMI launchers:

```text
multilingual_16k_plus_text_document.{bin,idx}
multilingual_4_16k_text_document.{bin,idx}
multilingual_under4k_text_document.{bin,idx}
```

The generated mix uses the same fixed weights as the current scripts:

```text
0.5 multilingual_16k_plus
0.3 multilingual_4_16k
0.2 multilingual_under4k
```

## LUMI Download

```bash
cd openeuro-longctx-datamix
export HF_TOKEN=hf_...
HF_REPO_ID=birgermoell/oellm-longctx-tokenized \
OUTPUT_DIR=/flash/project_462000963/bmoell/data_tokenized_multilingual_hf \
  lumi/download_hf_tokenized_artifact.sh
```

The download step rewrites:

```text
mix/data_path.args
mix/data_mix.json
mix/data_mix.txt
```

so all prefixes point to the local LUMI download path.

## Jouni Script Compatibility

For `yarn_multilingual.sbatch` and `longrope_multilingual.sbatch`, either:

```bash
MULTILINGUAL_DIR=/flash/project_462000963/bmoell/data_tokenized_multilingual_hf/bin
DATA_PATH="$(cat /flash/project_462000963/bmoell/data_tokenized_multilingual_hf/mix/data_path.args)"
```

or keep the existing hardcoded `DATA_PATH` because the filenames are identical:

```text
${MULTILINGUAL_DIR}/multilingual_16k_plus_text_document
${MULTILINGUAL_DIR}/multilingual_4_16k_text_document
${MULTILINGUAL_DIR}/multilingual_under4k_text_document
```

For smoke tests that only use the long tier:

```bash
DATA_PATH="${MULTILINGUAL_DIR}/multilingual_16k_plus_text_document"
```
