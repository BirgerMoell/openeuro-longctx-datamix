---
license: apache-2.0
language:
- mt
tags:
- openeurollm
- megatron-lm
- long-context
- yarn
- longrope
- tokenized-corpus
- finepdfs-edu
pretty_name: OpenEuroLLM Long-Context Megatron Tokenized Smoke Artifact
size_categories:
- n<1K
task_categories:
- text-generation
---

# OpenEuroLLM Long-Context Megatron Tokenized Smoke Artifact

This dataset repo is a small smoke-test artifact for the OpenEuroLLM long-context
training pipeline. It is intended to validate the NVIDIA → Hugging Face → LUMI
handoff before uploading larger multilingual tokenized corpora.

The files are **Megatron-LM indexed dataset artifacts**, not ordinary Hugging
Face `datasets` rows. Do not stream this with `load_dataset()` for training.
Download the files to local disk and pass the local prefixes to Megatron
`--data-path`.

## Contents

```text
bin/
  multilingual_16k_plus_text_document.bin
  multilingual_16k_plus_text_document.idx
  multilingual_4_16k_text_document.bin
  multilingual_4_16k_text_document.idx
  multilingual_under4k_text_document.bin
  multilingual_under4k_text_document.idx
  tier_summary.json
mix/
  data_mix.json
  data_mix.txt
  data_path.args
manifests/
  build_info.json
  checksums.json
  checksums.sha256
```

The three tier prefixes are compatible with Jouni Luoma's OpenEuroLLM Megatron
context-extension launchers:

```text
multilingual_16k_plus_text_document
multilingual_4_16k_text_document
multilingual_under4k_text_document
```

The fixed training weights are:

```text
0.5 multilingual_16k_plus
0.3 multilingual_4_16k
0.2 multilingual_under4k
```

## Source And Tokenizer

This smoke artifact was built from a bounded Maltese (`mt`) sample from
FinePDFs-Edu, converted with `openeuro-longctx-datamix`, then tokenized with
the OpenEuroLLM/Megatron tokenizer:

```text
tokenizer_model: openeurollm/tokenizer-256k
tokenizer_type: HuggingFaceTokenizer
vocab_size: 262144
```

See `manifests/build_info.json` for the exact local build metadata and Megatron
commit recorded at packaging time.

## Download For Megatron Training

Use the `longctx artifacts download` command from
[`BirgerMoell/openeuro-longctx-datamix`](https://github.com/BirgerMoell/openeuro-longctx-datamix).
It downloads the files and rewrites `mix/data_path.args` to point at your local
download directory.

```bash
git clone https://github.com/BirgerMoell/openeuro-longctx-datamix
cd openeuro-longctx-datamix
python -m venv .venv
source .venv/bin/activate
pip install -e .

longctx artifacts download \
  --repo-id birgermoell/oellm-longctx-tokenized-smoke \
  --output-dir /flash/project_462000963/bmoell/data_tokenized_smoke
```

Then use:

```bash
MULTILINGUAL_DIR=/flash/project_462000963/bmoell/data_tokenized_smoke/bin
DATA_PATH="$(cat /flash/project_462000963/bmoell/data_tokenized_smoke/mix/data_path.args)"
```

For Jouni-compatible smoke scripts that only use the long tier:

```bash
DATA_PATH="${MULTILINGUAL_DIR}/multilingual_16k_plus_text_document"
```

For full three-tier YaRN/LongRoPE training:

```bash
DATA_PATH="$(cat /flash/project_462000963/bmoell/data_tokenized_smoke/mix/data_path.args)"
```

## Verify Integrity

After download:

```bash
cd /flash/project_462000963/bmoell/data_tokenized_smoke
sha256sum -c manifests/checksums.sha256
```

All listed files should report `OK`.

## NVIDIA Build Command

The smoke artifact can be rebuilt on the NVIDIA GPU box with:

```bash
cd /home/ubuntu/birger/openeuro-longctx-datamix
source /home/ubuntu/birger/megatron_env/bin/activate

export PYTHON=/home/ubuntu/birger/megatron_env/bin/python3.10
export MEGATRON_DIR=/home/ubuntu/birger/NVIDIA-Megatron-LM-context-extension
export DATA_DIR=/home/ubuntu/birger/openeuro-longctx-datamix/data/smoke/megatron
export TOKENIZER=openeurollm/tokenizer-256k
export VOCAB_SIZE=262144
export LANGUAGES=mt
export MAX_DOCS_PER_LANG=100

scripts/nvidia_tokenize_tiers_pack.sh
```

For a full artifact, remove `MAX_DOCS_PER_LANG`, point `DATA_DIR` at the full
`data/megatron`, and upload the resulting artifact folder with:

```bash
longctx artifacts upload \
  --folder data/hf_artifacts/oellm-longctx-tokenized \
  --repo-id birgermoell/oellm-longctx-tokenized \
  --private
```

## Intended Use

This repo is for validating:

- Megatron `.bin/.idx` artifact transport through Hugging Face Hub
- local `DATA_PATH` rewriting on LUMI
- compatibility with OpenEuroLLM YaRN and LongRoPE launch scripts
- checksum-based artifact integrity checks

It is too small for model quality experiments.
