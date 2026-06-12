# Long-Context Dataset Expansion Plan

Current artifact:
`birgermoell/oellm-longctx-tokenized-streamed-all-v2`

The existing HF artifact is the right base: it is already tokenized as Megatron
indexed data from `HuggingFaceFW/finepdfs-edu`, and the full run report shows a
large long-document tail. The important missing piece is not "more of the same"
first. It is better control over which long lengths the trainer samples.

## What We Have

The full FinePDFs-Edu streamed run has useful material well beyond 16K:

| Bin | Use |
| --- | --- |
| <=16K | replay, local coherence, short-context retention |
| 16-32K | 32K adaptation support |
| 32-64K | 64K adaptation |
| 64-128K | 128K adaptation |
| >128K | 128K+ and constructed ultra-long source material |

The current coarse mix collapses everything above 16K into `16k_plus`, which is
fine for 32K but too blunt for 64K/128K/256K planning.

## Datasets To Make Next

Machine-readable recipes live in
`configs/longctx_dataset_recipes.json`.

### 1. FinePDFs lc128k Tokenized Tiers

Priority: immediate.

HF target:
`birgermoell/oellm-longctx-tokenized-finepdfs-lc128k-v1`

This re-streams the same source data but emits six length tiers:

```text
128k_plus   weight 0.25
64_128k     weight 0.25
32_64k      weight 0.20
16_32k      weight 0.15
4_16k       weight 0.10
under4k     weight 0.05
```

Small-GPU smoke:

```bash
cd openeuro-longctx-datamix

HF_REPO_ID=birgermoell/oellm-longctx-tokenized-finepdfs-lc128k-v1 \
RUN_ID=lc128k_smoke_20260612 \
TIER_PRESET=lc128k \
LANGUAGES=mt,cy,sv,en \
SHARDS=1 \
MAX_DOCS=2000 \
CHUNK_DOCS=512 \
CHUNK_DOCS_BY_TIER=128k_plus=8,64_128k=16,32_64k=32,16_32k=64,4_16k=256,under4k=512 \
UPLOAD_BATCH_CHUNKS=4 \
PRIVATE=1 \
scripts/nvidia_stream_upload_all_languages.sh
```

Full run after smoke:

```bash
HF_REPO_ID=birgermoell/oellm-longctx-tokenized-finepdfs-lc128k-v1 \
RUN_ID=lc128k_full_20260612 \
TIER_PRESET=lc128k \
CHUNK_DOCS=1024 \
CHUNK_DOCS_BY_TIER=128k_plus=32,64_128k=64,32_64k=128,16_32k=256,4_16k=1024,under4k=4096 \
UPLOAD_BATCH_CHUNKS=8 \
PRIVATE=1 \
scripts/nvidia_stream_upload_all_languages.sh
```

Acceptance checks:

- `mix/data_mix.json` contains all six tiers.
- `128k_plus` is not English-only.
- At least 30 FinePDFs languages contribute to `32_64k` or longer.
- A Leonardo 32K smoke can read `mix/data_path.args` without rebuilding data.

### 2. Retrieval / Anti-Cropping Curriculum

Priority: next after lc128k smoke.

HF target:
`birgermoell/oellm-longctx-retrieval-curriculum-v1`

Use `scripts/make_anti_cropping_traces.py` at 32K, 64K, and 128K. Keep this
small: about 2-5 percent of continued-pretraining tokens. It should teach the
model that early and middle context matter, not replace natural language
modeling.

Example raw JSONL builds:

```bash
python scripts/make_anti_cropping_traces.py \
  --model /home/ubuntu/birger/Megatron-Bridge-utils/tokenizers/openeurollm/tokenizer-256k \
  --output data/retrieval_curriculum/anti_cropping_32k.jsonl \
  --examples 2048 \
  --target-tokens 32768 \
  --fact-position cycle

python scripts/make_anti_cropping_traces.py \
  --model /home/ubuntu/birger/Megatron-Bridge-utils/tokenizers/openeurollm/tokenizer-256k \
  --output data/retrieval_curriculum/anti_cropping_64k.jsonl \
  --examples 1024 \
  --target-tokens 65536 \
  --fact-position cycle

python scripts/make_anti_cropping_traces.py \
  --model /home/ubuntu/birger/Megatron-Bridge-utils/tokenizers/openeurollm/tokenizer-256k \
  --output data/retrieval_curriculum/anti_cropping_128k.jsonl \
  --examples 512 \
  --target-tokens 131072 \
  --fact-position cycle
```

Required variants:

- answer evidence in first, middle, and final thirds
- single-key and multi-key retrieval
- aggregation/counting over repeated records
- no-answer negatives

Do not reuse eval keys or values from OneRuler/NIAH.

### 3. Ultra-Long Constructed Sequences

Priority: needed before 256K/512K/2M training.

HF target:
`birgermoell/oellm-longctx-ultra-concat-v1`

Natural 2M-token documents are rare. For 256K+ stages, build examples by
stitching coherent bundles:

- FinePDFs `128k_plus` documents
- EUR-Lex and national legal bundles
- Wikisource books and Wikipedia topic clusters
- open code repositories
- arXiv/open textbooks where licensing is acceptable

Rules:

- concatenate by language and broad topic, not random documents
- insert explicit document-boundary markers
- preserve natural order for books, legal bundles, and repositories
- keep source manifests so eval holdouts can be denylisted

### 4. Missing OpenEuroLLM Languages

Priority: quality/coverage, not first blocker.

HF target:
`birgermoell/oellm-longctx-missing-languages-v1`

FinePDFs-Edu misses `ga`, `sq`, and `lb`. Use HPLT first, and CulturaX only if
access and licensing are acceptable. These languages probably will not have much
native 128K+ material, so treat them as a high-quality low-resource supplement
rather than forcing synthetic ultra-long volume.

### 5. Evaluation Holdouts

Priority: create before large synthetic generation.

HF target:
`birgermoell/oellm-longctx-eval-holdouts-v1`

Hold out source IDs and synthetic key/value namespaces for:

- 32K, 64K, 128K, 256K, 512K, 1M, and 2M lengths
- depth buckets at least every 10 percent; every 5 percent near the first 20 percent
- multilingual single-key, multi-key, multi-value, aggregation, and no-answer tasks

## Training Mix Recommendation

For the next 64K/128K experiments:

```text
70-80% FinePDFs lc128k natural long-document tiers
10-20% short/medium replay from <=16K and 16-32K
2-5% retrieval and anti-cropping curriculum
0-10% code/legal/books if already curated
```

For 256K+:

```text
50-60% coherent constructed ultra-long bundles
20-30% natural 64K-128K+ FinePDFs
5-10% retrieval/aggregation curriculum
5-10% short/medium replay
```

The key is to avoid a pure "longest only" mix. Long-context training still needs
short-context retention and natural next-token statistics, while explicit
retrieval data prevents the model from learning that only the last chunk matters.

## Minimal 2M Experiment Data

For the <=2000 GPU-hour 2M-context experiment, do not try to create a huge
2M-token pretraining corpus. Make a small, surgical set:

| Dataset | Train/eval | Lengths | Purpose |
| --- | --- | --- | --- |
| FinePDFs lc128k tiers | train | 32K-128K+ | natural long-document CLM |
| constructed 256K bundles | train | 256K | natural-ish long CLM from related documents |
| synthetic 2M position curriculum | train | 32K, 64K, 128K, 256K | retrieval/aggregation examples with PoSE metadata |
| short recovery replay | train | 8K | preserve short-context behavior after positional surgery |
| 2M holdout eval | eval only | 512K, 1M, 2M | tiny exact-prefill NIAH/aggregation success check |

The synthetic train/eval pieces are generated by:
`scripts/make_2m_curriculum_data.py`.

Training curriculum, physically <=256K but tagged for a 2M position span:

```bash
python scripts/make_2m_curriculum_data.py \
  --output-dir /leonardo_scratch/large/userexternal/pmoell00/data/oellm_2m_curriculum/raw \
  --prefix oellm_2m_pose \
  --split train \
  --lengths 32768,65536,131072,262144 \
  --examples-per-length 256 \
  --languages en,sv,de,fr \
  --tasks single_key,multi_key,aggregation,no_answer \
  --depths 0.05,0.25,0.5,0.75,0.9 \
  --pose-max-context 2097152 \
  --model /home/ubuntu/birger/Megatron-Bridge-utils/tokenizers/openeurollm/tokenizer-256k
```

Tiny held-out exact-length eval set:

```bash
python scripts/make_2m_curriculum_data.py \
  --output-dir /leonardo_scratch/large/userexternal/pmoell00/data/oellm_2m_curriculum/eval \
  --prefix oellm_2m_holdout \
  --split eval \
  --lengths 524288,1048576,2097152 \
  --examples-per-length 12 \
  --languages en,sv \
  --tasks single_key,multi_key,aggregation,no_answer \
  --depths 0.05,0.5,0.9 \
  --pose-max-context 2097152 \
  --model /home/ubuntu/birger/Megatron-Bridge-utils/tokenizers/openeurollm/tokenizer-256k
```

For a fast local smoke without a tokenizer, omit `--model` and use tiny lengths:

```bash
python scripts/make_2m_curriculum_data.py \
  --output-dir /tmp/oellm_2m_curriculum_smoke \
  --prefix smoke \
  --split train \
  --lengths 512,1024 \
  --examples-per-length 4 \
  --languages en,sv \
  --tasks single_key,aggregation \
  --depths 0.05,0.9
```

Important: the `pose` metadata does not affect Megatron tokenization by itself.
The 256K training run must also sample skipped position IDs over `0..2097152`.
The raw text is the supervision; the metadata is the contract for the trainer and
for held-out scoring.
