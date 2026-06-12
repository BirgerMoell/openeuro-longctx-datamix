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
