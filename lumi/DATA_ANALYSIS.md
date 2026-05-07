# Data Analysis — OpenEuroLLM Long-Context Pipeline

**Date:** 2026-05-07  
**Language analyzed:** Maltese (mlt_Latn) — 1 shard (170 MB raw)  
**Tokenizer:** `openeurollm/tokenizer-256k` (vocab size 262,144)  
**Filter applied:** `longctx filter-long --min-tokens 4096`

---

## Pipeline Summary

| Stage | Docs | Tokens | Notes |
|-------|------|--------|-------|
| Raw download | — | — | 1 shard, 170 MB parquet |
| After `convert` | 16,490 | 130,966,164 | All text extracted |
| After `filter-long` | 5,463 | 113,726,214 | min 4096 tokens |
| Retention | 33% docs | 87% tokens | Filter keeps most tokens |

The filter removes 67% of documents but only 13% of tokens — most discarded documents are short snippets, not long-form text.

---

## Token Length Distribution (filtered corpus)

| Statistic | Tokens |
|-----------|--------|
| Min | 4,098 |
| p25 | 6,065 |
| Median | 10,036 |
| Mean | 20,817 |
| p75 | 19,026 |
| p90 | 37,098 |
| p95 | 61,191 |
| p99 | 200,208 |
| Max | 1,596,102 |

**Key observation:** The distribution has a very heavy right tail. The median document is ~10K tokens (fits in a 16K window), but the p99 is 200K and the maximum is **1.6M tokens** — likely a book or wiki article. The mean (20.8K) is far above the median (10K), confirming strong skew.

**Char/token ratio:** 2.33 (typical for Latin-script Maltese; English is ~4).

---

## Context Window Coverage

How many documents are long enough to fill various context windows natively (without padding):

| Context length | Docs | % of corpus | Tokens | Sequences |
|---------------|------|-------------|--------|-----------|
| 4,096 | 5,463 | 100.0% | 113,726,214 | ~27,765 |
| 8,192 | 3,283 | 60.1% | 101,164,430 | ~13,882 |
| 16,384 | 1,688 | 30.9% | 82,893,203 | ~6,942 |
| 32,768 | 655 | 12.0% | 59,809,352 | ~3,470 |
| 65,536 | 245 | 4.5% | 41,412,435 | ~1,735 |
| 131,072 | 104 | 1.9% | 28,784,078 | ~867 |
| 262,144 | 34 | 0.6% | 16,017,498 | ~433 |

*Sequences = total tokens in corpus / context length (approximate, assumes no cross-document packing)*

---

## Pre-filter Length Distribution

Before filtering, documents fell into these buckets:

| Length range | Docs | % |
|-------------|------|---|
| 0 – 512 tokens | 1,915 | 11.6% |
| 512 – 4,096 tokens | 9,112 | 55.3% |
| 4,096 – 16,384 tokens | 3,775 | 22.9% |
| 16,384 – 65,536 tokens | 1,443 | 8.8% |
| 65,536+ tokens | 245 | 1.5% |

More than two-thirds of documents (67%) are under 4K tokens — short web snippets, social media, or boilerplate text that was filtered out.

---

## Assessment: Suitability for Long-Context Training

### Suitable for context up to 16K ✅

60% of documents (3,283) natively exceed 8K tokens. With standard sequence packing (multiple documents packed into one training window), 8K and 16K context training is well-covered at ~14K–7K sequences respectively.

### Sparse at 32K–128K ⚠️

Only 12% of documents (655) exceed 32K tokens. At 32K context this yields ~3,470 training sequences — enough to demonstrate the pipeline but not enough for serious long-context fine-tuning. You would need more languages or a lower filter threshold with packing.

### Very thin above 128K ❌

Only 104 documents (1.9%) exceed 128K tokens. This is too sparse for training at 128K+ context windows from a single language. The 867 estimated sequences at 131K context is marginal.

### Outlier document

The longest document is **1,596,102 tokens** (~3.7M characters). This is ~80× longer than the median. It is likely a complete book, a large Wikipedia article collection, or a crawl artifact. It should be inspected to confirm it is genuine long-form content and not a data quality issue.

---

## Recommendations

### For immediate real-data training (this week)

1. **Tokenize the full 5,463-doc corpus** — the current tokenized file in `tmp_train/` only covers 50 documents. Run `longctx tokenize` on all of `data/long/mt.jsonl`.
2. **Use sequence packing** — pack multiple documents into each context window rather than padding. This maximizes GPU utilization, especially at 4K–16K context.
3. **Start at 4K–8K context** — where coverage is solid (60–100% of docs). This is a good baseline for long-context capability.

### For improving long-context data coverage

4. **Add more languages** — Maltese is one language. The pipeline supports `--languages` with multiple codes. Adding several languages from the OPUS/CulturaX data will multiply the training data.
5. **Inspect the 1.6M-token outlier** — confirm it is real long-form content before training on it.
6. **Consider a tiered filter** — for 32K+ context training, select from the 655 documents that are ≥32K tokens and treat them as a separate high-priority data source.

### For long-context evaluation

7. The 104 documents ≥128K tokens are valuable as **evaluation material** for testing whether the model can handle very long inputs, even if there are too few for training at that length.

---

## Files on LUMI

```
/scratch/project_462000963/bmoell/openeuro-longctx-datamix/
├── data/
│   ├── raw/mt/000_00000.parquet          # raw download (170 MB)
│   ├── megatron/mt.jsonl                 # all 16,490 converted docs
│   └── long/
│       ├── filter_summary.json           # filter statistics
│       └── mt.jsonl                      # 5,463 filtered docs (265 MB)
└── tmp_train/
    ├── mt_train.jsonl                    # first 50 docs (for testing)
    └── mt_train_text_document.{bin,idx}  # tokenized (50-doc subset, 1.9 MB)
```

To tokenize the full corpus for real training, run:

```bash
singularity exec "$SIF" python3 $MEGATRON_DIR/tools/preprocess_data.py \
  --input data/long/mt.jsonl \
  --output-prefix data/tokenized/mt_full \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model openeurollm/tokenizer-256k \
  --workers 4 \
  --append-eod
```
