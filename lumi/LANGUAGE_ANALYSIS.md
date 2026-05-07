# Language Analysis — Long-Context Document Availability

**Date:** 2026-05-07  
**Pipeline:** OpenEuroLLM long-context data pipeline (`longctx` CLI)  
**Tokenizer:** `openeurollm/tokenizer-256k` (vocab size 262,144)  
**Filter:** `longctx filter-long --min-tokens 4096`  
**Source dataset:** CulturaX / OPUS (1 sample shard per language)

---

## Summary

We are building long-context training data for OpenEuroLLM — a multilingual European language model. The pipeline downloads documents from CulturaX/OPUS, filters them to keep only those with ≥4,096 tokens, and tokenizes them for Megatron-LM training.

This report covers 16 target languages. **Full measurements are available for Maltese (mt)**, which served as the initial validation language. The analysis job for all remaining languages is running (LUMI job 18479845) and results will be added when complete.

---

## Target Languages

| Group | Languages |
|-------|-----------|
| Nordic | Swedish (sv), Norwegian (no), Danish (da), Finnish (fi) |
| Baltic | Estonian (et), Latvian (lv), Lithuanian (lt) |
| Central European | Polish (pl), Czech (cs), Slovak (sk), Hungarian (hu) |
| Southern European | Romanian (ro), Bulgarian (bg), Croatian (hr), Slovenian (sl) |
| Mediterranean | Maltese (mt) |

---

## Maltese (mt) — Full Analysis ✅

**Source:** CulturaX, 1 shard (170 MB parquet)  
**Status:** Fully measured and validated

### Pipeline Throughput

| Stage | Docs | Tokens | Notes |
|-------|------|--------|-------|
| Raw download | — | — | 1 shard, 170 MB parquet |
| After `convert` | 16,490 | 130,966,164 | All text extracted |
| After `filter-long` | 5,463 | 113,726,214 | min 4,096 tokens |
| Retention | 33% docs | 87% tokens | Most tokens are in long docs |

### Token Length Distribution (filtered corpus)

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

The distribution has a heavy right tail — the maximum document (1.6M tokens) is likely a book or large crawled long-form text. Mean (20.8K) is 2× the median (10K), confirming significant skew.

**Chars/token ratio:** 2.33 (Maltese Latin script is denser than English's ~4.0)

### Pre-Filter Length Distribution

| Length range | Docs | % |
|-------------|------|---|
| 0 – 512 tokens | 1,915 | 11.6% |
| 512 – 4,096 tokens | 9,112 | 55.3% |
| 4,096 – 16,384 tokens | 3,775 | 22.9% |
| 16,384 – 65,536 tokens | 1,443 | 8.8% |
| 65,536+ tokens | 245 | 1.5% |

Most documents (67%) are short web snippets under 4K tokens — filtered out as unsuitable for long-context training.

### Context Window Coverage (filtered corpus)

| Context length | Docs | % of corpus | Tokens | Est. sequences |
|---------------|------|-------------|--------|----------------|
| 4,096 | 5,463 | 100.0% | 113,726,214 | ~27,765 |
| 8,192 | 3,283 | 60.1% | 101,164,430 | ~13,882 |
| 16,384 | 1,688 | 30.9% | 82,893,203 | ~6,942 |
| 32,768 | 655 | 12.0% | 59,809,352 | ~3,470 |
| 65,536 | 245 | 4.5% | 41,412,435 | ~1,735 |
| 131,072 | 104 | 1.9% | 28,784,078 | ~867 |
| 262,144 | 34 | 0.6% | 16,017,498 | ~433 |

*Est. sequences = total tokens at that threshold / context length (no packing assumed)*

### Assessment

| Context target | Status | Sequences available |
|----------------|--------|---------------------|
| 4K–8K | ✅ Solid | ~14K–28K |
| 16K | ⚠️ Moderate | ~7K |
| 32K | ⚠️ Sparse | ~3.5K |
| 64K+ | ❌ Thin | <2K |
| 128K+ | ❌ Very thin | ~867 |

Maltese alone is sufficient to validate the pipeline up to 16K context, but insufficient for serious long-context fine-tuning at 32K+. The goal is to supplement with higher-resource languages.

---

## Other Languages — Analysis in Progress

*LUMI job 18479845 is running the download → convert → filter → analyze pipeline for all 15 remaining languages. Results will be added to `lang_stats.json` and this document.*

### Expected corpus characteristics (1 sample shard)

The estimates below are based on known CulturaX shard sizes and corpus composition for each language. The key driver of long-document availability is the presence of Wikipedia articles, news archives, and academic/legal text in the CulturaX crawl.

| Lang | CulturaX size | Expected docs/shard | Expected long docs (≥4K) | Expected median length | Notes |
|------|--------------|---------------------|--------------------------|------------------------|-------|
| sv | Very large (~50GB) | ~100K–500K | Many thousands | 8K–20K | Rich Wikipedia (~2.5M articles), news |
| no | Large (~20GB) | ~50K–200K | Many thousands | 8K–20K | Large Wikipedia, Bokmål+Nynorsk |
| da | Large (~15GB) | ~40K–150K | Thousands | 6K–15K | Good news corpus |
| fi | Large (~20GB) | ~50K–200K | Many thousands | 6K–15K | Agglutinative, denser tokens |
| et | Medium (~5GB) | ~20K–80K | Thousands | 5K–12K | Smaller but similar structure to Finnish |
| lv | Medium (~5GB) | ~15K–60K | Thousands | 5K–12K | Similar to Estonian |
| lt | Medium (~6GB) | ~20K–80K | Thousands | 5K–12K | Slightly larger than Latvian |
| pl | Very large (~40GB) | ~100K–400K | Many thousands | 8K–20K | Large Wikipedia, rich web corpus |
| cs | Large (~20GB) | ~60K–200K | Many thousands | 8K–18K | Large Wikipedia |
| sk | Medium (~8GB) | ~25K–100K | Thousands | 6K–15K | Smaller than Czech but similar |
| hu | Large (~15GB) | ~40K–150K | Many thousands | 7K–18K | Agglutinative, denser per token |
| ro | Large (~15GB) | ~40K–150K | Thousands | 7K–16K | Good news and web corpus |
| bg | Large (~12GB) | ~30K–120K | Thousands | 6K–15K | Cyrillic script |
| hr | Medium (~6GB) | ~20K–80K | Thousands | 5K–12K | Smaller web corpus |
| sl | Medium (~4GB) | ~15K–60K | Thousands | 5K–10K | Smallest of the Slavic targets |

**Key prediction:** Nordic languages (sv, no, da, fi) and Central European languages with large Wikipedias (pl, cs, hu) will provide 10–100× more long documents per shard than Maltese. This will push the ≥32K document count from ~650 (Maltese-only) to tens of thousands across all languages combined.

---

## Combined Dataset Projections

Assuming one shard per language and conservative estimates:

| Context | Maltese only | All 16 languages (est.) | Multiple shards per lang (est.) |
|---------|-------------|--------------------------|--------------------------------|
| ≥8K docs | 3,283 | 200K–500K | 1M–5M |
| ≥32K docs | 655 | 30K–100K | 200K–1M |
| ≥128K docs | 104 | 5K–20K | 30K–200K |

A dataset spanning all 16 languages at 1–2 shards each should provide:
- **Solid coverage** at 8K–16K context (hundreds of thousands of sequences)
- **Good coverage** at 32K context (tens of thousands of sequences)
- **Workable coverage** at 128K context (thousands of sequences)

This is the target volume for OpenEuroLLM long-context continual pre-training.

---

## Character-to-Token Ratio by Language

The tokenizer (`openeurollm/tokenizer-256k`) encodes text differently per language script. This affects how many tokens a given document produces.

| Script | Example languages | Est. chars/token |
|--------|------------------|------------------|
| Latin (Maltese) | mt | 2.33 (measured) |
| Latin (Nordic/Germanic) | sv, no, da, de | ~3.5–4.5 |
| Latin (Finno-Ugric) | fi, et, hu | ~3.0–4.0 (agglutinative words = longer) |
| Latin (Slavic) | pl, cs, sk, hr, sl, ro | ~3.0–4.0 |
| Cyrillic | bg | ~2.0–3.0 (Cyrillic tokens often shorter) |
| Latin (Baltic) | lv, lt | ~3.0–4.0 |

Maltese (2.33 chars/token) is notably dense — likely because Maltese has many loanwords from Arabic/Sicilian that are represented less efficiently. Nordic languages are expected to have 4–5 chars/token, meaning a 16K-token document in Swedish represents ~64K–80K characters (~12K–16K words). This is comparable to a short book chapter.

---

## Recommendations Based on Current Analysis

### Immediate (this week)

1. **Run the full 5,463-doc Maltese corpus through training** — job 18479525 is running. Validates real-data training end-to-end.
2. **Analyze all 16 languages** — job 18479845 is running. Will produce actual measurements within ~2 hours.
3. **Add at least Swedish + Finnish to the training mix** — these alone will multiply the available long documents by 10–100×.

### Near-term (next 2 weeks)

4. **Download 2–5 shards per language** for high-resource languages (sv, no, fi, pl, cs). Linear scaling — 5 shards = 5× the documents.
5. **Enable document packing** in Megatron training (`--reset-position-ids --reset-attention-mask --eod-mask-loss`). This turns 5K+7K document pairs into one 12K training sequence, dramatically improving GPU utilization at shorter context lengths.
6. **Inspect the Maltese 1.6M-token outlier** — if it is genuine long-form text (book or encyclopedia), it is extremely valuable. If it is a crawl artifact, filter it out.

### Longer-term

7. **Wikipedia extraction** — Wikipedia articles (4K–64K tokens, coherent, factual) are ideal long-context training data. Swedish Wikipedia alone has 2.5M articles. Adding Wikipedia would roughly double the ≥16K document count per language.
8. **EUR-Lex legal documents** — EU legislation is available in all 24 official EU languages, 32K–256K tokens per document, and already in CulturaX for many languages.

---

## Files

```
/scratch/project_462000963/bmoell/openeuro-longctx-datamix/
├── data/
│   ├── raw/mt/000_00000.parquet        # Maltese raw (170 MB)
│   ├── megatron/mt.jsonl               # Maltese converted (16,490 docs)
│   └── long/mt.jsonl                   # Maltese filtered (5,463 docs, 265 MB)
├── lang_stats.json                     # Per-language stats (updated by job 18479845)
└── lumi/
    ├── DATA_ANALYSIS.md                # Maltese deep-dive analysis
    ├── LANGUAGE_ANALYSIS.md            # This file — all-language report
    └── LONGDOC_STRATEGY.md             # Strategy for sourcing more long documents
```

Running jobs (LUMI, as of 2026-05-07):
- `18479860` — `train_real.sbatch`: Tokenize full 5,463-doc Maltese corpus + 20 training iterations
- `18479845` — `lang_analysis.sbatch`: Download 1 shard × 16 languages, analyze token length distributions
