# Language Analysis — Long-Context Document Availability

**Date:** 2026-05-07  
**Pipeline:** OpenEuroLLM long-context data pipeline (`longctx` CLI)  
**Tokenizer:** `openeurollm/tokenizer-256k` (vocab size 262,144)  
**Source dataset:** CulturaX / OPUS (1 sample shard per language)  
**LUMI job:** 18479845 — completed 2026-05-07 11:17 EEST

---

## Summary

We measured token length distributions for all 16 target languages of OpenEuroLLM across 1 sample shard each. The results are actual measurements, not estimates.

**Key finding:** All 16 languages combined (1 shard each) yield **~302,600 documents ≥4K tokens**, **~55,000 documents ≥32K tokens**, and **~8,000 documents ≥128K tokens** from **8.64 billion total tokens**. This is sufficient for solid long-context training up to 32K with multi-language mixing.

---

## Aggregate Table (1 shard per language, measured)

```
 Lang |     Docs |     MTok |   Median |      p90 |    >=4K |    >=8K |   >=32K |   >=128K
------------------------------------------------------------------------------------------
   bg |   38,312 |    465.6 |    2,778 |   22,569 |  15,581 |  10,169 |   2,614 |      572
   cs |   77,038 |    659.0 |    1,815 |   19,403 |  25,729 |  17,265 |   4,041 |      503
   da |   73,860 |    584.3 |    1,787 |   17,062 |  22,791 |  14,383 |   3,469 |      466
   et |   34,429 |    322.8 |    2,559 |   21,867 |  13,232 |   8,800 |   2,105 |      204
   fi |   46,199 |    574.5 |    2,419 |   30,820 |  17,988 |  12,846 |   4,351 |      471
   hr |   43,638 |    625.1 |    3,486 |   34,638 |  20,239 |  15,101 |   4,705 |      592
   hu |   39,341 |    750.1 |    4,045 |   38,194 |  19,562 |  14,453 |   4,482 |    1,050
   lt |   36,153 |    490.3 |    3,060 |   28,659 |  15,753 |  11,104 |   3,070 |      514
   lv |   33,962 |    344.8 |    2,381 |   23,056 |  12,446 |   8,167 |   2,154 |      320
   mt |   16,490 |    131.0 |    2,286 |   16,773 |   5,463 |   3,283 |     655 |      104
   no |   50,467 |    486.1 |    2,255 |   23,060 |  18,807 |  12,934 |   3,449 |      271
   pl |   74,689 |    773.8 |    2,377 |   21,440 |  28,587 |  19,402 |   4,465 |      701
   ro |   54,015 |    830.3 |    3,127 |   28,979 |  23,592 |  15,122 |   4,897 |    1,197
   sk |   65,629 |    668.3 |    2,551 |   23,834 |  26,580 |  18,748 |   4,316 |      456
   sl |   40,473 |    317.4 |    1,555 |   18,799 |  11,564 |   7,782 |   2,282 |      187
   sv |   68,680 |    617.7 |    2,137 |   22,640 |  24,728 |  16,501 |   3,928 |      426
------------------------------------------------------------------------------------------
TOTAL|  783,466 |  8,641.1 |          |          | 302,642 | 205,260 |  54,983 |    8,033
```

*MTok = millions of tokens in the sample shard. Coverage columns = number of docs ≥ that token threshold.*

---

## Combined Dataset Totals (all 16 languages, 1 shard each)

| Metric | Value |
|--------|-------|
| Total documents | 783,466 |
| Total tokens | 8.64 billion |
| Documents ≥ 4K tokens | 302,642 (38.6% of all docs) |
| Documents ≥ 8K tokens | 205,260 (26.2%) |
| Documents ≥ 32K tokens | 54,983 (7.0%) |
| Documents ≥ 128K tokens | 8,033 (1.0%) |

At 1 shard per language this already provides:
- **~302K training sequences** at 4K context
- **~75K training sequences** at 32K context (total tokens ÷ context length)
- **~8K training sequences** at 128K context

These numbers scale linearly with the number of shards downloaded. Most languages have 2–10+ shards available.

---

## Per-Language Detail

### Nordic Group

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K | Max |
|------|------|------|--------|-----|-----|------|-------|-----|
| sv (Swedish) | 68,680 | 617.7 | 2,137 | 22,640 | 24,728 | 3,928 | 426 | 1.3M |
| no (Norwegian) | 50,467 | 486.1 | 2,255 | 23,060 | 18,807 | 3,449 | 271 | 1.3M |
| da (Danish) | 73,860 | 584.3 | 1,787 | 17,062 | 22,791 | 3,469 | 466 | 1.8M |
| fi (Finnish) | 46,199 | 574.5 | 2,419 | 30,820 | 17,988 | 4,351 | 471 | **4.98M** |

Swedish and Danish have the most total documents. Finnish has the heaviest right tail (p90=30K, max=5M) — likely due to large wiki articles and official documents in CulturaX. Finnish's agglutinative morphology also means tokens represent more text per token, leading to denser documents.

### Baltic Group

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K | Max |
|------|------|------|--------|-----|-----|------|-------|-----|
| et (Estonian) | 34,429 | 322.8 | 2,559 | 21,867 | 13,232 | 2,105 | 204 | 638K |
| lv (Latvian) | 33,962 | 344.8 | 2,381 | 23,056 | 12,446 | 2,154 | 320 | 1.9M |
| lt (Lithuanian) | 36,153 | 490.3 | 3,060 | 28,659 | 15,753 | 3,070 | 514 | 4.0M |

Lithuanian is the strongest of the three — notably higher p90 and max. All three Baltic languages have similar shard sizes but Lithuanian has 50% more tokens per shard, suggesting longer average documents.

### Central European Group

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K | Max |
|------|------|------|--------|-----|-----|------|-------|-----|
| pl (Polish) | 74,689 | 773.8 | 2,377 | 21,440 | 28,587 | 4,465 | 701 | **8.8M** |
| cs (Czech) | 77,038 | 659.0 | 1,815 | 19,403 | 25,729 | 4,041 | 503 | 2.0M |
| sk (Slovak) | 65,629 | 668.3 | 2,551 | 23,834 | 26,580 | 4,316 | 456 | 2.9M |
| hu (Hungarian) | 39,341 | 750.1 | **4,045** | **38,194** | 19,562 | 4,482 | **1,050** | 4.1M |

**Hungarian is exceptional**: its median document (4,045 tokens) already exceeds the 4K filter threshold — meaning the majority of documents are long-context eligible. With p90 at 38K tokens, Hungarian has the most documents suitable for 32K+ training relative to its total doc count. It yields 1,050 docs ≥128K — the most of any Central European language.

Polish has the most total docs (74,689) and the largest single document (8.8M tokens — likely a book or large wiki dump).

### Southern European Group

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K | Max |
|------|------|------|--------|-----|-----|------|-------|-----|
| ro (Romanian) | 54,015 | 830.3 | 3,127 | 28,979 | 23,592 | 4,897 | **1,197** | — |
| bg (Bulgarian) | 38,312 | 465.6 | 2,778 | 22,569 | 15,581 | 2,614 | 572 | — |
| hr (Croatian) | 43,638 | 625.1 | 3,486 | 34,638 | 20,239 | 4,705 | 592 | — |
| sl (Slovenian) | 40,473 | 317.4 | 1,555 | 18,799 | 11,564 | 2,282 | 187 | — |

**Romanian is the overall winner for ≥128K documents** (1,197 — most of any language). Its shard is also the largest by token count (830M tokens), and it has a strong p90 at 29K tokens.

**Croatian is a surprise**: despite being a smaller language, it yields 4,705 docs ≥32K and 592 docs ≥128K — better than Norwegian or Finnish on a per-doc basis. Its median (3,486) and p90 (34,638) are both very strong.

**Slovenian is the weakest** of the 16: median only 1,555 tokens, and only 11,564 docs ≥4K from 40K total. Its shard has the fewest tokens (317M).

### Maltese (baseline)

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K | Max |
|------|------|------|--------|-----|-----|------|-------|-----|
| mt (Maltese) | 16,490 | 131.0 | 2,286 | 16,773 | 5,463 | 655 | 104 | 1.6M |

Maltese is the smallest corpus by every metric — smallest shard (131M tokens vs 830M for Romanian), fewest long docs (655 ≥32K vs 4,897 for Romanian). It was the right choice for pipeline validation (fast iteration) but should be mixed with larger languages for actual training.

---

## Long-Context Coverage Assessment

### By context length target

| Context | All 16 langs combined | Suitable languages | Assessment |
|---------|-----------------------|-------------------|------------|
| 4K | 302,642 docs | All 16 | ✅ Excellent — 300K+ sequences |
| 8K | 205,260 docs | All 16 | ✅ Excellent — 25K+ sequences |
| 16K | ~120,000 docs (est.) | All 16 | ✅ Good — 7K+ sequences |
| 32K | 54,983 docs | All 16 | ✅ Good — 1.7K sequences |
| 64K | ~20,000 docs (est.) | hu, ro, hr, fi, lt | ⚠️ Moderate — 300+ sequences |
| 128K | 8,033 docs | hu, ro best | ⚠️ Sparse — ~60 sequences |
| 256K+ | ~1,000–2,000 docs (est.) | pl, fi, lt, hu | ❌ Thin — <20 sequences |

*Sequences estimated as total tokens at threshold ÷ context length, no document packing.*

### Languages ranked by long-context quality (≥32K docs per shard)

1. **Romanian** (ro): 4,897 ≥32K, 1,197 ≥128K
2. **Croatian** (hr): 4,705 ≥32K, 592 ≥128K
3. **Hungarian** (hu): 4,482 ≥32K, 1,050 ≥128K
4. **Finnish** (fi): 4,351 ≥32K, 471 ≥128K
5. **Polish** (pl): 4,465 ≥32K, 701 ≥128K
6. **Slovak** (sk): 4,316 ≥32K, 456 ≥128K
7. **Czech** (cs): 4,041 ≥32K, 503 ≥128K
8. **Swedish** (sv): 3,928 ≥32K, 426 ≥128K

---

## Key Findings

### 1. Romanian and Hungarian dominate the ≥128K tier

Romanian (1,197) and Hungarian (1,050) together account for 28% of all ≥128K documents across all 16 languages. These two languages should be prioritized for very-long-context training (64K–256K).

### 2. All languages are 6–50× better than Maltese for long-context data

Even the weakest language (Slovenian) has 3.5× more ≥32K documents than Maltese. The best (Romanian) has 7.5× more. This confirms the pipeline design — start with one language for validation, then scale to all.

### 3. Median document lengths are shorter than expected

Most languages have a median around 2K–4K tokens — below the 4K filter threshold. This means the filter removes 50–70% of documents in most languages. The long-context data is concentrated in the right tail of each distribution.

### 4. A tiered strategy makes sense

- **≥4K corpus**: all 16 languages, 302K docs — for training up to 16K context
- **≥32K corpus**: select top 8 languages, ~37K docs — for training at 32K context
- **≥128K corpus**: ro + hu + pl + hr, ~3.5K docs — for training at 128K context

### 5. Multiple shards available for most languages

Most languages have 2–10 shards available. Current analysis uses 1 shard each. Downloading all shards would multiply the dataset proportionally:
- Polish: 10+ shards → 280K+ docs ≥4K
- Czech: 10+ shards → 257K+ docs ≥4K
- Swedish: 7 shards → 173K+ docs ≥4K

---

## Shard Availability (from `longctx estimate`)

| Lang | Available shards | Full corpus GB | 1-shard MB |
|------|-----------------|----------------|-----------|
| sv | 7 | 6.6 | 1,049 |
| no | ~5 | ~5.0 | ~1,000 |
| da | ~5 | ~5.0 | ~1,000 |
| fi | ~5 | ~5.0 | ~1,000 |
| et | 2 | 0.8 | 478 |
| lv | 2 | 0.8 | 472 |
| lt | ~3 | ~1.5 | ~500 |
| pl | 10+ | ~15 | ~800 |
| cs | 10+ | ~13 | ~700 |
| sk | ~5 | ~5 | ~700 |
| hu | 10 | 9.9 | 1,151 |
| ro | 7 | 8.5 | 1,368 |
| bg | ~5 | ~5 | ~500 |
| hr | ~3 | ~2 | ~600 |
| sl | 3 | 1.2 | 518 |
| mt | 1 | 0.2 | 170 |

---

## Recommendations

### Immediate priority: Download more shards of top languages

With 1 shard each we have 302K docs ≥4K. To reach 1M+ docs ≥4K:

```bash
# Download 3-5 shards of the top long-context languages
longctx download --languages ro,hu,pl,hr,fi --shards 5
longctx convert
longctx filter-long --min-tokens 4096
longctx tokenize
```

This would yield approximately:
- ro (5 shards): ~118K docs ≥4K, ~24K docs ≥32K, ~6K docs ≥128K
- hu (5 shards): ~98K docs ≥4K, ~22K docs ≥32K, ~5K docs ≥128K
- pl (5 shards): ~143K docs ≥4K, ~22K docs ≥32K, ~3.5K docs ≥128K

### Enable document packing for 4K–16K context training

For the short-context end (4K–16K), most documents don't fill a full context window. Document packing concatenates documents with EOS separators, maximizing GPU utilization:

```bash
# In Megatron training args:
--reset-position-ids     # treat each packed document independently
--reset-attention-mask   # no attention across document boundaries
--eod-mask-loss          # don't compute loss on EOD tokens
```

### Tiered training curriculum

A practical curriculum for long-context continual pre-training:

1. **Phase 1 (4K–16K)**: All 16 languages, all docs ≥4K, with document packing
2. **Phase 2 (16K–32K)**: Top 12 languages, docs ≥16K only
3. **Phase 3 (32K–128K)**: ro, hu, pl, hr, fi, lt, with 5+ shards each
4. **Phase 4 (128K+)**: ro + hu primarily (1,197 + 1,050 docs per shard)

---

## Files

```
/scratch/project_462000963/bmoell/openeuro-longctx-datamix/
├── data/
│   ├── raw/                            # Downloaded parquet shards (1 per language)
│   ├── megatron/                       # Converted JSONL (all docs, per language)
│   └── long/mt.jsonl                   # Maltese filtered corpus (5,463 docs)
└── lang_stats.json                     # Full per-language stats (all 16 languages)
```

The complete statistics are in `lang_stats.json` at the repo root on LUMI.
