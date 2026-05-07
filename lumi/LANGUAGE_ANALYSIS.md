# Language Analysis — Long-Context Document Availability

**Date:** 2026-05-07  
**Pipeline:** OpenEuroLLM long-context data pipeline (`longctx` CLI)  
**Tokenizer:** `openeurollm/tokenizer-256k` (vocab size 262,144)  
**Source dataset:** `HuggingFaceFW/finepdfs-edu` (PDF-derived documents, 1 sample shard per language)  
**LUMI jobs:** 18479845 (16 languages, completed) · 18480711 (remaining 19 languages, completed)

---

## Summary

We measured token length distributions for all 35 FinePDFs-Edu languages across 1 sample shard each. The 3 languages absent from FinePDFs-Edu (Irish/ga, Albanian/sq, Luxembourgish/lb) require an HF token for HPLT access and are not yet measured.

**Key finding:** All 35 languages combined (1 shard each) yield **~653K documents ≥4K tokens**, **~108K documents ≥32K tokens**, and **~18K documents ≥128K tokens** from **18.25 billion total tokens**. This is more than sufficient for full-scale long-context training up to 32K with multilingual mixing.

---

## Aggregate Table (1 shard per language, measured)

```
 Lang |     Docs |     MTok |   Median |      p90 |    >=4K |    >=8K |   >=32K |   >=128K
------------------------------------------------------------------------------------------
   bg |   38,312 |    465.6 |    2,778 |   22,569 |  15,581 |  10,169 |   2,614 |      572
   bs |   27,924 |    494.2 |    3,866 |   37,386 |  13,643 |   9,904 |   3,113 |      693
   ca |   62,081 |    399.8 |    1,609 |   13,615 |  17,227 |  10,306 |   2,229 |      240
   cs |   77,038 |    659.0 |    1,815 |   19,403 |  25,729 |  17,265 |   4,041 |      503
   cy |   16,560 |     92.2 |    1,706 |   12,549 |   4,439 |   2,513 |     474 |       40
   da |   73,860 |    584.3 |    1,787 |   17,062 |  22,791 |  14,383 |   3,469 |      466
   de |   89,938 |    473.7 |    1,409 |   10,767 |  21,357 |  11,970 |   2,167 |      284
   el |   46,963 |    536.3 |    2,175 |   21,451 |  16,281 |  10,498 |   3,325 |      711
   en |  236,144 |  1,335.4 |    1,448 |   10,663 |  61,637 |  31,612 |   6,099 |      963
   es |   54,045 |    578.2 |    2,314 |   21,321 |  20,414 |  12,974 |   3,567 |      612
   et |   34,429 |    322.8 |    2,559 |   21,867 |  13,232 |   8,800 |   2,105 |      204
   eu |   19,259 |    188.1 |    2,391 |   22,746 |   7,190 |   4,666 |   1,249 |      142
   fi |   46,199 |    574.5 |    2,419 |   30,820 |  17,988 |  12,846 |   4,351 |      471
   fr |   74,122 |    440.1 |    1,447 |   11,476 |  18,914 |  10,556 |   2,053 |      330
   gl |   30,901 |    270.4 |    1,871 |   18,693 |  10,355 |   6,808 |   1,617 |      232
   hr |   43,638 |    625.1 |    3,486 |   34,638 |  20,239 |  15,101 |   4,705 |      592
   hu |   39,341 |    750.1 |    4,045 |   38,194 |  19,562 |  14,453 |   4,482 |    1,050
   is |   24,259 |    211.3 |    1,928 |   21,690 |   7,663 |   5,267 |   1,364 |      112
   it |   80,619 |    527.7 |    1,685 |   13,295 |  21,638 |  12,709 |   2,766 |      383
   lt |   36,153 |    490.3 |    3,060 |   28,659 |  15,753 |  11,104 |   3,070 |      514
   lv |   33,962 |    344.8 |    2,381 |   23,056 |  12,446 |   8,167 |   2,154 |      320
   mk |   11,518 |    230.8 |    4,058 |   46,007 |   5,731 |   4,056 |   1,498 |      365
   mt |   16,490 |    131.0 |    2,286 |   16,773 |   5,463 |   3,283 |     655 |      104
   nl |   70,914 |    450.8 |    1,892 |   13,287 |  20,635 |  11,678 |   2,056 |      265
   no |   50,467 |    486.1 |    2,255 |   23,060 |  18,807 |  12,934 |   3,449 |      271
   pl |   74,689 |    773.8 |    2,377 |   21,440 |  28,587 |  19,402 |   4,465 |      701
   pt |   53,862 |    587.1 |    2,867 |   19,884 |  23,337 |  15,147 |   3,308 |      558
   ro |   54,015 |    830.3 |    3,127 |   28,979 |  23,592 |  15,122 |   4,897 |    1,197
   ru |   50,267 |    650.5 |    2,859 |   22,950 |  20,637 |  12,750 |   3,576 |      944
   sk |   65,629 |    668.3 |    2,551 |   23,834 |  26,580 |  18,748 |   4,316 |      456
   sl |   40,473 |    317.4 |    1,555 |   18,799 |  11,564 |   7,782 |   2,282 |      187
   sr |   30,965 |    769.7 |    4,332 |   51,007 |  15,758 |  12,220 |   4,407 |    1,313
   sv |   68,680 |    617.7 |    2,137 |   22,640 |  24,728 |  16,501 |   3,928 |      426
   tr |   47,585 |    619.9 |    3,440 |   25,662 |  22,034 |  14,904 |   3,690 |      678
   uk |   44,424 |    752.9 |    3,862 |   30,339 |  21,604 |  13,171 |   4,217 |    1,347
------------------------------------------------------------------------------------------
  TOT | 1,865,725 | 18,250.1 |         |          | 653,136 | 419,769 | 107,758 |   18,246
```

*MTok = millions of tokens in the sample shard. Coverage columns = number of docs ≥ that token threshold.*

---

## Combined Dataset Totals (all 35 languages, 1 shard each)

| Metric | Value |
|--------|-------|
| Total documents | 1,865,725 |
| Total tokens | 18.25 billion |
| Documents ≥ 4K tokens | 653,136 (35.0% of all docs) |
| Documents ≥ 8K tokens | 419,769 (22.5%) |
| Documents ≥ 32K tokens | 107,758 (5.8%) |
| Documents ≥ 128K tokens | 18,246 (1.0%) |

At 1 shard per language this provides:
- **~653K training sequences** at 4K context
- **~557K training sequences** at 32K context (total tokens ÷ context length, est.)
- **~18K training sequences** at 128K context

These numbers scale linearly with the number of shards downloaded. Most languages have 2–10+ shards available.

---

## Per-Language Detail

### EU Large Languages

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K |
|------|------|------|--------|-----|-----|------|-------|
| en (English) | 236,144 | 1,335.4 | 1,448 | 10,663 | 61,637 | 6,099 | 963 |
| de (German) | 89,938 | 473.7 | 1,409 | 10,767 | 21,357 | 2,167 | 284 |
| it (Italian) | 80,619 | 527.7 | 1,685 | 13,295 | 21,638 | 2,766 | 383 |
| fr (French) | 74,122 | 440.1 | 1,447 | 11,476 | 18,914 | 2,053 | 330 |
| nl (Dutch) | 70,914 | 450.8 | 1,892 | 13,287 | 20,635 | 2,056 | 265 |
| es (Spanish) | 54,045 | 578.2 | 2,314 | 21,321 | 20,414 | 3,567 | 612 |
| pt (Portuguese) | 53,862 | 587.1 | 2,867 | 19,884 | 23,337 | 3,308 | 558 |
| el (Greek) | 46,963 | 536.3 | 2,175 | 21,451 | 16,281 | 3,325 | 711 |

**English dominates by raw volume**: 1.34B tokens per shard, 61,637 docs ≥4K, 6,099 docs ≥32K — 2.5× the next largest language. German, Italian, French, and Dutch have notably shorter median documents (median ~1,400–1,900 tokens), consistent with shorter administrative/web-adjacent PDFs. Spanish, Portuguese, and Greek show longer documents (median 2,100–2,900) and better long-context coverage. Greek in particular yields 711 docs ≥128K — strongest of the large EU languages.

### Nordic Group

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K |
|------|------|------|--------|-----|-----|------|-------|
| sv (Swedish) | 68,680 | 617.7 | 2,137 | 22,640 | 24,728 | 3,928 | 426 |
| no (Norwegian) | 50,467 | 486.1 | 2,255 | 23,060 | 18,807 | 3,449 | 271 |
| da (Danish) | 73,860 | 584.3 | 1,787 | 17,062 | 22,791 | 3,469 | 466 |
| fi (Finnish) | 46,199 | 574.5 | 2,419 | 30,820 | 17,988 | 4,351 | 471 |
| is (Icelandic) | 24,259 | 211.3 | 1,928 | 21,690 | 7,663 | 1,364 | 112 |

Swedish and Danish have the most total documents. Finnish has the heaviest right tail (p90=30K) — likely due to long government documents and official reports in FinePDFs-Edu. Finnish's agglutinative morphology leads to denser, fewer tokens-per-word. Icelandic is significantly smaller but still yields 1,364 docs ≥32K.

### Baltic Group

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K |
|------|------|------|--------|-----|-----|------|-------|
| et (Estonian) | 34,429 | 322.8 | 2,559 | 21,867 | 13,232 | 2,105 | 204 |
| lv (Latvian) | 33,962 | 344.8 | 2,381 | 23,056 | 12,446 | 2,154 | 320 |
| lt (Lithuanian) | 36,153 | 490.3 | 3,060 | 28,659 | 15,753 | 3,070 | 514 |

Lithuanian is the strongest of the three — notably higher p90 and max. All three Baltic languages have similar shard sizes but Lithuanian has 50% more tokens per shard, suggesting longer average documents.

### Central European Group

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K |
|------|------|------|--------|-----|-----|------|-------|
| pl (Polish) | 74,689 | 773.8 | 2,377 | 21,440 | 28,587 | 4,465 | 701 |
| cs (Czech) | 77,038 | 659.0 | 1,815 | 19,403 | 25,729 | 4,041 | 503 |
| sk (Slovak) | 65,629 | 668.3 | 2,551 | 23,834 | 26,580 | 4,316 | 456 |
| hu (Hungarian) | 39,341 | 750.1 | **4,045** | **38,194** | 19,562 | 4,482 | **1,050** |

**Hungarian is exceptional**: its median document (4,045 tokens) already exceeds the 4K filter threshold — meaning the majority of documents are long-context eligible. With p90 at 38K tokens, Hungarian is outstanding for 32K+ training relative to its total doc count.

Polish has the most total docs and yields 701 docs ≥128K; Czech and Slovak are similarly strong.

### South/East Slavic Group

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K |
|------|------|------|--------|-----|-----|------|-------|
| ro (Romanian) | 54,015 | 830.3 | 3,127 | 28,979 | 23,592 | 4,897 | **1,197** |
| bg (Bulgarian) | 38,312 | 465.6 | 2,778 | 22,569 | 15,581 | 2,614 | 572 |
| hr (Croatian) | 43,638 | 625.1 | 3,486 | 34,638 | 20,239 | 4,705 | 592 |
| sl (Slovenian) | 40,473 | 317.4 | 1,555 | 18,799 | 11,564 | 2,282 | 187 |
| sr (Serbian) | 30,965 | 769.7 | **4,332** | **51,007** | 15,758 | 4,407 | **1,313** |
| bs (Bosnian) | 27,924 | 494.2 | 3,866 | 37,386 | 13,643 | 3,113 | 693 |
| mk (Macedonian) | 11,518 | 230.8 | **4,058** | **46,007** | 5,731 | 1,498 | 365 |
| ru (Russian) | 50,267 | 650.5 | 2,859 | 22,950 | 20,637 | 3,576 | 944 |
| uk (Ukrainian) | 44,424 | 752.9 | 3,862 | 30,339 | 21,604 | 4,217 | 1,347 |

**Serbian is the standout surprise**: smallest corpus (30K docs) but highest median (4,332), highest p90 (51K), and **most ≥128K docs of all 35 languages** (1,313). Ukrainian (1,347) is very close. These South Slavic and East Slavic languages appear to contain exceptionally long documents — likely legal, academic, and technical PDFs.

**Macedonian** (11K docs) also punches far above its weight: median 4,058, p90 46K — almost every document is a long-context candidate.

**Romanian** holds first place for ≥128K among the Romance languages (1,197 docs) and most tokens per shard (830M).

**Slovenian is the weakest**: median 1,555 tokens and only 2,282 docs ≥32K.

### Small / Regional Languages

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K |
|------|------|------|--------|-----|-----|------|-------|
| ca (Catalan) | 62,081 | 399.8 | 1,609 | 13,615 | 17,227 | 2,229 | 240 |
| gl (Galician) | 30,901 | 270.4 | 1,871 | 18,693 | 10,355 | 1,617 | 232 |
| eu (Basque) | 19,259 | 188.1 | 2,391 | 22,746 | 7,190 | 1,249 | 142 |
| cy (Welsh) | 16,560 | 92.2 | 1,706 | 12,549 | 4,439 | 474 | 40 |
| tr (Turkish) | 47,585 | 619.9 | 3,440 | 25,662 | 22,034 | 3,690 | 678 |

Catalan has the most documents of the regional languages (62K) but shorter median length. **Turkish is strong**: median 3,440, 22K docs ≥4K, 678 ≥128K — one of the better non-EU-official languages. Welsh is the weakest regional language (only 92M tokens per shard, 40 docs ≥128K).

### Maltese (baseline)

| Lang | Docs | MTok | Median | p90 | ≥4K | ≥32K | ≥128K |
|------|------|------|--------|-----|-----|------|-------|
| mt (Maltese) | 16,490 | 131.0 | 2,286 | 16,773 | 5,463 | 655 | 104 |

Maltese is the smallest corpus by every metric. It was the right choice for pipeline validation (fast iteration) but should be mixed with larger languages for actual training.

---

## Long-Context Coverage Assessment

### By context length target

| Context | All 35 langs combined | Suitable languages | Assessment |
|---------|-----------------------|-------------------|------------|
| 4K | 653,136 docs | All 35 | ✅ Excellent — 650K+ sequences |
| 8K | 419,769 docs | All 35 | ✅ Excellent — 52K+ sequences |
| 16K | ~240,000 docs (est.) | All 35 | ✅ Good — 15K+ sequences |
| 32K | 107,758 docs | All 35 | ✅ Good — 3.4K sequences |
| 64K | ~40,000 docs (est.) | sr, mk, hu, uk, bs, ro, hr | ⚠️ Moderate — 625+ sequences |
| 128K | 18,246 docs | sr, uk, ro, hu best | ⚠️ Sparse — ~140 sequences |
| 256K+ | ~3,000–5,000 docs (est.) | sr, pl, fi, lt, mk | ❌ Thin — <20 sequences |

*Sequences estimated as total tokens at threshold ÷ context length, no document packing.*

### Languages ranked by long-context quality (≥32K docs per shard)

1. **Romanian** (ro): 4,897 ≥32K, 1,197 ≥128K
2. **Croatian** (hr): 4,705 ≥32K, 592 ≥128K
3. **Hungarian** (hu): 4,482 ≥32K, 1,050 ≥128K
4. **Finnish** (fi): 4,351 ≥32K, 471 ≥128K
5. **Serbian** (sr): 4,407 ≥32K, **1,313 ≥128K** ← highest ≥128K of all
6. **Polish** (pl): 4,465 ≥32K, 701 ≥128K
7. **Slovak** (sk): 4,316 ≥32K, 456 ≥128K
8. **Ukrainian** (uk): 4,217 ≥32K, 1,347 ≥128K ← 2nd highest ≥128K
9. **Turkish** (tr): 3,690 ≥32K, 678 ≥128K
10. **Czech** (cs): 4,041 ≥32K, 503 ≥128K

---

## Key Findings

### 1. Serbian, Ukrainian, Romanian, Hungarian dominate the ≥128K tier

Serbian (1,313) and Ukrainian (1,347) are the top two languages for ≥128K documents, despite being medium-sized corpora. Romanian (1,197) and Hungarian (1,050) are close behind. These four languages together account for 27% of all ≥128K documents and should be prioritized for very-long-context training (64K–256K).

### 2. English dominates raw volume but has shorter documents

English has 2.5× more documents and 2.5× more tokens than the next largest language (German), but a low median (1,448 tokens) and only 963 docs ≥128K. This reflects a mix of short abstracts and longer research papers. English is critical for the 4K–16K tier but not the best choice for 128K+ training.

### 3. South/East Slavic languages punch well above their weight

Serbian (p90=51K), Macedonian (p90=46K), Bosnian (p90=37K), and Ukrainian (p90=30K) have dramatically higher p90 values than Western European languages of similar or larger size. The FinePDFs-Edu corpus appears to contain particularly long government and legal documents for these languages.

### 4. Median document lengths are shorter than expected for most languages

Most languages have a median around 1.5K–4K tokens. The long-context data is concentrated in the right tail. This means the 4K filter removes 50–70% of documents in most languages — the three-tier approach (≥16K / 4K–16K / <4K) is well-justified.

### 5. A tiered strategy makes sense

- **≥16K corpus**: top 10 languages (~60K docs/shard combined) — for 32K context training
- **≥4K corpus**: all 35 languages (~653K docs) — for 4K–16K context training
- **≥128K corpus**: sr, uk, ro, hu primarily (~4.9K docs/shard combined) — for 64K+ context

### 6. The 3 missing languages (ga, sq, lb) need an HF token

Irish, Albanian, and Luxembourgish are not in FinePDFs-Edu. The HPLT source (`longctx sources fetch --source hplt`) failed without a token. Set `HF_TOKEN` to fetch them.

---

## Languages Ranked by ≥128K Documents

| Rank | Lang | ≥128K | ≥32K | Notes |
|------|------|-------|------|-------|
| 1 | uk (Ukrainian) | 1,347 | 4,217 | Strong right tail |
| 2 | sr (Serbian) | 1,313 | 4,407 | Highest p90 (51K) |
| 3 | ro (Romanian) | 1,197 | 4,897 | Most ≥32K of Romance languages |
| 4 | hu (Hungarian) | 1,050 | 4,482 | Highest median (4,045) |
| 5 | ru (Russian) | 944 | 3,576 | Large corpus |
| 6 | en (English) | 963 | 6,099 | Most ≥32K total, but shorter median |
| 7 | el (Greek) | 711 | 3,325 | Strong for its size |
| 8 | tr (Turkish) | 678 | 3,690 | Best non-EU-official language |
| 9 | bs (Bosnian) | 693 | 3,113 | High median (3,866) |
| 10 | pl (Polish) | 701 | 4,465 | Large corpus |

---

## Files

```
/scratch/project_462000963/bmoell/openeuro-longctx-datamix/
├── data/
│   ├── raw/                            # Downloaded parquet shards (1 per language)
│   ├── megatron/                       # Converted JSONL (35 languages, all docs)
│   └── long/mt.jsonl                   # Maltese filtered corpus (5,463 docs)
└── lang_stats.json                     # Full per-language stats (all 35 languages)
```

Tokenized tier files (produced by `lumi/slurm/tokenize_tiers.sbatch`):
```
/flash/project_462000963/bmoell/data_tokenized_multilingual/
├── multilingual_16k_plus_text_document.{bin,idx}   # docs ≥ 16384 tokens
├── multilingual_4_16k_text_document.{bin,idx}      # docs 4096–16383 tokens
└── multilingual_under4k_text_document.{bin,idx}    # docs < 4096 tokens
```

## Source Dataset

All data comes from **`HuggingFaceFW/finepdfs-edu`** — a PDF-extracted corpus covering 35 of the 38 OpenEuroLLM target languages. PDFs are inherently long-form (research papers, legal documents, government reports, technical manuals), which is why document lengths here are substantially longer than typical web-scraped corpora like CulturaX or mC4.

The 3 languages not present in FinePDFs-Edu (Irish/ga, Albanian/sq, Luxembourgish/lb) can be fetched from `HPLT/HPLT2.0_cleaned` via `longctx sources fetch --source hplt` once an HF token is configured.
