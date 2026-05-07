# Strategy: Finding More Long Documents

**Context:** The current Maltese (mlt_Latn) corpus has only 12% of documents ≥32K tokens and 1.9% ≥128K tokens — too sparse for serious long-context training above 16K. This document describes how to get more.

---

## Why Long Documents Matter

For long-context training the model needs to *see* long sequences during training — not just short documents concatenated with EOS tokens. A 32K-token training sequence ideally comes from a single coherent document so the model learns real long-range dependencies (a book chapter, a legal brief, a Wikipedia article with its full citations). Packing short documents together teaches packing tolerance but not true long-range reasoning.

The current corpus situation:

| Context | Docs available | Sequences |
|---------|---------------|-----------|
| 8K | 3,283 (60%) | ~13,882 |
| 32K | 655 (12%) | ~3,470 |
| 128K | 104 (2%) | ~867 |

To train a model that generalises at 32K+ tokens, a practical minimum is ~10K–50K training sequences at that context length. We are currently 3–14× short of that for 32K, and ~12–60× short for 128K.

---

## Option 1: More Languages via the Existing Pipeline (Best ROI)

The pipeline already supports multiple languages with one flag change. Languages with larger CulturaX/OPUS corpora will have far more long documents.

**Recommended target languages for OpenEuroLLM:**

| Language | Code | CulturaX size | Expected long docs |
|----------|------|--------------|-------------------|
| Swedish | sv | Large (~10GB) | Many thousands |
| Norwegian | no | Large | Many thousands |
| Finnish | fi | Large | Many thousands |
| Danish | da | Medium | Thousands |
| German | de | Very large | Tens of thousands |
| French | fr | Very large | Tens of thousands |
| Estonian | et | Medium | Thousands |
| Latvian | lv | Medium | Thousands |
| Lithuanian | lt | Medium | Thousands |

**How to run:**

```bash
# Multi-language download (adjust --shards as needed)
longctx download --languages sv,no,fi,da --shards 2
longctx convert
longctx filter-long --min-tokens 4096
longctx tokenize
```

**Expected impact:** Swedish alone likely has 10–50× more long documents than Maltese. Running 5–10 languages should push the ≥32K document count from ~650 to tens of thousands.

---

## Option 2: More Shards of Existing Languages

The current pipeline uses `--sample --shards 1` — only 1 of potentially dozens of available shards per language. Downloading all shards multiplies the dataset proportionally.

```bash
# Download all available shards for Maltese (currently 1) and Swedish (many more)
longctx download --languages mt,sv --shards 99  # 99 = "as many as exist"
```

Check how many shards are available with:
```bash
longctx estimate --languages sv,no,fi,da,de
```

---

## Option 3: Wikipedia Dumps

Wikipedia articles are ideal long-context training data — coherent, factual, naturally long (4K–64K tokens per article), available for hundreds of languages. Maltese Wikipedia alone has ~6,000 articles; Swedish has ~2.5 million.

The current pipeline does not include Wikipedia. Adding it requires:

1. Download a Wikimedia dump (XML format, available at dumps.wikimedia.org)
2. Extract article text (e.g. with `wikiextractor`)
3. Add as a new source in the `longctx convert` step

**Expected article lengths:** Most Wikipedia articles are 4K–32K tokens — exactly the sweet spot we are missing.

```bash
# Sketch of a wiki download step to add to the pipeline
longctx download-wiki --languages sv,fi,no,da,de,fr \
  --output-dir data/raw/wiki
longctx convert --source wiki
longctx filter-long --min-tokens 4096
```

---

## Option 4: Books / Project Gutenberg

Books are the single best source of very-long documents (50K–500K tokens each). Project Gutenberg provides free public-domain books in many languages.

- **The 1.6M-token outlier** in the current Maltese dataset is almost certainly a book — it fits perfectly. Gutenberg-style content is already in CulturaX.
- For intentional book coverage: the [ROOTS corpus](https://huggingface.co/datasets/bigscience-data/roots) and [mC4](https://huggingface.co/datasets/mc4) both include book-like long documents.
- For English long-context baselines: [Books3](https://pile.eleuther.ai/) (part of The Pile) is a standard benchmark source.

Adding a Gutenberg pipeline would yield hundreds of 100K+ token documents per major European language.

---

## Option 5: Legal Documents (EUR-Lex)

EU legislation is available in all 24 official EU languages and is:
- Very long (directives and regulations are often 32K–256K tokens)
- Formally structured (good for reasoning tasks)
- Parallel across languages (useful for multilingual training)

EUR-Lex data is already in CulturaX for many languages. It can also be downloaded directly from `eur-lex.europa.eu` via their bulk download API.

---

## Option 6: Lower Filter Threshold + Packing

A softer approach that requires no new data sources: lower `--min-tokens` to 2048 and use Megatron's document packing (concatenating documents with EOS separators).

**What this gives:**
- 2048 threshold would keep roughly 3,000–4,000 additional documents from the 512–4096 bucket
- With packing at 8K context, two 4K documents become one training sequence

**Tradeoff:** The model sees multiple short documents per sequence rather than one long coherent document. This is fine for teaching the model to handle long contexts but does not teach genuine long-range coherence.

```bash
# Lower filter threshold
longctx filter-long --min-tokens 2048

# Enable packing in Megatron training
--reset-position-ids     # treat each packed document independently
--reset-attention-mask   # prevent attention across document boundaries
--eod-mask-loss          # don't compute loss on EOD tokens
```

---

## Recommended Immediate Actions

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| 1 | Add Swedish + Finnish to the pipeline | Low (1 line) | High — 10–50× more long docs |
| 2 | Run `longctx estimate` for all target languages to see available shards | Low | Medium — informs decision |
| 3 | Enable document packing in Megatron | Medium (few flags) | Medium — doubles usable sequences |
| 4 | Add Wikipedia extraction step | Medium | High — natural long articles |
| 5 | Add more shards per language | Low | Medium — linear scaling |

### Running the pipeline for multiple languages

```bash
# 1. Check what's available
longctx estimate --languages sv,no,fi,da,de,fr,et,lv,lt,mt

# 2. Download (start with Nordic languages)
longctx download --languages sv,no,fi,da --shards 2
longctx convert
longctx filter-long --min-tokens 4096
longctx tokenize

# 3. Run training on the combined dataset
# (update --data-path to include all language files)
```

---

## The 1.6M-Token Outlier

The current corpus contains one document with 1,596,102 tokens (~3.7M characters). This is most likely a book or large Wikipedia article collection crawled as a single page. Before training at long context, it should be verified:

```bash
# Inspect on LUMI
python3 -c "
import json
with open('data/long/mt.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d['token_count'] > 1000000:
            print('token_count:', d['token_count'])
            print('first 500 chars:', d['text'][:500])
            print('last 200 chars:', d['text'][-200:])
"
```

If it is genuine long-form content, it is extremely valuable training data. If it is a crawl artifact (e.g., repeated content, navigation menus concatenated), it should be filtered out.
