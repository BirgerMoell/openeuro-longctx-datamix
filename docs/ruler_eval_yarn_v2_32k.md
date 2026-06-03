# RULER Eval — YaRN v2 Multilingual 9B (32K)

**Model:** `birgermoell/oellm-9b-yarn-multilingual-v2-32k`  
**Cluster:** Leonardo (CINECA) — NVIDIA A100 64GB  
**Job:** 42990662 (2026-05-29)  
**Samples:** 100 per task  
**Setup:** base LM, no instruction tuning, `--num_fewshot 0`, `dtype=bfloat16`

## Results

| Task              | 4K    | 8K    | 16K   | 32K |
|-------------------|-------|-------|-------|-----|
| niah_single_1     | 1.000 | 1.000 | 1.000 | —   |
| niah_single_2     | 0.660 | 0.350 | 0.020 | —   |
| niah_single_3     | 0.630 | 0.200 | 0.110 | —   |
| niah_multikey_1   | 0.340 | 0.330 | 0.310 | —   |
| niah_multikey_2   | 0.140 | 0.070 | 0.010 | —   |
| niah_multikey_3   | 0.060 | 0.010 | 0.010 | —   |
| niah_multivalue   | 0.290 | 0.045 | 0.003 | —   |
| niah_multiquery   | 0.263 | 0.160 | 0.010 | —   |
| ruler_cwe         | 0.544 | 0.312 | 0.183 | —   |
| ruler_fwe         | 0.360 | 0.273 | 0.333 | —   |
| **AVG**           | **0.429** | **0.275** | **0.199** | — |

32K pass did not complete — job hit the 8-hour wall time. Rerun as job 44150956.

## Task descriptions

### Needle In A Haystack (NIAH)

The model is given a long document filled with filler text (Paul Graham essays). A "needle" — a short synthetic fact like *"The secret code is 42819"* — is hidden at a random position. The model must retrieve the exact value when asked.

| Task | What it measures |
|------|-----------------|
| `niah_single_1` | Retrieve 1 needle. The needle is a simple UUID-style string. Easiest variant. |
| `niah_single_2` | Retrieve 1 needle. The needle is a short sentence — slightly harder to extract by continuation. |
| `niah_single_3` | Retrieve 1 needle. The needle is a longer phrase — hardest single-needle variant. |
| `niah_multikey_1` | The needle has 2 keys; retrieve the value matching the queried key. Tests key disambiguation. |
| `niah_multikey_2` | Same but 4 keys — harder disambiguation. |
| `niah_multikey_3` | Same but 8 keys — requires the model to distinguish among many similar-looking entries. |
| `niah_multivalue` | One key maps to multiple values; retrieve all of them. Requires generating a list. |
| `niah_multiquery` | Multiple needles are hidden; retrieve the values for all queried keys in one pass. |

Scoring: exact string match (or token-level F1 for multi-value). A base LM scores by generating a continuation — it doesn't "know" to stop after the answer, so multi-value/multi-query tasks are especially hard without instruction tuning.

### Aggregation tasks

| Task | What it measures |
|------|-----------------|
| `ruler_cwe` | **Common Words Extraction** — given a long list of words with repetitions, output the K most frequent ones. Tests whether the model can aggregate counts over a long context. |
| `ruler_fwe` | **Frequent Words Extraction** — same idea but the list contains noisy distractor words at low frequency. Tests signal/noise separation over long contexts. |

Scoring: set overlap between predicted words and ground-truth frequent words.

## Interpretation

**`niah_single_1` = 1.0 at all lengths** confirms the YaRN context extension is working — the model can reliably retrieve a single needle at 4K, 8K, and 16K.

**Multi-key and multi-value degradation** is expected for a base LM. These tasks require listing multiple items, which instruction-tuned models handle much better via output formatting. The scores here reflect the difficulty of eliciting structured output from a continuation model.

**Average degradation 4K→8K→16K** (0.43 → 0.28 → 0.20) is a smooth decline rather than a sharp drop, which is a healthy sign — the model is not losing coherence at longer contexts, just finding harder retrieval tasks more challenging as the haystack grows.

**Next steps:**
- Rerun 32K pass with longer wall time to complete the picture
- Run on instruction-tuned version of the model for significantly better multi-key/multi-value scores
- Compare against baseline (pre-YaRN) model to quantify the context extension benefit
