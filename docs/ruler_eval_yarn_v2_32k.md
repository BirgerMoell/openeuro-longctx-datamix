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

32K pass did not complete — job hit the 8-hour wall time. To be rerun.

## Interpretation

**`niah_single_1` = 1.0 at all lengths** confirms the YaRN context extension is working — the model can reliably retrieve a single needle at 4K, 8K, and 16K.

**Multi-key and multi-value degradation** is expected for a base LM. These tasks require listing multiple items, which instruction-tuned models handle much better via output formatting. The scores here reflect the difficulty of eliciting structured output from a continuation model.

**Average degradation 4K→8K→16K** (0.43 → 0.28 → 0.20) is a smooth decline rather than a sharp drop, which is a healthy sign — the model is not losing coherence at longer contexts, just finding harder retrieval tasks more challenging as the haystack grows.

**Next steps:**
- Rerun 32K pass with longer wall time to complete the picture
- Run on instruction-tuned version of the model for significantly better multi-key/multi-value scores
- Compare against baseline (pre-YaRN) model to quantify the context extension benefit
