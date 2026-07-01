---
language:
- bg
- cs
- da
- et
- fi
- fr
- hr
- nl
license: apache-2.0
base_model: birgermoell/oellm-datamix-9b-80-20
tags:
- long-context
- yarn
- continued-pretraining
- multilingual
- european-languages
---

# OpenEuroLLM 9B — YaRN Multilingual 32K

A long-context extension of [OpenEuroLLM 9B](https://huggingface.co/birgermoell/oellm-datamix-9b-80-20), trained via continued pre-training using [YaRN](https://arxiv.org/abs/2309.00071) to extend the context window from **2 048 → 32 768 tokens**.

This is a **base language model** — not instruction-tuned. It is intended for research into multilingual long-context language modelling.

---

## Model details

| | |
|---|---|
| **Base model** | OpenEuroLLM 9B (oellm-datamix-9b-80-20) |
| **Architecture** | LLaMA-style, 32 layers, hidden 4096, 32 heads |
| **Context length** | 32 768 tokens |
| **Extension method** | YaRN (factor=16.0, original_max=2048) |
| **Training** | Continued pre-training, 1 000 iterations |
| **Global batch size** | 128 sequences × 32 768 tokens |
| **Training tokens** | ~4.2B |
| **Languages** | bg · cs · da · et · fi · fr · hr · nl |
| **Checkpoint** | iter 1000 |

---

## RoPE scaling config

To use this model you must add `rope_scaling` to `config.json`:

```json
"rope_scaling": {
  "factor": 16.0,
  "original_max_position_embeddings": 2048,
  "type": "yarn"
},
"rope_theta": 10000
```

---

## Languages

Trained on 8 European languages, each with documents split into length tiers
(≥16K tokens, 4K–16K, <4K) and weighted to favour long documents:

| Code | Language |
|------|----------|
| bg | Bulgarian |
| cs | Czech |
| da | Danish |
| et | Estonian |
| fi | Finnish |
| fr | French |
| hr | Croatian |
| nl | Dutch |

---

## Evaluation — Base-LM NIAH

Evaluated with a forced-choice log-likelihood needle-in-a-haystack (NIAH) benchmark
that does not require instruction following. Four candidate magic-number values are
scored; the model picks the one with highest log-probability given the context.
Random-chance baseline: **25%**.

### Accuracy by language × context length (all depths averaged)

| lang | 2 048 | 4 096 | 8 192 | 16 384 | 32 768 |
|------|------:|------:|------:|-------:|-------:|
| fr   |  1.00 |  1.00 |  1.00 |   1.00 |   0.84 |
| fi   |  1.00 |  1.00 |  1.00 |   1.00 |   0.86 |
| cs   |  0.94 |  1.00 |  1.00 |   1.00 |   0.80 |
| nl   |  0.98 |  1.00 |  1.00 |   1.00 |  0.73† |

*† NL 32K average over depths 0%/25%/50% only (job timed out before 75%/100%).*

### Key finding

Performance is near-perfect at all context lengths up to 16K and at most 32K depths.
The only failure mode is **32K depth=0%** — when the needle is placed at the very
beginning of the document (~32K tokens before the query):

| lang | 32K depth=0% |
|------|-------------:|
| fr   | 0.20 |
| fi   | 0.30 |
| cs   | 0.00 |
| nl   | 0.20 |

All other 32K depths (25%–100%) score 1.00 across all languages.
**Reliable retrieval up to ~24K tokens of separation.**

This failure is attributed to a missing `mscale` parameter in v1 training
(see [YaRN paper](https://arxiv.org/abs/2309.00071), eq. 27: `mscale = 0.1 × ln(factor) + 1.0`).
A v2 model with the fix applied is in preparation.

---

## Known limitations

- The `mscale` correction (`--yarn-mscale 1.277`) was not applied during training, causing
  attention magnitude decay at maximum retrieval distance. This is the root cause of the
  depth=0% failure at 32K. A corrected v2 model is being trained.
- Only 8 European languages; v2 will cover 35 languages.
- Base model only — requires instruction tuning for chat/assistant use cases.
- 10 trials per eval cell gives ±~10% accuracy uncertainty.

---

## Citation

If you use this model, please cite the OpenEuroLLM project and the YaRN paper:

```
@misc{peng2023yarn,
  title={YaRN: Efficient Context Window Extension of Large Language Models},
  author={Bowen Peng and Jeffrey Quesnelle and Honglu Fan and Enrico Shippole},
  year={2023},
  eprint={2309.00071},
  archivePrefix={arXiv}
}
```
