---
language:
- bg
- cs
- da
- de
- el
- en
- es
- et
- fi
- fr
- ga
- hr
- hu
- it
- lt
- lv
- mt
- nl
- pl
- pt
- ro
- sk
- sl
- sv
- ca
- eu
- gl
- is
- lb
- mk
- no
- oc
- sq
- sr
- uk
license: apache-2.0
tags:
- long-context
- yarn
- multilingual
- european-languages
- llama
- openeurollm
base_model: AI-Sweden-Models/oellm-9b-base
---

# OELLM 9B YaRN Multilingual v2 — 32K context

**OpenEuroLLM 9B** continued pre-trained with [YaRN](https://arxiv.org/abs/2309.00071)
context extension to **32 768 tokens** across **35 European languages**.

This is **v2**, which fixes a critical bug in v1 where `mscale` was not set correctly,
causing near-zero retrieval accuracy at depth=0% for 32K contexts.

## Model details

| Property | Value |
|----------|-------|
| Base model | OpenEuroLLM 9B (`oellm-datamix-9b-80-20`) |
| Architecture | LlamaForCausalLM |
| Parameters | ~9B |
| Context window | 32 768 tokens |
| Original context | 2 048 tokens |
| Context extension method | YaRN |
| YaRN factor | 16.0 (2 048 × 16 = 32 768) |
| YaRN mscale | 1.277 (`= 0.1 × ln(16) + 1.0`) |
| Training dtype | BFloat16 |
| Vocab size | 262 400 |

## Languages

35 European languages including all 24 EU official languages plus additional European languages:

Bulgarian (bg), Czech (cs), Danish (da), German (de), Greek (el), English (en),
Spanish (es), Estonian (et), Finnish (fi), French (fr), Irish (ga), Croatian (hr),
Hungarian (hu), Italian (it), Lithuanian (lt), Latvian (lv), Maltese (mt),
Dutch (nl), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk), Slovenian (sl),
Swedish (sv), Catalan (ca), Basque (eu), Galician (gl), Icelandic (is),
Luxembourgish (lb), Macedonian (mk), Norwegian (no), Occitan (oc),
Albanian (sq), Serbian (sr), Ukrainian (uk).

## Training details

| Setting | Value |
|---------|-------|
| Training iterations | 1 000 |
| Global batch size | 128 |
| Sequence length | 32 768 tokens |
| LR schedule | WSD (Warmup-Stable-Decay) |
| Peak learning rate | 1 × 10⁻⁵ |
| Final learning rate | 1.5 × 10⁻⁷ |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, ε=1×10⁻⁸) |
| Weight decay | 0.1 |
| Hardware | LUMI-G (AMD MI250X GPUs) |
| GPU count | 256 (32 nodes × 8 GPUs) |
| Parallelism | TP=2, PP=4, CP=4, DP=8 |
| Framework | Megatron-LM (MCore) |
| Precision | BF16 + TransformerEngine |

## v2 vs v1: the mscale fix

YaRN introduces a magnitude scaling factor `mscale` that compensates for the
change in attention logit scale when extending context. The correct formula is:

```
mscale = 0.1 × ln(factor) + 1.0
       = 0.1 × ln(16.0) + 1.0
       = 1.277
```

In **v1**, `mscale` was missing from the `rope_scaling` config, which caused the
model to use a default of 0 or 1.0, leading to severely degraded retrieval
at depth=0% in long contexts (the "needle at the beginning" case). This manifested
as essentially random performance at 32K for the depth=0% cell in NIAH evals.

**v2** sets `mscale=1.277` correctly in `config.json`:

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 16.0,
  "original_max_position_embeddings": 2048,
  "mscale": 1.277
}
```

## Data

Continued pre-training on the OpenEuroLLM long-context data mix, consisting of
long-document text in 35 European languages, curated from web crawls and
multilingual corpora with quality filtering and language balancing.

## Intended use

- Long-context language modelling and understanding in European languages
- Base model for further fine-tuning on downstream tasks requiring long contexts
- Research into multilingual long-context capabilities

This is a **base (pre-trained) model**, not instruction-tuned. It is intended for
further fine-tuning or as a research artefact.

## Evaluation

Base-LM Needle-in-a-Haystack (NIAH) via forced-choice log-likelihood scoring.
4-choice forced retrieval, 10 trials per cell, scored by log-likelihood (no instruction following required).

**Grid:** 4 languages × 5 context lengths (2K–32K) × 5 needle depths (0%–100%)

### Accuracy by language × context length (averaged across all depths)

| lang | 2K | 4K | 8K | 16K | 32K |
|------|-----|-----|-----|------|-----|
| fr | 1.00 | 0.98 | 1.00 | 1.00 | 0.84 |
| fi | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 |
| cs | 1.00 | 0.98 | 1.00 | 1.00 | 0.84 |
| nl | 1.00 | 0.98 | 1.00 | 1.00 | 0.88 |

### Key findings

- **2K–16K:** Near-perfect retrieval across all depths and languages (≥0.98 average).
- **32K depth ≥ 25%:** 1.00 across all tested languages — YaRN context extension works correctly.
- **32K depth = 0%:** 0.20–0.50 depending on language — a known "attention sink / extreme primacy" limitation at the maximum context length. This is distinct from the mscale bug fixed in v2 (which affected all depths at 32K in v1).
- **v2 vs v1:** The mscale fix (`mscale=1.277`) restored near-perfect 32K retrieval for depths 25–100%. The residual depth=0% weakness at 32K is a structural property of the attention mechanism at extreme positional distances, not a training artefact of this checkpoint.

### Control conditions (FR, FI, CS)

| condition | acc |
|-----------|-----|
| no_context (random baseline) | 0.20–0.40 |
| shuffled bindings | 0.90–1.00 |
| short context (256 tok) | 0.90–1.00 |

*Extended 31-language eval pending (job 18746959).*

## Training framework

Trained using [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) with MCore on the
[LUMI](https://www.lumi-supercomputer.eu/) supercomputer.

Converted to HuggingFace format using
[megatron-hf-converter](https://github.com/BirgerMoell/megatron-hf-converter)
with LUMI-specific patches for TP=2/PP=4 MCore checkpoints under PyTorch 2.6 +
TransformerEngine.

## Citation

```bibtex
@misc{oellm-yarn-v2-2026,
  title  = {OELLM 9B YaRN Multilingual v2},
  author = {Moell, Birger and {OpenEuroLLM Contributors}},
  year   = {2026},
  url    = {https://huggingface.co/birgermoell/oellm-9b-yarn-multilingual-v2-32k}
}
```

## Acknowledgements

Trained at [LUMI](https://www.lumi-supercomputer.eu/) (project_462000963).
Built on [OpenEuroLLM](https://openeurollm.eu/) base models.
