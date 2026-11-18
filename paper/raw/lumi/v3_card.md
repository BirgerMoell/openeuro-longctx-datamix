---
license: apache-2.0
tags: [long-context, rope, abf, niah, qwen3, openeurollm, base-model]
pipeline_tag: text-generation
---

# OELLM baby 9B — 128K context (ABF, θ=32M) — v3 (full training)

Fully-trained production version of [oellm-9b-128k-theta32m](https://huggingface.co/birgermoell/oellm-9b-128k-theta32m).
Same θ=32M / 128K config, but **2B tokens** of 128K continued-pretraining (vs the 0.3B
sweep-arm of the original). Base model, multilingual, Qwen3 dense 9B.

θ=32M fixes the far-position (depth-0) retrieval failure that uniform low-θ ABF suffers from
(RoPE high-dimension OOD; critical θ doubles per length-octave: 64K→8M, 128K→16M, and 32M
gives perfect 128K with headroom). Base model — not instruction-tuned. Config: rope_theta=32000000,
max_position_embeddings=131072.
