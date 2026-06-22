# Plan: Extending 128K → 256K (next stage, nice-to-have)

**Status:** plan only. Focus stays on making **128K** solid first; 256K is the next step once it is.
**Date:** 2026-06-22.

## TL;DR
The *training* step to 256K is easy and cheap (one more ABF stage, ~4–5h, trivial compute).
**All the real work is in data and eval** — the same two things that limited 128K, amplified.

## Method — one more ABF stage
`ckpt_131072_v2 → [256K] → ckpt_262144`: load with `--finetune`, raise `--seq-length` /
`--max-position-embeddings` to 262144, raise RoPE `--rotary-base`, short continued-pretrain.
- **θ:** NTK-aware ≈ 100k × (256K/4K) = ~6.4M; we used 5M at 128K, so **~10M at 256K**
  (double, slightly conservative). 256K is ~64× native — near where uniform θ scaling starts
  blurring short-range, so **A/B θ-scaling vs YaRN here** (Jouni has the YaRN code path).

## Config (64 nodes / 512 GCDs, TP=8)
| | 128K (current) | **256K (next)** |
|---|---|---|
| seq_length | 131072 | **262144** |
| rotary-base θ | 5,000,000 | **~10,000,000** |
| context-parallel | 8 | **16** (2 nodes / sequence) |
| DP @ 64 nodes | 8 | **4** |
| throughput | ~244 tok/s/GPU | **~120 tok/s/GPU** (est) |
| token budget | 2B | **~0.5–1B** (short finishing stage) |
| wall time | ~5h | **~4–5h** |

Compute is trivial (~0.5–1B tokens). CP=16 + flash-attn keeps per-GPU memory ~flat vs
128K@CP=8 + `--no-create-attention-mask-in-dataloader`.

## The two hard parts (NOT the training)

### 1. Data — the real bottleneck (the 128K lesson, compounded)
At 128K only ~19% of mix tokens were genuine long-range → far-depth (depth-0) retrieval failed.
At 256K it's worse — natural ≥256K docs are rarer. Mitigations:
- **Length-bias hard** with Jouni's `sample_idx.py` at `long_threshold=262144`. finepdfs has
  monster docs (up to ~83M tokens), so some ≥256K exists — but thin per language.
- **Construct 256K sequences:** concatenate same-topic/same-language docs; **repo-pack code**
  (whole repos easily exceed 256K); books.
- **Synthetic long-range / recall data** — at 256K you largely *manufacture* the long-range
  dependencies (needle/MRCR-style spanning the full window).
- **Honest risk:** multilingual ≥256K is very scarce; the 256K stage will skew English/code/
  synthetic. Most non-English langs won't have enough natural ≥256K.

### 2. Eval infrastructure — current eval breaks at 256K
`scripts/eval_base_lm_niah.py` does a **single-GPU HF forward**. Even with `logits_to_keep`,
a 256K forward for a 9B model won't fit one MI250X GCD. So 256K eval needs one of:
- **context-parallel / chunked-prefill** forward (shard the 256K sequence), or
- a **served model** (vLLM-ROCm, long-context), or
- **Megatron-native** CP-aware scoring (same stack as training).
Build this **before** the 256K run so the result is measurable.

## Order of operations
1. **Make 128K solid** (v2 length-biased rerun + broader RULER eval). Don't extend on a weak
   foundation — the depth-0 lesson.
2. **Build ≥256K data** (length-biased + synthetic + repo-pack) — gating item #1.
3. **Build a 256K-capable eval** (CP/chunked or served) — gating item #2.
4. Run the 256K ABF stage (~4–5h) and eval.

## Beyond 256K (context)
512K→1M is a different regime: dense attention (even with CP) gets prohibitive, and natural
data essentially runs out. That's where **sparse attention (DSA / IndexShare, GLM-5.2 style)**
becomes the architecture — a separate research project, not a continued-pretraining stage.
