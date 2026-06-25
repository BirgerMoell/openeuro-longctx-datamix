# Super-long context (512K → 1M → 2M): strategy & plan

**Status:** experimental next step. **Gated on 256K working first** (MIOpen prebuild in progress).
**Date:** 2026-06-25. Data: `birgermoell/oellm-longctx-tokenized-superlong-512k-1m-2m-v2`.

## TL;DR
We now have a **predictive law** for the one thing that actually mattered (RoPE θ), so the
θ values for 512K/1M/2M are already known. The real constraints at super-long are **(1) the
O(n²) attention compute wall, (2) eval infrastructure, and (3) where dense attention stops
being viable and sparse attention must take over.** 512K is a reasonable experimental target
with our current ABF+CP approach; 1M is a stretch; **2M dense is a proof-of-concept that likely
needs sparse attention** to be practical.

## 1. θ schedule — already determined by our scaling law
We measured: critical θ ≈ **doubles per context-length octave** (64K→8M, 128K→16M, 256K→32M).
Extrapolating:

| context | θ (rotary-base) | status |
|---|---|---|
| 256K | 32M | validating |
| **512K** | **64M** | predicted |
| **1M** | **128M** | predicted |
| **2M** | **256M** | predicted (law may break — validate w/ needle-PPL) |

Each stage: load previous checkpoint, `--finetune`, raise `--seq-length` and `--rotary-base`,
short continued-pretrain. **Caveat:** at θ ≥ 128M the law is unvalidated and very high θ can
erode short context — validate each stage with the cheap θ-sweep + depth-0 NIAH, and fall back
to **LongRoPE2 searched per-dim scaling** if uniform θ degrades.

## 2. Parallelism & the compute wall (the real bottleneck)
**Memory is fine** — context-parallel (CP) shards the sequence, so per-GCD tokens stay constant
if CP scales with length (keep ~16K tokens/GCD, as in 128K@CP8):

| context | CP | nodes / sequence | per-GCD seq | rel. attention compute vs 128K |
|---|---|---|---|---|
| 128K | 8 | 1 | 16K | 1× |
| 256K | 16 | 2 | 16K | 4× |
| 512K | 32 | 4 | 16K | 16× |
| 1M | 64 | 8 | 16K | 64× |
| 2M | 128 | 16 | 16K | **256×** |

**Attention is O(n²)** — a 2M step does ~256× the attention work of a 128K step, and CP ring-
communication spans 16 nodes for a *single* sequence (DP=1). So **wall-time, not memory, is the
wall.** 512K (16×) is tolerable for a short finishing stage; 2M (256×) is only feasible for a
tiny token budget (proof-of-concept), and CP=128 cross-node ring attention is unproven for us.

## 3. Data (this dataset)
512K/1M/2M sequences across arxiv, books, code-repo-pack, RFC specs, docsites, algebraic-stack,
and **`synthetic_recall`** — the key asset: at ≥512K there are essentially **no natural
documents with genuine long-range dependencies**, so retrieval ability must be *manufactured*
via synthetic recall/needle tasks spanning the full window. Mix per stage: genuine-long
(concat papers/books/repos) for fluency + **heavy synthetic_recall** for the long-range skill.
Same dtype/tokenizer as our 256k pipeline (verify before use). English/code-skewed (expected at
these lengths).

## 4. Eval infrastructure — the gating blocker
Our base-LM NIAH does a **single-GPU forward**; it already won't fit 256K, let alone 512K–2M.
Before any super-long run is measurable we need one of:
- **CP-aware / chunked-prefill** scoring (shard the forward across GPUs), or
- a **served model** (vLLM-ROCm long-context), or
- **Megatron-native** forced-choice scoring (reuse the training stack's CP forward).
**Build this first** — an unmeasurable 2M model is worthless. The Megatron-native path is most
aligned (we already run CP forwards in training).

## 5. Architecture: where dense ends and sparse begins
Dense attention (even with CP) hits the 256× compute wall by 2M. A *practical* 1M–2M model is a
different architecture: **sparse / linear attention** — sliding-window + global tokens, or
DeepSeek-style **DSA / native sparse attention**, or hybrid (a few full-attention layers +
mostly local). That's a **separate research track** (architecture change + retraining), not an
ABF continued-pretraining stage. Our ABF+CP push is the *dense baseline* that tells us how far
dense goes and provides a reference for the sparse work.

## 6. Staged plan with decision gates
0. **Prereq:** 256K works (MIOpen prebuild resolves the first-step hang) + a **≥256K-capable
   eval** exists. *Do not proceed without both.*
1. **512K @ θ=64M, CP=32** (4 nodes/seq), short budget (~0.3–0.5B tok, heavy synthetic_recall).
   Eval depth-0 @ 512K. **Gate:** depth-0 recovers → law holds, continue.
2. **1M @ θ=128M, CP=64** (8 nodes/seq), tiny budget. Eval @ 1M. **Gate:** does dense+CP=64 even
   run at acceptable throughput? If wall-time explodes → stop dense here.
3. **2M @ θ=256M, CP=128** (16 nodes/seq) — **proof-of-concept only**: can we get *any* working
   2M retrieval with dense+ABF? Expect very low throughput. Mainly a feasibility datapoint.
4. **If dense stalls (likely by 1M):** pivot to the **sparse-attention** research track using the
   same data — the real path to practical 1M–2M.

## 7. Honest risk assessment
- **Highest value / lowest risk:** 512K (likely works, modest cost) — a real deliverable.
- **Medium:** 1M (compute heavy, CP=64 unproven, but plausible).
- **Experimental:** 2M dense (256× compute, CP=128 ring across 16 nodes — proof-of-concept).
- **Biggest unknowns:** (a) does the 256K first-step hang fix generalize to CP=32/64/128?
  (b) eval infra at ≥512K; (c) does the θ-doubling law hold past 32M or do we need LongRoPE2.
- **Compute budget is not the limit** (~1.4M GPU-h); **wall-time per step and infra are.**

## Bottom line
Super-long is **θ-predicted and data-ready**, but it's an **infrastructure + architecture**
project, not a data one. Sequence: fix 256K → build a ≥256K eval → 512K (the real next
deliverable) → 1M (stretch) → 2M (proof-of-concept) → sparse-attention track for practical 1M+.
