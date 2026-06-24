# Depth-0 retrieval failure: diagnosis, eval caveat, and the θ sweep

**Date:** 2026-06-24. Why far-start (depth-0) NIAH retrieval fails at 64K/128K, what it
is *not*, and the experiment that should fix it.

## The symptom (apples-to-apples, 64K, by needle depth)
| depth (needle position) | v1 (catalogue) | v2-Jouni (≥64K tiered) | v2-ours (≥128K targeted) |
|---|---|---|---|
| 0.0 (far start) | 30% | **0%** | **5%** |
| 0.25 | 100% | 100% | 100% |
| 0.5 / 0.75 / 1.0 | 100% | 100% | 100% |
| 4K / 16K (all depths) | 100% | 100% | 100% |

Short + mid/late context are perfect. The entire deficit is **depth-0 at long context**, and
it is sharp (0%→100% between depth 0 and 0.25).

## Finding 1 — it is NOT a data-quantity problem
Two length-biased datasets (Jouni's ≥64K tiered; our ≥128K-targeted extraction) did **not**
lift depth-0 — if anything slightly *below* v1. More long-range tokens, even aggressively
≥128K-targeted, don't move it. So the lever is not data.

## Finding 2 — root cause is positional: undertrained high RoPE dimensions (OOD)
This is a documented, named failure mode. **LongRoPE2 (Microsoft, 2025,
[arXiv:2502.20082](https://arxiv.org/abs/2502.20082))**:

> *higher RoPE dimensions remain undertrained, leading to unexpectedly long rotation cycles …
> RoPE embeddings, especially in higher dimensions, do not complete their rotation cycles
> within the original context window.*

Depth-0 = maximum query→needle distance = governed by the **lowest-frequency / highest RoPE
dimensions**, which never completed a rotation during the original 4K pretraining → at long
range they are **out-of-distribution**, and uniform ABF (θ-scaling) does not bring them back
into a trained regime. An OOD positional code cannot be fixed by more data — matching Finding 1.

### YaRN is *directionally* right but not the answer
The "a better model used YaRN" intuition correctly points at *positional method, not data*. But:
- [Attention-perspective study (arXiv:2406.13282)](https://arxiv.org/abs/2406.13282): **NTK/ABF
  extrapolates ~128K; PI and YaRN only ~62K** — our ABF is actually *better* for far reach.
- YaRN is reported to **frequently fail NIAH** (Phi-3-mini, LLaMA-3-8B, LLaMA-3.1-8B).
So don't blindly switch to YaRN. Related: [Resonance RoPE (arXiv:2403.00071)](https://arxiv.org/abs/2403.00071)
rounds wavelengths to integer cycles to kill the OOD gaps.

### The SOTA fix (LongRoPE2)
1. **Get high-dim scaling right** via evolutionary search guided by **needle-driven perplexity**
   (PPL on *answer tokens only*, not averaged). Practical takeaway: **standard NTK/YaRN
   under-scale the critical high dimensions — you usually need a *larger* effective θ than the
   by-the-book formula.** Our θ (2M@64K, 5M@128K) is plausibly **too small**.
2. **Mixed context-window training** (short + long together). 128K with only ~10B tokens.

## Finding 3 — eval caveat: depth-0 = 0% is *below chance* (and why)
Chance for the 4-way forced choice is 25%, yet depth-0 scores ~0%. Investigated
`scripts/eval_base_lm_niah.py`:
- **Truncation ruled out:** `build_context` sizes to budget (no overshoot); the scorer uses
  `tokenizer(..., truncation=False)` — the full sequence reaches the model; the needle is
  present (at `pool[0]`, after the preamble — not clobbered by BOS).
- **Why below chance:** the 3 distractor candidate values are **placed into the context** as
  filler (`filler_kvs = distractor_kvs + …`; `distractors = [v for _,v in distractor_kvs[:3]]`).
  So at depth-0 the true value sits in the far (inaccessible) needle while all 3 wrong choices
  are in the **recent, visible** filler → a model that can't reach the far needle is **lured to
  an in-context distractor → systematically wrong → 0%, not 25%.**

**Implication:** this is an intentionally **adversarial** NIAH (distractors-in-context), not a
bug. The real signal (ABF far-retrieval fails) is valid, but **"0%" overstates severity** — the
honest read is "can't reach the far needle *and* gets pulled to visible distractors." For a
non-adversarial number, also run a **clean variant with distractors NOT in context**
(chance = 25%) to separate "can't retrieve" from "lured by distractors."

## Experiment: θ sweep at 128K (the cheap, decisive test of Finding 2)
From the v2 64K checkpoint (`output_real_v2_16n/ckpt_65536`), run the 128K stage at several θ
and eval depth-0:

| arm | θ (rotary-base) | note |
|---|---|---|
| baseline | 5,000,000 | already trained (`output_real_v2_16n/ckpt_131072`) |
| sweep A | 8,000,000 | |
| sweep B | 16,000,000 | |
| sweep C | 32,000,000 | push high dims into trained regime |

Each arm: short continued-pretrain (~0.3B tok, seq 131072, CP=8, 16 nodes), save
`ckpt_131072_th{θ}`, then convert + NIAH depth-0. **If depth-0 recovers with larger θ, the
fix is θ (high-dim OOD), and we then train the winning θ longer.** If θ plateaus, escalate to
LongRoPE2-style searched per-dim scaling (and Jouni's YaRN patch as one A/B arm), scored by
needle-driven PPL.

Script: `longctx-extend/theta_sweep.sbatch` (on LUMI).

## RESULTS — θ is the fix (2026-06-25)
Base-LM NIAH, depth-0 = far-start needle (the hardest position; the thing that was failing).

### depth-0 by θ and context length
| θ (rotary-base) | 64K depth-0 | 128K depth-0 |
|---|---|---|
| 2M / 5M (original) | 0–30% | **0%** |
| 8M | **100%** (all depths) | **0%** |
| **16M** | **100%** | **90%** (9/10), 128K overall 93% |
| 32M | 100% (eval in progress) | pending |

**Raising θ moved 128K depth-0 from 0% → 90%.** Short context (4K/16K) stayed 100%. This
confirms the diagnosis: depth-0 is a **θ / high-dim-OOD** problem, **not data** — two
length-biased datasets (Jouni ≥64K, ours ≥128K) all gave depth-0 ~0%; θ alone fixed it.

### The key relationship: critical θ scales with context length (≈ doubles per octave)
| context | θ that fixes depth-0 |
|---|---|
| 64K | 8M |
| 128K | 16M |
| **256K** | **~32M** (extrapolated) |
| (512K | ~64M) |

8M brings the high RoPE dims in-distribution to ~64K; 16M to ~128K. The required θ roughly
**doubles each time the context doubles** — consistent with NTK/critical-dimension theory, and
these are *measured* needed values (not the naive formula, which under-scales — the LongRoPE2
point).

## θ values for 256K
**Best estimate: θ ≈ 32M** for a 256K target, from the doubling relationship. De-risking:
the **32M checkpoint already exists** (sweep arm, trained at 128K) — if it holds 4K/16K/64K at
100% with no short-context regression, 32M is validated as safe before any 256K run.

**Clean θ schedule (matched to length):** `64K@8M → 128K@16M → 256K@32M`.

**To finalize 256K θ:** repeat the cheap sweep at 256K — short arms at **24M / 32M / 48M** from
the 128K@16M checkpoint, eval depth-0 on the staged 256K data (Birger's HF natural/structured
128K–256K sets + Jouni `sample_idx` ≥262144). Pick by depth-0 NIAH / needle-PPL.
**Caveat:** watch short-context retention — very high θ can erode 4K/16K (literature); if 32M
shows any short-ctx cost, use LongRoPE2-style searched per-dim scaling instead of uniform θ.

## Production
`real_v3_128k.sbatch` — full 2B-token 128K stage at **θ=16M** from the v2 64K ckpt (the
validated production value). Saves every 100 iters.
