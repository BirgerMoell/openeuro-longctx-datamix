# RoPE θ — suggestions for the next run

Recommendations for θ on the upcoming extension (newer prelude checkpoint), grounded in our own
sweep, the Qwen comparison, and the open metric question. Current values: `rope_theta_current.md`.

## TL;DR recommendation
| target | **recommended θ** | why |
|---|---|---|
| **128K** | **16M** | lowest value with real depth-0 signal in our sweep (90%); likely →~100% with a full budget + newer base. Meaningfully below our shipped 32M. |
| **256K** | **32M** | doubling of 16M; below our shipped 64M; plausibly sufficient (untested — was the cancelled sweep). |
| next *base* pretraining | **500k–1M** | shrinks the extension jump; standard practice (Llama 3 = 500k, Qwen = 1M); our 100k is low. |

## Why not 5M / 10M (yet)
Tempting from Qwen (10M @ 256K), but our **direct data says 8M → 0% depth-0 at 128K**, so 5–10M would
very likely fail our depth-0 bar. Qwen's 10M works because of **partial rotary + head_dim 256 + YaRN +
heavy long training** — a different geometry (`rope_theta_vs_qwen35.md`); it does not transfer to our
full-rotary / head_dim-128 model. Normalized for geometry, matching Qwen's aggressiveness on our model
is ~7.7M, which our sweep says fails.

## The one thing to settle first (cheap, decides everything)
There's an unresolved discrepancy: **your previous 128K test used θ=2M**, which our sweep puts at 0%
depth-0. Either (a) your test used a **less adversarial metric** (standard needle/RULER) → a much
lower θ is genuinely acceptable and we've held an over-strict bar; or (b) it didn't actually pass
far-position retrieval.

**Before committing θ, run one inference-only reconciliation:** eval the existing 32M/64M models *and*
your 2M / pre-summer models on **both** metrics — our adversarial depth-0 **and** a standard RULER
needle. If low-θ passes the standard bar, **use 5–10M with confidence**. This is ~free (no training)
and resolves the whole θ question.

## Nuances that could lower the floor
1. **Newer base + full training budget** ≠ the quick sweep (short budget). More adaptation tokens can
   push the θ floor down — so 16M may firm to ~100%, and even lower θ deserves a fair test.
2. **Metric bar** (above) — the biggest lever.
3. **Per-dimension scaling (LongRoPE2 / YaRN)** instead of uniform ABF preserves short-context and can
   reach the target at lower effective θ. Consider if short-context perplexity regresses.

## Suggested plan for the new run
- **Safe/grounded:** θ=16M @128K (or 32M @256K) — will likely work, lower than current.
- **Aggressive/informative:** bracket 10M **and** 16M, with depth-0 + short-context perplexity eval,
  so a lower floor is discovered rather than guessed.
- **Always eval:** depth-0 (θ-sensitive) **and** `scripts/eval_perplexity.py` short lengths
  (catches θ-over-scaling harm NIAH misses).
- **Next base:** raise base θ to 500k–1M (+ consider Qwen-style partial rotary / larger head_dim).

## θ-configurable launcher
Use `scripts/dsa/lumi/theta_sweep_256k.sbatch` (θ as env, no file edit):
```bash
sbatch --export=ALL,THETA=16000000,TAG=16M theta_sweep_256k.sbatch   # 256K @ 16M
```
For 128K, edit `--rotary-base` in `prelude_full.sbatch` (or its 128K stage). Cost: ~2k GPU-h per
1B-token 256K run — compute is not the constraint; wall-clock (~16 min/iter @256K) is.
