# Compute Budget — Long-Context Extension

**Date:** 2026-06-17
**Question:** How much compute does the real long-context extension need, and is it
worth spending compute on validation experiments first?

---

## Measured anchor (do not re-derive — this is real data)

From the validated 32K 2-node smoke test (job `45790387`, 8× A100-64GB, TP=4 PP=1 DP=2,
SP, GBS=2, full activation recompute, distributed optimizer):

| Metric | Value |
|--------|-------|
| Sustained throughput | **105 TFLOP/s/GPU** (model FLOPs, ~34% MFU) |
| Tokens/s/GPU @ 32K | **1415** |
| Aggregate (8 GPUs) | ~11,300 tok/s |
| Step time (GBS=2 × 32768) | ~5.8 s |
| GPU memory used | **53%** — large headroom |

Cross-check: 65,536 tok/step ÷ 5.8 s = 11,299 tok/s ✓

---

## Throughput vs context length

Per-token model FLOPs: `F(L) = 3.71e10 × (1 + L/32768)`. At 32K the dense (MLP + proj)
and attention-score terms are roughly equal, so cost grows sub-quadratically but faster
than linear as L increases. Holding sustained model-FLOP rate at 105 TFLOP/s/GPU:

| Context | tok/s/GPU | GPU-h / 1B tok | node-h / 1B tok | local-h / 1B tok |
|--------:|----------:|---------------:|----------------:|-----------------:|
| 2K   | 2665 | 104 |  26 |    832 |
| 4K   | 2517 | 110 |  28 |    880 |
| 8K   | 2264 | 123 |  31 |    984 |
| 16K  | 1887 | 147 |  37 |  1,176 |
| 32K  | 1415 | 196 |  49 |  1,568 |
| 64K  |  943 | 295 |  74 |  2,360 |
| 128K |  566 | 491 | 123 |  3,928 |
| 256K |  314 | 885 | 221 |  7,072 |

Units: 1 node = 4 GPU; Leonardo Booster bills ~32 local-h per node-hour (≈ 8 local-h/GPU-h).

**Uncertainty: ±30%.** Sustained MFU may dip at 256K (attention becomes bandwidth-bound)
or if memory forces smaller GBS / more recompute / more parallel nodes. Add ~10-15% on
top for RULER eval + checkpoint I/O.

---

## Cost of the real extension (stages 5-7: 64K → 128K → 256K)

Roadmap allots 500M-2B tokens per stage. Current model is already at 32K, so only three
new stages remain.

| Token budget/stage | 64K | 128K | 256K | **Total GPU-h** | node-h | local-h |
|-------------------:|----:|-----:|-----:|----------------:|-------:|--------:|
| 0.5B | 147 | 246 | 442 | **836**   | 209 |  6,700 |
| 1B   | 295 | 491 | 885 | **1,671** | 418 | 13,400 |
| 2B   | 590 | 982 | 1770| **3,342** | 836 | 26,700 |

256K alone is ~half the cost. If long 256K documents are scarce (open data question),
running fewer tokens there is the main lever.

---

## Cost of the validation experiment (the "should I even experiment?" answer)

Phased-extension hypothesis test: 2K → 4K → 8K → 16K → 32K, 500M tokens/stage = 2.5B tok,
compare per-stage RULER scores vs a single-jump baseline. All stages are at **short
context, where compute is cheap**:

| Item | GPU-h | node-h | local-h | % of 1B/stage real extension |
|------|------:|-------:|--------:|-----------------------------:|
| Validation experiment (500M/stage) | 340 | 85 | 2,720 | **~20%** |
| Validation experiment (200M/stage) | 136 | 34 | 1,088 | ~8% |
| One smoke test (e.g. 64K, 5 iters) | ~1 | 0.25 | 8 | negligible |
| RULER baseline on 32K v2 model | ~5-20 | 1-5 | tens | negligible |

---

## The reframe: compute is not the binding constraint

Remaining 2026 budget: **~8.86M local-h ≈ 277,000 node-h ≈ 1.1M GPU-h.**

| Program | local-h | % of remaining budget |
|---------|--------:|----------------------:|
| Validation experiment (500M/stage) | 2,720 | **0.03%** |
| Real extension, 1B/stage | 13,400 | **0.15%** |
| Real extension, 2B/stage | 26,700 | **0.30%** |
| Everything above combined | ~17,000 | **<0.2%** |

The entire long-context program — experiments *and* the real 256K extension — costs
**under 0.4% of the remaining allocation**. Smoke tests are rounding error (~8 local-h).

**Conclusion:** The chicken-and-egg fear is misplaced. You are not trading experiment
compute against real-run compute — both fit trivially. Spending ~0.03% to run the
validation experiment de-risks the design decision (phased vs single-jump) that governs
the *whole* extension. The real constraints are:

1. **Data** — availability of long (>128K) multilingual documents (fine-PDFs audit).
2. **Wall-clock / scheduling** — queue time and the 256K memory footprint, not budget.
3. **Eval validity** — multilingual RULER (OneRuler) not yet validated.

Recommendation: run the cheap experiments. Compute is abundant; design certainty and
data are what's scarce.
