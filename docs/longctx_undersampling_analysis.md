# Long-Range Undersampling Analysis (why 128K depth-0 retrieval is weak)

**Date:** 2026-06-22. Diagnoses the 64K→128K retrieval degradation in the baby_9b_dense
extension and motivates the **length-biased data mix** fix.

## Symptom
Base-LM NIAH (forced-choice) on the extended model:
- 4K = 100%, 16K = 100%, **64K = 86%**, **128K = weak at depth-0** (needle at the far start of
  the window): depth-0 ≈ 0%, mid/late depths fine. So the model can use *recent* long context
  but struggles to retrieve from the *far end* of a 128K window.

## Root cause: the long-range signal was undersampled
Long *documents* exist in the data — but Megatron packs tokens into fixed-length sequences. A
64K/128K sequence built from many short docs has **no genuine long-range dependency** (it's
unrelated short docs concatenated). What matters is the fraction of training tokens that live in
documents **≥ the stage's sequence length**.

### Document-length distribution (measured, catalogue sources)
% of a source's tokens living in docs ≥64K / ≥128K:

| source | %tok ≥64K | %tok ≥128K |
|---|---|---|
| finepdfs (eng/deu/fra/spa, avg) | ~37% | ~25% (some langs 31%) |
| finepdfs-edu (eng) | 33% | 22% |
| starcoder (code, file-level) | 12% | 8% |
| finemath | 3.4% | 2.1% |
| nemotron-cc (web) | 3.2% | 1.3% |

finepdfs is rich in long docs (good!); code/math/web are mostly short and **dilute** the mix.

### Effective long-range fraction in the ACTUAL training mix
Weighting the above by the run's mix weights:

| Stage | % of mix tokens in docs ≥ seqlen | packed-shorts (no long-range) | **genuine-long tokens trained** |
|---|---|---|---|
| 64K  | **28.4%** | 71.6% | 2.0B budget × 28% = **~0.57B** |
| 128K | **18.5%** | 81.5% | 1.0B budget × 19% = **~0.19B** |

### Why this explains the degradation
- 64K saw **~0.57B** genuine full-window tokens → 86% retrieval (decent).
- 128K saw only **~0.19B** — **3× less** — at the harder, longer span → weak depth-0.
- **Double hit at 128K:** smaller budget (1B vs 2B) *and* lower long-range fraction (19% vs 28%),
  because ≥128K docs are rarer AND diluted by ~30% short-doc sources (math/code/web).

## Fix: length-biased data mix (+ blend to keep short-context)
`scripts/build_longctx_mix.py` extracts long docs (diverse, range-capped) into new `.bin/.idx`:
- 64K stage: docs in [64K, 512K]; 128K stage: docs in [128K, 1M] (cap avoids monster docs
  dominating → many distinct long docs, less overfit).
- **Blend ~65% length-biased + ~35% original mix** at each stage — raises genuine-long from
  28%→~65% (64K) and 19%→~65% (128K) **without** nuking short-context (pure-long training
  causes short-context regression).
- Run as a **curriculum**: `ckpt_32768 → improved 64K → improved 128K` (a stronger 64K
  foundation compounds into 128K). 16K/32K already 100%, so not redone.

Expected: ~0.19B → multiple-B genuine ≥128K tokens at the final stage → much stronger depth-0.

## Scripts
- `scripts/build_longctx_mix.py` — the length-biased sampler (Megatron builder, validated).
- `scripts/undersample_analysis.py` — recomputes the effective-fraction table from any mix.
