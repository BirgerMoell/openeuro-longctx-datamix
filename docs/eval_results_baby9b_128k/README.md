# NIAH Eval Results — baby_9b_dense extended to 128K

**Model:** `baby_9b_dense` (Qwen3 9B) extended 4K→128K (all 37 OELLM languages), converted to HF
(`extended_hf_128k`, rope_theta 5e6, max_position 131072). See
[`../megatron_to_hf_conversion.md`](../megatron_to_hf_conversion.md) and
[`../RUN_baby9b_longctx_128k.md`](../RUN_baby9b_longctx_128k.md).

**Eval:** `scripts/eval_base_lm_niah.py` — base-LM **forced-choice log-likelihood** NIAH
(no instruction tuning needed). 4 candidate magic-numbers (all present in context, bound to
different keys) → model must read+bind the query key→value. **Chance = 25%.**
**Jobs:** `19415715` (full grid, dev-g), `19418402` (128K-only, dev-g) — both on LUMI 2026-06-22.

## Results so far (French — completed cells)
| Context | Accuracy | Notes |
|--------:|:--------:|-------|
| 4K   | **100%** (50/50) | perfect |
| 16K  | **100%** (50/50) | perfect |
| 64K  | **86%** (43/50)  | strong retrieval at 64K |
| 128K | (in progress)    | only depth-0 trials so far; full run on standard-g (`19419241`) |

French 4K→64K is a solid, citable result: **strong long-context retrieval holding to 64K**,
far above the 25% chance baseline — demonstrating the ABF extension worked. (For contrast, the
earlier YaRN v2-32k model degraded by 32K; see [`../ruler_eval_yarn_v2_32k.md`](../ruler_eval_yarn_v2_32k.md).)

## Status
- **128K full table (fr/fi/cs/nl):** running on standard-g, job `19419241` (12h, 10 trials/cell).
  128K is slow (~4 min/trial: the scorer re-runs the full 128K forward per candidate).
- **fi/cs/nl at 4K/16K/64K:** continuing under job `19415715`.
- Results stream to per-trial JSONL (durable); recompute any time.

## Files
- `raw_jsonl/fr_results.jsonl` — full-grid French (4K/16K/64K + start of 128K)
- `raw_jsonl/fr_results_128k_only.jsonl` — focused 128K-only French trials
- (more languages + the standard-g 128K results to be added as they finish)

## Reproduce / extend
```bash
# on LUMI
cd /scratch/project_465002530/users/bmoell/longctx-extend
sbatch --export=ALL,LANGS="fr fi cs nl",CTX="4096 16384 65536 131072",TRIALS="10" \
       eval_niah_extended.sbatch
```
Model: `/scratch/project_465002530/users/bmoell/longctx-extend/extended_hf_128k`.
Adding more of the 37 languages = add their needle templates in `scripts/eval_base_lm_niah.py`.

## TODO / known speedup
The scorer recomputes the long prefix forward once **per candidate** (4×). Caching the prefix
KV once and scoring all 4 candidate tails would be ~4× faster — worth doing for 128K evals.

## v1 128K performance — current/final (depth-stratified)

The definitive v1 baseline (to compare the length-biased v2 against). Base-LM forced-choice
NIAH, 4-way, chance 25%. 217 trials across cs/fi/fr (+nl just starting). Job 19419241.

**Overall: 57% (123/217).**

**By depth (the key signal — very stable):**
| needle depth (position in window) | accuracy |
|---|---|
| 0.0 (far start / oldest) | **0%** (0/55) |
| 0.25 | 30% (15/50) |
| 0.5 (middle) | 93% (38/41) |
| 0.75 | 97% (35/36) |
| 1.0 (end / most recent) | 100% (35/35) |

**By language:** cs 68% (34/50), fi 56% (37/66), fr 54% (52/96). (nl 0/5 — only depth-0 so far.)

**Read:** clean *recency gradient* — **near-perfect retrieval for the back ~60% of the 128K
window (depth ≥0.5 = 93–100%), total failure at the far front (depth 0 = 0%).** Consistent
across languages. This is the undersampling fingerprint (only ~0.19B genuine ≥128K tokens
trained). The model genuinely *uses* a 128K window for recent/mid content but can't anchor the
oldest quarter.

**v2 success metric = lifting depth-0 (0%) and depth-0.25 (30%)** via the length-biased data
(Jouni's tiered sample) + bigger 128K budget. Back-half is already saturated.
prelude-128k final: 96% (864/900), depth-0 97%, depth-0.5 88% (lost-in-middle); vs v3 ~100%
