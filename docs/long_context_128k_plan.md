# Long-Context Extension to 128K — Best-Practice Plan

**Date:** 2026-06-17
**Target:** 128K sequence length (revised down from 256K — the team is targeting 128K).
**Owners:** Birger (Leonardo) + Jouni Luoma (LUMI), long-context group.

---

## UPDATE 2026-06-17 (PM) — assets located, most blockers resolved

After locating the real assets, the picture is much simpler than the original blocker list
below. Corrections:

### The extension TARGET model = `baby_9b_dense` (NOT the Llama `oellm-9b-yarn`)
- **Qwen3 dense ~9B**: `num_layers=36, hidden=4096, ffn=12288, heads=32, num_query_groups=8
  (GQA), RMSNorm, SwiGLU, rope, rotary_base=100000, rope_scaling_factor=8.0, seq_length=4096
  native`.
- The earlier `oellm-9b-yarn-v2-32k` (Llama arch, MHA, 32 layers, HF format) is a **separate,
  older v2 model** — not the long-context target. Same 256k tokenizer though.

### Checkpoint — ALREADY on Leonardo, Megatron format (blocker #1 GONE)
- `/leonardo_work/OELLM_prod2026/production_training/baby_9b_dense/checkpoints/` — latest
  `iter_0076800`, **119 GB, `torch_dist` format**, readable. Baby run still progressing
  (10T-token target, `10T_baby_datamix.txt`).
- **`torch_dist` is parallelism-agnostic** → loadable at any TP/PP. **No HF→Megatron and no
  PP conversion needed.** The `megatron-hf-converter` is therefore only needed for exporting
  the final model to HF, not for training.

### Production training script — reuse it (blocker #2 GONE)
- `…/baby_9b_dense/job.sbatch` runs in container
  `/leonardo_work/OELLM_prod2026/container_images/nemo_25.11.01.sif` (NeMo 25.11.01 — has TE +
  Apex + flash-attn + all fused kernels). Uses TP=4 PP=1, GBS=2048 MBS=4, ckpt_format
  torch_dist, tokenizer at `…/baby_9b_dense/tokenizer`.
- **This replaces the bare-venv / legacy-model / `--no-*-fusion` workarounds entirely.** The
  32K smoke pipeline was a throwaway memory probe on a different (slower, unfused) path.
  Throughput in the NeMo container will be *higher* than the 105 TFLOP/s anchor → compute
  estimates below are conservative.

### Data — EXISTS, Megatron-ready (blocker #3 largely GONE)
- HF: `birgermoell/oellm-longctx-tokenized-streamed-all-v2` — bucketed `under4k / 4_16k /
  16k_plus`, run `lc16k_full_20260507` (9931 files) + smoke; ships `mix/data_path.args`
  (tier weights 0.5 / 0.3 / 0.2) and `mix/data_mix.json`. Tokenized with the 256k tokenizer.
- LUMI: Jouni's `…/oellm-v1-256k/long-ctx-sample` (~120 GB, 3-tier length-biased: short <16k
  60%/medium 16–64k 25%/long ≥64k 15%, ~30B-token target). Incomplete (largest files timed
  out). Designed for Qwen3 → 64k.

### What actually remains
1. **Write `extend_<len>.sbatch`** = copy `baby_9b_dense/job.sbatch`, then: `--finetune`
   (load weights only, reset optimizer + iteration), `--load` baby ckpt, `--save` to a NEW
   dir (do **not** overwrite the baby run), `--seq-length`/`--max-position-embeddings` =
   step length, **bump `--rotary-base` (ABF)** per step, `--data-path` from the long-ctx mix,
   small `--train-iters` + reduced GBS.
2. **Bucketing for >16k**: the HF data tops out at a `16k_plus` bucket; for 64k/128k steps,
   re-bucket / re-sample to expose 64k+ and 128k+ tiers (Jouni's LUMI sampler already uses a
   64k long-threshold — extend it to 128k).
3. **Verify tokenizer identity** between the baby run tokenizer and the data tokenizer (both
   "256k" — confirm vocab/merges match so token IDs align).
4. **Memory at 64k/128k** in the NeMo container (different profile than the smoke probe);
   pick TP/PP/recompute accordingly (torch_dist makes this free to change).
5. **RULER + OneRuler** eval pipeline — still to build.
6. **128k data sufficiency** — still the real long-term risk (natural docs rarely reach 128k).

The original blocker list and method/data/compute analysis below remain valid; treat this
UPDATE as the authoritative current state.

---

## Decision summary

| Question | Decision |
|---|---|
| Target length | **128K** |
| Stepping | **Progressive 4× steps** (Jouni): 4K → 16K → 64K → 128K (short finish) |
| Method A/B | **Two tracks** run in parallel, compared on RULER |
| Default to ship if A/B inconclusive | **Native ABF** (θ-scaling + long-data CPT) — the industry default (Llama-3.1, Qwen2.5, DeepSeek) |
| Compute concern | **Resolved — not the bottleneck** (see budget) |

### Two tracks (as agreed with Jouni)
- **Track A — Leonardo / Birger / YaRN.** Continue from the existing `oellm-9b-yarn-v2-32k`
  (already YaRN factor 16 → 32K). Bump YaRN factor 16 → 32 (64K) → 64 (128K), short CPT each.
  Lowest-effort path; reuses the validated 32K Leonardo pipeline.
- **Track B — LUMI / Jouni / native ABF.** From an early baby-run checkpoint. Increase RoPE
  base θ, brute-force CPT on a long-context datamix, no YaRN. Steps 4K → 16K → 64K → 128K.
- **Cheap hybrid** (fallback / third option): native to 64K, then a short 2× YaRN to 128K.

---

## Why these methods (grounded in the literature, not memory)

- **RoPE base (θ) increase = the dominant lever for native extension** ("ABF"). For 128K, θ
  must be large enough that the lowest frequency completes <1 rotation over 128K (Llama-3.1
  uses θ=500000; longer targets use millions). The current model has θ=10000 + YaRN scaling.
- **YaRN is more token-efficient** (Qwen2) but carries an inference-time scaling config that
  must be propagated through deployment *and* post-training (SFT/GRPO). Native ABF has no such
  baggage — cleaner long-term.
- **Data recipe matters more than the exact stepping** (Fu et al. 2024, "Data Engineering for
  Scaling LMs to 128K Context"): **keep the pretraining domain mixture, and upsample long
  documents *within* each domain** (per-source length upsampling). Do NOT rotate the domain mix
  toward whatever happens to be long. ~1–5B tokens of CPT reliably reaches 128K.
- **Progressive vs single-jump**: single-jump works with the right data (Fu et al.); Llama-3.1
  used 6 progressive stages. Progressive is safer and, since compute is cheap here, the
  de-risking is free. Jouni's 4× steps are a sound middle ground.

---

## Token budget & stepping

Per step: **0.5–1B tokens**, then RULER at that length **and all shorter lengths** (regression
check). Final 128K step can be short (Jouni's "finishing touch"). A regression at a shorter
length after extending = add more short-sequence data to the next step's mix.

---

## Compute (128K target) — NOT the bottleneck

Anchored on the measured 32K run (1415 tok/s/GPU @ 105 TFLOP/s/GPU, A100-64GB). See
`compute_budget_long_context.md` for the full model. Per-step cost:

| Step | node-h / 1B tokens |
|-----:|-------------------:|
| 4K   |  28 |
| 16K  |  37 |
| 64K  |  74 |
| 128K | 123 |

| Scenario | GPU-h | node-h | local-h | % of remaining 2026 budget |
|---|--:|--:|--:|--:|
| Track B native (1B/step, 0.5B@128K) | ~800 | 200 | 6,400 | 0.07% |
| Track A YaRN (32K→64K→128K, 1B/step) | ~980 | 245 | 7,850 | 0.09% |
| **Both tracks combined (generous)** | ~1,800 | 450 | 14,400 | **0.16%** |
| A/B at reduced 300M/step | ~600 | 150 | 4,800 | 0.05% |

Remaining 2026 allocation (`OELLM_prod2026`): **~8.86M local-h ≈ 277,000 node-h**. The entire
128K program — both tracks, full token budget — is **under 0.2%**. The "experiments cost as
much as the real run" worry is false: experiments use fewer tokens, run partly at cheap short
context, and the A/B shares its short-context steps. **Compute is abundant; data is scarce.**

---

## What is missing / blockers (prioritized)

1. **No loadable checkpoint yet.** The Leonardo "model" (`$WORK/models/oellm-9b-yarn-v2-32k`)
   is **HuggingFace format** (`pytorch_model.bin`), not Megatron. `$WORK/checkpoints/` is empty.
   → Need **HF → Megatron conversion** (and PP layout choice) before any Leonardo CPT.
   The LUMI YaRN-v2 1k Megatron checkpoint (PP=4) also needs PP=4→PP=2/1 conversion if used.

2. **Config mismatch for real continuation.** The real model is **MHA (32 KV heads, no GQA)**,
   **tied embeddings**, **θ=10000**, YaRN (factor 16, orig_max_pos 2048, mscale 1.277). The
   smoke-test sbatch uses GQA / untied / θ=500000 — correct for random-init memory tests, but
   **must be fixed** to continue the actual model. A dedicated `train_*_continue.sbatch` is
   needed with the matching architecture + the target step's YaRN/θ.

3. **Long-context data not ready.**
   - Jouni's filtered long-context sample (LUMI
     `/scratch/project_465002530/preprocessed/oellm-v1-256k/long-ctx-sample`) is **incomplete**
     — the largest files timed out (exactly the long docs we most need).
   - Need: finish filtering on the **actual** data (not just samples); **sample from the baby
     data**; **tune proportions for the 128K target**; tokenize with the matching tokenizer;
     transfer to Leonardo `$WORK/data/` (currently only a 28 MB smoke corpus).

4. **128K memory footprint unvalidated on Leonardo.** 64K smoke is running now (job 47078190,
   2 nodes). 128K will likely need 4 nodes (draft `train_32k_test_leonardo_4nodes.sbatch`
   exists as a starting point: TP=4, PP=2, DP=2).

5. **No RULER eval pipeline** on either cluster. Needed to compare tracks. Standard RULER for
   English; **OneRuler** (BirgerMoell/OneRuler-OELLM) for the 35 languages. Base-LM
   log-likelihood scoring (no instruction tuning).

6. **128K data sufficiency is the real risk.** Natural docs rarely reach 128K tokens (a
   300-page book ≈ 100K tokens). Strategy: use the longest docs (books, legal, manuals, code
   repos), and **document packing** to fill 128K where natural docs fall short. Multilingual
   long docs are scarce and EN-skewed — keep the 35 languages alive via short/medium docs; use
   synthetic long sequences only if a language has no natural coverage.

---

## Recommended order of operations

1. **Unblock the pipeline (no real data needed):**
   - Finish 64K (running) + run a **128K memory smoke** (4 nodes) to lock the parallel layout.
   - Build the **HF→Megatron conversion** for `oellm-9b-yarn-v2-32k`; verify a 1-step
     load-and-train (loss ~ sane, not 12.x random-init) — this proves the *real* model trains.
   - Write `train_continue_*.sbatch` with the **correct** arch (MHA, tied, θ, YaRN/step).
2. **Data (parallel, Jouni leading):** finish long-context filtering on full data, sample baby
   data, set 128K proportions, tokenize, transfer a first real shard to Leonardo.
3. **RULER baseline:** stand up RULER + OneRuler; baseline the 32K model at 4K/16K/32K.
4. **Run the A/B** at reduced token budget (300–500M/step) to pick the method, then scale the
   winner to the full budget.

**Bottom line:** Have the pipeline + scripts ready to run on a small real-data subset before
vacations (Birger's stated goal). The scale-up is then just more tokens — and the budget
easily covers it.
