# Long-Context Extension — Progress (baby_9b_dense on LUMI)

**Updated:** 2026-06-18

## TL;DR
The core long-context extension mechanic is **validated on LUMI**: the `baby_9b_dense`
(Qwen3 dense ~9B) checkpoint loads on LUMI/ROCm, extends from 4K to 16K via native ABF,
and produces a **sane loss (~2.24)** — stable, no NaNs, fits comfortably. A **tiny
end-to-end pipeline** (16k→32k→64k→128k) is now running to validate the full staged path.

## The model
Qwen3 dense (NOT 3.5), `Qwen3ForCausalLM`, ~9.1B params: 36 layers, hidden 4096, FFN 12288,
32 heads / 8 KV (GQA), head_dim 128, qk-layernorm, RMSNorm, SwiGLU, untied embeddings,
vocab 262272 (OpenEuroLLM 256k tokenizer), **plain RoPE θ=100000, native 4K**, full attention.

## Checkpoint locations
- **LUMI Megatron `torch_dist` (complete, 119 GB):**
  `/flash/project_462000963/training/qwen3_9b_hf_baby_ckpts/megatron/iter_0076800` (copied by joan.llop)
- **LUMI HF/transformers (17 GB):** `.../qwen3_9b_hf_baby_ckpts/hf/iter_0076800`
- **HF backup (private):** `birgermoell/baby_9b_dense-iter76800-megatron` (full, with model card)
- Source of truth: Leonardo `production_training/baby_9b_dense/checkpoints/iter_0076800` (baby run ongoing, ~644B/10T tokens)

## Extension recipe that WORKS (native ABF)
- Load the baby Megatron checkpoint with **`--finetune`** (loads weights, resets optimizer + iteration)
- Raise **`--rotary-base`** for the target length (ABF) — no YaRN/LongRoPE code needed
  (those aren't in the OELLM Megatron yet, only Jouni's personal repo)
- `--ckpt-format torch_dist` is **parallelism-agnostic** → load at any TP/PP/CP (no conversion)
- Keep the exact baby arch args (qk-layernorm, GQA 8, kv-channels 128, untied, RMSNorm, SwiGLU)
- Tokenizer must be the OpenEuroLLM 256k tokenizer (token-ID compatible with the long-ctx data)

## Stage 1 result (16K validation smoke — job 19338222, 1 node, TP=8)
- Baby loaded fine (TP=4-saved → TP=8 reshard worked on ROCm)
- **lm loss ~2.24** at θ=500000, 16K — sane (≈ a trained model), 0 NaN / 0 skipped
- Memory: ~34 GB / 64 GB per GPU (weights+opt ~19.5 GB, activations ~10.7 GB)
- Ran all iters → "after training is done"

## Memory-informed parallelism (per stage, 4 nodes / 32 GPUs, TP=8)
| Stage | seq | rotary-base | CP | fits |
|------|-----|-------------|----|------|
| 1 | 16384  | 500000   | 1 | ✓ (1 node sufficient) |
| 2 | 32768  | 1000000  | 1 | ✓ |
| 3 | 65536  | 2000000  | 2 | ✓ (matches Jouni's working 64k) |
| 4 | 131072 | 5000000  | 4 | the test — Jouni's 128k has been failing on LUMI |

## Current run
Tiny full pipeline (`extend_pipeline.sh`, job 19338612, 4 nodes, dev-g): runs all four
stages back-to-back, each `--finetune`-loading the previous stage's checkpoint, ~8 iters,
saving `output_pipeline/ckpt_<seq>`. 16k/32k/64k expected to pass; 128k is the real
experiment (will either work at tiny scale or pinpoint the failure for fixing).

## Where the LUMI work lives (Birger's space, separate from Jouni's)
`/scratch/project_465002530/users/bmoell/longctx-extend/`
- `extend_16k_smoke.sh` — stage-1 validation (adapted from Jouni's `train-gpt-long.sh`)
- `extend_pipeline.sh` — the tiny 4-stage pipeline
- `launch.sh` — per-rank launcher (copied from Jouni)
- Container: `laif-rocm-6.4.4-...sif`; Megatron: Jouni's `oellm-test/NVIDIA-Megatron-LM`
- Long-ctx data available: `/scratch/project_465002530/preprocessed/oellm-v1-256k/long-ctx-sample/`

## Key constraints / lessons
- **Leonardo strategic allocation is baby-run-only** → all extension compute runs on LUMI.
- Native ABF (raise θ) avoids the missing YaRN/LongRoPE code path.
- HF `upload_large_folder` of a 2000+ file checkpoint hits the 128-commits/hour limit and
  stalls — direct cluster copy is far better for big checkpoints.
- 128K on LUMI is unproven (Jouni: 64k works, 128k failing) — the tiny pipeline is meant to
  surface exactly where it breaks.

## Next
1. Read tiny-pipeline result; if 64k passes and 128k fails, debug the 128k CP/memory config.
2. Scale the winning per-stage recipe to real token budgets (~0.5–2B/stage) with the
   long-ctx data mix; save intermediate checkpoints (16k/32k/64k/128k).
3. Stand up RULER / OneRuler eval at each length.
