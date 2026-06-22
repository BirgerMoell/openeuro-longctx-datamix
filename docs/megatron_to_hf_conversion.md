# Megatron → HF Conversion (baby_9b_dense / Qwen3) on LUMI

Two routes exist; both produce a correct Qwen3 HF checkpoint from a Megatron `torch_dist` checkpoint.

## Route A — Joan Llop's official guide (reference)
OpenEuroLLM HPC guide (Leonardo-oriented), kept for reference:
**https://github.com/OpenEuroLLM/hpc-guides/blob/main/convert_baby_9B_checkpoint_from_megatron_to_hf_LEO.md**
(Private repo — needs OpenEuroLLM org access. If you can read it, paste the steps here so they're
captured offline.)

## Route B — our validated converter (used for the 128K extended model)
Built from `BirgerMoell/megatron-hf-converter` + a custom **Qwen3 saver** + loader patches,
because the stock converter only had a Llama saver (which silently drops Qwen3's qk-layernorm).

**Location on LUMI:** `/scratch/project_465002530/users/bmoell/conv/`
- `checkpoint/` — patched copy of Megatron's `tools/checkpoint`
- `checkpoint/saver_qwen3.py` — Qwen3 saver (emits `q_norm`/`k_norm`, `Qwen3Config`)
- `convert_baby_validate.sbatch` / `convert_extended_128k.sbatch` — runners
- `cmp_baby.py` — numerical validation vs a reference HF model

**Validation:** converting the baby and diffing vs Joan's known-good baby HF
(`/flash/project_462000963/training/qwen3_9b_hf_baby_ckpts/hf/iter_0076800`) gave **397/399
tensors bit-identical (max abs diff 0.0), all 72 q_norm/k_norm correct.** The only diff was
benign vocab padding (262656 vs 262272). → converter proven correct.

### The 8 fixes needed (Megatron version drift + Qwen3)
1. `--saver qwen3` is a module name (`saver_qwen3.py` on PYTHONPATH), not a file path.
2. `export CUDA_DEVICE_MAX_CONNECTIONS=1` (TP/CP requires it).
3. `loader_core.import_model_provider` must `return self.model_provider` (the `gpt_builder`-bound
   partial) — newer `model_provider(model_builder, ...)` signature.
4. Set fake DP/PP/CP/embedding/expert process groups (newer `GPTModel.__init__` needs them).
5. Init a single-process `torch.distributed` group (gloo) — torch_dist load needs it.
6. **Force TP=PP=CP=EP=1 after `load_args_from_checkpoint`** — torch_dist is parallelism-agnostic,
   so load at TP=1 and let `dist_checkpointing` reshard the saved shards into whole tensors.
   (Stock loader's per-TP-shard iteration is for legacy `mp_rank_*` checkpoints, not torch_dist.)
7. `--tokenizer-dir` must contain `tokenizer.model` (SentencePiece) — use the canonical
   `…/pyysalos/tokenizers/openeurollm/tokenizer-256k`, not the baby HF dir (json-only).
8. Saver epsilon: `getattr(mag_conf,'norm_epsilon',None) or getattr(mag_conf,'layernorm_epsilon',1e-5)`
   (arg name differs between the baby's and the extend run's Megatron).

### Run
```bash
cd /scratch/project_465002530/users/bmoell/conv
sbatch convert_extended_128k.sbatch     # ckpt_131072 -> extended_hf_128k
```
Output: `/scratch/project_465002530/users/bmoell/longctx-extend/extended_hf_128k/`
(config: qwen3, max_position_embeddings 131072, rope_theta 5000000, head_dim 128, 36 layers).

The saver auto-reads `rotary_base` and `max_position_embeddings` from the checkpoint, so the
same script converts any stage checkpoint (16K/32K/64K/128K).
