# LUMI Run Status — OpenEuroLLM Long-Context Data Pipeline

**Last updated:** 2026-05-05  
**Environment:** LUMI (CSC), partition `dev-g`, AMD Instinct MI250X, ROCm 7.0  
**Container:** `lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif` (PyTorch 2.10.0+ROCm7.0)  
**Project account:** `project_462000963`  
**Scratch path:** `/scratch/project_462000963/bmoell/`  
**Megatron fork:** [OpenEuroLLM/NVIDIA-Megatron-LM](https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM)  
**Tokenizer:** `openeurollm/tokenizer-256k` (vocab size = 2¹⁸ = **262,144** tokens)

---

## High-Level Goal

We are validating the full **OpenEuroLLM long-context data pipeline** on LUMI, an EU supercomputer with AMD MI250X GPUs. The goal is to prove end-to-end:

1. **Data pipeline** — `longctx estimate → download → convert → filter-long → tokenize`
2. **Megatron GPT training** — at least 5 training iterations with real loss values

This is a prerequisite before running large-scale multilingual training for the OpenEuroLLM project. The ML framework (NVIDIA Megatron-LM) is designed for NVIDIA GPUs and uses several apex fused CUDA kernels. Our challenge is making it work on AMD/ROCm hardware in the LUMI environment.

The entry point is `lumi/slurm/hello.sbatch`, submitted via `bash lumi/run-lumi.sh hello`.

---

## What Works ✅

### Data pipeline (all stages pass cleanly)

| Stage | Command | Status |
|-------|---------|--------|
| Estimate | `longctx estimate --languages mt` | ✅ |
| Download | `longctx download --sample --shards 1 --languages mt` | ✅ |
| Convert | `longctx convert` | ✅ — 16,490 docs, ~131M tokens |
| Filter | `longctx filter-long --min-tokens 4096` | ✅ — 5,463 docs kept (33%) |
| Tokenize (quick) | `preprocess_data.py` with `openeurollm/tokenizer-256k` | ✅ — bin/idx written |

The quick tokenize path writes a single-doc `.bin`/`.idx` file; full tokenization (without `--quick`) also works.

### GPU compute (pure PyTorch)

Tested on MI250X inside the Singularity container:

| Test | Result |
|------|--------|
| `torch.matmul` 1024×1024 | ✅ ~940ms first call (kernel compile), <1ms after |
| `nn.Embedding(262144, 64)` + `nn.Linear` + backward (5 iters) | ✅ ~2s first iter, ~3ms after |
| `F.scaled_dot_product_attention` (causal, math backend) | ✅ |
| `gloo dist.barrier()` with WORLD_SIZE=1 | ✅ ~108ms first call, <1ms after |
| Megatron `DotProductAttention` (QK^T, softmax, attn*V) — all 4 layers | ✅ (confirmed in job 18306279) |

### Megatron initialisation and forward pass

After applying ROCm compatibility patches (see below), all of the following now work:

- Model and optimizer build
- DataLoader build (including `--mock-data`)
- Apex extension compilation (`amp_C`, fused kernels)
- `[before the start of training step]` log line
- Full forward pass through all 4 transformer layers (attention + FFN)
- GPU sync after decoder: confirmed "GPU sync after decoder OK"
- `_postprocess` (output projection)
- `finalize_model_grads` called

---

## What Was Broken — Root Causes Found and Fixed

### Bug 1: GPU crash — wrong vocab size (FIXED)

**Symptom:** `vectorized_gather_kernel` SIGABRT (out-of-bounds embedding lookup)  
**Root cause:** The tokenizer `openeurollm/tokenizer-256k` has 262,144 vocab entries (2¹⁸), not 256,001 or 256,128 as initially assumed. Megatron's default `--vocab-size` was wrong.  
**Fix:** Added `--vocab-size 262144` to pretrain_gpt.py args in hello.sbatch.

### Bug 2: NCCL hang (FIXED)

**Symptom:** Hang immediately on process group init with "guessing device ID" warning from ProcessGroupNCCL  
**Root cause:** NCCL can't reliably detect ROCm device IDs in single-GPU Singularity container without explicit device binding.  
**Fix:** Switched to `--distributed-backend gloo` for single-node/single-GPU use.

### Bug 3: apex FusedLayerNorm hangs on MI250X (FIXED)

**Symptom:** Training hangs indefinitely with no output after `[DIAG] GPTModel: calling decoder (transformer layers)`. Diagnostic job (`gpu_diag5.sbatch`) confirmed `apex.normalization.FusedLayerNorm` hangs after 6+ minutes with only 2 lines of output.

**Root cause:** `megatron/core/fusions/fused_layer_norm.py` calls `FusedLayerNormAffineFunction.apply()` from `apex.normalization` when `hidden_size` is not in the supported list (256 is not supported; the list starts at 1024). This apex CUDA kernel deadlocks on MI250X.

**Fix applied to LUMI Megatron source** (`megatron/core/fusions/fused_layer_norm.py`):

```python
# Replaced this:
return FusedLayerNormAffineFunction.apply(
    input, weight, self.bias, self.hidden_size, self.eps
)

# With this (in the else branch of forward()):
# Use PyTorch native layer_norm (apex FusedLayerNormAffineFunction hangs on MI250X)
return torch.nn.functional.layer_norm(
    input, self.hidden_size, weight, self.bias, self.eps
)
```

After this fix, all 4 transformer layers run through successfully including attention.

### Bug 4: gloo doesn't support allreduce_coalesced for CUDA tensors (FIXED)

**Symptom:** Crash after backward pass with `RuntimeError: ProcessGroupGloo::allreduce_coalesced: unsupported device type cuda`

**Root cause:** Megatron's DDP gradient sync calls `_coalescing_manager` which batches multiple `all_reduce` calls and sends them as `allreduce_coalesced`. The gloo backend doesn't support this operation for CUDA tensors.

With WORLD_SIZE=1 (single GPU), no actual gradient synchronization is needed — the all-reduce is a no-op.

**Fix applied to LUMI Megatron source** (`megatron/core/distributed/param_and_grad_buffer.py`):

Added an early return in `start_grad_sync()` when world size is 1:

```python
# Skip all-reduce with single process - gloo does not support
# allreduce_coalesced for CUDA tensors, and it is a no-op with 1 rank.
import torch.distributed as _dist
if _dist.get_world_size(group=self.data_parallel_group) == 1:
    return
```

---

## Current Status

**Job 18306279** is running with both Bug 3 and Bug 4 fixes applied. Awaiting results to confirm 5 training iterations complete with loss values.

### Previous jobs and what they showed

| Job | Purpose | Result |
|-----|---------|--------|
| 18277969 | First hello.sbatch | Hung at decoder (FusedLayerNorm) |
| 18303968 | gpu_diag5 — apex component isolation | Confirmed FusedLayerNorm hangs after 6+ min |
| 18305680 | First fix attempt (FusedLayerNorm only) | Got past decoder, crashed at allreduce_coalesced |
| 18306279 | Both fixes | Running — awaiting results |

---

## Pending Next Steps

1. **Confirm training completes** — job 18306279 should print 5 iteration loss values
2. **Clean up Megatron diagnostic patches** — `git checkout` all patched files (training.py, schedules.py, gpt_model.py, dot_product_attention.py)
3. **Switch from `--mock-data` to real data** — use `--data-path $TRAIN_DIR/mt_train_text_document`
4. **Verify loss decreases** over 5 iterations
5. **Commit the final working configuration** to GitHub, including the two ROCm compatibility patches

---

## ROCm Compatibility Patches Summary

Two source files in the Megatron repo need patches for MI250X/ROCm 7.0:

### Patch 1: `megatron/core/fusions/fused_layer_norm.py`

In `FusedLayerNorm.forward()`, replace the `else` branch (non-persist path, used when `hidden_size` is not in the supported list) to use `torch.nn.functional.layer_norm` instead of `apex.normalization.FusedLayerNormAffineFunction`.

### Patch 2: `megatron/core/distributed/param_and_grad_buffer.py`

In `BucketGroup.start_grad_sync()`, add an early return when `dist.get_world_size(group=self.data_parallel_group) == 1`. This avoids the `allreduce_coalesced` call that gloo doesn't support for CUDA tensors.

---

## Current hello.sbatch Configuration

Relevant Megatron flags in Section 4 (tiny training run):

```bash
--num-layers 4 --hidden-size 256 --num-attention-heads 4
--seq-length 512 --max-position-embeddings 512
--micro-batch-size 1 --global-batch-size 1 --train-iters 5
--mock-data --tokenizer-type NullTokenizer --vocab-size 262144
--distributed-backend gloo
--transformer-impl local
--no-gradient-accumulation-fusion --no-bias-dropout-fusion --no-masked-softmax-fusion
--num-workers 0
```

ROCm env vars set before the training singularity exec:
```bash
export MIOPEN_USER_DB_PATH=$WORKDIR/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$WORKDIR/miopen_cache
export MIOPEN_FIND_ENFORCE=NONE
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export ROCR_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1; export RANK=0; export LOCAL_RANK=0
```

---

## File Layout

```
lumi/
├── run-lumi.sh          # top-level runner: hello / status / logs
├── slurm/
│   ├── hello.sbatch     # main end-to-end job (data pipeline + tiny training)
│   ├── gpu_diag.sbatch  # GPU matmul + tiny PyTorch training (all pass)
│   ├── gpu_diag2.sbatch # SDPA + DotProductAttention isolation tests
│   ├── gpu_diag3.sbatch # Megatron with stack-trace timeout wrapper
│   ├── gpu_diag4.sbatch # gloo barrier + Megatron timer tests
│   └── gpu_diag5.sbatch # apex component isolation (found FusedLayerNorm hang)
└── LUMI_STATUS.md       # this file
```
