# Leonardo 32K Smoke Test — Debug Log

**Date:** 2026-06-11  
**Goal:** Verify 10-step Megatron-LM training at `seq_len=32768` on Leonardo (1 node, 4× A100-SXM-64GB).  
**Script:** `lumi/slurm/train_32k_test_leonardo.sbatch`

---

## Architecture

9B model, TP=2, PP=2, 4 GPUs, GBS=4, MBS=1:

| Parameter | Value |
|-----------|-------|
| `--num-layers` | 32 |
| `--hidden-size` | 4096 |
| `--num-attention-heads` | 32 |
| `--num-query-groups` | 8 (GQA) |
| `--ffn-hidden-size` | 14336 |
| `--max-position-embeddings` | 32768 |
| `--seq-length` | 32768 |
| Params per GPU | 2.28B |

---

## Bugs found and fixed

### 1. FlashAttention install: GLIBCXX_3.4.29 not found

**Symptom:** `ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`  
**Cause:** Python spack module has RPATH baked to GCC 8.5.0 libstdc++ (lacks GLIBCXX_3.4.29). `module load gcc/12.2.0` sets PATH but does not override RPATH at runtime.  
**Fix:** Force GCC 12 libstdc++ at runtime:
```bash
GCC12_LIB=$(gcc -print-file-name=libstdc++.so.6)
export LD_PRELOAD=$GCC12_LIB${LD_PRELOAD:+:$LD_PRELOAD}
```
Added to `train_32k_test_leonardo.sbatch` and `install_flash_attn_leonardo.sbatch`.

---

### 2. Flash-attention not used in core transformer

**Symptom:** OOM in `baddbmm` (scaled dot-product attention) — pure O(T²) computation, allocates `(seq_len, seq_len)` bf16 ≈ 32 GB.  
**Cause:** `--use-flash-attn` only activates the `FlashSelfAttention` path in `megatron/legacy/model/transformer.py`. The default core transformer (`megatron/core/`) uses a plain `DotProductAttention` without a flash-attn path.  
**Fix:** Add `--use-legacy-models` to route through the legacy transformer that has native flash-attn + GQA support.

---

### 3. GQA reshape bug in legacy transformer

**Symptom:** `RuntimeError: view size is not compatible with input tensor's size and stride`  
**Location:** `megatron/legacy/model/transformer.py:720`  
**Cause:** `query_layer.view(...)` fails on a non-contiguous tensor produced by `repeat_interleave` for GQA key/value head broadcasting.  
**Fix:** Changed `.view(...)` → `.reshape(...)` on Leonardo directly:
```bash
sed -i 's/query_layer\.view(/query_layer.reshape(/g' \
  $WORK/NVIDIA-Megatron-LM/megatron/legacy/model/transformer.py
```

---

### 4. MLP OOM: allocator fragmentation

**Symptom:** OOM in MLP SwiGLU with 2.45 GB reserved-but-unallocated (fragmentation).  
**Fix:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — allows the allocator to return unused segments to the OS, eliminating fragmentation.

---

### 5. `@jit_fuser` on bias_dropout_add triggers TorchInductor compilation

**Symptom:** OOM in `bias_dropout_add_fused_train` — TorchInductor compilation allocates extra workspace.  
**Fix:** `--no-bias-dropout-fusion` disables the JIT-compiled path entirely.

---

### 6. `--recompute-activations` overrides `--recompute-granularity full`

**Symptom:** Config dump shows `recompute_granularity: selective` despite `--recompute-granularity full` in the sbatch.  
**Cause:** Inside `megatron/training/arguments.py` `validate_args()`:
```python
if args.recompute_activations:
    args.recompute_granularity = 'selective'   # clobbers the user's value
```
**Effect:** Selective recompute stores all intermediate activations except the core attention (~1 GB/layer), accumulating ~32 GB for 32 layers on a PP=1 or ~16 GB for 16 layers × 2 in-flight on PP=2. This is the dominant memory consumer (explains the consistent 60 GB at OOM time).  
**Fix:** Remove `--recompute-activations`, keep only `--recompute-granularity full` with `--recompute-method uniform --recompute-num-layers 16`.

---

## Memory budget analysis (per GPU, TP=2, PP=2)

| Component | Size |
|-----------|------|
| BF16 model params | 4.56 GB |
| FP32 grad buffer (pre-alloc) | 9.12 GB |
| FP32 master weights | 9.12 GB |
| Adam m1 (lazy, after first step) | 9.12 GB |
| Adam m2 (lazy, after first step) | 9.12 GB |
| BF16 .grad tensors | 4.56 GB |
| Activation checkpoints (full, 2 micro-batch × 16 layers × 256 MB) | 8 GB |
| NCCL + CUDA context | ~2.5 GB |
| **Total (after optimizer init)** | **~56 GB** |
| MLP intermediate (dense_h_to_4h, TP=2): (32768, 1, 14336) | **0.875 GB** |
| **Peak total** | **~57 GB** |
| A100-SXM-64GB capacity | 63.42 GB |
| Observed peak | ~60 GB |

The observed ~60 GB is ~3 GB over theoretical — likely cuBLAS workspace and fragmentation. This leaves ~530 MiB at the MLP backward recompute, which is not enough for the 896 MiB dense_h_to_4h output (TP=2) or 448 MiB (TP=4).

---

## Current status

All configurations tested on a single node fail with OOM at the MLP layer:

| Job | Config | OOM at | Free | Need | Short by |
|-----|--------|--------|------|------|----------|
| 45752954 | TP=2 PP=2, selective recompute | bias_dropout silu | 374 MiB | 448 MiB | 74 MiB |
| 45754649 | TP=2 PP=2, selective, no jit fuse | swiglu silu | 374 MiB | 448 MiB | 74 MiB |
| 45755485 | TP=4 PP=1, selective | dense_h_to_4h | 279 MiB | 448 MiB | 169 MiB |
| 45767785 | TP=2 PP=2, full recompute | dense_h_to_4h (backward) | 533 MiB | 896 MiB | 363 MiB |

**Root cause:** 9B model with Adam at 32K sequence fills ~60 GB per A100-64GB regardless of TP/PP, leaving insufficient headroom for the large MLP activations.

---

## Path forward

### Option A — 2 nodes (recommended)

With 2 nodes (8 GPUs), TP=2, PP=2, DP=2 + `--use-distributed-optimizer`:
- Adam states sharded across DP=2: saves ~9 GB/GPU
- Peak memory: ~47 GB → 16 GB headroom
- Script: `lumi/slurm/train_32k_test_leonardo_2nodes.sbatch` (TODO)

### Option B — 1 node at 16K

Run smoke test at `seq_len=16384` (half the memory for sequence-dependent tensors). Verifies the pipeline end-to-end; does not prove 32K fits on 1 node (it doesn't).

---

## Commit history

| Commit | Description |
|--------|-------------|
| `dfaf580` | Fix train_32k sbatch: add gcc/12.2.0 + LD_PRELOAD for flash-attn |
| `7c49ad0` | Switch 32K test to legacy model + flash-attn |
| `266a829` | Restore --ckpt-format torch (required by legacy model) |
| `e42ca78` | Restore --transformer-impl local (prevents TE import) |
| `bc42b70` | Add PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True |
| `99dace6` | Disable bias-dropout fusion (TorchInductor OOM) |
| `02df538` | Switch to TP=4 PP=1 (attempted MLP tensor reduction) |
| `d6ff3c7` | Use SGD no momentum (attempted optimizer savings) |
| `0364b63` | Fix recompute: drop --recompute-activations |
| `bf755ec` | Add --recompute-num-layers 16 |
