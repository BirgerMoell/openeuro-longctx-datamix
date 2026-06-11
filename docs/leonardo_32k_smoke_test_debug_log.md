# Leonardo 32K Smoke Test — Debug Log

**Date:** 2026-06-11  
**Status:** ✅ PASSED — 10/10 iterations completed, job 45790387  
**Script:** `lumi/slurm/train_32k_test_leonardo_2nodes.sbatch`

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

The observed ~60 GB is ~3 GB over theoretical — likely cuBLAS workspace and fragmentation. In the TP=2 runs the failing SwiGLU recompute already has the 896 MiB dense_h_to_4h output resident, then needs a fresh 448 MiB allocation for `silu(x[0])` / multiply. TP=4 should cut those tensors to roughly 448 MiB resident plus 224 MiB fresh allocation.

---

## Results

| Job | Nodes | Config | Result |
|-----|-------|--------|--------|
| 45752954 | 1 | TP=2 PP=2, selective recompute | OOM in bias_dropout (448 MiB, 74 MiB short) |
| 45754649 | 1 | TP=2 PP=2, no jit fusion | OOM in swiglu silu (448 MiB, 74 MiB short) |
| 45755485 | 1 | TP=4 PP=1, selective | OOM in dense_h_to_4h (448 MiB, 169 MiB short) |
| 45767785 | 1 | TP=2 PP=2, full recompute | OOM in dense_h_to_4h backward (896 MiB, 363 MiB short) |
| 45773570 | 2 | TP=2 PP=2 DP=2, dist. optimizer | OOM in swiglu silu (448 MiB, up to 420 MiB short) |
| 45775847 | 2 | TP=2 PP=2 DP=2, dist. optimizer, no grad overlap | OOM in swiglu silu (448 MiB, up to 436 MiB short) |
| 45777471 | 2 | TP=4 PP=1 DP=2, dist. optimizer | OOM in residual/RMSNorm recompute (256-512 MiB short) |
| **45779119** | **2** | **TP=4 PP=1 DP=2, sequence parallel, dist. optimizer** | **✅ 2/2 iterations, peak 55.4 GB** |
| **45790387** | **2** | **same config** | **✅ 10/10 iterations, peak 55.4 GB — SMOKE TEST PASSED** |

**Root cause of single-node failures:** 9B model + Adam at 32K fills ~60 GB per A100-64GB regardless of TP/PP configuration — the optimizer states, gradient buffers, activation checkpoints, and CUDA workspace leave only ~500 MiB free, not enough for the MLP backward recompute tensors.

**What made it work:** `--sequence-parallel` shards the residual stream (one 256 MB tensor per layer) across the TP=4 group, reducing per-rank activation memory by 4×. Combined with TP=4 (quarters the MLP intermediates), `--use-distributed-optimizer` (shards Adam states across DP=2), and 2 nodes — peak drops to 55.4 GB with 8 GB headroom.

---

## Verified working configuration (job 45790387)

```
2 nodes × 4× A100-SXM-64GB = 8 GPUs
TP=4, PP=1, DP=2
--sequence-parallel
--use-distributed-optimizer
--recompute-granularity full --recompute-method uniform --recompute-num-layers 32
GBS=2, MBS=1, seq_len=32768
Peak memory: 55.4 GB / 63.4 GB
Throughput: ~105 TFLOP/s/GPU (iterations 2–10)
```

10-iteration loss trace (random init, sanity check only):
```
iter  1: lm_loss=13.28, grad_norm=23.2
iter  2: lm_loss=13.09, grad_norm=27.7
iter  3: lm_loss=13.55, grad_norm=240.4
iter  4: lm_loss=12.55, grad_norm=37.5
iter  5: lm_loss=12.62, grad_norm=50.2
iter  6: lm_loss=12.78, grad_norm=113.2
iter  7: lm_loss=12.44, grad_norm=50.1
iter  8: lm_loss=12.37, grad_norm=42.2
iter  9: lm_loss=12.00, grad_norm=31.3
iter 10: lm_loss=12.39, grad_norm=41.1
```

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
| `0364b63` | Fix recompute: drop --recompute-activations flag |
| `92d8337` | Add debug log and 2-node sbatch |
| `55c88c5` | Set train-iters to 10; 2-node smoke test passes |
