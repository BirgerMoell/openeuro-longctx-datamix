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

This is a prerequisite before running large-scale multilingual training for the OpenEuroLLM project. The ML framework (NVIDIA Megatron-LM) is designed for NVIDIA GPUs and uses several apex fused CUDA kernels. Our challenge was making it work on AMD/ROCm hardware in the LUMI environment.

The entry point is `lumi/slurm/hello.sbatch`, submitted via `bash lumi/run-lumi.sh hello`.

---

## ✅ FULL STACK VALIDATED (job 18306279) — mock data

### Training output — 5 iterations on mock data

```
iteration 1/5 | lm loss: 1.253347E+01 | grad norm: 3.203 | elapsed: 16396.5ms
iteration 2/5 | lm loss: 1.254492E+01 | grad norm: 3.068 | elapsed:   161.4ms
iteration 3/5 | lm loss: 1.254339E+01 | grad norm: 2.979 | elapsed:    35.4ms
iteration 4/5 | lm loss: 1.251824E+01 | grad norm: 2.723 | elapsed:    33.9ms
iteration 5/5 | lm loss: 1.246977E+01 | grad norm: 2.584 | elapsed:    33.2ms

validation loss at iteration 5 | lm loss value: 1.250232E+01 | lm loss PPL: 2.689614E+05
validation loss at iteration 5 on test set | lm loss value: 1.243491E+01 | lm loss PPL: 2.514279E+05
Training OK!
```

**The loss of ~12.48 equals ln(262,144)** — exactly what a freshly initialized model produces at random on a vocab of 262,144 tokens. Grad norm is steadily decreasing, confirming proper gradient flow.

---

## ✅ REAL DATA TRAINING VALIDATED (job 18479860)

### Training output — 20 iterations on real Maltese corpus (5,463 docs)

Tokenized all 5,463 docs from `data/long/mt.jsonl` (~113M tokens) and trained a 4-layer GPT for 20 iterations:

```
iteration  1/20 | lm loss: 1.250298E+01 | grad norm: 5.748 | elapsed: 14761.3ms
iteration  2/20 | lm loss: 1.229649E+01 | grad norm: 5.161 | elapsed:   937.1ms
iteration  3/20 | lm loss: 1.215209E+01 | grad norm: 4.758 | elapsed:    31.7ms
iteration  5/20 | lm loss: 1.208117E+01 | grad norm: 3.032 | elapsed:    29.3ms
iteration 10/20 | lm loss: 1.187418E+01 | grad norm: 3.278 | elapsed:    29.0ms
iteration 15/20 | lm loss: 1.180686E+01 | grad norm: 2.469 | elapsed:    29.4ms
iteration 20/20 | lm loss: 1.167589E+01 | grad norm: 2.477 | elapsed:    28.2ms

validation loss at iteration 10 | lm loss value: 1.186658E+01 | lm loss PPL: 1.424263E+05
validation loss at iteration 20 | lm loss value: 1.176503E+01 | lm loss PPL: 1.286733E+05
validation loss at iteration 20 on validation set | lm loss value: 1.178377E+01 | lm loss PPL: 1.311069E+05
validation loss at iteration 20 on test set | lm loss value: 1.173791E+01 | lm loss PPL: 1.252302E+05
=== Done ===
```

**Loss dropped from 12.50 → 11.68 over 20 iterations** — a decrease of ~0.82 nats, confirming the model is learning from real Maltese text. This is meaningfully below the random baseline of ln(262,144) ≈ 12.48. The grad norm also decreases steadily (5.748 → 2.477).

The full pipeline is now end-to-end validated on real data:
`download → convert → filter-long → tokenize → Megatron GPT training`

Fix required: `export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}` inside the singularity heredoc, so `megatron.core` resolves from the local repo rather than the older container pip package (`megatron-core`).

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

### GPU compute

| Test | Result |
|------|--------|
| `torch.matmul` 1024×1024 | ✅ ~940ms first call (kernel compile), <1ms after |
| `nn.Embedding(262144, 64)` + `nn.Linear` + backward (5 iters) | ✅ |
| `F.scaled_dot_product_attention` (causal, math backend) | ✅ |
| Megatron DotProductAttention (all 4 layers) | ✅ |
| Megatron full forward + backward + optimizer step | ✅ |

### Megatron GPT training

Full training loop with 5 iterations, validation, and test evaluation — all complete with real loss values.

---

## Bugs Found and Fixed

Three bugs had to be diagnosed and fixed to get Megatron working on ROCm/MI250X. All are in the Megatron source on the LUMI scratch filesystem.

### Bug 1: Wrong vocab size → GPU crash

**Symptom:** `vectorized_gather_kernel` SIGABRT (out-of-bounds embedding lookup)  
**Root cause:** Tokenizer has 262,144 vocab entries; Megatron was using a smaller default.  
**Fix:** `--vocab-size 262144` in hello.sbatch.

### Bug 2: NCCL process group hangs on MI250X

**Symptom:** Training hangs immediately with "guessing device ID" from ProcessGroupNCCL  
**Root cause:** NCCL can't reliably auto-detect ROCm device IDs inside Singularity without explicit device binding setup.  
**Fix:** `--distributed-backend gloo` (single-node CPU-based communication).

### Bug 3: `apex.normalization.FusedLayerNorm` hangs on MI250X ← main blocker

**Symptom:** Training hangs indefinitely inside the first transformer layer with zero output.  
**Root cause:** `megatron/core/fusions/fused_layer_norm.py` calls `FusedLayerNormAffineFunction.apply()` from apex when `hidden_size` is not in a hardcoded list of supported sizes (the list starts at 1024; we use 256). This apex kernel deadlocks on MI250X.

**Fix** (`megatron/core/fusions/fused_layer_norm.py` — the `else` branch of `FusedLayerNorm.forward()`):

```python
# Before (hangs on MI250X):
return FusedLayerNormAffineFunction.apply(
    input, weight, self.bias, self.hidden_size, self.eps
)

# After (uses PyTorch native — no apex kernel):
return torch.nn.functional.layer_norm(
    input, self.hidden_size, weight, self.bias, self.eps
)
```

### Bug 4: `allreduce_coalesced` on CUDA tensors unsupported by gloo

**Symptom:** Crash after backward pass: `ProcessGroupGloo::allreduce_coalesced: unsupported device type cuda`  
**Root cause:** Megatron's DDP gradient sync batches `all_reduce` calls via `_coalescing_manager`, which internally calls `group.allreduce_coalesced(tensors)`. The gloo backend doesn't support this for CUDA tensors.  
With WORLD_SIZE=1, gradient synchronization is a no-op anyway.

**Fix** (`megatron/core/distributed/param_and_grad_buffer.py` — `BucketGroup.start_grad_sync()`):

```python
# Skip all-reduce with single process - gloo does not support
# allreduce_coalesced for CUDA tensors, and it is a no-op with 1 rank.
import torch.distributed as _dist
if _dist.get_world_size(group=self.data_parallel_group) == 1:
    return
```

---

## Debugging History (how we found the bugs)

The main hang was inside `self.decoder(...)` (TransformerBlock). We traced it progressively with diagnostic `print()` statements inserted into Megatron source files:

1. `training.py`: added `[DIAG] train_step: calling forward_backward_func`
2. `schedules.py`: added `[DIAG] schedules: calling final forward_step`
3. `pretrain_gpt.py`: added `[DIAG] forward_step: calling get_batch`
4. `gpt_model.py`: added prints at `_preprocess`, `decoder`, `_postprocess`
5. `dot_product_attention.py`: added prints at each operation inside attention

The attention prints never appeared — meaning the hang was before `DotProductAttention`. The last thing to run before attention is `input_layernorm` (a `FusedLayerNorm`). A standalone test (`gpu_diag5.sbatch`) confirmed `FusedLayerNorm` hangs after >6 minutes with zero GPU progress.

---

## Current hello.sbatch Configuration (working)

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

ROCm env vars:
```bash
export MIOPEN_USER_DB_PATH=$WORKDIR/miopen_cache
export MIOPEN_CUSTOM_CACHE_DIR=$WORKDIR/miopen_cache
export MIOPEN_FIND_ENFORCE=NONE
export ROCR_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost; export MASTER_PORT=29500
export WORLD_SIZE=1; export RANK=0; export LOCAL_RANK=0
```

---

## Next Steps

1. ✅ ~~**Clean up diagnostic prints**~~ — done; branch `rocm-mi250x-compat` on `BirgerMoell/NVIDIA-Megatron-LM` contains only the two fixes.
2. ✅ ~~**Switch from `--mock-data` to real data**~~ — done; job 18479860 confirmed loss drops on real Maltese text.
3. **Scale up** — increase model size and sequence length toward actual long-context training targets. Add more languages.
4. **Submit ROCm compatibility patches upstream** — both fixes should be contributed back to OpenEuroLLM/NVIDIA-Megatron-LM.
5. **Add more languages** — lang_analysis job (18479845) is measuring token length distributions for 16 target languages. Add Swedish, Finnish, and other high-resource languages to the training mix.

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
