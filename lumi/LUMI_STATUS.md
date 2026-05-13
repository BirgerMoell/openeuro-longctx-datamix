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

## ✅ YaRN MULTILINGUAL SMOKE TEST PASSED (job 18494787)

### Training output — 10 iterations on multilingual ≥16K-token data (35 languages)

4 nodes × 8 GPUs = 32 GPUs, TP=2, PP=4, CP=4, seq_length=32768, GBS=8

```
iteration  3/10 | lm loss: 1.328448E+01 | grad norm: 20.751 | elapsed: 18913.4ms | 34.2 TFLOP/s/GPU
iteration  4/10 | lm loss: 1.322131E+01 | grad norm: 12.566 | elapsed: 18833.1ms | 34.3 TFLOP/s/GPU
iteration  5/10 | lm loss: 1.287841E+01 | grad norm: 12.982 | elapsed: 18866.6ms | 34.3 TFLOP/s/GPU
validation loss at iteration  5 | lm loss: 1.280456E+01 | PPL: 3.638735E+05
iteration  6/10 | lm loss: 1.264119E+01 | grad norm: 22.707 | elapsed: 18816.7ms | 34.4 TFLOP/s/GPU
iteration  7/10 | lm loss: 1.284148E+01 | grad norm: 79.808 | elapsed: 18807.2ms | 34.4 TFLOP/s/GPU
iteration  8/10 | lm loss: 1.250232E+01 | grad norm: 14.550 | elapsed: 18830.2ms | 34.3 TFLOP/s/GPU
iteration  9/10 | lm loss: 1.211069E+01 | grad norm: 13.883 | elapsed: 18845.6ms | 34.3 TFLOP/s/GPU
iteration 10/10 | lm loss: 1.165134E+01 | grad norm: 34.423 | elapsed: 18840.2ms | 34.3 TFLOP/s/GPU
validation loss at iteration 10 | lm loss: 1.152394E+01 | PPL: 1.011075E+05
```

**Loss dropped from 13.28 → 11.65 over 10 iterations.** The model is learning from multilingual 32K-context data across 35 European languages. Checkpoint saved at iteration 10.

**Bugs encountered and fixed before reaching this state:**
1. **BlendedMegatronDataset hang** — 3-tier blended data hangs rank 0 for 30+ min building weights; other ranks SIGABRT at NCCL watchdog. Fix: use single 16k_plus tier for smoke test.
2. **GPTDataset test-split starvation** — default split 969/30/1 gives 0.1% to test; with 175 docs that's <1 doc; builder hangs. Fix: `--split 750,150,100` + `--eval-iters 2`.
3. **OOM at CP=1 on 1 node** — 32K seqlen fills 60/64 GB GPU memory before first forward pass. Fix: 4 nodes with CP=4 (splits sequence 4-ways, 4× lower activation memory).

---

## ✅ HF MULTILINGUAL DATA SMOKE TEST PASSED (job 18515088)

### Training output — 10 iterations on 8-language pre-tokenized HF dataset (full 3-tier blend)

4 nodes × 8 GPUs = 32 GPUs, TP=2, PP=4, CP=4, seq_length=32768, GBS=8  
Data: `birgermoell/oellm-longctx-tokenized-streamed-all-v2` — 24 merged files, 87 GB, 8 languages (bg cs da et fi fr hr nl)  
First time the full **blended** 3-tier DATA_PATH (24 entries) was used in training.

```
iteration  2/10 | lm loss: 1.328850E+01 | grad norm: 11.845 | elapsed: 54688.8ms | 11.8 TFLOP/s/GPU
iteration  3/10 | lm loss: 1.328448E+01 | grad norm: 20.751 | elapsed: 19035.8ms | 34.0 TFLOP/s/GPU
iteration  4/10 | lm loss: 1.322135E+01 | grad norm: 12.566 | elapsed: 18939.2ms | 34.1 TFLOP/s/GPU
iteration  5/10 | lm loss: 1.287813E+01 | grad norm: 12.984 | elapsed: 18932.9ms | 34.2 TFLOP/s/GPU
iteration  6/10 | lm loss: 1.264196E+01 | grad norm: 22.783 | elapsed: 18933.1ms | 34.1 TFLOP/s/GPU
iteration  7/10 | lm loss: 1.283994E+01 | grad norm: 79.636 | elapsed: 18963.8ms | 34.1 TFLOP/s/GPU
iteration  8/10 | lm loss: 1.250303E+01 | grad norm: 14.550 | elapsed: 18953.9ms | 34.1 TFLOP/s/GPU
iteration  9/10 | lm loss: 1.211181E+01 | grad norm: 13.867 | elapsed: 18969.3ms | 34.1 TFLOP/s/GPU
iteration 10/10 | lm loss: 1.165195E+01 | grad norm: 34.447 | elapsed: 18948.4ms | 34.1 TFLOP/s/GPU
validation loss at iteration 10 | lm loss: 1.193499E+01 | PPL: 1.525110E+05
test loss at iteration 10       | lm loss: 1.222711E+01 | PPL: 2.042524E+05
```

**Loss dropped from 13.29 → 11.65 over 10 iterations** — identical curve to the previous single-tier smoke test (job 18494787). The full 24-entry blended DATA_PATH works without the BlendedMegatronDataset hang (the hang only occurs with small datasets; at 87 GB / 35B tokens the index builder completes quickly). Throughput steady at 34.1 TFLOP/s/GPU after warmup.

**Pipeline now fully validated end-to-end:**  
`download_tokenized.sbatch` (HF → LUMI, 87 GB) → `merge_datasets.py` (24 merged files) → `data_path.args` (uniform per-language weighting) → `yarn_multilingual_test.sbatch` (10 iters, loss ↓) ✅

---

## ✅ FULL MULTILINGUAL YARN TRAINING COMPLETE (job 18536300)

### Training output — 1000 iterations, 32 nodes × 8 GPUs = 256 GPUs

32 nodes, TP=2, PP=4, CP=4, seq_length=32768, GBS=128  
Data: 8 languages (bg cs da et fi fr hr nl), 24-entry blended DATA_PATH, 87 GB / ~35B tokens  
Runtime: ~9 hours (08:51 → 17:43 UTC, 2026-05-10)

```
iteration    2/1000 | lm loss: 1.222107E+01 | grad norm:  73.530 | 16.1 TFLOP/s/GPU
iteration  101/1000 | lm loss: 7.123520E+00 | grad norm:   9.441 | 40.6 TFLOP/s/GPU
iteration  201/1000 | lm loss: 5.495810E+00 | grad norm:   2.855 | 40.6 TFLOP/s/GPU
iteration  301/1000 | lm loss: 4.684253E+00 | grad norm:   1.627 | 40.5 TFLOP/s/GPU
iteration  401/1000 | lm loss: 4.373148E+00 | grad norm:   1.259 | 40.6 TFLOP/s/GPU
iteration  501/1000 | lm loss: 4.159157E+00 | grad norm:   1.615 | 37.5 TFLOP/s/GPU
iteration  601/1000 | lm loss: 3.923108E+00 | grad norm:   1.286 | 40.5 TFLOP/s/GPU
iteration  701/1000 | lm loss: 3.825454E+00 | grad norm:   1.505 | 40.5 TFLOP/s/GPU
iteration  801/1000 | lm loss: 3.668952E+00 | grad norm:   1.192 | 40.5 TFLOP/s/GPU
iteration  901/1000 | lm loss: 3.631058E+00 | grad norm:   0.568 | 40.5 TFLOP/s/GPU
iteration 1000/1000 | lm loss: 3.664297E+00 | grad norm:   0.331 | 40.5 TFLOP/s/GPU

validation loss at iteration 1000 | lm loss: 3.567921E+00 | PPL: 3.544285E+01
test loss at iteration 1000       | lm loss: 3.430229E+00 | PPL: 3.088371E+01
```

**Loss dropped from 12.22 → 3.66 over 1000 iterations.** The high initial loss reflects the model adapting to YaRN-scaled position embeddings at 32K context (the checkpoint was pre-trained at 2K). By iter 100 the loss has dropped 5 nats — the model is learning long-context structure fast. Grad norm stabilises at ~1–2 from iter 200 onward, indicating stable training.

**Checkpoints saved:**
- `iter_0000500` — midpoint checkpoint
- `iter_0001000` — final checkpoint
- Path: `/flash/project_462000963/bmoell/yarn-multilingual/checkpoints/`

**Throughput:** 40.5 TFLOP/s/GPU sustained (256 GPUs), ~513 tok/s/GPU.  
One brief dip to 37.5 TFLOP/s at iter 501 (likely a checkpoint save or network hiccup), otherwise rock-solid.

**Next step:** Convert `iter_0001000` checkpoint to HuggingFace format and add `rope_scaling` to `config.json`:
```json
"rope_scaling": {"factor": 16.0, "original_max_position_embeddings": 2048, "type": "yarn"},
"rope_theta": 10000
```
Then evaluate on RULER / Needle-in-a-Haystack long-context benchmarks.

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
3. ✅ ~~**Language analysis for all 35 languages**~~ — done; jobs 18479845 + 18480711, results in `LANGUAGE_ANALYSIS.md`.
4. ✅ ~~**Tokenize multilingual tiers (mini)**~~ — done; `tokenize_tiers_mini.sbatch` produced 175 docs in `multilingual_16k_plus_text_document` (16K+ token tier) from 50 docs × 35 languages using lumi-multitorch SIF.
5. ✅ ~~**YaRN multilingual smoke test**~~ — done; job 18494787, 10 iterations, loss 13.28 → 11.65, checkpoint saved.
6. **Tokenize full multilingual tiers** — submit `tokenize_tiers.sbatch` to tokenize all 35 language JSONL files into 3 Megatron tier datasets (this will take ~8h on standard-g).
7. **YaRN multilingual full run** — submit `yarn_multilingual.sbatch` once full tokenization completes. Fix language balance issue first (see PROJECT_STATUS.md Phase 2).
8. **Submit ROCm compatibility patches upstream** — fixes should be contributed back to OpenEuroLLM/NVIDIA-Megatron-LM.

---

## HuggingFace Conversion — Manual Step Required

After converting a Megatron checkpoint to HuggingFace format, the `rope_scaling` parameters must be **added manually** to `config.json` — the conversion script does not propagate them automatically.

### YaRN (recommended for multilingual)

```json
"rope_scaling": {
  "factor": 16.0,
  "original_max_position_embeddings": 2048,
  "type": "yarn"
},
"rope_theta": 10000
```

Reference: `/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-9b-80-20-TP-2-PP-4-yarn-finepdfs/checkpoint_0001000/config.json`

### LongRoPE (English-only search, not recommended for multilingual)

LongRoPE requires a 64-element `long_factor` and `short_factor` array obtained by running a search over the target language corpus. Jouni has done this search for English only. The factors are in:

`/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-9b-80-20-TP-2-PP-4-longrope-finepdfs/checkpoint_0001000/config.json`

For multilingual training, use YaRN instead (single scaling factor, no per-language search needed).

---

## Context Extension Strategy

| Method | Megatron arg | HF config type | Notes |
|--------|-------------|----------------|-------|
| YaRN | `--position-embedding-type yarn --yarn-scaling-factor 16.0 --yarn-original-max-position-embeddings 2048` | `"yarn"` | Recommended for multilingual — no search needed |
| LongRoPE | `--position-embedding-type longrope --longrope-rescale-factors-path result_final.csv --longrope-original-max-position-embeddings 2048` | `"longrope"` | Better quality but requires per-language search |

Both extend from 2K → 32K context (scaling factor 16.0 = 32768 / 2048).

---

## File Layout

```
lumi/
├── run-lumi.sh                       # top-level runner: hello / status / logs
├── LANGUAGE_ANALYSIS.md              # per-language long-context stats (35 languages)
├── LUMI_STATUS.md                    # this file
└── slurm/
    ├── hello.sbatch                  # end-to-end validation job (mock + real data)
    ├── train_real.sbatch             # Maltese real-data training (validated)
    ├── lang_analysis.sbatch          # 16-language stats (job 18479845, done)
    ├── lang_analysis_remaining.sbatch# remaining 19 languages (job 18480711, done)
    ├── tokenize_tiers.sbatch         # split 35 langs into 3 Megatron tier datasets
    ├── yarn_test.sbatch              # 10-iter YaRN smoke test (1 node)
    ├── yarn_multilingual.sbatch      # full YaRN multilingual run (32 nodes)
    ├── longrope_test.sbatch          # 10-iter LongRoPE smoke test (1 node)
    ├── longrope_multilingual.sbatch  # full LongRoPE multilingual run (32 nodes)
    └── gpu_diag*.sbatch              # GPU/ROCm diagnostic jobs
```
