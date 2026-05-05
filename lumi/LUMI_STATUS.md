# LUMI Run Status — OpenEuroLLM Long-Context Data Pipeline

**Last updated:** 2026-05-05  
**Environment:** LUMI (CSC), partition `dev-g`, AMD Instinct MI250X, ROCm 7.0  
**Container:** `lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif` (PyTorch 2.10.0+ROCm7.0)  
**Project account:** `project_462000963`  
**Scratch path:** `/scratch/project_462000963/bmoell/`  
**Megatron fork:** [OpenEuroLLM/NVIDIA-Megatron-LM](https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM)  
**Tokenizer:** `openeurollm/tokenizer-256k` (vocab size = 2¹⁸ = **262,144** tokens)

---

## Goal

Validate the full end-to-end stack on LUMI:

1. **Data pipeline** — `longctx estimate → download → convert → filter-long → tokenize`
2. **Megatron GPT training** — at least 5 training iterations with real loss values printed

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

### Megatron initialisation

`pretrain_gpt.py` initialises correctly up to and including:
- Model and optimizer build
- DataLoader build (including `--mock-data`)
- Apex extension compilation (`amp_C`, fused kernels)
- `[before the start of training step]` log line printed

---

## What Is Broken ❌

### Megatron training hangs at first forward pass

After printing `[before the start of training step]`, the job hangs indefinitely with no output and no crash. Confirmed with both real data and `--mock-data`.

**Trace of where the hang is** (added diagnostic prints to Megatron source):

```
[DIAG] train_step: calling forward_backward_func
[DIAG] schedules: calling final forward_step
[DIAG] forward_step: calling get_batch          ← fast
[DIAG] forward_step: get_batch done, calling model
[DIAG] GPTModel: calling _preprocess (embedding lookup)
[DIAG] GPTModel: _preprocess done              ← fast
[DIAG] GPTModel: calling decoder (transformer layers)
                                                ← HANGS HERE
```

The hang is inside `self.decoder(...)` — i.e., within the transformer layers (attention + FFN).

---

## Debugging History

### Things tried that did NOT fix the hang

| Attempt | Result |
|---------|--------|
| `--distributed-backend nccl` | Hung immediately ("guessing device ID" warning from ProcessGroupNCCL) |
| `torchrun --standalone --nproc_per_node=1` | Hung (process spawning issues in Singularity) |
| `--distributed-backend gloo` | Initialization works; hang is elsewhere |
| `--num-workers 0` | No change — hang is not in the DataLoader |
| `--mock-data` | No change — hang is not in data loading |
| `AMD_SERIALIZE_KERNEL=3` | Caused complete deadlock (worse) |
| `MIOPEN_FIND_ENFORCE=NONE` | No change |
| `HSA_ENABLE_SDMA=0` | No change |
| `--no-masked-softmax-fusion --no-bias-dropout-fusion --no-gradient-accumulation-fusion` | No change |
| Direct `python pretrain_gpt.py` (vs `torchrun`) with `MASTER_ADDR/PORT/WORLD_SIZE/RANK/LOCAL_RANK` env vars | No change |
| Model size reduced (4→2 layers, hidden 256→64) | No change |

### Key fix already applied

The original crash was a GPU core dump (`vectorized_gather_kernel` SIGABRT) because the tokenizer `openeurollm/tokenizer-256k` has **262,144** vocab entries (not 256,001 or 256,128 as initially assumed). Fixed with `--vocab-size 262144`.

### Current hypothesis

The hang is in Megatron's `DotProductAttention.forward()`, most likely in one of:

1. **`torch.baddbmm`** — batched QK^T matrix multiply (rocBLAS tuning on first call?)
2. **`tensor_parallel.get_cuda_rng_tracker().fork()`** — saves/restores CUDA RNG state via `torch.cuda.get_rng_state()` / `torch.cuda.set_rng_state()`. This involves GPU synchronisation and is called even when `--attention-dropout 0.0`.
3. **`self.scale_mask_softmax`** — apex fused softmax (though unfused fallback is selected at runtime)

The RNG fork hypothesis is strongest: `fork()` runs unconditionally when `sequence_parallel=False` (our case), independently of dropout rate.

---

## Pending Next Steps

1. **Confirm hang location** — diagnostic prints were added to `dot_product_attention.py` at each operation (`baddbmm`, `scale_mask_softmax`, `fork()`, `bmm`). Submit a job to see which print is last.

2. **Test RNG fork hypothesis** — if the hang is at `fork()`, fix by either:
   - Patching `dot_product_attention.py` to skip `fork()` when `attention_dropout == 0`
   - Or setting `--attention-dropout 0.0` AND `--hidden-dropout 0.0` and monkey-patching `get_cuda_rng_tracker` to be a no-op

3. **If hang is in baddbmm** — try pre-warming rocBLAS with a standalone GEMM of the same dimensions before launching the training loop.

4. **Once training works** — remove `--mock-data`, switch back to real tokenized data (`--data-path $TRAIN_DIR/mt_train_text_document`), verify loss decreases over 5 iterations, and commit the final working `hello.sbatch`.

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
│   └── gpu_diag4.sbatch # gloo barrier + Megatron timer tests
└── LUMI_STATUS.md       # this file
```
