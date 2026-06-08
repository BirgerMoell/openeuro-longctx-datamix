# Megatron Training on Leonardo (CINECA)

Leonardo is a Tier-0 EuroHPC cluster at CINECA with NVIDIA A100-SXM-64GB GPUs (4 per boost node). This document covers the full setup for running Megatron-LM training there under the `OELLM_prod2026` project.

---

## Cluster basics

| Item | Value |
|------|-------|
| Login | `ssh pmoell00@login.leonardo.cineca.it` (certificate via `smallstep`) |
| Account | `OELLM_prod2026` |
| Partition | `boost_usr_prod` |
| GPUs per node | 4× A100-SXM-64GB |
| `$WORK` | `/leonardo_work/OELLM_prod2026/` — 1 TB, persistent |
| `$SCRATCH` | `/leonardo_scratch/large/userexternal/pmoell00/` — no quota, purged after 40 days |
| `$HOME` | 50 GB hard limit — keep only dotfiles here |

**Compute nodes have no internet.** All pip installs, HuggingFace downloads, and git clones must be done on a login node before submitting jobs.

---

## One-time setup

Run once from a login node. This clones the Megatron fork and builds the Python venv:

```bash
# on a Leonardo login node
bash ~/openeuro-longctx-datamix/lumi/setup_megatron_leonardo.sh
```

What it does:
- Clones `luomajouni/NVIDIA-Megatron-LM` to `$WORK/NVIDIA-Megatron-LM`
  - If the repo is private and clone fails, transfer a tarball from LUMI (see below)
- Creates `$WORK/megatron_venv` with torch 2.6+cu124, transformers, sentencepiece, pybind11, tensorboard

### Transferring the Megatron fork from LUMI (if clone fails)

```bash
# on LUMI
tar czf /tmp/megatron.tar.gz -C /flash/project_462000963/jouni/test NVIDIA-Megatron-LM

# on your local machine
scp bmoell@lumi.csc.fi:/tmp/megatron.tar.gz .
scp megatron.tar.gz pmoell00@login.leonardo.cineca.it:/leonardo_work/OELLM_prod2026/

# on Leonardo
tar xzf /leonardo_work/OELLM_prod2026/megatron.tar.gz -C /leonardo_work/OELLM_prod2026/
```

---

## Data transfer from LUMI

Data must live in `$WORK/data/` (persistent). Scratch is fine for outputs but not input data.

```bash
# on LUMI — pack the tokenized data
scp /scratch/project_462000963/bmoell/openeuro-longctx-datamix/data/multilingual_16k_plus_text_document.{bin,idx} \
    your_local_machine:/tmp/

# on your local machine — forward to Leonardo
scp /tmp/multilingual_16k_plus_text_document.{bin,idx} \
    pmoell00@login.leonardo.cineca.it:/leonardo_work/OELLM_prod2026/data/
```

The tokenizer lives at `$WORK/models/oellm-9b-yarn-v2-32k` (already present).

---

## Running training

### Sanity check (10 steps, random init)

Validates that Megatron loads, the data pipeline works, and TP=2 PP=2 runs cleanly:

```bash
cd ~/openeuro-longctx-datamix
sbatch lumi/slurm/train_smoke_test_leonardo.sbatch
```

Expected output after ~2 min init:
```
iteration  1/10  lm loss: 13.31  throughput: ~10 TFLOP/s/GPU  (JIT warmup)
iteration  2/10  lm loss: 13.12  throughput: ~58 TFLOP/s/GPU
...
iteration 10/10  lm loss: 12.74
=== Smoke test done ===
```

### Tiny training run (100 steps, GBS=32)

Validates checkpoint saving and the full training loop at realistic batch size:

```bash
sbatch lumi/slurm/train_tiny_leonardo.sbatch
```

- Wall time: ~50 min
- Checkpoints saved at iter 50 and 100 to `$SCRATCH/megatron-tiny-<jobid>/checkpoints/`
- TensorBoard logs in `$SCRATCH/megatron-tiny-<jobid>/tensorboard/`

### Monitoring jobs

```bash
squeue -u pmoell00                        # live queue
tail -f $SCRATCH/megatron-tiny-<jobid>.out  # stream output
```

An iteration line looks like:
```
[2026-06-08 12:22:13] iteration   2/100 | consumed samples: 8 | elapsed time per iteration (ms): 3723 |
  throughput per GPU (TFLOP/s/GPU): 56.7 | lm loss: 1.31E+01 | grad norm: 36.3 |
```

---

## Flags required without Apex / TransformerEngine

Leonardo's bare CUDA environment (no Apex, no TE) needs five fusion features disabled. These are set in both sbatch files and do not affect training correctness — only throughput:

| Flag | Why |
|------|-----|
| `--no-rope-fusion` | `apply_rope_fusion` requires TransformerEngine ≥ 1.4 |
| `--no-persist-layer-norm` | `persist_layer_norm` requires Apex FusedLayerNorm |
| `--sequence-parallel` removed | sequence parallel also requires Apex FusedLayerNorm |
| `--no-gradient-accumulation-fusion` | `fused_weight_gradient_mlp_cuda` is an Apex CUDA extension |
| `--no-masked-softmax-fusion` | `scaled_masked_softmax_cuda` is an Apex CUDA extension |

To recover full performance, install Apex with CUDA extensions on a compute node (requires internet access or pre-built wheels), or use the NVIDIA Megatron container which includes both Apex and TE.

**Throughput impact without fused kernels:** ~58 TFLOP/s/GPU vs ~80+ with fused kernels on A100.

---

## Continuing from a LUMI checkpoint

The 1k-step YaRN v2 checkpoint is at `/flash/project_462000963/bmoell/yarn-multilingual-v2-1k/checkpoints/` on LUMI. To continue training from it on Leonardo:

1. Transfer the checkpoint to `$WORK/checkpoints/yarn-v2-1k/` (it is in Megatron distributed format, ~70 GB for a 9B model with TP=2 PP=4)
2. Add `--load $WORK/checkpoints/yarn-v2-1k` to the sbatch
3. Adjust `--train-iters` to the new target (e.g. 2000 for another 1k steps)

Note: the LUMI checkpoint uses PP=4; Leonardo uses PP=2. A checkpoint conversion is required before loading. Use `convert_checkpoint.sbatch` in `lumi/slurm/` as a starting point.

---

## Storage summary

| Path | Use | Purge |
|------|-----|-------|
| `$WORK/NVIDIA-Megatron-LM/` | Megatron codebase | Never |
| `$WORK/megatron_venv/` | Python venv | Never |
| `$WORK/data/` | Tokenized .bin/.idx datasets | Never |
| `$WORK/models/` | Tokenizer, HF model | Never |
| `$WORK/hf_cache/` | HuggingFace cache | Never |
| `$SCRATCH/megatron-*/` | Job outputs, checkpoints | After 40 days |

Move important checkpoints from `$SCRATCH` to `$WORK` or back to LUMI before the 40-day purge.
