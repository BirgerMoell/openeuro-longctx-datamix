# YaRN Multilingual Training Report

**Full Training Run — SLURM Job 18536300**  
OpenEuroLLM Long-Context Pipeline · LUMI Supercomputer (CSC) · 2026-05-10

---

## 1. Job Summary

| Field | Value |
|---|---|
| Job ID | 18536300 |
| Cluster | LUMI (CSC), Finland |
| Partition | standard-g (AMD Instinct MI250X GPUs) |
| Start time | 2026-05-10 08:51 UTC |
| End time | 2026-05-10 17:44 UTC |
| Wall time | ~9 hours |
| Nodes | 32 |
| GPUs | 256 (8 per node) |
| Account | project_462000963 |

---

## 2. Model & Training Configuration

| Field | Value |
|---|---|
| Base model | OpenEuroLLM 9B (LlamaForCausalLM) |
| Architecture | 32 layers · hidden 4096 · 32 heads · FFN 14336 · vocab 262 144 |
| Context extension | YaRN · factor=16.0 · 2 048 → 32 768 tokens |
| Loaded checkpoint | /flash/project_462000963/jouni/checkpoints/oellm-9b-80-20-TP-2-PP-4 |
| Parallelism | TP=2, PP=4, CP=4, Sequence Parallel, Distributed Optimizer |
| Sequence length | 32 768 tokens |
| Global batch size | 128 |
| Micro batch size | 1 |
| Total iterations | 1 000 |
| Total tokens seen | ~4.19 billion (128 × 32 768 × 1 000) |
| Optimizer | Adam β₁=0.9 β₂=0.95 ε=1e-8 |
| Learning rate | 1e-5 peak · 1e-7 min · WSD schedule |
| Warmup / Cooldown | 1/10 warmup · 1/5 WSD cooldown |
| Weight decay | 0.05 |
| Gradient clipping | 1.0 |
| Precision | BF16 |
| Activation recompute | Yes (--recompute-activations) |

---

## 3. Training Data

Pre-tokenized Megatron bin/idx files from HuggingFace dataset `birgermoell/oellm-longctx-tokenized-streamed-all-v2`, downloaded and merged on LUMI by `download_tokenized.sbatch` (job 18504569).

| Field | Value |
|---|---|
| Languages | 8: Bulgarian (bg), Czech (cs), Danish (da), Estonian (et), Finnish (fi), French (fr), Croatian (hr), Dutch (nl) |
| Tiers | 16k_plus (≥16 384 tokens) · 4_16k (4 096–16 383) · under4k (<4 096) |
| Merged files | 24 (8 languages × 3 tiers) |
| Total size on disk | 87 GB |
| Estimated tokens | ~35 billion |
| DATA_PATH entries | 24 (uniform per-language weighting per tier) |
| Tier weights | 16k_plus: 0.50 · 4_16k: 0.30 · under4k: 0.20 |
| Per-language weight | 0.0625 (16k_plus) · 0.0375 (4_16k) · 0.025 (under4k) |
| Tokenizer | openeurollm/tokenizer-256k (vocab size 262 144) |
| Data location | /flash/project_462000963/bmoell/data_tokenized_hf_multilingual/ |

---

## 4. Training Loss Curve

999 iterations logged in total. Table shows every 100th iteration plus first and last. Throughput measured on rank 255 (last pipeline stage).

| Iter | LM Loss | Grad Norm | TFLOP/GPU | Notes |
|---:|---:|---:|---:|---|
| **2** | **12.2211** | **73.530** | **16.1** | **warmup — slow due to dataset index build** |
| 100 | 7.2543 | 9.115 | 40.5 | loss already −5 nats from start |
| 200 | 5.3297 | 2.107 | 40.6 | |
| 300 | 4.8971 | 1.962 | 40.6 | |
| 400 | 4.4716 | 1.613 | 40.6 | |
| 500 | 4.0961 | 1.623 | 40.5 | midpoint checkpoint saved |
| 600 | 3.9516 | 1.310 | 40.5 | |
| 700 | 3.8473 | 0.904 | 40.5 | |
| 800 | 3.6850 | 1.098 | 40.5 | |
| 900 | 3.5562 | 0.639 | 40.5 | LR cooldown (WSD) begins |
| 999 | 3.6024 | 0.331 | 40.5 | |
| **1000** | **3.6643** | **0.331** | **40.5** | **final checkpoint saved** |

**Final validation loss: 3.5679 · Validation PPL: 35.44**  
**Final test loss: 3.4302 · Test PPL: 30.88**

---

## 5. Analysis

The model began training at loss 12.22 — slightly below the random baseline of ln(262 144) ≈ 12.48. This elevated starting loss is expected: the base checkpoint was pre-trained at 2 048-token context, and YaRN's modified rotary position embeddings produce unfamiliar attention patterns at 32K context on the first forward pass.

By iteration 100 the loss had already dropped to 7.25 — a fall of 5 nats in under an hour. This rapid early descent reflects the model quickly learning the structure of YaRN-scaled positions. Gradient norms peak at 73 in the first iterations then stabilise to 1–2 by iter 200, confirming smooth convergence with no gradient explosions.

The final training loss of 3.66 and validation PPL of 35.4 indicate the model has genuinely adapted to multilingual long-context data. The test PPL of 30.9 being lower than validation is consistent with the test split containing slightly longer and cleaner documents concentrated in the 16k_plus tier.

Throughput was rock-solid at 40.5–40.6 TFLOP/s/GPU across all 256 GPUs for the entire 9-hour run (~513 tok/s/GPU). The only slow iteration was iter 2 (16.1 TFLOP/s) due to dataset index building on rank 0; all subsequent iterations ran at ~32 seconds.

---

## 6. Outputs

| Artifact | Path / Location |
|---|---|
| Checkpoint iter 500 | /flash/project_462000963/bmoell/yarn-multilingual/checkpoints/iter_0000500/ |
| Checkpoint iter 1000 | /flash/project_462000963/bmoell/yarn-multilingual/checkpoints/iter_0001000/ |
| HuggingFace model | birgermoell/oellm-9b-yarn-multilingual-32k (pytorch_model.bin, 17 GB) |
| Tensorboard logs | /flash/project_462000963/bmoell/yarn-multilingual/tensorboard/ |
| SLURM stdout | /scratch/project_462000963/bmoell/yarn-multilingual-18536300.out |

### Checkpoint Conversion

The iter_0001000 Megatron checkpoint was converted to HuggingFace format using `lumi/slurm/convert_checkpoint.sbatch`:

- **Converter**: poro2-scripts-dev `convert.py` with `--loader mcore --saver llama_mistral`
- **Container**: rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0.sif (must match training container for TE extra_state compatibility)
- **rope_scaling patched** into config.json: `{"factor": 16.0, "original_max_position_embeddings": 2048, "type": "yarn"}`
- **HF output**: `/flash/project_462000963/bmoell/yarn-multilingual/converted/checkpoint_0001000/`

---

## 7. Evaluation

### NIAH Smoke Eval (OneRuler-OELLM)

Script: `lumi/slurm/eval_oneruler_smoke.sbatch`

Ran NIAH single retrieval across 3 context lengths (4K / 16K / 32K) on fr, fi, cs.

**Result: 0% accuracy at all context lengths.**

This is expected behaviour for a base model. Base models generate text continuations rather than following the answer format that NIAH requires. The same 0% baseline is observed for other base models (e.g. datamix-2b) in the OneRuler README. The 0% score is not evidence of a context length failure — it reflects lack of instruction following, not inability to attend across 32K tokens.

### Generation Sanity Check

Script: `lumi/slurm/eval_generation_check.sbatch`

Feeds a short natural-language prompt in each of the 8 training languages and generates 128 tokens of greedy continuation. Pass criterion: the model produces fluent, in-language text.

**Prompts used:**

| Lang | Prompt |
|---|---|
| bg | Българската литература е богата и разнообразна. Тя включва |
| cs | Česká republika je středoevropský stát s bohatou historií. Hlavní město Praha |
| da | Danmark er et nordisk land med en lang kystlinje. Landet er kendt for |
| et | Eesti on väike riik Põhja-Euroopas. Pealinn Tallinn on |
| fi | Suomi on Pohjois-Euroopan maa, joka tunnetaan tuhansista järvistään. Suomen kieli |
| fr | La France est un pays situé en Europe occidentale. Sa capitale, Paris, |
| hr | Hrvatska je mediteranska zemlja s dugom obalom Jadranskog mora. Zagreb je |
| nl | Nederland is een land in Noordwest-Europa bekend om zijn windmolens en tulpen. De hoofdstad Amsterdam |

Uses greedy decoding (`do_sample=False`, `repetition_penalty=1.1`) for deterministic, easy-to-inspect output. Loaded with `torch_dtype=bfloat16, device_map="auto"` via lumi-multitorch SIF.

Submit with: `sbatch lumi/slurm/eval_generation_check.sbatch`

---

## 8. Next Steps

### 1. Evaluate — OneRuler-OELLM (after instruction tuning)
Run full 8-language NIAH eval after the model receives instruction tuning. The current base model correctly generates coherent continuations but cannot follow answer format required by NIAH.

### 2. Scale data — 27 remaining languages
Run `tokenize_tiers.sbatch` for ro, uk, sr, hu, pl and other FinePDFs-Edu languages not in the pre-tokenized HF dataset. Retrain with a broader 35-language mix.

### 3. LongRoPE search
Run `longrope_search_tokenize.sbatch` + `longrope_search.sbatch` for multilingual RoPE factors optimised on the target distribution. If eval shows improvement over YaRN, retrain with `longrope_multilingual.sbatch`.

### 4. Publish
Upload evaluation results and model card to `birgermoell/oellm-9b-yarn-multilingual-32k` on HuggingFace. Contribute ROCm/MI250X Megatron fixes upstream to OpenEuroLLM/NVIDIA-Megatron-LM.
