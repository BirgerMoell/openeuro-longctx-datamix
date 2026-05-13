# OpenEuroLLM Long-Context Training — Project Summary

**Model:** `birgermoell/oellm-9b-yarn-multilingual-32k`  
**Base:** OpenEuroLLM 9B · **Context:** 2 048 → 32 768 tokens (YaRN)  
**Cluster:** LUMI Supercomputer (CSC, Finland) · AMD Instinct MI250X  
**Last updated:** 2026-05-13

---

## 1. Overview

This project extends the OpenEuroLLM 9B base model from its native 2 048-token context window to 32 768 tokens using YaRN (Yet another RoPE extensioN). The training targets eight European languages from the FinePDFs-Edu corpus and is designed to support long-document understanding tasks across the OpenEuroLLM language set.

The pipeline covers four phases:
1. Data preparation — tokenize and tier multilingual text by document length
2. YaRN training — continued pre-training at 32K context on 256 GPUs
3. Checkpoint conversion — Megatron-LM → HuggingFace format
4. Evaluation — base-LM forced-choice NIAH across context lengths and languages

---

## 2. Training Data

### Source
Pre-tokenized Megatron bin/idx files from HuggingFace:  
`birgermoell/oellm-longctx-tokenized-streamed-all-v2`

Downloaded and merged on LUMI by `download_tokenized.sbatch` (job 18504569).

### Languages (8)
Bulgarian (bg) · Czech (cs) · Danish (da) · Estonian (et) · Finnish (fi) · French (fr) · Croatian (hr) · Dutch (nl)

### Length tiers and weighting

| Tier | Token range | Weight per language | Purpose |
|---|---|---|---|
| `16k_plus` | ≥ 16 384 tokens | 0.0625 | Primary long-context signal |
| `4_16k` | 4 096 – 16 383 | 0.0375 | Medium-length documents |
| `under4k` | < 4 096 | 0.0250 | Short documents for fluency |

- 24 merged files (8 languages × 3 tiers)
- 87 GB on disk, ~35 billion estimated tokens
- Tokenizer: `openeurollm/tokenizer-256k` (vocab 262 144)

### Document count at the 32K boundary
A scan of the raw JSONL files reveals how many documents are natively ≥ 32 768 tokens:

| lang | total | ≥32K | 16–32K | 4–16K | <4K | %≥32K |
|------|------:|-----:|-------:|------:|----:|------:|
| bg | 38 312 | 2 614 | 2 688 | 10 279 | 22 731 | 6.8% |
| cs | 77 038 | 4 041 | 5 138 | 16 550 | 51 309 | 5.2% |
| da | 73 860 | 3 469 | 4 224 | 15 098 | 51 069 | 4.7% |
| et | 34 429 | 2 105 | 2 631 | 8 496 | 21 197 | 6.1% |
| fi | 46 199 | 4 351 | 3 746 | 9 891 | 28 211 | 9.4% |
| fr | 74 122 | 2 053 | 2 870 | 13 991 | 55 208 | 2.8% |
| hr | 43 638 | 4 705 | 5 074 | 10 460 | 23 399 | 10.8% |
| nl | 70 914 | 2 056 | 3 542 | 15 037 | 50 279 | 2.9% |
| **ALL** | **458 512** | **25 394** | **29 913** | **99 802** | **303 403** | **5.5%** |

25 394 documents (~5.5%) are natively ≥ 32K tokens. The existing `16k_plus` bin/idx files already contain all of these; no re-tokenization is needed for a 64K training run.

---

## 3. YaRN Training Run

**SLURM Job:** 18536300  
**Script:** `lumi/slurm/yarn_multilingual.sbatch`

### Hardware
| Field | Value |
|---|---|
| Nodes | 32 |
| GPUs | 256 (8 × AMD Instinct MI250X per node) |
| Start | 2026-05-10 08:51 UTC |
| End | 2026-05-10 17:44 UTC |
| Wall time | ~9 hours |

### Model configuration
| Field | Value |
|---|---|
| Base model | OpenEuroLLM 9B (LlamaForCausalLM) |
| Architecture | 32 layers · hidden 4096 · 32 heads · FFN 14336 · vocab 262 144 |
| Loaded checkpoint | oellm-9b-80-20-TP-2-PP-4 |
| Context extension | YaRN · factor=16.0 · 2 048 → 32 768 tokens |
| Parallelism | TP=2, PP=4, CP=4 · Sequence Parallel · Distributed Optimizer |
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
| Activation recompute | Yes |

### Training loss

| Iter | LM Loss | Grad norm | TFLOP/GPU | Notes |
|---:|---:|---:|---:|---|
| **2** | **12.2211** | 73.530 | 16.1 | warmup — dataset index build |
| 100 | 7.2543 | 9.115 | 40.5 | −5 nats from start in ~45 min |
| 200 | 5.3297 | 2.107 | 40.6 | |
| 300 | 4.8971 | 1.962 | 40.6 | |
| 400 | 4.4716 | 1.613 | 40.6 | |
| 500 | 4.0961 | 1.623 | 40.5 | midpoint checkpoint saved |
| 600 | 3.9516 | 1.310 | 40.5 | |
| 700 | 3.8473 | 0.904 | 40.5 | |
| 800 | 3.6850 | 1.098 | 40.5 | |
| 900 | 3.5562 | 0.639 | 40.5 | LR cooldown (WSD) begins |
| **1000** | **3.6643** | 0.331 | 40.5 | final checkpoint saved |

**Validation loss: 3.5679 · Val PPL: 35.44**  
**Test loss: 3.4302 · Test PPL: 30.88**

Throughput was stable at 40.5–40.6 TFLOP/s/GPU (~513 tok/s/GPU) across all 256 GPUs for the full 9-hour run. The only slow iteration was iter 2 (16.1 TFLOP/s) due to dataset index building on rank 0.

### Outputs
| Artifact | Location |
|---|---|
| Checkpoint iter 500 | `/flash/project_462000963/bmoell/yarn-multilingual/checkpoints/iter_0000500/` |
| Checkpoint iter 1000 | `/flash/project_462000963/bmoell/yarn-multilingual/checkpoints/iter_0001000/` |
| HuggingFace model | `birgermoell/oellm-9b-yarn-multilingual-32k` |
| Tensorboard | `/flash/project_462000963/bmoell/yarn-multilingual/tensorboard/` |

---

## 4. Checkpoint Conversion

**Script:** `lumi/slurm/convert_checkpoint.sbatch`

The iter_0001000 Megatron checkpoint was converted to HuggingFace format using `poro2-scripts-dev convert.py` with `--loader mcore --saver llama_mistral`.

Key requirements discovered during conversion:
- Must use the **same container as training** (`rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0.sif`) — TransformerEngine `extra_state` serialized by te-2.4.0 cannot be read by older TE versions
- Must set `PYTHONPATH=$PORO2_MEGATRON` and `--megatron-path` — the container does not include Megatron in its Python path

After conversion, `rope_scaling` was patched into `config.json`:
```json
"rope_scaling": {"factor": 16.0, "original_max_position_embeddings": 2048, "type": "yarn"},
"rope_theta": 10000
```

HF output: `/flash/project_462000963/bmoell/yarn-multilingual/converted/checkpoint_0001000/`  
Uploaded to: `birgermoell/oellm-9b-yarn-multilingual-32k` on HuggingFace

---

## 5. Evaluations

### 5.1 Generation Sanity Check

**Script:** `lumi/slurm/eval_generation_check.sbatch`

A quick sanity check that the converted HF checkpoint generates coherent text. Feeds a short natural-language prompt in each of the 8 training languages and generates 128 tokens of greedy continuation (no sampling, `repetition_penalty=1.1`).

**Purpose:** Verify the model produces fluent in-language text before investing in evals. No instruction following required — tests basic generation quality.

| Lang | Prompt |
|---|---|
| bg | Българската литература е богата и разнообразна. Тя включва |
| cs | Česká република je středoevropský stát s bohatou historií. Hlavní město Praha |
| da | Danmark er et nordisk land med en lang kystlinje. Landet er kendt for |
| et | Eesti on väike riik Põhja-Euroopas. Pealinn Tallinn on |
| fi | Suomi on Pohjois-Euroopan maa, joka tunnetaan tuhansista järvistään. Suomen kieli |
| fr | La France est un pays situé en Europe occidentale. Sa capitale, Paris, |
| hr | Hrvatska je mediteranska zemlja s dugom obalom Jadranskog mora. Zagreb je |
| nl | Nederland is een land in Noordwest-Europa bekend om zijn windmolens en tulpen. De hoofdstad Amsterdam |

*Results: not yet captured in log — to be updated.*

---

### 5.2 OneRuler-OELLM NIAH (generation-based)

**Script:** `lumi/slurm/eval_oneruler_smoke.sbatch`  
**Job:** run during evaluation phase

OneRuler-OELLM is a multilingual NIAH benchmark for the OpenEuroLLM language set. The standard evaluation asks the model to generate an answer in a specific format (`<Answer>...</Answer>`).

**Result: 0% accuracy at 4K, 16K, and 32K context — all languages.**

This is the expected result for a base model. Base models generate text continuations; they do not follow the answer format required by the benchmark. The same 0% baseline is observed for all other base models in the OneRuler README (e.g. datamix-2b).

**Interpretation:** This is not evidence of a context-length failure. The model is generating plausible continuations rather than the bracketed answer format. Instruction tuning is required before generation-based evals are meaningful.

---

### 5.3 Base-LM NIAH — Forced-Choice Log-Likelihood

**Script:** `scripts/eval_base_lm_niah.py` · `lumi/slurm/eval_base_lm_niah.sbatch`  
**Job:** 18604459 · LUMI (CSC) · 2026-05-13  
**Full methodology:** `docs/eval_base_lm_niah.md`

#### Why this eval

Standard NIAH requires the model to generate an answer in a specific format, which only works after instruction tuning. This eval removes that dependency entirely by using **forced-choice log-likelihood scoring**: instead of generating, the model scores each candidate answer and picks the one with the highest probability.

#### How it works

1. **Context:** A document is generated in the target language containing dozens of key→value "magic number" fact pairs. Example (French):
   ```
   Voici un ensemble de faits.
   
   Le nombre magique spécial pour « river » est : 3827461.
   Le nombre magique spécial pour « forest » est : 9041823.
   ...
   Le nombre magique spécial pour « apple » est : 7319420.   ← query needle
   ...
   ```

2. **Prefix:** A plain base-LM completion stub is appended (no instruction format):
   ```
   Le nombre magique spécial pour « apple » est :
   ```

3. **Candidates:** 4 options — the true value plus 3 distractors. **All four values appear somewhere in the context attached to different keys.** The model must read and bind the correct key, not just recognise a plausible number.

4. **Scoring:** For each candidate C:
   ```
   score(C) = Σⱼ log P(tokenⱼ | context + prefix + C[0:j])
   ```
   One forward pass per candidate. The candidate with the highest score is the prediction.

5. **Grid:** 5 context lengths × 5 needle depths × 4 languages × 10 trials = 1 000 predictions (main grid) plus 3 controls per language.

6. **Controls:**
   - `no_context` — no context document → model guesses → expected ~25%
   - `shuffled` — key/value bindings rotated → correct answer changes → tests that model reads context
   - `short_ctx` — 256-token context → easy baseline to confirm scoring works

#### Results (SLURM job 18604459, partial — job still running)

**Accuracy by context length and needle depth (FR)**

| depth | 2048 | 4096 | 8192 | 16384 | 32768 |
|------:|-----:|-----:|-----:|------:|------:|
| 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.20** |
| 0.25 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | — |
| 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |

*(— = job still running at time of writing)*

**Summary table (all depths averaged per context length)**

| lang | 2048 | 4096 | 8192 | 16384 | 32768 |
|------|-----:|-----:|-----:|------:|------:|
| fr | 1.00 | 1.00 | 1.00 | 1.00 | — |
| fi | — | — | — | — | — |
| cs | — | — | — | — | — |
| nl | — | — | — | — | — |

#### Key finding so far

The model retrieves perfectly (100%) at **every context length from 2K to 16K**, across all needle depths. At **32K there is a positional failure at depth=0.00 only** (acc=0.20, near random chance). Depths 0.25–0.50 at 32K remain 100%.

This is a "lost at the very beginning" effect: when the needle sits at position 0 of a 32K context and the query is at the end, the distance is the maximum the model was trained on (~32K tokens). The YaRN RoPE interpolation degrades at exactly this boundary. From depth=0.25 onward (~24K tokens from the query), retrieval is perfect.

**Practical implication:** The model has a reliable retrieval window of approximately **24K tokens**, not 32K. The last ~8K token "slot" at the very beginning of the context is weakly attended at the training limit.

---

## 6. Known Bugs Fixed During Eval

| Bug | Symptom | Fix |
|---|---|---|
| `torch.OutOfMemoryError` at 32K | `log_softmax` on full `[32768, 262144]` logits ≈ 17 GB | Extract only completion-position rows before softmax |
| Shuffled control duplicate candidate | True candidate appeared twice in candidate list | Use `filler_kvs[1:3]` not `distractor_kvs[:3]` as distractors |
| 32K context buffer too tight | Total input could exceed 32768 tokens by ~1 token | Increased buffer from -20 to -50 tokens in context builder |
| JSONL written only at end of language | Crash mid-language lost all partial results | Write and flush incrementally after every trial |

---

## 7. Next Steps

### Immediate
- **Complete eval job 18604459** — get full results for FR 32K depths 0.75/1.00, controls, and languages fi/cs/nl
- **Update this document** with final summary table and control results

### Short term
- **64K context extension** — the existing `16k_plus` bin/idx files already contain all 25K natively-long documents. No re-tokenization needed. Training changes: `--yarn-scaling-factor 32.0`, `--seq-length 65536`, `--context-parallel-size 8`
- **More training languages** — tokenize remaining 27 FinePDFs-Edu languages (ro, uk, sr, hu, pl, …) and retrain with 35-language mix
- **LongRoPE search** — use `longrope_search.sbatch` to find per-dimension RoPE scaling factors optimised on the multilingual distribution; may outperform vanilla YaRN beyond 32K

### After instruction tuning
- **Full OneRuler-OELLM eval** — run `eval_oneruler_smoke.sbatch` on the instruction-tuned model
- **OneRuler base-LM adapter** — `scripts/oneruler_score_base_lm.py` can also re-score existing OneRuler JSONL using log-likelihood, without SFT

---

## 8. Repository Structure

```
lumi/slurm/
  yarn_multilingual.sbatch        # Full 32-node YaRN training
  convert_checkpoint.sbatch       # Megatron → HuggingFace conversion
  upload_to_hf.sbatch             # Push checkpoint to HuggingFace Hub
  eval_generation_check.sbatch    # Generation sanity check (8 languages)
  eval_oneruler_smoke.sbatch      # OneRuler NIAH (generation-based)
  eval_base_lm_niah.sbatch        # Base-LM NIAH (forced-choice scoring)

scripts/
  eval_base_lm_niah.py            # Main base-LM NIAH evaluator
  oneruler_score_base_lm.py       # OneRuler adapter for base-LM scoring

docs/
  project_summary.md              # This document
  eval_base_lm_niah.md            # Detailed eval methodology
  training_report_job18536300.md  # Training run report (PDF version also available)
```
