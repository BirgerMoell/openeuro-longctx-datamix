# OpenEuroLLM Long-Context Data Pipeline — Project Status

**Last updated:** 2026-05-07  
**Author:** Birger Moëll (AI Sweden) in collaboration with Jouni Luoma (TurkuNLP)  
**Compute:** LUMI supercomputer (CSC), AMD Instinct MI300X, `standard-g` partition

---

## What This Project Is

We are building and validating the **multilingual long-context continual pre-training pipeline** for the [OpenEuroLLM](https://openeurollm.eu/) 9B model. The goal is to extend the model's context window from 2K to 32K tokens using real multilingual data from 35 European languages.

The pipeline has two main components:

1. **Data pipeline** — download, convert, and tokenize FinePDFs-Edu data into length-stratified Megatron training datasets
2. **Training** — continued pre-training of the OpenEuroLLM 9B model with extended context using YaRN or LongRoPE position embedding scaling

---

## Current State (2026-05-07)

### ✅ Fully Validated

| Milestone | Evidence |
|-----------|----------|
| Megatron-LM running on AMD MI250X/ROCm | 5 training iterations, loss = ln(262144) ≈ 12.48 (correct random baseline) |
| Real data training working end-to-end | 20 iterations on Maltese, loss 12.50 → 11.68 (model is learning) |
| Language analysis for all 35 FinePDFs-Edu languages | Full stats in `lumi/LANGUAGE_ANALYSIS.md` |
| 3 ROCm/MI250X bugs found and fixed in Megatron | See `lumi/LUMI_STATUS.md` — vocab size, NCCL gloo, FusedLayerNorm apex hang |
| Mini multilingual tokenization (3 tiers, 35 languages) | `tokenize_tiers_mini.sbatch` produces valid Megatron bin/idx files using lumi-multitorch SIF |

### 🔄 In Progress

| Job | Description | Status |
|-----|-------------|--------|
| yarn_multilingual_test (18494663) | 10-iteration smoke test of YaRN 9B on multilingual 16k_plus data | Running on `dev-g` |

### 📋 Ready to Submit (awaiting smoke test)

| Script | Prerequisite | Description |
|--------|-------------|-------------|
| `yarn_multilingual.sbatch` | smoke test pass | Full 32-node YaRN multilingual training run |
| `longrope_search_tokenize.sbatch` | None | Build multilingual proxy dataset for LongRoPE search |
| `longrope_search.sbatch` | longrope_search_tokenize complete | Genetic algorithm search for multilingual RoPE factors |

---

## The Data

**Source:** `HuggingFaceFW/finepdfs-edu` — PDF-extracted corpus (research papers, legal documents, government reports). 35 of 38 OpenEuroLLM target languages. 3 missing (ga/sq/lb) require HPLT.

**Scale (1 sample shard per language):**

| Metric | Value |
|--------|-------|
| Total documents | 1,865,725 |
| Total tokens | 18.25 billion |
| Documents ≥ 4K tokens | 653,136 (35%) |
| Documents ≥ 32K tokens | 107,758 (5.8%) |
| Documents ≥ 128K tokens | 18,246 (1.0%) |

**Top languages for long-context training** (by ≥128K docs per shard):

| Rank | Language | ≥32K docs | ≥128K docs | Notable |
|------|----------|-----------|-----------|---------|
| 1 | Ukrainian (uk) | 4,217 | 1,347 | High right tail |
| 2 | Serbian (sr) | 4,407 | 1,313 | Highest p90 (51K tokens) |
| 3 | Romanian (ro) | 4,897 | 1,197 | Most ≥32K of all languages |
| 4 | Hungarian (hu) | 4,482 | 1,050 | Median already ≥4K |
| 5 | Russian (ru) | 3,576 | 944 | Large corpus |

Full per-language table: `lumi/LANGUAGE_ANALYSIS.md`

**Three-tier Megatron dataset structure** (matching Jouni Luoma's English setup):

```
multilingual_16k_plus_text_document.{bin,idx}    # docs ≥ 16384 tokens  →  weight 0.5
multilingual_4_16k_text_document.{bin,idx}        # docs 4096–16383     →  weight 0.3
multilingual_under4k_text_document.{bin,idx}      # docs < 4096         →  weight 0.2
```

**Mini sample (smoke test):** `tokenize_tiers_mini.sbatch` produces ~175 docs in the 16k_plus tier
from 50 docs × 35 languages, using the lumi-multitorch SIF (not the rocm container, which has a
fork-unsafe HuggingFace tokenizer). Files are in `/flash/project_462000963/bmoell/data_tokenized_multilingual/`.

---

## The Model

**Base:** OpenEuroLLM 9B — 32 layers, hidden size 4096, 32 attention heads, vocab 256K  
**Context extension:** 2K → 32K tokens (scaling factor 16.0)

### Context Extension Methods

| Method | Megatron args | Pros | Cons |
|--------|--------------|------|------|
| **YaRN** (recommended now) | `--position-embedding-type yarn --yarn-scaling-factor 16.0 --yarn-original-max-position-embeddings 2048` | Single parameter, no search, language-agnostic | Slightly lower quality than LongRoPE on its target distribution |
| **LongRoPE** (preferred long-term) | `--position-embedding-type longrope --longrope-rescale-factors-path result_final.csv --longrope-original-max-position-embeddings 2048` | Higher quality when factors searched on target distribution | Requires per-distribution genetic algorithm search (~4 nodes × 48h) |

**Current plan:** Run YaRN now for a working multilingual model. Run the LongRoPE search on a multilingual proxy corpus in parallel; if results are better, retrain with the new factors.

### HuggingFace Conversion — Manual Step

After converting a Megatron checkpoint to HuggingFace format, **add to `config.json` by hand**:

```json
"rope_scaling": {
  "factor": 16.0,
  "original_max_position_embeddings": 2048,
  "type": "yarn"
},
"rope_theta": 10000
```

For LongRoPE, replace `"type": "yarn"` with `"type": "longrope"` and add the `long_factor` / `short_factor` arrays from the search result. The conversion script does not propagate these parameters automatically.

---

## The Plan Forward

### Phase 1 — Smoke test and first multilingual run (this week)

1. **yarn_multilingual_test (18494663)** currently running on dev-g — 10 iterations on the 16k_plus
   multilingual tier. Confirms YaRN args, checkpoint loading, and training loss.
2. **If smoke test passes → submit `yarn_multilingual.sbatch`** — 32 nodes, 256 GPUs, ~1000 iterations (~24h).

### Phase 2 — Language-balanced full training (after smoke test)

The current three-tier data pipeline concatenates all 35 languages into per-tier datasets with no
per-language weighting. This gives length-tier weighting but not language weighting — high-resource
languages (ro, uk, sr) dominate. Two fixes to evaluate:

- **Sampling-based**: pre-balance each language before merging with temperature α≈0.3
  (e.g. `scripts/balance_languages.py`)
- **Megatron prefix weighting**: separate per-language-per-tier Megatron data prefixes with explicit
  weights in `DATA_PATH`

The full multilingual run (`yarn_multilingual.sbatch`) should use one of these approaches.

### Phase 3 — LongRoPE factors for multilingual (parallel with Phase 1)

3. **Submit `longrope_search_tokenize.sbatch`** — builds a multilingual proxy dataset (14 languages, ~1100 docs ≥32K) in HuggingFace Arrow format.
4. **Submit `longrope_search.sbatch`** — 4 nodes × 48h genetic algorithm search. Produces `result_final.csv` with per-frequency RoPE factors optimized for the multilingual distribution.
5. **Update `longrope_multilingual.sbatch`** with the new CSV path, then submit if YaRN results are unsatisfying.

### Phase 4 — Scale and evaluate

6. **Download more shards** — each language currently uses 1 shard. Top languages (ro, uk, sr, hu, pl) have 5–10+ shards available. Downloading 3–5 shards each would multiply the ≥32K dataset ~4×.
7. **Evaluate on long-context benchmarks** — once a trained checkpoint is converted to HuggingFace, run standard long-context eval (e.g. RULER, LongBench, Needle-in-a-Haystack).
8. **3 missing languages** — Irish (ga), Albanian (sq), Luxembourgish (lb) are not in FinePDFs-Edu and need to be fetched from HPLT.

---

## Scripts in `lumi/slurm/`

| Script | Purpose | Nodes | Time |
|--------|---------|-------|------|
| `tokenize_tiers_mini.sbatch` | Mini tokenization: 50 docs/lang → 3 Megatron tier datasets | 1 | 10min |
| `tokenize_tiers.sbatch` | Full tokenization: all 35 JSONL → 3 Megatron tier datasets | 1 | 8h |
| `yarn_multilingual_test.sbatch` | YaRN 9B smoke test (10 iters, 16k_plus tier only) | 1 | 1h |
| `yarn_multilingual.sbatch` | YaRN 9B multilingual full run (~1000 iters) | 32 | 24h |
| `longrope_test.sbatch` | LongRoPE 9B smoke test (10 iters) | 1 | 30min |
| `longrope_multilingual.sbatch` | LongRoPE 9B multilingual full run | 32 | 48h |
| `longrope_search_tokenize.sbatch` | Build multilingual proxy dataset for RoPE search | 1 | 4h |
| `longrope_search.sbatch` | Genetic search for multilingual RoPE factors | 4 | 48h |
| `train_real.sbatch` | Validated: 20-iter Maltese real-data training | 1 | 45min |

---

## Key Technical Findings

### ROCm/MI250X Compatibility Fixes

Three bugs had to be diagnosed and patched in Megatron-LM to run on LUMI's AMD GPUs:

1. **Vocab size**: `--vocab-size 262144` required (model has 256K vocab, Megatron defaults to smaller)
2. **NCCL on ROCm**: `--distributed-backend gloo` for single-node (NCCL hangs on ROCm without proper device binding)
3. **FusedLayerNorm**: `apex.normalization.FusedLayerNormAffineFunction` deadlocks on MI250X for hidden sizes not in its hardcoded list. Fix: replace with `torch.nn.functional.layer_norm` in the `else` branch of `FusedLayerNorm.forward()`.

Fixes are in branch `rocm-mi250x-compat` on `BirgerMoell/NVIDIA-Megatron-LM`. Should be contributed upstream to `OpenEuroLLM/NVIDIA-Megatron-LM`.

### Tokenizer Container Fix

The rocm-6.4.4 container's HuggingFace fast tokenizer crashes with "double free or corruption" when
Python's `multiprocessing.Pool` forks after tokenizer init (fork-unsafe). The `lumi-multitorch` SIF
does not have this problem. All tokenization jobs (`tokenize_tiers_mini.sbatch`,
`tokenize_tiers.sbatch`, `longrope_search_tokenize.sbatch`) use the lumi-multitorch SIF.

### BlendedMegatronDataset Hang (smoke test workaround)

When three unequal-sized tiers are provided as a blended `DATA_PATH`, Megatron's
`BlendedMegatronDataset` builder hangs on rank 0 computing sample weights for 30+ minutes.
Other ranks time out at the NCCL watchdog (`--distributed-timeout-minutes 30`) and SIGABRT.

**Smoke test workaround**: use only the `multilingual_16k_plus_text_document` tier (no blending).
The full training run (`yarn_multilingual.sbatch`) will keep the 3-tier blend once the full dataset
is large enough that index building completes quickly.

### GPTDataset Test-Split Starvation (smoke test workaround)

Megatron's default `--split 969,30,1` puts only 0.1% of data in the test partition. With ~175 docs
in the mini dataset, that is < 1 document — the GPTDataset index builder hangs trying to produce
the required `EVAL_ITERS × GLOBAL_BATCH_SIZE` test samples from essentially nothing.

**Smoke test fix**: `--split 750,150,100` (10% test) and `EVAL_ITERS=2`. With 175 docs, test gets
~17.5 documents — enough to produce 16 samples for the final evaluation.

### PYTHONPATH Fix

The LUMI container ships an older `megatron-core` pip package that shadows the local repo. Always set inside the singularity exec:

```bash
export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
```

### Why YaRN over LongRoPE for Multilingual

LongRoPE's per-frequency scaling factors are found by running a genetic algorithm search over the target corpus. The existing factors were searched on English-only data. Applying English-tuned factors to 35 languages is suboptimal. YaRN uses a single mathematically-derived scaling factor (16.0) that is language-agnostic. The multilingual LongRoPE search pipeline (`longrope_search_tokenize.sbatch` + `longrope_search.sbatch`) will produce properly-tuned multilingual factors when complete.

---

## Related Resources

- **OpenEuroLLM project:** https://openeurollm.eu/
- **Megatron fork with ROCm fixes:** `BirgerMoell/NVIDIA-Megatron-LM` branch `rocm-mi250x-compat`
- **FinePDFs-Edu dataset:** `HuggingFaceFW/finepdfs-edu`
- **LongRoPE search tool:** `poro2-longrope-search` (multinode fork of Microsoft LongRoPE)
