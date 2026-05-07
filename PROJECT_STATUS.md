# OpenEuroLLM Long-Context Data Pipeline — Project Status

**Last updated:** 2026-05-07  
**Author:** Birger Moëll (AI Sweden) in collaboration with Jouni Luoma (TurkuNLP)  
**Compute:** LUMI supercomputer (CSC), AMD Instinct MI300X, `standard-g` partition  
**Project account:** `project_462000963`

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
| Megatron-LM running on AMD MI250X/ROCm | job 18306279 — 5 training iterations, loss = ln(262144) ≈ 12.48 (correct random baseline) |
| Real data training working end-to-end | job 18479860 — 20 iterations on Maltese, loss 12.50 → 11.68 (model is learning) |
| Language analysis for all 35 FinePDFs-Edu languages | jobs 18479845 + 18480711 — full stats in `lumi/LANGUAGE_ANALYSIS.md` |
| 3 ROCm/MI250X bugs found and fixed in Megatron | see `lumi/LUMI_STATUS.md` — vocab size, NCCL gloo, FusedLayerNorm apex hang |

### 🔄 In Progress

| Job | ID | Description | Status |
|-----|-----|-------------|--------|
| tokenize_tiers | 18484511 | Split 35-language JSONL into 3 Megatron tier datasets | Queued (`standard-g`) |
| longrope_test | 18484512 | 10-iteration smoke test of LongRoPE 9B pipeline | Queued (`standard-g`) |

### 📋 Ready to Submit (awaiting prerequisites)

| Script | Prerequisite | Description |
|--------|-------------|-------------|
| `yarn_test.sbatch` | None | 10-iteration smoke test of YaRN 9B pipeline (preferred variant) |
| `yarn_multilingual.sbatch` | tokenize_tiers complete | Full 32-node YaRN multilingual training run |
| `longrope_multilingual.sbatch` | tokenize_tiers + LongRoPE factors | Full 32-node LongRoPE multilingual training run |
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
| 2 | Serbian (sr) | 4,407 | 1,313 | Highest p90 (51K) |
| 3 | Romanian (ro) | 4,897 | 1,197 | Most ≥32K of all languages |
| 4 | Hungarian (hu) | 4,482 | 1,050 | Median already ≥4K |
| 5 | Russian (ru) | 3,576 | 944 | Large corpus |

Full per-language table: `lumi/LANGUAGE_ANALYSIS.md`

**Three-tier Megatron dataset structure** (matching Jouni's English setup):

```
multilingual_16k_plus_text_document.{bin,idx}    # docs ≥ 16384 tokens  →  weight 0.5
multilingual_4_16k_text_document.{bin,idx}        # docs 4096–16383     →  weight 0.3
multilingual_under4k_text_document.{bin,idx}      # docs < 4096         →  weight 0.2
```

---

## The Model

**Base:** OpenEuroLLM 9B — 32 layers, hidden size 4096, 32 attention heads, vocab 256K  
**Checkpoint:** `/flash/project_462000963/jouni/checkpoints/oellm-9b-80-20-TP-2-PP-4` (TP=2, PP=4 sharded)  
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

For LongRoPE, replace `"type": "yarn"` with `"type": "longrope"` and add the `long_factor` / `short_factor` arrays from the search result. Reference: `/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-9b-80-20-TP-2-PP-4-yarn-finepdfs/checkpoint_0001000/config.json`

---

## The Plan Forward

### Phase 1 — Smoke test and first multilingual run (this week)

1. **Wait for job 18484511** (tokenize_tiers) to complete → verify 3 bin/idx files exist in `/flash/project_462000963/bmoell/data_tokenized_multilingual/`
2. **Submit `yarn_test.sbatch`** — 10-iteration test of the full 9B YaRN pipeline on 1 node. Confirms the checkpoint loads, YaRN args work, and training doesn't crash before committing 32 nodes.
3. **If yarn_test passes → submit `yarn_multilingual.sbatch`** — 32 nodes, 256 GPUs, ~1000 iterations (~24h). This is the first real multilingual long-context training run.

### Phase 2 — LongRoPE factors for multilingual (parallel with Phase 1)

4. **Submit `longrope_search_tokenize.sbatch`** — builds a multilingual proxy dataset (14 languages, ~1100 docs ≥32K) in HuggingFace Arrow format.
5. **Submit `longrope_search.sbatch`** — 4 nodes × 48h genetic algorithm search. Produces `result_final.csv` with per-frequency RoPE factors optimized for the multilingual distribution.
6. **Update `longrope_multilingual.sbatch`** with the new CSV path, then submit if YaRN results are unsatisfying.

### Phase 3 — Scale and evaluate

7. **Download more shards** — each language currently uses 1 shard. Top languages (ro, uk, sr, hu, pl) have 5–10+ shards available. Downloading 3–5 shards each would multiply the ≥32K dataset ~4×.
8. **Evaluate on long-context benchmarks** — once a trained checkpoint is converted to HuggingFace, run standard long-context eval (e.g. RULER, LongBench, Needle-in-a-Haystack).
9. **3 missing languages** — Irish (ga), Albanian (sq), Luxembourgish (lb) need `HF_TOKEN` set for HPLT fetch. Token is now stored on LUMI; re-run `lang_analysis_remaining.sbatch` to pick them up.

---

## Infrastructure

### LUMI Paths

| Resource | Path |
|----------|------|
| Repo (scratch) | `/scratch/project_462000963/bmoell/openeuro-longctx-datamix` |
| Megatron-LM | `/flash/project_462000963/jouni/test/NVIDIA-Megatron-LM` |
| Base checkpoint (Megatron) | `/flash/project_462000963/jouni/checkpoints/oellm-9b-80-20-TP-2-PP-4` |
| Tokenizer | `/flash/project_462000963/jouni/checkpoints/converted-checkpoints/oellm-datamix-9b-80-20` |
| LongRoPE search tool | `/flash/project_462000963/jouni/test/poro2-longrope-search` |
| Multilingual tokenized data (output) | `/flash/project_462000963/bmoell/data_tokenized_multilingual/` |
| YaRN training output | `/flash/project_462000963/bmoell/yarn-multilingual/` |
| LongRoPE search output | `/flash/project_462000963/bmoell/longrope-search/` |
| Container (training) | `/scratch/project_462000963/containers/staging/rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0.sif` |

### Scripts in `lumi/slurm/`

| Script | Purpose | Nodes | Time |
|--------|---------|-------|------|
| `tokenize_tiers.sbatch` | Split 35 JSONL → 3 Megatron tier datasets | 1 | 8h |
| `yarn_test.sbatch` | YaRN 9B smoke test (10 iters) | 1 | 30min |
| `yarn_multilingual.sbatch` | YaRN 9B multilingual full run (~1000 iters) | 32 | 24h |
| `longrope_test.sbatch` | LongRoPE 9B smoke test (10 iters) | 1 | 30min |
| `longrope_multilingual.sbatch` | LongRoPE 9B multilingual full run | 32 | 48h |
| `longrope_search_tokenize.sbatch` | Build multilingual proxy dataset for search | 1 | 4h |
| `longrope_search.sbatch` | Genetic search for multilingual RoPE factors | 4 | 48h |
| `train_real.sbatch` | Validated: 20-iter Maltese real-data training | 1 | 45min |
| `lang_analysis.sbatch` | Language stats — 16 languages (done) | 1 | 2h55 |
| `lang_analysis_remaining.sbatch` | Language stats — remaining 19 languages (done) | 1 | 2h55 |

---

## Key Technical Findings

### ROCm/MI250X Compatibility Fixes

Three bugs had to be diagnosed and patched in Megatron-LM to run on LUMI's AMD GPUs:

1. **Vocab size**: `--vocab-size 262144` required (model has 256K vocab, Megatron defaults to smaller)
2. **NCCL on ROCm**: `--distributed-backend gloo` for single-node (NCCL hangs on ROCm without proper device binding)
3. **FusedLayerNorm**: `apex.normalization.FusedLayerNormAffineFunction` deadlocks on MI250X for hidden sizes not in its hardcoded list. Fix: replace with `torch.nn.functional.layer_norm` in the `else` branch of `FusedLayerNorm.forward()`.

Fixes are in branch `rocm-mi250x-compat` on `BirgerMoell/NVIDIA-Megatron-LM`. Should be contributed upstream to `OpenEuroLLM/NVIDIA-Megatron-LM`.

### PYTHONPATH Fix

The container ships an older `megatron-core` pip package. Always set inside the singularity heredoc:

```bash
export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
```

### Why YaRN over LongRoPE for Multilingual

LongRoPE's per-frequency scaling factors are found by running a genetic algorithm search over the target corpus. Jouni's existing factors were searched on English-only data. Applying English-tuned factors to 35 languages is suboptimal. YaRN uses a single mathematically-derived scaling factor (16.0) that is language-agnostic. The multilingual LongRoPE search pipeline (`longrope_search_tokenize.sbatch` + `longrope_search.sbatch`) will produce better-tuned factors when complete.

---

## Related Resources

- **OpenEuroLLM project:** https://openeurollm.eu/
- **Megatron fork with ROCm fixes:** `BirgerMoell/NVIDIA-Megatron-LM` branch `rocm-mi250x-compat`
- **FinePDFs-Edu dataset:** `HuggingFaceFW/finepdfs-edu`
- **LongRoPE search tool:** `poro2-longrope-search` (multinode fork of Microsoft LongRoPE)
- **Jouni Luoma (TurkuNLP):** `/flash/project_462000963/jouni/` — reference scripts and checkpoints
