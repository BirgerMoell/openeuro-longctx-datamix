# Long-Context Extension Run — baby_9b_dense → 128K (multilingual)

**Started:** 2026-06-18 · **Cluster:** LUMI (AMD MI250X / ROCm) · **Job:** `19345889`
**Status:** RUNNING (real run, ~6k GPU-h). *First launch `19345342` hung on MIOpen kernel
search → cancelled & relaunched with the MIOpen fix below.*

## What we're doing
Extending the OpenEuroLLM **baby_9b_dense** base model (Qwen3 dense ~9B) from its **native 4K**
context to **128K**, on LUMI, with broad **multilingual** coverage so long context works across
all OpenEuroLLM languages. This is continued pretraining (mid-training), *separate from and
before* post-training (SFT/DPO/RLVR, which lives in `qwen35-posttrain`).

## The model
Qwen3 dense (`Qwen3ForCausalLM`), ~9.1B params: 36 layers, hidden 4096, FFN 12288, 32 heads /
8 KV (GQA), head_dim 128, qk-layernorm, RMSNorm, SwiGLU, untied embeddings, vocab 262k
(OpenEuroLLM 256k tokenizer). Native RoPE θ=100000, seq 4096. Loaded from the Megatron
`torch_dist` checkpoint `/flash/project_462000963/training/qwen3_9b_hf_baby_ckpts/megatron`
(iter 76800).

## Method — progressive native ABF (no YaRN)
**Adjusted Base Frequency**: at each stage we (1) load the previous stage's checkpoint with
`--finetune` (weights only, optimizer + iteration reset), (2) raise `--seq-length` and
`--max-position-embeddings`, (3) raise the RoPE base `--rotary-base` (θ) so the rotary
frequencies span the longer context, (4) continued-pretrain on long-context data. **Context
Parallelism (CP)** splits the long sequence across GPUs to fit memory.

No YaRN / LongRoPE — pure θ scaling. (Chosen because it needs no extra code, and the
LongRoPE/YaRN patches aren't in the OELLM Megatron yet.) θ values are NTK-aware, slightly
generous:

| Stage | seq_length | rotary_base (θ) | context-parallel | token budget |
|------:|-----------:|----------------:|:----------------:|-------------:|
| 1 | 16384  | 500,000   | 1 | 4B |
| 2 | 32768  | 1,000,000 | 1 | 3B |
| 3 | 65536  | 2,000,000 | 2 | 2B |
| 4 | 131072 | 5,000,000 | 8 | 1B |

(Base native: θ=100000 @ 4096.) Each stage saves a `torch_dist` checkpoint the next loads.

## Data — all 37 OpenEuroLLM languages
Length-rich, multilingual mix from the **known-good** tokenized catalogue
(`/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/`, 256k tokenizer, verified
`.bin/.idx`):

| share | content |
|------:|---------|
| ~23% | finepdfs **English** (boosted) |
| ~41% | finepdfs **other 36 EU languages** (~1.15% each — als…ukr, every OELLM language) |
| ~6%  | finepdfs-edu (English) |
| ~17% | starcoder (code) |
| ~7%  | finemath (math) |
| ~6%  | nemotron-cc (web) |

FinePDFs = naturally long PDF documents → genuine long-range signal (not just packed shorts).
(We avoided Jouni's `long-ctx-sample` after hitting a corrupt `arxiv.idx` there.)

## Training config
- **16 nodes / 128 GCDs**, `standard-g`, 48h. TP=8, PP=1, CP=(1,1,2,8), sequence-parallel,
  distributed optimizer.
- GBS 64, MBS 1, bf16, selective activation recompute.
- LR 1e-5 → 1e-6 cosine, warmup = iters/20, weight-decay 0.1, clip 1.0.
- `--ckpt-format torch_dist`; full checkpoint saved per stage → `output_real/ckpt_<seq>`.
- Container `laif-rocm-6.4.4-…sif`; Megatron `…/luomajou/oellm-test/NVIDIA-Megatron-LM`.
- **Compute: ~6,000 GPU-h (~2% of the 300k LUMI budget), ~2 days wall.**

### Required ROCm/MIOpen fix (credit: Jouni Luoma)
On AMD, MIOpen auto-tunes kernels for each new tensor shape. At long sequence lengths this
search hangs 30–60+ min and the idle ranks then **NCCL/comms time out → the job dies**. Since
this run changes seq length 4× (16K/32K/64K/128K), we set:
```bash
export MIOPEN_FIND_MODE=FAST                        # pick first working kernel; skip the search
export MIOPEN_USER_DB_PATH=/flash/.../bmoell/miopen_cache   # persistent perf-db across stages/runs
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
```
Without this, the first launch (`19345342`) sat 21 min with no training output and was headed
for the 30-min distributed timeout. The persistent `/flash` cache means tuned shapes are reused
on later stages/runs (Jouni's suggestion: prepopulate the cache once, keep it on /flash).

## Evaluation — base-model mode
The model is a **base** LM (no SFT), so RULER is run in **forced-completion / log-likelihood**
mode on **retrieval** tasks only — NIAH (single/multi key/value/query), Variable Tracking,
Common-Word-Extraction — **skipping QA / instruction tasks** (need instruction-following).
NIAH-as-completion directly measures long-range retrieval, which is the capability we're
extending. Multilingual via OneRuler-OELLM. Queued per stage (`ruler_eval.sh`); needs the
vLLM-ROCm generation stack wired.

## Validation already done (de-risking)
- 16K smoke: baby loads on LUMI/ROCm (TP=4→TP=8 reshard), **loss ~2.24**, fits ~53% mem.
- Tiny pipeline: **16K/32K/64K trained + checkpointed**, sane loss, 0 NaN.
- **128K confirmed working** (standalone, CP=4): loss decreasing 3.41→3.17→2.98, 0 NaN, but
  94% mem → real run uses **CP=8** for headroom. (This was the stage previously failing on LUMI.)

## Scripts (LUMI: `/scratch/project_465002530/users/bmoell/longctx-extend/`)
- `extend_real_to128k.sh` — **this run** (staged, all-language, 10B tokens, CP=(1,1,2,8))
- `extend_pipeline.sh` — tiny 4-stage validation
- `extend_128k_standalone.sh` — single 128K feasibility test
- `ruler_eval.sh` — per-stage base-LM RULER eval (skeleton; needs vLLM-ROCm)

## Language coverage per stage (IMPORTANT)
| Stage | seq | finepdfs languages | total blend sources | notes |
|------:|----:|:------------------:|:-------------------:|-------|
| 16K | 16384  | **37 (all OELLM)** | 41 | full multilingual |
| 32K | 32768  | **37 (all OELLM)** | 41 | full multilingual |
| 64K | 65536  | **37 (all OELLM)** | 41 | full multilingual (verified in job 19380028 log: als…ukr) |
| 128K | 131072 | **37 (all OELLM)** ← definitive run | 41 | *(an interim 128K attempt used a trimmed 8-source mix while chasing an OOM that turned out unrelated; the definitive 128K uses all 37)* |

The 37 finepdfs languages: als bos bul cat ces dan deu ekk ell eng eus fin fra gle glg hrv
hun isl ita kat lit lvs mkd mlt nld nno nob pol por ron slk slv spa srp swe tur ukr — plus
finepdfs-edu (eng), starcoder (code), finemath (math), nemotron-cc (web). English upsampled
20×, every other language floor-weighted (~1.15% each).

## Run history
- `19345342` — hung on MIOpen kernel search → cancelled.
- `19345889` — MIOpen fixed; trained **16K ✓ + 32K ✓** (37 langs, checkpoints saved), then
  **host-RAM OOM at 64K iter ~239** = the mid-stage eval point (eval dataloader over all 41
  sub-datasets spiked CPU RAM; GPU only 69%). Fix: disable mid-stage eval + `--num-workers 1`.
- `19377604` — resume attempt; exit-2 (an inline `#comment` leaked into the arg string) → fixed.
- `19380028` — resumed from `ckpt_32768`; trained **64K ✓ (all 37 langs)** → `ckpt_65536`
  saved, then **host-RAM OOM entering 128K** (mis-attributed to the 41 sources at the time).
- `19387078` — 128K from `ckpt_65536` with a **trimmed 8-source** mix → OOM'd AGAIN at the
  first 128K step, proving sources were NOT the cause.
- `19387894` — added **`--no-create-attention-mask-in-dataloader`** (the real fix: an explicit
  128K×128K host mask ≈ 16 GB/sample → host OOM; only bites at 128K) + `--num-workers 0`.
  8-source confirmation run.
- **Definitive 128K** — once the mask fix is confirmed, rerun 128K with **all 37 languages**
  (`extend_real_to128k.sh`, which has the full mix + mask fix; skips 16K/32K/64K).

## Key learnings / gotchas
- **Mid-stage eval → host-RAM OOM** with a many-source blend at long seq: the eval dataloader
  instantiates all sub-datasets and spikes CPU RAM. Set `--eval-interval` huge / `--eval-iters 0`
  for the training run; eval (RULER) separately. (Also `--num-workers 1` to trim dataloader RAM.)
- **MIOpen kernel search → NCCL timeout** at long seq: set `MIOPEN_FIND_MODE=FAST` (+ persistent
  `MIOPEN_USER_DB_PATH` on /flash). Without it the run hangs and dies on comms timeout. (Jouni)
- Must `export WORLD_SIZE=$SLURM_NTASKS` (else `world size 1 not divisible by TP×CP`).
- `long-ctx-sample/arxiv.idx` is corrupt → use the catalogue paths (verified).
- `torch_dist` is parallelism-agnostic → load at any TP/PP/CP, no conversion.
- 128K needs CP≥8 (or full recompute) for memory; CP=4 ran at 94%.
- Node count must be a multiple of 8 (so CP=8 @128K gives integer DP).
- dev-g 30-min cap is too short for staged runs / large index builds → use standard-g.
- Saves between independent `--finetune` stages can be weights-only; this run keeps full saves
  (resumable). Run is resumable per stage (skips stages whose checkpoint exists).
