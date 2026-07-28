# Long-Context Extension — Handoff & Experiment Guide (for Jouni)

**Purpose.** Everything needed to run and reason about long-context extension experiments the way we
did: how we think about **RoPE θ (theta)**, the exact recipe + scripts, the eval, results, the open
questions (annealing/LR), and where everything lives. You provided the data mix and the base scripts;
this documents what I built on top and what I learned so you can reproduce or extend it.

---

## 0. TL;DR
- We extend context by **native ABF**: raise RoPE's base frequency θ and continue-pretraining at the
  longer length, staged `4K→16K→32K→64K→128K→256K`.
- **The one non-obvious finding: far-position ("depth-0") retrieval failure is a θ problem, not a data
  problem.** The critical θ **doubles per context octave** (128K→32M, 256K→64M). Get θ wrong and the
  model can't retrieve from the *start* of the window no matter how much long data you add.
- Published: 128K (0.6T & 1T base) and 256K (1T base) models on HF. Scripts on GitHub.
- Open question you and David raised: **LR annealing / annealed base** — we use cosine decay but
  haven't swept it; likely relevant to the 1T-base "lost-in-the-middle."

---

## 1. How to think about θ (theta) — the core mental model

RoPE rotates each query/key dimension pair at a frequency set by the base θ (default ~10000):
```
frequency(dim i) ∝ θ^(−2i/d)
```
- **Low-index dims** rotate fast (short wavelength) → encode *local* position.
- **High-index dims** rotate slow (long wavelength) → encode *global/long-range* position.

When you train at 4K, the high-index (slow) dims only ever see rotations up to a small angle. At
128K those same dims must represent positions 32× further apart — **out of the distribution they were
trained on**. That OOD is *exactly* why a naively-extended model fails to retrieve from far positions
("depth-0"): the long-range dims produce garbage there.

**Raising θ lowers all frequencies → lengthens wavelengths → the far positions come back
in-distribution** once you train at the new θ. So θ is the knob that decides *how far* the model can
place tokens apart.

### The θ-law we found (the practical takeaway)
The **critical θ that fixes depth-0 doubles per context-length octave**:

| context | working θ | note |
|---|---|---|
| 64K | 8M | |
| 128K | 16M (90% depth-0) → **32M (100%)** | we use 32M |
| 256K | **64M** | confirmed depth-0 ≈ 100% |
| 512K | ~128M | hypothesis |
| 1M | ~256M | hypothesis — **validate, don't extrapolate blindly** |
| 2M | ~512M | hypothesis |

**How we found it:** a θ sweep at fixed data. At 128K, θ=2M/5M/8M gave depth-0 ≈ 0%; θ=16M → 90%;
θ=32M → 100%. Two *different* length-biased data mixes both failed depth-0 until θ was raised → data
is not the lever, θ is. (Consistent with LongRoPE2, arXiv:2502.20082, which frames it as high-dim RoPE
OOD.) **Caveat:** these are *uniform ABF* θ. LongRoPE2 argues for *per-dimension* scaling; we did not
compare — a worthwhile experiment (see §7).

---

## 2. The extension recipe (what to actually run)

**Native ABF, staged continued-pretraining.** Each stage: load the previous checkpoint, raise
`--seq-length` and `--rotary-base` (θ), train ~1–2B tokens, save. `--finetune` semantics (fresh LR
schedule, no optimizer state carried).

Per-stage Megatron args that matter:
```
--position-embedding-type rope --rotary-base <θ for this stage>
--seq-length <16384..262144> --max-position-embeddings <same or target>
--use-flash-attn
--tensor-model-parallel-size 8 --context-parallel-size <1|2|8|16>   # CP shards the long sequence
--sequence-parallel --recompute-activations --recompute-granularity <selective|full>
--micro-batch-size 1 --global-batch-size 64
--lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --lr-warmup-iters ~5% --lr-decay-iters <iters>
--train-iters <tokens / (seq*gbs)>
```
- **θ per stage** (what we used): 16K→500k, 32K→1M, 64K→2M, 128K→**32M**, 256K→**64M**.
  (Note the jump at 128K — that's the θ-law, not a smooth ramp.)
- **CP (context parallelism)** shards the sequence so long seqs fit: 128K used CP=8, 256K CP=16.
  Sliding-window/`a2a` needs heads÷CP; with our 8 KV heads ÷ TP=8 = 1 KV head/rank, so `a2a` is
  impossible — use `p2p`.
- **Token budget:** ~1–2B tokens/stage. This is a *short* phase (~238–476 iters). **We never swept
  this** — see open questions.

### The exact sbatch files (on GitHub)
`scripts/dsa/lumi/` — the real launch scripts that produced the published models:
- `real_v3_128k.sbatch` → 128K on the 0.6T "baby" base (θ=32M) — **scored a clean 100%**
- `prelude_full.sbatch` → 128K on the 1T "prelude" base (θ=32M)
- `real_256k_v2.sbatch` → 256K on the 1T base (θ=64M)
- `real_128k_2x.sbatch` → the 2×-budget diagnostic

These are based on **your** scripts; the LR annealing (`--lr-decay-style cosine`, warmup ≈ iters/20,
decay to min-lr) is inherited from them. See `scripts/dsa/lumi/README.md`.

---

## 3. Data (your mix)
The extension used **your length-biased multilingual long-context mix** — 152 datasets:
~34% finepdfs (eng/edu/per-lang), ~23% dclm, ~17% nemotron, ~9% starcoder, ~7% hplt3 (38 langs),
arxiv/pes2o. Precomputed blend string: `longctx-extend/jouni_blend.txt`.
- **256K** added genuine 256K-length concatenated docs: `blend_256k.txt` = 60% genuine-256K
  (arxiv/books/code/RFC concatenated) + 40% your multilingual mix.
- **512K/1M/2M** data is pre-tokenized in `superlong_data/` (`512k`/`1024k`/`2048k`) incl.
  synthetic-recall; `blend_512k.txt` is built.

**Key point for you:** for *depth-0* the data composition doesn't matter (θ does). But data quality /
length / annealing likely matters for *overall* quality and the mid-depth profile — untested (§7).

---

## 4. Evaluation (how we measure)
**Base-LM forced-choice NIAH** (`scripts/eval_base_lm_niah.py`) — works on base models (no
instruction-following):
- One query needle at a chosen **depth** (0–100%), many filler needles, forced-choice over 4
  candidates where **distractors are other in-context values** (so it tests retrieval+binding; chance
  25%, adversarial so a bad model can score <25%).
- Sweeps **context × depth × language**; control conditions (`no_context`, `shuffled`, `short_ctx`).
- **This is a single-needle smoke test.** For promotion you want RULER/HELMET/MRCR (multi-needle,
  aggregation, variable-tracking, long-doc QA) — not yet run.

**Depth-0 = needle at the far start = the θ-sensitive metric.** Watch that one.

---

## 5. Results (published)
| model | ctx | θ | overall | depth-0 | note |
|---|---|---|---|---|---|
| oellm-9b-128k-theta32m-**v3** | 128K | 32M | **100%** | 100% | 0.6T base |
| oellm-9b-128k-theta32m-**prelude** | 128K | 32M | 96% | 97% | 1T base; mild lost-in-middle (d0.5≈88%) |
| oellm-9b-256k-theta64m-**prelude** | 256K | 64M | ~90% (15-lang sweep finishing) | 93% | 1T base |

HF: `birgermoell/…` and `openeurollm/…`. **Interesting:** the *more-pretrained* 1T base is slightly
*worse* (lost-in-the-middle) than the 0.6T base — unexplained; connects to the annealing question.

---

## 6. Beyond 256K: DSA sparse attention (why, and status)
512K→2M dense is **inference-prohibitive** (O(L²) prefill, huge KV). We're adding **DeepSeek Sparse
Attention (DSA)**: a lightning indexer picks top-k≈2048 tokens per query; attention runs only over
those. Validated that our model's attention is **86–97% concentrated in top-2048** → great fit.
- Built on ROCm (Triton kernels, GQA adapter, chunked indexer) — see `docs/sparse_attention_dsa.md`.
- Training recipe (DeepSeek-confirmed): **dense warm-up** (freeze base, train indexer via KL) →
  **sparse adapt**. Currently debugging checkpoint robustness on flaky nodes.
- **Honest caveats:** the *indexer* is still O(L²) (only core attention is O(L·k)); KV cache is still
  O(L) (147 GB@1M) unless offloaded. These are separate deliverables.

---

## 7. Open experiments (where you can help — the LR/annealing thread)
These are **untested** and directly relevant to your and David's points:
1. **LR annealing sweep.** We use a gentle cosine decay (1e-5→1e-6, short stage). Test higher peak
   (3e-5) and longer decay. David: "LR annealing is absolutely required" — we do decay, but haven't
   validated it's *enough*.
2. **Annealed vs mid-training base.** Extend an *annealed* checkpoint vs the current base. Birger's
   hypothesis: extension on an annealed model works better. **First: confirm whether prelude
   (iter_0124800) / baby (iter_0076800) were annealed** — you'd know better than me.
3. **Extension length sweep.** 1B vs 4B vs 8B tokens/stage — the "how many steps for the anneal
   phase?" question. Our 2×-budget test was inconclusive (cut off by wall clock).
4. **Anneal-quality long mix.** Weight the extension data toward highest-quality long docs (closer to
   a pretraining-anneal mix), not just length-biased.
5. **θ validation, not extrapolation.** For 512K/1M/2M sweep θ around 128M/256M/512M; compare uniform
   ABF vs LongRoPE2 per-dim scaling; mixed short/long batches to avoid short-context regression.
6. **Stronger eval:** RULER/HELMET/MRCR + short-context regression + perplexity + lost-in-the-middle
   profile before any promotion.

The 1T-base lost-in-the-middle is the clue tying #1–#4 together: a more-pretrained base may need a
stronger/longer anneal to fully repurpose for long context.

---

## 8. Where everything lives
- **GitHub:** `github.com/BirgerMoell/openeuro-longctx-datamix`
  - extension scripts: `scripts/dsa/lumi/` · eval: `scripts/eval_base_lm_niah.py`
  - docs: `docs/sparse_attention_dsa.md`, `docs/dsa_training_recipe.md`,
    `docs/depth0_diagnosis_theta_sweep.md`, this file
- **HF models:** `birgermoell/oellm-9b-{128k,256k}-theta{32m,64m}-…`, mirrored to `openeurollm/…`
- **LUMI:** `/scratch/project_465002530/users/bmoell/longctx-extend/` (runs, blends, checkpoints:
  `output_256k_v2/ckpt_262144` is the 256K base for DSA) · `…/dsa_exp/` (DSA code) ·
  `…/superlong_data/` (512K/1M/2M data)
- **Megatron:** `/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM` (yours)

Questions → Birger. The θ-law + the annealing sweep are the two things most worth your eyes.
