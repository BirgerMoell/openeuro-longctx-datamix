# Sparse Attention (DSA) for OELLM — What We Learned & The Plan Forward

**Scope.** This document is the definitive reference for our DeepSeek-Sparse-Attention (DSA) work:
making fine-grained sparse attention run and train on our **GQA** model on **LUMI's AMD MI250X**
GPUs, so long context (512K → 1M → 2M) is *efficient at inference*, not just trainable. It captures
the motivation, the full engineering that is built and validated, every empirical finding, the
training recipe, the open issues, and the concrete plan forward.

Last updated: 2026-07-26.

---

## 1. Why sparse attention (the motivation)

We already extended the OELLM 9B (Qwen3 dense, GQA) to **256K** with the θ-scaling law and published
it (`oellm-9b-256k-theta64m-prelude`). Going further — 512K, 1M, 2M — the wall is **not training**
(context-parallel + FlashAttention makes long sequences trainable) but **inference cost**:

- **Dense prefill is O(L²).** At 512K a full attention-score matrix is ~1.1 TB (fp32); it OOMs on a
  single GCD past ~64K and is minutes-per-forward at 512K. That is not servable.
- **DSA is O(L·k).** A lightweight "lightning indexer" scores past tokens and selects the top-k
  (k≈2048); attention runs only over those. Prefill becomes ~250× cheaper at 512K, and — critically
  — the KV cache can be **offloaded** (keep bulk K/V in host memory, fetch only the ~2048 selected
  per query), turning GPU KV memory from O(L) to O(k). *That* is what makes 1M–2M servable on modest
  hardware.

**Key validation (this is the headline empirical result):** on our real 256K model the attention is
**highly concentrated** — the top-2048 keys hold **86–97% of all attention mass** (measured at layers
0/18/35). So a working sparse top-2048 loses almost nothing. **DSA is an excellent architectural fit
for this model.**

---

## 2. Starting point & the core obstacle

- Our Megatron (Jouni's `NVIDIA-Megatron-LM`) **already implements DSA** in
  `megatron/core/transformer/experimental_attention_variant/dsa.py` (822 lines: lightning indexer,
  top-k, KL loss, sparse-attention reference, loss logging, autoscaler). **But it is gated to MLA**
  (`assert config.multi_latent_attention`). Our model is **GQA**.
- The compiled fused DSA kernels found on disk
  (`.../bmoell/pylibs-overlay-euroeval-polluted/cudnn/deepseek_sparse_attention`,
  `indexer_forward/backward/top_k`, `sparse_attention_backward`, `score_recompute`) are **NVIDIA-only**
  (CUTLASS / `cute_dsl` / `sm90`/`sm100`, `import cuda.bindings.driver`). **Unusable on AMD/ROCm.**

So the work split into: (a) adapt Megatron's DSA to GQA; (b) write **ROCm-native** kernels to make it
fast/feasible; (c) find a **stable training recipe**.

---

## 3. What is built & validated (all committed under `scripts/dsa/`)

### 3.1 GQA adapter — `megatron_gqa_dsa.py`
- `GQADSAttention` — drop-in `core_attention`. Reconstructs the indexer's `x`/`qr` inputs, and
  **repeat-interleaves GQA KV heads** (`ng→np`) since the DSA core assumes equal q/kv head counts.
- `GQADSASelfAttention` — threads the real `hidden_states` to the indexer (needed at **TP>1**, where
  the query is head-sharded and can't be reshaped to `hidden_size`).
- `get_gqa_dsa_attention_spec` / `get_gqa_dsa_layer_spec` / `get_gqa_dsa_block_spec` — module/layer/
  block specs; the block spec assigns **per-layer dense vs DSA** from a pattern string (hybrid).

**Blockers cleared to run at all on ROCm (7 dev-g iterations):** config-bridge the MLA/rope fields the
indexer reads (`q_lora_rank=None`, `qk_pos_emb_head_dim`, `rope_type`, `rotary_percent`,
`rotary_base`); backend must be **`TESpecProvider`** (has `.linear`), not `LocalSpecProvider`;
`fast_hadamard_transform` is missing on ROCm → monkey-patch `dsa.rotate_activation` with a pure-torch
normalized **FWHT** (index_head_dim=64/128 are powers of two); GQA KV-head expansion.

**Validation ladder (all PASSED on MI250X):**
1. Attention module fwd+bwd (`smoke_dsa_gpu.py`).
2. Full GPTModel training step, TP=1 (`smoke_dsa_model.py`, loss 8.4→5.4).
3. TP>1 (`GQADSASelfAttention` threads hidden_states; TP=2 both ranks; needs
   `model_parallel_cuda_manual_seed`).
4. Hybrid per-layer dense/DSA (`smoke_dsa_hybrid.py`, pattern `FSFS`).
5. Full sparse pipeline end-to-end (`smoke_dsa_sparse.py`).

### 3.2 ROCm Triton sparse-attention kernel — `triton_dsa.py`
Fused, genuinely-sparse (O(L·k)) flash-style attention over the top-k keys; **no L² score matrix, no
gather-copies**. Forward + backward + `triton_dsa_attn` autograd Function.
- **Correct** vs the reference (max_err 2e-3 fwd; dq/dk/dv rel ~4e-3 bwd).
- **Runs at 64K/128K/256K where dense OOMs** (dense fp32 scores would be 69/275/1100 GB); scales
  linearly.
- Two real bugs fixed en route: (a) backward launch must pass the **2D slice** strides, not the 4D
  tensor strides (GPU memory-access fault); (b) compute the softmax-grad **`delta` in-kernel** from
  fp32-recomputed `p` (computing it in torch from the bf16 saved output caused 20–30% dq/dk error via
  cancellation).
- **Crossover ~64K:** below it, dense is faster (coalesced matmul beats scattered gather); above it,
  sparse is the *only* option. So: **dense for short-context warm-up, Triton-sparse for the 512K→2M
  extension/inference.**

### 3.3 Chunked / query-blocked indexer — `chunked_indexer.py`
The indexer's own scoring materializes `[b, sq, sk]`, which **OOMs at 512K** (69 GB @ CP=16). Fix:
key-blocked **running top-k** + query-blocking → **exact** top-k (match=1.0000 vs full), peak memory
O(q_block·block), independent of sq/sk (4.9 GB @128K vs 69 GB). Scales to 512K+.

### 3.4 Integration — `dsa_patches.py`, `gpt_builders_dsa.py`
`apply_sparse_dsa_patches()` monkey-patches the three O(L²) pieces for a sparse *run*:
1. `dsa.unfused_dsa_fn` → `triton_dsa_attn` (Triton O(L·k) attention)
2. `DSAIndexer.forward_with_scores` → chunked/query-blocked top-k
3. `dsa.compute_dsa_indexer_loss` → no-op (KL is O(L²), warm-up-only)

A **shadow `gpt_builders.py`** (first in `PYTHONPATH`, so Megatron's `pretrain_gpt` imports ours)
applies the FWHT patch + DSA config bridge + hybrid block spec + optional indexer-only freezing, and
loads the base checkpoint **excluding the indexer keys** (plain `dist_checkpointing.load` over base
keys — bypasses Megatron's `FullyParallelLoadStrategyWrapper` planner, whose cross-rank
`gather_object` **deadlocks** on the new indexer keys). Env knobs: `DSA_PATTERN`, `DSA_TOPK`,
`DSA_N_HEADS`, `DSA_HEAD_DIM`, `DSA_LOSS_COEFF`, `DSA_SPARSE_RUN`, `DSA_LOAD_BASE`, `DSA_RECALL_LOG`.

### 3.5 Hybrid layer selection — `dsa_layer_search.py`
Per GLM-5 (arXiv:2602.15763), **not every layer should be sparse** (they used ~1:1 full:sparse, an
arrangement search-discovered at 16K that length-generalizes). Our search (one cheap 2K forward on
`prelude_256k_hf`, ranking layers by top-12% attention concentration) yielded:
`FFFFSSFFFFFFFFSSFFFSSSSSSSSSSSSSSFFF` (18/36 sparse) — **early layers (esp L0, most diffuse) and the
last layers stay dense; the concentrated deep-middle (L19–32) goes sparse.** Model-specific and
interpretable.

---

## 4. The training recipe (what makes it stable) — the central lessons

DSA continued-training is **two stages**, confirmed against DeepSeek-V3.2 and MoBA/InfLLM-V2:

1. **Dense warm-up.** Keep attention **dense (full)** while the **lightning indexer** learns to
   predict attention via **KL loss**. This gives the indexer a stable target and never perturbs the
   model. (MoBA: "indexer warm-up runs full attention before switching to sparse … the same recipe
   converts a pretrained dense checkpoint into a sparse one.") DeepSeek: 1000 steps × (16 × 128K) =
   **2.1B tokens**, model frozen except the indexer.
2. **Sparse adaptation.** Introduce top-k (k=2048), train model+indexer. DeepSeek used 943.7B tokens;
   **GLM-5 ~20B; InfLLM-V2 shows it needs little data.** Graduated k-ramp (full→4096→2048) avoids a
   hard switch.

### Empirical findings that cost us runs (in order discovered)
1. **Sparse-from-step-1 is unstable.** Our first attempts ran top-k=2048 with a *random* indexer →
   the model attends to garbage → noisy loss 1.4–2.5, grad-norm ~5000. **Fix: dense warm-up**
   (`DSA_TOPK = seq_length` → the sparse kernel *is* dense → model unperturbed → loss stable ~1.3).
2. **Indexer LR must be high.** DeepSeek/GLM train the indexer at **~5e-3** (vs model ~1e-5). During
   dense warm-up the model is unperturbed, so a high global LR mostly drives the indexer. At LR 1e-4
   recall crept 0.37→0.40 over 46 iters; at 5e-3 it reached the plateau by iter 18 (≈2.5× faster).
3. **The 0.40 plateau was indexer *capacity*, not LR/data/metric.** A ceiling diagnostic showed the
   metric's achievable ceiling is ~0.78 (heads agree; overlap 0.76–0.85) and attention is 86–97%
   concentrated — so 0.40 meant the indexer was under-powered. **Fix: `DSA_N_HEADS` 4 → 16** → recall
   jumps to **0.60 out of the gate, spiking 0.83.**
4. **16-head indexer OOMs at 16K warm-up** (score intermediate = heads×seq², ~17 GB) → **warm up at
   8K** (indexer length-generalizes; `DSA_TOPK=8192`).
5. **Distributed pitfall:** rank-0-only diagnostic work inside the forward **desyncs the next
   collective and hangs** → compute recall on all ranks, print on rank 0.
6. **The recall probe itself must not be O(L²)** — materializing full attention for the metric OOM'd
   (64 GB @16K) → sample ~64 query rows.
7. **Training speed** is the practical limiter: the *unoptimized reference* DSA is ~6 min/iter (multi
   O(L²) passes × microbatches), so 150 warm-up iters need a 24h wall. Irrelevant to *inference*
   (Triton handles that); a DSA-forward optimization is the lever if faster *training* is wanted.

### Current recipe (validated, in use)
```
Base:           continue from output_256k_v2/ckpt_262144 (θ=64M, 256K), load base keys only
Warm-up:        dense (DSA_TOPK = seq); 16-head indexer (DSA_N_HEADS=16, head_dim=128);
                LR 3e-3 cosine→3e-4; seq 8192 (length-generalizes); KL on (coeff 0.1);
                hybrid pattern FFFFSSFFFFFFFFSSFFFSSSSSSSSSSSSSSFFF; recall logging on.
                Gate: top-2048 recall → ~0.78 ceiling.
Sparse adapt:   DSA_SPARSE_RUN=1 (Triton attn + chunked indexer + KL off); k full→4096→2048;
                train model to adapt. Gate: NIAH holds vs dense.
```

---

## 5. Current status (2026-07-26)

- **Warm-up running** (`dsa-fin`, 16-head, LR 3e-3, 8K, 24h wall) continuing from the 256K checkpoint.
  Recall at ~0.58–0.63 early (iter 16), climbing toward the ~0.78 ceiling; earlier 16-head run hit
  0.83 before a 6h timeout. Recipe is converging; this run has the wall-time to finish 150 iters.
- **Published models unaffected/independent:** `oellm-9b-128k-theta32m-{v3,prelude}` and
  `oellm-9b-256k-theta64m-prelude` are live on HF (both `birgermoell` and `openeurollm`).
- **512K data ready:** `blend_512k.txt` = 15% synthetic-recall + 50% genuine-512K superlong + 35%
  Jouni multilingual (168 sources, stable `/scratch`). 1M/2M data pre-tokenized (`superlong_data`,
  `1024k`/`2048k`).
- **Context note:** LUMI had a multi-week maintenance outage (mid-July) that paused runs; recovered
  2026-07-25.

---

## 6. Plan forward

### Phase A — finish the warm-up (in progress)
Let `dsa-fin` complete 150 iters; confirm rolling-avg recall settles ~0.72–0.78. If it stalls below
that: (i) more warm-up tokens (we're at ~20M vs DeepSeek's 2.1B — likely just needs longer), (ii)
larger indexer head_dim, (iii) LR tune. Save the warmed indexer checkpoint.

### Phase B — sparse adaptation
Flip `DSA_SPARSE_RUN=1` (Triton attn + chunked indexer + KL off), graduated k (full→4096→2048), train
the model to adapt to true sparsity at 8K–32K. Small budget (GLM/InfLLM-V2 → single-digit B tokens).
Gate: forced-choice NIAH at 128K holds vs the dense 256K model.

### Phase C — long-context sparse inference demo (the deliverable)
Demonstrate a **runnable** model: prefill timing + NIAH at **512K** (and 1M) with the Triton kernel —
showing it executes where dense OOMs, at O(L·k), with the accuracy the 86–97% mass-concentration
predicts. Then extend context with the θ-law (θ: 512K=128M, 1M=256M) *under sparsity*.

### Phase D — production hardening (as needed)
- **Optimize the Triton kernel** (autotune blocks, fold the b×np head-loop into the launch grid) —
  first version is ~5.5s/call@256K, unoptimized.
- **KV offloading** at inference (host-resident K/V, fetch top-k) → O(k) GPU KV for 1M–2M on one GPU.
- **DSA-forward training speedup** (fuse the indexer/KL passes) if faster *training* is wanted.
- **Sparse-under-CP** if we ever need to *train* at ≥512K (indexer needs cross-CP key gather; not
  required for the train-short / infer-long plan, since sparsity length-generalizes).

### Alternatives considered (documented, not chosen)
- **Dense 512K training** — feasible (CP+flash) but produces an inference-expensive model; rejected
  because the goal is a *servable* long-context model.
- **NVIDIA fused kernels / move to Leonardo** — the CUTLASS kernels work on NVIDIA but the models/data
  pipeline live on LUMI; a platform move is heavy. Triton-on-ROCm keeps everything on LUMI.

---

## 7. File index (`scripts/dsa/` + `docs/`)
- `megatron_gqa_dsa.py` — GQA adapter (attention/self-attention wrappers + specs).
- `triton_dsa.py` — ROCm Triton sparse attention (fwd+bwd+autograd).
- `chunked_indexer.py` — streaming/query-blocked top-k (512K-safe).
- `dsa_patches.py` — the three sparse monkey-patches + recall logging.
- `gpt_builders_dsa.py` — shadow builder (config bridge, hybrid spec, exclude-indexer base-load).
- `dsa_layer_search.py` — per-layer concentration search → hybrid pattern.
- `smoke_dsa_*.py`, `test_*.py` — the validation ladder.
- `docs/dsa_gqa_integration.md` — integration details & milestone log.
- `docs/dsa_training_recipe.md` — the recipe + literature cross-checks.
- `docs/dsa_sparse_attention_plan.md` — earlier research plan (GLM-5 summary).
- `docs/sparse_attention_dsa.md` — **this document.**

## 8. Key references
- DeepSeek-V3.2-Exp (DSA): dense warm-up (2.1B tok, indexer-only) → sparse (943.7B), k=2048.
- GLM-5 (arXiv:2602.15763): 1:1 full:sparse hybrid, search-discovered at 16K, length-generalizes.
- Kimi Linear (arXiv:2510.26692), MoBA (MoonshotAI), InfLLM-V2 (arXiv:2509.24663): dense→sparse
  warm-up consensus; sparse adaptation needs little data; hybrid ratios matter.
- LongRoPE2 (arXiv:2502.20082): informed the θ-law (depth-0 = RoPE high-dim OOD, not data).
