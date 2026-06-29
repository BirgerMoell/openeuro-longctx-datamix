# Sparse-attention long-context plan (DSA), adapted from GLM-5 (arXiv:2602.15763)

**Status:** independent research track, **additional** to the current dense-ABF / θ-law work and
the prelude extension. Promising next step *if* it works. Date: 2026-06-29.

This plan (1) summarizes exactly what GLM-5 did, then (2) translates it to **our model** (OELLM
Qwen3 **9B dense**, GQA 32 heads / 8 KV, 128K@θ=32M already working) — which is the real research
contribution, since DSA was designed for a 744B MoE + MLA, not a small dense GQA model.

---

## PART A — What GLM-5 did (faithful summary, with their numbers)

### A1. Staged context extension (mid-training)
Three stages, dense attention, after the 27T-token base pretrain:
| stage | seq length | tokens |
|---|---|---|
| 1 | 32K | **1T** |
| 2 | 128K | **500B** |
| 3 | 200K | **50B** |
(SFT max context later set to **202,752** tokens.)

### A2. DeepSeek Sparse Attention (DSA)
Replaces dense O(L²) attention with **content-based top-k selection** via a **lightning indexer**:
- An auxiliary **indexer** scores every past token for each query and retrieves the **top-k = 2048**
  most relevant key/value entries; main attention is computed **only over those 2048**.
- **Deterministic top-k** (plain `torch.topk`) — "slightly slower but deterministic … more
  consistent outputs and substantial RL gains" (vs a faster non-deterministic kernel).
- Net: **~1.5–2× less attention compute** at long sequence length.

**Indexer math (DSA / DeepSeek-V3.2 formulation):** for query token *t* and past token *s*,
```
I(t,s) = Σ_{j=1}^{H_I}  w_{t,j} · ReLU( q^I_{t,j} · k^I_s )
```
small number of indexer heads `H_I` (e.g. 1–4), low-dim indexer q/k (can be FP8), ReLU for speed.
Select `S_t = TopK_s( I(t,s), k )`; main attention: `softmax over s∈S_t of (q_t·k_s)/√d · v_s`.

**Two-stage "dense warm-up → sparse adaptation" training:**
- **Warm-up:** **1000 steps**, each **14 sequences × 202,752 tokens**, **max LR 5e-3** — model runs
  *dense*, and the indexer is trained (KL-style) to **mimic the true attention distribution**:
  `L_index = Σ_t KL( p_attn(t,·) ‖ Softmax(I(t,·)) )`, with the main model's grads detached from it.
- **Sparse adaptation:** **20B tokens** at the mid-training data/hyperparameters, now using top-k
  sparse attention end-to-end (indexer keeps training under the KL loss over selected tokens).

### A3. Architecture / other (context)
744B total / **40B active** MoE, **256 experts**, **80 layers**, **MLA** (576-dim latent KV;
"MLA-256" = head dim 192→256, heads −⅓), **MTP** (3 shared MTP layers). Reasoning RL: group 32,
batch 32, β=2, ε_low 0.2, ε_high 0.28.

### A4. Long-context data
Natural (**books, academic papers**) + **synthetic via "interleaved packing"** (citing **NextLong**
and **EntropyLong**) — i.e. construct long sequences that *contain genuine long-range dependencies*
(interleave related/cited passages so the answer requires distant context), not just concatenation.

---

## PART B — Adapted plan for our 9B dense model (the experiments)

**Prerequisite:** a working dense long-context base = our **θ=32M 128K model**
(`oellm-9b-128k-theta32m-v3`, or the prelude-128K once done). DSA is applied *on top* of a model
that already has good dense long-context, then extended further cheaply under sparsity.

### B0. Implementation (the gating engineering)
- Add a **lightning indexer** module to our Qwen3 attention (Megatron). `H_I = 2` indexer heads,
  indexer dim **64**, ReLU scoring, bf16 (skip FP8 on ROCm initially).
- Top-k selection with **deterministic `torch.topk`** (per GLM-5's RL-stability finding).
- GQA note: our 8 KV heads are shared across 32 q heads — the indexer selects KV *positions*
  (shared across the group), so selection is per-KV-head, k positions per query. This is the main
  adaptation vs MLA and must be implemented/validated.
- Reference impls to port: DeepSeek-V3.2 DSA kernels; fall back to a pure-PyTorch top-k + gather
  for correctness first, optimize later.

### B1. Stage 1 — dense warm-up (train the indexer)
Run the model **dense** at **128K** (our working regime), training **only the indexer** to predict
the attention distribution:
- Loss: `L = L_LM + λ · L_index`, `L_index = mean_t KL( stopgrad(p_attn(t,·)) ‖ softmax(I(t,·)) )`,
  λ ≈ 0.1; **indexer grads do not flow into the main model** (stop-grad), so LM quality is preserved.
- Scaled budget (vs GLM-5's 1000×14×202752 ≈ 2.8B tokens): **~1–2B tokens at 128K**, LR for the
  indexer **1e-3 → 1e-4** cosine (main model frozen or tiny LR 1e-6), gbs 64, CP=8, 16 nodes.
- Pass: indexer top-2048 recovers ≥ ~90% of dense attention mass (measure recall of top tokens).

### B2. Stage 2 — sparse adaptation
Switch to **top-k sparse attention** (k below), continue training end-to-end:
- **k schedule by length:** 128K → **k=1024**; 256K → **k=1536**; 512K–1M → **k=2048** (GLM-5 used
  2048 at 200K — for a 9B model with shorter target we can start smaller).
- Budget: **~10–20B tokens** (GLM-5 used 20B), Jouni length-biased + interleaved-packing synthetic.
- LR 1e-5 → 1e-6 cosine (WSD acceptable), gbs 64, bf16, clip 1.0, wd 0.1 — same family as our dense
  runs. Keep indexer KL loss on (now over the selected set).
- Pass: 128K NIAH depth-0 stays ≥ our dense θ=32M result (≈100%) at **1.5–2× lower attention cost**.

### B3. Stage 3 — extend further under sparsity (the payoff)
Because attention is now **O(L·k)** not O(L²), extend cheaply:
| stage | seq | θ (our law) | k | tokens |
|---|---|---|---|---|
| 256K | 262144 | 32–64M | 1536 | ~1–2B |
| 512K | 524288 | 64M | 2048 | ~1B |
| 1M | 1048576 | 128M | 2048 | ~0.5–1B |
θ from our **doubling-per-octave law**; CP still shards the *sequence* for activation memory, but
attention cost no longer explodes. This is where DSA beats dense-ABF+CP (which hit CP=128 / a2a
limits at 256K+).

### B4. Data (interleaved packing)
Implement **NextLong / EntropyLong-style interleaved packing**: build long sequences by
interleaving a target document with *related* passages (retrieved by similarity or citation) so the
needle/answer genuinely depends on distant context — plus our `synthetic_recall` superlong set and
Jouni length-biased mix. This is what makes sparse models actually *use* the long range.

### B5. Hyperparameters summary (our scale)
- Indexer: H_I=2, dim 64, ReLU; warm-up LR 1e-3→1e-4; KL weight λ=0.1; stop-grad to main model.
- Sparse: top-k via deterministic `torch.topk`; k = 1024(128K)→2048(≥512K).
- Main training: lr 1e-5/min 1e-6 cosine, gbs 64, mbs 1, adam(0.9,0.95), wd 0.1, clip 1.0, bf16,
  TP=8, CP=seq/16K, save-interval 100, distributed-timeout 20 + retry (our anti-hang pattern).

### B6. Compute estimate (our throughput)
Warm-up (~1–2B @128K) ≈ 1–2k GPU-h; sparse adapt (~15B, mixed lengths) ≈ 4–6k GPU-h; extensions
(256K/512K/1M, ~3B total, *sparse*) ≈ 3–5k GPU-h. **Total ≈ 10–13k GPU-h (~1% of allocation).**
Compute is not the constraint; the **engineering (indexer + sparse kernels on ROCm) is**.

### B7. Eval
- NIAH depth-stratified at 128K/256K/512K/1M (need the CP-aware/served eval from the superlong plan).
- **Indexer quality probe:** top-k recall of true attention mass.
- RULER multi-task once a serving path exists.
- Headline target: match dense-θ=32M depth-0 at 128K, then *extend* to 256K–1M where dense can't.

### B8. Risks / decision gates
1. **DSA-on-GQA is unproven** (DSA was built for MLA) — Stage B0/B1 is the make-or-break: does the
   indexer learn to predict GQA attention? Gate before spending on B2+.
2. **ROCm kernels:** no off-the-shelf DSA kernel for MI250X — start with pure-PyTorch top-k (correct
   but slow) to validate quality, then optimize. The 1.5–2× speedup only materializes with good kernels.
3. **Sequencing:** do B0–B2 at 128K (measurable) first; only extend (B3) after sparse 128K matches dense.
4. Independent of the dense track — if dense-ABF stalls at 256K (CP/a2a limits), **DSA is the path to
   1M**; if DSA stalls on kernels, dense-ABF still gives us ≤256K.

## Bottom line
GLM-5 = **staged dense extension (32K→128K→200K) + DSA (lightning indexer, top-k=2048, dense-warmup→
sparse-adapt 20B tok) + interleaved-packing data.** For us: **(a)** implement the lightning indexer on
Qwen3-GQA, **(b)** dense-warm-up the indexer at 128K, **(c)** sparse-adapt, **(d)** extend to 256K–1M
under O(L·k) attention using our θ-law. It's the credible route past the dense-attention wall to 1M+,
and ~1% of our compute — the cost is kernel/indexer engineering, not GPU-hours.
