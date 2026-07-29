# RoPE θ: our long-context models vs Qwen3.5 — comparison & recommendation

**Question raised (Risto, Jenia):** Qwen3.5 uses `rope_theta = 10M` for 256K; we use `64M`. Should we
adopt Qwen's value? Would it be a better baseline?

**Short answer:** No — not as a *raw number*. Qwen's 10M is tuned to a **different RoPE geometry**
(partial rotary + large head_dim), so it does not transfer to our full-rotary, head_dim-128 model.
Our own 128K sweep already shows why (θ=8M → depth-0 ≈ 0% on our model). The right move is a **θ-sweep
on our architecture** to find our true minimum (which may be well below 64M), and — for a *future*
base — to consider adopting Qwen's geometry, which is what makes a low θ sufficient.

Last updated: 2026-07-29.

---

## 1. Side-by-side configuration

| Field | **Our model** (oellm-9b) | **Qwen3.5-27B** |
|---|---|---|
| `rope_theta` | **32M @128K, 64M @256K** | **10M** |
| `max_position_embeddings` | 131072 / 262144 | 262144 (256K), native |
| `rope_scaling` (YaRN) | none (pure native ABF) | none in base (YaRN only for >256K) |
| **`partial_rotary_factor`** | **1.0 (full rotary)** | **0.25 (only 25% of dims rotary)** |
| `head_dim` | **128** | **256** |
| **rotary_dim** (`head_dim × partial`) | **128** | **64** |
| position scheme | standard RoPE (text) | **mRoPE** (multimodal, interleaved) |
| heads | 32 Q / 8 KV (GQA) | 24 Q / 4 KV |
| extension method | native ABF (raise θ, continue-train) | ABF + YaRN + (VL) mRoPE, heavy long-ctx training |

**Takeaway:** the two `rope_theta` numbers describe different objects. θ only has meaning *relative to*
`rotary_dim`, `head_dim`, the position scheme, and the training recipe.

---

## 2. Why the raw θ numbers are not comparable

RoPE dimension `i` rotates at `ω_i = θ^(−2i/D_rot)` over `i = 0 … D_rot/2−1`, where `D_rot` is the
**rotary** dimension (not head_dim, if partial). The critical θ for a target length depends on `D_rot`
and `head_dim`, so:

1. **Partial rotary (Qwen 0.25 vs our 1.0).** Qwen rotates only 64 of 256 dims. A different-shaped,
   smaller rotary block distributes frequencies differently → a *different* critical θ. Our full
   128-dim rotary is not the same spectrum.
2. **head_dim / rotary_dim (Qwen 256/64 vs our 128/128).** The frequency exponent uses `D_rot`; the
   two models spread their frequencies over different numbers of dims, so equal θ ≠ equal long-range
   coverage.
3. **mRoPE + VL.** Qwen3.5-27B is multimodal; its position encoding is split across sections
   ([11,11,10], interleaved) — a different mechanism again.
4. **Training recipe.** Qwen pairs θ with YaRN/NTK-by-parts and large long-context training budgets;
   ours is pure native ABF with a short (~1B-token) extension. More training/interpolation lets a
   *lower* raw θ work. θ and training-tokens trade off.

**Rough effective-wavelength check** (slowest-dim wavelength `λ_max ≈ 2π·θ^((D_rot−2)/D_rot)`):
- Ours @θ=64M, D_rot=128 → λ_max ≈ **3.0×10⁸** tokens
- Qwen @θ=10M, D_rot=64 → λ_max ≈ **3.8×10⁷** tokens

Both are ≫ 256K (so both "cover" the length), but even in this crude metric ours is ~8× more
aggressive — i.e. we are *not* obviously scaling to the same place, and our larger number is partly
just our full-rotary / head_dim-128 geometry, partly our short training budget.

---

## 3. What would happen if we simply set θ=10M on our model?

We have direct evidence it would **underperform our depth-0 bar**. Our 128K θ-sweep (same
architecture, same data), scored on base-LM forced-choice NIAH at depth-0:

| θ @128K | depth-0 accuracy |
|---|---|
| 2M / 5M / **8M** | **≈ 0%** |
| 16M | 90% |
| **32M** | **100%** |

θ=10M is between 8M and 16M → marginal *at 128K*, and would be well below threshold *at 256K*. So
copying Qwen's 10M onto our (full-rotary, head_dim-128) model would very likely fail far-position
retrieval — **worse**, not better. Their 10M works for *them* because of partial-rotary + head_dim
256 + YaRN + training, not because 10M is universally sufficient.

---

## 4. But the underlying instinct is right: are we over-scaled?

We proved 64M is **sufficient** at 256K (depth-0 ≈ 100%); we never proved it is **minimal**. It is
entirely possible that 16M or 32M also pass at 256K on our model — we jumped to 64M via the
doubling-law heuristic and only confirmed sufficiency. Over-scaling θ has a real cost (§ short-context
harm, `docs/sparse_attention_dsa.md` / the θ discussion): it stretches the frequency spectrum toward
long range and can blur short/medium-range positional resolution.

So the legitimate version of "use Qwen's number" is: **find our own minimum sufficient θ.**

---

## 5. Recommendation

**Do not copy θ=10M.** Instead:

1. **Run a θ-sweep on our architecture at 256K:** `{10M, 16M, 32M, 64M}`, scored on **both**
   - depth-0 (far-position retrieval — the θ-sensitive metric), and
   - **short-context perplexity** (4K/8K) + a short-context benchmark (catches the over-scaling harm).
   This finds our true 256K floor and tells us whether 64M is unnecessarily large. ~a few hundred GPU-h.
2. **Prefer per-dimension scaling for future runs.** If short-context regression appears, switch from
   uniform ABF to **LongRoPE2 / YaRN-style per-dim scaling**, which preserves low/mid frequencies
   (local resolution) and only stretches the truly-high dims — this is *how* Qwen keeps short context
   healthy at low raw θ.
3. **Architecture lesson for the next base (not retrofittable):** Qwen's **partial rotary (0.25) +
   larger head_dim (256)** is itself a long-context-friendly design that lets a *low* θ reach 256K.
   Worth considering when designing the next OELLM base if long context is a priority. We cannot apply
   it to the current base without re-pretraining.

**Bottom line:** Qwen's 10M is a useful *reference*, not a drop-in value. Matching their *number*
without their *geometry* would hurt us; matching their *approach* (minimum sufficient θ, per-dim
scaling, and ideally their rotary geometry next time) is the right lesson.

---

## Appendix: sources
- Qwen3.5-27B `config.json`: `rope_theta=10000000`, `max_position_embeddings=262144`,
  `partial_rotary_factor=0.25`, `head_dim=256`, mRoPE `[11,11,10]` interleaved, 24Q/4KV, 64 layers.
  (huggingface.co/Qwen/Qwen3.5-27B)
- Our θ-law + sweep: `docs/depth0_diagnosis_theta_sweep.md`.
- Short-context mechanism + mitigations: `docs/sparse_attention_dsa.md`, LongRoPE2 (arXiv:2502.20082).
