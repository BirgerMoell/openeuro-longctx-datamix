# DSA on GPU for our GQA model — integration setup (from Megatron's existing DSA)

**Key finding:** DSA is **already implemented** in our Megatron
(`megatron/core/transformer/experimental_attention_variant/dsa.py`, 822 lines: lightning indexer,
KL loss, top-k, sparse attention, loss logging). It is **gated to MLA**
(`assert config.multi_latent_attention, "Currently only MLA supports sparse attention"`).
Our model is **GQA** — so we **adapt the existing impl to GQA** rather than build from scratch.
This is the concrete setup to get it running on the MI250X.

## What exists and is reusable as-is
- `DSAIndexer` — lightning indexer. `_compute_index_scores(q,k)`:
  `einsum('sbhd,tbd->sbht') → relu → *weights → sum(heads) → [b,sq,sk]` (identical to our
  `scripts/dsa` prototype). Applies RoPE (`rope_type='rope'`, `rotary_base=θ`).
- `DSAttention.forward(query, key, value, attention_mask, x, qr, …)` — the sparse core_attention:
  detaches `x,qr` (indexer stop-grad), `indexer.forward_with_scores(x, qr) → topk_indices`,
  `output = unfused_dsa_fn(q,k,v, topk_indices, softmax_scale)`.
- `compute_dsa_indexer_loss(index_scores, topk_indices, query, key, …)` — KL(true-attn ‖ softmax(index)),
  operates on **generic q/k** — GQA-compatible.
- Config args (already in `TransformerConfig`): `dsa_indexer_n_heads`, `dsa_indexer_head_dim`,
  `dsa_indexer_topk`, `dsa_indexer_loss_coeff`, `dsa_indexer_use_sparse_loss`.
- Indexer input dim: `q_lora_rank if not None else hidden_size` — so **with `q_lora_rank=None`
  (GQA) the indexer projects straight from `hidden_size`.** No MLA latent needed.

## The ONE coupling to break for GQA
`DSAttention.forward` needs `x` (hidden_states) and `qr` (query rep) for the indexer.
MLA's `MLASelfAttention` threads these to `core_attention`; the **standard GQA `SelfAttention`
passes only (q,k,v,mask)**. So we need a thin **GQA SelfAttention that also passes
`hidden_states` as both `x` and `qr`** (valid because `q_lora_rank=None` → indexer's `linear_wq_b`
expects `hidden_size` input = the hidden states).

## Setup steps (to run on GPU)
1. **`GQADSASelfAttention(SelfAttention)`** — override the point where `core_attention(...)` is
   called to also pass `x=hidden_states`, `qr=hidden_states` (both = pre-projection hidden states,
   which the module already has in scope). Everything else (q/k/v projection, RoPE, output proj)
   unchanged from GQA SelfAttention.
2. **Module spec** — a non-MLA variant of `experimental_attention_variant_module_specs.py`:
   `SelfAttention`→`GQADSASelfAttention`, `core_attention=DSAttention`, add the `indexer` submodule
   (`DSAIndexerSubmodules` with `linear_wq_b`, `linear_wk`, weights). **Drop the MLA assertion.**
3. **Config** (add to the training args / config):
   ```
   dsa_indexer_n_heads = 2         # small, cheap
   dsa_indexer_head_dim = 64
   dsa_indexer_topk = 1024         # 128K: attend to top-1024; scale to 2048 at ≥512K
   dsa_indexer_loss_coeff = 0.1    # KL weight (indexer only; stop-grad to main model)
   dsa_indexer_use_sparse_loss = False   # dense warm-up first, then True for sparse phase
   q_lora_rank = None              # GQA: indexer projects from hidden_size
   rope_type = 'rope'; rotary_base = <stage θ, e.g. 32M @128K>
   ```
4. **GPU smoke test** (`dev-g`, 1 node): build a *tiny* GQA transformer (2 layers, hidden 512,
   32q/8kv) with the GQA-DSA spec; run fwd+bwd on the MI250X; assert (a) it runs on ROCm,
   (b) `dsa k=full` output ≈ dense output, (c) the indexer KL loss is finite and decreasing.
5. **Kernel note:** `unfused_dsa_fn` is the reference (correct, not fused). For real speedup at
   long context we later need a fused sparse-attn kernel on ROCm; validate correctness first with
   the unfused path.

## Two-phase training (matches GLM-5, using existing hooks)
- **Dense warm-up:** `dsa_indexer_use_sparse_loss=False`, train the indexer to predict attention
  (KL via `compute_dsa_indexer_loss`, stop-grad already built in). Gate: top-k recall ≥ ~0.9.
- **Sparse adaptation:** `use_sparse_loss=True`, train end-to-end with top-k. k=1024→2048 by length.

## Status / next action
- **Setup mapped & documented (this doc).** The single new piece of code is
  `GQADSASelfAttention` (thread hidden_states→x,qr) + the non-MLA spec; the rest is Megatron's
  existing DSA + config flags.
- **Next:** implement `GQADSASelfAttention` + spec (modular, in a patch dir), run the dev-g GPU
  smoke test, iterate on ROCm issues, then a 128K dense-warmup on the real checkpoint.
- Prototype + math reference: `scripts/dsa/`. Research plan: `docs/dsa_sparse_attention_plan.md`.
