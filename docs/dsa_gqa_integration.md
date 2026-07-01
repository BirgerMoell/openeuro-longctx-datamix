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

## Status: ✅ DSA RUNS ON THE GPU (MI250X) for our GQA model
GPU smoke test **PASSED** (`scripts/dsa/smoke_dsa_gpu.py`, dev-g, TP=1): builds the GQA-DSA
`SelfAttention` (core=`GQADSAttention`), forward → `out (128,1,512) bf16`, backward → grad flows.
Adapter: `scripts/dsa/megatron_gqa_dsa.py` (`GQADSAttention` drop-in + `get_gqa_dsa_attention_spec`).

**Blockers cleared (7 dev-g iterations), each a small bridge — no fork of Megatron core:**
1. `position_embedding_type`, `q_lora_rank`, `rotary_base` — not on base `TransformerConfig`;
   `setattr` the MLA/rope fields the indexer reads (`q_lora_rank=None`, `qk_pos_emb_head_dim`,
   `rope_type='rope'`, `rotary_percent`, `rotary_base`).
2. Backend: use **`TESpecProvider`** (has `.linear`), not `LocalSpecProvider`.
3. `fast_hadamard_transform` missing on ROCm → monkey-patch `dsa.rotate_activation` with a
   **pure-torch normalized FWHT** (index_head_dim=64 is a power of 2).
4. **GQA KV expansion**: the DSA core assumes equal q/kv head counts; `GQADSAttention.forward`
   `repeat_interleave`s key/value from `ng`→`np` (as `DotProductAttention` does).

**Prior work found (reuse for the FAST path):**
`/scratch/.../users/bmoell/pylibs-overlay-euroeval-polluted/cudnn/deepseek_sparse_attention/`
(`indexer_forward`, `indexer_backward`, `indexer_top_k`, `DSANamespace`) — compiled DSA kernels
(built Jun 23) + sibling `native_sparse_attention`. Use these to replace the unfused reference
`unfused_dsa_fn` for real long-context speedup.

**Next:**
1. **TP>1 for production:** the smoke reconstructs the indexer's `x` from the query (valid TP=1).
   For TP=8, thread the real `hidden_states` (full `hidden_size`) into `core_attention` — a thin
   custom `SelfAttention` (route 1). Then a tiny 2-layer end-to-end training step.
2. **Correctness gate:** at `dsa_indexer_topk >= seqlen`, DSA output must ≈ dense (add to smoke).
3. **Dense-warmup** on a real 128K checkpoint (`use_sparse_loss=False`, indexer KL), gate top-k
   recall ≥ 0.9; then **sparse adaptation** (`use_sparse_loss=True`) with the fused kernels.
- Prototype + math: `scripts/dsa/`. Research plan: `docs/dsa_sparse_attention_plan.md`.
