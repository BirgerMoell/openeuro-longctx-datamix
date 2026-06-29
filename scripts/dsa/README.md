# DSA (sparse attention) experiment — modular prototype

Self-contained prototype of **DeepSeek Sparse Attention (DSA)** as used in GLM-5
(arXiv:2602.15763), for our OELLM Qwen3 **9B dense / GQA** model. Designed as a **modular
experiment**: the core mechanism lives in pure PyTorch here and is validated *before* any
Megatron integration. Full research plan: `../../docs/dsa_sparse_attention_plan.md`.

## Files
- `lightning_indexer.py` — the module (framework-agnostic):
  - `LightningIndexer` — scores past tokens per query: `I(t,s)=Σ_j w_{t,j}·ReLU(q^I_{t,j}·k^I_s)`
  - `topk_sparse_mask` — deterministic top-k causal keep-mask (GLM-5 uses deterministic `topk`)
  - `attention` — dense or sparse (GQA-aware), `attention_probs` — teacher distribution
  - `indexer_kl_loss` — warm-up loss `KL(stopgrad(p_attn) ‖ softmax(I))` (stop-grad to main model)
  - `topk_recall` — quality metric: fraction of true attention mass captured by top-k
- `test_dsa.py` — correctness + behaviour tests (`python test_dsa.py`, CPU, ~seconds)

## Validated (proof-of-concept results)
```
[OK] sparse(k=T) == dense            (max err 0.00e+00)     # exactness
[OK] causal: no future-token selection
[OK] GQA 32q/8kv -> out normalized                          # our arch
[OK] random indexer recall ~0.59 (k/T baseline)
[OK] indexer learns: top-16/64 recall 0.57 -> 0.98          # the crux: it works
```
The indexer **learns to predict attention** (98% mass captured at k = T/4). This de-risks the
central DSA claim for our GQA model. (The raw KL number in the test is inflated by a deliberately
peaked synthetic teacher; the *recall* is the metric that matters.)

## Two-phase training (matches GLM-5)
1. **Dense warm-up:** run dense, train *only* the indexer via `indexer_kl_loss` (stop-grad to main
   model → LM untouched). Stop when `topk_recall(k)` ≥ ~0.9 at the target k.
2. **Sparse adaptation:** switch to `topk_sparse_mask(k)` + `attention(keep_mask=...)`, train
   end-to-end; keep the indexer KL on the selected set. k: 1024@128K → 2048@≥512K.

## Megatron integration (next step, modular)
Hook point found: `megatron/core/transformer/experimental_attention_variant/` +
`experimental_attention_variant_module_specs.py` — a drop-in slot for a custom `core_attention`.
Plan:
1. Wrap this module as a `core_attention` variant: takes (q,k,v) + hidden_states, returns context.
2. A run flag selects `dense | dsa_warmup | dsa_sparse` so it's a clean A/B vs the dense baseline.
3. Start with this pure-PyTorch path (correct, slower) to validate quality at 128K on a real
   checkpoint; only then write fused ROCm/Triton kernels for the 1.5–2× speedup.
4. CP interaction (context-parallel + top-k over a sharded sequence) is the hardest integration
   piece — prototype single-GPU/short-seq first.

## Why this is the right experiment shape
- **Modular:** core logic is testable in isolation; integration is a thin adapter.
- **Gated:** each phase has a measurable pass criterion (recall, then NIAH depth-0).
- **Additive & independent:** does not touch the dense-ABF / θ-law track; it's the route to 1M+
  once dense attention hits its wall.
