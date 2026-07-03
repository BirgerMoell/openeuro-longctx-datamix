"""
GQA adapter for Megatron's existing DSA (DeepSeek Sparse Attention).

Megatron already implements DSA (megatron/core/transformer/experimental_attention_variant/dsa.py)
but gates it to MLA. This module makes it usable with our GQA Qwen3:
  - `GQADSAttention` is a drop-in `core_attention` (standard q,k,v,mask signature) that
    reconstructs the indexer's `x`/`qr` from the query, so no MLA latent is needed.
  - `get_gqa_dsa_attention_spec` wires it into a normal `SelfAttention` (not MLASelfAttention),
    reusing Megatron's DSAIndexer + DSAttention + KL-loss unchanged.

Config to set (TransformerConfig): dsa_indexer_n_heads, dsa_indexer_head_dim, dsa_indexer_topk,
dsa_indexer_loss_coeff, dsa_indexer_use_sparse_loss; q_lora_rank=None; rope_type='rope', rotary_base=θ.

TP note: reconstructing x from the query is valid at TP=1 (query reshapes to full hidden_size).
For TP>1 the query is head-sharded — production needs the real hidden_states threaded in
(see docs/dsa_gqa_integration.md, route 1). This adapter targets the single-GPU smoke test + TP=1.
"""
import torch
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAttention, DSAttentionSubmodules, DSAIndexer, DSAIndexerSubmodules,
)


class GQADSAttention(DSAttention):
    """Drop-in DSA core_attention for GQA. The indexer needs a full-hidden_size per-token rep (x/qr).
    Production (TP>1): `GQADSASelfAttention` stashes the real hidden_states on `self._dsa_hidden`
    (the indexer SP-gathers it). Fallback (TP=1): reconstruct x from the query (np*hn == hidden)."""

    def forward(self, query, key, value, attention_mask, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        # GQA: expand KV heads to match query heads (like DotProductAttention's repeat_interleave).
        ng = key.shape[2]
        np = query.shape[2]
        if np // ng > 1:
            key = key.repeat_interleave(np // ng, dim=2)
            value = value.repeat_interleave(np // ng, dim=2)
        sq, b, np, hn = query.shape
        x = getattr(self, "_dsa_hidden", None)     # real hidden_states threaded by GQADSASelfAttention
        if x is None:
            x = query.reshape(sq, b, np * hn)      # TP=1 fallback: np*hn == hidden_size
        qr = x
        return super().forward(
            query, key, value, attention_mask, x, qr,
            attn_mask_type=attn_mask_type, attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )


class GQADSASelfAttention(SelfAttention):
    """SelfAttention that threads the real hidden_states to the DSA core (needed for TP>1, where the
    query is head-sharded and can't be reshaped to hidden_size)."""

    def forward(self, hidden_states, *args, **kwargs):
        # Stash the pre-projection hidden_states (full hidden_size, SP-sharded) on the core module;
        # the DSA indexer gathers it across sequence-parallel internally.
        self.core_attention._dsa_hidden = hidden_states
        return super().forward(hidden_states, *args, **kwargs)


def get_gqa_dsa_attention_spec(config, backend):
    """SelfAttention (GQA) with DSA sparse core_attention. Mirrors the MLA DSA spec but with
    a normal SelfAttention + our GQADSAttention wrapper. No `multi_latent_attention` required."""
    core_attention = ModuleSpec(
        module=GQADSAttention,
        submodules=DSAttentionSubmodules(
            indexer=ModuleSpec(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=backend.linear(),
                    linear_wk=backend.linear(),
                    k_norm=backend.layer_norm(rms_norm=False, for_qk=True),
                    linear_weights_proj=backend.linear(),
                ),
            )
        ),
    )
    qkv = (backend.column_parallel_layer_norm_linear()
           if config.qk_layernorm else backend.column_parallel_linear())
    return ModuleSpec(
        module=SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=qkv,
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=backend.layer_norm(for_qk=True) if config.qk_layernorm else None,
            k_layernorm=backend.layer_norm(for_qk=True) if config.qk_layernorm else None,
        ),
    )


def _dsa_core_attention_spec(backend):
    from megatron.core.transformer.spec_utils import ModuleSpec as _MS
    return _MS(
        module=GQADSAttention,
        submodules=DSAttentionSubmodules(
            indexer=_MS(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=backend.linear(),
                    linear_wk=backend.linear(),
                    k_norm=backend.layer_norm(rms_norm=False, for_qk=True),
                    linear_weights_proj=backend.linear(),
                ),
            )
        ),
    )


def get_gqa_dsa_layer_spec(backend, qk_layernorm=True):
    """Full GPT transformer-layer spec with DSA sparse attention: take the standard GQA layer and
    swap ONLY its dense core_attention for our DSA core. Everything else (qkv, MLP, norms, BDA)
    unchanged. Usable via `--spec <module> get_gqa_dsa_layer_spec`."""
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=qk_layernorm)
    spec.submodules.self_attention.module = GQADSASelfAttention  # threads hidden_states (TP>1)
    spec.submodules.self_attention.submodules.core_attention = _dsa_core_attention_spec(backend)
    return spec


def get_gqa_dsa_block_spec(backend, pattern, qk_layernorm=True):
    """HYBRID block: per-layer choice of dense vs DSA attention (GLM-5 uses ~1:1 full:sparse; the
    arrangement is model-specific and search-discoverable at short context — see dsa_layer_search.py).
    `pattern` is a string of len==num_layers: 'S'=sparse(DSA), 'F'=full(dense). Returns a
    TransformerBlockSubmodules with the mixed per-layer specs."""
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    dense = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=qk_layernorm)
    layer_specs = []
    for c in pattern:
        if c in "Ss":
            layer_specs.append(get_gqa_dsa_layer_spec(backend, qk_layernorm=qk_layernorm))
        else:
            layer_specs.append(dense)
    return TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=TENorm)
