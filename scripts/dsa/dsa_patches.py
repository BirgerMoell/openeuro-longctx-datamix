"""Apply the three monkey-patches that make DSA run SPARSE at long context (512K+) on ROCm:
  1. dsa.unfused_dsa_fn        -> Triton O(L·k) sparse attention (triton_dsa.triton_dsa_attn)
  2. DSAIndexer.forward_with_scores -> key-chunked top-k (no [b,sq,sk] materialization)
  3. dsa.compute_dsa_indexer_loss   -> no-op (KL is O(L^2) and only for warmup; off in the run)

Call `apply_sparse_dsa_patches()` once (e.g. from the shadow gpt_builders when DSA_SPARSE_RUN=1).
The indexer must already be warmed (train it with KL at <=256K where it fits; it length-generalizes).
"""
import os, torch
import megatron.core.transformer.experimental_attention_variant.dsa as _dsa
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer, rotate_activation

from triton_dsa import triton_dsa_attn
from chunked_indexer import chunked_topk

try:
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
except Exception:
    from megatron.core.tensor_parallel import gather_from_sequence_parallel_region


def _chunked_forward_with_scores(self, x, qr, mask=None, packed_seq_params=None):
    """Same projections as the original, but top-k via key-chunked running merge (O(sq*k) mem).
    Returns (None, topk_indices) — full index_scores not materialized (KL is patched off)."""
    assert packed_seq_params is None
    rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(None, None, x, self.config, packed_seq_params)
    if self.config.rope_type == "rope":
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False); mscale = 1.0
    else:
        rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
    if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
        x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
        qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)
    seqlen, bsz, _ = x.size()
    q, _ = self.linear_wq_b(qr)
    q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
    q = self._apply_rope(q, rotary_pos_emb, mscale)
    k, _ = self.linear_wk(x); k = self.k_norm(k)
    k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
    k = self._apply_rope(k, rotary_pos_emb, mscale).reshape(seqlen, bsz, self.index_head_dim)
    q = rotate_activation(q); k = rotate_activation(k)
    weights, _ = self.linear_weights_proj(x)
    weights = weights * (self.index_n_heads ** -0.5) * self.softmax_scale
    topk_k = min(self.index_topk, seqlen)
    block = int(os.environ.get("DSA_INDEX_BLOCK", "32768"))
    topk_indices = chunked_topk(q, weights, k, topk_k, mask=mask, block=block)
    return None, topk_indices


def _noop_indexer_loss(*args, **kwargs):
    return torch.zeros((), device=torch.cuda.current_device(), requires_grad=True)


def apply_sparse_dsa_patches():
    _dsa.unfused_dsa_fn = triton_dsa_attn                       # 1. Triton sparse attention
    DSAIndexer.forward_with_scores = _chunked_forward_with_scores  # 2. chunked top-k
    _dsa.compute_dsa_indexer_loss = _noop_indexer_loss          # 3. KL off (warmup-only)
    print("[dsa_patches] SPARSE DSA enabled: Triton attn + chunked indexer + KL off")
