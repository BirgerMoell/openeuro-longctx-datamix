"""
Lightning Indexer + top-k sparse attention (DSA), modular & framework-agnostic.

Adapted from DeepSeek Sparse Attention (DSA) as used in GLM-5 (arXiv:2602.15763).
This is a STANDALONE, testable module (pure PyTorch) — develop/validate here, then wire
into Megatron via the `experimental_attention_variant` hook (see README).

Math (indexer score for query t, past token s):
    I(t,s) = sum_{j=1..H_I}  w_{t,j} * ReLU( q^I_{t,j} . k^I_s )
Select S_t = TopK_s(I(t,s), k) over s<=t (causal); main attention runs only over S_t.

Two training phases:
  - dense warm-up: model runs DENSE; indexer trained to mimic the true attention
    distribution via KL( stopgrad(p_attn) || softmax(I) ). Indexer grads do NOT touch
    the main model (stop-grad), so LM quality is untouched.
  - sparse adaptation: switch to top-k; train end-to-end (indexer KL kept on selected set).

Shapes use [B, T, H, D] (batch, seq, heads, head_dim). GQA: q has Hq heads, k/v have Hkv.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e30


class LightningIndexer(nn.Module):
    """Lightweight scorer that ranks past tokens for each query. Cheap: H_I small heads,
    low index_dim, ReLU. Produces index logits I [B, T, T] (query x key)."""

    def __init__(self, hidden_size: int, n_index_heads: int = 2, index_dim: int = 64):
        super().__init__()
        self.n_index_heads = n_index_heads
        self.index_dim = index_dim
        self.wq = nn.Linear(hidden_size, n_index_heads * index_dim, bias=False)
        self.wk = nn.Linear(hidden_size, index_dim, bias=False)          # shared key proj
        self.wht = nn.Linear(hidden_size, n_index_heads, bias=False)     # per-token head weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, H]
        B, T, _ = hidden_states.shape
        qi = self.wq(hidden_states).view(B, T, self.n_index_heads, self.index_dim)
        ki = self.wk(hidden_states)                                      # [B, T, index_dim]
        # per-(head) dot products: [B, T(query), H_I, T(key)]
        scores = torch.einsum("bthd,bsd->bths", qi, ki)
        scores = F.relu(scores)
        wht = self.wht(hidden_states)                                    # [B, T, H_I]
        index_logits = torch.einsum("bths,bth->bts", scores, wht)        # [B, T(q), T(k)]
        return index_logits


def _causal_mask(T: int, device) -> torch.Tensor:
    # True where key s <= query t (allowed)
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def topk_sparse_mask(index_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Build a boolean keep-mask [B, T, T]: for each query t, keep its top-k causal keys.
    Uses deterministic torch.topk (per GLM-5: deterministic top-k => stable RL)."""
    B, T, _ = index_logits.shape
    causal = _causal_mask(T, index_logits.device)
    masked = index_logits.masked_fill(~causal, NEG_INF)
    kk = min(k, T)
    idx = masked.topk(kk, dim=-1).indices                               # [B, T, kk]
    keep = torch.zeros_like(masked, dtype=torch.bool)
    keep.scatter_(-1, idx, True)
    keep &= causal                                                      # never select future
    return keep


def _expand_kv_for_gqa(k: torch.Tensor, v: torch.Tensor, n_q_heads: int):
    """GQA: repeat each KV head to cover its query group. k,v: [B,T,Hkv,D]."""
    B, T, Hkv, D = k.shape
    rep = n_q_heads // Hkv
    k = k.repeat_interleave(rep, dim=2)
    v = v.repeat_interleave(rep, dim=2)
    return k, v


def attention(q, k, v, keep_mask=None, causal=True):
    """Dense (keep_mask=None) or sparse (keep_mask given) attention.
    q:[B,T,Hq,D] k,v:[B,T,Hkv,D] keep_mask:[B,T,T] (shared across heads). Returns [B,T,Hq,D]."""
    B, T, Hq, D = q.shape
    if k.shape[2] != Hq:
        k, v = _expand_kv_for_gqa(k, v, Hq)
    logits = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(D)        # [B,Hq,T,T]
    mask = _causal_mask(T, q.device)[None, None] if causal else torch.ones(1, 1, T, T, dtype=torch.bool, device=q.device)
    if keep_mask is not None:
        mask = mask & keep_mask[:, None]                                # [B,Hq,T,T]
    logits = logits.masked_fill(~mask, NEG_INF)
    p = logits.softmax(dim=-1)
    out = torch.einsum("bhts,bshd->bthd", p, v)
    return out, p


def attention_probs(q, k, causal=True):
    """True dense attention probabilities aggregated over query heads (mean) -> [B,T,T].
    Used as the indexer target during warm-up."""
    B, T, Hq, D = q.shape
    Hkv = k.shape[2]
    if Hkv != Hq:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=2)
    logits = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(D)
    mask = _causal_mask(T, q.device)[None, None]
    logits = logits.masked_fill(~mask, NEG_INF)
    p = logits.softmax(dim=-1).mean(dim=1)                              # mean over heads -> [B,T,T]
    return p


def indexer_kl_loss(index_logits, target_probs):
    """KL( stopgrad(p_attn) || softmax(index_logits) ), causal, averaged over valid queries.
    Trains the indexer to predict where attention goes. Stop-grad on target (the main model)."""
    B, T, _ = index_logits.shape
    causal = _causal_mask(T, index_logits.device)[None]
    li = index_logits.masked_fill(~causal, NEG_INF)
    logp_idx = li.log_softmax(dim=-1)
    p = target_probs.detach().clamp_min(1e-12)
    kl = (p * (p.log() - logp_idx)).sum(dim=-1)                          # [B,T]
    return kl.mean()


def topk_recall(index_logits, target_probs, k):
    """Quality metric: fraction of true attention mass captured by the indexer's top-k.
    1.0 = indexer perfectly identifies the tokens attention cares about."""
    keep = topk_sparse_mask(index_logits, k)                            # [B,T,T]
    captured = (target_probs.detach() * keep).sum(dim=-1)               # [B,T]
    total = target_probs.detach().sum(dim=-1).clamp_min(1e-12)
    return (captured / total).mean().item()
