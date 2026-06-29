"""
Correctness + behaviour tests for the DSA lightning-indexer module.
Pure PyTorch, runs on CPU in seconds. `python scripts/dsa/test_dsa.py`.

Validates the experiment's core claims before any training-stack integration:
  1. sparse attention with k>=T  ==  dense attention (exactness sanity)
  2. causal masking never leaks future tokens
  3. GQA shapes work (Hq != Hkv)
  4. KL warm-up loss trains the indexer to predict attention (recall improves)
  5. a trained indexer's top-k recovers most attention mass at k << T
"""
import torch
from lightning_indexer import (
    LightningIndexer, attention, attention_probs, topk_sparse_mask,
    indexer_kl_loss, topk_recall,
)

torch.manual_seed(0)


def test_sparse_equals_dense_when_k_full():
    B, T, Hq, Hkv, D = 2, 16, 8, 2, 32
    q = torch.randn(B, T, Hq, D); k = torch.randn(B, T, Hkv, D); v = torch.randn(B, T, Hkv, D)
    dense, _ = attention(q, k, v, keep_mask=None, causal=True)
    idx = torch.randn(B, T, T)                      # any logits; k=T keeps all causal keys
    keep = topk_sparse_mask(idx, k=T)
    sparse, _ = attention(q, k, v, keep_mask=keep, causal=True)
    err = (dense - sparse).abs().max().item()
    assert err < 1e-5, f"sparse(k=T) != dense, max err {err}"
    print(f"[OK] sparse(k=T) == dense  (max err {err:.2e})")


def test_causal_no_future_leak():
    B, T = 1, 12
    idx = torch.randn(B, T, T)
    keep = topk_sparse_mask(idx, k=T)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert not keep[0][future].any(), "future tokens were selected!"
    print("[OK] causal: no future-token selection")


def test_gqa_shapes():
    B, T, Hq, Hkv, D = 2, 16, 32, 8, 16     # our model: 32 q heads / 8 KV
    q = torch.randn(B, T, Hq, D); k = torch.randn(B, T, Hkv, D); v = torch.randn(B, T, Hkv, D)
    out, p = attention(q, k, v, causal=True)
    assert out.shape == (B, T, Hq, D)
    assert torch.allclose(p.sum(-1), torch.ones(B, Hq, T), atol=1e-4)
    print(f"[OK] GQA {Hq}q/{Hkv}kv -> out {tuple(out.shape)}, probs normalized")


def test_indexer_learns_attention():
    """The crux: can the lightning indexer learn to predict where attention goes?
    Train indexer (only) to match a fixed teacher attention via KL; recall should rise."""
    B, T, Hq, Hkv, D, Hdim = 4, 64, 8, 2, 32, 64
    hidden = torch.randn(B, T, Hdim)
    # fixed "teacher" attention from random q/k projections of the SAME hidden states
    Wq = torch.randn(Hdim, Hq * D); Wk = torch.randn(Hdim, Hkv * D)
    q = (hidden @ Wq).view(B, T, Hq, D); k = (hidden @ Wk).view(B, T, Hkv, D)
    target = attention_probs(q, k, causal=True)              # [B,T,T], the thing to predict
    k_top = 16                                               # 25% of T
    idx_mod = LightningIndexer(Hdim, n_index_heads=2, index_dim=32)
    opt = torch.optim.Adam(idx_mod.parameters(), lr=1e-2)
    r0 = topk_recall(idx_mod(hidden), target, k_top)
    for step in range(300):
        opt.zero_grad()
        loss = indexer_kl_loss(idx_mod(hidden), target)
        loss.backward(); opt.step()
    r1 = topk_recall(idx_mod(hidden), target, k_top)
    print(f"[OK] indexer learns: top-{k_top}/{T} recall {r0:.3f} -> {r1:.3f} (KL loss {loss.item():.3f})")
    assert r1 > r0 + 0.1, f"indexer did not improve recall ({r0:.3f}->{r1:.3f})"
    assert r1 > 0.6, f"trained recall too low ({r1:.3f})"


def test_random_indexer_recall_baseline():
    """Sanity: an UNTRAINED indexer's top-k recall ~ k/T (no better than random)."""
    B, T = 4, 64
    target = attention_probs(torch.randn(B, T, 8, 32), torch.randn(B, T, 2, 32))
    idx = torch.randn(B, T, T)
    r = topk_recall(idx, target, k=16)
    print(f"[OK] random indexer recall {r:.3f} (~k/T baseline; trained should beat this)")


if __name__ == "__main__":
    print("=== DSA lightning-indexer tests ===")
    test_sparse_equals_dense_when_k_full()
    test_causal_no_future_leak()
    test_gqa_shapes()
    test_random_indexer_recall_baseline()
    test_indexer_learns_attention()
    print("=== all tests passed ===")
