"""Key-blocked (streaming) top-k for the DSA indexer — avoids materializing the [b,sq,sk] index
scores, which OOMs at 512K (69 GB @ CP=16). Scores keys in blocks, keeps a running top-k merge.
O(sq*k) memory. Used to monkey-patch DSAIndexer.forward_with_scores for sparse runs >256K.

`compute_index_scores_block(q, weights, k_block)` mirrors the indexer's `_compute_index_scores`
for one key block; `chunked_topk` merges block top-ks into the global top-k.
"""
import torch


def index_scores_block(q, weights, k_block):
    # q [sq,b,ih,hd]  weights [sq,b,ih]  k_block [blk,b,hd] -> scores [b,sq,blk]
    s = torch.einsum('sbhd,tbd->sbht', q.float(), k_block.float())
    s = torch.relu(s) * weights.float().unsqueeze(-1)
    return s.sum(dim=2).transpose(0, 1)                       # [b,sq,blk]


def chunked_topk(q, weights, k, topk, mask=None, block=32768):
    """Return topk_indices [b, sq, topk] using key-blocked running top-k (no [b,sq,sk] tensor).
    mask (optional) is [b, sq, sk] additive (e.g. causal -inf); sliced per block."""
    sq, b, ih, hd = q.shape
    sk = k.shape[0]
    kk = min(topk, sk)
    run_s = None; run_i = None
    for j0 in range(0, sk, block):
        j1 = min(j0 + block, sk)
        s = index_scores_block(q, weights, k[j0:j1])          # [b,sq,blk]
        if mask is not None:
            s = s + mask[..., j0:j1]                           # mask [sq,sk] or [b,sq,sk]; slice last dim
        idx = torch.arange(j0, j1, device=q.device).view(1, 1, -1).expand(b, sq, j1 - j0)
        if run_s is None:
            cs, ci = s, idx
        else:
            cs = torch.cat([run_s, s], dim=-1)
            ci = torch.cat([run_i, idx], dim=-1)
        m = min(kk, cs.shape[-1])
        top = cs.topk(m, dim=-1)
        run_s = top.values
        run_i = torch.gather(ci, -1, top.indices)
    return run_i                                              # [b,sq,topk]
