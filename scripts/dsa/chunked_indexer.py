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


def chunked_topk(q, weights, k, topk, mask=None, block=32768, q_block=8192):
    """Return topk_indices [b, sq, topk] via query-blocked + key-blocked running top-k. Peak memory
    is O(q_block * block), independent of sq/sk — so it scales to 512K+ (needed with CP=1 where sq is
    unsharded). mask (optional) is [sq,sk] or [b,sq,sk] additive (causal); sliced per block."""
    sq, b, ih, hd = q.shape
    sk = k.shape[0]
    kk = min(topk, sk)
    out = torch.empty(b, sq, kk, dtype=torch.long, device=q.device)
    m3 = (mask is not None and mask.dim() == 3)
    for i0 in range(0, sq, q_block):
        i1 = min(i0 + q_block, sq)
        qb, wb = q[i0:i1], weights[i0:i1]
        run_s = None; run_i = None
        for j0 in range(0, sk, block):
            j1 = min(j0 + block, sk)
            s = index_scores_block(qb, wb, k[j0:j1])           # [b, qb, blk]
            if mask is not None:
                s = s + (mask[:, i0:i1, j0:j1] if m3 else mask[i0:i1, j0:j1])
            idx = torch.arange(j0, j1, device=q.device).view(1, 1, -1).expand(b, i1 - i0, j1 - j0)
            if run_s is None:
                cs, ci = s, idx
            else:
                cs = torch.cat([run_s, s], dim=-1); ci = torch.cat([run_i, idx], dim=-1)
            m = min(kk, cs.shape[-1])
            top = cs.topk(m, dim=-1)
            run_s = top.values; run_i = torch.gather(ci, -1, top.indices)
        out[:, i0:i1] = run_i
    return out                                                 # [b,sq,topk]
