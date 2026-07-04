"""ROCm-native efficient sparse attention to replace Megatron's `unfused_dsa_fn`.

The reference builds the full [b,np,sq,skv] scores in FP32 + a materialized [b,sq,skv] top-k mask
(scatter) + full softmax — O(L^2), fp32, ~17 min/iter at 16K on MI250X. This version GATHERS only
the top-k keys/values per query and attends over them: O(L*k), bf16, query-blocked for bounded
memory. Drop-in: same signature/return as `unfused_dsa_fn`. Monkey-patch:
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa
    dsa.unfused_dsa_fn = efficient_dsa_fn
"""
import torch

NEG = float("-inf")


def efficient_dsa_fn(query, key, value, topk_indices, softmax_scale, block=2048):
    # query [sq,b,np,hn]  key [skv,b,np,hn]  value [skv,b,np,hnv]  topk_indices [b,sq,k] (global key pos)
    sq, b, np, hn = query.shape
    skv = key.shape[0]
    hnv = value.shape[3]
    k = topk_indices.shape[-1]
    q = query.permute(1, 2, 0, 3)          # [b,np,sq,hn]
    kk = key.permute(1, 2, 0, 3)           # [b,np,skv,hn]
    vv = value.permute(1, 2, 0, 3)         # [b,np,skv,hnv]
    out = torch.empty(b, np, sq, hnv, dtype=value.dtype, device=query.device)
    qpos_all = torch.arange(sq, device=query.device)

    for s0 in range(0, sq, block):
        s1 = min(s0 + block, sq)
        bs = s1 - s0
        qb = q[:, :, s0:s1]                                    # [b,np,bs,hn]
        idxb = topk_indices[:, s0:s1].clamp_(0, skv - 1)      # [b,bs,k] (clamp padding)
        idx_bnk = idxb.unsqueeze(1).expand(b, np, bs, k)     # [b,np,bs,k]
        # gather selected keys/values: [b,np,bs,k,hn]/[...,hnv]
        kg = torch.gather(kk.unsqueeze(2).expand(b, np, bs, skv, hn),
                          3, idx_bnk.unsqueeze(-1).expand(b, np, bs, k, hn))
        vg = torch.gather(vv.unsqueeze(2).expand(b, np, bs, skv, hnv),
                          3, idx_bnk.unsqueeze(-1).expand(b, np, bs, k, hnv))
        # scores [b,np,bs,k]
        sc = (qb.unsqueeze(3) * kg).sum(-1).float() * softmax_scale
        # causal: mask keys whose global pos > query pos
        qpos = qpos_all[s0:s1].view(1, 1, bs, 1)
        keypos = idxb.unsqueeze(1)                            # [b,1,bs,k]
        sc = sc.masked_fill(keypos > qpos, NEG)
        # mask duplicate selected indices (match scatter idempotency): keep only the FIRST occurrence.
        # dup[j] = True iff idxb[j] equals some STRICTLY-earlier position.
        eq = idxb.unsqueeze(-1) == idxb.unsqueeze(-2)                 # [b,bs,k,k]
        earlier = torch.tril(torch.ones(k, k, dtype=torch.bool, device=idxb.device), -1)
        dup = (eq & earlier).any(-1)                                  # [b,bs,k]
        sc = sc.masked_fill(dup.unsqueeze(1), NEG)
        sc = sc.softmax(-1).to(value.dtype)                  # [b,np,bs,k]
        out[:, :, s0:s1] = (sc.unsqueeze(-1) * vg).sum(3)     # [b,np,bs,hnv]

    return out.permute(2, 0, 1, 3).reshape(sq, b, np * hnv)
