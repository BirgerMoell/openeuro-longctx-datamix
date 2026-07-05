"""Fused sparse DSA attention in Triton — ROCm-compatible (MI250X / gfx90a).

Replaces Megatron's `unfused_dsa_fn` (which materializes the full [b,np,sq,skv] scores + mask, O(L^2),
plus heavy copies). This computes, per query, a flash-style online softmax over ONLY its top-k
selected keys — O(L*k), no L^2 matrix, no gather-copies. Triton runs on ROCm (container has triton-3.2).

Interface matches `unfused_dsa_fn`:
    query [sq,b,np,hn]  key [skv,b,np,hn]  value [skv,b,np,hnv]  topk_indices [b,sq,k]  -> [sq,b,np*hnv]
Forward kernel here; backward via a torch.autograd.Function (added next).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_fwd(
    Q, K, V, TOPK, Out, Lse,
    sq, skv, k, scale,
    sqb_q, snp_q, sd_q,          # Q strides: [sq, (b*np folded), hn] -> we pass per-(b,np) base
    sk_k, sd_k,                  # K strides along skv and hn (per (b,np) base)
    sk_v, sd_v,                  # V strides
    st_sq, st_k,                 # TOPK strides along sq and k (per b base; shared across np)
    so_sq, so_d,                 # Out strides
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, HN: tl.constexpr,
):
    pid_m = tl.program_id(0)     # query block
    bh = tl.program_id(1)        # flattened (b*np) for Q/K/V/Out; TOPK uses b only (passed via base ptr)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)          # queries [BLOCK_M]
    m_mask = offs_m < sq
    offs_d = tl.arange(0, HN)
    q = tl.load(Q + bh * 0 + offs_m[:, None] * sqb_q + offs_d[None, :] * sd_q,
                mask=m_mask[:, None], other=0.0).to(tl.float32)   # [BLOCK_M, HN]

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HN], tl.float32)

    for j0 in range(0, k, BLOCK_K):
        offs_k = j0 + tl.arange(0, BLOCK_K)                    # [BLOCK_K]
        k_mask = offs_k < k
        idx = tl.load(TOPK + offs_m[:, None] * st_sq + offs_k[None, :] * st_k,
                      mask=m_mask[:, None] & k_mask[None, :], other=0)      # [BLOCK_M, BLOCK_K] key positions
        causal = idx <= offs_m[:, None]                       # key pos <= query pos
        valid = m_mask[:, None] & k_mask[None, :] & causal
        # gather K [BLOCK_M, BLOCK_K, HN] and compute scores [BLOCK_M, BLOCK_K]
        kptr = K + idx[:, :, None] * sk_k + offs_d[None, None, :] * sd_k
        kblk = tl.load(kptr, mask=valid[:, :, None], other=0.0).to(tl.float32)  # [M,K,HN]
        s = tl.sum(q[:, None, :] * kblk, axis=2) * scale       # [M,K]
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vptr = V + idx[:, :, None] * sk_v + offs_d[None, None, :] * sd_v
        vblk = tl.load(vptr, mask=valid[:, :, None], other=0.0).to(tl.float32)  # [M,K,HN]
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * vblk, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(Out + offs_m[:, None] * so_sq + offs_d[None, :] * so_d, out.to(Out.dtype.element_ty),
             mask=m_mask[:, None])
    tl.store(Lse + offs_m, m_i + tl.log(l_i), mask=m_mask)     # save for backward


def sparse_attn_forward(query, key, value, topk_indices, scale, BLOCK_M=32, BLOCK_K=64):
    sq, b, np, hn = query.shape
    skv = key.shape[0]
    k = topk_indices.shape[-1]
    out = torch.empty(sq, b, np, hn, device=query.device, dtype=query.dtype)
    lse = torch.empty(b, np, sq, device=query.device, dtype=torch.float32)
    grid = (triton.cdiv(sq, BLOCK_M), b * np)
    for bi in range(b):
        for hi in range(np):
            q = query[:, bi, hi]; kk = key[:, bi, hi]; vv = value[:, bi, hi]
            o = out[:, bi, hi]; ti = topk_indices[bi]
            _sparse_attn_fwd[(triton.cdiv(sq, BLOCK_M),)](
                q, kk, vv, ti, o, lse[bi, hi],
                sq, skv, k, scale,
                q.stride(0), 0, q.stride(1),
                kk.stride(0), kk.stride(1),
                vv.stride(0), vv.stride(1),
                ti.stride(0), ti.stride(1),
                o.stride(0), o.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, HN=hn,
            )
    return out.reshape(sq, b, np * hn), lse
