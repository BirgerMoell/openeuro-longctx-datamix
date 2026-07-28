"""Fused sparse DSA attention in Triton — ROCm-compatible (MI250X / gfx90a).

Replaces Megatron's `unfused_dsa_fn` (which materializes the full [b,np,sq,skv] scores + mask, O(L^2),
plus heavy copies). This computes, per query, a flash-style online softmax over ONLY its top-k
selected keys — O(L*k), no L^2 matrix, no gather-copies. Triton runs on ROCm (container has triton-3.2).

Interface matches `unfused_dsa_fn`, but K/V remain in native grouped-query form:
    query [sq,b,np,hn]  key [skv,b,ng,hn]  value [skv,b,ng,hnv]
    topk_indices [b,sq,k]  -> [sq,b,np*hnv]
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
    # One program per query.  Earlier [M,K,D] broadcasts produced incorrect
    # reductions with ROCm Triton 3.2; [K,D] is both simpler and auditable.
    query_id = tl.program_id(0)
    offs_d = tl.arange(0, HN)
    q = tl.load(Q + query_id * sqb_q + offs_d * sd_q).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HN], tl.float32)

    for j0 in range(0, k, BLOCK_K):
        offs_k = j0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k
        idx = tl.load(
            TOPK + query_id * st_sq + offs_k * st_k,
            mask=k_mask,
            other=-1,
        )
        valid = k_mask & (idx >= 0) & (idx < skv) & (idx <= query_id)
        safe_idx = tl.maximum(idx, 0)
        kptr = K + safe_idx[:, None] * sk_k + offs_d[None, :] * sd_k
        kblk = tl.load(kptr, mask=valid[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * kblk, axis=1) * scale
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.where(valid, tl.exp(s - m_new), 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        vptr = V + safe_idx[:, None] * sk_v + offs_d[None, :] * sd_v
        vblk = tl.load(vptr, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * vblk, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(Out + query_id * so_sq + offs_d * so_d, out.to(Out.dtype.element_ty))
    tl.store(Lse + query_id, m_i + tl.log(l_i))


def sparse_attn_forward(query, key, value, topk_indices, scale, BLOCK_M=1, BLOCK_K=32):
    sq, b, np, hn = query.shape
    skv, kb, ng, khn = key.shape
    if value.shape[:3] != (skv, b, ng):
        raise ValueError(f"key/value GQA shape mismatch: {key.shape} vs {value.shape}")
    if kb != b or khn != hn or value.shape[-1] != hn:
        raise ValueError(
            "Triton DSA currently requires equal Q/K/V head dimensions and matching batch: "
            f"q={query.shape}, k={key.shape}, v={value.shape}"
        )
    if np % ng != 0:
        raise ValueError(f"query heads ({np}) must be divisible by KV heads ({ng})")
    if topk_indices.shape[:2] != (b, sq):
        raise ValueError(f"top-k shape {topk_indices.shape} does not match batch/query {(b, sq)}")
    heads_per_group = np // ng
    k = topk_indices.shape[-1]
    out = torch.empty(sq, b, np, hn, device=query.device, dtype=query.dtype)
    lse = torch.empty(b, np, sq, device=query.device, dtype=torch.float32)
    for bi in range(b):
        for hi in range(np):
            gi = hi // heads_per_group
            q = query[:, bi, hi]; kk = key[:, bi, gi]; vv = value[:, bi, gi]
            o = out[:, bi, hi]; ti = topk_indices[bi]
            _sparse_attn_fwd[(sq,)](
                q, kk, vv, ti, o, lse[bi, hi],
                sq, skv, k, scale,
                q.stride(0), 0, q.stride(1),
                kk.stride(0), kk.stride(1),
                vv.stride(0), vv.stride(1),
                ti.stride(0), ti.stride(1),
                o.stride(0), o.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, HN=hn,
            )
    return out, lse


@triton.jit
def _sparse_attn_bwd(
    Q, K, V, TOPK, DO, Lse, DQ, DK, DV,
    sq, skv, k, scale,
    sq_q, sd_q, sdo_q, sdo_d,
    sk_k, sd_k, sk_v, sd_v,
    sdq_q, sdq_d, sdk_k, sdk_d, sdv_k, sdv_d,
    st_sq, st_k,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, HN: tl.constexpr,
):
    query_id = tl.program_id(0)
    offs_d = tl.arange(0, HN)
    q = tl.load(Q + query_id * sq_q + offs_d * sd_q).to(tl.float32)
    do = tl.load(DO + query_id * sdo_q + offs_d * sdo_d).to(tl.float32)
    lse = tl.load(Lse + query_id)
    # pass 1: delta_i = sum_j p_ij * dp_ij, from the kernel's own fp32 recompute (avoids bf16-O cancellation error)
    delta = 0.0
    for j0 in range(0, k, BLOCK_K):
        offs_k = j0 + tl.arange(0, BLOCK_K); k_mask = offs_k < k
        idx = tl.load(TOPK + query_id * st_sq + offs_k * st_k, mask=k_mask, other=-1)
        valid = k_mask & (idx >= 0) & (idx < skv) & (idx <= query_id)
        safe_idx = tl.maximum(idx, 0)
        kb = tl.load(K + safe_idx[:, None] * sk_k + offs_d[None, :] * sd_k, mask=valid[:, None], other=0.0).to(tl.float32)
        vb = tl.load(V + safe_idx[:, None] * sk_v + offs_d[None, :] * sd_v, mask=valid[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * kb, axis=1) * scale
        p = tl.where(valid, tl.exp(s - lse), 0.0)
        delta += tl.sum(p * tl.sum(do[None, :] * vb, axis=1), axis=0)
    dq = tl.zeros([HN], tl.float32)
    for j0 in range(0, k, BLOCK_K):
        offs_k = j0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k
        idx = tl.load(TOPK + query_id * st_sq + offs_k * st_k, mask=k_mask, other=-1)
        valid = k_mask & (idx >= 0) & (idx < skv) & (idx <= query_id)
        safe_idx = tl.maximum(idx, 0)
        kblk = tl.load(K + safe_idx[:, None] * sk_k + offs_d[None, :] * sd_k, mask=valid[:, None], other=0.0).to(tl.float32)
        vblk = tl.load(V + safe_idx[:, None] * sk_v + offs_d[None, :] * sd_v, mask=valid[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * kblk, axis=1) * scale
        p = tl.where(valid, tl.exp(s - lse), 0.0)
        dp = tl.sum(do[None, :] * vblk, axis=1)
        ds = p * (dp - delta) * scale
        dq += tl.sum(ds[:, None] * kblk, axis=0)
        tl.atomic_add(DV + safe_idx[:, None] * sdv_k + offs_d[None, :] * sdv_d,
                      p[:, None] * do[None, :], mask=valid[:, None])
        tl.atomic_add(DK + safe_idx[:, None] * sdk_k + offs_d[None, :] * sdk_d,
                      ds[:, None] * q[None, :], mask=valid[:, None])
    tl.store(DQ + query_id * sdq_q + offs_d * sdq_d, dq.to(DQ.dtype.element_ty))


class _SparseDSA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, topk_indices, scale, BLOCK_M, BLOCK_K):
        out, lse = sparse_attn_forward(query, key, value, topk_indices, scale, BLOCK_M, BLOCK_K)
        ctx.save_for_backward(query, key, value, topk_indices, out, lse)
        ctx.scale = scale; ctx.blocks = (BLOCK_M, BLOCK_K)
        sq, b, np, hn = query.shape
        return out.reshape(sq, b, np * hn)

    @staticmethod
    def backward(ctx, dout):
        query, key, value, topk_indices, out, lse = ctx.saved_tensors
        sq, b, np, hn = query.shape
        skv, _, ng, _ = key.shape
        heads_per_group = np // ng
        k = topk_indices.shape[-1]
        BLOCK_M, BLOCK_K = ctx.blocks; scale = ctx.scale
        dout = dout.reshape(sq, b, np, hn)
        dq = torch.empty_like(query)
        dk = torch.zeros(key.shape, device=key.device, dtype=torch.float32)
        dv = torch.zeros(value.shape, device=value.device, dtype=torch.float32)
        for bi in range(b):
            for hi in range(np):
                gi = hi // heads_per_group
                qs, ks, vs = query[:, bi, hi], key[:, bi, gi], value[:, bi, gi]
                dos, dqs = dout[:, bi, hi], dq[:, bi, hi]
                dks, dvs = dk[:, bi, gi], dv[:, bi, gi]
                ti = topk_indices[bi]
                _sparse_attn_bwd[(sq,)](
                    qs, ks, vs, ti, dos, lse[bi, hi], dqs, dks, dvs,
                    sq, skv, k, scale,
                    qs.stride(0), qs.stride(1), dos.stride(0), dos.stride(1),
                    ks.stride(0), ks.stride(1), vs.stride(0), vs.stride(1),
                    dqs.stride(0), dqs.stride(1), dks.stride(0), dks.stride(1),
                    dvs.stride(0), dvs.stride(1), ti.stride(0), ti.stride(1),
                    BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, HN=hn,
                )
        return dq, dk.to(key.dtype), dv.to(value.dtype), None, None, None, None


def triton_dsa_attn(query, key, value, topk_indices, scale, BLOCK_M=1, BLOCK_K=32):
    """Drop-in for unfused_dsa_fn (differentiable). query[sq,b,np,hn] ... -> [sq,b,np*hn]."""
    return _SparseDSA.apply(query, key, value, topk_indices, scale, BLOCK_M, BLOCK_K)
