"""Selected-set KL loss for sparse DSA indexer training.

DeepSeek keeps the indexer objective active during sparse adaptation.  The teacher
distribution is dense-attention probability restricted to the selected key set;
there is no reason to reconstruct an O(L^2) matrix to compute it.
"""

import os

import torch
import torch.distributed as dist


def _tp_size(pg_collection):
    tp = getattr(pg_collection, "tp", None)
    if tp is None:
        return 1
    if hasattr(tp, "size"):
        return tp.size()
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=tp)
    return 1


def selected_set_indexer_loss(
    selected_scores,
    selected_indices,
    query,
    key,
    softmax_scale,
    loss_coeff,
    pg_collection,
    q_block=None,
):
    """Compute ``KL(dense-attention-on-selected || indexer-on-selected)``.

    Query heads may be grouped-query attention heads. Key heads remain in their
    native grouped form; each contiguous query-head group maps to one KV head.
    Main-model tensors are detached so this objective updates only the indexer.
    """
    if selected_scores is None:
        raise ValueError("selected-set KL requires selected indexer scores")
    if selected_scores.shape != selected_indices.shape:
        raise ValueError(
            f"score/index shape mismatch: {selected_scores.shape} vs {selected_indices.shape}"
        )

    sq, batch, n_query_heads, q_dim = query.shape
    sk, key_batch, n_kv_heads, k_dim = key.shape
    if key_batch != batch or k_dim != q_dim:
        raise ValueError(
            f"query/key mismatch: query={tuple(query.shape)} key={tuple(key.shape)}"
        )
    if n_query_heads % n_kv_heads != 0:
        raise ValueError(
            f"GQA requires query heads divisible by KV heads: {n_query_heads} vs {n_kv_heads}"
        )
    if selected_scores.shape[:2] != (batch, sq):
        raise ValueError(
            f"selected tensors must begin [batch,sq]=[{batch},{sq}], got {selected_scores.shape}"
        )

    heads_per_group = n_query_heads // n_kv_heads
    q_block = q_block or int(os.environ.get("DSA_KL_Q_BLOCK", "128"))
    if q_block <= 0:
        raise ValueError(f"DSA_KL_Q_BLOCK must be positive, got {q_block}")

    valid_all = selected_indices >= 0
    safe_all = selected_indices.clamp(min=0).to(torch.long)
    teacher_blocks = []

    # Teacher computation is deliberately detached from the main model. The selected
    # indexer scores retain their graph and are the only differentiable KL input.
    with torch.no_grad():
        q_detached = query.detach().float()
        k_detached = key.detach().float()
        for i0 in range(0, sq, q_block):
            i1 = min(i0 + q_block, sq)
            safe = safe_all[:, i0:i1]
            valid = valid_all[:, i0:i1]
            target = torch.zeros(
                safe.shape, device=query.device, dtype=torch.float32
            )

            for bi in range(batch):
                selected = safe[bi]
                for group in range(n_kv_heads):
                    h0 = group * heads_per_group
                    h1 = h0 + heads_per_group
                    q_group = q_detached[i0:i1, bi, h0:h1]
                    selected_k = k_detached[:, bi, group][selected]
                    logits = torch.einsum("qhd,qkd->hqk", q_group, selected_k)
                    logits.mul_(softmax_scale)
                    logits.masked_fill_(~valid[bi].unsqueeze(0), float("-inf"))
                    target[bi].add_(torch.softmax(logits, dim=-1).sum(dim=0))

            teacher_blocks.append(target)

        target = torch.cat(teacher_blocks, dim=1)
        if _tp_size(pg_collection) > 1:
            dist.all_reduce(target, group=pg_collection.tp)
        target = target.masked_fill(~valid_all, 0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    masked_scores = selected_scores.float().masked_fill(~valid_all, float("-inf"))
    indexer_prob = torch.softmax(masked_scores, dim=-1)
    indexer_prob = indexer_prob.masked_fill(~valid_all, 0.0)
    eps = 1e-10
    kl = target * ((target + eps).log() - (indexer_prob + eps).log())
    return kl.sum(dim=-1).mean() * float(loss_coeff)
