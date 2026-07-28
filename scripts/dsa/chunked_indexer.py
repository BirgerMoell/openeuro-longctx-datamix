"""Causal, key-blocked top-k selection for the DSA lightning indexer.

The upstream implementation materializes ``[batch, query, key]`` scores.  This module
scores one query/key tile at a time and retains only the selected scores and indices.
It is exact (not approximate) for causal self-attention, but its arithmetic is still
O(L^2); the blocking fixes peak score memory, not million-token compute.
"""

import os

import torch


def index_scores_block(q, weights, k_block):
    """Mirror ``DSAIndexer._compute_index_scores`` for one key block."""
    # q [sq,b,ih,hd], weights [sq,b,ih], k_block [blk,b,hd]
    scores = torch.einsum("sbhd,tbd->sbht", q.float(), k_block.float())
    scores = torch.relu(scores) * weights.float().unsqueeze(-1)
    return scores.sum(dim=2).transpose(0, 1)  # [b,sq,blk]


def _retained_bytes(batch, seqlen, topk):
    # Selected scores are fp32 and indices are int32.
    return batch * seqlen * topk * 8


def chunked_topk(q, weights, k, topk, mask=None, block=8192, q_block=512):
    """Return exact causal selected scores and indices without a full score matrix.

    Args:
        q: ``[sq,b,index_heads,head_dim]``.
        weights: ``[sq,b,index_heads]``.
        k: ``[sk,b,head_dim]``.
        topk: Maximum number of selected keys per query.
        mask: Optional additive mask ``[sq,sk]`` or ``[b,sq,sk]``.  Sparse DSA
            normally passes ``None`` and uses the implicit causal mask here.
        block: Key tile size.
        q_block: Query tile size.

    Returns:
        ``(selected_scores, selected_indices)`` with shapes ``[b,sq,k]``.
        Scores are fp32. Indices are int32; unavailable early-causal slots use -1.

    This function deliberately fails closed when merely retaining selected scores and
    indices would exceed ``DSA_MAX_RETAINED_SELECTION_BYTES`` (default 2 GiB).
    """
    if q_block <= 0 or block <= 0:
        raise ValueError(f"block sizes must be positive, got q_block={q_block}, block={block}")
    if mask is not None and mask.dim() not in (2, 3):
        raise ValueError(f"mask must be rank 2 or 3, got shape {tuple(mask.shape)}")

    sq, batch, _, _ = q.shape
    sk = k.shape[0]
    kk = min(int(topk), sk)
    if kk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    retained = _retained_bytes(batch, sq, kk)
    limit = int(os.environ.get("DSA_MAX_RETAINED_SELECTION_BYTES", str(2 * 1024**3)))
    if retained > limit:
        raise RuntimeError(
            "DSA selected-set retention exceeds the configured safety limit: "
            f"{retained / 1024**3:.2f} GiB > {limit / 1024**3:.2f} GiB. "
            "Flat exact selection is not a million-token solution; use a hierarchical index."
        )

    score_blocks = []
    index_blocks = []
    mask_is_batched = mask is not None and mask.dim() == 3

    for i0 in range(0, sq, q_block):
        i1 = min(i0 + q_block, sq)
        qb = q[i0:i1]
        wb = weights[i0:i1]
        query_positions = torch.arange(i0, i1, device=q.device).view(1, -1, 1)

        running_scores = None
        running_indices = None
        # No query in this tile can attend beyond i1-1.  Avoid scoring later key tiles.
        key_stop = min(sk, i1)
        for j0 in range(0, key_stop, block):
            j1 = min(j0 + block, key_stop)
            scores = index_scores_block(qb, wb, k[j0:j1])
            key_positions = torch.arange(j0, j1, device=q.device)
            causal = key_positions.view(1, 1, -1) <= query_positions
            scores = scores.masked_fill(~causal, float("-inf"))
            if mask is not None:
                mask_block = (
                    mask[:, i0:i1, j0:j1]
                    if mask_is_batched
                    else mask[i0:i1, j0:j1].unsqueeze(0)
                )
                scores = scores + mask_block

            indices = key_positions.view(1, 1, -1).expand(batch, i1 - i0, j1 - j0)
            if running_scores is not None:
                scores = torch.cat((running_scores, scores), dim=-1)
                indices = torch.cat((running_indices, indices), dim=-1)

            keep = min(kk, scores.shape[-1])
            selected = scores.topk(keep, dim=-1)
            running_scores = selected.values
            running_indices = torch.gather(indices, -1, selected.indices)

        if running_scores is None:
            raise RuntimeError("causal selection produced no key candidates")

        # Early causal rows have fewer than kk valid keys. Pad with explicit sentinels.
        if running_scores.shape[-1] < kk:
            pad = kk - running_scores.shape[-1]
            running_scores = torch.cat(
                (
                    running_scores,
                    torch.full(
                        (*running_scores.shape[:-1], pad),
                        float("-inf"),
                        device=q.device,
                        dtype=running_scores.dtype,
                    ),
                ),
                dim=-1,
            )
            running_indices = torch.cat(
                (
                    running_indices,
                    torch.full(
                        (*running_indices.shape[:-1], pad),
                        -1,
                        device=q.device,
                        dtype=running_indices.dtype,
                    ),
                ),
                dim=-1,
            )

        valid = torch.isfinite(running_scores)
        running_indices = torch.where(valid, running_indices, -1)
        score_blocks.append(running_scores)
        index_blocks.append(running_indices.to(torch.int32))

    return torch.cat(score_blocks, dim=1), torch.cat(index_blocks, dim=1)
