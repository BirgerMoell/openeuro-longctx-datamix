"""Hierarchical, causal block routing for long-context DSA.

The correctness-first 512K backend keeps every token in the current block and
routes a small number of earlier global blocks using the learned DSA indexer.
It never creates a query-by-key score matrix: block routing costs O(L^2 / B^2)
and scoring selected tokens costs O(L * K / B), where ``B`` is block size.
"""

from __future__ import annotations

import os

import torch

from chunked_indexer import _retained_bytes, index_scores_block


def _validate_query_blocks(query_positions: torch.Tensor, block_size: int) -> None:
    if query_positions.ndim != 1:
        raise ValueError(f"query_positions must be rank 1, got {query_positions.shape}")
    if query_positions.numel() % block_size:
        raise ValueError(
            f"local query length {query_positions.numel()} must be divisible by block {block_size}"
        )
    blocks = query_positions.reshape(-1, block_size)
    expected = blocks[:, :1] + torch.arange(
        block_size, device=query_positions.device, dtype=query_positions.dtype
    )
    if not torch.equal(blocks, expected):
        raise ValueError(
            "each local query block must be globally contiguous; choose a block size that "
            "divides each Megatron CP half-chunk"
        )
    if torch.any(blocks[:, 0].remainder(block_size) != 0):
        raise ValueError("local query blocks must start on global block boundaries")


def hierarchical_block_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    global_k: torch.Tensor,
    query_positions: torch.Tensor,
    topk: int,
    *,
    block_size: int = 256,
    routed_blocks: int = 1,
):
    """Select current-block tokens plus learned earlier global blocks.

    ``q`` and ``weights`` are CP-local indexer activations. ``global_k`` is in
    canonical token order.  A block's first query routes for the whole block;
    using the mean of all queries would leak future-token information into early
    routing decisions. Key blocks use means of entirely earlier blocks. The
    discrete route is non-differentiable, while selected scores retain gradients
    to the indexer projections used by the coarse router.
    """
    if block_size <= 0 or routed_blocks < 0:
        raise ValueError(
            f"invalid block router settings: block={block_size}, routed={routed_blocks}"
        )
    sq, batch, index_heads, head_dim = q.shape
    if weights.shape != (sq, batch, index_heads):
        raise ValueError(f"q/weight shape mismatch: q={q.shape}, weights={weights.shape}")
    if global_k.ndim != 3 or global_k.shape[1:] != (batch, head_dim):
        raise ValueError(f"q/global_k shape mismatch: q={q.shape}, k={global_k.shape}")
    if query_positions.device != q.device:
        raise ValueError("query_positions must be on the same device as q")
    if global_k.shape[0] % block_size:
        raise ValueError(
            f"global key length {global_k.shape[0]} must be divisible by block {block_size}"
        )
    _validate_query_blocks(query_positions, block_size)

    candidates = block_size * (1 + routed_blocks)
    if topk < candidates:
        raise ValueError(
            f"DSA_TOPK={topk} cannot retain current + routed blocks ({candidates} tokens)"
        )
    retained = _retained_bytes(batch, sq, int(topk))
    limit = int(os.environ.get("DSA_MAX_RETAINED_SELECTION_BYTES", str(2 * 1024**3)))
    if retained > limit:
        raise RuntimeError(
            "DSA selected-set retention exceeds the configured safety limit: "
            f"{retained / 1024**3:.2f} GiB > {limit / 1024**3:.2f} GiB"
        )

    query_blocks = sq // block_size
    key_blocks = global_k.shape[0] // block_size
    q_rep = q.reshape(query_blocks, block_size, batch, index_heads, head_dim)[:, 0]
    w_rep = weights.reshape(query_blocks, block_size, batch, index_heads)[:, 0]
    k_summary = global_k.reshape(key_blocks, block_size, batch, head_dim).mean(dim=1)

    current_blocks = query_positions.reshape(query_blocks, block_size)[:, 0] // block_size
    if torch.any(current_blocks >= key_blocks):
        raise ValueError("query positions extend past the global key sequence")

    block_scores = index_scores_block(q_rep, w_rep, k_summary)
    block_ids = torch.arange(key_blocks, device=q.device).view(1, 1, key_blocks)
    earlier = block_ids < current_blocks.view(1, query_blocks, 1)
    block_scores = block_scores.masked_fill(~earlier, float("-inf"))

    if routed_blocks:
        route_count = min(routed_blocks, key_blocks)
        routes = block_scores.topk(route_count, dim=-1)
        route_ids = torch.where(
            torch.isfinite(routes.values),
            routes.indices,
            torch.full_like(routes.indices, -1),
        )
        if route_count < routed_blocks:
            padding = torch.full(
                (batch, query_blocks, routed_blocks - route_count),
                -1,
                device=q.device,
                dtype=route_ids.dtype,
            )
            route_ids = torch.cat((route_ids, padding), dim=-1)
    else:
        route_ids = torch.empty(
            batch, query_blocks, 0, device=q.device, dtype=torch.long
        )

    current = current_blocks.view(1, query_blocks, 1).expand(batch, -1, -1)
    selected_blocks = torch.cat((current, route_ids), dim=-1)
    offsets = torch.arange(block_size, device=q.device).view(1, 1, 1, block_size)
    block_valid = selected_blocks >= 0
    token_indices = selected_blocks.clamp(min=0).unsqueeze(-1) * block_size + offsets
    token_indices = torch.where(
        block_valid.unsqueeze(-1), token_indices, torch.full_like(token_indices, -1)
    ).reshape(batch, query_blocks, candidates)

    k_by_batch = global_k.permute(1, 0, 2)
    safe_flat = token_indices.clamp(min=0).reshape(batch, -1)
    selected_k = torch.gather(
        k_by_batch,
        1,
        safe_flat.unsqueeze(-1).expand(-1, -1, head_dim),
    ).reshape(batch, query_blocks, candidates, head_dim)
    token_scores = torch.einsum(
        "rbhd,brkd->brhk", q_rep.float(), selected_k.float()
    )
    token_scores = torch.relu(token_scores) * w_rep.permute(1, 0, 2).float().unsqueeze(-1)
    token_scores = token_scores.sum(dim=2)

    token_indices = token_indices.repeat_interleave(block_size, dim=1)
    token_scores = token_scores.repeat_interleave(block_size, dim=1)
    causal = token_indices <= query_positions.view(1, sq, 1)
    valid = (token_indices >= 0) & causal
    token_indices = torch.where(valid, token_indices, torch.full_like(token_indices, -1))
    token_scores = token_scores.masked_fill(~valid, float("-inf"))

    if topk > candidates:
        pad = topk - candidates
        token_indices = torch.cat(
            (
                token_indices,
                torch.full(
                    (batch, sq, pad), -1, device=q.device, dtype=token_indices.dtype
                ),
            ),
            dim=-1,
        )
        token_scores = torch.cat(
            (
                token_scores,
                torch.full(
                    (batch, sq, pad),
                    float("-inf"),
                    device=q.device,
                    dtype=token_scores.dtype,
                ),
            ),
            dim=-1,
        )

    if not torch.all(((token_indices < 0) | (token_indices <= query_positions.view(1, sq, 1)))):
        raise RuntimeError("hierarchical router produced non-causal indices")
    if torch.any((token_indices >= 0).sum(dim=-1) == 0):
        raise RuntimeError("hierarchical router produced a query with no valid keys")
    return token_scores, token_indices.to(torch.int32)
