"""Context-parallel sequence helpers for the standalone DSA overlay.

Megatron balances causal context parallelism by giving CP rank ``r`` two
chunks: ``r`` and ``2 * cp_size - r - 1``.  Collectives return tensors in CP
rank order, not token order.  Sparse attention needs canonical global token
indices, so this module owns the reorder and its autograd rule.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _group_size(group) -> int:
    if group is None:
        return 1
    if hasattr(group, "size"):
        return int(group.size())
    return int(dist.get_world_size(group=group))


def _group_rank(group) -> int:
    if group is None:
        return 0
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank(group=group))
    if hasattr(group, "rank"):
        return int(group.rank())
    raise RuntimeError("cannot determine context-parallel rank")


def cp_global_positions(local_length: int, cp_size: int, cp_rank: int, *, device=None):
    """Return the global positions in a Megatron zig-zag CP-local tensor."""
    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"invalid CP topology: size={cp_size} rank={cp_rank}")
    if local_length <= 0 or local_length % 2:
        raise ValueError(
            f"zig-zag CP local length must be positive and even, got {local_length}"
        )
    chunk = local_length // 2
    first = torch.arange(
        cp_rank * chunk, (cp_rank + 1) * chunk, device=device, dtype=torch.int64
    )
    mirror = 2 * cp_size - cp_rank - 1
    second = torch.arange(
        mirror * chunk, (mirror + 1) * chunk, device=device, dtype=torch.int64
    )
    return torch.cat((first, second), dim=0)


def reorder_cp_rank_gather(gathered: torch.Tensor) -> torch.Tensor:
    """Reorder ``[cp, local_seq, ...]`` from rank order to global token order."""
    if gathered.ndim < 2:
        raise ValueError(f"gathered tensor must be at least rank 2, got {gathered.shape}")
    cp_size, local_length = gathered.shape[:2]
    if cp_size <= 0 or local_length <= 0 or local_length % 2:
        raise ValueError(
            f"invalid gathered CP shape: cp={cp_size}, local_length={local_length}"
        )
    chunk = local_length // 2
    first = gathered[:, :chunk].reshape(cp_size * chunk, *gathered.shape[2:])
    second = gathered.flip(0)[:, chunk:].reshape(cp_size * chunk, *gathered.shape[2:])
    return torch.cat((first, second), dim=0)


def pack_global_gradient_for_cp(global_gradient: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Inverse layout of :func:`reorder_cp_rank_gather`, packed by source rank."""
    global_length = global_gradient.shape[0]
    if cp_size <= 0 or global_length % (2 * cp_size):
        raise ValueError(
            f"global length {global_length} is not divisible by 2*CP ({2 * cp_size})"
        )
    chunk = global_length // (2 * cp_size)
    blocks = global_gradient.reshape(2 * cp_size, chunk, *global_gradient.shape[1:])
    by_rank = [
        torch.cat((blocks[rank], blocks[2 * cp_size - rank - 1]), dim=0)
        for rank in range(cp_size)
    ]
    return torch.stack(by_rank, dim=0)


class _GatherGlobalSequence(torch.autograd.Function):
    """All-gather CP-local activations and return canonical global token order.

    Every CP rank consumes the global result.  Backward therefore sums each
    source rank's gradient contributions across the CP group before returning
    the local zig-zag slice.
    """

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group):
        cp_size = _group_size(group)
        cp_rank = _group_rank(group)
        ctx.group = group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        if cp_size == 1:
            return local_tensor
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("CP all-gather requires initialized torch.distributed")
        if local_tensor.shape[0] % 2:
            raise ValueError(
                f"CP-local sequence must contain two equal chunks, got {local_tensor.shape[0]}"
            )
        contiguous = local_tensor.contiguous()
        rank_tensors = [torch.empty_like(contiguous) for _ in range(cp_size)]
        dist.all_gather(rank_tensors, contiguous, group=group)
        return reorder_cp_rank_gather(torch.stack(rank_tensors, dim=0))

    @staticmethod
    def backward(ctx, global_gradient: torch.Tensor):
        if ctx.cp_size == 1:
            return global_gradient, None
        packed = pack_global_gradient_for_cp(global_gradient, ctx.cp_size).contiguous()
        dist.all_reduce(packed, group=ctx.group)
        return packed[ctx.cp_rank], None


def gather_global_sequence(local_tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Differentiably gather a Megatron CP-local sequence into global order."""
    return _GatherGlobalSequence.apply(local_tensor, group)


def cp_topology(group=None):
    """Return ``(size, rank)`` for a possibly absent CP group."""
    return _group_size(group), _group_rank(group)
