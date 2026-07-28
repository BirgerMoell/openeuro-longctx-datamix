#!/usr/bin/env python3
"""Dense-reference correctness gates for the OpenEuroLLM sparse DSA path."""

import argparse
import math
import os

import torch

from chunked_indexer import chunked_topk, index_scores_block
from dsa_sparse_loss import selected_set_indexer_loss


class _OneRankGroup:
    def size(self):
        return 1


class _OneRankPGCollection:
    tp = _OneRankGroup()


def _causal_indices(batch, seqlen, topk, device):
    result = torch.full((batch, seqlen, topk), -1, dtype=torch.int32, device=device)
    for bi in range(batch):
        for qi in range(seqlen):
            count = min(topk, qi + 1)
            result[bi, qi, :count] = torch.randperm(qi + 1, device=device)[:count].to(
                torch.int32
            )
    return result


def test_chunked_indexer(device):
    torch.manual_seed(7)
    sq, batch, index_heads, dim, topk = 37, 2, 16, 16, 11
    q = torch.randn(sq, batch, index_heads, dim, device=device, requires_grad=True)
    weights = torch.randn(sq, batch, index_heads, device=device, requires_grad=True)
    key = torch.randn(sq, batch, dim, device=device, requires_grad=True)

    selected_scores, selected_indices = chunked_topk(
        q, weights, key, topk, block=13, q_block=9
    )
    dense = index_scores_block(q, weights, key)
    positions = torch.arange(sq, device=device)
    dense = dense.masked_fill(
        positions.view(1, 1, -1) > positions.view(1, -1, 1), float("-inf")
    )
    expected = dense.topk(topk, dim=-1)
    expected_indices = torch.where(
        torch.isfinite(expected.values), expected.indices, -1
    ).to(torch.int32)

    # GEMM tiling changes fp32 accumulation order slightly between full and blocked einsums.
    torch.testing.assert_close(selected_scores, expected.values, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(selected_indices, expected_indices, rtol=0, atol=0)
    assert selected_indices.dtype == torch.int32
    assert torch.all(
        (selected_indices < 0)
        | (selected_indices <= positions.view(1, -1, 1))
    )

    finite_sum = torch.where(
        torch.isfinite(selected_scores), selected_scores, torch.zeros_like(selected_scores)
    ).sum()
    finite_sum.backward()
    for name, tensor in (("q", q), ("weights", weights), ("key", key)):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), name
        assert torch.count_nonzero(tensor.grad) > 0, name
    print("PASS chunked causal top-k: exact scores/indices, int32 sentinels, finite gradients")


def test_selection_retention_guard(device):
    torch.manual_seed(9)
    seqlen, batch, index_heads, dim, topk = 4, 1, 2, 4, 4
    q = torch.randn(seqlen, batch, index_heads, dim, device=device)
    weights = torch.randn(seqlen, batch, index_heads, device=device)
    key = torch.randn(seqlen, batch, dim, device=device)
    previous = os.environ.get("DSA_MAX_RETAINED_SELECTION_BYTES")
    os.environ["DSA_MAX_RETAINED_SELECTION_BYTES"] = "64"
    try:
        try:
            chunked_topk(q, weights, key, topk)
        except RuntimeError as error:
            assert "selected-set retention exceeds" in str(error)
        else:
            raise AssertionError("selection retention guard did not fail closed")
    finally:
        if previous is None:
            os.environ.pop("DSA_MAX_RETAINED_SELECTION_BYTES", None)
        else:
            os.environ["DSA_MAX_RETAINED_SELECTION_BYTES"] = previous
    print("PASS selection retention guard: oversized B×L×k state fails closed")


def _reference_selected_kl(scores, indices, query, key, scale, coeff):
    sq, batch, n_query_heads, _ = query.shape
    sk, _, n_kv_heads, _ = key.shape
    expanded_key = key.repeat_interleave(n_query_heads // n_kv_heads, dim=2)
    logits = torch.einsum("qbhd,kbhd->bhqk", query.float(), expanded_key.float()) * scale

    selection_count = torch.zeros(batch, sq, sk, dtype=torch.int32, device=query.device)
    valid = indices >= 0
    selection_count.scatter_add_(
        -1, indices.clamp(min=0).long(), valid.to(torch.int32)
    )
    selection = selection_count > 0
    logits = logits.masked_fill(~selection.unsqueeze(1), float("-inf"))
    target = torch.softmax(logits, dim=-1).sum(dim=1)
    target = target / target.sum(dim=-1, keepdim=True)
    selected_target = torch.gather(target, -1, indices.clamp(min=0).long())
    selected_target = selected_target.masked_fill(~valid, 0.0)

    probs = torch.softmax(scores.float().masked_fill(~valid, float("-inf")), dim=-1)
    probs = probs.masked_fill(~valid, 0.0)
    eps = 1e-10
    return (
        selected_target
        * ((selected_target + eps).log() - (probs + eps).log())
    ).sum(dim=-1).mean() * coeff


def test_selected_set_kl(device):
    torch.manual_seed(11)
    sq, batch, n_query_heads, n_kv_heads, dim, topk = 23, 2, 8, 2, 16, 9
    query = torch.randn(sq, batch, n_query_heads, dim, device=device)
    key = torch.randn(sq, batch, n_kv_heads, dim, device=device)
    indices = _causal_indices(batch, sq, topk, device)
    raw_scores = torch.randn(batch, sq, topk, device=device, requires_grad=True)
    scores = raw_scores.masked_fill(indices < 0, float("-inf"))
    scale, coeff = dim**-0.5, 0.1

    actual = selected_set_indexer_loss(
        scores,
        indices,
        query,
        key,
        scale,
        coeff,
        _OneRankPGCollection(),
        q_block=7,
    )
    expected = _reference_selected_kl(scores, indices, query, key, scale, coeff)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.backward()
    assert raw_scores.grad is not None and torch.isfinite(raw_scores.grad).all()
    assert torch.count_nonzero(raw_scores.grad) > 0
    print(
        f"PASS selected-set KL: native GQA matches full dense reference ({actual.item():.7f})"
    )


def _reference_sparse_attention(query, key, value, indices, scale):
    sq, batch, n_query_heads, _ = query.shape
    sk, _, n_kv_heads, _ = key.shape
    expanded_key = key.repeat_interleave(n_query_heads // n_kv_heads, dim=2)
    expanded_value = value.repeat_interleave(n_query_heads // n_kv_heads, dim=2)
    logits = torch.einsum("qbhd,kbhd->bhqk", query.float(), expanded_key.float()) * scale
    selection_count = torch.zeros(batch, sq, sk, dtype=torch.int32, device=query.device)
    valid = indices >= 0
    selection_count.scatter_add_(
        -1, indices.clamp(min=0).long(), valid.to(torch.int32)
    )
    selection = selection_count > 0
    probs = torch.softmax(logits.masked_fill(~selection.unsqueeze(1), float("-inf")), dim=-1)
    output = torch.einsum("bhqk,kbhd->qbhd", probs, expanded_value.float())
    return output.to(query.dtype).reshape(sq, batch, -1)


def test_triton_native_gqa(device):
    from triton_dsa import triton_dsa_attn

    torch.manual_seed(19)
    sq, batch, n_query_heads, n_kv_heads, dim, topk = 64, 2, 4, 2, 32, 16
    # Strided views exercise independent Q/K/V/DO strides in the custom backward.
    q_storage = torch.randn(
        sq, batch, n_query_heads, dim * 2, device=device, dtype=torch.float32
    )
    k_storage = torch.randn(
        sq, batch, n_kv_heads, dim * 2, device=device, dtype=torch.float32
    )
    v_storage = torch.randn(
        sq, batch, n_kv_heads, dim * 2, device=device, dtype=torch.float32
    )
    query = q_storage[..., ::2].requires_grad_()
    key = k_storage[..., ::2].requires_grad_()
    value = v_storage[..., ::2].requires_grad_()
    indices = _causal_indices(batch, sq, topk, device)
    scale = dim**-0.5

    actual = triton_dsa_attn(query, key, value, indices, scale, BLOCK_M=1, BLOCK_K=16)
    q_ref = query.detach().clone().requires_grad_()
    k_ref = key.detach().clone().requires_grad_()
    v_ref = value.detach().clone().requires_grad_()
    expected = _reference_sparse_attention(q_ref, k_ref, v_ref, indices, scale)
    if not torch.allclose(actual, expected, rtol=3e-4, atol=3e-4):
        flat_actual = actual.reshape(sq, batch, n_query_heads, dim)
        flat_expected = expected.reshape(sq, batch, n_query_heads, dim)
        print("TRITON FORWARD DEBUG")
        for query_row in (0, 1, 2, 3, 15, 16, 63):
            error = (flat_actual[query_row] - flat_expected[query_row]).abs()
            print(
                f"q={query_row} max={error.max().item():.6f} "
                f"mean={error.mean().item():.6f}"
            )
        print("actual q3/b0/h0", flat_actual[3, 0, 0, :8].tolist())
        print("expect q3/b0/h0", flat_expected[3, 0, 0, :8].tolist())
        print("indices q3/b0", indices[0, 3].tolist())
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)

    dout_storage = torch.randn(
        *actual.shape[:-1], actual.shape[-1] * 2, device=device, dtype=actual.dtype
    )
    dout = dout_storage[..., ::2]
    actual_grads = torch.autograd.grad((actual * dout).sum(), (query, key, value))
    expected_grads = torch.autograd.grad((expected * dout).sum(), (q_ref, k_ref, v_ref))
    for name, actual_grad, expected_grad in zip(
        ("dq", "dk", "dv"), actual_grads, expected_grads
    ):
        assert torch.isfinite(actual_grad).all(), name
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-3, atol=3e-3)
    print("PASS Triton sparse attention: native GQA forward/backward matches dense reference")


def test_noninterleaved_rope(device):
    try:
        from megatron.core.models.common.embeddings.rope_utils import (
            _apply_rotary_pos_emb_bshd,
        )
    except ModuleNotFoundError as error:
        if error.name != "megatron":
            raise
        print("SKIP RoPE convention: Megatron is not installed on this host")
        return False

    torch.manual_seed(23)
    seqlen, dim = 9, 16
    x = torch.randn(seqlen, 2, 3, dim, device=device)
    inv_freq = 1.0 / (10_000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    half_freqs = torch.outer(torch.arange(seqlen, device=device).float(), inv_freq)
    freqs = torch.cat((half_freqs, half_freqs), dim=-1).view(seqlen, 1, 1, dim)
    x1, x2 = x.chunk(2, dim=-1)
    manual = x * freqs.cos() + torch.cat((-x2, x1), dim=-1) * freqs.sin()
    actual = _apply_rotary_pos_emb_bshd(x, freqs, rotary_interleaved=False)
    torch.testing.assert_close(actual, manual, rtol=0, atol=1e-6)
    print("PASS RoPE convention: Megatron non-interleaved application matches manual reference")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--triton-only", action="store_true")
    parser.add_argument(
        "--require-megatron-rope",
        action="store_true",
        help="Fail instead of skipping the RoPE check when Megatron is unavailable.",
    )
    args = parser.parse_args()
    if args.cpu_only:
        device = torch.device("cpu")
    else:
        assert torch.cuda.is_available(), "GPU test requested but torch.cuda.is_available() is false"
        device = torch.device("cuda")

    if args.triton_only:
        test_triton_native_gqa(device)
        return

    test_chunked_indexer(device)
    test_selection_retention_guard(device)
    test_selected_set_kl(device)
    rope_tested = test_noninterleaved_rope(device)
    if args.require_megatron_rope and not rope_tested:
        raise AssertionError("Megatron is required for the non-interleaved RoPE gate")
    if device.type == "cuda":
        test_triton_native_gqa(device)
    print("ALL DSA CORRECTNESS TESTS PASSED")


if __name__ == "__main__":
    main()
