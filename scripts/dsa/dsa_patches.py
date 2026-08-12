"""Correctness-first sparse DSA patches for the OpenEuroLLM GQA model.

The patch keeps the official two-gradient-path design:

* LM loss differentiates through sparse attention into the main model.
* Selected-set KL differentiates into the detached lightning indexer only.

Selection is causal, K/V stay in native GQA form, and unsupported masks fail
closed. ``flat_exact`` remains the 8K reference router. ``block_cp`` adds a
hierarchical router and differentiable global K/V exchange for a correctness-
first 512K context-parallel pipeline.
"""

import os

import torch
import torch.distributed as dist

import megatron.core.transformer.experimental_attention_variant.dsa as _dsa
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    rotate_activation,
)

from chunked_indexer import chunked_topk
from cp_utils import cp_global_positions, cp_topology, gather_global_sequence
from dsa_sparse_loss import selected_set_indexer_loss
from hierarchical_indexer import hierarchical_block_topk
from triton_dsa import triton_dsa_attn

try:
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
except Exception:
    from megatron.core.tensor_parallel import gather_from_sequence_parallel_region


def _group_size(group):
    if group is None:
        return 1
    if hasattr(group, "size"):
        return group.size()
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=group)
    return 1


def _chunked_forward_with_scores(self, x, qr, mask=None, packed_seq_params=None):
    """Original indexer projections plus the selected fail-closed router."""
    if packed_seq_params is not None:
        raise NotImplementedError("packed sequences are not supported by sparse DSA")

    rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
        None, None, x, self.config, packed_seq_params
    )
    if self.config.rope_type == "rope":
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
        mscale = 1.0
    else:
        rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

    if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
        x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
        qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

    seqlen, batch, _ = x.size()
    q, _ = self.linear_wq_b(qr)
    q = q.reshape(seqlen, batch, self.index_n_heads, self.index_head_dim)
    q = self._apply_rope(q, rotary_pos_emb, mscale)

    k, _ = self.linear_wk(x)
    k = self.k_norm(k)
    k = k.reshape(seqlen, batch, 1, self.index_head_dim)
    k = self._apply_rope(k, rotary_pos_emb, mscale)
    k = k.reshape(seqlen, batch, self.index_head_dim)

    q = rotate_activation(q)
    k = rotate_activation(k)
    weights, _ = self.linear_weights_proj(x)
    weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

    router = os.environ.get("DSA_ROUTER", "flat_exact").lower()
    cp_group = getattr(self.pg_collection, "cp", None)
    cp_size, cp_rank = cp_topology(cp_group)
    if router == "flat_exact":
        if cp_size != 1:
            raise RuntimeError(
                "flat_exact cannot produce global CP indices; use DSA_ROUTER=block_cp"
            )
        return chunked_topk(
            q,
            weights,
            k,
            min(self.index_topk, seqlen),
            mask=mask,
            block=int(os.environ.get("DSA_INDEX_BLOCK", "8192")),
            q_block=int(os.environ.get("DSA_INDEX_Q_BLOCK", "512")),
        )
    if router != "block_cp":
        raise RuntimeError(f"unknown DSA_ROUTER={router!r}; expected flat_exact or block_cp")
    if mask is not None:
        raise NotImplementedError("block_cp accepts only the implicit causal mask")

    query_positions = cp_global_positions(seqlen, cp_size, cp_rank, device=q.device)
    global_k = gather_global_sequence(k, cp_group)
    max_seq = int(os.environ.get("DSA_CP_MAX_SEQ", "524288"))
    if global_k.shape[0] > max_seq and os.environ.get("DSA_ALLOW_LONGER_CP", "0") != "1":
        raise RuntimeError(
            f"block_cp global sequence {global_k.shape[0]} exceeds validated cap {max_seq}; "
            "set DSA_ALLOW_LONGER_CP=1 only for a separately reviewed experiment"
        )
    return hierarchical_block_topk(
        q,
        weights,
        global_k,
        query_positions,
        self.index_topk,
        block_size=int(os.environ.get("DSA_BLOCK_SIZE", "256")),
        routed_blocks=int(os.environ.get("DSA_ROUTED_BLOCKS", "1")),
    )


def _selected_indexer_loss_bridge(
    index_scores,
    topk_indices,
    query,
    key,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    pg_collection,
):
    if not sparse_loss:
        raise RuntimeError(
            "sparse DSA must use selected-set KL (set DSA_SPARSE=1); "
            "dense O(L^2) KL is not available in this path"
        )
    return selected_set_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        pg_collection,
    )


def _sparse_dsa_forward(
    self,
    query,
    key,
    value,
    attention_mask,
    x,
    qr,
    attn_mask_type=None,
    attention_bias=None,
    packed_seq_params=None,
):
    """DSAttention.forward without any full causal mask or repeated GQA K/V."""
    if packed_seq_params is not None:
        raise NotImplementedError("packed sequences are not supported by sparse DSA")
    if attention_bias is not None:
        raise NotImplementedError("attention bias is not supported by sparse DSA")
    if attn_mask_type != _dsa.AttnMaskType.causal:
        raise NotImplementedError(
            f"sparse DSA requires an explicit causal mask type, got {attn_mask_type}"
        )
    if query.shape[0] != key.shape[0]:
        raise NotImplementedError(
            "the input must be aligned CP-local causal self-attention: "
            f"sq={query.shape[0]} sk={key.shape[0]}"
        )

    # Standard Megatron may hand us a prebuilt causal attention_mask. It is intentionally
    # ignored only because the explicit causal enum above and aligned self-attention fully
    # define the supported mask. Arbitrary padding/custom masks are not accepted.
    if attention_mask is not None and attention_mask.ndim not in (2, 4):
        raise NotImplementedError(
            f"unsupported attention mask shape for causal sparse DSA: {attention_mask.shape}"
        )

    router = os.environ.get("DSA_ROUTER", "flat_exact").lower()
    cp_group = getattr(self.indexer.pg_collection, "cp", None)
    cp_size, cp_rank = cp_topology(cp_group)
    if router == "flat_exact" and cp_size != 1:
        raise RuntimeError("flat_exact sparse attention requires context parallel size 1")
    if router not in ("flat_exact", "block_cp"):
        raise RuntimeError(f"unknown DSA_ROUTER={router!r}")

    selected_scores, selected_indices = self.indexer.forward_with_scores(
        x.detach(), qr.detach(), mask=None, packed_seq_params=None
    )
    query_positions = cp_global_positions(
        query.shape[0], cp_size, cp_rank, device=query.device
    )
    if router == "block_cp":
        key = gather_global_sequence(key, cp_group)
        value = gather_global_sequence(value, cp_group)
    if selected_indices.shape[:2] != (query.shape[1], query.shape[0]):
        raise RuntimeError(
            f"router returned wrong selection shape {selected_indices.shape} for q={query.shape}"
        )
    if torch.any(selected_indices.to(torch.int64) > query_positions.view(1, -1, 1)):
        raise RuntimeError("router returned non-causal global token indices")
    output = triton_dsa_attn(
        query,
        key,
        value,
        selected_indices,
        self.softmax_scale,
        query_positions=query_positions,
    )

    if not getattr(_dsa, "_oellm_runtime_logged", False):
        retained = selected_indices.numel() * (
            selected_indices.element_size() + selected_scores.element_size()
        )
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "[dsa runtime] "
                f"router={router} cp={cp_size} local_q={query.shape[0]} global_k={key.shape[0]} "
                f"topk={selected_indices.shape[-1]} retained={retained / 1024**2:.1f}MiB "
                f"qpos=[{query_positions.min().item()},{query_positions.max().item()}]",
                flush=True,
            )
        _dsa._oellm_runtime_logged = True

    if self.training and torch.is_grad_enabled():
        coeff = float(getattr(self.config, "dsa_indexer_loss_coeff", 0.0))
        if coeff <= 0:
            raise RuntimeError(
                "sparse adaptation requires a positive DSA indexer loss coefficient"
            )
        indexer_loss = _dsa.compute_dsa_indexer_loss(
            selected_scores,
            selected_indices,
            query.detach(),
            key.detach(),
            self.softmax_scale,
            coeff,
            True,
            self.indexer.pg_collection,
        )
        _dsa.DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=indexer_loss,
            layer_number=self.layer_number,
            num_layers=self.config.num_layers,
            avg_group=cp_group if cp_size > 1 else None,
        )
        output = _dsa.DSAIndexerLossAutoScaler.apply(output, indexer_loss)

    return output


def apply_sparse_dsa_patches():
    """Install sparse attention, selection, and selected-set KL exactly once."""
    if getattr(_dsa, "_oellm_sparse_correctness_patch", False):
        return
    _dsa._oellm_original_dsattention_forward = _dsa.DSAttention.forward
    _dsa.unfused_dsa_fn = triton_dsa_attn
    DSAIndexer.forward_with_scores = _chunked_forward_with_scores
    _dsa.compute_dsa_indexer_loss = _selected_indexer_loss_bridge
    _dsa.DSAttention.forward = _sparse_dsa_forward
    _dsa._oellm_native_gqa_sparse = True
    _dsa._oellm_sparse_correctness_patch = True
    print(
        "[dsa_patches] sparse DSA enabled: fail-closed router + native GQA Triton + "
        "selected-set KL",
        flush=True,
    )


# ---- Per-layer indexer attention-mass recall -------------------------------------------------
_recall_state = {"enabled": False, "every": 36, "ks": (512, 1024, 2048), "counts": {}}


def _recall_selected_indices(
    index_scores,
    topk_indices,
    query_rows,
    query_global_positions,
    k_eval,
    sk,
):
    sampled_scores = index_scores[:, query_rows]
    if index_scores.shape[-1] == sk:
        positions = torch.arange(sk, device=index_scores.device).view(1, 1, -1)
        causal = positions <= query_global_positions.view(1, -1, 1)
        sampled_scores = sampled_scores.masked_fill(~causal, float("-inf"))
        chosen = sampled_scores.topk(min(k_eval, sk), dim=-1)
        return chosen.indices.to(torch.long)

    sampled_indices = topk_indices[:, query_rows]
    chosen = sampled_scores.topk(min(k_eval, sampled_scores.shape[-1]), dim=-1)
    return torch.gather(sampled_indices.to(torch.long), -1, chosen.indices)


def _maybe_log_recall(
    index_scores,
    topk_indices,
    query,
    key,
    softmax_scale,
    pg_collection,
):
    state = _recall_state
    if not state["enabled"] or index_scores is None:
        return
    layer_info = getattr(_dsa, "_oellm_current_layer", None)
    if layer_info is None:
        return
    layer, num_layers = layer_info
    step = state["counts"].get(layer, 0) + 1
    state["counts"][layer] = step
    every = state["every"]
    # Stagger probes: with every=num_layers, exactly one layer is sampled per step.
    if (step - 1) % every != (layer - 1) % every:
        return

    with torch.no_grad():
        sq, batch, n_query_heads, dim = query.shape
        sk, _, n_kv_heads, _ = key.shape
        if n_query_heads % n_kv_heads:
            raise RuntimeError(
                f"recall probe cannot map GQA heads: {n_query_heads} vs {n_kv_heads}"
            )
        nq = min(64, sq)
        query_rows = torch.linspace(0, sq - 1, nq, device=query.device).round().long()
        cp = getattr(pg_collection, "cp", None)
        cp_size, cp_rank = cp_topology(cp)
        all_query_positions = cp_global_positions(sq, cp_size, cp_rank, device=query.device)
        query_global_positions = all_query_positions[query_rows]
        key_positions = torch.arange(sk, device=query.device).view(1, 1, -1)
        causal = key_positions <= query_global_positions.view(1, -1, 1)
        heads_per_group = n_query_heads // n_kv_heads
        target = torch.zeros(batch, nq, sk, device=query.device, dtype=torch.float32)

        qf = query.detach().float()
        kf = key.detach().float()
        for bi in range(batch):
            for group in range(n_kv_heads):
                h0 = group * heads_per_group
                h1 = h0 + heads_per_group
                logits = torch.einsum(
                    "qhd,kd->hqk", qf[query_rows, bi, h0:h1], kf[:, bi, group]
                )
                logits.mul_(softmax_scale)
                logits.masked_fill_(~causal, float("-inf"))
                target[bi].add_(torch.softmax(logits, dim=-1).sum(dim=0))

        tp = getattr(pg_collection, "tp", None)
        if _group_size(tp) > 1:
            dist.all_reduce(target, group=tp)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-20)

        for k_eval in state["ks"]:
            indices = _recall_selected_indices(
                index_scores,
                topk_indices,
                query_rows,
                query_global_positions,
                k_eval,
                sk,
            )
            valid = (indices >= 0) & (indices < sk) & (
                indices <= query_global_positions.view(1, -1, 1)
            )
            captured_per_row = torch.gather(target, -1, indices.clamp(min=0))
            captured_per_row = (captured_per_row * valid).sum(dim=-1)
            sums_and_counts = [
                captured_per_row.sum(),
                captured_per_row.new_tensor(captured_per_row.numel()),
            ]
            for quartile in range(4):
                lo = quartile * sk // 4
                hi = (quartile + 1) * sk // 4
                in_quartile = (query_global_positions >= lo) & (query_global_positions < hi)
                selected = captured_per_row[:, in_quartile]
                sums_and_counts.extend(
                    (selected.sum(), selected.new_tensor(selected.numel()))
                )
            sums_and_counts = torch.stack(sums_and_counts)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sums_and_counts, op=dist.ReduceOp.SUM)
            metrics = sums_and_counts[0::2] / sums_and_counts[1::2].clamp_min(1)
            if not dist.is_initialized() or dist.get_rank() == 0:
                quartiles = ",".join(f"{value:.3f}" for value in metrics[1:].tolist())
                print(
                    f"[dsa recall] layer={layer}/{num_layers} layer-step={step} "
                    f"cp={cp_size} sampled-queries={int(sums_and_counts[1].item())} "
                    f"top-{min(k_eval, index_scores.shape[-1])} "
                    f"mass={metrics[0].item():.3f} quartiles=[{quartiles}]",
                    flush=True,
                )


def apply_indexer_recall_logging(every=36, k_eval=None):
    """Wrap the active indexer loss with staggered, correctly attributed recall probes.

    For context-parallel block routing this computes an exact dense teacher only
    for 64 sampled local queries in one layer per update.  It never materializes
    an O(L^2) tensor, but it does deliberately add a bounded O(64*L) diagnostic
    so a long sparse run is gated by measured attention-mass recall rather than
    by training loss alone.
    """
    if getattr(_dsa, "_oellm_recall_wrapper", False):
        return
    ks_env = os.environ.get("DSA_RECALL_KS", "512,1024,2048")
    ks = tuple(int(value) for value in ks_env.split(",") if value.strip())
    if k_eval is not None:
        ks = (int(k_eval),)
    _recall_state.update(enabled=True, every=int(every), ks=ks, counts={})
    original = _dsa.compute_dsa_indexer_loss

    def wrapped(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        pg_collection,
    ):
        _maybe_log_recall(
            index_scores,
            topk_indices,
            query,
            key,
            softmax_scale,
            pg_collection,
        )
        return original(
            index_scores,
            topk_indices,
            query,
            key,
            softmax_scale,
            loss_coeff,
            sparse_loss,
            pg_collection,
        )

    _dsa.compute_dsa_indexer_loss = wrapped
    _dsa._oellm_recall_wrapper = True
    print(
        f"[dsa_patches] per-layer recall logging on: stagger={every}, ks={ks}",
        flush=True,
    )
