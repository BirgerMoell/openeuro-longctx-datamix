"""Correctness-first sparse DSA patches for the OpenEuroLLM GQA model.

The patch keeps the official two-gradient-path design:

* LM loss differentiates through sparse attention into the main model.
* Selected-set KL differentiates into the detached lightning indexer only.

Selection is causal and score-memory-blocked, K/V stay in native GQA form, and
unsupported masks/context parallelism fail closed. Exact flat selection is still
O(L^2) arithmetic and is intentionally guarded as an 8K correctness bridge.
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
from dsa_sparse_loss import selected_set_indexer_loss
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
    """Original indexer projections plus exact causal blocked selected scores."""
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

    selected_scores, selected_indices = chunked_topk(
        q,
        weights,
        k,
        min(self.index_topk, seqlen),
        mask=mask,
        block=int(os.environ.get("DSA_INDEX_BLOCK", "8192")),
        q_block=int(os.environ.get("DSA_INDEX_Q_BLOCK", "512")),
    )
    return selected_scores, selected_indices


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
            f"only aligned causal self-attention is supported: sq={query.shape[0]} sk={key.shape[0]}"
        )
    if _group_size(getattr(self.indexer.pg_collection, "cp", None)) != 1:
        raise NotImplementedError(
            "context parallel sparse DSA is not implemented correctly yet; refusing to run"
        )

    # Standard Megatron may hand us a prebuilt causal attention_mask. It is intentionally
    # ignored only because the explicit causal enum above and aligned self-attention fully
    # define the supported mask. Arbitrary padding/custom masks are not accepted.
    if attention_mask is not None and attention_mask.ndim not in (2, 4):
        raise NotImplementedError(
            f"unsupported attention mask shape for causal sparse DSA: {attention_mask.shape}"
        )

    selected_scores, selected_indices = self.indexer.forward_with_scores(
        x.detach(), qr.detach(), mask=None, packed_seq_params=None
    )
    output = _dsa.unfused_dsa_fn(
        query, key, value, selected_indices, self.softmax_scale
    )

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
        "[dsa_patches] sparse DSA correctness path enabled: causal blocked selection + "
        "native GQA Triton + selected-set KL",
        flush=True,
    )


# ---- Per-layer indexer attention-mass recall -------------------------------------------------
_recall_state = {"enabled": False, "every": 36, "ks": (512, 1024, 2048), "counts": {}}


def _recall_selected_indices(index_scores, topk_indices, query_rows, k_eval, sk):
    sampled_scores = index_scores[:, query_rows]
    if index_scores.shape[-1] == sk:
        positions = torch.arange(sk, device=index_scores.device).view(1, 1, -1)
        causal = positions <= query_rows.view(1, -1, 1)
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
        key_positions = torch.arange(sk, device=query.device).view(1, 1, -1)
        causal = key_positions <= query_rows.view(1, -1, 1)
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
                index_scores, topk_indices, query_rows, k_eval, sk
            )
            valid = (indices >= 0) & (indices < sk) & (
                indices <= query_rows.view(1, -1, 1)
            )
            captured_per_row = torch.gather(target, -1, indices.clamp(min=0))
            captured_per_row = (captured_per_row * valid).sum(dim=-1)
            metrics = [captured_per_row.mean()]
            for quartile in range(4):
                lo = quartile * nq // 4
                hi = (quartile + 1) * nq // 4
                metrics.append(captured_per_row[:, lo:hi].mean())
            metrics = torch.stack(metrics)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            if not dist.is_initialized() or dist.get_rank() == 0:
                quartiles = ",".join(f"{value:.3f}" for value in metrics[1:].tolist())
                print(
                    f"[dsa recall] layer={layer}/{num_layers} layer-step={step} "
                    f"top-{min(k_eval, index_scores.shape[-1])} "
                    f"mass={metrics[0].item():.3f} quartiles=[{quartiles}]",
                    flush=True,
                )


def apply_indexer_recall_logging(every=36, k_eval=None):
    """Wrap the active indexer loss with staggered, correctly attributed recall probes."""
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
