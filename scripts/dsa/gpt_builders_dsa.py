"""Shadow gpt_builders.py for DSA dense-warmup / sparse-adapt.

Placed FIRST in PYTHONPATH so `from gpt_builders import gpt_builder` in pretrain_gpt.py picks THIS up.
It: (1) applies the ROCm FWHT patch, (2) bridges the DSA config fields onto the config,
(3) builds a HYBRID per-layer dense/DSA block spec (pattern from $DSA_PATTERN),
(4) for dense-warmup ($DSA_FREEZE_MODEL=1) freezes everything except the indexer.

Env knobs: DSA_PATTERN, DSA_TOPK (2048), DSA_N_HEADS (4), DSA_HEAD_DIM (128),
DSA_LOSS_COEFF (0.1), DSA_SPARSE (0=warmup/dense,1=sparse), DSA_FREEZE_MODEL (1=indexer-only).
"""
import os, sys, torch
from megatron.core.models.gpt import GPTModel
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from megatron_gqa_dsa import get_gqa_dsa_block_spec

# ROCm has no fast_hadamard_transform -> pure-torch normalized FWHT
import megatron.core.transformer.experimental_attention_variant.dsa as _dsa
def _fwht(x):
    d = x.shape[-1]; shp = x.shape; y = x.reshape(-1, d).float(); h = 1
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        y = torch.stack([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], dim=2).reshape(-1, d); h *= 2
    return (y * (d ** -0.5)).reshape(shp).to(x.dtype)
_dsa.rotate_activation = _fwht

DEFAULT_PATTERN = "FFFFSSFFFFFFFFSSFFFSSSSSSSSSSSSSSFFF"  # from dsa_layer_search on our 9B


def gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0("building GPT model (DSA hybrid) ...")
    if config is None:
        config = core_transformer_config_from_args(args)
    # DSA config bridge (fields the indexer reads that base TransformerConfig lacks)
    for k, v in dict(
        q_lora_rank=None, qk_pos_emb_head_dim=config.kv_channels, rope_type="rope",
        rotary_percent=args.rotary_percent, rotary_base=args.rotary_base,
        dsa_indexer_n_heads=int(os.environ.get("DSA_N_HEADS", "4")),
        dsa_indexer_head_dim=int(os.environ.get("DSA_HEAD_DIM", "128")),
        dsa_indexer_topk=int(os.environ.get("DSA_TOPK", "2048")),
        dsa_indexer_loss_coeff=float(os.environ.get("DSA_LOSS_COEFF", "0.1")),
        dsa_indexer_use_sparse_loss=(os.environ.get("DSA_SPARSE", "0") == "1"),
    ).items():
        setattr(config, k, v)

    pattern = os.environ.get("DSA_PATTERN", DEFAULT_PATTERN)
    assert len(pattern) == config.num_layers, \
        f"DSA_PATTERN len {len(pattern)} != num_layers {config.num_layers}"
    spec = get_gqa_dsa_block_spec(TESpecProvider(), pattern, qk_layernorm=args.qk_layernorm)
    print_rank_0(f"DSA pattern ({pattern.count('S')}/{len(pattern)} sparse): {pattern} "
                 f"| topk={config.dsa_indexer_topk} sparse={config.dsa_indexer_use_sparse_loss}")

    model = GPTModel(
        config=config, transformer_layer_spec=spec, vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings, pre_process=pre_process,
        post_process=post_process, fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type, rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base, rope_scaling=args.use_rope_scaling, mtp_block_spec=None,
        vp_stage=vp_stage, pg_collection=pg_collection,
    )

    # Load the BASE weights ourselves, EXCLUDING the indexer keys, so the checkpoint key-set matches
    # exactly (the indexer is new). This avoids Megatron's FullyParallelLoadStrategyWrapper planner,
    # whose cross-rank gather_object deadlocks on the extra indexer keys. Pass Megatron NO --load;
    # point DSA_LOAD_BASE at the iter dir (e.g. .../ckpt_262144/iter_0000059).
    load_base = os.environ.get("DSA_LOAD_BASE", "")
    if load_base:
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
        ssd = model.sharded_state_dict()
        ssd_base = {k: v for k, v in ssd.items() if "indexer" not in k}
        strat = get_default_load_sharded_strategy(load_base)
        state = dist_checkpointing.load(ssd_base, load_base, strat)
        res = model.load_state_dict(state, strict=False)
        miss = [k for k in getattr(res, "missing_keys", []) if "indexer" not in k]
        print_rank_0(f"DSA base-load from {load_base}: {len(state)} tensors; indexer at init; "
                     f"non-indexer missing: {len(miss)}")

    # Dense-warmup: freeze everything except the indexer (keep the 256K model exactly intact)
    if os.environ.get("DSA_FREEZE_MODEL", "0") == "1":
        fz = tr = 0
        for n, p in model.named_parameters():
            if "indexer" in n:
                p.requires_grad = True; tr += p.numel()
            else:
                p.requires_grad = False; fz += p.numel()
        print_rank_0(f"DSA dense-warmup: froze {fz/1e9:.2f}B, training indexer {tr/1e6:.1f}M params")
    return model
