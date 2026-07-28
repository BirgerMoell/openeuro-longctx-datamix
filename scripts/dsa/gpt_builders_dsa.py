"""Shadow gpt_builders.py for DSA dense-warmup / sparse-adapt.

Placed FIRST in PYTHONPATH so `from gpt_builders import gpt_builder` in pretrain_gpt.py picks THIS up.
It: (1) applies the ROCm FWHT patch, (2) bridges the DSA config fields onto the config,
(3) builds a per-layer dense/DSA block spec (all DSA by default; pattern from $DSA_PATTERN),
(4) for dense-warmup ($DSA_FREEZE_MODEL=1) freezes everything except the indexer.

Env knobs: DSA_PATTERN, DSA_TOPK (2048), DSA_N_HEADS (16), DSA_HEAD_DIM (128),
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

# SPARSE RUN: enable Triton O(L*k) attention, exact causal blocked index selection, and
# selected-set KL. Selection is still O(L^2) arithmetic, so this is an 8K correctness bridge.
if os.environ.get("DSA_SPARSE_RUN", "0") == "1":
    if os.environ.get("DSA_SPARSE", "0") != "1":
        raise RuntimeError("DSA_SPARSE_RUN=1 requires DSA_SPARSE=1 (selected-set KL)")
    if os.environ.get("DSA_FREEZE_MODEL", "0") == "1":
        raise RuntimeError("sparse adaptation must train the main model; set DSA_FREEZE_MODEL=0")
    from dsa_patches import apply_sparse_dsa_patches
    apply_sparse_dsa_patches()

# Warm-up: log indexer top-k RECALL (the real convergence signal; lm loss is noisy at small batch)
if os.environ.get("DSA_RECALL_LOG", "0") == "1":
    from dsa_patches import apply_indexer_recall_logging
    apply_indexer_recall_logging(every=int(os.environ.get("DSA_RECALL_EVERY", "36")))

# Million-token scaling cannot retain global O(L²) layers. The earlier 18/18 search pattern is
# useful only as a diagnostic; production DSA defaults to every layer sparse-capable.
DEFAULT_PATTERN = "S" * 36


def gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0("building GPT model (DSA) ...")
    if config is None:
        config = core_transformer_config_from_args(args)
    # DSA config bridge (fields the indexer reads that base TransformerConfig lacks)
    for k, v in dict(
        q_lora_rank=None, qk_pos_emb_head_dim=config.kv_channels, rope_type="rope",
        rotary_percent=args.rotary_percent, rotary_base=args.rotary_base,
        dsa_indexer_n_heads=int(os.environ.get("DSA_N_HEADS", "16")),
        dsa_indexer_head_dim=int(os.environ.get("DSA_HEAD_DIM", "128")),
        dsa_indexer_topk=int(os.environ.get("DSA_TOPK", "2048")),
        dsa_indexer_loss_coeff=float(os.environ.get("DSA_LOSS_COEFF", "0.1")),
        dsa_indexer_use_sparse_loss=(os.environ.get("DSA_SPARSE", "0") == "1"),
    ).items():
        setattr(config, k, v)
    require_non_interleaved = os.environ.get("DSA_REQUIRE_NON_INTERLEAVED_ROPE", "1") == "1"
    rotary_interleaved = bool(getattr(config, "rotary_interleaved", False))
    if require_non_interleaved and rotary_interleaved:
        raise RuntimeError(
            "DSA indexer requires non-interleaved RoPE, but config.rotary_interleaved=True"
        )
    print_rank_0(
        "DSA RoPE convention: "
        f"interleaved={rotary_interleaved} required_non_interleaved={require_non_interleaved} "
        f"pos_dim={config.qk_pos_emb_head_dim} base={config.rotary_base}"
    )
    # Training-side metric logging reads args rather than TransformerConfig.
    args.dsa_indexer_loss_coeff = config.dsa_indexer_loss_coeff

    pattern = os.environ.get("DSA_PATTERN", DEFAULT_PATTERN).upper()
    if len(pattern) != config.num_layers:
        raise RuntimeError(
            f"DSA_PATTERN len {len(pattern)} != num_layers {config.num_layers}"
        )
    invalid_pattern = set(pattern) - {"S", "F"}
    if invalid_pattern:
        raise RuntimeError(f"DSA_PATTERN contains invalid layer codes: {invalid_pattern}")
    if (
        os.environ.get("DSA_SPARSE_RUN", "0") == "1"
        and "F" in pattern
        and os.environ.get("DSA_ALLOW_DENSE_LAYERS", "0") != "1"
    ):
        raise RuntimeError(
            "sparse adaptation defaults to all-S because global dense layers cannot scale; "
            "set DSA_ALLOW_DENSE_LAYERS=1 only for a bounded short-context diagnostic"
        )
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
    # FAIL-CLOSED base load. On cold start we REQUIRE DSA_LOAD_BASE and abort unless every
    # non-indexer key loads (0 missing). Resume (Megatron --load of a *complete* warm-up ckpt with
    # latest_checkpointed_iteration.txt) is gated in the sbatch, not here. Never silently random-init.
    load_base = os.environ.get("DSA_LOAD_BASE", "")
    resume = os.environ.get("DSA_RESUME", "0") == "1"
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
        assert len(state) > 0, f"FAIL-CLOSED: base load returned 0 tensors from {load_base}"
        assert len(miss) == 0, f"FAIL-CLOSED: {len(miss)} non-indexer keys missing (e.g. {miss[:3]})"
    elif not resume:
        raise RuntimeError("FAIL-CLOSED: cold start with no DSA_LOAD_BASE and DSA_RESUME!=1 — "
                           "refusing to train from random init.")

    # Hybrid F/S blocks are checkpoint-heterogeneous: indexer tensors exist only in DSA layers.
    # Keep this disabled while loading the homogeneous base checkpoint above (whose transformer
    # tensors share one layer-sharded key), then enable it for warm-up save/resume so each layer
    # gets its own checkpoint key.  Otherwise dist-checkpoint validation expects indexer shards in
    # every layer and rejects the intentional holes at dense layers as an invalid access pattern.
    config.hetereogenous_dist_checkpoint = True
    model.config.hetereogenous_dist_checkpoint = True
    model.decoder.config.hetereogenous_dist_checkpoint = True
    print_rank_0(
        "DSA checkpoint layout: "
        f"builder={config.hetereogenous_dist_checkpoint} "
        f"model={model.config.hetereogenous_dist_checkpoint} "
        f"decoder={model.decoder.config.hetereogenous_dist_checkpoint}"
    )

    # Freeze base during indexer warm-up (DeepSeek: train ONLY the indexer). Record a hash of the
    # non-indexer weights so we can assert they stay bit-identical after training.
    if os.environ.get("DSA_FREEZE_MODEL", "0") == "1":
        fz = tr = 0; h = 0
        first_indexer_param = None
        for n, p in model.named_parameters():
            if "indexer" in n:
                p.requires_grad = True; tr += p.numel()
                if first_indexer_param is None:
                    first_indexer_param = (n, p)
            else:
                p.requires_grad = False; fz += p.numel()
                h ^= hash(p.detach().float().sum().item())
        print_rank_0(f"DSA FROZEN warm-up: froze {fz/1e9:.2f}B (base), training indexer {tr/1e6:.1f}M "
                     f"| non-indexer-weight-hash={h} | loss-coeff={args.dsa_indexer_loss_coeff} "
                     f"| grad-probe={os.environ.get('DSA_GRAD_PROBE', '0')}")

        # Selective activation checkpointing executes the first core-attention forward under
        # no-grad. With every upstream parameter frozen, its inputs otherwise also have
        # requires_grad=False, so backward never recomputes the DSA core and the attached KL loss
        # silently disappears. Make the embedding output a leaf activation when recomputation is
        # enabled; this restores the checkpoint backward path without unfreezing/updating the
        # embedding weights. Pipeline stages after the first already receive grad-bearing inputs.
        if (config.recompute_granularity is not None
                and os.environ.get("DSA_FORCE_INPUT_GRAD", "1") == "1"
                and pre_process and hasattr(model, "embedding")):
            def _force_frozen_activation_grad(_module, _inputs, output):
                if not torch.is_tensor(output):
                    raise TypeError(f"Expected tensor embedding output, got {type(output)}")
                return output.detach().requires_grad_(True)

            model.embedding.register_forward_hook(_force_frozen_activation_grad)
            print_rank_0("DSA frozen-recompute guard: embedding output will require grad")

        if os.environ.get("DSA_GRAD_PROBE", "0") == "1" and first_indexer_param is not None:
            if not getattr(_dsa, "_oellm_grad_probe_installed", False):
                original_indexer_loss = _dsa.compute_dsa_indexer_loss
                loss_probe_seen = False

                def _probe_indexer_loss(*loss_args, **loss_kwargs):
                    nonlocal loss_probe_seen
                    loss = original_indexer_loss(*loss_args, **loss_kwargs)
                    if not loss_probe_seen and (not torch.distributed.is_initialized()
                                                or torch.distributed.get_rank() == 0):
                        print_rank_0(
                            f"DSA LOSS PROBE: value={loss.detach().float().item():.9e} "
                            f"requires_grad={loss.requires_grad} grad_enabled={torch.is_grad_enabled()}"
                        )

                        if loss.requires_grad:
                            def _probe_loss_backward(grad):
                                print_rank_0(
                                    f"DSA LOSS BACKWARD PROBE: grad={grad.detach().float().item():.9e}"
                                )
                                return grad

                            loss.register_hook(_probe_loss_backward)
                        loss_probe_seen = True
                    return loss

                _dsa.compute_dsa_indexer_loss = _probe_indexer_loss
                _dsa._oellm_grad_probe_installed = True

            probe_name, probe_param = first_indexer_param
            probe_logged = False

            def _probe_indexer_grad(grad):
                nonlocal probe_logged
                if not probe_logged and (not torch.distributed.is_initialized()
                                         or torch.distributed.get_rank() == 0):
                    print_rank_0(
                        f"DSA GRAD PROBE {probe_name}: norm={grad.float().norm().item():.9e} "
                        f"max={grad.float().abs().max().item():.9e} "
                        f"nonzero={torch.count_nonzero(grad).item()}/{grad.numel()}"
                    )
                    probe_logged = True
                return grad

            probe_param.register_hook(_probe_indexer_grad)

    # Sparse adaptation must exercise both independent gradient paths: LM loss into
    # the main model and selected-set KL into the detached indexer. Probe one tensor
    # from each family during bounded correctness runs.
    if (os.environ.get("DSA_SPARSE_RUN", "0") == "1"
            and os.environ.get("DSA_GRAD_PROBE", "0") == "1"):
        probe_params = {}
        for name, param in model.named_parameters():
            family = "indexer" if "indexer" in name else "main"
            if family not in probe_params and param.requires_grad:
                probe_params[family] = (name, param)
            if len(probe_params) == 2:
                break
        if set(probe_params) != {"main", "indexer"}:
            raise RuntimeError(
                f"sparse grad probe could not find both parameter families: {probe_params.keys()}"
            )
        seen = {"main": False, "indexer": False}

        def _make_sparse_grad_probe(family, name):
            def _probe(grad):
                if not seen[family] and (
                    not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                ):
                    print_rank_0(
                        f"DSA SPARSE {family.upper()} GRAD PROBE {name}: "
                        f"norm={grad.float().norm().item():.9e} "
                        f"max={grad.float().abs().max().item():.9e} "
                        f"nonzero={torch.count_nonzero(grad).item()}/{grad.numel()}"
                    )
                    seen[family] = True
                return grad

            return _probe

        for family, (name, param) in probe_params.items():
            param.register_hook(_make_sparse_grad_probe(family, name))
        print_rank_0(
            "DSA sparse dual-gradient probes installed: "
            + ", ".join(f"{family}={name}" for family, (name, _) in probe_params.items())
        )
    return model
