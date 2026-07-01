"""GPU smoke test: build GQA-DSA attention on the MI250X, run fwd/bwd. TP=1, tiny.
Run inside the container on 1 GPU. Prints the first real blocker if any."""
import os, sys, torch

def main():
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(0)
    from megatron.core import parallel_state as ps
    ps.initialize_model_parallel(tensor_model_parallel_size=1)
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.spec_utils import build_module
    sys.path.insert(0, os.path.dirname(__file__))
    from megatron_gqa_dsa import get_gqa_dsa_attention_spec

    H, NH, NKV, HD = 512, 8, 2, 64
    cfg = TransformerConfig(
        num_layers=1, hidden_size=H, num_attention_heads=NH, num_query_groups=NKV,
        kv_channels=HD, use_cpu_initialization=True, bf16=True,
        add_bias_linear=False, qk_layernorm=True, normalization="RMSNorm",
        position_embedding_type="rope", rotary_base=32000000,
        # DSA config:
        dsa_indexer_n_heads=2, dsa_indexer_head_dim=64, dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=0.1, dsa_indexer_use_sparse_loss=False,
        q_lora_rank=None, rope_type="rope",
    )
    # backend provider (local impl, per the DSA spec comment)
    try:
        from megatron.core.models.backends import LocalSpecProvider as Backend
    except Exception:
        from megatron.core.models.gpt.gpt_layer_specs import LocalSpecProvider as Backend  # fallback
    backend = Backend()
    spec = get_gqa_dsa_attention_spec(cfg, backend)
    attn = build_module(spec, config=cfg, layer_number=1).cuda().bfloat16()
    print("built GQA-DSA attention:", type(attn).__name__, "core=", type(attn.core_attention).__name__)

    SQ, B = 128, 1
    hidden = torch.randn(SQ, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    from megatron.core.packed_seq_params import PackedSeqParams  # noqa
    out = attn(hidden, attention_mask=None)
    y = out[0] if isinstance(out, (tuple, list)) else out
    print("forward OK, out shape:", tuple(y.shape), "dtype:", y.dtype)
    y.float().sum().backward()
    print("backward OK. grad on hidden:", hidden.grad is not None)
    print("=== DSA GPU SMOKE PASSED ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("=== DSA GPU SMOKE FAILED:", type(e).__name__, str(e)[:200])
