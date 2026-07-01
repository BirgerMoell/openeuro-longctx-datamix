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
    # ROCm lacks fast_hadamard_transform; replace the indexer's Hadamard rotation with a
    # pure-torch normalized FWHT (faithful, dim must be power of 2 — index_head_dim=64 ✓).
    import megatron.core.transformer.experimental_attention_variant.dsa as _dsa
    def _fwht_rotate(x):
        d = x.shape[-1]; shp = x.shape
        y = x.reshape(-1, d).float(); h = 1
        while h < d:
            y = y.view(-1, d // (2 * h), 2, h)
            top = y[:, :, 0] + y[:, :, 1]; bot = y[:, :, 0] - y[:, :, 1]
            y = torch.stack([top, bot], dim=2).reshape(-1, d); h *= 2
        return (y * (d ** -0.5)).reshape(shp).to(x.dtype)
    _dsa.rotate_activation = _fwht_rotate

    H, NH, NKV, HD = 512, 8, 2, 64
    cfg = TransformerConfig(
        num_layers=1, hidden_size=H, num_attention_heads=NH, num_query_groups=NKV,
        kv_channels=HD, use_cpu_initialization=True, bf16=True,
        add_bias_linear=False, qk_layernorm=True, normalization="RMSNorm",
        # DSA config (these ARE TransformerConfig fields):
        dsa_indexer_n_heads=2, dsa_indexer_head_dim=64, dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=0.1, dsa_indexer_use_sparse_loss=False,
    )
    # The DSA indexer reads MLA/rope config fields base TransformerConfig lacks; bridge them.
    for k, v in dict(q_lora_rank=None, qk_pos_emb_head_dim=HD, rope_type="rope",
                     rotary_percent=1.0, rotary_base=32000000).items():
        setattr(cfg, k, v)
    # backend: TESpecProvider supplies the linear/layernorm builders (has .linear);
    # DSAttention itself is the local sparse module. (LocalSpecProvider lacks .linear.)
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    backend = TESpecProvider()
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
