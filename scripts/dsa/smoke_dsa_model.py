"""Full-model DSA training-step smoke test: build a tiny GPTModel whose layers use DSA sparse
attention, run forward(loss)+backward+optimizer for a few steps on the MI250X. TP=1, tiny.
Validates DSA end-to-end inside a real Megatron GPTModel (not just the attention module)."""
import os, sys, torch

def main():
    import torch.distributed as dist
    for k, v in dict(MASTER_ADDR="localhost", MASTER_PORT="29593", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0").items():
        os.environ.setdefault(k, v)
    TP = int(os.environ.get("DSA_TP", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend="nccl"); torch.cuda.set_device(local)
    from megatron.core import parallel_state as ps
    ps.initialize_model_parallel(tensor_model_parallel_size=TP)
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(123)  # required for TP>1 (model-parallel-rng)
    if dist.get_rank() == 0: print(f"TP={TP} world={dist.get_world_size()}")
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt import GPTModel
    sys.path.insert(0, os.path.dirname(__file__))
    from megatron_gqa_dsa import get_gqa_dsa_layer_spec
    # ROCm: replace the indexer Hadamard with a pure-torch FWHT
    import megatron.core.transformer.experimental_attention_variant.dsa as _dsa
    def _fwht(x):
        d = x.shape[-1]; shp = x.shape; y = x.reshape(-1, d).float(); h = 1
        while h < d:
            y = y.view(-1, d // (2 * h), 2, h)
            y = torch.stack([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], dim=2).reshape(-1, d); h *= 2
        return (y * (d ** -0.5)).reshape(shp).to(x.dtype)
    _dsa.rotate_activation = _fwht

    H, NH, NKV, HD, V, SQ, B = 512, 8, 2, 64, 4096, 128, 2
    cfg = TransformerConfig(
        num_layers=2, hidden_size=H, num_attention_heads=NH, num_query_groups=NKV,
        kv_channels=HD, use_cpu_initialization=True, bf16=True, add_bias_linear=False,
        qk_layernorm=True, normalization="RMSNorm", gated_linear_unit=True, ffn_hidden_size=1024,
        tensor_model_parallel_size=TP, sequence_parallel=(TP > 1),
        dsa_indexer_n_heads=2, dsa_indexer_head_dim=64, dsa_indexer_topk=32,
        dsa_indexer_loss_coeff=0.1, dsa_indexer_use_sparse_loss=False,
    )
    for k, v in dict(q_lora_rank=None, qk_pos_emb_head_dim=HD, rope_type="rope",
                     rotary_percent=1.0, rotary_base=32000000).items():
        setattr(cfg, k, v)
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    layer_spec = get_gqa_dsa_layer_spec(TESpecProvider(), qk_layernorm=True)
    model = GPTModel(config=cfg, transformer_layer_spec=layer_spec, vocab_size=V,
                     max_sequence_length=SQ, position_embedding_type="rope",
                     rotary_base=32000000, pre_process=True, post_process=True).cuda().bfloat16()
    print("built GPTModel with DSA layers; params(M):", sum(p.numel() for p in model.parameters())/1e6)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ids = torch.randint(0, V, (B, SQ), device="cuda")
    pos = torch.arange(SQ, device="cuda").unsqueeze(0).expand(B, -1)
    amask = torch.tril(torch.ones(SQ, SQ, device="cuda", dtype=torch.bool)).view(1, 1, SQ, SQ)
    losses = []
    for step in range(4):
        opt.zero_grad()
        out = model(ids, pos, attention_mask=~amask, labels=ids)
        loss = out.mean() if out.dim() > 0 else out
        loss.backward(); opt.step(); losses.append(loss.item())
        print(f"step {step}: loss {loss.item():.4f}")
    print("losses:", [round(l, 3) for l in losses])
    assert all(l == l for l in losses), "NaN loss"
    print("=== DSA FULL-MODEL TRAIN STEP PASSED ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("=== DSA MODEL SMOKE FAILED:", type(e).__name__, str(e)[:200])
