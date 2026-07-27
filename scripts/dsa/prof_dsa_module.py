"""Profile the real DSA attention module fwd+bwd with torch.profiler -> printed op table."""
import os, sys, torch

def main():
    import torch.distributed as dist
    for k,v in dict(MASTER_ADDR="localhost",MASTER_PORT="29603",RANK="0",WORLD_SIZE="1",LOCAL_RANK="0").items(): os.environ.setdefault(k,v)
    dist.init_process_group(backend="nccl"); torch.cuda.set_device(0)
    from megatron.core import parallel_state as ps; ps.initialize_model_parallel(1)
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed; model_parallel_cuda_manual_seed(123)
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.spec_utils import build_module
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from megatron_gqa_dsa import get_gqa_dsa_attention_spec
    import megatron.core.transformer.experimental_attention_variant.dsa as _dsa
    def _fwht(x):
        d=x.shape[-1];shp=x.shape;y=x.reshape(-1,d).float();h=1
        while h<d:
            y=y.view(-1,d//(2*h),2,h);y=torch.stack([y[:,:,0]+y[:,:,1],y[:,:,0]-y[:,:,1]],2).reshape(-1,d);h*=2
        return (y*(d**-0.5)).reshape(shp).to(x.dtype)
    _dsa.rotate_activation=_fwht
    H,NH,NKV,HD=4096,32,8,128
    cfg=TransformerConfig(num_layers=1,hidden_size=H,num_attention_heads=NH,num_query_groups=NKV,kv_channels=HD,
        use_cpu_initialization=True,bf16=True,add_bias_linear=False,qk_layernorm=True,normalization="RMSNorm",
        dsa_indexer_n_heads=4,dsa_indexer_head_dim=128,dsa_indexer_topk=2048,dsa_indexer_loss_coeff=0.1,dsa_indexer_use_sparse_loss=False)
    for k,v in dict(q_lora_rank=None,qk_pos_emb_head_dim=HD,rope_type="rope",rotary_percent=1.0,rotary_base=64000000).items(): setattr(cfg,k,v)
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    attn=build_module(get_gqa_dsa_attention_spec(cfg,TESpecProvider()),config=cfg,layer_number=1).cuda().bfloat16()
    attn.train()
    SQ,B=8192,1
    hid=torch.randn(SQ,B,H,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    def step():
        o=attn(hid,attention_mask=None); y=o[0] if isinstance(o,tuple) else o; y.float().sum().backward()
    for _ in range(3): step()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        for _ in range(3): step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18))
    print("=== PROFILE DONE ===")

if __name__=="__main__":
    try: main()
    except Exception as e:
        import traceback; traceback.print_exc(); print("FAILED",e)
