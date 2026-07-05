"""Integration smoke: full DSA GPTModel with the SPARSE patches (Triton attn + chunked indexer +
KL off), train a few steps. Validates the integrated sparse pipeline end-to-end."""
import os, sys, torch
def main():
    import torch.distributed as dist
    for k,v in dict(MASTER_ADDR="localhost",MASTER_PORT="29613",RANK="0",WORLD_SIZE="1",LOCAL_RANK="0").items(): os.environ.setdefault(k,v)
    dist.init_process_group(backend="nccl"); torch.cuda.set_device(0)
    from megatron.core import parallel_state as ps; ps.initialize_model_parallel(1)
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed; model_parallel_cuda_manual_seed(123)
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt import GPTModel
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import megatron.core.transformer.experimental_attention_variant.dsa as _dsa
    def _fwht(x):
        d=x.shape[-1];shp=x.shape;y=x.reshape(-1,d).float();h=1
        while h<d:
            y=y.view(-1,d//(2*h),2,h);y=torch.stack([y[:,:,0]+y[:,:,1],y[:,:,0]-y[:,:,1]],2).reshape(-1,d);h*=2
        return (y*(d**-0.5)).reshape(shp).to(x.dtype)
    _dsa.rotate_activation=_fwht
    from dsa_patches import apply_sparse_dsa_patches
    apply_sparse_dsa_patches()                          # <-- the 3 sparse monkey-patches
    from megatron_gqa_dsa import get_gqa_dsa_block_spec
    H,NH,NKV,HD,V,SQ,B=512,8,2,64,4096,8192,1
    PAT="FSFS"
    cfg=TransformerConfig(num_layers=len(PAT),hidden_size=H,num_attention_heads=NH,num_query_groups=NKV,kv_channels=HD,
        use_cpu_initialization=True,bf16=True,add_bias_linear=False,qk_layernorm=True,normalization="RMSNorm",
        gated_linear_unit=True,ffn_hidden_size=1024,dsa_indexer_n_heads=2,dsa_indexer_head_dim=64,dsa_indexer_topk=512,
        dsa_indexer_loss_coeff=0.0,dsa_indexer_use_sparse_loss=False)
    for k,v in dict(q_lora_rank=None,qk_pos_emb_head_dim=HD,rope_type="rope",rotary_percent=1.0,rotary_base=64000000).items(): setattr(cfg,k,v)
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    spec=get_gqa_dsa_block_spec(TESpecProvider(),PAT,qk_layernorm=True)
    model=GPTModel(config=cfg,transformer_layer_spec=spec,vocab_size=V,max_sequence_length=SQ,
        position_embedding_type="rope",rotary_base=64000000,pre_process=True,post_process=True).cuda().bfloat16()
    print("built sparse-DSA model at SQ=",SQ)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    ids=torch.randint(0,V,(B,SQ),device="cuda"); pos=torch.arange(SQ,device="cuda").unsqueeze(0)
    am=torch.tril(torch.ones(SQ,SQ,device="cuda",dtype=torch.bool)).view(1,1,SQ,SQ)
    losses=[]
    for s in range(3):
        opt.zero_grad(); o=model(ids,pos,attention_mask=~am,labels=ids); loss=o.mean(); loss.backward(); opt.step(); losses.append(round(loss.item(),3))
    print("losses:",losses); assert all(l==l for l in losses)
    print("=== SPARSE DSA SMOKE PASSED ===")
if __name__=="__main__":
    try: main()
    except Exception as e:
        import traceback; traceback.print_exc(); print("=== FAILED:",type(e).__name__,str(e)[:300])
