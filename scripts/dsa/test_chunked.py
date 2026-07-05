"""Validate chunked top-k == full top-k, and check memory feasibility."""
import os, sys, torch
def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chunked_indexer import chunked_topk, index_scores_block
    dev="cuda"; torch.manual_seed(0)
    for (sq, ih, hd, topk, blk) in [(4096,4,128,512,1024),(16384,4,128,2048,4096)]:
        b, sk = 1, sq
        q=torch.randn(sq,b,ih,hd,device=dev,dtype=torch.bfloat16)
        w=torch.randn(sq,b,ih,device=dev,dtype=torch.bfloat16)
        k=torch.randn(sk,b,hd,device=dev,dtype=torch.bfloat16)
        # causal additive mask [b,sq,sk]
        mask=torch.where(torch.arange(sk,device=dev)[None,None,:]<=torch.arange(sq,device=dev)[None,:,None],0.0,float('-inf')).expand(b,sq,sk)
        # full reference
        full = index_scores_block(q,w,k) + mask                # [b,sq,sk]
        idx_full = full.topk(min(topk,sk),dim=-1)[1]
        idx_chunk = chunked_topk(q,w,k,topk,mask=mask,block=blk)
        # compare as SETS per query (top-k order may differ on ties)
        eq = (idx_full.sort(-1)[0]==idx_chunk.sort(-1)[0]).float().mean().item()
        print(f"sq={sq} topk={topk} blk={blk}: topk-set match={eq:.4f}")
    print("=== CHUNKED TOPK TEST DONE ===")
if __name__=="__main__":
    import torch.distributed as dist
    for k,v in dict(MASTER_ADDR="localhost",MASTER_PORT="29611",RANK="0",WORLD_SIZE="1",LOCAL_RANK="0").items(): os.environ.setdefault(k,v)
    dist.init_process_group(backend="nccl"); torch.cuda.set_device(0)
    try: main()
    except Exception as e:
        import traceback; traceback.print_exc(); print("FAILED",e)
