"""Diagnose the recall ceiling: how much do the 32 heads DISAGREE on top-k? If head-mean top-k
poorly represents individual heads, my recall metric has a low ceiling regardless of indexer quality.
Measures, on the real 256K model at 16K: per-head top-2048 vs head-mean top-2048 agreement (the
theoretical max my metric could reach), and pairwise head top-k overlap (attention diffuseness)."""
import os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def main():
    m = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="eager"); m.eval()
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    txt = ("The theta scaling law for rotary embeddings and long-context retrieval. "*2000)
    ids = tok(txt, return_tensors="pt").input_ids[:, :4096].to(next(m.parameters()).device)
    k = 2048
    with torch.no_grad():
        out = m(ids, output_attentions=True, use_cache=False)
    # sample a few layers (early/mid/late) + 64 query rows
    L = len(out.attentions); qi = torch.linspace(min(k,3900), ids.shape[1]-1, 64, device=ids.device).long()
    for li in [0, L//2, L-1]:
        A = out.attentions[li][0]                          # [heads, q, kk]
        a = A[:, qi]                                       # [heads, 64, sk]
        mean_tk = a.mean(0).topk(k, -1).indices            # [64, k] head-mean top-k
        # per-head recall vs head-mean (ceiling of my metric)
        head_tk = a.topk(k, -1).indices                    # [heads,64,k]
        rec_vs_mean = (head_tk.unsqueeze(-1)==mean_tk.unsqueeze(0).unsqueeze(-2)).any(-1).float().mean().item()
        # pairwise head agreement (diffuseness): head0 vs head1 top-k overlap
        h0,h1 = head_tk[0], head_tk[min(1,head_tk.shape[0]-1)]
        pair = (h0.unsqueeze(-1)==h1.unsqueeze(-2)).any(-1).float().mean().item()
        # how concentrated is attention? mass in top-2048
        mass = a.mean(0).topk(k,-1).values.sum(-1).mean().item()
        print(f"layer {li}: per-head-vs-mean recall={rec_vs_mean:.3f} | head0-head1 overlap={pair:.3f} | top{k} mass={mass:.3f}")
    print("=== CEILING DIAG DONE ===")
if __name__=="__main__":
    try: main()
    except Exception as e:
        import traceback; traceback.print_exc(); print("FAILED",e)
