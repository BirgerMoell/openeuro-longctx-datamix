"""Short-context DSA layer search: measure per-layer attention COMPRESSIBILITY on the real model
and emit a hybrid dense/sparse pattern. Layers whose attention is naturally concentrated (high
top-k mass) lose least from sparsification -> DSA ('S'); diffuse layers stay dense ('F').

Principled + cheap (GLM-5 searched the arrangement at 16K; it length-generalizes). One forward pass
with eager attention at a modest context; no per-pattern training needed.

Usage: python dsa_layer_search.py <hf_model> [--ctx 2048] [--sparse-frac 0.5] [--topk-frac 0.125]
"""
import sys, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SAMPLE = (
    "The theta scaling law for rotary position embeddings states that the critical rotary base "
    "doubles per context-length octave. Long-context retrieval at the far start of the window is "
    "an out-of-distribution problem in the high RoPE dimensions, not a data problem. "
) * 400  # repeated real text -> plenty of tokens; realistic-ish attention structure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--sparse-frac", type=float, default=0.5, help="fraction of layers to make DSA")
    ap.add_argument("--topk-frac", type=float, default=0.125, help="top fraction of keys = 'kept' (k=2048@16K)")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    model.eval()
    ids = tok(SAMPLE, return_tensors="pt").input_ids[:, : args.ctx]
    dev = next(model.parameters()).device
    nL = model.config.num_hidden_layers
    mass = torch.zeros(nL)

    with torch.no_grad():
        for s in range(args.samples):
            chunk = ids[:, s * 16 :].to(dev)[:, : args.ctx]
            if chunk.shape[1] < 64: break
            out = model(chunk, output_attentions=True, use_cache=False)
            for L, A in enumerate(out.attentions):        # A: [b, heads, q, k], causal
                a = A[0].float()                          # [heads, q, k]
                q = a.shape[1]
                # per query, mass captured by the top ceil(topk_frac * valid_keys)
                valid = torch.arange(1, q + 1, device=a.device)      # #valid keys per query pos
                k = torch.clamp((valid.float() * args.topk_frac).ceil().long(), min=1)  # [q]
                srt, _ = a.sort(dim=-1, descending=True)              # [heads, q, k]
                cum = srt.cumsum(dim=-1)
                idx = (k - 1).clamp(max=a.shape[-1] - 1)
                capt = cum.gather(-1, idx.view(1, q, 1).expand(a.shape[0], q, 1)).squeeze(-1)  # [heads,q]
                total = a.sum(dim=-1).clamp(min=1e-9)                 # ~1.0
                mass[L] += (capt / total).mean().item()
    mass /= max(args.samples, 1)

    order = mass.argsort(descending=True)                 # most-compressible first
    n_sparse = int(round(args.sparse_frac * nL))
    sparse_layers = set(order[:n_sparse].tolist())
    pattern = "".join("S" if i in sparse_layers else "F" for i in range(nL))
    print("per-layer top-{:.0%} attention mass (higher = more compressible):".format(args.topk_frac))
    for i in range(nL):
        print(f"  L{i:02d}: {mass[i]:.3f}  {'S(DSA)' if i in sparse_layers else 'F(dense)'}")
    print(f"\nRECOMMENDED PATTERN ({n_sparse}/{nL} sparse): {pattern}")


if __name__ == "__main__":
    main()
