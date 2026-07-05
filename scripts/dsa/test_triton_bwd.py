"""Validate Triton DSA backward: dq/dk/dv vs autograd through the reference (dedup-free region)."""
import os, sys, torch

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from triton_dsa import triton_dsa_attn
    from megatron.core.transformer.experimental_attention_variant.dsa import unfused_dsa_fn
    torch.manual_seed(0); dev = "cuda"
    sq, b, np, hn, k = 4096, 1, 4, 128, 512
    scale = hn ** -0.5
    idx = torch.zeros(b, sq, k, dtype=torch.long, device=dev)
    for i in range(sq):
        n = min(i + 1, k); perm = torch.randperm(i + 1, device=dev)[:n]
        idx[0, i, :n] = perm
        if n < k: idx[0, i, n:] = perm[torch.randint(0, n, (k - n,), device=dev)]

    def mk():
        return [torch.randn(sq, b, np, hn, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    torch.manual_seed(1); qr, kr, vr = mk()
    torch.manual_seed(1); qt, kt, vt = mk()

    g = torch.randn(sq, b, np * hn, device=dev, dtype=torch.bfloat16)
    o_ref = unfused_dsa_fn(qr, kr, vr, idx, scale); (o_ref * g).sum().backward()
    o_tri = triton_dsa_attn(qt, kt, vt, idx, scale);  (o_tri * g).sum().backward()

    def cmp(name, a, bb, sl):
        a = a.float()[sl]; bb = bb.float()[sl]
        e = (a - bb).abs().max().item(); r = e / (a.abs().max().item() + 1e-9)
        print(f"  {name}: max_err={e:.3e} rel={r:.2e}")
    print("dedup-free region (queries/keys >= k):")
    cmp("dq", qr.grad, qt.grad, slice(k, None))
    cmp("dk", kr.grad, kt.grad, slice(k, None))
    cmp("dv", vr.grad, vt.grad, slice(k, None))
    print("=== TRITON BWD TEST DONE ===")

if __name__ == "__main__":
    import torch.distributed as dist
    for kk, vv in dict(MASTER_ADDR="localhost", MASTER_PORT="29609", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0").items():
        os.environ.setdefault(kk, vv)
    dist.init_process_group(backend="nccl"); torch.cuda.set_device(0)
    try: main()
    except Exception as e:
        import traceback; traceback.print_exc(); print("=== FAILED:", type(e).__name__, str(e)[:300])
