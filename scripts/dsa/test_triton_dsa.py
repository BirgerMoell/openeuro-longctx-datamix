"""Correctness + speed of the Triton sparse-attn forward vs Megatron's unfused reference."""
import os, sys, time, torch

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from triton_dsa import sparse_attn_forward
    from megatron.core.transformer.experimental_attention_variant.dsa import unfused_dsa_fn
    torch.manual_seed(0); dev = "cuda"
    for (sq, np, hn, k) in [(4096, 4, 128, 512), (16384, 4, 128, 2048)]:
        b, skv = 1, sq
        q = torch.randn(sq, b, np, hn, device=dev, dtype=torch.bfloat16)
        kx = torch.randn(skv, b, np, hn, device=dev, dtype=torch.bfloat16)
        v = torch.randn(skv, b, np, hn, device=dev, dtype=torch.bfloat16)
        scale = hn ** -0.5
        # k DISTINCT causal keys per query (queries >= k-1 are fully distinct -> compare only those,
        # sidestepping the early-query dedup, which is a separate refinement).
        idx = torch.zeros(b, sq, k, dtype=torch.long, device=dev)
        for i in range(sq):
            n = min(i + 1, k)
            perm = torch.randperm(i + 1, device=dev)[:n]
            idx[0, i, :n] = perm
            if n < k: idx[0, i, n:] = perm[torch.randint(0, n, (k - n,), device=dev)]
        o_ref = unfused_dsa_fn(q, kx, v, idx, scale)
        o_tri, _ = sparse_attn_forward(q, kx, v, idx, scale)
        oref = o_ref.float()[k:]; otri = o_tri.float()[k:]        # dedup-free queries only
        err = (oref - otri).abs().max().item()
        rel = err / (oref.abs().max().item() + 1e-9)
        # speed
        for _ in range(2): unfused_dsa_fn(q, kx, v, idx, scale); sparse_attn_forward(q, kx, v, idx, scale)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(3): unfused_dsa_fn(q, kx, v, idx, scale)
        torch.cuda.synchronize(); t_ref = (time.time() - t) / 3
        t = time.time()
        for _ in range(3): sparse_attn_forward(q, kx, v, idx, scale)
        torch.cuda.synchronize(); t_tri = (time.time() - t) / 3
        print(f"sq={sq} k={k}: max_err={err:.3e} rel={rel:.2e} | ref={t_ref*1e3:.0f}ms triton={t_tri*1e3:.0f}ms "
              f"speedup={t_ref/max(t_tri,1e-6):.1f}x")
    print("=== TRITON DSA TEST DONE ===")

if __name__ == "__main__":
    import torch.distributed as dist
    for k, v in dict(MASTER_ADDR="localhost", MASTER_PORT="29605", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0").items():
        os.environ.setdefault(k, v)
    dist.init_process_group(backend="nccl"); torch.cuda.set_device(0)
    try: main()
    except Exception as e:
        import traceback; traceback.print_exc(); print("=== FAILED:", type(e).__name__, str(e)[:300])
