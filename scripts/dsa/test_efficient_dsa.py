"""Compare efficient_dsa_fn vs Megatron's reference unfused_dsa_fn: correctness + speed on the MI250X."""
import os, sys, time, torch

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from efficient_dsa import efficient_dsa_fn
    from megatron.core.transformer.experimental_attention_variant.dsa import unfused_dsa_fn
    torch.manual_seed(0); dev = "cuda"

    for (sq, np, hn, k) in [(2048, 4, 128, 512), (8192, 4, 128, 1024)]:
        b, skv, hnv = 1, sq, hn
        q = torch.randn(sq, b, np, hn, device=dev, dtype=torch.bfloat16)
        kx = torch.randn(skv, b, np, hn, device=dev, dtype=torch.bfloat16)
        v = torch.randn(skv, b, np, hnv, device=dev, dtype=torch.bfloat16)
        scale = hn ** -0.5
        # topk_indices [b,sq,k]: for each query pick k causal positions (<=q), sorted-ish
        idx = torch.zeros(b, sq, k, dtype=torch.long, device=dev)
        for i in range(sq):
            n = min(i + 1, k)
            perm = torch.randperm(i + 1, device=dev)[:n]
            idx[0, i, :n] = perm
            if n < k: idx[0, i, n:] = perm[0] if n > 0 else 0   # pad with a valid (dup) index
        for _ in range(2):  # warmup (compile/autotune)
            unfused_dsa_fn(q, kx, v, idx, scale); efficient_dsa_fn(q, kx, v, idx, scale)
        torch.cuda.synchronize(); t = time.time()
        o_ref = unfused_dsa_fn(q, kx, v, idx, scale); torch.cuda.synchronize()
        t_ref = time.time() - t
        t = time.time()
        o_eff = efficient_dsa_fn(q, kx, v, idx, scale); torch.cuda.synchronize()
        t_eff = time.time() - t
        err = (o_ref.float() - o_eff.float()).abs().max().item()
        rel = err / (o_ref.float().abs().max().item() + 1e-9)
        print(f"sq={sq} k={k}: max_err={err:.4e} rel={rel:.2e} | ref={t_ref*1e3:.0f}ms eff={t_eff*1e3:.0f}ms "
              f"speedup={t_ref/max(t_eff,1e-6):.1f}x")
    print("=== EFFICIENT DSA TEST DONE ===")

if __name__ == "__main__":
    import torch.distributed as dist
    for k, v in dict(MASTER_ADDR="localhost", MASTER_PORT="29599", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0").items():
        os.environ.setdefault(k, v)
    dist.init_process_group(backend="nccl"); torch.cuda.set_device(0)
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc(); print("=== FAILED:", type(e).__name__, str(e)[:200])
