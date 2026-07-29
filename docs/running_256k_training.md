# Running 256K Context Extension — Runbook

How to extend an OELLM base to **256K** context via staged native ABF (RoPE-θ scaling), on LUMI with
Megatron. This is the exact pipeline that produced
[`oellm-9b-256k-theta64m-prelude`](https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude).

> Method background: `docs/handoff_longcontext_for_jouni.md` (θ mental model, the θ-law) and
> `docs/rope_theta_vs_qwen35.md` (how to choose θ).

---

## 1. The staged chain (you extend, you don't jump)

Context is extended in stages, raising `--seq-length` and `--rotary-base` (θ) at each, continuing
from the previous checkpoint:

```
base(4K, θ=100k) → 16K(θ=500k) → 32K(θ=1M) → 64K(θ=2M) → 128K(θ=32M) → 256K(θ=64M)
```

- **128K stages:** `scripts/dsa/lumi/prelude_full.sbatch` (1T base) or `real_v3_128k.sbatch` (0.6T base)
- **256K stage:** `scripts/dsa/lumi/real_256k_v2.sbatch`  ← this runbook

Each stage is ~1–2B tokens (~a short phase). θ jumps sharply at 128K (the θ-law: doubles per octave).

---

## 2. Files

| file | role |
|---|---|
| [`scripts/dsa/lumi/real_256k_v2.sbatch`](../scripts/dsa/lumi/real_256k_v2.sbatch) | **256K launcher** (128K ckpt → 256K, θ=64M, CP=16) |
| [`scripts/dsa/lumi/prelude_full.sbatch`](../scripts/dsa/lumi/prelude_full.sbatch) | 128K prerequisite (the stages you extend from) |
| [`scripts/dsa/lumi/README.md`](../scripts/dsa/lumi/README.md) | script → HF-model map + per-stage LR/anneal |
| [`scripts/eval_base_lm_niah.py`](../scripts/eval_base_lm_niah.py) | NIAH eval (depth × context × language) |
| [`scripts/eval_perplexity.py`](../scripts/eval_perplexity.py) | short-context regression / perplexity |

---

## 3. Prerequisites (LUMI)

- **A 128K checkpoint** to extend from (Megatron torch_dist), e.g.
  `/scratch/project_465002530/users/bmoell/longctx-extend/output_prelude_2/ckpt_131072`.
  *(Extending the newer prelude checkpoint? Point `LOAD` at your new 128K result.)*
- **256K data blend** on stable `/scratch` (NOT `/flash`): `blend_256k.txt` =
  60% genuine-256K concatenated docs (arXiv/books/code/RFC) + 40% multilingual (Jouni mix).
- Container, Megatron, tokenizer (paths are set in the sbatch — verify they exist).

---

## 4. Config knobs (what to edit in `real_256k_v2.sbatch`)

The single stage line at the bottom:
```bash
run_stage 262144 64000000 16 1000000000 64 "$LOAD" "$OUT/ckpt_262144"
#         └seq─┘ └──θ───┘ CP └─tokens──┘ gbs
```
| arg | value | change it if… |
|---|---|---|
| seq | 262144 (256K) | fixed for 256K |
| **θ (rotary-base)** | **64000000** | **lower θ experiment** — e.g. 32000000 (see `rope_theta_vs_qwen35.md`; our floor is ~32M, 16M@128K gave 90%) |
| CP | 16 | context-parallel shards the 256K sequence; a2a impossible (1 KV head/rank) → uses p2p |
| tokens | 1e9 (~59 iters) | budget; ~0.3e9 for a quick probe, more for a clean run |
| gbs | 64 | global batch |

Other key args already set: `--lr 8e-6 --min-lr 8e-7 --lr-decay-style cosine` (annealing),
`--recompute-granularity selective`, `--ckpt-format torch_dist`, `--finetune`,
16 nodes × 8 GPUs, 5× in-job retry loop (handles the intermittent 256K NCCL first-step hang).

**Also update per run:** `LOAD` (your 128K ckpt), `DATA`/`blend_256k.txt`, `OUT`, `MEG`, `CONTAINER`.

---

## 5. Run

```bash
cd /scratch/project_465002530/users/bmoell/longctx-extend
sbatch real_256k_v2.sbatch
# θ-configurable variant (θ/tag as env, no file edit):
sbatch --export=ALL,THETA=32000000,TAG=32M theta_sweep_256k.sbatch
```
Watch: `logs/<jobid>.out` for `256K STAGE … ATTEMPT=n`, `iteration N/…`, `256K stage OK`,
final `ckpt_262144`. Expect **~16 min/iter** at 256K → ~16h for 1B tokens (16 nodes).

**Cost:** ~2,000 GPU-h for a 1B-token run (16 nodes × 8 GPU × ~16h). Compute is not the constraint.

---

## 6. Convert + evaluate

1. **Convert** Megatron→HF (Megatron-Bridge; PYTHONPATH must put `python-packages` first).
2. **NIAH:** `scripts/eval_base_lm_niah.py --model <hf> --context-lengths 4096 65536 131072 262144
   --depths 0.0 0.25 0.5 0.75 1.0 --languages en de fr … --trials 6`.
   The 256K forward needs **multi-GPU** (`device_map="auto"`, full 8-GCD node — one GCD OOMs at 256K).
   Watch **depth-0** (the θ-sensitive metric).
3. **Perplexity / short-context:** `scripts/eval_perplexity.py --model <hf> --seq-lengths 512 2048
   8192 131072 262144` — catches θ-over-scaling harm that NIAH misses.

---

## 7. Known pitfalls (learned the hard way)

- **`/flash` is a flaky burst buffer** — files intermittently fail to read mid-job. Keep data + the
  extend dir on **stable `/scratch`**.
- **256K NCCL first-step hang** (CP>1) — intermittent, node-dependent. The retry loop + 20-min
  fast-fail handles it (broke through on attempt 3–5 historically).
- **256K eval OOM on 1 GCD** — use a full node + `device_map="auto"`.
- **`logits_to_keep` in the eval is essential** at 256K (else full-seq logits over 262k vocab OOM).
- **θ is architecture-specific** — don't copy Qwen's 10M; our full-rotary/head_dim-128 floor is
  higher (see `rope_theta_vs_qwen35.md`).

---

## 8. Reference result
`oellm-9b-256k-theta64m-prelude` @256K: ~90% (7-lang partial), **depth-0 = 93%** (θ=64M confirms
far-position retrieval at 256K), 128K retention 98%. Mild lost-in-the-middle from the 1T base.
