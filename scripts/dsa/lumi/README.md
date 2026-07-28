# Long-context extension launch scripts (LUMI / Megatron)

The exact sbatch files that produced the published context-extended models. All use **Megatron's
cosine LR annealing** (`OptimizerParamScheduler`), per stage: `--lr-decay-style cosine`,
`--lr-warmup-iters ≈ iters/20` (~5% warmup), `--lr-decay-iters = iters` (cosine decay to `--min-lr`).

| script | model | stages (seq @ θ) | LR anneal | tokens/stage |
|---|---|---|---|---|
| `real_v3_128k.sbatch` | [oellm-9b-128k-theta32m-v3](https://huggingface.co/birgermoell/oellm-9b-128k-theta32m-v3) (0.6T base) | 16K→32K→64K→128K @ θ=32M | **1e-5 → 1e-6 cosine** | ~1–2B |
| `prelude_full.sbatch` | [oellm-9b-128k-theta32m-prelude](https://huggingface.co/openeurollm/oellm-9b-128k-theta32m-prelude) (1T base) | 16K→…→128K @ θ=32M | **1e-5 → 1e-6 cosine** | ~1–2B |
| `real_256k_v2.sbatch` | [oellm-9b-256k-theta64m-prelude](https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude) | 256K @ θ=64M (from 128K) | **8e-6 → 8e-7 cosine** | ~1B |
| `real_128k_2x.sbatch` | 128K 2×-budget diagnostic (base-stickiness test) | 128K @ θ=32M | 1e-5 → 1e-6 cosine | ~4B |
| `real_512k.sbatch` | 512K (dense, staged from 256K) | 512K @ θ=128M | 8e-6 → 8e-7 cosine | ~1B |

**Annealing note (re: the LR-decay discussion):** these use a *gentle finetune-style* decay (10× over
a short stage, low peak LR), not a high-peak pretraining anneal. Whether the base checkpoints were
themselves annealed, and whether a higher-peak / longer extension anneal helps (esp. the 1T-base
lost-in-the-middle), is **untested** — candidate sweep. Method details: `docs/dsa_training_recipe.md`,
`docs/sparse_attention_dsa.md`.
