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

## DSA launchers

| script | purpose | status |
|---|---|---|
| `dsa_warmup_failclosed.sbatch` | Frozen-base, all-layer 8K indexer warm-up | Validated; job 20291047 completed 300 steps |
| `dsa_sparse_8k_correctness.sbatch` | One-step sparse update with selected-set KL and dual-gradient probes | Validated recipe; source of job 20336946 |
| `dsa_sparse_8k_sustained.sbatch` | Resumable 500-update sparse adaptation | Deferred quality gate; do not run before round trips |
| `dsa_sparse_8k_roundtrip.sbatch` | GPU/RCCL tests, sparse update, full save, fresh-process reload | **Passed; job 20927044, commit a8e3551** |
| `dsa_sparse_64k_cp2_roundtrip.sbatch` | Two-node CP2 round trip at 32K local tokens/rank | **Passed; job 20932303** |
| `dsa_sparse_512k_cp16_roundtrip.sbatch` | 16-node CP16 512K round trip | **Passed on real superlong-v2 data; job 20996514, commit 8e23ddf** |
| `dsa_sparse_512k_k2048_calibration.sbatch` | 16-node CP16 real-data adaptation, 128×16 block budget | **Mechanically passed across jobs 21265492/21284719; recall-quality gate failed** |
| `sparse_512k.sbatch` | Historical 512K proposal | **Archived and fail-closed** |

The round-trip scripts use the standalone `block_cp` overlay: global zig-zag reorder, differentiable
K/V gather, a 256-token current block plus one learned earlier block, global-position Triton
causality, selected-set KL, and dual-gradient probes. Every script starts a fresh Python process to
reload model+optimizer+RNG before the second update. There are no blind retries. This full-K/V
gather is capped at 512K; 1M–2M still requires selected-row exchange and streamed/recomputed state.
See [the canonical DSA overview](../../../docs/sparse_attention_dsa.md).

### Passing 512K superlong invocation

Job `20996514` used the immutable launcher with these overrides:

```text
DSA_OUT=/scratch/project_465002530/users/bmoell/longctx-extend/output_dsa_sparse512k_cp16_superlong_roundtrip
DATA_BLEND_FILE=/scratch/project_465002530/users/bmoell/superlong_data/mix/data_path.args
DATA_CACHE_PATH=/scratch/project_465002530/users/bmoell/longctx-extend/cache_dsa_sparse_524288_superlong
```

Preflight found 48 data pairs, sequence 524288, CP16/local 32768, TP8/128 ranks, all 36 layers
`S`, block size 256, one routed block, top-k 512, full/uniform recomputation, and dataset surplus
0.5. Iterations 301/302 had finite LM losses `2.093314`/`2.125192`, finite indexer losses
`0.643614`/`0.662684`, nonzero main/indexer probes, and zero NaN/skipped updates. A fresh process
loaded iteration 301 with optimizer, RNG, and scheduler state before update 302. Both checkpoints
contain `.metadata`, `common.pt`, and 256 distributed shards; tracker is 302. Runtime was 608
seconds on 128 GPU slots (21.62 GPU-hours), and the launcher emitted its explicit PASS signal.

### Completed 512K k=2048 calibration

Use immutable source commit `84feff7` with:

```text
MEGATRON_ROOT=/scratch/project_465002530/users/bmoell/deps/NVIDIA-Megatron-LM-b359462c
DSA_OUT=/scratch/project_465002530/users/bmoell/longctx-extend/output_dsa_sparse512k_k2048_calibration
DATA_BLEND_FILE=/scratch/project_465002530/users/bmoell/superlong_data/mix/data_path.args
```

Preflight job `21265221` verified Megatron revision
`b359462c12858cedd2238a22eca0dca7aa6b8872`, the overlay, checkpoint, 48-prefix data blend,
CP/block geometry, and dataset surplus. Job `21265492` completed updates 301–336 and checkpoint
336, then suffered a transient RCCL timeout in the first phase-2 CP all-gather. Recovery job
`21284719` loaded 336, completed updates 337–372 and checkpoint 372, then loaded 372 in a third
process and completed update/checkpoint 373. Checkpoints 336/372/373 each have `.metadata`,
`common.pt`, and 256 distributed shards; tracker is 373. All updates were finite with zero
skipped/NaN iterations, and the final launcher signal was `DSA 512K k2048 calibration PASS`.

Steady updates took about 81.7 seconds at roughly 50.1 tokens/s/GPU, with peak memory around
17.7 GiB allocated and 25.7 GiB reserved per rank. Total scheduler exposure across first launch
`21050508`, preflight `21265221`, phase-1/transient-failure job `21265492`, and recovery `21284719`
was about 315.0 GPU-hours.

Do not treat the PASS signal as a model-quality result. Mean sampled top-2,048 attention-mass
recall declined from 0.103 to 0.092; 25/36 layers worsened and phase-2 position quartiles were
[0.206, 0.069, 0.053, 0.041]. Do not submit a longer run from iteration 373. The next bounded
experiment must add explicit local/sink coverage, per-GQA routing, and longer dense-teacher indexer
adaptation before repeating loss/logit and retrieval gates.
