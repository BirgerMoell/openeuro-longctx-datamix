# DeepSeek Sparse Attention overlay

This directory contains the OpenEuroLLM-owned DSA implementation, correctness tests, and LUMI
launchers. It integrates with a pinned external Megatron checkout at runtime; it does not require a
private Megatron fork.

Read [the canonical sparse-attention overview](../../docs/sparse_attention_dsa.md) before launching
training. It records the architecture, LUMI evidence, known limitations, and gated route to
512K–2M.

## Current state

Validated:

- 300-step all-layer indexer warm-up at 8K;
- exact causal blocked top-k versus a dense reference;
- selected-set KL with a native-GQA teacher;
- ROCm Triton native-GQA forward/backward versus dense attention;
- non-interleaved indexer RoPE;
- one full sparse update with finite LM/indexer losses; and
- nonzero gradients in both the main model and indexer;
- Megatron CP zig-zag reorder and summed collective backward on two CPU ranks; and
- causal hierarchical block routing without future-query leakage; and
- an 8K GPU/RCCL save/reload round trip through complete iterations 301 and 302 (job 20927044);
- a 64K/CP2 round trip at the final 32K local shape (job 20932303); and
- a 512K/CP16 round trip on the checksum-verified 48-prefix superlong-v2 blend, including a
  fresh-process full-state reload (job 20996514); and
- a 73-update 512K/CP16 k=2048 calibration through checkpoints 336/372/373 and two full-state
  reload boundaries (jobs 21265492 and 21284719).

The first k=2048 calibration launch, job `21050508`, failed before preflight because its shared
external Megatron checkout had been removed. It performed no training and created no output. The
replacement prerequisite is an immutable, project-owned checkout matching `MEGATRON_REVISION`.
The checkout is now staged at
`/scratch/project_465002530/users/bmoell/deps/NVIDIA-Megatron-LM-b359462c`; preflight job `21265221`
passed. Replacement job `21265492` completed phase 1/checkpoint 336, then hit a transient CP RCCL
timeout before update 337. Recovery job `21284719` loaded 336 and completed the remaining updates,
checkpoint 372, a fresh reload, update/checkpoint 373, and the explicit calibration PASS.

This is a mechanical pipeline result, not a quality pass. Mean top-2048 dense-attention-mass recall
declined from 0.103 to 0.092, with 25/36 layers worsening and strong position bias. Do not resume
iteration 373 for a longer adaptation; repair local/sink coverage and per-GQA routing first.

Not yet implemented or validated:

- quality-successful sparse adaptation (the bounded 73-update run failed its recall gate);
- explicit 512–1,024-token local/sink coverage and per-GQA-group routing;
- selected-row rather than replicated-global K/V transport for 1M–2M;
- sparse prefill/decode and KV-cache integration; and
- model export and serving.

## Production-path files

- `gpt_builders.py` — import shim expected by Megatron's `pretrain_gpt.py`
- `MEGATRON_REVISION` — exact external Megatron commit used by the validated LUMI gates
- `gpt_builders_dsa.py` — config bridge, fail-closed checkpoint load, freeze logic, gradient probes
- `megatron_gqa_dsa.py` — GQA-aware DSA module and layer specifications
- `chunked_indexer.py` — exact causal query/key-blocked selection and retention guard
- `hierarchical_indexer.py` — subquadratic causal block router for the 512K bridge
- `cp_utils.py` — differentiable Megatron CP gather and zig-zag/global reorder
- `dsa_sparse_loss.py` — selected-set KL with detached native-GQA teacher
- `dsa_patches.py` — sparse path, unsupported-mode guards, and recall logging
- `triton_dsa.py` — ROCm Triton sparse attention forward/backward
- `test_dsa_correctness.py` — dense-reference correctness suite
- `test_cp_distributed.py` — multi-rank collective/autograd suite
- `test_hierarchical_indexer.py` — deployed block-router geometry/causality suite
- `lumi/dsa_warmup_failclosed.sbatch` — frozen-indexer warm-up launcher
- `lumi/dsa_sparse_8k_correctness.sbatch` — one-step sparse integration gate
- `lumi/dsa_sparse_8k_roundtrip.sbatch` — GPU/RCCL and checkpoint round trip
- `lumi/dsa_sparse_64k_cp2_roundtrip.sbatch` — two-node final-local-shape gate
- `lumi/dsa_sparse_512k_cp16_roundtrip.sbatch` — gated 16-node 512K round trip
- `lumi/dsa_sparse_512k_k2048_calibration.sbatch` — 73-update, k=2048 real-data calibration
- `lumi/dsa_sparse_calibration.sh` — fail-closed multi-process calibration driver
- `../validate_megatron_indexed_mix.py` — indexed-pair and real GPT blend validator

Other modules in this directory are earlier prototypes, diagnostics, or layer-search experiments.
They are useful for research history but are not the current production path.

## Correctness tests

Local CPU test:

```bash
python3 scripts/dsa/test_dsa_correctness.py --cpu-only
```

This tests exact selection and selected-set KL. It reports a clear RoPE skip when Megatron is not
installed.

LUMI CPU gate with Megatron required:

```bash
PYTHONPATH="scripts/dsa:$MEGATRON_ROOT:$PYTHONPATH" \
  python3 scripts/dsa/test_dsa_correctness.py --cpu-only --require-megatron-rope
```

LUMI GPU gate:

```bash
PYTHONPATH="scripts/dsa:$MEGATRON_ROOT:$PYTHONPATH" \
  python3 scripts/dsa/test_dsa_correctness.py
```

The GPU gate additionally compares native-GQA Triton forward/backward, including global CP query
positions, with a dense reference. The local collective test is:

```bash
python3 scripts/dsa/test_cp_distributed.py --backend gloo --spawn-procs 2
```

## Runtime integration

Place `scripts/dsa` before Megatron on `PYTHONPATH`. Megatron imports `gpt_builders` by module
name, so the shim selects `gpt_builders_dsa.gpt_builder`. The builder then installs sparse patches
only when `DSA_SPARSE_RUN=1`.

Sparse adaptation fails closed unless:

- `DSA_SPARSE=1`;
- `DSA_FREEZE_MODEL=0`;
- the selected-set KL coefficient is positive;
- every layer is `S` (unless a bounded diagnostic explicitly sets
  `DSA_ALLOW_DENSE_LAYERS=1`);
- the indexer uses non-interleaved RoPE; and
- `block_cp` uses full/uniform recomputation, all 36 sparse layers, aligned 256-token blocks, and
  no more than the validated 524288-token cap.

`flat_exact` remains O(L²) arithmetic and is an 8K oracle. `block_cp` is subquadratic selection but
replicates global K/V, so it is a bounded 512K correctness bridge rather than a 1M–2M production
backend. Do not submit the archived `sparse_512k.sbatch`.

## Verified superlong data

The passing 512K gate used
`/scratch/project_465002530/users/bmoell/superlong_data/mix/data_path.args`, not a short-context
stand-in. It contains 48 weighted Megatron prefixes and had SHA-256
`8debcb373049ab52bbd8a03912da431b57acdf4dca0156c5ae73731a9099f5b8` at submission. All 48
indexed pairs passed checksum/read tests and real 302-sample blend builds at 512K, 1M, and 2M.
See `docs/superlong_context_plan.md` for report paths and `scripts/validate_megatron_indexed_mix.py`
for the repeatable validation command.
