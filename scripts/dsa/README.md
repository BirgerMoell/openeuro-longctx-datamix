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
- nonzero gradients in both the main model and indexer.

Not yet implemented or validated:

- sustained sparse adaptation and sparse checkpoint reload;
- context-parallel global top-k;
- scalable selected-set-KL retention at 512K–2M;
- subquadratic/hierarchical indexer candidate generation;
- sparse prefill/decode and KV-cache integration; and
- model export and serving.

## Production-path files

- `gpt_builders.py` — import shim expected by Megatron's `pretrain_gpt.py`
- `MEGATRON_REVISION` — exact external Megatron commit used by the validated LUMI gates
- `gpt_builders_dsa.py` — config bridge, fail-closed checkpoint load, freeze logic, gradient probes
- `megatron_gqa_dsa.py` — GQA-aware DSA module and layer specifications
- `chunked_indexer.py` — exact causal query/key-blocked selection and retention guard
- `dsa_sparse_loss.py` — selected-set KL with detached native-GQA teacher
- `dsa_patches.py` — sparse path, unsupported-mode guards, and recall logging
- `triton_dsa.py` — ROCm Triton sparse attention forward/backward
- `test_dsa_correctness.py` — dense-reference correctness suite
- `lumi/dsa_warmup_failclosed.sbatch` — frozen-indexer warm-up launcher
- `lumi/dsa_sparse_8k_correctness.sbatch` — one-step sparse integration gate

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

The GPU gate additionally compares native-GQA Triton forward/backward with a dense reference.

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
- context parallelism is one.

The exact indexer avoids allocating an L×L score tensor, but it still performs O(L²) arithmetic.
Do not describe the current end-to-end system as O(Lk), and do not submit the archived 512K sparse
launcher.
