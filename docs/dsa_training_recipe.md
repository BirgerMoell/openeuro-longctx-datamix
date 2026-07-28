# DSA training recipe

**Status:** operational 8K recipe, updated 2026-07-28
**Canonical design/status:** [sparse_attention_dsa.md](sparse_attention_dsa.md)

This recipe separates indexer warm-up from sparse adaptation and preserves the two independent
gradient paths described in the DeepSeek-V3.2 DSA design.

## Phase 1: frozen-indexer warm-up

Run dense content attention and train only the lightning indexer.

Required invariants:

- `DSA_FREEZE_MODEL=1`;
- `DSA_SPARSE=0` and no sparse-run patch;
- positive `DSA_LOSS_COEFF`;
- all intended DSA layers present in `DSA_PATTERN`;
- the base checkpoint loads with zero missing non-indexer tensors;
- non-indexer weights remain frozen;
- indexer loss and gradients are finite and nonzero; and
- recall is logged per layer at top-512/1024/2048.

The completed pilot used 8K, all 36 layers, 16 indexer heads of dimension 128, and a 1e-3→1e-4
cosine indexer LR. Job 20291047 completed 300 steps and saved checkpoints every 50 steps. The
fail-closed launcher is `scripts/dsa/lumi/dsa_warmup_failclosed.sbatch`.

Do not infer that a short-context indexer automatically generalizes to 512K–2M. That is an
experiment to measure, not a premise.

## Phase 2: sparse correctness gate

Load the complete iteration-300 warm checkpoint, do not save, and execute one sparse update.

Required invariants:

- `DSA_SPARSE_RUN=1`;
- `DSA_SPARSE=1` so selected-set KL remains active;
- `DSA_FREEZE_MODEL=0`;
- `DSA_LOSS_COEFF>0`;
- all 36 global-attention layers are `S`;
- k=2048;
- TP=8, PP=1, CP=1;
- non-interleaved indexer RoPE;
- exact causal blocked selection;
- finite LM/indexer losses and grad norm; and
- nonzero gradient probes for both main-model and indexer parameters.

Job 20336946 passed this gate. The reproducible launcher is
`scripts/dsa/lumi/dsa_sparse_8k_correctness.sbatch`.

## Phase 3: sustained sparse adaptation

The next run should remain at 8K and CP=1 for 100–500 steps. This is deliberately a training-quality
test rather than a scale test.

Recommended initial settings:

| Parameter | Value |
|---|---:|
| Sequence length | 8,192 |
| Top-k | 2,048 |
| DSA layers | 36/36 |
| TP / PP / CP | 8 / 1 / 1 |
| Main/indexer LR | 7.3e-6 constant initially |
| Selected-set KL coefficient | 0.1 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Checkpoint interval | 25–50 steps |

Keep fixed held-out batches for dense/sparse comparisons. Save and reload at least one intermediate
checkpoint, then continue for several steps.

Stop immediately on:

- NaN/Inf in LM loss, indexer loss, recall, or gradients;
- a zero gradient in either parameter family;
- a checkpoint that cannot reload exactly;
- persistent recall collapse;
- a material short-context loss regression; or
- an unsupported mask/parallelism error.

A gradual k schedule (for example 4096→2048) is a possible ablation if the hard switch is unstable.
It is not part of the validated baseline and should not replace the k=2048 comparison.

## Scaling gates

Do not increase context merely because the 8K step works.

1. Benchmark 16K and 32K at CP=1.
2. Implement CP=2 global top-k and prove equality with CP=1 on small inputs.
3. Extend CP tests to 4+ with deterministic tie handling and selected-K/V exchange.
4. Stream or recompute selected-set KL; the current B×L×k retained tensors reach about 16 GiB per
   layer at 1M×2048.
5. Benchmark the exact O(L²) indexer. Add hierarchical candidate generation only if recall and
   quality are measured against the exact oracle.
6. Move progressively through 64K, 128K, 256K, 512K, and 1M.
7. Treat sparse prefill and sparse decoding as separate inference deliverables.

The archived `scripts/dsa/lumi/sparse_512k.sbatch` is intentionally disabled because it combines
unsupported CP with an obsolete sparse-loss configuration.

## Data and evaluation

Use a mixture that retains normal short-context language modeling while increasing coherent
long-form and distributed-dependency examples. Kimi K3 supports progressive extension and
long-context data curation, but its KDA/NoPE architecture is not a replacement for this DSA/RoPE
implementation.

Minimum evaluation per gate:

- held-out LM loss at short and stage context;
- per-layer and query-quartile attention-mass recall;
- NIAH/RULER-style retrieval across depths;
- dense-versus-sparse fixed-batch comparison;
- checkpoint save/reload/continue; and
- separate indexer, selection, sparse-core, KL, and end-to-end timings.

## References

- [DeepSeek-V3.2 technical report](https://arxiv.org/html/2512.02556)
- [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- [Kimi K3 technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
