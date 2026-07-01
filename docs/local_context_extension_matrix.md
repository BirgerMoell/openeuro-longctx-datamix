# Local Context-Extension Matrix

This matrix is for `hot-poodle` (`ubuntu@77.87.121.41`, 2 x NVIDIA L4) using
`openeurollm/datamix-2b-80-20` as the main proxy model and the local streamed
FinePDFs long-context artifact:

```text
/home/ubuntu/birger/openeuro-longctx-datamix/data/local_stream_subset_4096_artifact/mix/data_mix.json
```

The goal is to rank techniques, not train a final model. Keep token budgets equal
inside each comparison.

## Technique Axis

| Label | `--rope-method` | Why test it |
| --- | --- | --- |
| native/control | `none` | short-context retention and sanity baseline |
| linear PI | `linear` | cheap scalar interpolation baseline |
| dynamic NTK | `dynamic` | common inference/training baseline |
| YaRN | `yarn` | current repo/cluster default |
| LongRoPE flat | `longrope_uniform` | LongRoPE machinery with scalar-like factors |
| LongRoPE2-lite ramp | `longrope_ramp` | non-uniform factors that stretch low-frequency dimensions more |
| LongRoPE2-lite inverse | `longrope_inverse_ramp` | sanity check for the opposite factor allocation |
| LongRoPE2-lite two-band | `longrope_two_band` | protects one band and stretches the rest |
| searched factors | `longrope_json` | later: factors from a small held-out search |
| DroPE / NoPE | `drope_nope` | remove RoPE after pretraining, then recalibrate |

The LongRoPE2-lite variants are not the full Microsoft LongRoPE2 optimization.
They are local proxy candidates to test whether non-uniform factorization is worth
a larger LongRoPE2 search.

DroPE is a different kind of candidate. It treats RoPE as a training scaffold and
then nulls positional embeddings during recalibration/inference. Test it as a
short recalibration method, not as a RoPE scaling curve.

## Data/Objectives Axis

| Label | Data | Question |
| --- | --- | --- |
| long-doc CLM | local FinePDFs artifact, tiers `16k_plus,4_16k` | does positional method learn from natural long docs? |
| tier mix | local artifact, all tiers | does short-document retention improve? |
| long-only | tier `16k_plus` only | does target-length emphasis help or overfit? |
| trace-after | first long-doc CLM, then 5-10% reasoning traces | do traces improve reasoning without hurting retrieval? |
| trace-mixed | 5-15% traces mixed during CLM | does early reasoning-trace mixing harm context extension? |

Use evidence-grounded traces if possible: search/read/answer, proof with cited
facts, or agent trajectories compiled into long-context QA. Avoid generic
chain-of-thought dumps as the first trace experiment.

## Phase 0: Scaling-Only Eval

Run NIAH/RULER-style evaluation with no training for `linear`, `dynamic`, `yarn`,
and the LongRoPE2-lite candidates. This tells us what a training run must beat.

Primary metric: max-length NIAH at depths `0.0`, `0.05`, `0.1`, plus 2K/4K
retention.

## Phase 1: 4K Full Fine-Tuning

Full backward on `datamix-2b-80-20` fits 4K on one L4. Run matched-token jobs:

```bash
cd /home/ubuntu/birger/openeuro-longctx-datamix
PY=/home/ubuntu/birger/swedish-medical-benchmark/.venv-grpo/bin/python
MIX=data/local_stream_subset_4096_artifact/mix/data_mix.json

CUDA_VISIBLE_DEVICES=0 $PY scripts/train_hf_longctx.py \
  --model openeurollm/datamix-2b-80-20 \
  --mix-json $MIX \
  --output-dir runs/local_ctx/yarn_4k_full_tiermix \
  --rope-method yarn \
  --target-context 4096 \
  --seq-len 4096 \
  --tiers 16k_plus,4_16k,under4k \
  --steps 100 \
  --learning-rate 2e-5 \
  --optimizer adamw8bit \
  --eval-every 25
```

Repeat with `--rope-method linear`, `dynamic`, `longrope_ramp`, and
`longrope_two_band`. Keep `steps`, `seq-len`, and selected tiers fixed.

Decision: promote only methods that improve 4K eval loss/NIAH without hurting 2K
retention.

## Phase 2: 8K LoRA

Full 8K backward OOMs on one L4, so use LoRA first:

```bash
CUDA_VISIBLE_DEVICES=1 $PY scripts/train_hf_longctx.py \
  --model openeurollm/datamix-2b-80-20 \
  --mix-json $MIX \
  --output-dir runs/local_ctx/yarn_8k_lora_longdocs \
  --rope-method yarn \
  --target-context 8192 \
  --seq-len 8192 \
  --tiers 16k_plus \
  --lora \
  --lora-r 16 \
  --lora-target attention \
  --steps 100 \
  --learning-rate 1e-4 \
  --eval-every 25
```

Compare `yarn`, `longrope_ramp`, `longrope_two_band`, and `linear`. If LoRA
ranking agrees with the 4K full-finetune ranking, use LoRA for broader 8K sweeps.
If not, treat LoRA as a memory workaround only and confirm winners with a smaller
full-finetune proxy.

## Phase 2b: DroPE Recalibration

DroPE should get a small, separate run because it changes the inductive bias more
sharply than YaRN/LongRoPE. Start at 4K full fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_hf_longctx.py \
  --model openeurollm/datamix-2b-80-20 \
  --mix-json $MIX \
  --output-dir runs/local_ctx/drope_4k_full_tiermix \
  --rope-method drope_nope \
  --target-context 4096 \
  --seq-len 4096 \
  --tiers 16k_plus,4_16k,under4k \
  --steps 100 \
  --learning-rate 2e-5 \
  --optimizer adamw8bit \
  --eval-every 25
```

Then try 8K LoRA only if 4K DroPE preserves short-context behavior:

```bash
CUDA_VISIBLE_DEVICES=1 $PY scripts/train_hf_longctx.py \
  --model openeurollm/datamix-2b-80-20 \
  --mix-json $MIX \
  --output-dir runs/local_ctx/drope_8k_lora_longdocs \
  --rope-method drope_nope \
  --target-context 8192 \
  --seq-len 8192 \
  --tiers 16k_plus \
  --lora \
  --steps 100 \
  --learning-rate 1e-4 \
  --optimizer adamw8bit
```

Decision gate: DroPE is interesting only if it improves extrapolation while
keeping 2K/4K retrieval and short perplexity close to the base model. A sharp
short-context regression means it needs more careful recalibration or Q/K norm
variants before it is worth scaling.

## Phase 3: LongRoPE2-Lite Search

Do a small factor search before training longer runs:

1. Generate 10-30 `long_factor` arrays: uniform, ramp, inverse ramp, two-band,
   random monotone ramps around factor 4 or 8.
2. Run scaling-only NIAH and held-out long-doc loss.
3. Train only the top 2-3 at 4K/8K.
4. Save the winning factors as JSON and rerun via `--rope-method longrope_json`.

Promotion rule: LongRoPE2-lite must beat YaRN on far-distance retrieval or
held-out long-doc loss at equal tokens, while preserving short-context metrics.

## Phase 4: Reasoning Trace Ablation

Use the same winning positional method from Phases 1-3.

Prepare a JSONL file with one of these schemas:

```json
{"text": "...full trace or compiled trajectory..."}
{"prompt": "...", "response": "..."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=1 $PY scripts/train_hf_longctx.py \
  --model openeurollm/datamix-2b-80-20 \
  --mix-json $MIX \
  --output-dir runs/local_ctx/yarn_8k_lora_trace05 \
  --rope-method yarn \
  --target-context 8192 \
  --seq-len 8192 \
  --tiers 16k_plus \
  --lora \
  --trace-jsonl /path/to/reasoning_traces.jsonl \
  --trace-weight 0.05 \
  --steps 100 \
  --learning-rate 1e-4 \
  --optimizer adamw8bit
```

Compare:

- `trace_weight=0.00` control
- `trace_weight=0.05`
- `trace_weight=0.15`
- traces after long-doc CLM vs traces mixed from the start

Expected outcome: reasoning traces may improve multi-hop/key-chain tasks, but can
hurt pure retrieval if added too early or too heavily. If NIAH depth-0 drops, move
traces later in the pipeline and keep context extension as CLM/retrieval first.

## Minimal First Run

Run these four first:

| Run | Context | Tuning | Method | Data |
| --- | --- | --- | --- | --- |
| `linear_4k_full` | 4K | full | linear | tier mix |
| `yarn_4k_full` | 4K | full | YaRN | tier mix |
| `longrope_ramp_4k_full` | 4K | full | LongRoPE2-lite | tier mix |
| `drope_4k_full` | 4K | full | DroPE/NoPE | tier mix |
| `yarn_8k_lora` | 8K | LoRA | YaRN | 16K+ only |

That gives a fast answer to: scalar baseline vs YaRN vs non-uniform factors, and
whether DroPE-style recalibration and 8K adapter training are viable on the L4s.

References:

- DroPE: https://sakana.ai/drope/ and https://arxiv.org/abs/2512.12167
- LongRoPE2: https://arxiv.org/abs/2502.20082
- LongRoPE: https://arxiv.org/abs/2402.13753
- YaRN: https://openreview.net/forum?id=wHBfxhZu1u
