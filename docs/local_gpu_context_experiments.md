# Local GPU Context-Extension Experiments

This is a small-GPU experiment plan for choosing which long-context recipe is
worth scaling to the OpenEuroLLM 9B Megatron/LUMI pipeline.

The aim is not to reproduce the full 9B run locally. Use the two local GPUs as a
proxy lab: run small models, short token budgets, and sharp evaluations that
predict which data and positional-training choices deserve cluster time.

## Short Answer

Yes, two smaller GPUs are useful if the experiments are framed as ablations on a
proxy model. They are not enough to answer "what is the final 9B score?", but
they are enough to answer:

- staged extension vs single jump
- balanced length curriculum vs target-length-only data
- high-signal long documents vs merely long documents
- YaRN/linear scaling vs sparse position supervision
- plain CLM vs RoPE-perturbed self-distillation
- long-context reasoning traces as a small post-extension mix

## Use The Existing Repo As The Spine

This repo already has the right data and evaluation anchors:

- `longctx filter-long` can build length-bucketed long-document slices.
- `longctx mix` can emit reproducible Megatron-style weighted mixtures.
- `scripts/eval_base_lm_niah.py` provides base-LM forced-choice retrieval
  scoring without requiring instruction tuning.
- `scripts/oneruler_score_base_lm.py` can score OneRuler examples for base LMs.
- `docs/ruler_eval_yarn_v2_32k.md` already identifies a key failure mode:
  single-needle retrieval drops at 32K, especially when the useful fact is at
  the beginning of the context.

Keep the local experiments compatible with these outputs: every run should save
its config, data mix, training log, and eval summaries under `runs/local_ctx/...`.

## Hardware Triage

Current non-cluster GPU host:

| Field | Value |
| --- | --- |
| SSH | `ssh ubuntu@77.87.121.41` |
| Hostname | `hot-poodle` |
| GPUs | 2 x NVIDIA L4, ~23 GB each |
| Repo | `/home/ubuntu/birger/openeuro-longctx-datamix` |
| Default CUDA env | `/home/ubuntu/birger/megatron_hf_conversion/.venv` |
| Post-cleanup disk | about 75 GB free on `/` as of 2026-06-08 |

Run this first on the actual GPU box:

```bash
cd /home/ubuntu/birger/openeuro-longctx-datamix
/home/ubuntu/birger/megatron_hf_conversion/.venv/bin/python scripts/probe_context_memory.py \
  --model <hf-or-local-proxy-model> \
  --lengths 2048 4096 8192 16384 32768 \
  --mode train
```

Interpretation:

| Result | Good local target |
| --- | --- |
| OOM above 8K | Use 0.5B-1B proxy, test EndPrompt/RoPE perturbations first |
| Train step works at 16K | Run staged 4K->8K->16K and length-mix ablations |
| Train step works at 32K | Run direct 32K vs staged 32K fairly |
| Inference only at 32K | Still useful for NIAH/RULER eval and data scoring |

Use BF16 where possible, gradient checkpointing, FlashAttention/SDPA if
available, micro-batch 1, and compare by total tokens rather than by steps.

Good first probes on `hot-poodle`:

```bash
# Tiny cached sanity check.
/home/ubuntu/birger/megatron_hf_conversion/.venv/bin/python scripts/probe_context_memory.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --lengths 2048 4096 8192 16384 32768 \
  --mode train \
  --attn-implementation sdpa

# More relevant 2B OpenEuroLLM datamix proxy.
# Use 80/20 as the default because it matches the planned 9B distribution.
/home/ubuntu/birger/megatron_hf_conversion/.venv/bin/python scripts/probe_context_memory.py \
  --model openeurollm/datamix-2b-80-20 \
  --lengths 2048 4096 8192 16384 32768 \
  --mode train \
  --attn-implementation sdpa
```

For a cheaper superlong-position plumbing test that does not require any HF
model or tokenizer cache, run the tiny byte-level smoke in
`docs/tiny_superlong_context_smoke.md`. This is the first thing to try before a
real 2B proxy run because it exercises 2M RoPE positions, PoSE-style offsets,
training, and forced-choice retrieval eval in one small script.

## Proxy Model Choice

Prefer a small RoPE-based causal LM whose architecture is close enough to the
target model to exercise the same positional machinery.

Default choice on `hot-poodle`: `openeurollm/datamix-2b-80-20`.

Why this one:

- It is a base Llama-style model, not instruction tuned.
- It uses the OpenEuroLLM tokenizer/vocab family.
- It starts at 2K native context, so context extension is a real adaptation.
- The 80/20 data mix is the closest local proxy for the planned 9B distribution.
- It is cached on the GPU host and fits 16K inference on one L4.

Use `openeurollm/datamix-2b-50-50` as the multilingual stress-control. Use Qwen
0.5B only for debugging scripts and memory behavior; it has a different tokenizer,
different RoPE settings, and a native 32K context, so it is a poor scientific proxy
for OpenEuroLLM context extension.

Recommended tiers:

| GPU memory | Proxy size | Purpose |
| --- | --- | --- |
| 2 x 12GB | 0.5B-1.1B | positional/data ablations only |
| 2 x 24GB | 1B-3B | most local experiments |
| 2 x 48GB | 3B-7B | confirm the top one or two recipes |

Use the same tokenizer as OpenEuroLLM when feasible for data-pipeline checks,
but do not force it if it complicates the proxy training. The local signal that
matters most is relative ranking between recipes under equal token budget.

## Evaluation Battery

Run before and after every experiment.

```bash
python scripts/eval_base_lm_niah.py \
  --model <model-or-adapter-merged-model> \
  --output runs/local_ctx/<run>/niah \
  --languages en sv de fr \
  --context-lengths 2048 4096 8192 16384 32768 \
  --depths 0.0 0.05 0.1 0.25 0.5 0.75 1.0 \
  --trials 10
```

Primary metrics:

- NIAH max-length average.
- NIAH far-distance cells: depth `0.0`, `0.05`, `0.1` at the maximum context.
- Short-context retention: 2K/4K NIAH and held-out short-text perplexity.
- Multilingual spread: at least English, Swedish, German, French; add Finnish
  and Czech for morphology/script stress.
- RULER/OneRuler smoke: `niah_single_1`, `niah_single_2`, `cwe`, `fwe`.

Promotion rule: a recipe is worth a larger run if it improves max-length NIAH by
at least 5-10 percentage points at equal tokens, does not regress 2K/4K by more
than 2-3 points, and repeats on at least two seeds or two proxy sizes.

## Experiment 0: Training-Free Baselines

Question: how much do we get from RoPE scaling alone?

Variants:

- no context patch, evaluate only up to native context
- linear/NTK-style scaling
- YaRN scaling with target factor
- current 32K converted model, if it fits for inference

Why: this measures the baseline that all training runs must beat. If training
does not beat scaling-only on far-distance NIAH, the data or objective is not
teaching useful positional behavior.

## Experiment 1: Staged Extension vs Single Jump

Question: does phased extension still win at small scale?

Use the same total token budget for all variants:

| Variant | Schedule |
| --- | --- |
| single jump | native -> 32K only |
| staged | native -> 4K -> 8K -> 16K -> 32K |
| mixed curriculum | every epoch samples 4K/8K/16K/32K |
| final polish | staged, then extra 32K-only tail |

Keep everything else fixed: model, data, optimizer, LoRA/full fine-tuning mode,
global token budget, and evals.

Expected useful answer: staged or mixed should preserve shorter contexts better;
if single jump matches it locally, the cluster roadmap can be simplified.

## Experiment 2: Length Mixture

Question: is target-length-only data hurting generalization?

Build three small datasets from the existing FinePDFs/tokenized artifacts:

| Variant | Length mix |
| --- | --- |
| target-only | mostly `>=16k`, packed/truncated to max length |
| repo default | `>=16k` / `4-16k` / `<4k` around `50/30/20` |
| balanced lengths | roughly uniform over 4K/8K/16K/32K training sequences |

Evaluate at all lengths, not only the maximum. The recent MMProLong paper found
balanced sequence-length distributions beat target-length-focused data, and
retrieval-heavy mixtures with modest reasoning data were strongest.

## Experiment 3: LongFilter-Lite Data Selection

Question: are the long documents actually useful as long-context examples?

Do a lightweight information-gain filter:

1. Sample candidate long documents.
2. Pick target spans near the end.
3. Score target-token loss with only a local window, e.g. 2K.
4. Score the same targets with a longer window, e.g. 16K or 32K.
5. Rank documents by loss reduction from the longer context.

Train equal-token runs on:

- random long documents
- top-scoring high-information-gain documents
- bottom-scoring long-but-local documents

This is the local version of the LongFilter/EntropyLong idea: length alone is
not enough; useful examples should make distant context measurably predictive.
PolicyLong suggests repeating the screening after a small amount of training,
so try one on-policy refresh for the best local candidate.

## Experiment 4: Positional Robustness

Question: can cheap positional supervision fix "lost at the beginning"?

Two local-friendly variants are worth testing:

| Method | Local implementation idea |
| --- | --- |
| RoPE-perturbed self-distillation | same tokens, altered `position_ids`, KL consistency to the normal view |
| terminal anchoring / EndPrompt | short physical sequence, terminal block assigned positions near target length |

These are attractive because they target positional brittleness without requiring
every training batch to be a full 32K sequence. The RoPE-perturbed method has one
extra forward pass. EndPrompt-style training is even cheaper, but it needs custom
`position_ids` support in the trainer.

Decision gate: if either method improves depth `0.0`/`0.05` at 32K while keeping
2K/4K unchanged, it is a strong candidate for a 9B follow-up.

## Experiment 5: LoRA vs Full Fine-Tuning

Question: can adapter training rank context-extension recipes reliably?

Variants:

- full fine-tune of a <=1B proxy
- LoRA rank 8/16/32 on attention projections
- LoRA rank 16/32 on all linear layers
- QLoRA if memory is tight

Do not assume LoRA is faithful for positional adaptation; measure it. If LoRA and
full fine-tuning rank recipes similarly on the proxy, use LoRA for broad sweeps.
If not, use full fine-tuning on the smallest proxy and reserve adapters for SFT.

## Experiment 6: Reasoning Traces After Retrieval Works

Question: when should reasoning traces enter the mix?

Only test this after a context-extension recipe can retrieve reliably. Add a
small SFT/post-training mix:

| Variant | Mix |
| --- | --- |
| retrieval only | long docs + synthetic key/value retrieval |
| retrieval + 5% traces | ACC-style compiled trajectories or long QA traces |
| retrieval + 15% traces | stress test for forgetting and formatting drift |

Use evidence-grounded traces, not free-form chain-of-thought dumps. ACC supports
the idea of compiling trajectories into long-context QA, and LongRLVR argues that
long-context reasoning needs dense/verifiable grounding signals rather than only
final-answer rewards.

Local decision: if traces improve multi-hop/key-chain tasks without hurting NIAH
or short perplexity, keep them for post-extension SFT. If they hurt retrieval,
move them later in the pipeline.

## Suggested Four-Week Local Schedule

| Week | Work | Output |
| --- | --- | --- |
| 1 | memory probe, eval baseline, scaling-only baselines | max feasible context and baseline plots |
| 2 | staged vs single jump, length mixture | first recipe ranking |
| 3 | LongFilter-lite and one on-policy refresh | data-quality decision |
| 4 | RoPE perturbation / EndPrompt, LoRA vs full check | shortlist for 9B cluster run |

The first serious cluster candidate should be the smallest recipe that:

- fixes the 32K far-distance NIAH failure,
- improves RULER `niah_single_1/2` and `cwe/fwe`,
- preserves 2K/4K behavior,
- works across at least four European languages,
- and has a plausible implementation path in the OpenEuroLLM Megatron fork.

## What Not To Spend Local GPU Time On First

- Full 9B context-extension training.
- 128K+ training before 32K depth-0 failures are understood.
- RLVR/GRPO before there is a stable long-context SFT/retrieval model.
- All 38 languages in every run; use a representative multilingual subset.
- Target-length-only training without a shorter-length retention eval.

## Paper Trail

- MMProLong: balanced length distributions, retrieval-heavy data, and modest
  reasoning data for long-context continued pre-training:
  https://arxiv.org/abs/2605.13831
- ACC: compile agent trajectories into long-context QA pairs:
  https://arxiv.org/abs/2605.21850
- PolicyLong: on-policy long-context data screening:
  https://arxiv.org/abs/2604.07809
- RoPE-perturbed self-distillation for positional robustness:
  https://arxiv.org/abs/2604.14339
- EndPrompt terminal anchoring for cheap sparse positional supervision:
  https://arxiv.org/abs/2605.14589
- LongRLVR: dense verifiable context rewards for long-context reasoning:
  https://arxiv.org/abs/2603.02146
- MEMENTO: train models to compress/manage long reasoning context:
  https://arxiv.org/abs/2604.09852
- RoPE limitations in long contexts:
  https://arxiv.org/abs/2605.15514
- LongFilter: select documents with measurable long-range information gain:
  https://openreview.net/pdf/4d1d0f808795415e8b2fd323482ebd30506f0966.pdf
