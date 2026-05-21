# Context Extension Roadmap

Notes and decisions from planning meeting, 2026-05-21.

---

## Decision: Phased YaRN Extension

Rather than jumping directly to 256K, extend in discrete stages using YaRN (not LongRoPE
— LongRoPE requires a parameter search which adds complexity; YaRN has a fixed, derivable
configuration). Each stage produces an independently usable model checkpoint.

**Target stages:**

| Stage | Context length | RoPE factor | Notes |
|---|---|---|---|
| Base | 2K | 1× | Existing pre-trained model |
| 1 | 4K | 2× | Sanity check — minimal extension |
| 2 | 8K | 4× | |
| 3 | 16K | 8× | |
| 4 | 32K | 16× | Current v2 model ✓ |
| 5 | 64K | 32× | First new stage |
| 6 | 128K | 64× | |
| 7 | 256K | 128× | Target |

Each stage gets a short continued pre-training run on a mix of short and long sequences.
The key insight is that **the same token budget should be used at each stage** — extending
to 4K costs roughly the same as extending to 256K in terms of training tokens. The model
only needs to learn new positional patterns, not new world knowledge, so 500M–2B tokens
per stage is sufficient.

---

## Training Data

### Volume
10–20 billion tokens total across all stages, with long-context documents as the primary
new ingredient at each extension stage.

### Source: Fine-PDFs
Use fine-PDFs (high-quality PDF-extracted documents) as the primary long-context data
source. These are natural long documents rather than stitched-together short ones,
reducing cross-document contamination artifacts.

Open questions:
- Does the fine-PDF corpus have sufficient multilingual coverage for all 38 OELLM
  languages, or will some languages need synthetic long-document construction?
- What is the minimum document length threshold to include in long-context training?

### Construction for very long sequences (128K–256K)
For the longest stages, individual documents may not reach 128K+ tokens. Strategy:
- Find 2–3 super-long examples per language (e.g. full books, long legal documents,
  comprehensive technical manuals)
- Build a dedicated long-sequence dataset from these, targeting 256K token sequences
- Split the dataset by context length bucket: sequences of 16K, 32K, 64K, 128K, 256K
  are all included; the model sees a curriculum of increasing lengths

### Data mix
Maintain approximately **80% English / 20% other languages** throughout — consistent
with the base model's training distribution. Shifting the mix too far toward multilingual
content during context extension risks degrading English performance.

---

## Evaluation: RULER

Use **RULER** (NVIDIA) for evaluation at each stage, since it works with non-instruction-
tuned base language models (scored by log-likelihood / forced completion, not generation).

Repository: https://github.com/NVIDIA/RULER

RULER covers:
- NIAH single, multi-key, multi-value, multi-query
- Variable tracking (VT)
- Common word extraction (CWE)
- Question answering (QA) — this one does require instruction following; skip for base LM

RULER is English-only in the standard distribution. For multilingual coverage, the
OneRuler fork (https://github.com/BirgerMoell/OneRuler-OELLM) adapts it to 38 languages.
The adapted base-LM scoring approach (forced-choice log-likelihood) can run RULER tasks
without instruction tuning.

**Eval plan per stage:**
Run RULER at the new maximum context length + all shorter lengths to verify that shorter
context performance is preserved. A regression at 4K after extending to 8K is a red flag
that the training mix needs more short-sequence data.

**Making the eval harder:**
The current 5-depth NIAH is relatively easy (only 1 distractor number per context). RULER
supports configuring distractor density — increase to 10–20 distractors at longer context
lengths to get a more discriminative signal.

---

## Post-Training Pipeline

Once the context-extended base model reaches the target length, the planned post-training
sequence is:

```
Context extension (this roadmap)
    ↓
SFT (supervised fine-tuning on instruction data)   ← talk with Amaru
    ↓
Tool use
    ↓
Reasoning
    ↓
Verifiable rewards
    ↓
GRPO (Group Relative Policy Optimization)
```

The SFT step should include long-context instruction data so that the context extension
is preserved through post-training — instruction tuning on short data only can degrade
long-context capability.

---

## Immediate Next Steps

- [ ] **Validate the phased extension hypothesis at small scale**: run 2K→4K→8K→16K→32K
  extension on the existing base model with a fixed small token budget (e.g. 500M tokens
  per stage). Compare per-stage RULER scores to verify that staged extension improves over
  a single-step jump. This is the key empirical question to answer first.

- [ ] **Set up RULER eval pipeline**: clone NVIDIA/RULER, configure for base-LM scoring
  (log-likelihood), run on the existing 32K v2 model as a baseline before starting new
  stages.

- [ ] **Audit fine-PDF corpus**: determine available document lengths per language,
  identify which languages have natural documents >32K tokens and which will need
  synthetic long-sequence construction.

- [ ] **Prepare sbatch templates** for each extension stage on LUMI / Leonardo /
  MareNostrum (see `docs/long_context_200k_considerations.md` for infrastructure details).

- [ ] **Coordinate with Amaru** on SFT data and post-training pipeline requirements —
  particularly what long-context instruction data exists or needs to be created.

---

## Open Questions

1. **What is the base training context length for the next full pre-training run?**
   Options: start at 2K (same as current) or start at 4K to reduce the number of
   extension stages needed. Starting at 4K costs more in pre-training but saves one
   extension stage.

2. **Should the 80/20 EN/multilingual mix be rebalanced at longer context lengths?**
   Long multilingual documents are scarcer — if the 20% multilingual target can't be
   met with real documents, synthetic sequences may introduce noise.

3. **How many independent model checkpoints to maintain?** One per stage (7 models)
   gives flexibility for ablations and deployment at multiple context lengths, but
   multiplies storage and eval cost.

4. **RULER multilingual gap**: the standard RULER English eval is well-understood.
   For non-English languages, the OneRuler adapted eval is not yet validated against
   human performance on the tasks. Is this good enough for a research artifact, or does
   it need native-speaker validation before being cited?
