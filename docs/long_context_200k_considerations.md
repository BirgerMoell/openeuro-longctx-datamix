# Scaling to 200K+ Context: Key Considerations

This document outlines the main technical challenges and design decisions for extending
the YaRN multilingual model beyond 32K tokens toward 200K+ context lengths.

Current state: the v2 model uses YaRN with factor=16.0 (2048→32768 tokens) and
mscale=1.277. The discussion below assumes a 9B parameter model on AMD MI250X hardware.

---

## 1. RoPE Scaling

At 200K tokens from a 2048-token base, the required scaling factor is ~100× (vs 16× for
32K). This pushes uniform frequency interpolation to its limits.

**Problems with uniform scaling at 100×:**
- High-frequency RoPE dimensions are compressed so aggressively they lose the ability to
  distinguish nearby token positions
- The attention temperature correction (mscale) would reach ~1.46 — larger corrections
  introduce their own distortions

**Better approaches:**
- **Non-uniform scaling** (LongRoPE / LongRoPE v2): compress low-frequency dimensions
  more than high-frequency ones, preserving local position sensitivity
- **Dynamic NTK**: adjust the effective scale factor based on the actual sequence length
  at inference time, rather than fixing it at training time
- **Staged extension**: 2K→8K→32K→128K→200K, with a short continued pre-training phase
  at each step rather than jumping directly to the target length

---

## 2. Attention Memory and Compute

Standard full attention is O(n²) in both time and memory. At 200K tokens:

- Attention score matrix per layer: 200K² = 40 billion values
- KV cache for a 9B model (32 layers, 32 heads, head_dim=128): ~105 GB at bf16

This exceeds the memory of a single MI250X GPU (128 GB) and makes single-GPU training
impractical even with FlashAttention.

**Solutions:**

| Approach | Trade-off |
|---|---|
| **Ring attention / context parallelism** | Splits the KV cache across GPUs; full attention preserved; requires inter-GPU communication per layer |
| **Sliding window + global tokens** (Longformer-style) | O(n×window) cost; loses full cross-sequence attention |
| **KV cache quantization** | 4-bit KV reduces memory 4×; small accuracy cost |
| **Streaming KV eviction** (H2O, StreamingLLM) | Keeps only recent + "sink" tokens; strong for generation, weaker for retrieval |

For training a 9B model at 200K, context parallelism across 4–8 GPUs is likely the
minimum viable approach.

---

## 3. Training Data

Very few documents are naturally longer than 200K tokens. Constructing training sequences
requires deliberate data engineering.

**Options:**
- **Book concatenation**: stitch related chapters or full books into single sequences
- **Code repositories**: a full repo with all files concatenated is often 50K–500K tokens
- **Legal/scientific corpora**: long contracts, patents, multi-paper reviews
- **Synthetic construction**: generate long-context tasks (multi-document QA, timeline
  reconstruction) where the model must use the full sequence

**Multilingual caveat**: long native-language documents exist for maybe 5–10 of the 38
OELLM languages. For lower-resource languages (mt, ga, lb, etc.) the effective maximum
natural document length may be 10–50K tokens, making 200K sequences require cross-document
stitching — which teaches the model that context boundaries don't matter.

**Cross-document contamination**: stitching unrelated documents together without clear
separators causes the model to learn to ignore earlier context, since the first document
is irrelevant to the final document's continuation. Always include explicit document
boundary tokens and consider whether cross-document attention should be masked.

---

## 4. The "Lost in the Middle" Problem

Even at 32K we observe systematic failure at depth=0% (needle at the very start of
context). The underlying causes compound at 200K:

1. **Attention sinks**: the first 1–4 tokens accumulate disproportionate attention weight,
   making the model "look through" nearby content to always attend back to position 0–3
2. **Extreme RoPE interpolation**: at relative distance 200K, the position encodings are
   so compressed that early tokens are nearly indistinguishable from each other

**Mitigation strategies:**
- Long-context SFT data that specifically requires early-context retrieval — forces the
  model to learn to attend to position 0 for content, not just as a sink
- Position ID remapping experiments (shift the sink tokens to not occupy position 0)
- Attention sink regularization during fine-tuning

Without explicit intervention, retrieval accuracy from the first ~5% of a 200K context
is likely to be near chance.

---

## 5. Training Infrastructure

At 200K sequence length, standard data + tensor parallelism is not sufficient.

**Required additions:**

- **Context parallelism (CP)**: splits the sequence dimension across GPUs. Megatron-LM
  supports this natively. For 200K on a 9B model, CP=4 or CP=8 is likely needed.
- **Sequence parallelism**: distributes LayerNorm and dropout across the sequence dimension
  (complementary to tensor parallelism, already supported in Megatron).
- **Activation recomputation within attention**: standard gradient checkpointing saves
  activations at layer boundaries; at 200K you also need to recompute within the
  FlashAttention kernel itself to avoid storing the full attention matrix.

**Rough estimate for LUMI standard-g:**
A 9B model at 200K with CP=4 needs a node with ≥4 MI250X GPUs per replica. At 32 nodes
× 8 GCDs (treating each die as a GPU), a configuration of TP=4, PP=2, CP=4, DP=8 is
plausible but would require careful tuning of micro-batch size and gradient accumulation
steps to keep GPU utilization high.

---

## 6. Training Recipe: Staged Extension

Jumping directly from 2K to 200K is inefficient and likely unstable. The standard
approach is staged:

```
Stage 1: Pre-train at 2K (done — base model)
Stage 2: Extend to 8K  — ~1B tokens, YaRN factor=4
Stage 3: Extend to 32K — ~500M tokens, YaRN factor=16  (done — v2 model)
Stage 4: Extend to 128K — ~200M tokens, non-uniform RoPE
Stage 5: Extend to 200K+ — ~100M tokens, non-uniform RoPE
Stage 6: Long-context SFT — task-specific fine-tuning at target length
```

Token counts at each stage decrease because the model only needs to learn new positional
patterns, not new world knowledge. The longest stages (4 and 5) are expensive but
shorter than pre-training.

---

## 7. Evaluation at 200K

The current 5-depth NIAH grid (25 cells per language) does not adequately characterize
200K context models. Extensions needed:

**More granular depth sampling**: test every 5% rather than every 25%, especially in the
first 10% where failure modes are concentrated.

**Multi-hop retrieval**: place two related facts at different positions; the model must
combine them to answer. Tests whether the model can hold multiple retrieved facts in
working memory.

**Aggregation tasks**: count occurrences, find the most common item, summarize across
the full context. These require attending to the entire sequence, not just a single needle.

**Distractor density**: at 32K we use ~3 distractors. At 200K a realistic test would use
50–200 distractors to assess signal-to-noise robustness.

**Real-task benchmarks**: book-length QA (e.g. NarrativeQA at full book length),
repository-level code understanding, multi-document summarization.

**OneRuler**: the adapted base-LM version of OneRuler (see
`docs/eval_base_lm_niah.md`) covers NIAH single/multi-key/multi-value and noexist tasks
across 38 languages. Extending it to 200K context lengths would give a strong multilingual
long-context benchmark.

---

## 8. Multilingual-Specific Considerations

**Tokenization efficiency**: some languages tokenize much less efficiently than English.
At 200K tokens, a language with 2× the tokens-per-word ratio effectively has only 100K
words of semantic content — the same as an English 200K-token sequence has ~200K words.
Reporting results in tokens AND in approximate word counts helps interpret cross-language
differences.

**Resource imbalance in long-context data**: high-resource languages (en, de, fr) have
abundant long documents; low-resource languages (mt, ga, lb) do not. A 200K multilingual
model trained on stitched short documents for these languages may learn degenerate
positional representations for those languages specifically.

**Script and tokenization diversity**: languages using non-Latin scripts (bg, el, uk, mk,
sr) tend to have higher token/character ratios, compressing their effective context window
further.

---

## Summary: What Would Need to Change

| Component | 32K (current) | 200K (target) |
|---|---|---|
| RoPE scaling | YaRN uniform, factor=16 | Non-uniform (LongRoPE), factor~100 |
| Attention | Single-GPU FlashAttention | Context parallelism across 4–8 GPUs |
| KV cache | ~6 GB at 32K | ~105 GB at 200K → requires quantization or CP |
| Training data | Standard pre-training mix | Long-document curation + synthetic tasks |
| Training recipe | Single extension stage | Staged (32K→128K→200K) |
| Eval | 5-depth NIAH, 25 cells | Multi-hop, aggregation, 200-distractor NIAH |
| Infrastructure | 1 GPU (eval), 32-node (train) | 32-node with CP=4+ mandatory |
