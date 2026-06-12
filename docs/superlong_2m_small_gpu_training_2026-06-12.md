# 2M Superlong Small-GPU Training Smoke - 2026-06-12

This note records the small-GPU experiments run on `hot-poodle`
(`ubuntu@77.87.121.41`, 2 x NVIDIA L4) to turn the earlier tiny
superlong-context smoke test into a more realistic training check.

## Question

Can a real pretrained model be trained on a small GPU in a way that exercises
a 2M-token positional span, rather than only proving that a tiny model accepts
large `position_ids`?

## Trainer

Script:

```bash
scripts/train_hf_superlong_smoke.py
```

Key properties:

- Uses a real Hugging Face causal LM (`openeurollm/datamix-2b-80-20` in these runs).
- Freezes most of the model and trains only the final decoder layers plus final norm.
- Uses suffix-only loss so it does not materialize full `seq_len x vocab` logits.
- Supports `pose_bridge` position IDs: evidence is placed in an earlier virtual
  segment and the query/answer segment is placed near position 2,097,152.
- Supports YaRN/linear/dynamic RoPE scaling. The 2B model has original
  `max_position_embeddings=2048`; these runs use YaRN factor `1024.0`.
- Supports a log-space training curriculum from a smaller virtual span to 2M.
- Supports optional contrastive forced-choice retrieval loss.

## What the first real run showed

The first HF run used Qwen 0.5B and passed mechanically, but it was too easy:
the model solved the forced-choice code retrieval before training. That proved
the plumbing, not learning.

The harder OpenEuroLLM 2B runs changed the task so wrong answers also appeared
in the context. That prevents the model from winning by simply preferring any
seen number.

## Important bug fixed

The first `pose_far` strategy applied one common offset to all tokens. For RoPE
that mostly preserves relative distances, so it is not a true 2M relative-distance
test. The trainer now uses `pose_bridge`, which places the evidence and query in
different virtual segments with a large positional gap.

Gradient checkpointing was also made opt-in. With frozen early layers, default
checkpointing emitted warnings and could prevent gradients from reaching the
trainable tail.

## Results

All runs used the real OpenEuroLLM 2B model on L4 GPUs. Peak memory stayed well
below 23 GB per GPU.

| Run | Task | Trainable | Objective | Baseline | Best/final | Peak memory | Remote output |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| unscaled 2M bridge | random 7-digit code | last 4 layers | generative | 3/18 | 3/18 at step 600 | 7.6 GB | `/home/ubuntu/birger/runs/hf_superlong/oellm2b_bridge_4k_last4_800_20260612` |
| YaRN 2M bridge | random 7-digit code | last 4 layers | generative | 2/18 | 1/18 at step 400 | 7.6 GB | `/home/ubuntu/birger/runs/hf_superlong/oellm2b_yarn_bridge_4k_last4_800_20260612` |
| YaRN 2M bridge | status label, all depths | last 4 layers | generative | 5/36 | 5/36 at step 400 | 7.6 GB | `/home/ubuntu/birger/runs/hf_superlong/oellm2b_yarn_bridge_status_4k_last4_600_20260612` |
| YaRN 2M bridge + curriculum | status label, all depths | last 8 layers | generative | 5/24 | 6/24 at step 400 | 10.7 GB | `/home/ubuntu/birger/runs/hf_superlong/oellm2b_yarn_bridge_status_4k_last8_curriculum_800_20260612` |
| YaRN 2M bridge + curriculum | status label, depth 0.9 | last 8 layers | generative | 2/16 | 3/16 at step 300, 0/16 final | 10.7 GB | `/home/ubuntu/birger/runs/hf_superlong/oellm2b_yarn_bridge_status_depth09_last8_curriculum_600_20260612` |
| YaRN 2M bridge + curriculum | status label, depth 0.9 | last 8 layers | generative + contrastive | 1/16 | 3/16 at step 150, 5/16 final | 16.3 GB | `/home/ubuntu/birger/runs/hf_superlong/oellm2b_yarn_bridge_status_depth09_contrastive_2k_last8_300_20260612` |

## Interpretation

The realistic small-GPU experiment is now useful because it distinguishes three
things:

1. The real model can run 2M virtual position IDs with YaRN and PoSE-style bridge
   positions on an L4.
2. Plain generative answer loss is not enough to produce reliable 2M bridge
   retrieval improvement in a tiny partial-tuning run.
3. Direct retrieval optimization helps: the contrastive run improved from 1/16
   to 5/16 on fresh forced-choice examples while still using the 2M bridge.

This is not yet a strong success. It is a credible minimal positive signal and,
more importantly, it tells us what the next serious experiment should be.

## Recommended Next Experiment

For a proper small-but-real 2M extension experiment:

- Use YaRN or LongRoPE scaling from the start.
- Use `pose_bridge`, not common-offset `pose_far`, for superlong-distance checks.
- Train with a curriculum over both virtual distance and retrieval depth:
  `32k -> 128k -> 512k -> 2M`, then depth `0.9 -> 0.5 -> 0.05`.
- Include contrastive retrieval loss for a small fraction of batches, because
  generative-only loss did not move the retrieval metric.
- Increase eval trials to reduce noise once the setup is fixed.
- On Leonardo, use the 9B model with distributed optimizer and a larger batch of
  constructed long documents; this L4 experiment should stay as the fast gate.

## Reproducing The Best Small Run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_hf_superlong_smoke.py \
  --model openeurollm/datamix-2b-80-20 \
  --local-files-only \
  --output-dir /path/to/oellm2b_yarn_bridge_status_depth09_contrastive_2k_last8_300 \
  --seq-len 2048 \
  --max-position 2097152 \
  --train-start-position 32768 \
  --rope-scaling yarn \
  --position-strategy pose_bridge \
  --eval-position-strategy pose_bridge \
  --train-steps 300 \
  --last-n-layers 8 \
  --no-train-lm-head \
  --learning-rate 1e-5 \
  --contrastive-weight 1.0 \
  --contrastive-wrong-candidates 3 \
  --eval-before \
  --eval-every 150 \
  --eval-lengths 2048,4096 \
  --eval-depths 0.9 \
  --eval-trials 8 \
  --distractors 24 \
  --wrong-candidates 3 \
  --value-kind status \
  --dtype bfloat16
```
