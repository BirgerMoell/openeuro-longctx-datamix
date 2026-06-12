# Tiny Superlong Context Smoke

This is a small-GPU plumbing test for the 2M-context plan. It does not try to
prove OpenEuroLLM quality. It checks that the training and eval machinery can:

- use large RoPE position IDs up to a 2M span
- train on retrieval traces with PoSE-style position offsets
- run forced-choice retrieval eval at longer physical context lengths
- record memory, config, metrics, and final eval artifacts

Script:

```bash
scripts/tiny_superlong_smoke.py
```

The model is a tiny byte-level RoPE transformer created from scratch, so this
does not need a Hugging Face model download or tokenizer cache.

## Fast CPU Smoke

```bash
python scripts/tiny_superlong_smoke.py \
  --output-dir /tmp/tiny_superlong_smoke \
  --seq-len 128 \
  --max-position 2097152 \
  --train-steps 2 \
  --batch-size 1 \
  --d-model 64 \
  --layers 2 \
  --heads 4 \
  --eval-every 1 \
  --eval-lengths 128,256 \
  --eval-depths 0.05,0.9 \
  --eval-trials 1 \
  --dtype float32
```

## Small-GPU Sanity Run

This should fit comfortably on a single L4-class GPU.

```bash
python scripts/tiny_superlong_smoke.py \
  --output-dir runs/tiny_superlong/l4_8k_pose2m \
  --seq-len 8192 \
  --max-position 2097152 \
  --position-strategy pose_offset \
  --train-steps 200 \
  --batch-size 1 \
  --grad-accum 1 \
  --d-model 256 \
  --layers 4 \
  --heads 4 \
  --eval-every 25 \
  --eval-lengths 8192,16384,32768 \
  --eval-depths 0.05,0.5,0.9 \
  --eval-trials 4 \
  --dtype bfloat16
```

## Ambitious L4 Run

Try this only after the 8K run succeeds.

```bash
python scripts/tiny_superlong_smoke.py \
  --output-dir runs/tiny_superlong/l4_16k_pose2m \
  --seq-len 16384 \
  --max-position 2097152 \
  --position-strategy pose_far \
  --train-steps 100 \
  --batch-size 1 \
  --d-model 256 \
  --layers 4 \
  --heads 4 \
  --eval-every 20 \
  --eval-lengths 16384,32768,65536 \
  --eval-depths 0.05,0.5,0.9 \
  --eval-trials 2 \
  --dtype bfloat16
```

## What Counts As Success

- The run finishes without OOM.
- `run_config.json`, `metrics.jsonl`, and `final_eval.json` are written.
- `max_memory_gb` is recorded on CUDA.
- Accuracy is not the main criterion for the shortest smoke, but the forced-choice
  eval should execute at all requested lengths.

If the 16K/64K eval OOMs, keep the 8K training run and reduce eval lengths. The
point is to test the superlong-position path cheaply before spending Leonardo
time.
