# Tiny Superlong Small-GPU Results — 2026-06-12

Host: `hot-poodle` (`ubuntu@77.87.121.41`)

GPUs: `2 x NVIDIA L4`, 23 GB each. Both GPUs were idle before launch.

The remote repo had uncommitted local changes that blocked `git pull`, so the
standalone script was copied to:

```text
/home/ubuntu/birger/tiny_superlong_smoke.py
```

This avoided touching the remote worktree.

## Run 1: 8K Physical, 2M Position Span

Output:

```text
/home/ubuntu/birger/runs/tiny_superlong/l4_8k_pose2m
```

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/birger/tiny_superlong_smoke.py \
  --output-dir /home/ubuntu/birger/runs/tiny_superlong/l4_8k_pose2m \
  --seq-len 8192 \
  --max-position 2097152 \
  --position-strategy pose_offset \
  --train-steps 200 \
  --d-model 256 \
  --layers 4 \
  --heads 4 \
  --eval-lengths 8192,16384,32768 \
  --eval-depths 0.05,0.5,0.9 \
  --eval-trials 4 \
  --dtype bfloat16
```

Result:

- Completed 200/200 steps.
- Tokens: 1,638,400.
- Runtime: about 33 seconds.
- Peak PyTorch memory: 0.50 GB.
- Artifacts written: `run_config.json`, `metrics.jsonl`, `run.log`,
  `final_eval.json`.
- Final eval executed at 8K, 16K, and 32K.

Final eval accuracy was noisy/chance-level, which is expected for a tiny random
byte-level model after a very short run. The purpose was plumbing, not model
quality.

## Run 2: 16K Physical, 2M Far-Position Span

Output:

```text
/home/ubuntu/birger/runs/tiny_superlong/l4_16k_pose2m
```

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/birger/tiny_superlong_smoke.py \
  --output-dir /home/ubuntu/birger/runs/tiny_superlong/l4_16k_pose2m \
  --seq-len 16384 \
  --max-position 2097152 \
  --position-strategy pose_far \
  --train-steps 100 \
  --d-model 256 \
  --layers 4 \
  --heads 4 \
  --eval-lengths 16384,32768,65536 \
  --eval-depths 0.05,0.5,0.9 \
  --eval-trials 2 \
  --dtype bfloat16
```

Result:

- Completed 100/100 steps.
- Tokens: 1,638,400.
- Runtime: about 33 seconds.
- Peak PyTorch memory: 0.97 GB.
- Artifacts written: `run_config.json`, `metrics.jsonl`, `run.log`,
  `final_eval.json`.
- Final eval executed at 16K, 32K, and 65K.

## Conclusion

The tiny superlong path works mechanically on the small GPU box:

- large 2M-span position IDs are accepted
- PoSE-style offset/far-position training runs
- forced-choice retrieval eval runs beyond the training length
- the memory footprint is tiny on L4

Next small-GPU step: run the same script with a somewhat larger tiny model
(`d_model=512`, `layers=6`) or run the HF 0.5B/2B proxy path if we want a
quality signal rather than a pure plumbing signal.
