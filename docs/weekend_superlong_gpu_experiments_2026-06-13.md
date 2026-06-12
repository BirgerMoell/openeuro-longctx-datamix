# Weekend Superlong GPU Experiments - 2026-06-13

These are the small-GPU experiments queued for the weekend on `hot-poodle`
(`ubuntu@77.87.121.41`, 2 x NVIDIA L4). They build on the 2M bridge smoke test
from `docs/superlong_2m_small_gpu_training_2026-06-12.md`.

## Main Hypotheses

1. Contrastive-only retrieval training should move the forced-choice metric more
   reliably than generative answer loss.
2. A depth curriculum (`0.9 -> 0.5 -> 0.05`) should work better than sampling all
   retrieval depths from step 1.
3. More trainable depth may matter: last 12 layers could beat last 8 layers.
4. Physical context length may matter: 4k training windows may transfer better to
   8k evaluation than 2k windows.
5. RoPE scaling type matters: YaRN should be compared against linear and dynamic
   scaling under the same task.
6. Random code retrieval is much harder than finite status-label retrieval and is
   the next realism step if status retrieval improves.

## Runner

```bash
scripts/run_weekend_superlong_experiments.sh
```

Run one queue per GPU:

```bash
RUN_ROOT=/home/ubuntu/birger/runs/hf_superlong/weekend_20260613 \
  bash scripts/run_weekend_superlong_experiments.sh 0

RUN_ROOT=/home/ubuntu/birger/runs/hf_superlong/weekend_20260613 \
  bash scripts/run_weekend_superlong_experiments.sh 1
```

Each experiment writes:

- `run_config.json`
- `before_eval.json`
- `metrics.jsonl`
- `final_eval.json`
- `exit_code.txt`
- `launcher.log`

## GPU 0 Queue

- `g0_yarn_status_depthsched_contrastive_only_2k_last8_seed2026061301`
- `g0_yarn_status_depthsched_mixed_temp025_2k_last8_seed2026061302`
- `g0_yarn_code_depth09_contrastive_only_2k_last8_seed2026061303`
- `g0_linear_status_depthsched_contrastive_only_2k_last8_seed2026061304`
- `g0_yarn_status_depthsched_contrastive_only_2k_last8_seed2026061311`

This queue focuses on safer 2k experiments: contrastive-only vs mixed loss,
status vs random-code retrieval, and YaRN vs linear scaling.

## GPU 1 Queue

- `g1_yarn_status_depthsched_contrastive_only_4k_last8_wrong3_seed2026061305`
- `g1_yarn_status_depthsched_contrastive_only_2k_last12_seed2026061306`
- `g1_dynamic_status_depthsched_contrastive_only_2k_last8_seed2026061307`
- `g1_yarn_code_depthsched_contrastive_only_2k_last12_seed2026061308`
- `g1_yarn_status_depthsched_contrastive_only_4k_last8_wrong2_seed2026061312`

This queue tests bigger but probed-safe variants: 4k physical context, last 12
trainable layers, dynamic RoPE scaling, and code retrieval with more trainable
depth.

## Memory Probes

Completed before launching:

- 4k, last 8, 3 wrong contrastive candidates: fits at about 20.7 GB.
- 2k, last 12, 3 wrong contrastive candidates: fits at about 16.1 GB.
- LM-head adaptation: OOM at about 22 GB, so it is excluded from the weekend queue.

The runner sets:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

to reduce fragmentation risk, especially for the 4k contrastive runs.
