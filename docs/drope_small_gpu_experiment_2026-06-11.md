# DroPE Small-GPU Experiment — 2026-06-11

## Research premise

DroPE is not just "train NoPE at a longer context". The claimed recipe is:

1. Pretrain with positional embeddings such as RoPE.
2. Drop positional embeddings after pretraining.
3. Recalibrate briefly at the original pretraining context length.
4. Evaluate longer contexts zero-shot.

This matters because the earlier local run `drope_4k_full_adafactor_100s`
trained NoPE directly at 4K. The experiment below instead recalibrates at 2K,
matching the OpenEuroLLM proxy model's original context.

## Run

Host: `hot-poodle` (`ubuntu@77.87.121.41`), 1 x NVIDIA L4.

Run directory:

```text
/home/ubuntu/birger/openeuro-longctx-datamix/runs/local_ctx/drope_2k_recal_200s_codex
```

Command shape:

```bash
python scripts/train_hf_longctx.py \
  --model openeurollm/datamix-2b-80-20 \
  --mix-json data/local_stream_subset_4096_artifact/mix/data_mix.json \
  --output-dir runs/local_ctx/drope_2k_recal_200s_codex \
  --rope-method drope_nope \
  --target-context 8192 \
  --seq-len 2048 \
  --tiers 16k_plus,4_16k,under4k \
  --steps 200 \
  --learning-rate 2e-5 \
  --optimizer adafactor \
  --eval-every 50 \
  --eval-batches 4 \
  --save-final
```

Total tokens: 409,600. Runtime: about 6.4 minutes.

## Loss results

| run | training length | tokens | final eval loss |
| --- | ---: | ---: | ---: |
| `yarn_4k_full_adafactor_100s` | 4K | 409,600 | 8.1689 |
| `linear_4k_full_adafactor_100s` | 4K | 409,600 | 8.2714 |
| `drope_2k_recal_200s_codex` | 2K | 409,600 | 8.4173 |
| `longrope_ramp_4k_full_adafactor_100s` | 4K | 409,600 | 8.5630 |
| `drope_4k_full_adafactor_100s` | 4K | 409,600 | 9.2102 |

Faithful 2K DroPE recalibration is much better than the earlier 4K NoPE-style
attempt, but still behind YaRN and linear interpolation on held-out loss.

## NIAH retrieval smoke

Forced-choice base-LM NIAH, languages `en sv`, depths `0.0 0.5 1.0`,
5 trials per cell.

| run | en 2K | en 4K | en 8K | sv 2K | sv 4K | sv 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `drope_2k_recal_200s_codex` | 0.267 | 0.333 | 0.267 | 0.200 | 0.133 | 0.267 |
| `yarn_4k_full_adafactor_100s` | 1.000 | 0.667 | n/a | 1.000 | 0.667 | n/a |

The DroPE model does not recover strong native-context retrieval under this
budget. Its 8K scores are not worse than its 2K scores, but that is because all
three lengths are weak.

## Anti-cropping trace follow-up

I also ran a YaRN 8K LoRA continuation with a small anti-cropping trace mix.
The idea was to teach the model that information near the beginning, middle,
and end of long contexts can all be answer-critical, instead of relying only on
plain long-document language modeling loss.

Run directory:

```text
/home/ubuntu/birger/openeuro-longctx-datamix/runs/local_ctx/yarn_8k_lora_anticrop10_100s_codex
```

Setup:

- Base: `openeurollm/datamix-2b-80-20`
- Context: YaRN to 8192, training `seq_len=8192`
- Adapter: LoRA attention adapter
- Trace mix: `data/anti_cropping_traces_8k_en_sv_de_fr_256.jsonl`
- Trace weight: `0.10`
- Steps: 100

Loss improved over the earlier plain YaRN 8K LoRA run:

| run | final eval loss |
| --- | ---: |
| `yarn_8k_lora_longdocs_100s` | 10.0046 |
| `yarn_8k_lora_anticrop10_100s_codex` | 9.6904 |

PEFT-aware NIAH eval, languages `en sv`, depths `0.0 0.5 1.0`, 2 trials per
cell:

| run | en 2K | en 4K | en 8K | sv 2K | sv 4K | sv 8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yarn_8k_lora_anticrop10_100s_codex` | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.500 |

This is encouraging for loss and native/medium-context retrieval, but it still
does not solve full 8K retrieval. The failures are concentrated at the front
and middle needle placements; end-of-context retrieval was 2/2 in both
languages.

## Interpretation

This experiment supports three conclusions:

1. The faithful DroPE recipe is worth distinguishing from the earlier 4K NoPE
   run. It recovers loss much better.
2. With only 409K recalibration tokens on the OpenEuroLLM 2B proxy, DroPE does
   not recover enough retrieval behavior to beat YaRN.
3. For a production 9B long-context path, YaRN remains the safer near-term
   choice; LongRoPE/search remains the higher-upside route if we can afford the
   search. DroPE needs either a larger recalibration budget, Q/K norm variants,
   or a more faithful implementation from Sakana's repo before cluster scale.

## Caveat

The tokenizer load emits the known Mistral regex warning for saved local
tokenizers. It affects these local run-tokenizers uniformly, but it should be
fixed before a larger Mattermost-derived benchmark.

## Sources

- DroPE blog: https://pub.sakana.ai/DroPE/
- DroPE paper: https://arxiv.org/abs/2512.12167
- YaRN paper: https://arxiv.org/abs/2309.00071
- LongRoPE paper: https://arxiv.org/abs/2402.13753
- Position Interpolation paper: https://arxiv.org/abs/2306.15595
