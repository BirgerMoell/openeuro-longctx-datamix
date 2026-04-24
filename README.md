# openeuro-longctx-datamix

**A CLI that turns [`HuggingFaceFW/finepdfs-edu`](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) into a long-context data mix for [OpenEuroLLM](https://openeurollm.eu) / [NVIDIA-Megatron-LM (OpenEuroLLM fork)](https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM).**

PDF-derived corpora are naturally long: unlike web text, documents routinely exceed 10K–100K tokens, which is exactly what you need for **context extension** (RoPE scaling / YaRN / LongRoPE continued pretraining). This tool:

1. Shards and downloads FinePDFs-Edu by language.
2. Streams parquet → JSONL in bounded RAM.
3. Optionally filters to long-only documents (`--min-tokens 8192` etc.).
4. Shells out to the Megatron fork's `tools/preprocess_data.py` to tokenize into `.bin`/`.idx`.
5. Emits a ready-to-paste **weighted `--data-path` block** with tempered sampling so low-resource languages aren't drowned out by English.

> **Note for AI coding agents.** This README is the source of truth. Read [`AGENTS.md`](AGENTS.md) for a concise, agent-oriented runbook. Machine-readable invariants live under the [`<!-- SPEC -->`](#spec) sections below and should be preserved when editing the tool.

---

## Language coverage

**35 of 38 OpenEuroLLM target languages are included** (all data from FinePDFs-Edu). The mapping `ISO 639-1 → FinePDFs folder` lives in [`src/longctx/languages.py`](src/longctx/languages.py).

<!-- SPEC: language-coverage -->
| Group | ISO 639-1 codes | Count |
| --- | --- | --- |
| EU official (present in FinePDFs-Edu) | `bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro sk sl sv` | **22** |
| Additional European | `bs ca cy eu gl is mk no ru sr tr uk` | **13** |
| **Total covered** | — | **35** |
| **Missing from FinePDFs-Edu** | `ga` (Irish Gaelic), `sq` (Albanian), `lb` (Luxembourgish) | **3** |

For the missing languages, bring your own JSONL (HPLT, CulturaX, Wikipedia) in the same schema (`{"text": "..."}`) and drop it into `data/megatron/` — the rest of the pipeline will pick them up.
<!-- /SPEC -->

---

## Install

```bash
git clone https://github.com/birgermoell/openeuro-longctx-datamix
cd openeuro-longctx-datamix
python -m venv .venv && source .venv/bin/activate
pip install -e .
longctx --help
```

Requires Python ≥ 3.10. Runtime deps: `huggingface_hub`, `pyarrow`, `tqdm`.

### Point at your Megatron fork (needed only for `tokenize`)

```bash
git clone https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM
export MEGATRON_LM=$PWD/NVIDIA-Megatron-LM
```

---

## End-to-end quickstart

```bash
# 1. Plan disk usage (safe dry-run; no downloads)
longctx estimate --languages bg,de,fr,sv,fi,en

# 2. Sample download (1 shard per language) — fast, ~30 GB for all 35 langs
longctx download --sample --shards 1 \
                 --languages bg,de,fr,sv,fi,en \
                 --output-dir data/raw

# 3. Parquet → JSONL (streams row groups; ~300 MB RAM peak)
longctx convert --output-dir data/raw --megatron-dir data/megatron

# 4. Keep long docs only (core context-extension step)
longctx filter-long --megatron-dir data/megatron \
                    --long-dir data/long \
                    --min-tokens 8192

# 5. Tokenize with Megatron's preprocess_data.py → .bin/.idx
longctx tokenize --input-dir data/long \
                 --output-dir data/bin \
                 --tokenizer-type HuggingFaceTokenizer \
                 --tokenizer-model meta-llama/Llama-3.1-8B \
                 --vocab-size 128256 \
                 --workers 16

# 6. Emit a weighted data mix (tempered sampling, α=0.3)
longctx mix --bin-dir data/bin --mix-dir data/mix --alpha 0.3
```

The last step writes `data/mix/data_path.args`, which you paste into Megatron:

```bash
DATA_PATH="$(cat data/mix/data_path.args)"

bash $MEGATRON_LM/examples/llama/train_llama3_8b_context_extension.sh \
    checkpoints/llama3_8b_32k_yarn \
    tensorboard/llama3_8b_32k_yarn \
    meta-llama/Llama-3.1-8B \
    "$DATA_PATH" \
    yarn
```

See [`examples/train_openeurollm_longctx.sh`](examples/train_openeurollm_longctx.sh) for a worked launcher.

---

## Subcommand reference

<!-- SPEC: cli -->
```
longctx estimate     --languages bg,fr,...  --output-dir data/raw
longctx download     --languages bg,fr,...  --output-dir data/raw
                     [--sample] [--shards N]
longctx convert      --output-dir data/raw  --megatron-dir data/megatron
                     [--max-docs-per-shard N] [--keep-token-count]
longctx filter-long  --megatron-dir data/megatron --long-dir data/long
                     --min-tokens 8192 [--max-tokens 131072]
                     [--languages bg,...]
longctx tokenize     --input-dir data/long  --output-dir data/bin
                     --tokenizer-type HuggingFaceTokenizer
                     --tokenizer-model meta-llama/Llama-3.1-8B
                     [--vocab-size 128256] [--megatron-path $MEGATRON_LM]
                     [--workers N] [--partitions K]
                     [--append-eod] [--dry-run]
longctx mix          --bin-dir data/bin --mix-dir data/mix
                     [--alpha 0.3] [--floor 0.005]
                     [--languages bg,...] [--suffix _text_document]
longctx run          --languages bg,...  [--sample] [--filter-long]
                     --output-dir data/raw --megatron-dir data/megatron
                     [--long-dir data/long --min-tokens 8192]
```
<!-- /SPEC -->

---

## Output layout

<!-- SPEC: directory-layout -->
```
data/
├── raw/                        # from `download`
│   ├── manifest.json
│   └── <lc>/<lc>_<script>_<shard>.parquet
├── megatron/                   # from `convert`
│   ├── conversion_summary.json
│   └── <lc>.jsonl              # {"text": "...", "token_count": N}
├── long/                       # from `filter-long`
│   ├── filter_summary.json
│   └── <lc>.jsonl
├── bin/                        # from `tokenize`
│   ├── tokenize_summary.json
│   └── <lc>_text_document.{bin,idx}
└── mix/                        # from `mix`
    ├── data_mix.json           # machine-readable weights
    ├── data_mix.txt            # human-readable table
    └── data_path.args          # paste after `--data-path`
```

`<lc>` = ISO 639-1 language code (e.g. `bg`, `de`, `fr`). `<script>` = ISO 15924 (`Latn`, `Cyrl`, `Grek`).
<!-- /SPEC -->

---

## The mix math

<!-- SPEC: mixing-policy -->
Let `tokens_i` be the bin size (in bytes, which ≈ tokens for Megatron binaries) of language `i`. We compute:

```
p_i = tokens_i / Σ tokens_j                       # natural frequency
w_i ∝ p_i^alpha                                   # tempered sampling
w_i = max(w_i, floor)                             # protect low-resource langs
w_i /= Σ w_j                                      # renormalize
```

| α     | Behaviour                               | Use case                                 |
| ----- | --------------------------------------- | ---------------------------------------- |
| `1.0` | Natural frequency                       | English-dominant runs (not recommended)  |
| `0.5` | Moderate upsampling                     | Balanced multilingual pretraining        |
| `0.3` | **Default — strong upsampling**         | Multilingual continued pretraining       |
| `0.0` | Uniform                                 | Tiny eval sets / rapid ablations         |

`floor = 0.005` guarantees each included language sees at least ~0.5% of batches regardless of size. Tune via `--alpha` and `--floor`.
<!-- /SPEC -->

---

## How this wires into the OpenEuroLLM Megatron fork

<!-- SPEC: megatron-integration -->
The fork's training launcher (`examples/llama/train_llama3_8b_context_extension.sh`) accepts a single `DATA_PATH` argument that is forwarded to `pretrain_gpt.py` as `--data-path`. Megatron-LM parses that as either:

* a single prefix: `--data-path /path/to/bin/en_text_document`, or
* a weighted mixture: `--data-path 0.5 /path/a 0.3 /path/b 0.2 /path/c`.

`longctx mix` emits the weighted form in `data_path.args`. The prefixes it writes are **absolute paths** to `<bin_dir>/<lc>_text_document` (no `.bin`/`.idx` suffix), matching Megatron's expectation.

Tokenizer settings in `longctx tokenize` mirror the fork's `_add_tokenizer_args`:

* `--tokenizer-type` ∈ `{HuggingFaceTokenizer, SentencePieceTokenizer, GPTSentencePieceTokenizer, Llama2Tokenizer, TikTokenizer, NullTokenizer}`
* `--tokenizer-model` = HF repo id or local path
* `--vocab-size` required for some types (e.g. `128256` for Llama-3 HF tokenizer)

`--append-eod` (on by default) inserts `<eod>` between documents, which is what you want for causal LM pretraining.
<!-- /SPEC -->

---

## Troubleshooting

<!-- SPEC: troubleshooting -->
| Symptom | Cause | Fix |
| --- | --- | --- |
| `preprocess_data.py not found` | `$MEGATRON_LM` not set or repo not cloned | `git clone https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM && export MEGATRON_LM=$PWD/NVIDIA-Megatron-LM` |
| Rate-limited HF downloads | No `HF_TOKEN` | `export HF_TOKEN=hf_...` before `longctx download` |
| `No manifest at data/raw/manifest.json` | `convert` run before `download` | Run `longctx download` first |
| `No tokenized datasets found` in `mix` | Tokenize step didn't produce `.bin`/`.idx` | Check `tokenize_summary.json`; verify tokenizer-model path |
| Huge English share (> 30%) | α too high | Re-run `longctx mix --alpha 0.3` (or 0.1 to push further) |
| Memory blowup in `convert` | Parquet has oversized row groups | Convert with `--max-docs-per-shard N` to cap; file an issue |
| `filter-long` drops ~100% of docs | `token_count` missing from JSONL | Re-run `convert` with `--keep-token-count` (default), or lower `--min-tokens` |
<!-- /SPEC -->

---

## What "context extension" means here

<!-- SPEC: context-extension -->
Long-context training = take a model pretrained on short sequences (typically 4K–8K), adjust the positional-embedding scheme (YaRN, LongRoPE, NTK, PI), and do a **short continued pretraining run** on sequences at the target length (32K/64K/128K+). The OpenEuroLLM Megatron fork supports `yarn` and `longrope` via `--position-embedding-type`.

This data mix is designed for that phase:

* **Source = FinePDFs-Edu**: PDF-derived, natively long, educational quality filter.
* **Hard filter at `--min-tokens`**: default 8192. Raise to target context length (e.g. 32768) for 32K runs so every sample is long enough.
* **Tempered sampling**: the 24 EU official languages should all show up in the batch stream. Without it, English crowds them out during context extension and you get lopsided gains.
* **Upsampling floor**: keeps Maltese / Icelandic / Welsh visible even at large batch sizes.
<!-- /SPEC -->

---

## Extending

**Add a new data source.** Write an adapter under `src/longctx/sources/` that produces the same JSONL schema under `data/megatron/<lc>.jsonl`. All downstream steps are source-agnostic.

**Add the three missing languages.** Provide `data/megatron/{ga,sq,lb}.jsonl` from HPLT or CulturaX; `filter-long`, `tokenize`, `mix` will treat them identically.

**Swap tokenizer.** `longctx tokenize --tokenizer-type SentencePieceTokenizer --tokenizer-model path/to/tokenizer.model` — works for any tokenizer the Megatron fork accepts.

---

## License

Apache 2.0. FinePDFs-Edu is subject to its own license on the Hugging Face Hub.
