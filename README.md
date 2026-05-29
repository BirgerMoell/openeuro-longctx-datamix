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

**All 38 OpenEuroLLM target languages are reachable** — 35 from FinePDFs-Edu (primary source), and the 3 missing ones (`ga`, `sq`, `lb`) via the `sources` subcommand which streams from HPLT or CulturaX.

<!-- SPEC: language-coverage -->
| Group | ISO 639-1 codes | Count | Source |
| --- | --- | --- | --- |
| EU official (FinePDFs-Edu) | `bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro sk sl sv` | **22** | `download` |
| Additional European (FinePDFs-Edu) | `bs ca cy eu gl is mk no ru sr tr uk` | **13** | `download` |
| Missing from FinePDFs-Edu | `ga sq lb` | **3** | `sources fetch` (HPLT / CulturaX) |
| **Total reachable** | — | **38** | — |
<!-- /SPEC -->

The mapping `ISO 639-1 → FinePDFs folder` lives in [`src/longctx/languages.py`](src/longctx/languages.py); adapter-specific mappings live under [`src/longctx/sources/`](src/longctx/sources).

### Uploaded tokenized artifact snapshot

The public Hugging Face dataset [`birgermoell/oellm-longctx-tokenized-streamed-all-v2`](https://huggingface.co/datasets/birgermoell/oellm-longctx-tokenized-streamed-all-v2) currently contains the streamed Megatron artifacts under `runs/lc16k_full_20260507/`. The table below is computed from the uploaded `.bin`/`.idx` files and chunk manifests in that run. Sizes are tokenized artifact sizes in GiB, not compressed source parquet size.

Length buckets are based on the FinePDFs-Edu source `token_count` field:

| Bucket | Source document length |
| --- | --- |
| `<4k` | fewer than 4,000 source tokens |
| `4-16k` | 4,000 to fewer than 16,000 source tokens |
| `>=16k` | 16,000 or more source tokens |

<!-- SPEC: hf-tokenized-artifact-snapshot -->
| ISO | Language | Chunks | Tokenized GiB | `>=16k` chunks/GiB | `4-16k` chunks/GiB | `<4k` chunks/GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `bg` | Bulgarian | 15 | 4.6 | 4 / 3.4 | 4 / 0.9 | 7 / 0.3 |
| `bs` | Bosnian | 9 | 3.7 | 3 / 3.0 | 3 / 0.5 | 3 / 0.2 |
| `ca` | Catalan | 20 | 4.3 | 4 / 2.6 | 4 / 1.1 | 12 / 0.6 |
| `cs` | Czech | 53 | 15.2 | 8 / 10.5 | 14 / 3.3 | 31 / 1.5 |
| `cy` | Welsh | 3 | 0.3 | 1 / 0.1 | 1 / 0.1 | 1 / 0.0 |
| `da` | Danish | 22 | 6.3 | 4 / 4.2 | 4 / 1.4 | 14 / 0.7 |
| `de` | German | 313 | 62.7 | 45 / 34.4 | 45 / 16.8 | 223 / 11.6 |
| `el` | Greek | 20 | 8.7 | 5 / 6.7 | 5 / 1.3 | 10 / 0.7 |
| `en` | English | 302 | 109.9 | 20 / 63.0 | 60 / 29.5 | 222 / 17.3 |
| `es` | Spanish | 254 | 106.1 | 54 / 76.6 | 54 / 20.7 | 146 / 8.8 |
| `et` | Estonian | 7 | 1.5 | 2 / 1.0 | 2 / 0.3 | 3 / 0.2 |
| `eu` | Basque | 6 | 1.1 | 2 / 0.8 | 2 / 0.2 | 2 / 0.1 |
| `fi` | Finnish | 20 | 7.3 | 5 / 5.7 | 5 / 1.1 | 10 / 0.5 |
| `fr` | French | 129 | 33.6 | 21 / 19.4 | 21 / 8.6 | 87 / 5.6 |
| `gl` | Galician | 4 | 0.9 | 1 / 0.6 | 1 / 0.2 | 2 / 0.1 |
| `hr` | Croatian | 16 | 6.1 | 4 / 4.8 | 4 / 0.9 | 8 / 0.4 |
| `hu` | Hungarian | 38 | 18.2 | 10 / 15.0 | 10 / 2.3 | 18 / 0.8 |
| `is` | Icelandic | 7 | 0.9 | 2 / 0.7 | 2 / 0.2 | 3 / 0.1 |
| `it` | Italian | 146 | 38.1 | 25 / 23.7 | 25 / 8.7 | 96 / 5.8 |
| `lt` | Lithuanian | 11 | 3.3 | 3 / 2.5 | 3 / 0.6 | 5 / 0.2 |
| `lv` | Latvian | 7 | 2.1 | 2 / 1.5 | 2 / 0.4 | 3 / 0.2 |
| `mk` | Macedonian | 6 | 1.0 | 2 / 0.8 | 2 / 0.1 | 2 / 0.0 |
| `mt` | Maltese | 3 | 0.4 | 1 / 0.2 | 1 / 0.1 | 1 / 0.0 |
| `nl` | Dutch | 74 | 18.3 | 13 / 10.5 | 13 / 4.9 | 48 / 2.9 |
| `no` | Norwegian Bokmal | 16 | 4.9 | 4 / 3.4 | 4 / 1.0 | 8 / 0.4 |
| `pl` | Polish | 85 | 28.1 | 15 / 20.1 | 25 / 5.7 | 45 / 2.3 |
| `pt` | Portuguese | 118 | 47.9 | 25 / 32.5 | 43 / 11.9 | 50 / 3.5 |
| `ro` | Romanian | 28 | 16.5 | 7 / 13.2 | 7 / 2.4 | 14 / 0.9 |
| `ru` | Russian | 148 | 63.2 | 37 / 47.9 | 37 / 11.0 | 74 / 4.3 |
| `sk` | Slovak | 20 | 6.9 | 4 / 4.8 | 6 / 1.5 | 10 / 0.6 |
| `sl` | Slovenian | 12 | 2.3 | 3 / 1.6 | 3 / 0.4 | 6 / 0.3 |
| `sr` | Serbian | 12 | 7.2 | 4 / 6.3 | 4 / 0.6 | 4 / 0.2 |
| `sv` | Swedish | 33 | 12.2 | 7 / 8.5 | 7 / 2.5 | 19 / 1.2 |
| `tr` | Turkish | 20 | 9.1 | 5 / 6.8 | 5 / 1.8 | 10 / 0.6 |
| `uk` | Ukrainian | 28 | 17.8 | 7 / 14.2 | 7 / 2.8 | 14 / 0.8 |
| **Total** | **35 languages** | **2,005** | **670.4** | **359 / 450.8** | **435 / 145.8** | **1,211 / 73.8** |
<!-- /SPEC -->

This snapshot includes the 35 FinePDFs-Edu languages. The three languages not present in FinePDFs-Edu (`ga`, `sq`, `lb`) still need to be sourced through `longctx sources fetch` before they can be included in an equivalent tokenized artifact.

### Filling in the missing three

```bash
# HPLT is ungated and covers all 38 targets — one command gets ga/sq/lb:
longctx sources fetch --source hplt \
                      --languages ga,sq,lb \
                      --megatron-dir data/megatron

# CulturaX is gated; accept terms at https://huggingface.co/datasets/uonlp/CulturaX
# and `export HF_TOKEN=hf_...` first, then:
longctx sources fetch --source culturax \
                      --languages ga,sq,lb \
                      --megatron-dir data/megatron
```

After that, `filter-long`, `tokenize`, and `mix` work unchanged — all three missing languages flow through the same long-context pipeline as the 35 FinePDFs-Edu ones.

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

## Running on LUMI (YaRN context extension)

Two ready-to-submit SLURM jobs live under `lumi/slurm/`. They are identical in every way (model, parallelism, optimizer, 1 000 iterations) so their results are directly comparable — the only difference is the data mix.

### Prerequisite: data on LUMI

Both jobs read data that is already available at:

| Dataset | Path on LUMI |
| --- | --- |
| Multilingual tokenized tiers (31 languages) | `/flash/project_462000963/bmoell/data_tokenized_hf_multilingual/merged/` |
| MegaMath-full (oellm-v1-256k tokenizer) | `/scratch/project_462000963/preprocessed/oellm-v1-256k/LLM360/MegaMath-full/` |
| StarCoder 10%-sample (oellm-v1-256k tokenizer) | `/scratch/project_462000963/preprocessed/oellm-v1-256k/catalogue/starcoder-10p-sample/` |

### Variant A — multilingual only (baseline)

```bash
sbatch lumi/slurm/yarn_multilingual_v2_1k.sbatch
```

Data mix: 31-language FinePDFs-Edu tier mix, weights 75 % `≥16k` / 15 % `4–16k` / 10 % `<4k`.  
Output: `/flash/project_462000963/bmoell/yarn-multilingual-v2-1k/`

### Variant B — multilingual + code + math

```bash
sbatch lumi/slurm/yarn_multilingual_v2_1k_code_math.sbatch
```

Data mix:

| Source | Share | Details |
| --- | --- | --- |
| Multilingual (31 langs × 3 tiers) | **80 %** | Same tier weights as Variant A |
| MegaMath-full | **10 %** | 5 subsets weighted by file size: `megamath-web` (905 GB), `megamath-text-code-block` (181 GB), `megamath-web-pro` (54 GB), `megamath-qa` (26 GB), `megamath-translated-code` (26 GB) |
| StarCoder 10%-sample | **10 %** | Single entry, 104 GB |

Output: `/flash/project_462000963/bmoell/yarn-multilingual-v2-1k-code-math/`

The `data_path.args` file is regenerated at job start (written to `$HF_DATA_DIR/data_path_v2_code_math.args`) so no manual setup is needed before submitting.

### Running both at once (A/B comparison)

```bash
# Submit both from your local machine — they run independently
sbatch lumi/slurm/yarn_multilingual_v2_1k.sbatch
sbatch lumi/slurm/yarn_multilingual_v2_1k_code_math.sbatch

# Monitor
ssh bmoell@lumi.csc.fi "squeue -u bmoell"
```

### Changing the project allocation

Both files currently bill to `project_462000963`. Once `project_462002530` is active, update the `--account` line in both sbatch files:

```bash
sed -i 's/--account=project_462000963/--account=project_462002530/' \
    lumi/slurm/yarn_multilingual_v2_1k.sbatch \
    lumi/slurm/yarn_multilingual_v2_1k_code_math.sbatch
```

---

## Subcommand reference

<!-- SPEC: cli -->
```
longctx estimate         --languages bg,fr,...  --output-dir data/raw
longctx download         --languages bg,fr,...  --output-dir data/raw
                         [--sample] [--shards N]
longctx convert          --output-dir data/raw  --megatron-dir data/megatron
                         [--max-docs-per-shard N] [--keep-token-count]
longctx filter-long      --megatron-dir data/megatron --long-dir data/long
                         --min-tokens 8192 [--max-tokens 131072]
                         [--languages bg,...]
longctx sources list
longctx sources fetch    --source hplt|culturax  --languages ga,sq,lb
                         --megatron-dir data/megatron
                         [--sample] [--max-docs N] [--overwrite]
longctx tokenize         --input-dir data/long  --output-dir data/bin
                         --tokenizer-type HuggingFaceTokenizer
                         --tokenizer-model meta-llama/Llama-3.1-8B
                         [--vocab-size 128256] [--megatron-path $MEGATRON_LM]
                         [--workers N] [--partitions K]
                         [--append-eod] [--dry-run]
longctx mix              --bin-dir data/bin --mix-dir data/mix
                         [--alpha 0.3] [--floor 0.005]
                         [--languages bg,...] [--suffix _text_document]
                         [--weights name=0.5,other=0.5]
longctx stream-upload    --repo-id birgermoell/<dataset>
                         --languages mt,sv --tokenizer-model <tok>
                         --megatron-path $MEGATRON_LM
                         [--shards N|--all-shards] [--chunk-docs N]
longctx artifacts pack   --bin-dir data/bin --mix-dir data/mix
                         --output-dir data/hf_artifacts/<name>
longctx artifacts upload --folder data/hf_artifacts/<name>
                         --repo-id birgermoell/<dataset> [--private]
longctx artifacts download --repo-id birgermoell/<dataset>
                           --output-dir /flash/.../data_tokenized
longctx run              --languages bg,...  [--sample] [--filter-long]
                         --output-dir data/raw --megatron-dir data/megatron
                         [--long-dir data/long --min-tokens 8192]
```
<!-- /SPEC -->

### Source adapters

<!-- SPEC: source-adapters -->
| Source | HF dataset | Gated | Langs covered | Notes |
| --- | --- | --- | --- | --- |
| `hplt` | `HPLT/HPLT2.0_cleaned` | No | All 38 (incl. `ga sq lb`) | Primary for missing langs; uses ISO 639-3 + script codes internally |
| `culturax` | `uonlp/CulturaX` | **Yes** | All 38 | Accept terms + set `HF_TOKEN` |

Adapters write `{lc}.jsonl` in the canonical `{text, token_count}` schema into `--megatron-dir`, so every downstream step (`filter-long`, `tokenize`, `mix`) works identically to the FinePDFs-Edu path. See [`src/longctx/sources/`](src/longctx/sources/) to add new adapters.
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

For Jouni-compatible multilingual tier training, use explicit weights:

```bash
longctx mix --bin-dir data/tokenized_multilingual --mix-dir data/tokenized_multilingual/mix \
  --weights multilingual_16k_plus=0.5,multilingual_4_16k=0.3,multilingual_under4k=0.2
```

See [`docs/hf_tokenized_artifacts.md`](docs/hf_tokenized_artifacts.md) for the NVIDIA → Hugging Face → LUMI artifact handoff.

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
