# AGENTS.md — runbook for AI coding agents

This is a Python CLI package that produces a long-context multilingual data mix for the [OpenEuroLLM NVIDIA-Megatron-LM fork](https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM). Read `README.md` for the full spec — **this file is the fast-path summary**.

## Mental model

The pipeline is a strict DAG. Each step reads from the previous step's output dir. Directory layout is the contract — don't reinvent it.

```
download → data/raw/{manifest.json, <lc>/*.parquet}
convert  → data/megatron/{conversion_summary.json, <lc>.jsonl}
filter   → data/long/{filter_summary.json, <lc>.jsonl}
tokenize → data/bin/{tokenize_summary.json, <lc>_text_document.{bin,idx}}
mix      → data/mix/{data_mix.json, data_mix.txt, data_path.args}
```

## Invariants (do not break)

1. **Schema of JSONL lines is `{"text": str, "token_count": int}`.** `filter-long` depends on `token_count`; losing it forces a re-tokenize pass.
2. **Megatron binary naming is `<lc>_text_document.{bin,idx}`.** Hardcoded by Megatron's `preprocess_data.py` when `--json-keys text`; `longctx mix` relies on the `_text_document` suffix.
3. **`data_path.args` contains absolute prefixes without the `.bin`/`.idx` suffix.** Megatron's `--data-path` parser demands this.
4. **Language codes are ISO 639-1.** The folder map `ISO 639-1 → ISO 639-3+script` lives in `src/longctx/languages.py`. Don't duplicate it.
5. **The three missing languages `ga`, `sq`, `lb` are not in FinePDFs-Edu.** Do not silently include them; surface them as missing.

## Adding a subcommand

1. Create `src/longctx/commands/<name>.py` exporting `cmd_<name>(args) -> None`.
2. Re-export it in `src/longctx/commands/__init__.py`.
3. Register the subparser in `src/longctx/cli.py`.
4. Update the `CLI` and `directory-layout` SPEC blocks in `README.md` if the layout changes.

## Adding a new source

1. Write `src/longctx/sources/<source>.py` exposing `NAME`, `DATASET_ID`, `GATED: bool`, `SUPPORTED: dict[iso_639_1, adapter_config]`, and `fetch(lang, output_path, *, sample=False, max_docs=None) -> dict`.
2. Register it in `src/longctx/sources/__init__.py` under `SOURCES`.
3. Downstream commands are source-agnostic — they read every `*.jsonl` under `data/megatron/`.
4. Don't modify `LANG_MAP` (FinePDFs-Edu specific); per-adapter mappings live on the adapter.

**Existing adapters** (both cover all 38 OpenEuroLLM target langs including `ga`, `sq`, `lb`):

| Name       | HF id                  | Gated | Config codes          |
|------------|------------------------|-------|-----------------------|
| `hplt`     | `HPLT/HPLT2.0_cleaned` | no    | ISO 639-3 + script    |
| `culturax` | `uonlp/CulturaX`       | yes   | ISO 639-1 pass-through|

## Running tests / smoke checks

```bash
pip install -e .
longctx --help
longctx estimate --languages mt,cy,is        # smallest langs, fast
longctx download --sample --shards 1 --languages mt
longctx convert
longctx filter-long --min-tokens 4096
longctx tokenize --dry-run \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.1-8B \
    --vocab-size 128256
```

`--dry-run` on `tokenize` prints the exact `preprocess_data.py` invocation without running it — use it to diff against README claims.

## Things that tempt agents but are wrong

- **Don't swap the sampling exponent without asking.** α=0.3 is a deliberate default for multilingual continued pretraining; changing it silently changes training dynamics.
- **Don't add a `--merge` step that concatenates JSONL across languages.** Megatron's MMapIndexedDataset mixer does weighted sampling at batch time; concatenating destroys that.
- **Don't tokenize with `--split-sentences`.** That's for BERT-style MLM, not causal LM pretraining.
- **Don't set `--append-eod=False`.** You want EOD between documents for next-token-prediction.
- **Don't commit `data/`.** It's in `.gitignore` for a reason — a full run produces >1 TB.

## Where to look first when something breaks

1. `data/raw/manifest.json` — did download actually finish?
2. `data/megatron/conversion_summary.json` — token counts per lang.
3. `data/bin/tokenize_summary.json` — did Megatron tokenize each file?
4. `data/mix/data_mix.json` — are weights sane? (no NaN, sum ≈ 1)
5. Fork script: `$MEGATRON_LM/examples/llama/train_llama3_8b_context_extension.sh`.
