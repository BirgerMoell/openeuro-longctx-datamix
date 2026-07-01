# Natural 128K/256K Dataset

This note defines the companion dataset for the 128K rescue run and a 256K
extension stage:

`birgermoell/oellm-longctx-tokenized-natural-128k-256k-v1`

It is deliberately separate from
`birgermoell/oellm-longctx-tokenized-streamed-all-v2` so FinePDFs-Edu remains an
immutable artifact and code/books/arXiv can be weighted explicitly.

## Source Families

| Prefix | Share | Role |
| --- | ---: | --- |
| `natural_code_repo_pack_128k` | 25% | Cross-file references, configs, tests, docs, symbol lookup |
| `natural_books_concat_128k` | 15% | Long narrative/entity state and delayed callbacks |
| `natural_arxiv_full_concat_128k` | 10% | Technical structure, sections, equations, appendices, cross-references |
| `natural_code_repo_pack_256k` | 25% | Same as code 128K, but with longer repo/bundle span |
| `natural_books_concat_256k` | 15% | Longer book-scale state for 256K extension |
| `natural_arxiv_full_concat_256k` | 10% | Longer technical bundles for 256K extension |

The production recipe is
[`configs/natural_128k_dataset_recipe.json`](../configs/natural_128k_dataset_recipe.json).

## Build

```bash
longctx natural-pack \
  --recipe configs/natural_128k_dataset_recipe.json \
  --output-dir data/natural128k256k/raw \
  --target-tokens 131072,262144 \
  --min-fill-ratio 0.85
```

For The Stack v2, first hydrate the SWH file contents into one JSONL row per
repository:

```json
{
  "repo_name": "owner/repo",
  "snapshot_id": "swh:1:snp:...",
  "files": [
    {
      "path": "README.md",
      "language": "Markdown",
      "content": "...",
      "is_vendor": false,
      "is_generated": false
    }
  ]
}
```

The packer writes strict tokenizer-ready rows:

```json
{"text": "...", "token_count": 131072}
{"text": "...", "token_count": 262144}
```

Provenance and source-family metadata live under
`data/natural128k256k/raw/manifests/`.

## Tokenize And Upload

```bash
longctx tokenize \
  --input-dir data/natural128k256k/raw \
  --output-dir data/natural128k256k/bin \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /path/to/openeurollm/tokenizer-256k \
  --vocab-size 262144 \
  --workers 16 \
  --megatron-path "$MEGATRON_LM"

longctx mix \
  --bin-dir data/natural128k256k/bin \
  --mix-dir data/natural128k256k/mix \
  --weights natural_code_repo_pack_128k=0.25,natural_books_concat_128k=0.15,natural_arxiv_full_concat_128k=0.10,natural_code_repo_pack_256k=0.25,natural_books_concat_256k=0.15,natural_arxiv_full_concat_256k=0.10

longctx artifacts pack \
  --bin-dir data/natural128k256k/bin \
  --mix-dir data/natural128k256k/mix \
  --output-dir data/hf_artifacts/oellm-longctx-tokenized-natural-128k-256k-v1 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /path/to/openeurollm/tokenizer-256k \
  --vocab-size 262144 \
  --notes "Natural 128K/256K long-context continuation data: repo-packed code, books, full arXiv."

longctx artifacts upload \
  --folder data/hf_artifacts/oellm-longctx-tokenized-natural-128k-256k-v1 \
  --repo-id birgermoell/oellm-longctx-tokenized-natural-128k-256k-v1 \
  --private
```

## Smoke Test

```bash
longctx natural-pack \
  --recipe configs/natural_pack_smoke_recipe.json \
  --output-dir /tmp/oellm-natural-pack-smoke \
  --target-tokens 80,160 \
  --min-fill-ratio 0.5 \
  --allow-short \
  --overwrite
```

Expected output: six JSONL files, one per source family and target length, plus
`manifests/natural_pack_manifest.json`.
