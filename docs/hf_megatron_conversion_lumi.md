# HF ↔ Megatron checkpoint conversion on LUMI

How to convert between HuggingFace and Megatron-core (torch_dist) checkpoints for our Qwen3-dense 9B
(GQA) models on LUMI. Two directions, two different tools.

## Container (both directions)
```
/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
```
Run under `srun singularity exec -B /pfs -B /scratch -B /flash <container> bash -lc "..."`.
Always set `export CUDA_DEVICE_MAX_CONNECTIONS=1`.

---

## Direction 1 — HF → Megatron (import a base to extend)
**Tool: Megatron-Bridge.** Used to bring a HF base (e.g. prelude `iter_0124800`) into Megatron before
context extension.

- Bridge: `/flash/project_465002530/tools/Megatron-Bridge-LUMI`
- **PYTHONPATH gotcha (critical):** `python-packages` must come FIRST, else `ImportError:
  ProcessGroupCollection`.

```bash
BRIDGE=/flash/project_465002530/tools/Megatron-Bridge-LUMI
HF_MODEL=/scratch/.../prelude_hf/iter_0124800          # source HF dir
MEG_OUT=/scratch/.../longctx-extend/prelude_meg        # output Megatron dir

srun singularity exec -B /pfs,/scratch,/flash "$CONTAINER" bash -lc "
  export PYTHONPATH=$BRIDGE/python-packages:$BRIDGE/3rdparty/Megatron-LM:$BRIDGE/src
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  python $BRIDGE/examples/conversion/convert_checkpoints.py import \
    --hf-model $HF_MODEL --megatron-path $MEG_OUT
"
```
Reference sbatch: `longctx-extend/convert_prelude.sbatch`.
⚠️ Note `Megatron-Bridge-LUMI` lives on **`/flash`** (flaky/often full) — if unavailable, that's the
first thing to check.

---

## Direction 2 — Megatron → HF (export a trained checkpoint to publish/eval)
**Tool: the `conv/checkpoint/convert.py` converter** (`--loader core --saver qwen3`). Used after each
extension stage to produce an HF model for eval + HF upload.

- Converter: `/scratch/project_465002530/users/bmoell/conv/checkpoint/`
- Megatron-LM (for `--loader core`): `/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM`
- Tokenizer: `/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k`

```bash
CONV=/scratch/project_465002530/users/bmoell/conv
SHARED_MEG=/scratch/project_465002530/users/luomajou/oellm-test/NVIDIA-Megatron-LM
MEG_CKPT=/scratch/.../longctx-extend/output_256k_v2/ckpt_262144   # Megatron torch_dist ckpt
OUT=/scratch/.../longctx-extend/prelude_256k_hf                    # HF output dir
TOK=/scratch/project_465002530/users/pyysalos/tokenizers/openeurollm/tokenizer-256k

srun singularity exec -B /pfs -B /scratch -B /flash "$CONTAINER" bash -lc "
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export PYTHONPATH=$SHARED_MEG:$CONV/checkpoint
  cd $CONV
  python3 -u $CONV/checkpoint/convert.py \
    --model-type GPT --loader core --saver qwen3 \
    --load-dir $MEG_CKPT --save-dir $OUT \
    --tokenizer-dir $TOK --save-dtype bf16
"
```
Reference sbatch: `conv/convert_256k.sbatch` (also `convert_prelude_128k.sbatch`, `convert_128k2x.sbatch`).
Output is a standard HF dir (config.json, pytorch_model.bin, tokenizer). Verify
`rope_theta` / `max_position_embeddings` in the produced `config.json`.

---

## Ready-made sbatch templates (LUMI)
- HF→Megatron: `longctx-extend/convert_prelude.sbatch`
- Megatron→HF: `conv/convert_256k.sbatch`, `conv/convert_prelude_128k.sbatch`
To reuse: edit `HF_MODEL`/`MEG_OUT` (import) or `MEG_CKPT`/`OUT`/`TOK` (export), then `sbatch <file>`.

## Gotchas
- **PYTHONPATH order** for the Bridge (python-packages first) — the #1 failure.
- **`/flash`** hosts the Bridge and is frequently full/flaky — keep working data on `/scratch`.
- **DSA/indexer checkpoints** need special handling (heterogeneous keys) — see
  `docs/sparse_attention_dsa.md`; the two converters above are for standard dense/GQA checkpoints.
- After export, sanity-check with a short NIAH/generation run before publishing.
