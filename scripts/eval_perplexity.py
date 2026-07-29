#!/usr/bin/env python3
"""Windowed perplexity eval for our long-context HF checkpoints.

Measures mean token NLL / perplexity over non-overlapping windows at several sequence lengths — the
metric that catches short-context regression from aggressive RoPE-θ scaling (NIAH does not).

Compare a θ-extended model against its base (or across θ values) at SHORT lengths (512–8K) to see if
extension hurt local modelling, and at LONG lengths (128K/256K) to confirm long-context gain.

Usage:
  python eval_perplexity.py --model <hf-repo-or-local-path> \
      --seq-lengths 512 2048 8192 32768 131072 \
      --text-file held_out.txt --num-windows 50 --output ppl.json

Data: pass --text-file (plain text, e.g. held-out multilingual docs) OR --hf-dataset name[:config].
Report is per-seq-length perplexity (and loss). For a fair θ comparison, run the SAME data/lengths on
each model and on the pre-extension base.
"""
import argparse, json, math, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_text(args, tok):
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8", errors="ignore")
    if args.hf_dataset:
        from datasets import load_dataset
        name, *cfg = args.hf_dataset.split(":")
        ds = load_dataset(name, cfg[0] if cfg else None, split=args.hf_split, streaming=True)
        buf, n = [], 0
        for ex in ds:
            t = ex.get(args.text_key) or ""
            if t:
                buf.append(t); n += len(t)
            if n > args.max_chars:
                break
        return "\n\n".join(buf)
    raise SystemExit("provide --text-file or --hf-dataset")


@torch.no_grad()
def windowed_ppl(model, ids, L, num_windows, device):
    """Mean per-token NLL over up to num_windows non-overlapping windows of length L."""
    total_nll, total_tok, used = 0.0, 0, 0
    for start in range(0, ids.shape[0] - L, L):
        if used >= num_windows:
            break
        w = ids[start:start + L].unsqueeze(0).to(device)
        out = model(w, labels=w)                      # HF shifts + masks internally
        # out.loss is mean over (L-1) tokens; recover the sum to aggregate fairly across windows
        ntok = L - 1
        total_nll += out.loss.item() * ntok
        total_tok += ntok
        used += 1
    if total_tok == 0:
        return None
    mean_nll = total_nll / total_tok
    return {"seq_len": L, "windows": used, "loss": mean_nll, "ppl": math.exp(min(20.0, mean_nll))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    ap.add_argument("--num-windows", type=int, default=50)
    ap.add_argument("--text-file"); ap.add_argument("--hf-dataset")
    ap.add_argument("--hf-split", default="train"); ap.add_argument("--text-key", default="text")
    ap.add_argument("--max-chars", type=int, default=50_000_000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    text = load_text(args, tok)
    ids = tok(text, return_tensors="pt", truncation=False).input_ids[0]
    print(f"tokenized {ids.shape[0]:,} tokens", flush=True)

    results = []
    for L in sorted(args.seq_lengths):
        if ids.shape[0] < L + 1:
            print(f"  seq_len {L}: not enough tokens, skipping", flush=True); continue
        r = windowed_ppl(model, ids, L, args.num_windows, device)
        if r:
            print(f"  seq_len {L:>7}: ppl={r['ppl']:.3f} loss={r['loss']:.4f} ({r['windows']} windows)",
                  flush=True)
            results.append(r)
    out = {"model": args.model, "results": results}
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2)); print(f"wrote {args.output}")
    print("=== PPL EVAL DONE ===")


if __name__ == "__main__":
    main()
