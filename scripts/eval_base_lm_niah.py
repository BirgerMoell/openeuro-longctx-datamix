#!/usr/bin/env python3
"""
Base-LM NIAH: forced-choice log-likelihood scoring.

No instruction following required. Measures long-context retrieval and
key-value binding by comparing log-probabilities of candidate completions.

Eval design
-----------
  1. Generate a context with N key→value needle pairs (all numeric).
  2. Insert the query needle at a specified depth (0–100% of context).
  3. All 3 distractor candidates are values from *other* keys in the context
     → tests retrieval + binding, not memorisation of the number.
  4. Score each candidate as sum(log P(token | prefix)) under the model.
  5. Predict = argmax logprob. Accuracy reported per context-length × depth cell.

Controls
--------
  no_context   — prefix only, no context        → should fail (≈25%)
  shuffled     — key/value bindings swapped      → correct answer changes
  short_ctx    — 256-token context               → confirms scoring works

Usage
-----
  python eval_base_lm_niah.py \\
      --model /flash/.../converted/checkpoint_0001000 \\
      --output /scratch/.../eval-base-lm-niah \\
      --languages fr fi cs nl \\
      --context-lengths 2048 4096 8192 16384 32768 \\
      --depths 0.0 0.25 0.5 0.75 1.0 \\
      --trials 10
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Language templates ─────────────────────────────────────────────────────────
# Each template has:
#   needle   — one key→value fact line inserted into the context
#   prefix   — the base-LM completion prefix (no answer format, just a stub)
#   preamble — short header before the needle list

LANG_TEMPLATES = {
    "en": dict(
        needle='The special magic number for "{key}" is: {value}.',
        prefix='The special magic number for "{key}" is:',
        preamble="Below is a collection of facts.\n\n",
    ),
    "fr": dict(
        needle='Le nombre magique spécial pour « {key} » est : {value}.',
        prefix='Le nombre magique spécial pour « {key} » est :',
        preamble="Voici un ensemble de faits.\n\n",
    ),
    "fi": dict(
        needle='Sanan "{key}" erityinen taikuusnumero on: {value}.',
        prefix='Sanan "{key}" erityinen taikuusnumero on:',
        preamble="Alla on joukko faktoja.\n\n",
    ),
    "nl": dict(
        needle='Het speciale magische getal voor "{key}" is: {value}.',
        prefix='Het speciale magische getal voor "{key}" is:',
        preamble="Hieronder staat een verzameling feiten.\n\n",
    ),
    "cs": dict(
        needle='Speciální magické číslo pro „{key}" je: {value}.',
        prefix='Speciální magické číslo pro „{key}" je:',
        preamble="Níže je sbírka faktů.\n\n",
    ),
    "da": dict(
        needle='Det specielle magiske tal for "{key}" er: {value}.',
        prefix='Det specielle magiske tal for "{key}" er:',
        preamble="Nedenfor er en samling fakta.\n\n",
    ),
    "et": dict(
        needle='Sõna "{key}" eriline võlunumber on: {value}.',
        prefix='Sõna "{key}" eriline võlunumber on:',
        preamble="Allpool on kogum fakte.\n\n",
    ),
    "hr": dict(
        needle='Poseban čarobni broj za "{key}" je: {value}.',
        prefix='Poseban čarobni broj za "{key}" je:',
        preamble="Ispod je zbirka činjenica.\n\n",
    ),
}

KEYS = [
    "river", "forest", "apple", "castle", "mountain", "ocean", "bridge", "tower",
    "garden", "library", "mirror", "candle", "thunder", "crystal", "lantern", "whisper",
    "anchor", "compass", "feather", "horizon", "marble", "needle", "pillow", "shadow",
    "temple", "vessel", "winter", "arrow", "basket", "circle", "desert", "engine",
    "falcon", "glacier", "harbor", "island", "jungle", "kettle", "ladder", "meadow",
    "noble", "olive", "planet", "quartz", "ribbon", "silver", "timber", "umbrella",
    "valley", "walnut", "xenon", "amber", "bronze", "carbon", "diamond", "emerald",
    "flame", "gravel", "helium", "indigo", "jasper", "kelp", "lotus", "meteor",
]


def rand_value() -> str:
    return str(random.randint(1_000_000, 9_999_999))


# ── Context builder ────────────────────────────────────────────────────────────

def build_context(
    tokenizer,
    tmpl: dict,
    query_key: str,
    query_value: str,
    filler_kvs: list,
    target_tokens: int,
    depth: float,
    shuffle_bindings: bool = False,
) -> str:
    """
    Assemble a context string targeting `target_tokens` tokens.

    shuffle_bindings: rotate all values by 1 so every key maps to the
    *wrong* value → correct answer changes (control condition).
    """
    if shuffle_bindings:
        # Rotate: query key gets filler_kvs[0]'s value, etc.
        all_kvs = [(query_key, query_value)] + filler_kvs
        values = [v for _, v in all_kvs]
        values = values[1:] + [values[0]]
        all_kvs = [(k, v) for (k, _), v in zip(all_kvs, values)]
        query_key_val = (all_kvs[0][0], all_kvs[0][1])
        filler_kvs = all_kvs[1:]
    else:
        query_key_val = (query_key, query_value)

    query_needle = tmpl["needle"].format(key=query_key_val[0], value=query_key_val[1])
    filler_lines = [tmpl["needle"].format(key=k, value=v) for k, v in filler_kvs]

    # Estimate tokens per line and how many filler lines we need
    sample_line = filler_lines[0] if filler_lines else query_needle
    toks_per_line = max(1, len(tokenizer.encode(sample_line + "\n")))
    preamble_toks = len(tokenizer.encode(tmpl["preamble"]))
    query_toks = len(tokenizer.encode(query_needle + "\n"))

    # Leave ~50-token headroom for the prefix stub + candidate appended by the scorer
    budget = max(0, target_tokens - preamble_toks - query_toks - 50)
    need_filler = budget // toks_per_line

    # Cycle filler pool to fill the budget
    pool = []
    while len(pool) < need_filler and filler_lines:
        pool.extend(filler_lines)
    pool = pool[:need_filler]

    # Insert query needle at specified depth
    insert_idx = int(len(pool) * depth)
    pool.insert(insert_idx, query_needle)

    return tmpl["preamble"] + "\n".join(pool) + "\n\n"


# ── Scorer ─────────────────────────────────────────────────────────────────────

def score_completion(model, tokenizer, prefix: str, candidate: str, device) -> float:
    """Sum of log P(candidate tokens | prefix) under the model."""
    # Add a space before candidate so tokenisation matches in-context numbers
    full = prefix + " " + candidate.strip()
    enc_full = tokenizer(full, return_tensors="pt", truncation=False).to(device)
    enc_pre = tokenizer(prefix, return_tensors="pt", truncation=False).to(device)
    pre_len = enc_pre["input_ids"].shape[1]

    with torch.no_grad():
        out = model(**enc_full)

    logits = out.logits[0]  # [L, V]
    lp = F.log_softmax(logits, dim=-1)
    ids = enc_full["input_ids"][0][pre_len:]

    if len(ids) == 0:
        return float("-inf")
    return sum(lp[pre_len - 1 + j, ids[j]].item() for j in range(len(ids)))


# ── Single trial ───────────────────────────────────────────────────────────────

def run_trial(
    model,
    tokenizer,
    device,
    tmpl: dict,
    target_tokens: int,
    depth: float,
    trial_idx: int,
    no_context: bool = False,
    shuffle_bindings: bool = False,
) -> dict:
    rng_keys = random.sample(KEYS, 4)  # query + 3 distractors
    query_key = rng_keys[0]
    query_value = rand_value()
    distractor_kvs = [(k, rand_value()) for k in rng_keys[1:]]
    # Pad filler with more pairs so context actually reaches target length
    extra_keys = [k for k in KEYS if k not in rng_keys]
    random.shuffle(extra_keys)
    filler_kvs = distractor_kvs + [(k, rand_value()) for k in extra_keys[:50]]

    if no_context:
        context = ""
    else:
        context = build_context(
            tokenizer, tmpl, query_key, query_value,
            filler_kvs, target_tokens, depth,
            shuffle_bindings=shuffle_bindings,
        )

    prefix = context + tmpl["prefix"].format(key=query_key)

    # Candidates: true answer + 3 distractors from context
    if shuffle_bindings:
        # After rotation in build_context, query_key maps to filler_kvs[0][1].
        # Distractors must NOT include filler_kvs[0][1] to avoid duplicates.
        true_candidate = filler_kvs[0][1]
        distractors = [query_value] + [v for _, v in filler_kvs[1:3]]
    else:
        true_candidate = query_value
        distractors = [v for _, v in distractor_kvs[:3]]

    candidates = [true_candidate] + distractors
    random.shuffle(candidates)
    true_idx = candidates.index(true_candidate)

    scores = [score_completion(model, tokenizer, prefix, c, device) for c in candidates]
    pred_idx = scores.index(max(scores))

    actual_ctx_tokens = len(tokenizer.encode(context)) if context else 0

    return {
        "trial": trial_idx,
        "target_tokens": target_tokens,
        "actual_ctx_tokens": actual_ctx_tokens,
        "depth": depth,
        "no_context": no_context,
        "shuffle_bindings": shuffle_bindings,
        "query_key": query_key,
        "true_value": true_candidate,
        "candidates": candidates,
        "scores": scores,
        "true_idx": true_idx,
        "pred_idx": pred_idx,
        "correct": pred_idx == true_idx,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--languages", nargs="+", default=["fr", "fi", "cs", "nl"])
    ap.add_argument("--context-lengths", nargs="+", type=int,
                    default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--depths", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading model (bfloat16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print("Model loaded.\n", flush=True)

    SEP = "─" * 72
    all_results = []

    for lang in args.languages:
        if lang not in LANG_TEMPLATES:
            print(f"WARNING: no template for '{lang}', skipping", flush=True)
            continue
        tmpl = LANG_TEMPLATES[lang]
        lang_results = []
        print(f"\n{'═'*72}")
        print(f"Language: {lang.upper()}")
        print('═'*72, flush=True)

        # ── Main grid: context length × depth ─────────────────────────────
        for ctx_len in args.context_lengths:
            for depth in args.depths:
                cell_correct = 0
                for t in range(args.trials):
                    r = run_trial(model, tokenizer, device, tmpl,
                                  ctx_len, depth, t)
                    r["lang"] = lang
                    r["condition"] = "main"
                    lang_results.append(r)
                    cell_correct += r["correct"]
                acc = cell_correct / args.trials
                print(f"  ctx={ctx_len:6d}  depth={depth:.2f}  "
                      f"acc={acc:.2f}  ({cell_correct}/{args.trials})", flush=True)

        # ── Controls ───────────────────────────────────────────────────────
        print(f"\n{SEP}")
        print("Controls:", flush=True)

        # no_context (256 token target, any depth)
        nc_correct = 0
        for t in range(args.trials):
            r = run_trial(model, tokenizer, device, tmpl, 256, 0.5, t,
                          no_context=True)
            r["lang"] = lang
            r["condition"] = "no_context"
            lang_results.append(r)
            nc_correct += r["correct"]
        print(f"  no_context    acc={nc_correct/args.trials:.2f}  "
              f"({nc_correct}/{args.trials})", flush=True)

        # shuffled bindings (short context)
        sh_correct = 0
        for t in range(args.trials):
            r = run_trial(model, tokenizer, device, tmpl, 2048, 0.5, t,
                          shuffle_bindings=True)
            r["lang"] = lang
            r["condition"] = "shuffled"
            lang_results.append(r)
            sh_correct += r["correct"]
        print(f"  shuffled      acc={sh_correct/args.trials:.2f}  "
              f"({sh_correct}/{args.trials})", flush=True)

        # short baseline (256 tokens, centre)
        sb_correct = 0
        for t in range(args.trials):
            r = run_trial(model, tokenizer, device, tmpl, 256, 0.5, t)
            r["lang"] = lang
            r["condition"] = "short_ctx"
            lang_results.append(r)
            sb_correct += r["correct"]
        print(f"  short_ctx     acc={sb_correct/args.trials:.2f}  "
              f"({sb_correct}/{args.trials})", flush=True)

        all_results.extend(lang_results)
        # Save per-language JSONL
        with open(out_dir / f"{lang}_results.jsonl", "w") as f:
            for r in lang_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("SUMMARY — accuracy by lang × ctx_length (main condition, all depths)")
    print('═'*72)
    main_results = [r for r in all_results if r["condition"] == "main"]

    by_lang_ctx = defaultdict(list)
    for r in main_results:
        by_lang_ctx[(r["lang"], r["target_tokens"])].append(r["correct"])

    header = f"{'lang':>4}  " + "  ".join(f"{c:>6}" for c in args.context_lengths)
    print(header)
    for lang in args.languages:
        row = f"{lang:>4}  "
        for ctx in args.context_lengths:
            vals = by_lang_ctx.get((lang, ctx), [])
            acc = sum(vals) / len(vals) if vals else float("nan")
            row += f"  {acc:6.2f}"
        print(row)

    summary = {
        "languages": args.languages,
        "context_lengths": args.context_lengths,
        "depths": args.depths,
        "trials": args.trials,
        "by_lang_ctx": {
            f"{lang}_{ctx}": {
                "n": len(v),
                "correct": sum(v),
                "acc": sum(v) / len(v) if v else None,
            }
            for (lang, ctx), v in by_lang_ctx.items()
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "all_results.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
