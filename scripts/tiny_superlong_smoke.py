#!/usr/bin/env python3
"""Tiny super-long-context smoke test for small GPUs.

This is not meant to model OpenEuroLLM quality. It checks that the ingredients
for the 2M experiment are mechanically sound:

* RoPE can receive large position IDs up to a 2M span.
* A tiny causal LM can train on retrieval traces with PoSE-style offsets.
* Forced-choice retrieval eval works at longer physical context lengths.

The tokenizer is byte-level, so the script has no Hugging Face dependency and
can run on a small GPU box even when no model cache is present.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD = 0
EOS = 1
BYTE_OFFSET = 2
VOCAB_SIZE = 258


class ByteTokenizer:
    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [b + BYTE_OFFSET for b in text.encode("utf-8", errors="replace")]
        if add_eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = bytes(max(0, token - BYTE_OFFSET) for token in ids if token >= BYTE_OFFSET)
        return data.decode("utf-8", errors="replace")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return x * self.weight


def apply_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    rope_scale: float,
) -> torch.Tensor:
    # x: [batch, heads, seq, head_dim]
    pos = position_ids.to(dtype=inv_freq.dtype) / rope_scale
    freqs = torch.einsum("bs,d->bsd", pos, inv_freq)
    cos = freqs.cos().unsqueeze(1)
    sin = freqs.sin().unsqueeze(1)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, rope_base: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("--d-model must be divisible by --heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model, bias=False),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model, bias=False),
        )
        inv = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, rope_scale: float) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, position_ids, self.inv_freq, rope_scale)
        k = apply_rope(k, position_ids, self.inv_freq, rope_scale)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        x = residual + self.out(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyRoPELM(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_mult: int,
        rope_base: float,
        rope_scale: float,
    ) -> None:
        super().__init__()
        self.rope_scale = rope_scale
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.blocks = nn.ModuleList(
            TinyBlock(d_model=d_model, n_heads=n_heads, mlp_mult=mlp_mult, rope_base=rope_base)
            for _ in range(n_layers)
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, position_ids, self.rope_scale)
        return self.lm_head(self.norm(x))


@dataclass
class Example:
    prompt: str
    answer: str
    wrong_answers: list[str]
    depth: float
    key: str


KEYS = [
    "aurora", "birch", "cobalt", "delta", "ember", "fjord", "granite", "harbor",
    "iris", "juniper", "kelvin", "lumen", "meridian", "nimbus", "onyx", "polar",
]
STATUSES = ["green", "amber", "blue", "silver", "paused", "closed", "pending"]


def rand_code(rng: random.Random) -> str:
    return str(rng.randrange(1_000_000, 9_999_999))


def rand_key(rng: random.Random) -> str:
    return rng.choice(KEYS) + f"-{rng.randrange(1000, 9999)}"


def filler_line(rng: random.Random, idx: int) -> str:
    return (
        f"Background record {idx}: unrelated project {rand_key(rng)} has status "
        f"{rng.choice(STATUSES)} and checksum {rand_code(rng)}.\n"
    )


def make_example(rng: random.Random, tokenizer: ByteTokenizer, target_tokens: int, depth: float) -> Example:
    key = rand_key(rng)
    answer = rand_code(rng)
    wrong = [rand_code(rng) for _ in range(3)]
    evidence = f"Critical record: project {key} has verification code {answer}.\n"
    distractors = "".join(
        f"Nearby record {idx}: project {rand_key(rng)} has verification code {rand_code(rng)}.\n"
        for idx in range(12)
    )
    query = f"\nQuestion: What is the verification code for project {key}?\nAnswer: "
    preamble = "Audit log. Use the matching record and ignore unrelated records.\n\n"

    sample = filler_line(rng, 999)
    fixed = len(tokenizer.encode(preamble + evidence + distractors + query))
    filler_count = max(0, (target_tokens - fixed) // max(1, len(tokenizer.encode(sample))) + 2)
    before_count = max(0, int(round(filler_count * depth)))
    after_count = max(0, filler_count - before_count)
    before = "".join(filler_line(rng, idx) for idx in range(1, before_count + 1))
    after = "".join(
        filler_line(rng, idx) for idx in range(before_count + 1, before_count + after_count + 1)
    )
    prompt = preamble + before + evidence + distractors + after + query
    # Keep the query at the end. If the prompt overshoots, trim filler from the
    # middle/tail rather than trimming the answer-bearing evidence.
    ids = tokenizer.encode(prompt)
    if len(ids) > target_tokens:
        keep_prefix = tokenizer.encode(preamble + before + evidence + distractors)
        keep_suffix = tokenizer.encode(query)
        budget_mid = max(0, target_tokens - len(keep_prefix) - len(keep_suffix))
        after_ids = tokenizer.encode(after)[:budget_mid]
        prompt = tokenizer.decode(keep_prefix + after_ids + keep_suffix)
    elif len(ids) < target_tokens:
        prompt = prompt + (" " * (target_tokens - len(ids)))
    return Example(prompt=prompt, answer=answer, wrong_answers=wrong, depth=depth, key=key)


def train_sequence(
    rng: random.Random,
    tokenizer: ByteTokenizer,
    seq_len: int,
    depths: list[float],
) -> list[int]:
    ex = make_example(rng, tokenizer, seq_len, rng.choice(depths))
    ids = tokenizer.encode(ex.prompt + ex.answer + "\n", add_eos=True)
    if len(ids) < seq_len + 1:
        ids.extend([EOS] * (seq_len + 1 - len(ids)))
    return ids[: seq_len + 1]


def make_position_ids(
    rng: random.Random,
    batch_size: int,
    seq_len: int,
    max_position: int,
    device: torch.device,
    strategy: str,
) -> torch.Tensor:
    base = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    if strategy == "contiguous":
        return base
    if strategy == "pose_offset":
        max_offset = max(0, max_position - seq_len)
        offsets = [rng.randrange(0, max_offset + 1) if max_offset else 0 for _ in range(batch_size)]
        return base + torch.tensor(offsets, device=device, dtype=torch.long).unsqueeze(1)
    if strategy == "pose_far":
        max_offset = max(0, max_position - seq_len)
        low = int(max_offset * 0.75)
        offsets = [rng.randrange(low, max_offset + 1) if max_offset else 0 for _ in range(batch_size)]
        return base + torch.tensor(offsets, device=device, dtype=torch.long).unsqueeze(1)
    raise ValueError(f"unknown position strategy: {strategy}")


def candidate_logprob(
    model: TinyRoPELM,
    tokenizer: ByteTokenizer,
    prompt: str,
    candidate: str,
    max_position: int,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer.encode(prompt)
    cand_ids = tokenizer.encode(candidate + "\n")
    ids = torch.tensor([prompt_ids + cand_ids], dtype=torch.long, device=device)
    pos = torch.arange(ids.shape[1], device=device, dtype=torch.long).unsqueeze(0)
    logits = model(ids[:, :-1], pos[:, :-1]).float()
    logp = F.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    score = 0.0
    for i, token in enumerate(cand_ids):
        score += float(logp[0, start + i, token].detach().cpu())
    return score


@torch.no_grad()
def evaluate(
    model: TinyRoPELM,
    tokenizer: ByteTokenizer,
    rng: random.Random,
    eval_lengths: list[int],
    depths: list[float],
    trials: int,
    max_position: int,
    device: torch.device,
) -> dict:
    model.eval()
    rows = []
    for length in eval_lengths:
        for depth in depths:
            correct = 0
            for _ in range(trials):
                ex = make_example(rng, tokenizer, length, depth)
                candidates = [ex.answer] + ex.wrong_answers
                rng.shuffle(candidates)
                scores = {
                    cand: candidate_logprob(model, tokenizer, ex.prompt, cand, max_position, device)
                    for cand in candidates
                }
                pred = max(scores, key=scores.get)
                correct += int(pred == ex.answer)
            rows.append(
                {
                    "length": length,
                    "depth": depth,
                    "trials": trials,
                    "correct": correct,
                    "accuracy": correct / trials,
                }
            )
    model.train()
    return {"rows": rows}


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-position", type=int, default=2_097_152)
    p.add_argument("--original-context", type=int, default=2048)
    p.add_argument("--position-strategy", choices=["contiguous", "pose_offset", "pose_far"], default="pose_offset")
    p.add_argument("--train-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-mult", type=int, default=4)
    p.add_argument("--rope-base", type=float, default=500000.0)
    p.add_argument("--rope-scale", type=float, default=None)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-lengths", default="2048,4096,8192")
    p.add_argument("--eval-depths", default="0.05,0.5,0.9")
    p.add_argument("--eval-trials", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--save-final", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = ByteTokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    if device.type == "cpu":
        dtype = torch.float32
    rope_scale = args.rope_scale or (args.max_position / args.original_context)
    model = TinyRoPELM(
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        mlp_mult=args.mlp_mult,
        rope_base=args.rope_base,
        rope_scale=rope_scale,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    depths = parse_floats(args.eval_depths)
    eval_lengths = parse_ints(args.eval_lengths)
    run_config = vars(args) | {
        "device": str(device),
        "actual_dtype": str(dtype),
        "rope_scale": rope_scale,
        "vocab_size": VOCAB_SIZE,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    metrics_path = out_dir / "metrics.jsonl"
    start = time.time()

    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            rows = [train_sequence(rng, tokenizer, args.seq_len, depths) for _ in range(args.batch_size)]
            batch = torch.tensor(rows, dtype=torch.long, device=device)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            position_ids = make_position_ids(
                rng,
                args.batch_size,
                args.seq_len,
                args.max_position,
                device,
                args.position_strategy,
            )
            logits = model(input_ids, position_ids).float()
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1)) / args.grad_accum
            loss.backward()
            total_loss += float(loss.detach().cpu()) * args.grad_accum
        optimizer.step()

        record = {
            "step": step,
            "train_loss": total_loss,
            "tokens": step * args.batch_size * args.grad_accum * args.seq_len,
            "elapsed_sec": time.time() - start,
            "max_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
            ),
        }
        if args.eval_every and step % args.eval_every == 0:
            record["eval"] = evaluate(
                model,
                tokenizer,
                rng,
                eval_lengths=eval_lengths,
                depths=depths,
                trials=args.eval_trials,
                max_position=args.max_position,
                device=device,
            )
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    final_eval = evaluate(
        model,
        tokenizer,
        rng,
        eval_lengths=eval_lengths,
        depths=depths,
        trials=args.eval_trials,
        max_position=args.max_position,
        device=device,
    )
    (out_dir / "final_eval.json").write_text(json.dumps(final_eval, indent=2) + "\n", encoding="utf-8")
    if args.save_final:
        torch.save({"model": model.state_dict(), "config": run_config}, out_dir / "tiny_superlong_model.pt")


if __name__ == "__main__":
    main()
