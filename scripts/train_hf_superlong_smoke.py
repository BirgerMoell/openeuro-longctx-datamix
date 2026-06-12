#!/usr/bin/env python3
"""More realistic small-GPU superlong-context smoke with a real HF model.

Compared with `tiny_superlong_smoke.py`, this uses:

* a pretrained Hugging Face causal LM and tokenizer
* suffix-only retrieval loss to avoid materializing full seq_len x vocab logits
* optional large position offsets over a 2M span
* frozen base model with the last N transformer layers trainable
* before/after forced-choice retrieval eval

It is still a smoke test, not a final training recipe.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


KEYS = [
    "aurora", "birch", "cobalt", "delta", "ember", "fjord", "granite", "harbor",
    "iris", "juniper", "kelvin", "lumen", "meridian", "nimbus", "onyx", "polar",
    "quartz", "raven", "saffron", "tundra", "umbra", "vector", "willow", "zenith",
]
STATUSES = ["green", "amber", "blue", "silver", "paused", "closed", "pending"]


@dataclass
class RetrievalExample:
    prompt_ids: list[int]
    answer_ids: list[int]
    wrong_answer_ids: list[list[int]]
    depth: float
    key: str
    answer: str
    evidence_start: int
    query_start: int


def rand_code(rng: random.Random) -> str:
    return str(rng.randrange(1_000_000, 9_999_999))


def rand_value(rng: random.Random, value_kind: str, exclude: set[str] | None = None) -> str:
    exclude = exclude or set()
    if value_kind == "code":
        value = rand_code(rng)
        while value in exclude:
            value = rand_code(rng)
        return value
    if value_kind == "status":
        choices = [status for status in STATUSES if status not in exclude]
        if not choices:
            choices = STATUSES[:]
        return rng.choice(choices)
    raise ValueError(f"unknown value kind: {value_kind}")


def rand_key(rng: random.Random, stem: str | None = None) -> str:
    return (stem or rng.choice(KEYS)) + f"-{rng.randrange(1000, 9999)}"


def filler_line(rng: random.Random, idx: int) -> str:
    return (
        f"Background record {idx}: unrelated project {rand_key(rng)} has status "
        f"{rng.choice(STATUSES)} and checksum {rand_code(rng)}.\n"
    )


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def filler_ids_to_budget(tokenizer, rng: random.Random, budget: int, start_idx: int) -> tuple[list[int], int]:
    ids: list[int] = []
    idx = start_idx
    while len(ids) < budget:
        line_ids = encode(tokenizer, filler_line(rng, idx))
        remaining = budget - len(ids)
        ids.extend(line_ids[:remaining])
        idx += 1
    return ids, idx


def make_retrieval_example(
    tokenizer,
    rng: random.Random,
    target_prompt_tokens: int,
    depth: float,
    distractors: int,
    wrong_candidates: int,
    value_kind: str,
) -> RetrievalExample:
    stem = rng.choice(KEYS)
    key = rand_key(rng, stem=stem)
    answer = rand_value(rng, value_kind)
    field_name = "verification code" if value_kind == "code" else "status label"
    preamble = "Audit log. Use the matching record and ignore unrelated records.\n\n"
    evidence = f"Critical record: project {key} has {field_name} {answer}.\n"
    query = f"\nQuestion: What is the {field_name} for project {key}?\nAnswer:"

    distractor_rows: list[tuple[str, str]] = []
    for idx in range(max(1, distractors)):
        # Half the distractors share the target stem, so the model must bind the
        # exact key to the exact code instead of just copying any nearby number.
        distractor_key = rand_key(rng, stem=stem if idx % 2 == 0 else None)
        while distractor_key == key:
            distractor_key = rand_key(rng, stem=stem)
        distractor_value = rand_value(rng, value_kind, exclude={answer})
        distractor_rows.append((distractor_key, distractor_value))

    preamble_ids = encode(tokenizer, preamble)
    evidence_ids = encode(tokenizer, evidence)
    query_ids = encode(tokenizer, query)

    def distractor_ids(rows: list[tuple[str, str]]) -> list[int]:
        text = "".join(
            f"Nearby record {idx}: project {dkey} has {field_name} {dvalue}.\n"
            for idx, (dkey, dvalue) in enumerate(rows, start=1)
        )
        return encode(tokenizer, text)

    min_rows = max(1, wrong_candidates)
    local_rows = distractor_rows[:]
    local_ids = distractor_ids(local_rows)
    fixed_len = len(preamble_ids) + len(evidence_ids) + len(local_ids) + len(query_ids)
    while fixed_len > target_prompt_tokens and len(local_rows) > min_rows:
        local_rows.pop()
        local_ids = distractor_ids(local_rows)
        fixed_len = len(preamble_ids) + len(evidence_ids) + len(local_ids) + len(query_ids)

    filler_budget = max(0, target_prompt_tokens - fixed_len)
    before_budget = int(round(filler_budget * depth))
    after_budget = max(0, filler_budget - before_budget)
    before_ids, next_idx = filler_ids_to_budget(tokenizer, rng, before_budget, 1)
    after_ids, next_idx = filler_ids_to_budget(tokenizer, rng, after_budget, next_idx)

    evidence_start = len(preamble_ids) + len(before_ids)
    query_start = evidence_start + len(evidence_ids) + len(local_ids) + len(after_ids)
    prompt_ids = preamble_ids + before_ids + evidence_ids + local_ids + after_ids + query_ids

    if len(prompt_ids) < target_prompt_tokens:
        extra_ids, _ = filler_ids_to_budget(tokenizer, rng, target_prompt_tokens - len(prompt_ids), next_idx)
        after_ids += extra_ids
        query_start += len(extra_ids)
        prompt_ids = preamble_ids + before_ids + evidence_ids + local_ids + after_ids + query_ids

    answer_ids = encode(tokenizer, " " + answer)
    candidate_pool = [value for _, value in local_rows if value != answer]
    if value_kind == "status":
        candidate_pool = sorted(set(candidate_pool))
    wrong_answers = rng.sample(candidate_pool, k=min(wrong_candidates, len(candidate_pool)))
    wrong_answer_ids = [encode(tokenizer, " " + wrong) for wrong in wrong_answers]
    return RetrievalExample(
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        wrong_answer_ids=wrong_answer_ids,
        depth=depth,
        key=key,
        answer=answer,
        evidence_start=evidence_start,
        query_start=query_start,
    )


def make_position_ids(
    rng: random.Random,
    length: int,
    max_position: int,
    device: torch.device,
    strategy: str,
    example: RetrievalExample | None = None,
) -> torch.Tensor:
    base = torch.arange(length, device=device, dtype=torch.long).unsqueeze(0)
    if strategy == "contiguous":
        return base
    max_offset = max(0, max_position - length)
    if strategy == "pose_offset":
        offset = rng.randrange(0, max_offset + 1) if max_offset else 0
    elif strategy == "pose_far":
        low = int(max_offset * 0.75)
        offset = rng.randrange(low, max_offset + 1) if max_offset else 0
    elif strategy == "pose_bridge":
        if example is None:
            if length <= 1:
                return base
            return torch.linspace(
                0,
                max_position - 1,
                steps=length,
                device=device,
                dtype=torch.float32,
            ).round().long().unsqueeze(0)
        query_start = min(max(0, example.query_start), length)
        tail_len = max(1, length - query_start)
        tail_start = max(0, max_position - tail_len)
        evidence_virtual = int(round(example.depth * max(0, tail_start - 1)))
        pre_start = max(0, evidence_virtual - min(example.evidence_start, query_start))
        pre_positions = torch.arange(query_start, device=device, dtype=torch.long) + pre_start
        if query_start and int(pre_positions[-1]) >= tail_start:
            shift = int(pre_positions[-1]) - tail_start + 1
            pre_positions = torch.clamp(pre_positions - shift, min=0)
        tail_positions = torch.arange(tail_len, device=device, dtype=torch.long) + tail_start
        return torch.cat([pre_positions, tail_positions], dim=0).unsqueeze(0)
    else:
        raise ValueError(f"unknown position strategy: {strategy}")
    return base + offset


def get_core_and_head(model):
    core = getattr(model, "model", None)
    head = getattr(model, "lm_head", None)
    if core is None or head is None:
        raise RuntimeError("Expected a Llama/Qwen-style model with .model and .lm_head")
    return core, head


def suffix_loss(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    answer_start: int,
) -> torch.Tensor:
    core, head = get_core_and_head(model)
    outputs = core(
        input_ids=input_ids[:, :-1],
        position_ids=position_ids[:, :-1],
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    labels = input_ids[:, 1:]
    # Candidate token at full index j is predicted by logit row j-1. The first
    # answer token is at full index answer_start.
    row_start = max(0, answer_start - 1)
    target_hidden = hidden[:, row_start:, :]
    target_labels = labels[:, row_start:]
    logits = head(target_hidden).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_labels.reshape(-1))


@torch.no_grad()
def score_candidate(
    model,
    example: RetrievalExample,
    candidate_ids: list[int],
    rng: random.Random,
    max_position: int,
    position_strategy: str,
    device: torch.device,
) -> float:
    prompt_ids = example.prompt_ids
    ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long, device=device)
    position_ids = make_position_ids(
        rng,
        ids.shape[1],
        max_position=max_position,
        device=device,
        strategy=position_strategy,
        example=example,
    )
    core, head = get_core_and_head(model)
    outputs = core(
        input_ids=ids[:, :-1],
        position_ids=position_ids[:, :-1],
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    row_start = len(prompt_ids) - 1
    rows = hidden[:, row_start:, :]
    logits = head(rows).float()[0]
    logp = F.log_softmax(logits, dim=-1)
    return float(sum(logp[idx, token].item() for idx, token in enumerate(candidate_ids)))


def differentiable_candidate_score(
    model,
    example: RetrievalExample,
    candidate_ids: list[int],
    rng: random.Random,
    max_position: int,
    position_strategy: str,
    device: torch.device,
) -> torch.Tensor:
    prompt_ids = example.prompt_ids
    ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long, device=device)
    position_ids = make_position_ids(
        rng,
        ids.shape[1],
        max_position=max_position,
        device=device,
        strategy=position_strategy,
        example=example,
    )
    core, head = get_core_and_head(model)
    outputs = core(
        input_ids=ids[:, :-1],
        position_ids=position_ids[:, :-1],
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    row_start = len(prompt_ids) - 1
    rows = hidden[:, row_start:, :]
    logits = head(rows).float()[0]
    logp = F.log_softmax(logits, dim=-1)
    token_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    return logp.gather(1, token_ids.unsqueeze(1)).sum()


def contrastive_retrieval_loss(
    model,
    example: RetrievalExample,
    rng: random.Random,
    max_position: int,
    position_strategy: str,
    device: torch.device,
    max_wrong: int,
    temperature: float,
) -> torch.Tensor:
    wrong = example.wrong_answer_ids[:]
    rng.shuffle(wrong)
    candidates = [example.answer_ids] + wrong[:max_wrong]
    scores = torch.stack(
        [
            differentiable_candidate_score(
                model,
                example,
                candidate,
                rng,
                max_position=max_position,
                position_strategy=position_strategy,
                device=device,
            )
            for candidate in candidates
        ]
    )
    labels = torch.zeros(1, dtype=torch.long, device=device)
    return F.cross_entropy((scores / temperature).unsqueeze(0), labels)


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    rng: random.Random,
    lengths: list[int],
    depths: list[float],
    trials: int,
    max_position: int,
    position_strategy: str,
    device: torch.device,
    distractors: int,
    wrong_candidates: int,
    value_kind: str,
) -> dict[str, Any]:
    model.eval()
    rows = []
    for length in lengths:
        for depth in depths:
            correct = 0
            for _ in range(trials):
                ex = make_retrieval_example(
                    tokenizer,
                    rng,
                    length,
                    depth,
                    distractors=distractors,
                    wrong_candidates=wrong_candidates,
                    value_kind=value_kind,
                )
                candidates = [ex.answer_ids] + ex.wrong_answer_ids
                rng.shuffle(candidates)
                scores = [
                    score_candidate(
                        model,
                        ex,
                        candidate,
                        rng,
                        max_position=max_position,
                        position_strategy=position_strategy,
                        device=device,
                    )
                    for candidate in candidates
                ]
                pred_idx = max(range(len(scores)), key=lambda i: scores[i])
                correct += int(candidates[pred_idx] == ex.answer_ids)
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


def freeze_for_smoke(model, last_n_layers: int, train_lm_head: bool) -> dict[str, Any]:
    for param in model.parameters():
        param.requires_grad = False
    core, head = get_core_and_head(model)
    layers = getattr(core, "layers", None)
    if layers is None:
        raise RuntimeError("Expected model.model.layers")
    trainable_modules = []
    if last_n_layers > 0:
        for layer in layers[-last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
            trainable_modules.append(layer.__class__.__name__)
    norm = getattr(core, "norm", None)
    if norm is not None:
        for param in norm.parameters():
            param.requires_grad = True
        trainable_modules.append("final_norm")
    if train_lm_head:
        for param in head.parameters():
            param.requires_grad = True
        trainable_modules.append("lm_head")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": trainable / total,
        "trainable_modules": trainable_modules,
    }


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def scheduled_max_position(step: int, train_steps: int, start_position: int | None, end_position: int) -> int:
    if start_position is None or start_position >= end_position or train_steps <= 1:
        return end_position
    progress = (step - 1) / (train_steps - 1)
    log_start = math.log(max(2, start_position))
    log_end = math.log(max(2, end_position))
    return int(round(math.exp(log_start + progress * (log_end - log_start))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--max-position", type=int, default=2_097_152)
    p.add_argument("--train-start-position", type=int, default=None)
    p.add_argument("--rope-scaling", choices=["none", "linear", "dynamic", "yarn"], default="none")
    p.add_argument("--rope-factor", type=float, default=None)
    p.add_argument("--rope-original-max-position", type=int, default=None)
    p.add_argument(
        "--position-strategy",
        choices=["contiguous", "pose_offset", "pose_far", "pose_bridge"],
        default="pose_bridge",
    )
    p.add_argument(
        "--eval-position-strategy",
        choices=["contiguous", "pose_offset", "pose_far", "pose_bridge"],
        default="pose_bridge",
    )
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--last-n-layers", type=int, default=2)
    p.add_argument("--no-train-lm-head", action="store_true")
    p.add_argument("--contrastive-weight", type=float, default=0.0)
    p.add_argument("--contrastive-wrong-candidates", type=int, default=3)
    p.add_argument("--contrastive-temperature", type=float, default=1.0)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--eval-before", action="store_true")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-lengths", default="4096,8192,16384")
    p.add_argument("--eval-depths", default="0.05,0.5,0.9")
    p.add_argument("--eval-trials", type=int, default=4)
    p.add_argument("--distractors", type=int, default=32)
    p.add_argument("--wrong-candidates", type=int, default=7)
    p.add_argument("--value-kind", choices=["code", "status"], default="code")
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--save-final", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    original_max_position = int(
        args.rope_original_max_position
        or getattr(config, "max_position_embeddings", 0)
        or args.seq_len
    )
    target_max_position = max(int(getattr(config, "max_position_embeddings", 0) or 0), args.max_position)
    config.max_position_embeddings = target_max_position
    if args.rope_scaling != "none":
        rope_factor = float(args.rope_factor or (target_max_position / original_max_position))
        config.rope_scaling = {
            "rope_type": args.rope_scaling,
            "factor": rope_factor,
        }
        if args.rope_scaling in {"dynamic", "yarn"}:
            config.rope_scaling["original_max_position_embeddings"] = original_max_position
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    trainable_info = freeze_for_smoke(
        model,
        last_n_layers=args.last_n_layers,
        train_lm_head=not args.no_train_lm_head,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    eval_lengths = parse_ints(args.eval_lengths)
    eval_depths = parse_floats(args.eval_depths)
    run_config = vars(args) | {
        "device": str(device),
        "actual_dtype": str(dtype),
        "trainable": trainable_info,
        "model_type": getattr(config, "model_type", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "original_max_position_embeddings": original_max_position,
        "target_max_position_embeddings": target_max_position,
        "effective_rope_scaling": getattr(config, "rope_scaling", None),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_config, indent=2), flush=True)

    if args.eval_before:
        before = evaluate(
            model,
            tokenizer,
            rng,
            lengths=eval_lengths,
            depths=eval_depths,
            trials=args.eval_trials,
            max_position=args.max_position,
            position_strategy=args.eval_position_strategy,
            device=device,
            distractors=args.distractors,
            wrong_candidates=args.wrong_candidates,
            value_kind=args.value_kind,
        )
        (output_dir / "before_eval.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"before_eval": before}), flush=True)

    metrics_path = output_dir / "metrics.jsonl"
    start_time = time.time()
    model.train()
    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            depth = rng.choice(eval_depths)
            ex = make_retrieval_example(
                tokenizer,
                rng,
                args.seq_len,
                depth,
                distractors=args.distractors,
                wrong_candidates=args.wrong_candidates,
                value_kind=args.value_kind,
            )
            eos = tokenizer.eos_token_id
            ids = ex.prompt_ids + ex.answer_ids + ([eos] if eos is not None else [])
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            position_ids = make_position_ids(
                rng,
                input_ids.shape[1],
                max_position=scheduled_max_position(
                    step,
                    args.train_steps,
                    args.train_start_position,
                    args.max_position,
                ),
                device=device,
                strategy=args.position_strategy,
                example=ex,
            )
            train_max_position = scheduled_max_position(
                step,
                args.train_steps,
                args.train_start_position,
                args.max_position,
            )
            loss = suffix_loss(model, input_ids, position_ids, len(ex.prompt_ids))
            if args.contrastive_weight:
                loss = loss + args.contrastive_weight * contrastive_retrieval_loss(
                    model,
                    ex,
                    rng,
                    max_position=train_max_position,
                    position_strategy=args.position_strategy,
                    device=device,
                    max_wrong=args.contrastive_wrong_candidates,
                    temperature=args.contrastive_temperature,
                )
            loss = loss / args.grad_accum
            loss.backward()
            total_loss += float(loss.detach().cpu()) * args.grad_accum
        optimizer.step()

        record: dict[str, Any] = {
            "step": step,
            "train_loss": total_loss,
            "tokens": step * args.seq_len * args.grad_accum,
            "elapsed_sec": time.time() - start_time,
            "max_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
        }
        if args.eval_every and step % args.eval_every == 0:
            record["eval"] = evaluate(
                model,
                tokenizer,
                rng,
                lengths=eval_lengths,
                depths=eval_depths,
                trials=args.eval_trials,
                max_position=args.max_position,
                position_strategy=args.eval_position_strategy,
                device=device,
                distractors=args.distractors,
                wrong_candidates=args.wrong_candidates,
                value_kind=args.value_kind,
            )
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    final_eval = evaluate(
        model,
        tokenizer,
        rng,
        lengths=eval_lengths,
        depths=eval_depths,
        trials=args.eval_trials,
        max_position=args.max_position,
        position_strategy=args.eval_position_strategy,
        device=device,
        distractors=args.distractors,
        wrong_candidates=args.wrong_candidates,
        value_kind=args.value_kind,
    )
    (output_dir / "final_eval.json").write_text(json.dumps(final_eval, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"final_eval": final_eval}), flush=True)
    if args.save_final:
        model.save_pretrained(output_dir / "final")
        tokenizer.save_pretrained(output_dir / "final")


if __name__ == "__main__":
    main()
