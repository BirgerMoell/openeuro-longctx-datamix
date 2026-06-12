#!/usr/bin/env python3
"""Create synthetic long-context train/eval JSONL for 2M-context experiments.

This script intentionally writes raw JSONL, not Megatron indexed data. The normal
tokenization path can consume the `"text"` field later, while the extra metadata
is useful for evaluation and for future PoSE-style position-id training.

The intended use is:

* train curriculum: physical sequences up to 256K tokens, with metadata saying
  they should be sampled over a 2M positional span.
* eval holdout: a tiny number of exact 512K/1M/2M retrieval examples.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


LANG = {
    "en": {
        "preamble": "Audit log. Use the matching records and ignore unrelated records.\n\n",
        "single_fact": "Record {idx}: project {key} has verification code {value}.\n",
        "multi_fact": "Record {idx}: project {key} stores value {value} for slot {slot}.\n",
        "count_fact": "Record {idx}: region {key} reports event type {event}.\n",
        "filler": "Background record {idx}: unrelated project {key} has status {status} and checksum {value}.\n",
        "single_query": "\nQuestion: What is the verification code for project {key}?\nAnswer: {answer}\n",
        "multi_query": "\nQuestion: What values are stored for project {key} in slots A and B?\nAnswer: {answer}\n",
        "count_query": "\nQuestion: How many ALERT records are reported for region {key}?\nAnswer: {answer}\n",
        "no_answer_query": "\nQuestion: What is the verification code for project {key}?\nAnswer: NOT_FOUND\n",
    },
    "sv": {
        "preamble": "Granskningslogg. Anvand matchande poster och ignorera orelaterade poster.\n\n",
        "single_fact": "Post {idx}: projekt {key} har verifieringskod {value}.\n",
        "multi_fact": "Post {idx}: projekt {key} lagrar varde {value} for plats {slot}.\n",
        "count_fact": "Post {idx}: region {key} rapporterar handelsetyp {event}.\n",
        "filler": "Bakgrundspost {idx}: orelaterat projekt {key} har status {status} och kontrollsumma {value}.\n",
        "single_query": "\nFraga: Vilken verifieringskod har projekt {key}?\nSvar: {answer}\n",
        "multi_query": "\nFraga: Vilka varden lagras for projekt {key} pa plats A och B?\nSvar: {answer}\n",
        "count_query": "\nFraga: Hur manga ALERT-poster rapporteras for region {key}?\nSvar: {answer}\n",
        "no_answer_query": "\nFraga: Vilken verifieringskod har projekt {key}?\nSvar: NOT_FOUND\n",
    },
    "de": {
        "preamble": "Pruefprotokoll. Verwende passende Eintraege und ignoriere fremde Eintraege.\n\n",
        "single_fact": "Eintrag {idx}: Projekt {key} hat den Pruefcode {value}.\n",
        "multi_fact": "Eintrag {idx}: Projekt {key} speichert Wert {value} fuer Slot {slot}.\n",
        "count_fact": "Eintrag {idx}: Region {key} meldet Ereignistyp {event}.\n",
        "filler": "Hintergrundeintrag {idx}: fremdes Projekt {key} hat Status {status} und Pruefsumme {value}.\n",
        "single_query": "\nFrage: Wie lautet der Pruefcode fuer Projekt {key}?\nAntwort: {answer}\n",
        "multi_query": "\nFrage: Welche Werte speichert Projekt {key} in Slot A und B?\nAntwort: {answer}\n",
        "count_query": "\nFrage: Wie viele ALERT-Eintraege werden fuer Region {key} gemeldet?\nAntwort: {answer}\n",
        "no_answer_query": "\nFrage: Wie lautet der Pruefcode fuer Projekt {key}?\nAntwort: NOT_FOUND\n",
    },
    "fr": {
        "preamble": "Journal d'audit. Utilise les enregistrements correspondants et ignore les autres.\n\n",
        "single_fact": "Enregistrement {idx}: le projet {key} a le code de verification {value}.\n",
        "multi_fact": "Enregistrement {idx}: le projet {key} stocke la valeur {value} pour le slot {slot}.\n",
        "count_fact": "Enregistrement {idx}: la region {key} signale le type d'evenement {event}.\n",
        "filler": "Enregistrement de fond {idx}: le projet sans rapport {key} a le statut {status} et la somme {value}.\n",
        "single_query": "\nQuestion : quel est le code de verification du projet {key} ?\nReponse : {answer}\n",
        "multi_query": "\nQuestion : quelles valeurs le projet {key} stocke-t-il dans les slots A et B ?\nReponse : {answer}\n",
        "count_query": "\nQuestion : combien d'enregistrements ALERT sont signales pour la region {key} ?\nReponse : {answer}\n",
        "no_answer_query": "\nQuestion : quel est le code de verification du projet {key} ?\nReponse : NOT_FOUND\n",
    },
}

KEYS = [
    "aurora", "birch", "cobalt", "delta", "ember", "fjord", "granite", "harbor",
    "iris", "juniper", "kelvin", "lumen", "meridian", "nimbus", "onyx", "polar",
    "quartz", "raven", "saffron", "tundra", "umbra", "vector", "willow", "zenith",
]
STATUSES = ["green", "amber", "blue", "silver", "paused", "closed", "pending"]
TASKS = ["single_key", "multi_key", "aggregation", "no_answer"]


class TokenCounter:
    def __init__(self, model: str | None, approx_chars_per_token: float) -> None:
        self.approx_chars_per_token = approx_chars_per_token
        self.tokenizer = None
        if model:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model)

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            return max(1, int(len(text) / self.approx_chars_per_token))
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def code(rng: random.Random) -> str:
    return str(rng.randrange(1_000_000, 9_999_999))


def key(rng: random.Random) -> str:
    return rng.choice(KEYS) + f"-{rng.randrange(1000, 9999)}"


def make_evidence(
    template: dict[str, str],
    rng: random.Random,
    task: str,
) -> tuple[str, str, dict[str, Any]]:
    needle_key = key(rng)
    if task == "single_key":
        answer = code(rng)
        evidence = template["single_fact"].format(idx=0, key=needle_key, value=answer)
        metadata = {"needle_key": needle_key, "answer": answer}
    elif task == "multi_key":
        value_a = code(rng)
        value_b = code(rng)
        answer = f"A={value_a}; B={value_b}"
        evidence = "".join(
            [
                template["multi_fact"].format(idx=0, key=needle_key, value=value_a, slot="A"),
                template["multi_fact"].format(idx=1, key=needle_key, value=value_b, slot="B"),
            ]
        )
        metadata = {"needle_key": needle_key, "answer": answer, "values": {"A": value_a, "B": value_b}}
    elif task == "aggregation":
        count = rng.randrange(3, 9)
        answer = str(count)
        records = [
            template["count_fact"].format(idx=idx, key=needle_key, event="ALERT")
            for idx in range(count)
        ]
        for idx in range(count, count + 8):
            records.append(template["count_fact"].format(idx=idx, key=key(rng), event="NOTICE"))
        rng.shuffle(records)
        evidence = "".join(records)
        metadata = {"needle_key": needle_key, "answer": answer, "alert_count": count}
    elif task == "no_answer":
        answer = "NOT_FOUND"
        evidence = "".join(
            template["single_fact"].format(idx=idx, key=key(rng), value=code(rng))
            for idx in range(16)
        )
        metadata = {"needle_key": needle_key, "answer": answer}
    else:
        raise ValueError(f"unsupported task: {task}")

    # Add local distractors close to the evidence so lexical matching alone is
    # not enough; the query still needs the exact key.
    distractors = "".join(
        template["single_fact"].format(idx=idx + 100, key=key(rng), value=code(rng))
        for idx in range(12)
    )
    return evidence + distractors, answer, metadata


def make_query(template: dict[str, str], task: str, needle_key: str, answer: str) -> str:
    if task == "single_key":
        return template["single_query"].format(key=needle_key, answer=answer)
    if task == "multi_key":
        return template["multi_query"].format(key=needle_key, answer=answer)
    if task == "aggregation":
        return template["count_query"].format(key=needle_key, answer=answer)
    if task == "no_answer":
        return template["no_answer_query"].format(key=needle_key)
    raise ValueError(f"unsupported task: {task}")


def filler_line(template: dict[str, str], rng: random.Random, idx: int) -> str:
    return template["filler"].format(
        idx=idx,
        key=key(rng),
        status=rng.choice(STATUSES),
        value=code(rng),
    )


def build_example(
    counter: TokenCounter,
    rng: random.Random,
    lang: str,
    task: str,
    target_tokens: int,
    depth: float,
    pose_max_context: int,
    split: str,
    measure_final: bool,
) -> dict[str, Any]:
    template = LANG[lang]
    evidence, answer, metadata = make_evidence(template, rng, task)
    query = make_query(template, task, metadata["needle_key"], answer)
    preamble = template["preamble"]

    sample_filler = filler_line(template, rng, 999999)
    filler_tokens = max(1, counter.count(sample_filler))
    fixed_tokens = counter.count(preamble + evidence + query)
    filler_count = max(0, (target_tokens - fixed_tokens) // filler_tokens + 2)
    before_count = min(filler_count, max(0, int(round(filler_count * depth))))
    after_count = max(0, filler_count - before_count)

    before = "".join(filler_line(template, rng, idx) for idx in range(1, before_count + 1))
    after = "".join(
        filler_line(template, rng, idx)
        for idx in range(before_count + 1, before_count + after_count + 1)
    )
    text = preamble + before + evidence + after + query
    estimated_tokens = fixed_tokens + filler_count * filler_tokens
    actual_tokens = counter.count(text) if measure_final else None

    item = {
        "text": text,
        "lang": lang,
        "split": split,
        "task": task,
        "target_tokens": target_tokens,
        "estimated_tokens": estimated_tokens,
        "actual_tokens": actual_tokens,
        "depth": depth,
        "answer": answer,
        "needle_key": metadata["needle_key"],
        "pose": {
            "physical_context_tokens": target_tokens,
            "max_position_tokens": pose_max_context,
            "position_id_strategy": "sample_skip_offsets_during_training",
        },
    }
    item.update({f"meta_{k}": v for k, v in metadata.items() if k not in {"answer", "needle_key"}})
    return item


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="oellm_2m_curriculum")
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--lengths", default="32768,65536,131072,262144")
    parser.add_argument("--examples-per-length", type=int, default=12)
    parser.add_argument("--languages", default="en,sv,de,fr")
    parser.add_argument("--tasks", default="single_key,multi_key,aggregation,no_answer")
    parser.add_argument("--depths", default="0.05,0.5,0.9")
    parser.add_argument("--pose-max-context", type=int, default=2_097_152)
    parser.add_argument("--model", default=None, help="Optional tokenizer model/path for token counting")
    parser.add_argument("--approx-chars-per-token", type=float, default=3.8)
    parser.add_argument("--measure-final", action="store_true",
                        help="Tokenize every final text. Accurate but slow for 1M/2M examples.")
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()

    lengths = parse_csv_ints(args.lengths)
    languages = parse_csv_strings(args.languages)
    tasks = parse_csv_strings(args.tasks)
    depths = parse_csv_floats(args.depths)
    unknown_langs = sorted(set(languages) - set(LANG))
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if unknown_langs:
        raise SystemExit(f"Unsupported languages: {', '.join(unknown_langs)}")
    if unknown_tasks:
        raise SystemExit(f"Unsupported tasks: {', '.join(unknown_tasks)}")

    rng = random.Random(args.seed)
    counter = TokenCounter(args.model, args.approx_chars_per_token)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "prefix": args.prefix,
        "split": args.split,
        "lengths": lengths,
        "examples_per_length": args.examples_per_length,
        "languages": languages,
        "tasks": tasks,
        "depths": depths,
        "pose_max_context": args.pose_max_context,
        "token_counter": args.model or f"approx_chars_per_token={args.approx_chars_per_token}",
        "shards": [],
    }

    for target_tokens in lengths:
        rows = []
        for idx in range(args.examples_per_length):
            lang = languages[idx % len(languages)]
            task = tasks[(idx // len(languages)) % len(tasks)]
            depth = depths[idx % len(depths)]
            rows.append(
                build_example(
                    counter=counter,
                    rng=rng,
                    lang=lang,
                    task=task,
                    target_tokens=target_tokens,
                    depth=depth,
                    pose_max_context=args.pose_max_context,
                    split=args.split,
                    measure_final=args.measure_final,
                )
            )

        shard = output_dir / f"{args.prefix}_{args.split}_{target_tokens}.jsonl"
        write_jsonl(shard, rows)
        token_values = [
            row["actual_tokens"] if row["actual_tokens"] is not None else row["estimated_tokens"]
            for row in rows
        ]
        manifest["shards"].append(
            {
                "path": str(shard),
                "target_tokens": target_tokens,
                "examples": len(rows),
                "min_tokens": min(token_values),
                "max_tokens": max(token_values),
                "mean_tokens": statistics.mean(token_values),
            }
        )
        print(json.dumps(manifest["shards"][-1], indent=2), flush=True)

    manifest_path = output_dir / f"{args.prefix}_{args.split}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
