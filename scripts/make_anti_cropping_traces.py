#!/usr/bin/env python3
"""Build synthetic anti-cropping retrieval traces for long-context CLM.

Each example places the answer-bearing fact near the beginning of the context,
fills the middle with distractor facts, and repeats the query plus answer near
the end. This gives next-token supervision for retrieving early evidence.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


LANG = {
    "en": {
        "preamble": "Below is a long audit log. Use the earliest matching record.\n\n",
        "fact": "Record {idx}: the verification code for project {key} is {value}.\n",
        "query": "\nQuestion: What is the verification code for project {key}?\nAnswer: {value}\n",
        "filler": "Background note {idx}: this unrelated project has status {status} and checksum {value}.\n",
    },
    "sv": {
        "preamble": "Nedan finns en lång granskningslogg. Använd den tidigaste matchande posten.\n\n",
        "fact": "Post {idx}: verifieringskoden för projekt {key} är {value}.\n",
        "query": "\nFråga: Vilken verifieringskod har projekt {key}?\nSvar: {value}\n",
        "filler": "Bakgrundsnotis {idx}: detta orelaterade projekt har status {status} och kontrollsumma {value}.\n",
    },
    "de": {
        "preamble": "Unten steht ein langes Prüfprotokoll. Verwende den frühesten passenden Eintrag.\n\n",
        "fact": "Eintrag {idx}: der Prüfcode für Projekt {key} lautet {value}.\n",
        "query": "\nFrage: Wie lautet der Prüfcode für Projekt {key}?\nAntwort: {value}\n",
        "filler": "Hintergrundnotiz {idx}: dieses nicht verwandte Projekt hat Status {status} und Prüfsumme {value}.\n",
    },
    "fr": {
        "preamble": "Voici un long journal d'audit. Utilise le premier enregistrement correspondant.\n\n",
        "fact": "Enregistrement {idx}: le code de vérification du projet {key} est {value}.\n",
        "query": "\nQuestion : quel est le code de vérification du projet {key} ?\nRéponse : {value}\n",
        "filler": "Note de contexte {idx}: ce projet sans rapport a le statut {status} et la somme {value}.\n",
    },
}

KEYS = [
    "aurora", "birch", "cobalt", "delta", "ember", "fjord", "granite", "harbor",
    "iris", "juniper", "kelvin", "lumen", "meridian", "nimbus", "onyx", "polar",
    "quartz", "raven", "saffron", "tundra", "umbra", "vector", "willow", "zenith",
]

STATUSES = ["green", "amber", "blue", "silver", "paused", "closed", "pending"]
FACT_POSITIONS = ["start", "middle", "end"]


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_example(
    tokenizer,
    rng: random.Random,
    lang: str,
    target_tokens: int,
    fact_position: str,
) -> str:
    template = LANG[lang]
    key = rng.choice(KEYS) + f"-{rng.randrange(100, 999)}"
    value = str(rng.randrange(1_000_000, 9_999_999))

    evidence = template["fact"].format(idx=0, key=key, value=value)

    # Add several plausible distractors right after the true fact.
    for idx in range(1, 16):
        dkey = rng.choice(KEYS) + f"-{rng.randrange(100, 999)}"
        dval = str(rng.randrange(1_000_000, 9_999_999))
        evidence += template["fact"].format(idx=idx, key=dkey, value=dval)

    query = template["query"].format(key=key, value=value)

    sample_filler = template["filler"].format(idx=999, status="pending", value="1234567")
    filler_tokens = max(1, token_len(tokenizer, sample_filler))
    current_tokens = token_len(tokenizer, template["preamble"] + evidence + query)
    needed = max(0, target_tokens - current_tokens)
    filler_count = needed // filler_tokens + 2

    if fact_position == "start":
        before_fillers = 0
    elif fact_position == "middle":
        before_fillers = filler_count // 2
    elif fact_position == "end":
        before_fillers = int(filler_count * 0.85)
    else:
        raise ValueError(f"unsupported fact_position: {fact_position}")
    after_fillers = filler_count - before_fillers

    def filler(filler_idx: int) -> str:
        return template["filler"].format(
            idx=filler_idx,
            status=rng.choice(STATUSES),
            value=str(rng.randrange(1_000_000, 9_999_999)),
        )

    before = "".join(filler(idx) for idx in range(1, before_fillers + 1))
    after = "".join(
        filler(idx) for idx in range(before_fillers + 1, before_fillers + after_fillers + 1)
    )
    return template["preamble"] + before + evidence + after + query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openeurollm/datamix-2b-80-20")
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--target-tokens", type=int, default=7800)
    parser.add_argument("--languages", nargs="+", default=["en", "sv", "de", "fr"])
    parser.add_argument(
        "--fact-position",
        choices=[*FACT_POSITIONS, "cycle"],
        default="start",
        help="Where to place the answer-bearing evidence before the final query.",
    )
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lengths = []
    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(args.examples):
            lang = args.languages[idx % len(args.languages)]
            fact_position = (
                FACT_POSITIONS[idx % len(FACT_POSITIONS)]
                if args.fact_position == "cycle"
                else args.fact_position
            )
            text = build_example(tokenizer, rng, lang, args.target_tokens, fact_position)
            lengths.append(token_len(tokenizer, text))
            f.write(
                json.dumps(
                    {
                        "text": text,
                        "lang": lang,
                        "task": "single_key_retrieval",
                        "fact_position": fact_position,
                        "target_tokens": args.target_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        json.dumps(
            {
                "output": str(out_path),
                "examples": args.examples,
                "min_tokens": min(lengths),
                "max_tokens": max(lengths),
                "mean_tokens": sum(lengths) / len(lengths),
                "languages": args.languages,
                "fact_position": args.fact_position,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
