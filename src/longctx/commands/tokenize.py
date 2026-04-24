"""Tokenize JSONL into Megatron-format .bin/.idx via the fork's preprocess_data.py.

Shells out to:
  python $MEGATRON_LM/tools/preprocess_data.py \
      --input <jsonl> \
      --output-prefix <bin_dir>/<lc>_text_document \
      --tokenizer-type ... --tokenizer-model ... --append-eod \
      --workers N --partitions K

This matches the OpenEuroLLM NVIDIA-Megatron-LM fork at
https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM, which accepts either
--data-path <prefix> or --data-path <w1> <p1> <w2> <p2> ... for mixtures.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _resolve_megatron_path(user_value: str | None) -> Path:
    env_val = os.environ.get("MEGATRON_LM")
    candidate = user_value or env_val
    if not candidate:
        print("[error] --megatron-path not set and $MEGATRON_LM env var is empty.\n"
              "        Point it at your NVIDIA-Megatron-LM checkout, e.g.:\n"
              "        git clone https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM\n"
              "        export MEGATRON_LM=$PWD/NVIDIA-Megatron-LM")
        sys.exit(2)
    path = Path(candidate).expanduser().resolve()
    script = path / "tools" / "preprocess_data.py"
    if not script.exists():
        print(f"[error] preprocess_data.py not found at {script}")
        sys.exit(2)
    return path


def cmd_tokenize(args) -> None:
    megatron = _resolve_megatron_path(args.megatron_path)
    script = megatron / "tools" / "preprocess_data.py"

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.languages:
        wanted = [l.strip() for l in args.languages.split(",") if l.strip()]
        jsonl_files = [in_dir / f"{lc}.jsonl" for lc in wanted]
    else:
        jsonl_files = sorted(in_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"[error] No JSONL files in {in_dir}. Run `longctx convert` (or filter-long) first.")
        sys.exit(1)

    report: dict[str, dict] = {}

    for jf in jsonl_files:
        lc = jf.stem
        if not jf.exists():
            print(f"[skip] {lc}: {jf} missing")
            continue

        # Megatron preprocess_data.py writes {output_prefix}_{json_key}_document.{bin,idx},
        # i.e. passing `out_dir/lc` yields `out_dir/lc_text_document.{bin,idx}`.
        bare_prefix = out_dir / lc
        full_prefix = out_dir / f"{lc}_text_document"
        cmd = [
            sys.executable, str(script),
            "--input", str(jf),
            "--output-prefix", str(bare_prefix),
            "--json-keys", "text",
            "--tokenizer-type", args.tokenizer_type,
            "--tokenizer-model", args.tokenizer_model,
            "--workers", str(args.workers),
            "--partitions", str(args.partitions),
        ]
        if args.append_eod:
            cmd.append("--append-eod")
        if args.vocab_size is not None:
            cmd.extend(["--vocab-size", str(args.vocab_size)])

        print(f"\n[{lc}] tokenizing → {full_prefix}.{{bin,idx}}")
        print("  $ " + " ".join(cmd))
        if args.dry_run:
            report[lc] = {"cmd": cmd, "dry_run": True}
            continue

        try:
            subprocess.run(cmd, check=True, cwd=str(megatron))
        except subprocess.CalledProcessError as e:
            print(f"[error] {lc} tokenization failed (exit {e.returncode})")
            report[lc] = {"cmd": cmd, "error": f"exit {e.returncode}"}
            continue

        bin_path = Path(f"{full_prefix}.bin")
        idx_path = Path(f"{full_prefix}.idx")
        report[lc] = {
            "prefix": str(full_prefix),
            "bin_mb": bin_path.stat().st_size / 1e6 if bin_path.exists() else None,
            "idx_mb": idx_path.stat().st_size / 1e6 if idx_path.exists() else None,
        }

    with open(out_dir / "tokenize_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTokenize summary: {out_dir / 'tokenize_summary.json'}")
