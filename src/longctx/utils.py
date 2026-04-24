"""Small shared helpers used across commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def check_disk_space(path: str | Path = "/") -> dict[str, float]:
    total, used, free = shutil.disk_usage(str(path))
    return {"total_gb": total / 1e9, "used_gb": used / 1e9, "free_gb": free / 1e9}


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def human_bytes(n: float) -> str:
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n/div:.1f} {unit}"
    return f"{int(n)} B"
