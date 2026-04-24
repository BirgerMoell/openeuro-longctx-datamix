"""End-to-end convenience: download + convert (+ optional filter-long).

Does not run tokenize or mix — those need a tokenizer path / GPU workers and
are best launched explicitly once the JSONL looks right.
"""

from __future__ import annotations

from longctx.commands.convert import cmd_convert
from longctx.commands.download import cmd_download
from longctx.commands.filter_long import cmd_filter_long


def cmd_run(args) -> None:
    cmd_download(args)
    cmd_convert(args)
    if getattr(args, "filter_long", False):
        cmd_filter_long(args)
