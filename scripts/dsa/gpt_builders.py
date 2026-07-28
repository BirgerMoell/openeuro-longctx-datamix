"""Megatron import shim for the repository-owned DSA builder overlay.

Put scripts/dsa before the Megatron checkout on PYTHONPATH. Megatron's
pretrain_gpt.py imports gpt_builders by module name, while the implementation
remains explicitly named gpt_builders_dsa.py in this repo.
"""

from gpt_builders_dsa import gpt_builder

__all__ = ["gpt_builder"]
