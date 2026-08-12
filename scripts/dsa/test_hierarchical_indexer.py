#!/usr/bin/env python3
"""CPU regressions for the long-context block router's deployed geometries."""

import unittest

import torch

from hierarchical_indexer import hierarchical_block_topk


class HierarchicalIndexerTest(unittest.TestCase):
    def test_msa_calibration_geometry_is_causal_and_complete(self):
        block = 128
        routed = 15
        topk = block * (1 + routed)
        sequence = topk * 2
        batch = 1
        index_heads = 2
        head_dim = 8

        torch.manual_seed(7)
        query = torch.randn(sequence, batch, index_heads, head_dim)
        weights = torch.randn(sequence, batch, index_heads)
        key = torch.randn(sequence, batch, head_dim)
        positions = torch.arange(sequence)

        scores, indices = hierarchical_block_topk(
            query,
            weights,
            key,
            positions,
            topk,
            block_size=block,
            routed_blocks=routed,
        )

        self.assertEqual(scores.shape, (batch, sequence, topk))
        self.assertEqual(indices.shape, (batch, sequence, topk))
        self.assertEqual(indices.dtype, torch.int32)
        self.assertTrue(torch.all((indices < 0) | (indices <= positions.view(1, -1, 1))))
        self.assertTrue(torch.all((indices >= 0).sum(dim=-1) > 0))

        # Once enough history exists, the deployed geometry must retain all
        # 2,048 positions rather than silently padding the sparse budget.
        self.assertEqual(int((indices[0, -1] >= 0).sum()), topk)

    def test_topk_must_cover_current_and_routed_blocks(self):
        block = 8
        query = torch.randn(16, 1, 1, 4)
        weights = torch.randn(16, 1, 1)
        key = torch.randn(16, 1, 4)
        with self.assertRaises(ValueError):
            hierarchical_block_topk(
                query,
                weights,
                key,
                torch.arange(16),
                topk=15,
                block_size=block,
                routed_blocks=1,
            )


if __name__ == "__main__":
    unittest.main()
