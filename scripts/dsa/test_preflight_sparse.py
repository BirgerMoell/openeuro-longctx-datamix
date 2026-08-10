#!/usr/bin/env python3
"""Small regression tests for the fail-closed sparse launcher preflight."""

from pathlib import Path
import tempfile
import unittest

from lumi.preflight_sparse import (
    validate_mid_level_dataset_surplus,
    validate_training_launcher,
)


class SparsePreflightTest(unittest.TestCase):
    def test_dataset_surplus_rejects_known_unsafe_values(self):
        for value in (float("nan"), float("inf"), -1.0, 0.005, 0.499):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                validate_mid_level_dataset_surplus(value)

    def test_dataset_surplus_accepts_current_margin(self):
        validate_mid_level_dataset_surplus(0.5)
        validate_mid_level_dataset_surplus(1.0)

    def test_launcher_requires_surplus_and_full_recompute(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "train.sh"
            launcher.write_text(
                "--recompute-granularity full --recompute-method uniform "
                "--mid-level-dataset-surplus $MID_LEVEL_DATASET_SURPLUS\n"
            )
            validate_training_launcher(launcher)

            launcher.write_text(
                "--recompute-granularity full --recompute-method uniform\n"
            )
            with self.assertRaises(RuntimeError):
                validate_training_launcher(launcher)


if __name__ == "__main__":
    unittest.main()
