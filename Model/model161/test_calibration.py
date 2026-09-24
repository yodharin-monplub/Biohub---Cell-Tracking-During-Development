from __future__ import annotations

import unittest

import numpy as np

from model161.solve_target import rank_map


class RankMapTest(unittest.TestCase):
    def test_reassigns_exact_raw_scores_by_calibrated_rank(self):
        raw = np.array([0.8, 0.2, 0.6, 0.1])
        calibrated = np.array([0.1, 0.8, 0.2, 0.9])
        mapped = rank_map(raw, calibrated)
        np.testing.assert_array_equal(mapped, [0.1, 0.6, 0.2, 0.8])
        np.testing.assert_array_equal(np.sort(mapped), np.sort(raw))
        self.assertEqual(int((mapped >= 0.4).sum()), int((raw >= 0.4).sum()))

    def test_rejects_nonfinite_and_shape_mismatch(self):
        with self.assertRaises(ValueError):
            rank_map(np.array([0.3]), np.array([0.1, 0.2]))
        with self.assertRaises(ValueError):
            rank_map(np.array([0.3]), np.array([np.nan]))


if __name__ == "__main__":
    unittest.main()
