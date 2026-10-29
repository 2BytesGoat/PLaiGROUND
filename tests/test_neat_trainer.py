"""Tests for the optional NEAT speed-selection fitness term."""

from __future__ import annotations

import unittest

import numpy as np

from plaiground.algorithms.genetic.neat.trainer import apply_time_penalty


class TimePenaltyTest(unittest.TestCase):
    def test_zero_weight_is_a_noop(self) -> None:
        scores = np.array([561.0, 570.0, -100.0])
        times = np.array([4.68, 7.12, np.inf])
        adjusted = apply_time_penalty(scores, times, 0.0)
        np.testing.assert_array_equal(adjusted, scores)

    def test_penalizes_finishers_only(self) -> None:
        scores = np.array([561.0, 570.0, -100.0, 400.0], dtype=np.float32)
        times = np.array([4.68, 7.12, np.inf, np.inf])
        adjusted = apply_time_penalty(scores, times, 20.0)
        self.assertAlmostEqual(float(adjusted[0]), 561.0 - 20.0 * 4.68)
        self.assertAlmostEqual(float(adjusted[1]), 570.0 - 20.0 * 7.12)
        self.assertEqual(float(adjusted[2]), -100.0)
        self.assertEqual(float(adjusted[3]), 400.0)

    def test_does_not_mutate_input_scores(self) -> None:
        scores = np.array([561.0, 570.0])
        times = np.array([4.68, np.inf])
        apply_time_penalty(scores, times, 20.0)
        np.testing.assert_array_equal(scores, np.array([561.0, 570.0]))

    def test_negative_weight_is_a_noop(self) -> None:
        scores = np.array([561.0])
        times = np.array([4.68])
        adjusted = apply_time_penalty(scores, times, -5.0)
        np.testing.assert_array_equal(adjusted, scores)


if __name__ == "__main__":
    unittest.main()
