"""Tests for the DQN anchor collector summary."""

from __future__ import annotations

import contextlib
import io
import unittest

from tools.collect_dqn_buffer import _fmt_stats


class CollectDqnBufferStatsTest(unittest.TestCase):
    def test_best_time_ignores_deaths(self) -> None:
        rows = [
            {
                "start_kind": "canonical",
                "run_time": 2.4333,
                "finished": False,
                "timed_out": False,
            },
            {
                "start_kind": "canonical",
                "run_time": 4.6833,
                "finished": True,
                "timed_out": False,
            },
            {
                "start_kind": "canonical",
                "run_time": 6.0,
                "finished": False,
                "timed_out": True,
            },
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            stats = _fmt_stats("test", rows)

        self.assertEqual(stats["best_time"], 4.6833)
        self.assertEqual(stats["best_finished_time"], 4.6833)
        self.assertEqual(stats["best_death_time"], 2.4333)
        self.assertEqual(stats["canonical_finished"], 1)
        self.assertEqual(stats["canonical_deaths"], 1)
        self.assertEqual(stats["timeouts"], 1)
        self.assertEqual(stats["median_time"], 4.6833)
        self.assertEqual(stats["below_1_75"], 0)

    def test_no_finished_episode_has_no_best_time(self) -> None:
        rows = [
            {
                "start_kind": "canonical",
                "run_time": 2.4333,
                "finished": False,
                "timed_out": False,
            }
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            stats = _fmt_stats("test", rows)

        self.assertIsNone(stats["best_time"])
        self.assertIsNone(stats["best_finished_time"])
        self.assertEqual(stats["best_death_time"], 2.4333)
        self.assertEqual(stats["canonical_finished"], 0)


if __name__ == "__main__":
    unittest.main()
