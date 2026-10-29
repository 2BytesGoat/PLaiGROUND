"""Tests for extracting start-aligned expert transitions."""

from __future__ import annotations

import unittest

import numpy as np

from tools.build_start_expert_buffer import _match_starts


class StartExpertBufferTest(unittest.TestCase):
    def _data(self):
        return {
            "route_signature": np.asarray(["a"] * 5 + ["b"] * 3, dtype="U8"),
            "step": np.asarray([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64),
            "route_success": np.asarray(
                [True] * 5 + [True] * 3, dtype=bool
            ),
        }

    def test_start_action_only(self) -> None:
        starts = [
            {"canonical": True},
            {
                "cell": [2, 3],
                "facing": 1,
                "source_route": "a",
                "run_time": 2 / 15,
            },
        ]
        selected, log = _match_starts(
            starts, self._data(), action_repeat=4
        )
        self.assertEqual(selected, [2])
        self.assertTrue(log[0]["matched"])
        self.assertEqual(log[0]["transitions_added"], 1)

    def test_suffix_includes_remaining_route(self) -> None:
        starts = [
            {
                "cell": [2, 3],
                "facing": 1,
                "source_route": "a",
                "run_time": 2 / 15,
            }
        ]
        selected, log = _match_starts(
            starts, self._data(), action_repeat=4, suffix=True
        )
        self.assertEqual(selected, [2, 3, 4])
        self.assertEqual(log[0]["transitions_added"], 3)

    def test_missing_source_route_is_reported(self) -> None:
        starts = [
            {
                "cell": [2, 3],
                "facing": 1,
                "source_route": "missing",
                "run_time": 2 / 15,
            }
        ]
        selected, log = _match_starts(
            starts, self._data(), action_repeat=4
        )
        self.assertEqual(selected, [])
        self.assertFalse(log[0]["matched"])


if __name__ == "__main__":
    unittest.main()
