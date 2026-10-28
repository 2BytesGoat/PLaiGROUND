"""Tests for the runtime start-pool spawn guard."""

from __future__ import annotations

import unittest

from plaiground.envs.single_agent import SingleAgentGodotEnv


class SpawnValidationTest(unittest.TestCase):
    def test_accepts_matching_spawn_cell(self) -> None:
        SingleAgentGodotEnv._assert_spawn_applied(
            {"cell": [4, 7], "facing": 1},
            {"spawn_cell": [4, 7]},
        )

    def test_rejects_silent_fallback(self) -> None:
        with self.assertRaises(ValueError):
            SingleAgentGodotEnv._assert_spawn_applied(
                {"cell": [7, 7], "facing": 1},
                {"spawn_cell": [4, 7]},
            )

    def test_does_not_require_spawn_cell_for_canonical(self) -> None:
        SingleAgentGodotEnv._assert_spawn_applied(
            {"canonical": True},
            {"spawn_cell": [10, 14]},
        )


if __name__ == "__main__":
    unittest.main()
