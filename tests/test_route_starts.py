"""Tests for the DQN route-start extraction gates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plaiground.data.route_starts import (
    _valid_state,
    extract_checkpoint_routes,
)


def _analysis(size: int = 8, time_to_exit: float = 1.0) -> dict:
    grid = [[time_to_exit] * size for _ in range(size)]
    return {"time_to_exit": grid}


def _checkpoint(**overrides) -> dict:
    checkpoint = {
        "cell": [4, 7],
        "facing": 1,
        "on_floor": True,
        "state_name": "Move",
        "powerup_stack": [-1, -1, -1],
        "run_time": 1.2,
        "best_run_time": 3.0,
        "run_active": True,
        "started_walking": True,
    }
    checkpoint.update(overrides)
    return checkpoint


class RouteStartTest(unittest.TestCase):
    def test_valid_state_requires_grounded_time_and_stack(self) -> None:
        self.assertTrue(_valid_state(_checkpoint()))
        self.assertFalse(_valid_state(_checkpoint(on_floor=False)))
        self.assertFalse(_valid_state(_checkpoint(state_name="Jump")))
        self.assertFalse(_valid_state(_checkpoint(run_time=0.0)))
        self.assertFalse(
            _valid_state(_checkpoint(powerup_stack=[0], world_state=None))
        )

    def test_checkpoint_archive_builds_canonical_plus_state_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots = root / "snapshots.json"
            snapshots.write_text(
                json.dumps([_checkpoint()]), encoding="utf-8"
            )
            routes = [
                {
                    "route_signature": "sig",
                    "run_time": 2.0,
                    "checkpoint_file": "snapshots.json",
                }
            ]
            payload = extract_checkpoint_routes(
                routes,
                root / "routes.json",
                _analysis(),
                root / "run",
                "1-1",
                0.5,
            )
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload["train_spawns"], 1)
            self.assertTrue(payload["train"][0]["canonical"])
            self.assertEqual(payload["train"][1]["cell"], [4, 7])
            self.assertTrue(
                (root / "run" / "spawns" / "1-1_start_pool.json").exists()
            )
            self.assertTrue(
                (root / "run" / "spawns" / "1-1_spawn_pool.json").exists()
            )

    def test_low_time_to_exit_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "snapshots.json").write_text(
                json.dumps([_checkpoint()]), encoding="utf-8"
            )
            routes = [
                {
                    "route_signature": "sig",
                    "run_time": 2.0,
                    "checkpoint_file": "snapshots.json",
                }
            ]
            payload = extract_checkpoint_routes(
                routes,
                root / "routes.json",
                _analysis(time_to_exit=0.1),
                root / "run",
                "1-1",
                0.5,
            )
            self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
