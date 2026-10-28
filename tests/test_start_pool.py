"""Tests for clean-start pool construction from filtered routes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plaiground.data.start_pool import build_start_pools


def route(signature: str, checkpoint: str, run_time: float) -> dict:
    return {
        "route_signature": signature,
        "run_time": run_time,
        "cluster_id": "c000",
        "checkpoint_file": checkpoint,
        "split": "train",
    }


class StartPoolTest(unittest.TestCase):
    def test_train_and_holdout_pools_stay_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive"
            filtered = root / "filtered"
            output = root / "pool"
            (archive / "checkpoints").mkdir(parents=True)
            filtered.mkdir()
            analysis = {"time_to_exit": [[2.0, 2.0], [2.0, 2.0]]}
            analysis_path = root / "analysis.json"
            analysis_path.write_text(json.dumps(analysis))

            train_checkpoint = archive / "checkpoints" / "train.json"
            holdout_checkpoint = archive / "checkpoints" / "holdout.json"
            state = {
                "cell": [1, 1],
                "facing": 1,
                "position": [10.0, 20.0],
                "velocity": [0.0, 0.0],
                "state_name": "Move",
                "powerup_stack": [-1, -1, -1],
                "run_time": 1.0,
                "run_active": True,
                "started_walking": True,
                "on_floor": True,
                "world_state": {},
            }
            train_checkpoint.write_text(
                json.dumps(
                    [
                        {**state, "run_time": 1.2},
                        {**state, "run_time": 0.9},
                        {**state, "on_floor": False},
                    ]
                )
            )
            holdout_checkpoint.write_text(json.dumps([state]))
            (archive / "routes.json").write_text("{}")
            (filtered / "train_clean_routes.json").write_text(
                json.dumps([route("train", "checkpoints/train.json", 2.0)])
            )
            (filtered / "holdout_clean_routes.json").write_text(
                json.dumps([route("holdout", "checkpoints/holdout.json", 2.1)])
            )

            result = build_start_pools(
                filtered_dir=filtered,
                archive_dir=archive,
                analysis_path=analysis_path,
                output_dir=output,
                min_time_to_exit=0.5,
            )
            summary = result["summary"]
            self.assertEqual(summary["train_states"], 1)
            self.assertEqual(summary["holdout_states"], 1)
            self.assertEqual(summary["shared_state_keys"], 1)
            self.assertEqual(result["train_pool"][0]["weight"], 0.7)
            self.assertEqual(result["train_pool"][1]["weight"], 0.3)
            self.assertEqual(result["train_pool"][1]["run_time"], 0.9)
            self.assertFalse(
                any(item.get("canonical") for item in result["holdout_pool"])
            )
            self.assertTrue((output / "train_start_pool.json").exists())
            self.assertTrue((output / "holdout_start_pool.json").exists())

    def test_rejects_invalid_weight_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_start_pools(
                    filtered_dir=Path(tmp),
                    archive_dir=Path(tmp),
                    analysis_path=Path(tmp) / "missing.json",
                    output_dir=Path(tmp) / "pool",
                    canonical_weight=0.5,
                    state_weight=0.2,
                )


if __name__ == "__main__":
    unittest.main()
