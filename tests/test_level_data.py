"""Tests for level route-tier classification and pool preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plaiground.data.level_data import (
    ALL_SUCCESS,
    NEAR_CLEAN,
    STRICT_CLEAN,
    LevelDataError,
    classify_all_routes,
    prepare_level_data,
    select_quality,
)


def make_route(
    signature: str,
    run_time: float,
    cells: list[list[int]],
    *,
    revisit: float,
    backtracks: int,
    flips: float,
    checkpoint: str | None = None,
) -> dict:
    return {
        "route_signature": signature,
        "run_time": run_time,
        "path_cells": cells,
        "path_length": len(cells),
        "revisit_ratio": revisit,
        "backtracks": backtracks,
        "action_flip_rate": flips,
        "checkpoint_file": checkpoint or f"checkpoints/{signature}.json",
        "clean": False,
    }


def state(cell: list[int], run_time: float = 1.0) -> dict:
    return {
        "cell": cell,
        "facing": 1,
        "position": [24.0, 48.0],
        "velocity": [0.0, 0.0],
        "state_name": "Move",
        "powerup_stack": [-1, -1, -1],
        "run_time": run_time,
        "run_active": True,
        "started_walking": True,
        "on_floor": True,
        "world_state": {},
    }


ANALYSIS = {
    "symbols": [
        ["W", "W", "W", "W", "W"],
        ["W", ".", ".", ".", "W"],
        ["W", ".", ".", ".", "W"],
        ["W", "W", "W", "W", "W"],
        ["W", "W", "W", "W", "W"],
    ],
    "time_to_exit": [
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
    ],
}


def strict_routes() -> list[dict]:
    return [
        make_route(
            "strict-a",
            2.0,
            [[1, 2], [2, 2], [3, 2], [2, 1], [1, 1]],
            revisit=0.0,
            backtracks=0,
            flips=0.1,
            checkpoint="checkpoints/strict-a.json",
        ),
        make_route(
            "strict-b",
            2.1,
            [[2, 2], [2, 3], [3, 3], [4, 3], [4, 2]],
            revisit=0.05,
            backtracks=0,
            flips=0.2,
            checkpoint="checkpoints/strict-b.json",
        ),
    ]


class LevelDataTest(unittest.TestCase):
    def test_classify_recomputes_clean_tiers_and_dedupes(self) -> None:
        slow = make_route(
            "strict-a",
            3.0,
            [[1, 2], [2, 2], [3, 2], [2, 1], [1, 1]],
            revisit=0.0,
            backtracks=0,
            flips=0.1,
        )
        near = make_route(
            "near",
            2.2,
            [[1, 2], [1, 1], [2, 1], [3, 1], [4, 1]],
            revisit=0.15,
            backtracks=1,
            flips=0.2,
        )
        all_success = make_route(
            "messy",
            2.4,
            [[1, 2], [1, 3], [1, 4], [2, 4], [3, 4]],
            revisit=0.5,
            backtracks=3,
            flips=0.9,
        )
        tiers, summary = classify_all_routes(
            [slow, strict_routes()[0], near, all_success]
        )
        self.assertEqual(summary["input_routes"], 4)
        self.assertEqual(summary["unique_routes"], 3)
        self.assertEqual(len(tiers[STRICT_CLEAN]), 1)
        self.assertEqual(len(tiers[NEAR_CLEAN]), 2)
        self.assertEqual(len(tiers[ALL_SUCCESS]), 3)
        self.assertEqual(tiers[STRICT_CLEAN][0]["run_time"], 2.0)
        self.assertEqual(tiers[STRICT_CLEAN][0]["quality"], STRICT_CLEAN)
        self.assertNotIn("clean", tiers[STRICT_CLEAN][0])
        self.assertEqual(tiers[NEAR_CLEAN][1]["quality"], NEAR_CLEAN)

    def test_auto_quality_prefers_strict_then_near(self) -> None:
        summary = {
            "qualities": {
                STRICT_CLEAN: {"routes": 2},
                NEAR_CLEAN: {"routes": 5},
                ALL_SUCCESS: {"routes": 10},
            }
        }
        self.assertEqual(select_quality(summary, min_routes=1), STRICT_CLEAN)
        summary["qualities"][STRICT_CLEAN]["routes"] = 0
        self.assertEqual(select_quality(summary, min_routes=1), NEAR_CLEAN)
        summary["qualities"][NEAR_CLEAN]["routes"] = 0
        with self.assertRaises(LevelDataError):
            select_quality(summary, min_routes=1)

    def test_prepare_level_data_writes_tiers_pools_and_eval_union(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            archive = run_dir / "archive"
            checkpoints = archive / "checkpoints"
            checkpoints.mkdir(parents=True)
            (archive / "routes.json").write_text(
                json.dumps(
                    {
                        route["route_signature"]: route
                        for route in strict_routes()
                    }
                )
            )
            (checkpoints / "strict-a.json").write_text(
                json.dumps([state([1, 2])])
            )
            (checkpoints / "strict-b.json").write_text(
                json.dumps([state([2, 2])])
            )
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps(ANALYSIS))
            output = root / "out"

            summary = prepare_level_data(
                level="1-1",
                run_dir=run_dir,
                generation=7,
                analysis_path=analysis,
                output_root=output,
                quality="auto",
                min_routes=1,
                seed=0,
                holdout_fraction=0.25,
            )

            self.assertEqual(summary["selected_quality"], STRICT_CLEAN)
            self.assertIn("strict-clean routes", summary["selection_reason"])
            self.assertEqual(summary["quality"]["input_routes"], 2)
            self.assertTrue((output / "strict_clean_routes_g7.json").exists())
            self.assertTrue((output / "near_clean_routes_g7.json").exists())
            self.assertTrue((output / "all_success_routes_g7.json").exists())
            accepted = summary["accepted"]
            self.assertEqual(accepted["filter"]["filtered_routes"], 2)
            self.assertEqual(
                accepted["filter"]["train_routes"]
                + accepted["filter"]["holdout_routes"],
                2,
            )
            self.assertEqual(accepted["pool"]["train_states"], 1)
            self.assertEqual(accepted["pool"]["holdout_states"], 1)
            self.assertEqual(accepted["pool"]["unique_eval_states"], 2)
            self.assertTrue(
                (output / "pool_clean_g7" / "eval_clean_start_pool.json").exists()
            )
            self.assertIsNotNone(summary["all_success"])
            self.assertTrue(
                (
                    output
                    / "pool_all_success_g7"
                    / "eval_all_success_start_pool.json"
                ).exists()
            )
            self.assertTrue((output / "level_data_summary.json").exists())

    def test_prepare_level_data_can_fall_back_to_near_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            archive = run_dir / "archive"
            checkpoints = archive / "checkpoints"
            checkpoints.mkdir(parents=True)
            near = make_route(
                "near",
                2.2,
                [[1, 2], [1, 1], [2, 1], [3, 1], [4, 1]],
                revisit=0.15,
                backtracks=1,
                flips=0.2,
                checkpoint="checkpoints/near.json",
            )
            (archive / "routes.json").write_text(json.dumps({"near": near}))
            (checkpoints / "near.json").write_text(
                json.dumps([state([1, 2])])
            )
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps(ANALYSIS))
            output = root / "out"

            summary = prepare_level_data(
                level="1-1",
                run_dir=run_dir,
                generation=None,
                analysis_path=analysis,
                output_root=output,
                quality="auto",
                min_routes=1,
                seed=0,
            )

            self.assertEqual(summary["selected_quality"], NEAR_CLEAN)
            self.assertEqual(summary["selected_tag"], "near_clean")
            self.assertIn("near-clean fallback", summary["selection_reason"])
            self.assertTrue(
                (
                    output
                    / "pool_near_clean"
                    / "eval_near_clean_start_pool.json"
                ).exists()
            )

    def test_prepare_level_data_errors_when_no_clean_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            archive = run_dir / "archive"
            (archive / "checkpoints").mkdir(parents=True)
            route = make_route(
                "messy",
                2.4,
                [[1, 2], [1, 3], [1, 4], [2, 4], [3, 4]],
                revisit=0.5,
                backtracks=3,
                flips=0.9,
                checkpoint="checkpoints/messy.json",
            )
            (archive / "routes.json").write_text(json.dumps({"messy": route}))
            (archive / "checkpoints" / "messy.json").write_text(
                json.dumps([state([1, 2])])
            )
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps(ANALYSIS))
            with self.assertRaises(LevelDataError):
                prepare_level_data(
                    level="1-1",
                    run_dir=run_dir,
                    generation=1,
                    analysis_path=analysis,
                    output_root=root / "out",
                    quality="auto",
                    min_routes=1,
                )


if __name__ == "__main__":
    unittest.main()
