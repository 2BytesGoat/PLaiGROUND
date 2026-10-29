"""Tests for the config-driven run resolver."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plaiground.experiments.runner import (
    load_run,
    resolve_path,
    train_start_pool,
)

ROOT = Path(__file__).resolve().parent.parent


class RunResolverTest(unittest.TestCase):
    def test_loads_existing_ppo_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = load_run(
                ROOT / "experiments/0001_ppo_clean_start_1_1",
                output_dir=Path(tmp) / "run",
            )
            self.assertEqual(run.algorithm_name, "ppo")
            self.assertEqual(run.level, "1-1")
            self.assertEqual(run.levels, ["1-1"])
            self.assertEqual(run.n_envs, 8)
            self.assertEqual(run.obs_encoding, "lean")
            self.assertEqual(run.preprocessing, "lean")
            self.assertEqual(run.hidden, [512, 512])
            self.assertEqual(run.model_path.name, "ppo_pilot.zip")
            self.assertEqual(run.output_dir, (Path(tmp) / "run").resolve())
            self.assertFalse(run.reassert_held_jump)

    def test_loads_dqn_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = load_run(
                ROOT / "experiments/0003_dqn_clean_start_1_1",
                output_dir=Path(tmp) / "run",
            )
            self.assertEqual(run.algorithm_name, "dqn")
            self.assertEqual(run.obs_encoding, "lean")
            self.assertEqual(run.preprocessing, "lean")
            self.assertEqual(run.hidden, [512, 512])
            self.assertEqual(run.activation, "relu")
            self.assertEqual(run.model_path.name, "dqn_final.zip")
            self.assertEqual(run.checkpoint_dir.name, "checkpoints")
            self.assertTrue(run.evaluation["train_pool"])
            self.assertTrue(run.evaluation["holdout_pool"])

    def test_held_stick_encoding_defaults_to_reassert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = load_run(
                ROOT / "experiments/0058_ddqn_fresh_1_1_lean_stick",
                output_dir=Path(tmp) / "run",
            )
            self.assertEqual(run.obs_encoding, "lean_stick")
            self.assertTrue(run.reassert_held_jump)

    def test_timesteps_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = load_run(
                ROOT / "experiments/0001_ppo_clean_start_1_1",
                timesteps=1234,
                output_dir=Path(tmp) / "run",
            )
            self.assertEqual(run.timesteps, 1234)

    def test_level_and_speedup_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = load_run(
                ROOT / "experiments/0079_show_chain_1_1_to_1_4_0073",
                output_dir=Path(tmp) / "run",
                level="1-4",
                speedup=2.0,
            )
            self.assertEqual(run.level, "1-4")
            self.assertEqual(run.levels, ["1-4"])
            self.assertEqual(run.speedup, 2.0)

    def test_watch_manifest_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = load_run(
                ROOT / "experiments/0079_show_chain_1_1_to_1_4_0073",
                output_dir=Path(tmp) / "run",
            )
            self.assertEqual(
                run.levels, ["1-1", "1-2", "1-3", "1-4"]
            )
            self.assertEqual(run.speedup, 1.0)
            self.assertTrue(run.reassert_held_jump)

    def test_resolve_path_is_repository_relative(self) -> None:
        resolved = resolve_path("experiments/0001_ppo_clean_start_1_1")
        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved, ROOT / "experiments/0001_ppo_clean_start_1_1")

    def test_loads_level_tagged_start_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_path = root / "pool.json"
            pool_path.write_text(
                json.dumps(
                    {
                        "level": "1-1",
                        "starts": [
                            {"canonical": True, "weight": 0.7},
                            {
                                "cell": [4, 7],
                                "facing": 1,
                                "run_time": 1.0,
                                "weight": 0.3,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[experiment]",
                        'id = "9001"',
                        "[workflow]",
                        'name = "reinforcement.ppo"',
                        "[env]",
                        'level = "1-1"',
                        "[observation]",
                        'encoding = "raw"',
                        'preprocessing = "none"',
                        "[algorithm]",
                        'name = "ppo"',
                        "timesteps = 10",
                        "[data]",
                        f'train_pool = "{pool_path}"',
                        "[output]",
                        f'dir = "{root / "run"}"',
                    ]
                )
            )
            run = load_run(config_path)
            pool = train_start_pool(run)
            self.assertEqual(pool[0]["level"], "1-1")
            self.assertEqual(pool[1]["level"], "1-1")


if __name__ == "__main__":
    unittest.main()
