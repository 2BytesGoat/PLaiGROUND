"""Tests for the shared deterministic evaluation engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from plaiground.evaluation import rollout
from plaiground.experiments.runner import Run


class FakeModel:
    def predict(self, observations, deterministic=True):
        return np.zeros((len(observations), 1), dtype=np.int64), None


class FakeEnv:
    def __init__(self, count: int, finish_steps: dict[int, int]):
        self.count = count
        self.finish_steps = finish_steps
        self.time = 0
        self.reset_spawns = None

    def reset(self, spawns=None):
        self.time = 0
        self.reset_spawns = spawns
        observations = [
            {"obs": np.zeros(4, dtype=np.float32)} for _ in range(self.count)
        ]
        infos = [{} for _ in range(self.count)]
        for spawn in spawns or []:
            infos[spawn["agent"]]["spawn_cell"] = spawn["cell"]
        return observations, infos

    def step(self, actions):
        self.time += 1
        observations = []
        rewards = []
        dones = []
        infos = []
        for index in range(self.count):
            finish = self.finish_steps.get(index)
            done = finish is not None and self.time >= finish
            observations.append({"obs": np.zeros(4, dtype=np.float32)})
            rewards.append(1.0 if done else 0.0)
            dones.append(done)
            infos.append(
                {
                    "finished": bool(done),
                    # Deliberately zero: the evaluator must derive the time.
                    "run_time": 0.0 if done else 0.0,
                    "deaths": 0,
                }
            )
        return (
            observations,
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            np.zeros(self.count, dtype=bool),
            infos,
        )

    def close(self):
        pass


class EvaluationTest(unittest.TestCase):
    def test_run_starts_reports_finish_timeout_and_spawn_checks(self) -> None:
        starts = [
            {"canonical": True, "kind": "canonical"},
            {"cell": [3, 4], "facing": 1, "kind": "state"},
            {"cell": [5, 6], "facing": -1, "kind": "state"},
        ]
        env = FakeEnv(3, {0: 3, 1: 2})
        rows = rollout.run_starts(
            FakeModel(),
            env,
            starts,
            preprocessor=None,
            max_decisions=10,
            action_repeat=4,
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0]["finished"])
        self.assertTrue(rows[1]["finished"])
        self.assertFalse(rows[2]["finished"])
        self.assertTrue(rows[2]["timed_out"])
        self.assertAlmostEqual(rows[0]["run_time"], 3 * 4 / 60)
        self.assertAlmostEqual(rows[1]["run_time"], 2 * 4 / 60)
        self.assertTrue(all(row["spawn_ok"] for row in rows))
        self.assertEqual(rows[1]["start_cell"], [3, 4])
        self.assertEqual(env.reset_spawns[0]["agent"], 1)

    def test_force_start_jump_rejects_state_starts(self) -> None:
        env = FakeEnv(1, {0: 2})
        with self.assertRaises(ValueError):
            rollout.run_starts(
                FakeModel(),
                env,
                [{"cell": [1, 1], "facing": 1}],
                preprocessor=None,
                max_decisions=5,
                action_repeat=4,
                force_start_jump=True,
            )

    def test_prefix_starts_are_rejected(self) -> None:
        env = FakeEnv(1, {})
        with self.assertRaises(ValueError):
            rollout.run_starts(
                FakeModel(),
                env,
                [{"prefix_actions": [1, 0]}],
                preprocessor=None,
                max_decisions=5,
                action_repeat=4,
            )

    def test_summarize_uses_numeric_values(self) -> None:
        rows = [
            {
                "agent": 0,
                "finished": True,
                "timed_out": False,
                "run_time": 2.0,
                "reward": 1.0,
                "deaths": 0,
                "efficiency": 0.5,
                "flow": 0.25,
            },
            {
                "agent": 1,
                "finished": False,
                "timed_out": True,
                "run_time": None,
                "reward": 0.0,
                "deaths": 1,
                "efficiency": None,
                "flow": None,
            },
        ]
        summary = rollout.summarize(rows)
        self.assertEqual(summary["agents"], 2)
        self.assertEqual(summary["finishes"], 1)
        self.assertEqual(summary["timeouts"], 1)
        self.assertEqual(summary["deaths"], 0)
        self.assertEqual(summary["spawn_failures"], 0)
        self.assertEqual(summary["best_time"], 2.0)
        self.assertEqual(summary["median_time"], 2.0)
        self.assertAlmostEqual(summary["efficiency_mean"], 0.5)

    def test_summarize_separates_death_from_timeout(self) -> None:
        rows = [
            {
                "finished": True,
                "timed_out": False,
                "run_time": 4.0,
                "reward": 500.0,
                "deaths": 0,
                "spawn_ok": True,
            },
            {
                "finished": False,
                "timed_out": False,
                "run_time": 2.4,
                "reward": -50.0,
                "deaths": 1,
                "spawn_ok": True,
            },
            {
                "finished": False,
                "timed_out": True,
                "run_time": 10.0,
                "reward": -100.0,
                "deaths": 0,
                "spawn_ok": True,
            },
        ]
        summary = rollout.summarize(rows)
        self.assertEqual(summary["finishes"], 1)
        self.assertEqual(summary["deaths"], 1)
        self.assertEqual(summary["timeouts"], 1)

    def test_evaluate_set_rejects_spawn_mismatch(self) -> None:
        class WrongSpawnEnv(FakeEnv):
            def reset(self, spawns=None):
                observations, infos = super().reset(spawns=spawns)
                for info in infos:
                    if "spawn_cell" in info:
                        info["spawn_cell"] = [99, 99]
                return observations, infos

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = Run(
                config={"env": {}, "algorithm": {"name": "ppo"}, "output": {}},
                config_path=root / "config.toml",
                output_dir=root / "run",
                timesteps=10,
                resume=None,
                preprocessing_path=root / "pre.pkl",
                show=False,
                device="cpu",
                verbose=0,
            )
            starts = [
                {"canonical": True, "kind": "canonical"},
                {"cell": [3, 4], "facing": 1, "kind": "state"},
            ]
            env = WrongSpawnEnv(2, {0: 3, 1: 2})
            with mock.patch.object(
                rollout, "load_start_pool", return_value=starts
            ), mock.patch.object(
                rollout, "make_eval_env", return_value=env
            ):
                with self.assertRaises(RuntimeError):
                    rollout.evaluate_set(
                        FakeModel(),
                        run,
                        "/nonexistent",
                        None,
                        "train",
                        root / "train_start_pool.json",
                        "smoke",
                    )

    def test_evaluate_set_writes_rows_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = Run(
                config={"env": {}, "algorithm": {"name": "ppo"}, "output": {}},
                config_path=root / "config.toml",
                output_dir=root / "run",
                timesteps=10,
                resume=None,
                preprocessing_path=root / "pre.pkl",
                show=False,
                device="cpu",
                verbose=0,
            )
            starts = [
                {"canonical": True, "kind": "canonical"},
                {"cell": [3, 4], "facing": 1, "kind": "state"},
            ]
            env = FakeEnv(2, {0: 3, 1: 2})
            with mock.patch.object(
                rollout, "load_start_pool", return_value=starts
            ), mock.patch.object(
                rollout, "make_eval_env", return_value=env
            ):
                summary = rollout.evaluate_set(
                    FakeModel(),
                    run,
                    "/nonexistent",
                    None,
                    "train",
                    root / "train_start_pool.json",
                    "smoke",
                )
            self.assertEqual(summary["finishes"], 2)
            self.assertEqual(summary["spawn_ok"], 2)
            csv_path = run.eval_dir / "smoke" / "train.csv"
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("start_kind", header)
            self.assertIn("run_time", header)


if __name__ == "__main__":
    unittest.main()
