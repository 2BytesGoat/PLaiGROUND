"""Tests for experiment manifests and code/game provenance."""

from __future__ import annotations

import unittest
from pathlib import Path

from plaiground.experiments import load_experiment
from plaiground.experiments.provenance import git_metadata

ROOT = Path(__file__).resolve().parent.parent


class ExperimentConfigTest(unittest.TestCase):
    def test_current_ppo_experiment_loads(self) -> None:
        config = load_experiment(ROOT / "experiments/0001_ppo_clean_start_1_1")
        self.assertEqual(config["experiment"]["id"], "0001")
        self.assertEqual(config["workflow"]["name"], "reinforcement.ppo")
        self.assertEqual(config["observation"]["encoding"], "lean")
        self.assertEqual(config["model"]["hidden"], [512, 512])
        self.assertEqual(config["game"]["commit"], "adb9278")

    def test_git_metadata_has_commit_and_branch(self) -> None:
        metadata = git_metadata(ROOT)
        self.assertTrue(metadata["commit"])
        self.assertIsInstance(metadata["dirty"], bool)
        self.assertIn("diff_sha256", metadata)


if __name__ == "__main__":
    unittest.main()
