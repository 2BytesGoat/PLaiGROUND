"""Tests for creating and loading the lean preprocessing pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from plaiground.experiments.runner import load_run
from plaiground.observations import lean_io


class LeanIOTest(unittest.TestCase):
    def _run(self, root: Path):
        config_path = root / "config.toml"
        config_path.write_text(
            "\n".join(
                [
                    "[experiment]",
                    'id = "9002"',
                    "[workflow]",
                    'name = "reinforcement.ppo"',
                    "[env]",
                    'level = "1-1"',
                    "n_envs = 1",
                    "[observation]",
                    'encoding = "lean"',
                    'preprocessing = "lean"',
                    "[model]",
                    "hidden = [64, 64]",
                    'activation = "tanh"',
                    "[algorithm]",
                    'name = "ppo"',
                    "timesteps = 100",
                    "[output]",
                    f'dir = "{root / "run"}"',
                ]
            ),
            encoding="utf-8",
        )
        return load_run(config_path)

    def test_creates_pipeline_without_a_game_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._run(root)
            with mock.patch.object(
                lean_io,
                "probe_lean_tile_names",
                return_value=["E", "W", "Y"],
            ):
                preprocessor = lean_io.prepare_lean_preprocessor(
                    run, "/nonexistent"
                )
            self.assertIsNotNone(preprocessor)
            self.assertTrue(run.preprocessing_path.exists())
            metadata = json.loads(
                (run.preprocessing_path.parent / "preprocessing.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["obs_size"], 188)
            self.assertEqual(metadata["model_input_size"], preprocessor.output_size)
            self.assertEqual(metadata["tile_names"], ["E", "W", "Y"])

    def test_resume_loads_the_saved_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._run(root)
            with mock.patch.object(
                lean_io,
                "probe_lean_tile_names",
                return_value=["E", "W", "Y"],
            ):
                first = lean_io.prepare_lean_preprocessor(run, "/nonexistent")

            resumed = load_run(
                run.config_path,
                resume=run.preprocessing_path.parent / "checkpoint.zip",
            )
            loaded = lean_io.prepare_lean_preprocessor(resumed, "/nonexistent")
            self.assertEqual(loaded.output_size, first.output_size)


if __name__ == "__main__":
    unittest.main()
