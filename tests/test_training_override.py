"""The training override must ship with every launch (A41)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from plaiground.envs.runtime import (
    DISABLE_TRAINING_OVERRIDE,
    TRAINING_OVERRIDE,
    ensure_training_override,
    remove_training_override,
)


class TrainingOverrideTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _executable(self, name: str = "Game") -> Path:
        macos = self.root / f"{name}.app" / "Contents" / "MacOS"
        macos.mkdir(parents=True)
        executable = macos / name
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        return executable

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def test_writes_beside_exported_executable(self) -> None:
        executable = self._executable()
        written = ensure_training_override(executable)
        target = executable.parent / "override.cfg"
        self.assertEqual(written, [target])
        self.assertEqual(self._read(target), TRAINING_OVERRIDE)

    def test_accepts_the_app_bundle_path(self) -> None:
        executable = self._executable()
        written = ensure_training_override(self.root / "Game.app")
        self.assertEqual(written, [executable.parent / "override.cfg"])
        self.assertTrue((executable.parent / "override.cfg").is_file())

    def test_writes_into_a_marked_source_project(self) -> None:
        executable = self._executable()
        project = self.root / "project"
        project.mkdir()
        (executable.parent / "project_path").write_text(
            f"{project}\n", encoding="utf-8"
        )
        written = ensure_training_override(executable)
        self.assertIn(project / "override.cfg", written)
        self.assertEqual(self._read(project / "override.cfg"), TRAINING_OVERRIDE)
        self.assertTrue((executable.parent / "override.cfg").is_file())

    def test_is_idempotent_and_repairs_stale_content(self) -> None:
        executable = self._executable()
        target = ensure_training_override(executable)[0]
        target.write_text("stale\n", encoding="utf-8")
        ensure_training_override(executable)
        self.assertEqual(self._read(target), TRAINING_OVERRIDE)

    def test_disable_flag_writes_nothing(self) -> None:
        executable = self._executable()
        with mock.patch.dict(os.environ, {DISABLE_TRAINING_OVERRIDE: "1"}):
            self.assertEqual(ensure_training_override(executable), [])
        self.assertFalse((executable.parent / "override.cfg").exists())

    def test_remove_leaves_a_custom_file_alone(self) -> None:
        executable = self._executable()
        target = ensure_training_override(executable)[0]
        target.write_text("custom\n", encoding="utf-8")
        self.assertEqual(remove_training_override(executable), [])
        self.assertTrue(target.is_file())

    def test_remove_deletes_the_harness_file(self) -> None:
        executable = self._executable()
        target = ensure_training_override(executable)[0]
        self.assertEqual(remove_training_override(executable), [target])
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
