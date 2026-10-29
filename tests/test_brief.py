"""Tests for the compact repo/run brief."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_tool(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


brief = load_tool("plaiground_brief", "brief.py")


class BriefTest(unittest.TestCase):
    def build_tree(self, root: Path) -> None:
        ledger = root / "docs" / "reference" / "level-ledger.md"
        ledger.parent.mkdir(parents=True)
        ledger.write_text(
            "# Level ledger\n\n"
            "| Level | Par / +0.30 | Route source | Selected model | "
            "Canonical | Accepted coverage | Status |\n"
            "|---|---:|---|---|---:|---|---|\n"
            "| 1-5 | 2.9088 / 3.2088 | `0080`/`0081` 1-5 | "
            "`0085` final 1.655M | **3.1667** | canonical finish | "
            "selected 1-5 expert |\n",
            encoding="utf-8",
        )

        experiment = root / "experiments" / "0001_demo"
        experiment.mkdir(parents=True)
        (experiment / "config.toml").write_text(
            "[experiment]\n"
            'id = "0001"\n'
            'name = "demo"\n'
            'status = "complete"\n'
            "[algorithm]\n"
            'name = "dqn"\n'
            "timesteps = 1000\n"
            "[output]\n"
            'dir = "runs/demo"\n'
            "[result]\n"
            'selected_checkpoint = "dqn/checkpoints/ddqn_1000_steps.zip"\n',
            encoding="utf-8",
        )

        run = root / "runs" / "demo"
        (run / "dqn" / "checkpoints").mkdir(parents=True)
        (run / "dqn" / "checkpoints" / "ddqn_1000_steps.zip").write_bytes(b"")
        (run / "eval" / "ddqn_1000_steps").mkdir(parents=True)
        (run / "eval" / "ddqn_1000_steps" / "summary.json").write_text(
            json.dumps(
                {
                    "tag": "ddqn_1000_steps",
                    "canonical": {
                        "1-5": {
                            "agents": 1,
                            "finishes": 1,
                            "deaths": 0,
                            "timeouts": 0,
                            "best_time": 3.1667,
                        }
                    },
                    "sets": {
                        "train_start_pool": {
                            "agents": 12,
                            "finishes": 12,
                            "best_time": 3.1667,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "train.log").write_text(
            "start\nObjectDB instances leaked\nat: cleanup\nlast useful line\n",
            encoding="utf-8",
        )

    def test_parse_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_tree(root)
            rows = brief.parse_ledger(
                root / "docs" / "reference" / "level-ledger.md"
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["level"], "1-5")
            self.assertEqual(rows[0]["selected"], "0085 final 1.655M")
            self.assertEqual(rows[0]["canonical"], "3.1667")
            self.assertEqual(rows[0]["coverage"], "canonical finish")
            self.assertEqual(rows[0]["status"], "selected 1-5 expert")

    def test_repo_brief_uses_ledger_and_recent_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_tree(root)
            output = brief.render_repo_brief(root, jobs=[])
            self.assertIn("Selected levels", output)
            self.assertIn("1-5", output)
            self.assertIn("3.1667", output)
            self.assertIn("ddqn_1000_steps.zip", output)
            self.assertIn("Running  none", output)

    def test_run_brief_reports_manifest_eval_and_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_tree(root)
            output = brief.render_run_brief(root, root / "runs" / "demo")
            self.assertIn("Run demo", output)
            self.assertIn("experiments/0001_demo/config.toml", output)
            self.assertIn("checkpoint 1 total", output)
            self.assertIn("1/1 3.1667s", output)
            self.assertIn("last useful line", output)
            self.assertNotIn("ObjectDB instances leaked", output)
            self.assertNotIn("at: cleanup", output)

    def test_resolve_run_accepts_unique_substring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_tree(root)
            self.assertEqual(
                brief.resolve_run(root, "dem"), root / "runs" / "demo"
            )
            self.assertIsNone(brief.resolve_run(root, "missing"))


if __name__ == "__main__":
    unittest.main()
