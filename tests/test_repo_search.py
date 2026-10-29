"""Tests for the local repository index and context search."""

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


repo_index = load_tool("plaiground_repo_index", "repo_index.py")
context_tool = load_tool("plaiground_context", "context.py")


class RepoSearchTest(unittest.TestCase):
    def build_tree(self, root: Path) -> None:
        source = root / "src" / "plaiground"
        source.mkdir(parents=True)
        (source / "demo.py").write_text(
            '"""Demo module for pool search."""\n\n'
            "class Widget:\n"
            '    """A searchable widget."""\n\n'
            "    def run(self) -> None:\n"
            "        pass\n\n"
            "def build_pool() -> None:\n"
            '    """Build a start pool."""\n'
            "    pass\n",
            encoding="utf-8",
        )

        experiment = root / "experiments" / "0001_demo"
        experiment.mkdir(parents=True)
        (experiment / "config.toml").write_text(
            "[experiment]\n"
            'id = "0001"\n'
            'name = "demo"\n'
            'status = "complete"\n'
            "[workflow]\n"
            'name = "reinforcement.ddqn"\n'
            "[env]\n"
            'level = "1-5"\n'
            "[result]\n"
            'selected_checkpoint = "dqn/demo.zip"\n'
            'note = "1-5 expert replay pool note"\n'
            "[output]\n"
            'dir = "runs/demo"\n',
            encoding="utf-8",
        )
        (experiment / "README.md").write_text(
            "# Demo experiment\n\nStatus: complete\n\n"
            "This covers the 1-5 expert replay pool.\n",
            encoding="utf-8",
        )

        docs = root / "docs"
        docs.mkdir()
        (docs / "status.md").write_text(
            "# Status\n\n## Current focus\n\n1-5 expert replay pool is next.\n",
            encoding="utf-8",
        )

        run = root / "runs" / "demo"
        (run / "archive").mkdir(parents=True)
        (run / "eval" / "tag").mkdir(parents=True)
        (run / "dqn" / "checkpoints").mkdir(parents=True)
        (run / "history.csv").write_text(
            "generation,finishes,best_time\n20,384,4.6833\n", encoding="utf-8"
        )
        (run / "archive" / "routes.json").write_text(
            json.dumps(
                {
                    "clean": {"run_time": 4.68, "clean": True, "generation": 20},
                    "rough": {"run_time": 8.0, "clean": False, "generation": 19},
                }
            ),
            encoding="utf-8",
        )
        (run / "eval" / "tag" / "summary.json").write_text(
            json.dumps(
                {
                    "tag": "tag",
                    "canonical": {
                        "1-5": {"agents": 1, "finishes": 1, "best_time": 4.68}
                    },
                    "sets": {
                        "train_start_pool": {
                            "agents": 12,
                            "finishes": 12,
                            "best_time": 4.68,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "dqn" / "checkpoints" / "ddqn_1000_steps.zip").write_bytes(b"")
        (run / "neat-checkpoint-2").write_bytes(b"")
        (run / "neat-checkpoint-20").write_bytes(b"")

    def test_index_and_code_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_tree(root)
            index = repo_index.build_index(root)
            self.assertIn("src/plaiground/demo.py", index["modules"])
            self.assertEqual(index["experiments"][0]["levels"], ["1-5"])
            markdown = repo_index.render_markdown(index)
            self.assertIn("src/plaiground/demo.py", markdown)
            self.assertIn("0001", markdown)
            self.assertNotIn(str(root), markdown)
            self.assertEqual(index["runs"][0]["archive"]["clean"], 1)
            self.assertIn("neat-checkpoint-20", index["runs"][0]["key_files"])

    def test_context_ranks_level_and_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_tree(root)
            index = repo_index.build_index(root)
            tokens, phrases = context_tool.tokenize(
                "1-5 expert replay pool"
            )
            self.assertIn("1-5", phrases)
            code = context_tool.search_code(index, tokens, phrases, 5)
            self.assertTrue(
                any(item["path"] == "src/plaiground/demo.py" for item in code)
            )
            experiments = context_tool.search_experiments(
                index, tokens, phrases, 5, levels=["1-5"]
            )
            self.assertEqual(experiments[0]["id"], "0001")
            runs = context_tool.search_runs(
                index, tokens, phrases, 5, levels=["1-5"]
            )
            self.assertEqual(runs[0]["path"], "runs/demo")


if __name__ == "__main__":
    unittest.main()
