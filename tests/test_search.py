"""Tests for the bounded search tool."""

from __future__ import annotations

import importlib.util
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


search = load_tool("plaiground_search", "search.py")


class BoundedSearchTest(unittest.TestCase):
    def build_tree(self, root: Path) -> None:
        (root / "src").mkdir(parents=True)
        (root / "src" / "good.py").write_text(
            "needle one\nno match\nneedle two\n", encoding="utf-8"
        )
        (root / "src" / "other.md").write_text("needle docs\n", encoding="utf-8")
        (root / "runs" / "demo").mkdir(parents=True)
        (root / "runs" / "demo" / "big.txt").write_text(
            "needle hidden\n", encoding="utf-8"
        )
        (root / "data").mkdir()
        (root / "data" / "blob.txt").write_text("needle data\n", encoding="utf-8")
        (root / ".opencode").mkdir()
        (root / ".opencode" / "notes.md").write_text(
            "needle visible\n", encoding="utf-8"
        )
        (root / "binary.bin").write_bytes(b"needle\x00\x01")
        (root / "huge.py").write_text("needle\n" + "x" * 5000, encoding="utf-8")

    def test_default_prunes_generated_and_binary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_tree(root)
            result = search.search_lines(
                "needle", root, max_bytes=1000, max_matches=20
            )
            paths = {item["path"] for item in result["matches"]}
            self.assertEqual(
                paths, {"src/good.py", "src/other.md", ".opencode/notes.md"}
            )
            self.assertEqual(len(result["matches"]), 4)
            self.assertFalse(result["truncated"])

    def test_glob_filters_to_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_tree(root)
            result = search.search_lines(
                "needle", root, globs=["*.py"], max_bytes=1000, max_matches=20
            )
            self.assertEqual(
                [item["path"] for item in result["matches"]],
                ["src/good.py", "src/good.py"],
            )

    def test_match_cap_reports_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_tree(root)
            result = search.search_lines(
                "needle", root, max_bytes=1000, max_matches=1
            )
            self.assertEqual(len(result["matches"]), 1)
            self.assertTrue(result["truncated"])
            self.assertTrue(result["limit_hit"])

    def test_files_only_returns_unique_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_tree(root)
            result = search.search_files("needle", root, max_bytes=1000)
            self.assertEqual(
                result["paths"],
                [".opencode/notes.md", "src/good.py", "src/other.md"],
            )

    def test_invalid_regex_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                search.search_lines("[", root)


if __name__ == "__main__":
    unittest.main()
