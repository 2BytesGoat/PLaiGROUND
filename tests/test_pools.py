"""Tests for shared start/spawn pool loaders."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plaiground.data.pools import flatten_reset_state, load_start_pool


class PoolLoaderTest(unittest.TestCase):
    def test_flattens_legacy_nested_state(self) -> None:
        item = flatten_reset_state(
            {
                "cell": [4, 7],
                "facing": 1,
                "state": {"run_time": 1.25, "position": [10.0, 20.0]},
            }
        )
        self.assertEqual(item["run_time"], 1.25)
        self.assertEqual(item["position"], [10.0, 20.0])

    def test_loads_weighted_start_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.json"
            path.write_text(
                json.dumps(
                    {
                        "starts": [
                            {"canonical": True, "weight": 0.7},
                            {
                                "cell": [4, 7],
                                "facing": -1,
                                "run_time": 1.0,
                                "weight": 0.3,
                            },
                        ]
                    }
                )
            )
            pool = load_start_pool(path)
            self.assertEqual(len(pool), 2)
            self.assertTrue(pool[0]["canonical"])
            self.assertEqual(pool[1]["cell"], [4, 7])
            self.assertEqual(pool[1]["facing"], -1)
            self.assertAlmostEqual(pool[1]["weight"], 0.3)

    def test_attaches_payload_level_to_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.json"
            path.write_text(
                json.dumps(
                    {
                        "level": "1-1",
                        "starts": [
                            {"canonical": True, "weight": 1.0},
                        ],
                    }
                )
            )
            pool = load_start_pool(path)
            self.assertEqual(pool[0]["level"], "1-1")

    def test_rejects_empty_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.json"
            path.write_text(json.dumps({"starts": []}))
            with self.assertRaises(ValueError):
                load_start_pool(path)


if __name__ == "__main__":
    unittest.main()
