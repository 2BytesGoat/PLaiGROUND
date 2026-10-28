"""Tests for post-processing clean NEAT route filters."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plaiground.data import route_filter


def make_route(
    signature: str,
    cells: list[list[int]],
    run_time: float,
    *,
    checkpoint: bool = True,
) -> dict:
    return {
        "route_signature": signature,
        "run_time": run_time,
        "path_cells": cells,
        "clean": True,
        "checkpoint_file": (
            f"checkpoints/{signature}.json" if checkpoint else None
        ),
    }


class CleanRouteFilterTest(unittest.TestCase):
    def test_mapping_keys_become_route_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes.json"
            path.write_text(
                json.dumps(
                    {
                        "mapped-signature": {
                            "run_time": 2.0,
                            "path_cells": [[0, 0], [1, 0]],
                            "clean": True,
                            "checkpoint_file": "checkpoints/a.json",
                        }
                    }
                )
            )
            routes = route_filter.load_routes(path)
        self.assertEqual(routes[0]["route_signature"], "mapped-signature")

    def test_exact_duplicates_keep_fastest_signature(self) -> None:
        slow = make_route("same", [[0, 0], [1, 0], [2, 0]], 2.5)
        fast = make_route("same", [[0, 0], [1, 0], [2, 0]], 2.0)
        result = route_filter.filter_routes([slow, fast], max_per_cluster=0)
        self.assertEqual(len(result["filtered_routes"]), 1)
        self.assertEqual(result["filtered_routes"][0]["run_time"], 2.0)
        self.assertEqual(result["summary"]["duplicate_signatures"], 1)

    def test_near_duplicates_are_clustered_and_capped(self) -> None:
        routes = [
            make_route("h1", [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], 2.0),
            make_route("h2", [[0, 0], [1, 0], [2, 0], [3, 0], [4, 1]], 2.1),
            make_route("h3", [[0, 0], [1, 0], [2, 0], [3, 0], [4, 2]], 2.2),
            make_route("h4", [[0, 0], [1, 0], [2, 0], [3, 0], [4, 3]], 2.3),
            make_route("h5", [[0, 0], [1, 0], [2, 0], [3, 0], [4, 4]], 2.4),
            make_route("h6", [[0, 0], [1, 0], [2, 0], [3, 0], [4, 5]], 2.5),
            make_route("v1", [[0, 0], [0, 1], [0, 2]], 2.2),
        ]
        result = route_filter.filter_routes(
            routes,
            jaccard=0.5,
            max_per_cluster=3,
        )
        summary = result["summary"]
        self.assertEqual(summary["clusters"], 2)
        self.assertEqual(summary["filtered_routes"], 4)
        horizontal = next(
            cluster
            for cluster in result["clusters"]
            if cluster["representative_signature"] == "h1"
        )
        self.assertEqual(horizontal["route_count"], 6)
        self.assertEqual(horizontal["kept_count"], 3)
        kept = [
            route
            for route in result["filtered_routes"]
            if route["cluster_id"] == horizontal["cluster_id"]
        ]
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept[0]["cluster_rank"], 0)
        self.assertTrue(kept[0]["cluster_representative"])

    def test_cluster_holdout_split_is_disjoint(self) -> None:
        routes = [
            make_route("a", [[0, 0], [1, 0]], 2.0),
            make_route("b", [[10, 10], [11, 10]], 2.1),
            make_route("c", [[20, 20], [21, 20]], 2.2),
            make_route("d", [[30, 30], [31, 30]], 2.3),
        ]
        result = route_filter.filter_routes(
            routes,
            jaccard=0.5,
            max_per_cluster=1,
            holdout_fraction=0.5,
            seed=7,
        )
        summary = result["summary"]
        self.assertEqual(summary["clusters"], 4)
        self.assertEqual(summary["holdout_clusters"], 2)
        self.assertEqual(summary["train_clusters"], 2)
        train_clusters = {
            route["cluster_id"]
            for route in result["filtered_routes"]
            if route["split"] == "train"
        }
        holdout_clusters = {
            route["cluster_id"]
            for route in result["filtered_routes"]
            if route["split"] == "holdout"
        }
        self.assertFalse(train_clusters & holdout_clusters)
        self.assertEqual(len(train_clusters), 2)
        self.assertEqual(len(holdout_clusters), 2)

    def test_no_split_keeps_all_filtered_as_train(self) -> None:
        routes = [
            make_route("a", [[0, 0], [1, 0]], 2.0),
            make_route("b", [[10, 10], [11, 10]], 2.1),
        ]
        result = route_filter.filter_routes(
            routes,
            max_per_cluster=1,
            holdout_fraction=0.0,
        )
        self.assertEqual(result["summary"]["holdout_clusters"], 0)
        self.assertEqual(result["summary"]["train_clusters"], 2)
        self.assertTrue(
            all(route["split"] == "train" for route in result["filtered_routes"])
        )

    def test_outputs_are_written_and_deterministic(self) -> None:
        routes = [
            make_route("a", [[0, 0], [1, 0], [2, 0]], 2.0),
            make_route("b", [[0, 0], [1, 0], [2, 0], [3, 0]], 2.1),
            make_route("c", [[10, 10], [11, 10]], 2.2),
        ]
        first = route_filter.filter_routes(
            routes,
            max_per_cluster=1,
            holdout_fraction=0.5,
            seed=11,
        )
        second = route_filter.filter_routes(
            routes,
            max_per_cluster=1,
            holdout_fraction=0.5,
            seed=11,
        )
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            route_filter.write_outputs(first, output)
            self.assertTrue((output / "filtered_clean_routes.json").exists())
            self.assertTrue((output / "route_clusters.json").exists())
            self.assertTrue((output / "summary.json").exists())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["filtered_routes"], 2)


if __name__ == "__main__":
    unittest.main()
