from __future__ import annotations

import unittest
from types import SimpleNamespace

from plaiground.algorithms.genetic.neat.portfolio import (
    PortfolioEntry,
    _merge_extra_genomes,
    build_entries,
    select_portfolio,
    time_bin,
)


class PortfolioTest(unittest.TestCase):
    def test_time_bins(self) -> None:
        self.assertEqual(time_bin(4.0), "speed")
        self.assertEqual(time_bin(5.0), "median")
        self.assertEqual(time_bin(7.0), "slow")
        self.assertEqual(time_bin(12.0), "exploration")
        self.assertEqual(time_bin(None), "exploration")

    def test_selection_fills_requested_size(self) -> None:
        entries = [
            PortfolioEntry(
                genome_id=index,
                run_time=4.0 + index * 0.01,
                clean=True,
                cluster_id=f"c{index % 3:03d}",
                bin="speed" if index < 3 else "median",
                route_signature=f"r{index}",
            )
            for index in range(10)
        ]
        selected = select_portfolio(
            entries,
            total=8,
            quotas={"speed": 0.25, "median": 0.5, "slow": 0.25},
            max_per_cluster=2,
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({entry.genome_id for entry in selected}), 8)

    def test_cluster_cap_is_respected_then_filled(self) -> None:
        entries = [
            PortfolioEntry(
                genome_id=index,
                run_time=5.0,
                clean=True,
                cluster_id="c000",
                bin="median",
                route_signature=f"r{index}",
            )
            for index in range(10)
        ]
        selected = select_portfolio(
            entries,
            total=8,
            quotas={"median": 1.0},
            max_per_cluster=2,
        )
        self.assertEqual(len(selected), 8)

    def test_extra_finishers_survive_post_reproduction_key_collisions(self) -> None:
        candidates = {1: SimpleNamespace(key=1), 5: SimpleNamespace(key=5)}
        extras = {
            1: SimpleNamespace(key=1, tag="parent-a"),
            2: SimpleNamespace(key=2, tag="parent-b"),
        }
        remap, added = _merge_extra_genomes(candidates, extras)
        self.assertEqual(added, 2)
        self.assertNotEqual(remap[1], 1)
        self.assertEqual(remap[2], 2)
        self.assertEqual(candidates[remap[1]].tag, "parent-a")
        self.assertEqual(candidates[remap[2]].tag, "parent-b")
        self.assertEqual(candidates[1].key, 1)

    def test_live_finishers_without_route_archive_are_classified(self) -> None:
        population = {1: object(), 2: object(), 3: object()}
        entries, stats = build_entries(
            population,
            [],
            live_finish_times={1: 5.5, 2: 7.25},
        )
        by_id = {entry.genome_id: entry for entry in entries}
        self.assertEqual(stats["finishers"], 2)
        self.assertEqual(stats["live_finishers"], 2)
        self.assertEqual(by_id[1].bin, "median")
        self.assertEqual(by_id[2].bin, "slow")
        self.assertEqual(by_id[3].bin, "exploration")
        self.assertTrue(by_id[1].is_finisher)
        self.assertTrue(by_id[2].is_finisher)
        self.assertFalse(by_id[3].is_finisher)

    def test_slow_finisher_is_kept_when_preserving_finishers(self) -> None:
        entries = [
            PortfolioEntry(
                genome_id=1,
                run_time=12.0,
                clean=False,
                cluster_id=None,
                bin="exploration",
                route_signature=None,
                is_finisher=True,
            ),
            PortfolioEntry(
                genome_id=2,
                run_time=None,
                clean=False,
                cluster_id=None,
                bin="exploration",
                route_signature=None,
                is_finisher=False,
            ),
        ]
        selected = select_portfolio(
            entries,
            total=1,
            preserve_finishers=True,
        )
        self.assertEqual([entry.genome_id for entry in selected], [1])

    def test_selection_keeps_all_live_finishers_when_population_has_room(self) -> None:
        entries = [
            PortfolioEntry(
                genome_id=index,
                run_time=5.0 + index * 0.01,
                clean=True,
                cluster_id=f"c{index % 8:03d}",
                bin="median",
                route_signature=f"r{index}",
            )
            for index in range(173)
        ] + [
            PortfolioEntry(
                genome_id=1000 + index,
                run_time=None,
                clean=False,
                cluster_id=None,
                bin="exploration",
                route_signature=None,
            )
            for index in range(211)
        ]
        selected = select_portfolio(entries, total=384)
        selected_finishers = [
            entry for entry in selected if entry.bin != "exploration"
        ]
        self.assertEqual(len(selected), 384)
        self.assertEqual(len(selected_finishers), 173)


if __name__ == "__main__":
    unittest.main()
