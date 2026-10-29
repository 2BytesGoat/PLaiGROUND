from __future__ import annotations

import unittest

from plaiground.algorithms.genetic.neat.portfolio import (
    PortfolioEntry,
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


if __name__ == "__main__":
    unittest.main()
