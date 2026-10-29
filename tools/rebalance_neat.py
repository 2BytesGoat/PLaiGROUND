#!/usr/bin/env python3
"""Rebalance a NEAT checkpoint around a portfolio of route/time representatives."""

from __future__ import annotations

import argparse
import json

from plaiground.algorithms.genetic.neat.portfolio import (
    DEFAULT_QUOTAS,
    rebalance_population,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--jaccard", type=float, default=0.5)
    parser.add_argument("--max-per-cluster", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed-quota", type=float, default=DEFAULT_QUOTAS["speed"])
    parser.add_argument(
        "--median-quota", type=float, default=DEFAULT_QUOTAS["median"]
    )
    parser.add_argument("--slow-quota", type=float, default=DEFAULT_QUOTAS["slow"])
    parser.add_argument(
        "--exploration-quota",
        type=float,
        default=DEFAULT_QUOTAS["exploration"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quotas = {
        "speed": args.speed_quota,
        "median": args.median_quota,
        "slow": args.slow_quota,
        "exploration": args.exploration_quota,
    }
    result = rebalance_population(
        checkpoint=args.checkpoint,
        routes_path=args.routes,
        output_prefix=args.output_prefix,
        quotas=quotas,
        jaccard=args.jaccard,
        max_per_cluster=args.max_per_cluster,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
