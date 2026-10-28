#!/usr/bin/env python3
"""CLI for post-processing clean NEAT routes.

The implementation lives in :mod:`plaiground.data.route_filter`; this script
only handles arguments and presentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plaiground.data.route_filter import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_JACCARD,
    DEFAULT_MAX_PER_CLUSTER,
    DEFAULT_SEED,
    filter_routes,
    load_routes,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--level", default="1-1")
    parser.add_argument("--jaccard", type=float, default=DEFAULT_JACCARD)
    parser.add_argument(
        "--max-per-cluster",
        type=int,
        default=DEFAULT_MAX_PER_CLUSTER,
        help="Keep at most this many routes per cluster (0 = keep all).",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help="Fraction of whole route clusters to reserve for evaluation.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = filter_routes(
        load_routes(args.routes),
        jaccard=args.jaccard,
        max_per_cluster=args.max_per_cluster,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        level=args.level,
    )
    write_outputs(result, args.output_dir)
    summary = result["summary"]
    print(
        f"[filter] input={summary['input_routes']} "
        f"unique={summary['unique_signatures']} "
        f"clusters={summary['clusters']} "
        f"filtered={summary['filtered_routes']} "
        f"train={summary['train_routes']} "
        f"holdout={summary['holdout_routes']} "
        f"checkpoints={summary['filtered_routes_with_checkpoint']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
