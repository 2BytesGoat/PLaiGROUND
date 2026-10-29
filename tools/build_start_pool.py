#!/usr/bin/env python3
"""Build train/holdout clean-start pools and their eval union."""

from __future__ import annotations

import argparse
from pathlib import Path

from plaiground.data.start_pool import build_start_pools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        required=True,
        help="Directory produced by tools/filter_routes.py.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        required=True,
        help="NEAT archive containing routes.json and checkpoints/.",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        required=True,
        help="Game level analysis JSON with a time_to_exit grid.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--level", default="1-1")
    parser.add_argument("--min-time-to-exit", type=float, default=0.5)
    parser.add_argument("--canonical-weight", type=float, default=0.70)
    parser.add_argument("--state-weight", type=float, default=0.30)
    parser.add_argument(
        "--eval-pool-name",
        default="eval_start_pool.json",
        help="Filename for the deduplicated train+holdout evaluation union.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_start_pools(
        filtered_dir=args.filtered_dir,
        archive_dir=args.archive_dir,
        analysis_path=args.analysis,
        output_dir=args.output_dir,
        level=args.level,
        min_time_to_exit=args.min_time_to_exit,
        canonical_weight=args.canonical_weight,
        state_weight=args.state_weight,
        eval_pool_name=args.eval_pool_name,
    )
    summary = result["summary"]
    print(
        f"[pool] train_routes={summary['train_routes']} "
        f"holdout_routes={summary['holdout_routes']} "
        f"train_states={summary['train_states']} "
        f"holdout_states={summary['holdout_states']} "
        f"shared_state_keys={summary['shared_state_keys']} "
        f"eval_states={summary['unique_eval_states']} "
        f"state_weight_each={summary['state_weight_each']:.6f}",
        flush=True,
    )
    print(f"[pool] eval={result['eval_pool_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
