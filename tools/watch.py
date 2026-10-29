#!/usr/bin/env python3
"""Preview a saved PPO or DQN checkpoint in the game window.

The experiment manifest determines the algorithm, observation encoding, and
environment settings. Use ``--start-pool``/``--start-index`` to preview a
specific train or holdout start; otherwise the canonical start is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plaiground.data.pools import load_start_pool
from plaiground.envs.factory import make_eval_env
from plaiground.envs.runtime import resolve_env_path
from plaiground.evaluation.rollout import (
    load_model,
    load_preprocessor,
    run_starts,
    summarize,
)
from plaiground.experiments.runner import load_run, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment directory or config.toml path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Checkpoint to preview; defaults to [output].model_file.",
    )
    parser.add_argument(
        "--start-pool",
        type=Path,
        default=None,
        help="Preview one entry from a start pool instead of canonical.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Physics speedup; 1.0 is real time.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the game window.",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = load_run(
        args.config,
        show=not args.headless,
        device=args.device,
        verbose=args.verbose,
    )
    run.ensure_dirs()
    model_path = (
        resolve_path(args.model) if args.model is not None else run.model_path
    )
    if not model_path.exists():
        raise SystemExit(f"missing model: {model_path}")
    model = load_model(run.algorithm_name, model_path, run.device)
    preprocessor = load_preprocessor(run)

    if args.start_pool is not None:
        starts = load_start_pool(args.start_pool)
        if not 0 <= args.start_index < len(starts):
            raise SystemExit(
                f"start index {args.start_index} out of range for "
                f"{len(starts)} starts"
            )
        starts = [starts[args.start_index]]
    else:
        starts = [{"canonical": True}]

    level = run.level
    tag = args.tag or model_path.stem
    env = make_eval_env(
        run,
        resolve_env_path(),
        level=level,
        num_agents=1,
        port=run.eval_port,
        log_name=f"watch-{tag}.log",
        home_name=f"watch_{tag}",
        show_window=not args.headless,
        speedup=args.speedup,
    )
    try:
        rows = run_starts(
            model,
            env,
            starts,
            preprocessor=preprocessor,
            max_decisions=run.max_decisions,
            action_repeat=run.action_repeat,
            force_start_jump=(
                bool(starts[0].get("canonical"))
                and bool(run.env.get("force_start_jump", False))
            ),
        )
    finally:
        env.close()

    summary = summarize(rows)
    row = rows[0]
    print(
        f"[watch] algorithm={run.algorithm_name} level={level} "
        f"start={row.get('start_kind')} cell={row.get('start_cell') or '-'} "
        f"finished={row['finished']} time={row['run_time'] or '-'} "
        f"steps={row['steps']} reward={row['reward']:.2f}"
    )
    print(
        f"[watch] finish={summary['finishes']}/{summary['agents']} "
        f"timeouts={summary['timeouts']} spawn_ok={row['spawn_ok']}",
        flush=True,
    )
    return 0 if row["finished"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
