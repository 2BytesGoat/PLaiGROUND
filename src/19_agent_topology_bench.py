"""Benchmark agents per Godot process and process scaling.

This is a pure environment-throughput screen, not a PPO learning run. It uses
random actions so resets and per-agent episode boundaries are exercised.

Examples:
    .venv/bin/python src/19_agent_topology_bench.py \
        --agents-per-process 1 2 4 8 16 32 64 100

    .venv/bin/python src/19_agent_topology_bench.py \
        --agents-per-process 8 --processes 1 2 4 8
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import numpy as np

from wrappers.multi_agent_vec_env import MultiAgentGodotVecEnv


DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / "environments" / "DragonJump"

PROFILE_KEYS = (
    "wait",
    "apply_action",
    "physics",
    "info",
    "reward",
    "done",
    "obs",
    "reset",
    "reset_info",
    "pack",
    "send",
)


def _profile_usec(env: MultiAgentGodotVecEnv) -> dict[str, int]:
    totals: dict[str, int] = {}
    for profile in env.profiles:
        for key, value in profile.get("usec", {}).items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _profile_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in after
    }


def resolve_env_path() -> str:
    env_dir = os.getenv("ENV_PATH", str(DEFAULT_ENV_PATH.parent))
    env_name = os.getenv("ENV_NAME", DEFAULT_ENV_PATH.name)
    path = Path(env_dir) / env_name
    if not path.exists() and not path.with_suffix(".app").exists():
        raise FileNotFoundError(f"Godot environment not found: {path}")
    return str(path)


def run_topology(
    args: argparse.Namespace,
    agents_per_process: int,
    processes: int,
    run_index: int,
    warmup_actions: np.ndarray,
    repeat_actions: np.ndarray,
) -> list[dict]:
    total_agents = agents_per_process * processes
    port = args.base_port + run_index * 20
    env = MultiAgentGodotVecEnv(
        env_path=args.env_path,
        level=args.level,
        n_agents_per_process=agents_per_process,
        n_processes=processes,
        seed=args.seed,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        max_decisions=args.max_decisions,
        base_port=port,
        output_dir=Path(args.output_dir) / f"n{agents_per_process}_p{processes}",
        obs_encoding=args.obs_encoding,
        reward_profile=args.reward_profile,
        profile_training=args.profile,
        training_no_cosmetics=args.no_cosmetics,
    )
    rows: list[dict] = []
    agent_slice = slice(0, total_agents)
    try:
        env.reset()
        for step in range(args.warmup):
            env.step_async(warmup_actions[step, agent_slice])
            env.step_wait()

        for repeat in range(args.repeat):
            profile_before_reset = _profile_usec(env) if args.profile else {}
            env.reset()
            profile_after_reset = _profile_usec(env) if args.profile else {}
            reset_profile = (
                _profile_delta(profile_before_reset, profile_after_reset)
                if args.profile
                else {}
            )
            resets_before = env.reset_count
            profile_before = _profile_usec(env) if args.profile else {}
            started = time.perf_counter()
            for step in range(args.steps_per_agent):
                env.step_async(repeat_actions[step, agent_slice])
                env.step_wait()
            elapsed = time.perf_counter() - started
            profile_after = _profile_usec(env) if args.profile else {}
            step_profile = (
                _profile_delta(profile_before, profile_after)
                if args.profile
                else {}
            )
            decisions = args.steps_per_agent * total_agents
            row = {
                "agents_per_process": agents_per_process,
                "processes": processes,
                "total_agents": total_agents,
                "steps_per_agent": args.steps_per_agent,
                "repeat": repeat,
                "decisions": decisions,
                "elapsed_s": elapsed,
                "decisions_per_s": decisions / elapsed if elapsed else 0.0,
                "per_agent_decisions_per_s": (
                    decisions / elapsed / total_agents if elapsed else 0.0
                ),
                "resets": env.reset_count - resets_before,
            }
            if args.profile:
                row["profile_reset_ms"] = (
                    reset_profile.get("reset", 0) / 1000.0
                )
                for key in PROFILE_KEYS:
                    usec = step_profile.get(key, 0)
                    row[f"profile_{key}_ms"] = usec / 1000.0
                    row[f"profile_{key}_ms_per_decision"] = (
                        usec / 1000.0 / decisions if decisions else 0.0
                    )
            rows.append(row)
            print(
                f"n={agents_per_process:>3} p={processes:>2} "
                f"agents={total_agents:>3} r={repeat} "
                f"{row['decisions_per_s']:>9.1f} decisions/s "
                f"{row['per_agent_decisions_per_s']:>7.1f}/agent "
                f"resets={row['resets']}",
                flush=True,
            )
            if args.profile:
                parts = [
                    f"{key}={row[f'profile_{key}_ms_per_decision']:.2f}"
                    for key in PROFILE_KEYS
                ]
                print("      ms/decision: " + " ".join(parts), flush=True)
    finally:
        env.close()
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        key = (row["agents_per_process"], row["processes"])
        groups.setdefault(key, []).append(row)
    summary: list[dict] = []
    for (agents_per_process, processes), group in sorted(groups.items()):
        item = {
            "agents_per_process": agents_per_process,
            "processes": processes,
            "total_agents": group[0]["total_agents"],
            "mean_decisions_per_s": statistics.mean(
                row["decisions_per_s"] for row in group
            ),
            "mean_per_agent_decisions_per_s": statistics.mean(
                row["per_agent_decisions_per_s"] for row in group
            ),
            "mean_elapsed_s": statistics.mean(
                row["elapsed_s"] for row in group
            ),
            "mean_resets": statistics.mean(row["resets"] for row in group),
        }
        for key in PROFILE_KEYS:
            field = f"profile_{key}_ms_per_decision"
            if field in group[0]:
                item[f"mean_{field}"] = statistics.mean(
                    row[field] for row in group
                )
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_action_stream(
    args: argparse.Namespace,
    max_agents: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-agent action streams that do not depend on topology width.

    Each agent gets its own RNG, so agent 0 sees the same actions whether the
    run has 1, 8, or 32 agents in the process. This keeps agent-count sweeps
    comparable instead of changing the random stream when the width changes.
    """
    base = args.action_seed if args.action_seed is not None else args.seed
    rngs = [
        np.random.default_rng(base + 10_000 + agent)
        for agent in range(max_agents)
    ]

    def draw(rows: int) -> np.ndarray:
        if max_agents == 0:
            return np.empty((rows, 0), dtype=np.int64)
        return np.column_stack(
            [rng.integers(0, 2, size=rows, dtype=np.int64) for rng in rngs]
        )

    return draw(args.warmup), draw(args.steps_per_agent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents-per-process", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 100])
    parser.add_argument("--processes", type=int, nargs="+", default=[1])
    parser.add_argument("--level", default="1-11")
    parser.add_argument("--steps-per-agent", type=int, default=500)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--obs-encoding", default="multi_hot")
    parser.add_argument("--reward-profile", default="time")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable training-only Godot timers and write ms/decision columns.",
    )
    parser.add_argument(
        "--no-cosmetics",
        action="store_true",
        help="Training-only A/B: suppress effect scenes, player particles, and animations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action-seed",
        type=int,
        default=None,
        help="Use one fixed action stream seed for every topology.",
    )
    parser.add_argument("--base-port", type=int, default=13008)
    parser.add_argument("--env-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/topology_bench"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.env_path = args.env_path or resolve_env_path()
    args.output_dir = args.output_dir.resolve()
    combos = [
        (agents, processes)
        for agents in args.agents_per_process
        for processes in args.processes
    ]
    if args.dry_run:
        for index, (agents, processes) in enumerate(combos):
            print(
                f"n={agents} p={processes} agents={agents * processes} "
                f"port={args.base_port + index * 20}"
            )
        return 0

    max_agents = max(agents * processes for agents, processes in combos)
    warmup_actions, repeat_actions = build_action_stream(args, max_agents)

    rows: list[dict] = []
    for index, (agents, processes) in enumerate(combos):
        rows.extend(
            run_topology(
                args,
                agents,
                processes,
                index,
                warmup_actions,
                repeat_actions,
            )
        )
    summary = summarize(rows)
    write_csv(args.output_dir / "raw.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    print("--- summary")
    for row in summary:
        print(
            f"n={row['agents_per_process']:>3} p={row['processes']:>2} "
            f"agents={row['total_agents']:>3} "
            f"{row['mean_decisions_per_s']:>9.1f} decisions/s "
            f"{row['mean_per_agent_decisions_per_s']:>7.1f}/agent"
        )
        if args.profile:
            parts = [
                f"{key}={row[f'mean_profile_{key}_ms_per_decision']:.2f}"
                for key in PROFILE_KEYS
                if f"mean_profile_{key}_ms_per_decision" in row
            ]
            print("      ms/decision: " + " ".join(parts))
    print(f"summary: {args.output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
