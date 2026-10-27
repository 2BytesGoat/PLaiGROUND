"""Sequential 512 multi-hot curriculum with replay and all-seen gates.

Start from the accepted 1-1 model, train the next level, then evaluate every
level seen so far. If an old level is forgotten, replay it before moving on.
This is a continual-learning check, not a per-level specialist sweep.

Example:
    .venv/bin/python src/21_curriculum_8x8.py \
        --initial-model runs/topology/all_levels_8x8_w512/1-1/best_model.zip
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

PILOT = Path(__file__).with_name("10_ppo_pilot.py")
LEVELS = [f"1-{i}" for i in range(1, 17)]


def parse_levels(value: str) -> list[str]:
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [level for level in levels if level not in LEVELS]
    if unknown:
        raise ValueError(f"Unknown level(s): {unknown}")
    return levels


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=PILOT.parent.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def train_stage(
    args: argparse.Namespace,
    model: Path,
    level: str,
    output_dir: Path,
    timesteps: int,
    index: int,
) -> Path:
    command = [
        sys.executable,
        str(PILOT),
        "--level", level,
        "--n-envs", str(args.processes),
        "--agents-per-process", str(args.agents_per_process),
        "--timesteps", str(timesteps),
        "--max-decisions", str(args.max_decisions),
        "--action-repeat", str(args.action_repeat),
        "--speedup", str(args.speedup),
        "--reward-profile", args.reward_profile,
        "--obs-encoding", args.obs_encoding,
        "--net-arch", *[str(value) for value in args.net_arch],
        "--eval-agents", str(args.eval_agents),
        "--eval-every", str(args.eval_every),
        "--patience", str(args.patience),
        "--min-finish-rate", str(args.min_finish_rate),
        "--save-freq", str(args.save_freq),
        "--seed", str(args.seed),
        "--base-port", str(args.base_port + index * 20),
        "--eval-port", str(args.eval_port + index * 20),
        "--resume", str(model),
        "--output-dir", str(output_dir),
    ]
    run_command(command, output_dir / "train.log")
    best = output_dir / "best_model.zip"
    if best.exists():
        return best
    final = output_dir / "ppo_pilot.zip"
    if final.exists():
        return final
    raise FileNotFoundError(f"no model produced in {output_dir}")


def evaluate_level(
    args: argparse.Namespace,
    model: Path,
    level: str,
    output_dir: Path,
    index: int,
) -> dict:
    command = [
        sys.executable,
        str(PILOT),
        "--level", level,
        "--eval-only",
        "--resume", str(model),
        "--eval-agents", str(args.eval_agents),
        "--max-decisions", str(args.max_decisions),
        "--action-repeat", str(args.action_repeat),
        "--speedup", str(args.speedup),
        "--obs-encoding", args.obs_encoding,
        "--eval-port", str(args.eval_port + index * 20),
        "--output-dir", str(output_dir),
    ]
    run_command(command, output_dir / "eval.log")
    return read_summary(output_dir / "eval" / "summary.csv")


def evaluate_seen(
    args: argparse.Namespace,
    model: Path,
    seen: list[str],
    output_dir: Path,
    stage_index: int,
    round_index: int,
) -> tuple[bool, dict[str, dict]]:
    results: dict[str, dict] = {}
    for level_index, level in enumerate(seen):
        result = evaluate_level(
            args,
            model,
            level,
            output_dir / f"{level}_r{round_index}",
            stage_index * 100 + round_index * 20 + level_index,
        )
        results[level] = result
        finishes = int(result.get("finishes", 0) or 0)
        rate = finishes / max(1, args.eval_agents)
        print(
            f"  eval {level}: {finishes}/{args.eval_agents} "
            f"best={result.get('best_time', '?')}",
            flush=True,
        )
        if rate < args.min_finish_rate:
            return False, results
    return True, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--levels", default=",".join(LEVELS[1:]))
    parser.add_argument("--start-level", default="1-2")
    parser.add_argument(
        "--seen-levels",
        default="1-1",
        help="Previously accepted levels, comma-separated.",
    )
    parser.add_argument("--stage-timesteps", type=int, default=300_000)
    parser.add_argument("--replay-timesteps", type=int, default=50_000)
    parser.add_argument("--max-replay-rounds", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-finish-rate", type=float, default=0.90)
    parser.add_argument("--eval-agents", type=int, default=100)
    parser.add_argument("--agents-per-process", type=int, default=8)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--obs-encoding", default="multi_hot")
    parser.add_argument("--reward-profile", default="time")
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-port", type=int, default=14808)
    parser.add_argument("--eval-port", type=int, default=15800)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/curriculum/512_8x8_replay"),
    )
    args = parser.parse_args()

    if not args.initial_model.exists():
        raise FileNotFoundError(args.initial_model)
    levels = parse_levels(args.levels)
    if args.start_level not in levels:
        raise ValueError(f"--start-level {args.start_level} not in --levels")
    levels = levels[levels.index(args.start_level):]
    seen = parse_levels(args.seen_levels)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    current_model = args.initial_model.resolve()
    started = time.time()
    summary_rows: list[dict] = []

    for stage_index, level in enumerate(levels):
        print(f"\n=== curriculum {level} from {current_model} ===", flush=True)
        stage_dir = args.output_root / "stages" / level
        stage_model = train_stage(
            args,
            current_model,
            level,
            stage_dir,
            args.stage_timesteps,
            stage_index * 4,
        )
        stage_seen = seen + [level]
        eval_root = args.output_root / "evals" / level
        passed, results = evaluate_seen(
            args, stage_model, stage_seen, eval_root, stage_index, 0
        )

        round_index = 0
        while not passed and round_index < args.max_replay_rounds:
            round_index += 1
            failed = [
                seen_level
                for seen_level, result in results.items()
                if int(result.get("finishes", 0) or 0) / max(1, args.eval_agents)
                < args.min_finish_rate
            ]
            if not failed:
                break
            print(f"  replay round {round_index}: {failed}", flush=True)
            replay_dir = args.output_root / "replay" / level / f"round{round_index}"
            for replay_index, replay_level in enumerate(failed):
                stage_model = train_stage(
                    args,
                    stage_model,
                    replay_level,
                    replay_dir / replay_level,
                    args.replay_timesteps,
                    stage_index * 4 + round_index * 3 + replay_index,
                )
            passed, results = evaluate_seen(
                args,
                stage_model,
                stage_seen,
                eval_root,
                stage_index,
                round_index,
            )

        status = "accepted" if passed else "failed_gate"
        print(f"[{level}] {status}; model={stage_model}", flush=True)
        snapshot = args.output_root / "snapshots" / level
        snapshot.mkdir(parents=True, exist_ok=True)
        snapshot_model = snapshot / "ppo_pilot.zip"
        snapshot_model.write_bytes(stage_model.read_bytes())
        summary_rows.append(
            {
                "level": level,
                "status": status,
                "stage_model": str(stage_model),
                "snapshot": str(snapshot_model),
                "replay_rounds": round_index,
                "current_model": str(current_model),
                "observed_levels": ",".join(stage_seen),
                "best_times": ";".join(
                    f"{name}:{result.get('best_time', '')}"
                    for name, result in results.items()
                ),
                "wall_s": f"{time.time() - started:.0f}",
            }
        )
        summary_path = args.output_root / "curriculum_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        if not passed:
            print(
                f"[{level}] failed after replay; stopping. See {summary_path}",
                flush=True,
            )
            return 1
        current_model = stage_model
        seen.append(level)

    print(f"[curriculum] done in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
