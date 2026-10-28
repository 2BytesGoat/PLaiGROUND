#!/usr/bin/env python3
"""Clean the 1-1 NEAT finisher archive and check start-cell coverage.

The archive stores raw successful trajectories. This tool:

1. Collapses duplicate route signatures into the fastest clean trajectory per
   route and exports the action/path data needed for prefix replay.
2. Enumerates every stationary standable start cell (derived from the real
   exported 1-1 analysis), with optional both facings.
3. Runs the best clean archived NEAT genomes from every start and reports
   finish/time coverage, near-exit starts, and the canonical start separately.

It is deliberately a start-cell coverage test, not an arbitrary dynamic-state
test. No powerups are present on 1-1.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]

import neat

from plaiground.envs.godot import GodotEnv

DEFAULT_ARCHIVE = REPO / "runs" / "neat_1_1_8x24_archive" / "archive"
DEFAULT_CLEAN = REPO / "runs" / "neat_1_1_8x24_archive" / "clean"
DEFAULT_COVERAGE = REPO / "runs" / "neat_1_1_8x24_archive" / "coverage"
DEFAULT_ANALYSIS = (
    REPO.parent / "dragon-jump-remaster" / "runs" / "level_analysis" / "1-1.json"
)
DEFAULT_CONFIG = REPO / "src" / "plaiground" / "algorithms" / "genetic" / "neat" / "neat_1_1_config.txt"
DEFAULT_PORT = 15700
MAX_STEPS = 150
ACTION_REPEAT = 4
SPEEDUP = 8.0
LEVEL = "1-1"


def resolve_env_path() -> str:
    config_path = REPO / "src" / ".config"
    load_dotenv(config_path, override=True)
    env_dir = os.getenv("ENV_PATH", "./environments")
    env_name = os.getenv("ENV_NAME", "DragonJump")
    if os.path.isabs(env_dir):
        resolved = Path(env_dir)
    else:
        resolved = None
        for base in (Path.cwd(), config_path.parent, REPO):
            candidate = (base / env_dir).resolve()
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            resolved = (Path.cwd() / env_dir).resolve()
    return str(resolved / env_name)


def isolated_home(output_dir: Path, name: str) -> dict[str, str]:
    root = (output_dir / "userdata" / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["HOME"] = str(root)
    child_env["XDG_DATA_HOME"] = str(root / "xdg_data")
    child_env["XDG_CONFIG_HOME"] = str(root / "xdg_config")
    return child_env


def load_config(path: Path) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )


def load_genome(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def make_env(cells: int, port: int, output_dir: Path, level: str) -> GodotEnv:
    log_dir = output_dir / "godot"
    log_dir.mkdir(parents=True, exist_ok=True)
    return GodotEnv(
        env_path=resolve_env_path(),
        port=port,
        show_window=False,
        seed=42,
        action_repeat=ACTION_REPEAT,
        speedup=SPEEDUP,
        nb_agents=cells,
        level=level,
        convert_action_space=True,
        log_path=str(log_dir / "coverage.log"),
        env=isolated_home(output_dir, "coverage"),
        reward_profile="time",
        obs_encoding="legacy",
    )


def standable_starts(
    analysis_path: Path,
    both_facings: bool,
    min_exit_cells: int,
) -> list[dict]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    symbols = analysis["symbols"]
    flow = analysis["flow"]
    exit_cell = [int(analysis["exit"][0]), int(analysis["exit"][1])]
    player_start = [
        int(analysis["player_start"][0]),
        int(analysis["player_start"][1]),
    ]
    height = len(symbols)
    width = len(symbols[0])
    starts: list[dict] = []
    for y in range(height):
        for x in range(width):
            symbol = str(symbols[y][x])
            if x == exit_cell[0] and y == exit_cell[1]:
                continue
            if any(mark in symbol for mark in ("W", "Y", "R", "Q")):
                continue
            if y + 1 >= height or "W" not in str(symbols[y + 1][x]):
                continue
            flow_value = flow[y][x]
            if flow_value is None or float(flow_value) < min_exit_cells:
                continue
            facings = (1, -1) if both_facings else (1,)
            for facing in facings:
                near_exit = (
                    abs(x - exit_cell[0]) <= 2
                    and abs(y - exit_cell[1]) <= 2
                )
                starts.append(
                    {
                        "cell": [x, y],
                        "facing": facing,
                        "near_exit": near_exit,
                        "canonical": [x, y] == player_start and facing == 1,
                    }
                )
    return starts


def clean_archive(args) -> int:
    archive_dir = args.archive_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    routes = json.loads((archive_dir / "routes.json").read_text(encoding="utf-8"))
    finishers_path = archive_dir / "finishers.jsonl"
    finishers = []
    for line in finishers_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            finishers.append(json.loads(line))

    best_by_signature: dict[str, dict] = {}
    for record in finishers:
        if not record.get("clean"):
            continue
        signature = str(record.get("route_signature", ""))
        if not signature:
            continue
        previous = best_by_signature.get(signature)
        if previous is None or float(record["run_time"]) < float(
            previous["run_time"]
        ):
            best_by_signature[signature] = record

    clean_routes = []
    for signature, record in best_by_signature.items():
        route = routes.get(signature, {})
        genome_file = route.get("genome_file")
        if not genome_file:
            continue
        if int(record.get("path_length", 0)) < args.min_path_cells:
            continue
        clean_routes.append(
            {
                "route_signature": signature,
                "run_time": float(record["run_time"]),
                "path_cells": record.get("path_cells", []),
                "path_facings": record.get("path_facings", []),
                "actions": record.get("actions", []),
                "checkpoint_file": route.get("checkpoint_file")
                or record.get("checkpoint_file"),
                "path_length": int(record.get("path_length", 0)),
                "unique_cells": int(record.get("unique_cells", 0)),
                "revisit_ratio": float(record.get("revisit_ratio", 0.0)),
                "backtracks": int(record.get("backtracks", 0)),
                "action_flip_rate": float(record.get("action_flip_rate", 0.0)),
                "genome_file": genome_file,
                "generation": int(record.get("generation", 0)),
                "genome_id": int(record.get("genome_id", 0)),
            }
        )
    clean_routes.sort(key=lambda item: item["run_time"])

    (output_dir / "clean_routes.json").write_text(
        json.dumps(clean_routes, indent=2), encoding="utf-8"
    )
    top_genomes = []
    seen = set()
    for route in clean_routes:
        genome = route["genome_file"]
        if genome in seen:
            continue
        seen.add(genome)
        top_genomes.append(
            {
                "genome_file": genome,
                "run_time": route["run_time"],
                "route_signature": route["route_signature"],
            }
        )
        if len(top_genomes) >= args.top_k:
            break

    for name in ("best_time_genome.pkl", "best_clean_genome.pkl"):
        path = archive_dir.parent / name
        if not path.exists():
            continue
        relative = os.path.relpath(path, archive_dir)
        if relative in seen:
            continue
        top_genomes.append(
            {
                "genome_file": relative,
                "run_time": None,
                "route_signature": None,
            }
        )
        seen.add(relative)

    summary = {
        "level": args.level,
        "observation": "legacy_58",
        "clean_routes": len(clean_routes),
        "raw_finishers": len(finishers),
        "top_genomes": top_genomes,
        "best_clean_time": min(
            (route["run_time"] for route in clean_routes), default=None
        ),
    }
    (output_dir / "top_genomes.json").write_text(
        json.dumps(top_genomes, indent=2), encoding="utf-8"
    )
    (output_dir / "clean_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"[clean] {len(clean_routes)} clean routes from {len(finishers)} finishers; "
        f"top={len(top_genomes)} genomes; best={summary['best_clean_time']}",
        flush=True,
    )
    return len(clean_routes)


def run_genome(
    env: GodotEnv,
    net,
    starts: list[dict],
    max_steps: int,
) -> list[dict]:
    spawns = [
        {
            "agent": index,
            "cell": item["cell"],
            "facing": item["facing"],
        }
        for index, item in enumerate(starts)
    ]
    obs, _infos = env.reset(spawns=spawns)
    count = len(starts)
    done = np.zeros(count, dtype=bool)
    finished = np.zeros(count, dtype=bool)
    finish_times = np.full(count, np.inf, dtype=np.float64)
    steps = np.zeros(count, dtype=np.int32)
    actual_spawns: list[list[int] | None] = [None] * count
    paths: list[list[list[int]]] = [[] for _ in range(count)]

    for step in range(max_steps):
        actions = []
        for index in range(count):
            if done[index]:
                actions.append([0])
                continue
            if step == 0:
                # Legacy coverage of pre-force-start genomes: force the old
                # first jump. New training/eval does not inject it.
                jump = 1
            else:
                jump = int(net.activate(obs[index]["obs"])[0] > 0.5)
            actions.append([jump])
        obs, _rewards, dones, _timeouts, infos = env.step(
            np.asarray(actions, dtype=np.int64)
        )
        for index in range(count):
            if done[index]:
                continue
            info = infos[index] if index < len(infos) else {}
            cell = info.get("cell")
            if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                paths[index].append([int(cell[0]), int(cell[1])])
            if actual_spawns[index] is None:
                spawn = info.get("spawn_cell", starts[index]["cell"])
                if isinstance(spawn, (list, tuple)) and len(spawn) >= 2:
                    actual_spawns[index] = [int(spawn[0]), int(spawn[1])]
            steps[index] = step + 1
            if dones[index]:
                done[index] = True
                finished[index] = bool(
                    info.get("finished", float(_rewards[index]) > 0.0)
                )
                if finished[index]:
                    finish_times[index] = float(
                        info.get("run_time", np.inf)
                    )
        if done.all():
            break

    results = []
    for index, start in enumerate(starts):
        results.append(
            {
                **start,
                "actual_spawn": actual_spawns[index],
                "spawn_ok": actual_spawns[index] == start["cell"],
                "finished": bool(finished[index]),
                "run_time": (
                    float(finish_times[index])
                    if finished[index] and np.isfinite(finish_times[index])
                    else None
                ),
                "steps": int(steps[index]),
                "path_cells": paths[index],
            }
        )
    return results


def coverage(args) -> int:
    clean_dir = args.clean_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    top_path = clean_dir / "top_genomes.json"
    if not top_path.exists():
        raise FileNotFoundError(f"{top_path} missing; run clean first")
    top_genomes = json.loads(top_path.read_text(encoding="utf-8"))
    starts = standable_starts(
        args.level_analysis,
        args.both_facings,
        args.min_exit_cells,
    )
    if not starts:
        raise RuntimeError("no standable starts found")
    (output_dir / "starts.json").write_text(
        json.dumps(starts, indent=2), encoding="utf-8"
    )

    config = load_config(args.config)
    env = make_env(len(starts), args.port, output_dir, args.level)
    runs: dict[str, list[dict]] = {}
    try:
        for index, item in enumerate(top_genomes):
            genome_path = (args.archive_dir / item["genome_file"]).resolve()
            if not genome_path.exists():
                print(f"[coverage] missing genome {genome_path}", flush=True)
                continue
            genome = load_genome(genome_path)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            print(
                f"[coverage {index + 1}/{len(top_genomes)}] "
                f"{genome_path.name} starts={len(starts)}",
                flush=True,
            )
            results = run_genome(env, net, starts, args.max_steps)
            runs[str(item["genome_file"])] = results
            finishes = sum(item["finished"] for item in results)
            print(
                f"  finishes={finishes}/{len(results)} "
                f"canonical={next((r['run_time'] for r in results if r['canonical'] and r['finished']), None)}",
                flush=True,
            )
    finally:
        env.close()

    union = {}
    for genome_file, results in runs.items():
        for item in results:
            key = f"{item['cell'][0]},{item['cell'][1]},{item['facing']}"
            if not item["finished"]:
                continue
            previous = union.get(key)
            if previous is None or float(item["run_time"]) < float(
                previous["run_time"]
            ):
                union[key] = {
                    **item,
                    "genome_file": genome_file,
                }

    total = len(starts)
    near_exit = [item for item in starts if item["near_exit"]]
    meaningful = [item for item in starts if not item["near_exit"]]
    canonical = [item for item in starts if item["canonical"]]

    def covered(items: list[dict]) -> int:
        return sum(
            1
            for item in items
            if f"{item['cell'][0]},{item['cell'][1]},{item['facing']}" in union
        )

    summary = {
        "level": args.level,
        "observation": "legacy_58",
        "starts": total,
        "both_facings": bool(args.both_facings),
        "min_exit_cells": args.min_exit_cells,
        "genomes": len(runs),
        "max_steps": args.max_steps,
        "union_finishes": covered(starts),
        "union_meaningful_finishes": covered(meaningful),
        "meaningful_starts": len(meaningful),
        "union_near_exit_finishes": covered(near_exit),
        "near_exit_starts": len(near_exit),
        "canonical_finishes": covered(canonical),
        "canonical_starts": len(canonical),
        "per_genome": {
            genome_file: {
                "finishes": sum(item["finished"] for item in results),
                "starts": len(results),
                "best_time": min(
                    (
                        item["run_time"]
                        for item in results
                        if item["finished"] and item["run_time"] is not None
                    ),
                    default=None,
                ),
            }
            for genome_file, results in runs.items()
        },
        "failures": [
            item
            for item in starts
            if f"{item['cell'][0]},{item['cell'][1]},{item['facing']}" not in union
        ],
    }
    (output_dir / "coverage_runs.json").write_text(
        json.dumps(runs, indent=2), encoding="utf-8"
    )
    (output_dir / "coverage_union.json").write_text(
        json.dumps(list(union.values()), indent=2), encoding="utf-8"
    )
    (output_dir / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary["union_finishes"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean")
    clean.add_argument("--level", default=LEVEL)
    clean.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    clean.add_argument("--output-dir", type=Path, default=DEFAULT_CLEAN)
    clean.add_argument("--top-k", type=int, default=10)
    clean.add_argument("--min-path-cells", type=int, default=5)
    clean.set_defaults(func=clean_archive)

    cov = subparsers.add_parser("coverage")
    cov.add_argument("--level", default=LEVEL)
    cov.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    cov.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN)
    cov.add_argument("--output-dir", type=Path, default=DEFAULT_COVERAGE)
    cov.add_argument("--level-analysis", type=Path, default=DEFAULT_ANALYSIS)
    cov.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cov.add_argument("--both-facings", action="store_true")
    cov.add_argument(
        "--min-exit-cells",
        type=int,
        default=5,
        help="Exclude starts whose BFS distance-to-exit is below this.",
    )
    cov.add_argument("--max-steps", type=int, default=MAX_STEPS)
    cov.add_argument("--port", type=int, default=DEFAULT_PORT)
    cov.set_defaults(func=coverage)

    args = parser.parse_args()
    args.archive_dir = args.archive_dir.resolve()
    if args.command == "clean":
        args.output_dir = args.output_dir.resolve()
        return 0 if clean_archive(args) else 1
    args.config = args.config.resolve()
    args.clean_dir = args.clean_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.level_analysis = args.level_analysis.resolve()
    return 0 if coverage(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
