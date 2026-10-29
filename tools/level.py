#!/usr/bin/env python3
"""Prepare and inspect one Dragon Jump level's expert pipeline.

``data`` is the repeatable post-NEAT step: it classifies route tiers, filters
the selected tier, builds train/holdout start pools, and writes the
deduplicated evaluation union. ``status`` is a read-only view of the level's
experiments and generated data summaries.

This tool does not train a policy or select a checkpoint. Long runs and the
accept/repair decision stay explicit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from plaiground.data.level_data import (
    DEFAULT_PIPELINE_SEED,
    QUALITY_CHOICES,
    LevelDataError,
    prepare_level_data,
)
from plaiground.data.route_filter import (
    DEFAULT_JACCARD,
    DEFAULT_MAX_PER_CLUSTER,
)
from plaiground.experiments.config import load_experiment

REPO = Path(__file__).resolve().parents[1]
GAME_REPO = REPO.parent / "dragon-jump-remaster"
LEVEL_DATA = GAME_REPO / "godot" / "src" / "resources" / "level_data"


def _default_analysis(level: str) -> Path:
    return GAME_REPO / "runs" / "level_analysis" / f"{level}.json"


def _infer_generation(run_dir: Path) -> int | None:
    generations = []
    for path in run_dir.glob("neat-checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            generations.append(int(suffix))
    if generations:
        return max(generations)
    best_clean = run_dir / "best_clean.json"
    if best_clean.exists():
        try:
            payload = json.loads(best_clean.read_text(encoding="utf-8"))
            value = payload.get("generation")
            return int(value) if value is not None else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
    return None


def _level_meta(level: str) -> dict[str, Any]:
    path = LEVEL_DATA / f"{level}.tres"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    display = re.search(r'^display_name = "([^"]*)"', text, re.M)
    par = re.search(r"^par_time = ([0-9.]+)", text, re.M)
    return {
        "display_name": display.group(1) if display else None,
        "par_time": float(par.group(1)) if par else None,
    }


def _experiment_levels(config: dict[str, Any]) -> list[str]:
    env = config.get("env", {})
    value = env.get("levels", env.get("level"))
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _experiments_for_level(level: str) -> list[dict[str, Any]]:
    experiments = []
    for config_path in sorted((REPO / "experiments").glob("*/config.toml")):
        try:
            config = load_experiment(config_path)
        except (OSError, ValueError):
            continue
        if level not in _experiment_levels(config):
            continue
        experiments.append(config)
    return experiments


def _experiment_row(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    workflow = config.get("workflow", {}).get("name")
    output = config.get("output", {}).get("dir")
    return {
        "id": str(experiment.get("id", "")),
        "name": str(experiment.get("name", "")),
        "status": experiment.get("status"),
        "workflow": workflow,
        "output": output,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _count_checkpoints(run_dir: Path) -> int:
    return len(list((run_dir / "dqn" / "checkpoints").glob("*.zip")))


def _status_payload(level: str) -> dict[str, Any]:
    meta = _level_meta(level)
    experiments = [_experiment_row(config) for config in _experiments_for_level(level)]
    data_summaries = []
    for experiment in experiments:
        output = experiment.get("output")
        if not output:
            continue
        run_dir = (REPO / output).resolve()
        summary_path = run_dir / "level_data_summary.json"
        summary = _read_json(summary_path)
        if summary is not None:
            data_summaries.append(
                {
                    "experiment": experiment["id"],
                    "run_dir": str(run_dir),
                    "summary_path": str(summary_path),
                    "selected_quality": summary.get("selected_quality"),
                    "selection_reason": summary.get("selection_reason"),
                    "accepted": summary.get("accepted", {}).get("pool", {}),
                    "warnings": summary.get("warnings", []),
                }
            )
    return {
        "level": level,
        "display_name": meta.get("display_name"),
        "par_time": meta.get("par_time"),
        "gate": (
            float(meta["par_time"]) + 0.30
            if meta.get("par_time") is not None
            else None
        ),
        "experiments": experiments,
        "data_summaries": data_summaries,
    }


def _print_status(payload: dict[str, Any]) -> None:
    level = payload["level"]
    name = payload.get("display_name") or ""
    heading = f"Level {level}"
    if name:
        heading += f" ({name})"
    print(heading, flush=True)
    par = payload.get("par_time")
    gate = payload.get("gate")
    if par is not None:
        print(
            f"  par={par:.4f}s  provisional gate={gate:.4f}s (par + 0.30s)",
            flush=True,
        )
    else:
        print("  par: unknown (level file not found)", flush=True)

    experiments = payload["experiments"]
    if not experiments:
        print("  experiments: none", flush=True)
    else:
        print("  experiments:", flush=True)
        seen_runs: set[str] = set()
        for experiment in experiments:
            print(
                f"    {experiment['id']} {experiment['name']} "
                f"[{experiment.get('status')}] {experiment.get('workflow')}",
                flush=True,
            )
            output = experiment.get("output")
            if not output:
                continue
            run_dir = (REPO / output).resolve()
            run_key = str(run_dir)
            if run_key in seen_runs:
                continue
            seen_runs.add(run_key)
            checkpoints = len(list(run_dir.glob("neat-checkpoint-*")))
            if checkpoints:
                print(
                    f"      run={run_dir} neat_checkpoints={checkpoints}",
                    flush=True,
                )
            dqn_checkpoints = _count_checkpoints(run_dir)
            if dqn_checkpoints:
                print(
                    f"      run={run_dir} dqn_checkpoints={dqn_checkpoints}",
                    flush=True,
                )

    summaries = payload["data_summaries"]
    if summaries:
        print("  data summaries:", flush=True)
        for summary in summaries:
            pool = summary.get("accepted", {})
            reason = summary.get("selection_reason")
            reason_text = f" ({reason})" if reason else ""
            print(
                f"    {summary['summary_path']} "
                f"selected={summary.get('selected_quality')}{reason_text} "
                f"eval_states={pool.get('unique_eval_states')} "
                f"train_states={pool.get('train_states')} "
                f"holdout_states={pool.get('holdout_states')}",
                flush=True,
            )
            for warning in summary.get("warnings", []):
                print(f"      WARNING: {warning}", flush=True)
    else:
        print("  data summaries: none", flush=True)

    print("  next:", flush=True)
    neat_experiments = [
        experiment
        for experiment in experiments
        if experiment.get("workflow") == "genetic.neat"
    ]
    expert_experiments = [
        experiment
        for experiment in experiments
        if experiment.get("workflow") == "reinforcement.ddqn"
    ]
    if not neat_experiments:
        print("    create a NEAT transfer manifest; see docs/runbooks/new-level.md")
    elif not summaries:
        latest = neat_experiments[-1]
        output = latest.get("output") or "<run-dir>"
        print(
            "    tools/level.py data --level "
            f"{level} --run-dir {output} --analysis {_default_analysis(level)}"
        )
    elif not expert_experiments:
        print("    create the expert manifest from the selected pool")
    else:
        print("    evaluate every checkpoint; see docs/runbooks/new-level.md")


def _parse_data_args(args: argparse.Namespace) -> None:
    if args.analysis is None:
        args.analysis = _default_analysis(args.level)
    if args.output_root is None:
        args.output_root = args.run_dir
    if args.generation is None:
        args.generation = _infer_generation(args.run_dir)


def _run_data(args: argparse.Namespace) -> int:
    _parse_data_args(args)
    try:
        summary = prepare_level_data(
            level=args.level,
            run_dir=args.run_dir,
            generation=args.generation,
            analysis_path=args.analysis,
            output_root=args.output_root,
            archive_dir=args.archive_dir,
            quality=args.quality,
            min_routes=args.min_routes,
            tag=args.tag,
            build_all_success=not args.no_all_success,
            jaccard=args.jaccard,
            max_per_cluster=args.max_per_cluster,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
            canonical_weight=args.canonical_weight,
            state_weight=args.state_weight,
            min_time_to_exit=args.min_time_to_exit,
        )
    except (LevelDataError, FileNotFoundError, ValueError) as exc:
        print(f"level data: {exc}", file=sys.stderr, flush=True)
        return 2

    quality = summary["quality"]["qualities"]
    accepted = summary["accepted"]
    pool = accepted["pool"]
    print(
        f"[level] {args.level} archive={summary['quality']['input_routes']} "
        f"strict={quality['strict_clean']['routes']} "
        f"near={quality['near_clean']['routes']} "
        f"all={quality['all_success']['routes']} "
        f"selected={summary['selected_quality']} "
        f"({summary['selection_reason']})",
        flush=True,
    )
    print(
        f"[level] accepted routes={accepted['routes']} "
        f"filtered={accepted['filter']['filtered_routes']} "
        f"train/holdout routes="
        f"{accepted['filter']['train_routes']}/"
        f"{accepted['filter']['holdout_routes']} "
        f"states={pool['train_states']}/"
        f"{pool['holdout_states']} "
        f"eval={pool['unique_eval_states']}",
        flush=True,
    )
    if summary.get("all_success"):
        diagnostic = summary["all_success"]
        print(
            f"[level] all-success diagnostic: "
            f"states={diagnostic['pool']['train_states']}/"
            f"{diagnostic['pool']['holdout_states']} "
            f"eval={diagnostic['pool']['unique_eval_states']}",
            flush=True,
        )
    for warning in summary.get("warnings", []):
        print(f"[level] WARNING: {warning}", flush=True)
    print(f"[level] wrote {summary['summary_path']}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser(
        "data",
        help="classify routes, filter, and build train/holdout/eval pools",
    )
    data.add_argument("--level", required=True)
    data.add_argument("--run-dir", type=Path, required=True)
    data.add_argument(
        "--generation",
        type=int,
        default=None,
        help="NEAT generation tag; inferred from neat-checkpoint-* if omitted.",
    )
    data.add_argument(
        "--analysis",
        type=Path,
        default=None,
        help="Game level analysis JSON (default: sibling game repo 1-9.json).",
    )
    data.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="NEAT archive (default: <run-dir>/archive).",
    )
    data.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Write derived files here (default: <run-dir>).",
    )
    data.add_argument(
        "--quality",
        choices=("auto", *QUALITY_CHOICES),
        default="auto",
        help="Accepted tier; auto prefers strict-clean, then near-clean.",
    )
    data.add_argument("--min-routes", type=int, default=25)
    data.add_argument(
        "--tag",
        default=None,
        help="Accepted output tag (default: clean or near_clean).",
    )
    data.add_argument(
        "--no-all-success",
        action="store_true",
        help="Skip the all-success diagnostic filtered/pool artifacts.",
    )
    data.add_argument("--jaccard", type=float, default=DEFAULT_JACCARD)
    data.add_argument(
        "--max-per-cluster", type=int, default=DEFAULT_MAX_PER_CLUSTER
    )
    data.add_argument("--holdout-fraction", type=float, default=0.25)
    data.add_argument("--seed", type=int, default=DEFAULT_PIPELINE_SEED)
    data.add_argument("--canonical-weight", type=float, default=0.5)
    data.add_argument("--state-weight", type=float, default=0.5)
    data.add_argument("--min-time-to-exit", type=float, default=0.5)
    data.set_defaults(handler=_run_data)

    status = subparsers.add_parser(
        "status", help="show the level's experiments and data summaries"
    )
    status.add_argument("level")
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_run_status)
    return parser


def _run_status(args: argparse.Namespace) -> int:
    payload = _status_payload(args.level)
    if args.json:
        print(json.dumps(payload, indent=2), flush=True)
    else:
        _print_status(payload)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
