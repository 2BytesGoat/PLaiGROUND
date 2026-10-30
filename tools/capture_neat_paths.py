#!/usr/bin/env python3
"""Capture clean NEAT routes and the selected expert's time on their starts.

The output is one JSONL record per route. It preserves the route signature,
path cells/facings, the shortest successful NEAT action sequence found in the
archive's ``finishers.jsonl``, the filtered split/cluster when available, and
the selected expert's successful start times for that route.

Example:
    .venv/bin/python tools/capture_neat_paths.py \
      --level 1-10 \
      --routes runs/neat_transfer_1_10_from_1_9/filtered_clean_g131/filtered_clean_routes.json \
      --finishers runs/neat_transfer_1_10_from_1_9/archive/finishers.jsonl \
      --expert-metadata runs/distill_campaign_student/raw/expert_1-10/metadata.json \
      --output runs/distill_campaign_student/paths/1-10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_routes(path: Path, clean_only: bool) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "routes" in data:
        data = data["routes"]
    if isinstance(data, dict):
        routes = list(data.values())
    elif isinstance(data, list):
        routes = data
    else:
        raise SystemExit(f"{path}: expected a route list or mapping")
    if clean_only:
        routes = [route for route in routes if route.get("clean", True)]
    return routes


def load_actions(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    best: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            signature = record.get("route_signature")
            actions = record.get("actions")
            if not signature or not isinstance(actions, list) or not actions:
                continue
            try:
                run_time = float(record.get("run_time", float("inf")))
            except (TypeError, ValueError):
                run_time = float("inf")
            previous = best.get(str(signature))
            if previous is None or run_time < float(previous["run_time"]):
                best[str(signature)] = {
                    "actions": [int(value) for value in actions],
                    "run_time": run_time,
                    "generation": record.get("generation"),
                    "genome_id": record.get("genome_id"),
                }
    return best


def load_expert_starts(path: Path | None) -> dict[str, list[dict]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_route: dict[str, list[dict]] = {}
    for row in data.get("starts", []):
        signature = str(row.get("source_route", ""))
        if not signature:
            continue
        by_route.setdefault(signature, []).append(
            {
                "split": row.get("split"),
                "start_cell": row.get("start_cell"),
                "start_facing": row.get("start_facing"),
                "run_time_at_state": row.get("run_time_at_state"),
                "finished": bool(row.get("finished")),
                "run_time": row.get("run_time"),
                "deaths": row.get("deaths"),
                "timed_out": row.get("timed_out"),
            }
        )
    return by_route


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--finishers", type=Path, default=None)
    parser.add_argument("--expert-metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-unclean",
        action="store_true",
        help="Keep routes with clean=false instead of filtering them out.",
    )
    args = parser.parse_args()

    routes = load_routes(args.routes, clean_only=not args.include_unclean)
    actions = load_actions(args.finishers)
    expert_starts = load_expert_starts(args.expert_metadata)
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    missing_actions = 0
    for route in routes:
        signature = str(route.get("route_signature", ""))
        action_record = actions.get(signature, {})
        if not action_record:
            missing_actions += 1
        starts = expert_starts.get(signature, [])
        successful = [
            start for start in starts if start.get("finished") and start.get("run_time")
        ]
        best_expert_time = (
            min(float(start["run_time"]) for start in successful) if successful else None
        )
        records.append(
            {
                "level": args.level,
                "route_signature": signature,
                "source_run": str(args.routes.parent),
                "run_time": route.get("run_time"),
                "clean": bool(route.get("clean", True)),
                "split": route.get("split"),
                "cluster_id": route.get("cluster_id"),
                "generation": route.get("generation"),
                "genome_id": route.get("genome_id"),
                "path_length": route.get("path_length"),
                "revisit_ratio": route.get("revisit_ratio"),
                "backtracks": route.get("backtracks"),
                "action_flip_rate": route.get("action_flip_rate"),
                "checkpoint_file": route.get("checkpoint_file"),
                "genome_file": route.get("genome_file"),
                "path_cells": route.get("path_cells"),
                "path_facings": route.get("path_facings"),
                "neat_actions": action_record.get("actions"),
                "neat_action_count": len(action_record.get("actions", [])),
                "neat_action_run_time": action_record.get("run_time"),
                "expert_starts": starts,
                "best_expert_time": best_expert_time,
            }
        )

    output_path = args.output / "paths.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "level": args.level,
        "route_source": str(args.routes),
        "finishers": str(args.finishers) if args.finishers else None,
        "expert_metadata": str(args.expert_metadata) if args.expert_metadata else None,
        "routes": len(records),
        "routes_with_actions": sum(
            1 for record in records if record["neat_actions"] is not None
        ),
        "routes_missing_actions": missing_actions,
        "expert_starts": sum(len(record["expert_starts"]) for record in records),
        "expert_finished_starts": sum(
            sum(1 for start in record["expert_starts"] if start["finished"])
            for record in records
        ),
        "routes_with_expert_finish": sum(
            1 for record in records if record["best_expert_time"] is not None
        ),
        "best_route_time": min(
            (float(record["run_time"]) for record in records if record["run_time"] is not None),
            default=None,
        ),
        "best_expert_time": min(
            (
                float(record["best_expert_time"])
                for record in records
                if record["best_expert_time"] is not None
            ),
            default=None,
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
