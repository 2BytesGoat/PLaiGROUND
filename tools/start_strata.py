#!/usr/bin/env python3
"""Stratify standable starts by shortest-path distance to the exit.

The source distribution is every stationary standable cell from the exported
level analysis, not the union of visited route tiles. The top distance quartile
is the hard set; ``--medium-min``..``--medium-max`` is the medium band; below
``--medium-min`` is the near-exit/easy band and is excluded from the training
sampler. The canonical start is kept separately as an anchor. Level and medium
band are explicit so the same generator works for 1-1 and later levels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = (
    REPO.parent / "dragon-jump-remaster" / "runs" / "level_analysis" / "1-1.json"
)
DEFAULT_COVERAGE = REPO / "runs" / "neat_1_1_8x24_archive" / "coverage_min5"
DEFAULT_OUTPUT = REPO / "runs" / "neat_1_1_8x24_archive" / "strata"


def standable_cells(analysis: dict) -> list[dict]:
    symbols = analysis["symbols"]
    flow = analysis["flow"]
    exit_cell = [int(analysis["exit"][0]), int(analysis["exit"][1])]
    height = len(symbols)
    width = len(symbols[0])
    cells = []
    for y in range(height):
        for x in range(width):
            symbol = str(symbols[y][x])
            if [x, y] == exit_cell:
                continue
            if any(mark in symbol for mark in ("W", "Y", "R", "Q")):
                continue
            if y + 1 >= height or "W" not in str(symbols[y + 1][x]):
                continue
            distance = flow[y][x]
            if distance is None or float(distance) < 0:
                continue
            cells.append(
                {
                    "cell": [x, y],
                    "distance": int(float(distance)),
                    "near_exit": (
                        abs(x - exit_cell[0]) <= 2
                        and abs(y - exit_cell[1]) <= 2
                    ),
                }
            )
    return cells


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="1-1")
    parser.add_argument("--level-analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--medium-min", type=int, default=9)
    parser.add_argument("--medium-max", type=int, default=12)
    args = parser.parse_args()

    if args.medium_max < args.medium_min:
        raise ValueError("--medium-max must be >= --medium-min")
    analysis = json.loads(args.level_analysis.read_text(encoding="utf-8"))
    cells = standable_cells(analysis)
    distances = [item["distance"] for item in cells]
    hard_threshold = percentile(distances, 0.75)
    canonical_cell = [
        int(analysis["player_start"][0]),
        int(analysis["player_start"][1]),
    ]

    starts = []
    for item in cells:
        for facing in (1, -1):
            is_canonical = item["cell"] == canonical_cell and facing == 1
            if is_canonical:
                stratum = "canonical"
            elif item["distance"] >= hard_threshold:
                stratum = "hard"
            elif args.medium_min <= item["distance"] <= args.medium_max:
                stratum = "medium"
            else:
                stratum = "easy"
            starts.append(
                {
                    **item,
                    "facing": facing,
                    "canonical": is_canonical,
                    "stratum": stratum,
                }
            )

    coverage_summary_path = args.coverage_dir / "coverage_summary.json"
    coverage_union_path = args.coverage_dir / "coverage_union.json"
    coverage = {}
    if coverage_summary_path.exists():
        summary = json.loads(coverage_summary_path.read_text(encoding="utf-8"))
        union = json.loads(coverage_union_path.read_text(encoding="utf-8"))
        min_coverage_distance = int(summary.get("min_exit_cells", 5))
        by_key = {
            f"{item['cell'][0]},{item['cell'][1]},{item['facing']}": item
            for item in union
        }
        for start in starts:
            if start["distance"] < min_coverage_distance:
                start["coverage_finished"] = None
                start["coverage_time"] = None
                start["coverage_genome"] = None
                continue
            key = f"{start['cell'][0]},{start['cell'][1]},{start['facing']}"
            hit = by_key.get(key)
            start["coverage_finished"] = bool(hit)
            start["coverage_time"] = (
                float(hit["run_time"]) if hit and hit.get("run_time") is not None else None
            )
            start["coverage_genome"] = hit.get("genome_file") if hit else None
        coverage = summary

    by_stratum: dict[str, list[dict]] = {}
    for start in starts:
        by_stratum.setdefault(start["stratum"], []).append(start)
    for group in by_stratum.values():
        group.sort(key=lambda item: (item["distance"], item["cell"], item["facing"]))

    summary = {
        "level": args.level,
        "standable_cells": len(cells),
        "starts": len(starts),
        "distance_p75": hard_threshold,
        "hard_threshold": hard_threshold,
        "medium_range": [args.medium_min, args.medium_max],
        "coverage_source": str(args.coverage_dir),
        "coverage_available": bool(coverage),
        "by_stratum": {},
        "failures": [
            {
                "cell": start["cell"],
                "facing": start["facing"],
                "stratum": start["stratum"],
                "distance": start["distance"],
            }
            for start in starts
            if start.get("coverage_finished") is False
        ],
    }
    for name, group in sorted(by_stratum.items()):
        evaluated = [
            item for item in group if item.get("coverage_finished") is not None
        ]
        finished = [item for item in evaluated if item["coverage_finished"]]
        times = [
            float(item["coverage_time"])
            for item in finished
            if item.get("coverage_time") is not None
        ]
        summary["by_stratum"][name] = {
            "starts": len(group),
            "cells": len({tuple(item["cell"]) for item in group}),
            "evaluated": len(evaluated),
            "covered": len(finished),
            "coverage_rate": (
                len(finished) / len(evaluated) if evaluated else None
            ),
            "best_time": min(times) if times else None,
            "median_time": (
                sorted(times)[len(times) // 2] if times else None
            ),
            "worst_time": max(times) if times else None,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "start_strata.json").write_text(
        json.dumps(starts, indent=2), encoding="utf-8"
    )
    (args.output_dir / "strata_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plan = {
        "level": args.level,
        "note": "Exclude easy/near-exit starts; keep canonical as an anchor.",
        "weights": {"canonical": 0.15, "medium": 0.45, "hard": 0.40, "easy": 0.0},
        "canonical_starts": [
            item for item in starts if item["stratum"] == "canonical"
        ],
        "medium_starts": [
            item for item in starts if item["stratum"] == "medium"
        ],
        "hard_starts": [
            item for item in starts if item["stratum"] == "hard"
        ],
        "easy_starts_excluded": [
            item for item in starts if item["stratum"] == "easy"
        ],
        "focus_starts": [
            item
            for item in starts
            if item.get("coverage_finished") is False
        ],
    }
    (args.output_dir / "sampling_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["by_stratum"], indent=2), flush=True)
    print(f"hard threshold: {hard_threshold}", flush=True)
    print(f"failures: {len(summary['failures'])}", flush=True)
    print(f"wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
