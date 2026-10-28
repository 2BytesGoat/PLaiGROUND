#!/usr/bin/env python3
"""Build a weighted PPO start pool from level strata and clean NEAT routes.

Reset mixture defaults:
  canonical 15%
  medium cell starts 30%
  hard cell starts 30%
  clean-route prefix states 25%
  easy/near-exit starts 0%

Optional focus cells remove all of that cell's starts from medium/hard, add one
randomly chosen facing to training, and leave the mirror facing out as a
held-out symmetry test. Prefix entries replay a recorded NEAT action prefix
from the canonical start, then hand control to PPO. No full state snapshot is
required.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "runs" / "neat_1_1_8x24_archive"
DEFAULT_STRATA = BASE / "strata" / "start_strata.json"
DEFAULT_CLEAN = BASE / "clean" / "clean_routes.json"
DEFAULT_OUTPUT = BASE / "ppo" / "start_pool.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="1-1")
    parser.add_argument("--strata", type=Path, default=DEFAULT_STRATA)
    parser.add_argument("--clean-routes", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--canonical-weight", type=float, default=0.15)
    parser.add_argument("--medium-weight", type=float, default=0.30)
    parser.add_argument("--hard-weight", type=float, default=0.30)
    parser.add_argument("--prefix-weight", type=float, default=0.25)
    parser.add_argument(
        "--focus-cells",
        default="",
        help="Semicolon-separated x,y cells to split into one train facing + one holdout facing.",
    )
    parser.add_argument("--focus-weight", type=float, default=0.0)
    parser.add_argument("--focus-seed", type=int, default=7)
    args = parser.parse_args()

    starts = json.loads(args.strata.read_text(encoding="utf-8"))
    by_stratum: dict[str, list[dict]] = {}
    for start in starts:
        by_stratum.setdefault(str(start["stratum"]), []).append(start)
    medium = by_stratum.get("medium", [])
    hard = by_stratum.get("hard", [])
    canonical = by_stratum.get("canonical", [])
    if not canonical or not medium or not hard:
        raise RuntimeError("strata file is missing canonical/medium/hard starts")

    focus_cells: list[tuple[int, int]] = []
    if args.focus_cells:
        for entry in args.focus_cells.split(";"):
            if not entry.strip():
                continue
            x_text, y_text = entry.split(",", 1)
            cell = (int(x_text), int(y_text))
            if cell in focus_cells:
                raise ValueError(f"duplicate focus cell {cell}")
            focus_cells.append(cell)
    if focus_cells and args.focus_weight <= 0.0:
        raise ValueError("--focus-weight must be positive when --focus-cells is set")

    focus_set = set(focus_cells)
    focus_starts: list[dict] = []
    heldout_starts: list[dict] = []
    focus_rng = random.Random(args.focus_seed)
    for cell in focus_cells:
        candidates = [
            start
            for start in starts
            if tuple(start["cell"]) == cell
            and start.get("stratum") in ("medium", "hard")
        ]
        if not candidates:
            raise ValueError(f"focus cell {cell} has no medium/hard start")
        chosen = focus_rng.choice(sorted(candidates, key=lambda item: item["facing"]))
        focus_starts.append(
            {
                "kind": "focus",
                "level": args.level,
                "stratum": "focus",
                "cell": list(cell),
                "facing": int(chosen["facing"]),
                "distance": int(chosen["distance"]),
                "weight": args.focus_weight / len(focus_cells),
            }
        )
        heldout_starts.extend(
            start for start in candidates if start["facing"] != chosen["facing"]
        )

    # The focus cell is represented once in training (random facing) and its
    # mirror remains out of the pool for the held-out symmetry check.
    medium = [
        start for start in medium if tuple(start["cell"]) not in focus_set
    ]
    hard = [start for start in hard if tuple(start["cell"]) not in focus_set]

    pool: list[dict] = []
    pool.append(
        {
            "kind": "canonical",
            "level": args.level,
            "canonical": True,
            "weight": args.canonical_weight,
        }
    )
    for start in focus_starts:
        pool.append(start)
    for start in medium:
        pool.append(
            {
                "kind": "medium",
                "level": args.level,
                "cell": list(start["cell"]),
                "facing": int(start["facing"]),
                "distance": int(start["distance"]),
                "weight": args.medium_weight / len(medium),
            }
        )
    for start in hard:
        pool.append(
            {
                "kind": "hard",
                "level": args.level,
                "cell": list(start["cell"]),
                "facing": int(start["facing"]),
                "distance": int(start["distance"]),
                "weight": args.hard_weight / len(hard),
            }
        )

    clean_routes = json.loads(args.clean_routes.read_text(encoding="utf-8"))
    clean_routes = [
        route
        for route in clean_routes
        if isinstance(route.get("actions"), list) and len(route["actions"]) >= 8
    ]
    if not clean_routes:
        raise RuntimeError("clean archive has no replayable routes")
    rng = random.Random(args.seed)
    prefixes = []
    for _ in range(args.prefix_count):
        route = rng.choice(clean_routes)
        actions = [int(action) for action in route["actions"]]
        max_prefix = max(5, int(len(actions) * 0.8) - 1)
        prefix_len = rng.randint(5, min(max_prefix, len(actions) - 2))
        prefixes.append(
            {
                "kind": "prefix",
                "level": args.level,
                "prefix_actions": actions[:prefix_len],
                "prefix_len": prefix_len,
                "source_route": route.get("route_signature"),
                "source_time": route.get("run_time"),
                "weight": args.prefix_weight / args.prefix_count,
            }
        )
    pool.extend(prefixes)

    total = sum(float(item["weight"]) for item in pool)
    if abs(total - 1.0) > 1e-6:
        for item in pool:
            item["weight"] = float(item["weight"]) / total

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool, indent=2), encoding="utf-8")
    summary = {
        "level": args.level,
        "starts": len(pool),
        "canonical": 1,
        "medium": len(medium),
        "hard": len(hard),
        "focus": len(focus_starts),
        "prefix": len(prefixes),
        "weights": {
            "canonical": args.canonical_weight,
            "medium": args.medium_weight,
            "hard": args.hard_weight,
            "focus": args.focus_weight,
            "prefix": args.prefix_weight,
        },
        "seed": args.seed,
        "focus_starts": focus_starts,
        "heldout_starts": heldout_starts,
    }
    (args.output.parent / "start_pool_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    chosen_focus_keys = {
        (tuple(item["cell"]), int(item["facing"])) for item in focus_starts
    }
    heldout_eval = [
        start
        for start in starts
        if start.get("stratum") in ("canonical", "medium", "hard")
        and (tuple(start["cell"]), int(start["facing"])) not in chosen_focus_keys
    ]
    (args.output.parent / "eval_starts_heldout.json").write_text(
        json.dumps(heldout_eval, indent=2), encoding="utf-8"
    )
    (args.output.parent / "focus_starts.json").write_text(
        json.dumps(focus_starts, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
