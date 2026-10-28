#!/usr/bin/env python3
"""Write code/game provenance metadata for an experiment run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plaiground.experiments import load_experiment
from plaiground.experiments.provenance import write_run_metadata

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--command", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment(args.experiment)
    game_repo = (REPO / config["game"]["repo"]).resolve()
    path = write_run_metadata(
        args.run_dir,
        experiment=config,
        game_repo=game_repo,
        command=args.command or " ".join(sys.argv),
    )
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
