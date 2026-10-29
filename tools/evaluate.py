#!/usr/bin/env python3
"""Evaluate a saved PPO/DQN checkpoint from an experiment manifest.

Canonical and start-pool evaluation use one persistent Godot process per set.
The manifest's ``[evaluation]`` section controls whether canonical is run and
which pool/set files are evaluated.
"""

from __future__ import annotations

import json

from plaiground.evaluation.rollout import evaluate_run
from plaiground.experiments.runner import (
    load_run,
    parse_eval_args,
    resolve_path,
)


def main() -> int:
    args = parse_eval_args(__doc__)
    run = load_run(
        args.config,
        show=args.show,
        device=args.device,
        verbose=args.verbose,
        level=args.level,
        speedup=args.speedup,
    )
    model_path = (
        resolve_path(args.model) if args.model is not None else run.model_path
    )
    if not model_path.exists():
        raise SystemExit(f"missing model: {model_path}")
    run.ensure_dirs()
    summary = evaluate_run(run, model_path, args.tag)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
