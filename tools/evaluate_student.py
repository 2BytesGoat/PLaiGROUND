#!/usr/bin/env python3
"""Evaluate a distilled PyTorch student on canonical and held-out starts.

The model wrapper exposes the same ``predict(observations, deterministic=True)``
interface as the SB3 policies, so the shared rollout evaluator is reused.

Examples:
    .venv/bin/python tools/evaluate_student.py \
      --checkpoint runs/distill_campaign_student/students/raw191_512/student.pt \
      --level 1-6 --canonical \
      --holdout-pool runs/neat_speed_1_6_from_0090/pool_state_70/holdout_start_pool.json \
      --output runs/distill_campaign_student/eval/raw191_512/1-6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from plaiground.envs.godot import GodotEnv
from plaiground.envs.runtime import isolated_home, resolve_env_path
from plaiground.envs.single_agent import RESET_STATE_KEYS
from plaiground.evaluation.rollout import run_starts, summarize
from plaiground.models.student import GridStudent, StudentMLP, TorchPolicy
from plaiground.observations.lean_io import load_lean_preprocessor
from plaiground.observations.lean_stick import LeanStickPreprocessor


def load_start_pool(path: Path, level: str, split: str | None = None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("starts", data) if isinstance(data, dict) else data
    result = []
    seen: set[tuple] = set()
    for entry in entries:
        if entry.get("level") not in (None, level):
            continue
        item = dict(entry)
        item.setdefault("split", "holdout" if "holdout" in path.name else "train")
        if split is not None and item.get("split") != split:
            continue
        if item.get("canonical"):
            key = ("canonical",)
        else:
            key = (
                tuple(int(v) for v in item.get("cell", [])),
                int(item.get("facing", 1)),
                round(float(item.get("run_time_at_state", item.get("run_time", 0.0))), 6),
                str(item.get("source_route", "")),
            )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def spawns_for(starts: list[dict]) -> list[dict]:
    spawns = []
    for index, entry in enumerate(starts):
        if entry.get("canonical"):
            continue
        spawn = {
            "agent": index,
            "cell": [int(entry["cell"][0]), int(entry["cell"][1])],
            "facing": 1 if int(entry.get("facing", 1)) >= 0 else -1,
        }
        for key in RESET_STATE_KEYS:
            if key in entry:
                spawn[key] = entry[key]
        spawns.append(spawn)
    return spawns


def make_env(
    level: str,
    num_agents: int,
    port: int,
    output_dir: Path,
    *,
    action_repeat: int,
    speedup: float,
    timeout: int,
    show: bool,
    reassert_held_jump: bool,
) -> GodotEnv:
    GodotEnv.DEFAULT_TIMEOUT = timeout
    return GodotEnv(
        env_path=resolve_env_path(),
        port=port,
        show_window=show,
        seed=42,
        action_repeat=action_repeat,
        speedup=speedup,
        nb_agents=num_agents,
        level=level,
        convert_action_space=True,
        log_path=str(output_dir / "godot.log"),
        env=isolated_home(output_dir, "home"),
        reward_profile="default",
        obs_encoding="lean_stick",
        reassert_held_jump="true" if reassert_held_jump else "false",
    )


def evaluate(
    *,
    checkpoint_path: Path,
    level: str,
    starts: list[dict],
    port: int,
    output_dir: Path,
    action_repeat: int,
    max_decisions: int,
    speedup: float,
    timeout: int,
    show: bool,
    reassert_held_jump: bool,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_kind = str(checkpoint["input_kind"])
    input_dim = int(checkpoint["input_dim"])
    hidden = [int(value) for value in checkpoint["hidden"]]
    if input_kind == "grid191":
        model = GridStudent(input_dim, hidden)
    else:
        model = StudentMLP(input_dim, hidden)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    preprocessor = None
    if input_kind == "onehot957":
        loaded = load_lean_preprocessor(
            Path("runs/ddqn_fresh_1_1_lean_stick/preprocessing.pkl")
        )
        preprocessor = (
            loaded
            if isinstance(loaded, LeanStickPreprocessor)
            else LeanStickPreprocessor(loaded)
        )
    policy = TorchPolicy(model, expected_input_dim=input_dim)

    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(
        level,
        len(starts),
        port,
        output_dir,
        action_repeat=action_repeat,
        speedup=speedup,
        timeout=timeout,
        show=show,
        reassert_held_jump=reassert_held_jump,
    )
    try:
        rows = run_starts(
            policy,
            env,
            starts,
            preprocessor=preprocessor,
            max_decisions=max_decisions,
            action_repeat=action_repeat,
            force_start_jump=False,
        )
    finally:
        env.close()
    summary = summarize(rows)
    return {
        "level": level,
        "checkpoint": str(checkpoint_path),
        "input_kind": input_kind,
        "summary": summary,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--holdout-pool", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=32400)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--speedup", type=float, default=12.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-reassert-held-jump", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    starts: list[dict] = []
    if args.canonical:
        starts.append({"canonical": True, "kind": "canonical", "split": "canonical"})
    for path in args.holdout_pool:
        starts.extend(load_start_pool(path, args.level, split="holdout"))
    if not starts:
        raise SystemExit("no starts selected; pass --canonical and/or --holdout-pool")
    result = evaluate(
        checkpoint_path=args.checkpoint,
        level=args.level,
        starts=starts,
        port=args.port,
        output_dir=args.output.parent / args.level,
        action_repeat=args.action_repeat,
        max_decisions=args.max_decisions,
        speedup=args.speedup,
        timeout=args.timeout,
        show=args.show,
        reassert_held_jump=not args.no_reassert_held_jump,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
