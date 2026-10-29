#!/usr/bin/env python3
"""Diagnose deterministic DQN rollouts with Q-value and action telemetry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plaiground.data.pools import load_start_pool
from plaiground.envs.factory import make_eval_env
from plaiground.envs.runtime import resolve_env_path
from plaiground.envs.single_agent import RESET_STATE_KEYS
from plaiground.evaluation.rollout import load_model, load_preprocessor
from plaiground.experiments.runner import load_run, resolve_path


def _spawn_for(entry: dict[str, Any]) -> dict[str, Any] | None:
    if entry.get("canonical"):
        return None
    spawn = {
        "agent": 0,
        "cell": [int(entry["cell"][0]), int(entry["cell"][1])],
        "facing": 1 if int(entry.get("facing", 1)) >= 0 else -1,
    }
    for key in RESET_STATE_KEYS:
        if key in entry:
            spawn[key] = entry[key]
    return spawn


def _q_values(model: Any, policy_obs: np.ndarray) -> np.ndarray:
    q_net = getattr(model, "q_net", None)
    if q_net is None:
        q_net = model.policy.q_net
    with torch.no_grad():
        tensor = torch.as_tensor(policy_obs, device=model.device)
        return q_net(tensor).detach().cpu().numpy()[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--start-pool", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--tag", default="diagnostic")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run = load_run(
        args.config,
        show=False,
        device=args.device,
        verbose=0,
    )
    run.ensure_dirs()
    model_path = resolve_path(args.model)
    model = load_model(run.algorithm_name, model_path, run.device)
    preprocessor = load_preprocessor(run)

    if args.start_pool is not None:
        starts = load_start_pool(args.start_pool)
        if not 0 <= args.start_index < len(starts):
            raise ValueError(
                f"start index {args.start_index} out of range for "
                f"{len(starts)} starts"
            )
        start = starts[args.start_index]
    else:
        start = {"canonical": True}

    out_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else run.eval_dir / "diagnostics" / args.tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_eval_env(
        run,
        resolve_env_path(),
        level=run.level,
        num_agents=1,
        port=run.eval_port,
        log_name=f"diagnose-{args.tag}.log",
        home_name=f"diagnose_{args.tag}",
        show_window=False,
    )
    spawn = _spawn_for(start)
    rows: list[dict[str, Any]] = []
    try:
        obs, infos = env.reset(spawns=[spawn]) if spawn else env.reset()
        info = infos[0] if infos else {}
        spawn_ok = True
        if spawn is not None:
            actual = info.get("spawn_cell")
            expected = spawn["cell"]
            spawn_ok = bool(
                isinstance(actual, (list, tuple))
                and len(actual) == 2
                and [int(actual[0]), int(actual[1])] == expected
            )
        for step in range(run.max_decisions):
            raw = np.asarray([obs[0]["obs"]], dtype=np.float32)
            policy_obs = (
                np.asarray(preprocessor.transform(raw), dtype=np.float32)
                if preprocessor is not None
                else raw
            )
            q = _q_values(model, policy_obs)
            action = int(np.argmax(q))
            rows.append(
                {
                    "step": step,
                    "action": action,
                    "q0": float(q[0]),
                    "q1": float(q[1]),
                    "q_margin": float(abs(q[0] - q[1])),
                    "run_time": float(info.get("run_time", 0.0)),
                    "global_x": (
                        float(info["global_position"][0])
                        if isinstance(info.get("global_position"), (list, tuple))
                        else ""
                    ),
                    "global_y": (
                        float(info["global_position"][1])
                        if isinstance(info.get("global_position"), (list, tuple))
                        else ""
                    ),
                }
            )
            obs, rewards, dones, _timeouts, infos = env.step(
                np.asarray([[action]], dtype=np.int64)
            )
            info = infos[0] if infos else {}
            if bool(dones[0]):
                break
    finally:
        env.close()

    actions = [row["action"] for row in rows]
    flips = sum(
        1 for index in range(1, len(actions)) if actions[index] != actions[index - 1]
    )
    margins = [row["q_margin"] for row in rows]
    summary = {
        "tag": args.tag,
        "model": str(model_path),
        "config": str(run.config_path),
        "level": run.level,
        "start": start,
        "spawn_ok": spawn_ok,
        "finished": bool(info.get("finished", False)),
        "timed_out": bool(info.get("timed_out", False)),
        "run_time": float(info.get("run_time", 0.0)),
        "decisions": len(rows),
        "action_flips": flips,
        "action_flip_rate": flips / max(1, len(rows) - 1),
        "mean_q_margin": float(np.mean(margins)) if margins else 0.0,
        "min_q_margin": float(np.min(margins)) if margins else 0.0,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (out_dir / "actions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["step"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["finished"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
