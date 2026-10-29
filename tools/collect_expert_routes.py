#!/usr/bin/env python3
"""Replay successful NEAT routes in the current observation environment.

The collector is intentionally independent of the DQN replay buffer: it saves
successful ``(obs, action, reward, next_obs, done)`` transitions so the same
expert data can be used by behavior cloning and by expert replay.

Example:
    .venv/bin/python tools/collect_expert_routes.py \
      --config experiments/0073_bc_ddqn_1_4_lean_stick \
      --limit 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plaiground.envs.godot import GodotEnv
from plaiground.envs.runtime import isolated_home, resolve_env_path
from plaiground.experiments.runner import load_run, resolve_path
from plaiground.observations.lean_io import load_lean_preprocessor
from plaiground.observations.lean_stick import LeanStickPreprocessor


def _load_actions(path: Path) -> dict[str, dict]:
    """Return the shortest finisher action sequence per route signature."""
    best: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
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
                    "clean": bool(record.get("clean", False)),
                    "generation": record.get("generation"),
                    "genome_id": record.get("genome_id"),
                }
    return best


def _flag_stats(obs: np.ndarray) -> dict:
    flags = np.asarray(obs)[:, -3:]
    names = ("on_ceiling", "jump_held", "stick_active")
    stats = {}
    for index, name in enumerate(names):
        values, counts = np.unique(flags[:, index], return_counts=True)
        stats[name] = {
            "mean": float(flags[:, index].mean()),
            "counts": {
                float(value): int(count)
                for value, count in zip(values, counts)
            },
        }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--routes", type=Path, default=None)
    parser.add_argument("--finishers", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run = load_run(args.config, show=args.show)
    run.ensure_dirs()
    data = run.data

    routes_value = args.routes or data.get("expert_routes")
    if routes_value is None:
        raise SystemExit("missing [data] expert_routes")
    routes_path = resolve_path(routes_value)
    routes = json.loads(routes_path.read_text(encoding="utf-8"))
    if not isinstance(routes, list):
        raise SystemExit(
            f"{routes_path} must be a list of clean routes; "
            "use filtered/train_clean_routes.json, not routes.json"
        )
    if args.limit is not None:
        routes = routes[: max(0, int(args.limit))]

    finishers_value = args.finishers or data.get("finishers")
    if finishers_value is None:
        raise SystemExit("missing [data] finishers")
    finishers_path = resolve_path(finishers_value)
    actions_by_signature = _load_actions(finishers_path)

    output_value = args.output or data.get("expert_data")
    if output_value is None:
        raise SystemExit("missing [data] expert_data")
    output_path = resolve_path(output_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = load_lean_preprocessor(run.preprocessing_path)
    if not isinstance(preprocessor, LeanStickPreprocessor):
        preprocessor = LeanStickPreprocessor(preprocessor)

    env_path = resolve_env_path()
    GodotEnv.DEFAULT_TIMEOUT = run.timeout
    log_path = run.output_dir / "godot" / "expert_replay.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = GodotEnv(
        env_path=env_path,
        port=run.base_port,
        show_window=args.show,
        seed=run.seed,
        action_repeat=run.action_repeat,
        speedup=run.speedup,
        nb_agents=1,
        level=run.level,
        convert_action_space=True,
        log_path=str(log_path),
        env=isolated_home(run.output_dir, "expert_replay"),
        reward_profile=str(run.env.get("reward_profile", "default")),
        obs_encoding=run.obs_encoding,
        reassert_held_jump="true",
    )
    advertised = str(env.static_info.get("obs_encoding", ""))
    advertised_size = int(env.static_info.get("obs_size", 0))
    if advertised != run.obs_encoding or advertised_size != 191:
        env.close()
        raise SystemExit(
            f"game advertised {advertised!r}/{advertised_size}; "
            f"expected {run.obs_encoding!r}/191"
        )
    print(
        f"expert replay: level={run.level} encoding={run.obs_encoding} "
        f"routes={len(routes)} output={output_path}",
        flush=True,
    )

    obs_raw: list[np.ndarray] = []
    obs_enc: list[np.ndarray] = []
    next_raw: list[np.ndarray] = []
    next_enc: list[np.ndarray] = []
    actions_all: list[int] = []
    rewards_all: list[float] = []
    dones_all: list[float] = []
    signatures: list[str] = []
    steps_all: list[int] = []
    levels: list[str] = []
    route_times: list[float] = []
    replay_times: list[float] = []
    finished_all: list[bool] = []
    route_success_all: list[bool] = []
    route_log: list[dict] = []

    try:
        for route_index, route in enumerate(routes):
            signature = str(route.get("route_signature", ""))
            finisher = actions_by_signature.get(signature)
            if finisher is None:
                print(
                    f"[routereplay] {route_index + 1}/{len(routes)} "
                    f"{signature[:12]} no finisher actions; skipped",
                    flush=True,
                )
                route_log.append(
                    {
                        "route_signature": signature,
                        "finished": False,
                        "error": "missing finisher actions",
                    }
                )
                continue
            action_sequence = finisher["actions"]
            route_start = len(actions_all)
            obs, infos = env.reset()
            info = infos[0] if infos else {}
            finished = False
            replay_time = None
            steps = 0
            for step, action in enumerate(action_sequence):
                raw = np.asarray(obs[0]["obs"], dtype=np.float32)
                if raw.shape != (191,):
                    raise RuntimeError(
                        f"expected 191-value observation, got {raw.shape}"
                    )
                encoded = np.asarray(
                    preprocessor.transform(raw), dtype=np.float32
                )
                (
                    next_obs,
                    rewards,
                    dones,
                    _timeouts,
                    infos,
                ) = env.step(
                    np.asarray([[int(action)]], dtype=np.int64)
                )
                next_value = np.asarray(
                    next_obs[0]["obs"], dtype=np.float32
                )
                next_encoded = np.asarray(
                    preprocessor.transform(next_value), dtype=np.float32
                )
                reward = float(rewards[0])
                done = bool(dones[0])
                info = infos[0] if infos else {}

                obs_raw.append(raw)
                obs_enc.append(encoded)
                next_raw.append(next_value)
                next_enc.append(next_encoded)
                actions_all.append(int(action))
                rewards_all.append(reward)
                dones_all.append(float(done))
                signatures.append(signature)
                steps_all.append(step)
                levels.append(run.level)
                route_times.append(float(route.get("run_time", np.nan)))
                finished_all.append(bool(info.get("finished", False)))
                replay_times.append(
                    float(info.get("run_time", np.nan))
                )
                steps += 1
                obs = next_obs
                if done:
                    finished = bool(info.get("finished", False))
                    replay_time = info.get("run_time")
                    break

            for _ in range(len(actions_all) - route_start):
                route_success_all.append(bool(finished))
            route_log.append(
                {
                    "route_signature": signature,
                    "source_time": route.get("run_time"),
                    "finisher_time": finisher["run_time"],
                    "actions": len(action_sequence),
                    "replayed_steps": steps,
                    "finished": finished,
                    "replayed_run_time": replay_time,
                }
            )
            print(
                f"[routereplay] {route_index + 1}/{len(routes)} "
                f"{signature[:12]} steps={steps}/{len(action_sequence)} "
                f"finished={finished} time={replay_time} "
                f"source={route.get('run_time')}",
                flush=True,
            )
    finally:
        env.close()

    if not obs_raw:
        raise SystemExit("no expert transitions collected")

    payload = {
        "obs_raw": np.asarray(obs_raw, dtype=np.float32),
        "obs": np.asarray(obs_enc, dtype=np.float32),
        "next_obs_raw": np.asarray(next_raw, dtype=np.float32),
        "next_obs": np.asarray(next_enc, dtype=np.float32),
        "actions": np.asarray(actions_all, dtype=np.int64),
        "rewards": np.asarray(rewards_all, dtype=np.float32),
        "dones": np.asarray(dones_all, dtype=np.float32),
        "route_signature": np.asarray(signatures, dtype="U64"),
        "step": np.asarray(steps_all, dtype=np.int32),
        "level": np.asarray(levels, dtype="U8"),
        "route_run_time": np.asarray(route_times, dtype=np.float32),
        "replayed_run_time": np.asarray(replay_times, dtype=np.float32),
        "finished": np.asarray(finished_all, dtype=bool),
        "route_success": np.asarray(route_success_all, dtype=bool),
    }
    if payload["route_success"].shape[0] != payload["actions"].shape[0]:
        raise RuntimeError(
            "route_success length does not match expert transitions"
        )
    np.savez_compressed(output_path, **payload)

    metadata = {
        "config": str(run.config_path),
        "routes": str(routes_path),
        "finishers": str(finishers_path),
        "output": str(output_path),
        "level": run.level,
        "routes_requested": len(routes),
        "routes_finished": sum(1 for item in route_log if item["finished"]),
        "transitions": len(obs_raw),
        "obs_raw_size": 191,
        "obs_size": int(payload["obs"].shape[1]),
        "actions_zero": int(np.sum(payload["actions"] == 0)),
        "actions_one": int(np.sum(payload["actions"] == 1)),
        "flag_stats": _flag_stats(payload["obs_raw"]),
        "route_log": route_log,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"expert data={output_path} "
        f"transitions={metadata['transitions']} "
        f"routes={metadata['routes_finished']}/{metadata['routes_requested']} "
        f"metadata={metadata_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
