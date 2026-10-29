"""Algorithm-agnostic deterministic evaluation for canonical and pool starts.

Both PPO and DQN expose ``model.predict(observations, deterministic=True)``.
That is the only policy interface this module needs, so canonical evaluation,
start-pool evaluation, and the CSV/JSON summaries live in one place.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from plaiground.data.pools import load_start_pool
from plaiground.envs.factory import make_eval_env
from plaiground.envs.godot import GodotEnv
from plaiground.envs.runtime import resolve_env_path
from plaiground.envs.single_agent import RESET_STATE_KEYS
from plaiground.experiments.runner import Run, resolve_path
from plaiground.observations.lean_io import load_lean_preprocessor


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def policy_observations(
    preprocessor: Any | None,
    observations: list[dict],
) -> np.ndarray:
    raw = np.asarray(
        [item["obs"] for item in observations], dtype=np.float32
    )
    if preprocessor is None:
        return raw
    return np.asarray(preprocessor.transform(raw), dtype=np.float32)


def predict_actions(model: Any, observations: np.ndarray) -> np.ndarray:
    actions, _ = model.predict(observations, deterministic=True)
    return np.asarray(actions, dtype=np.int64).reshape(-1)


def _validate_starts(starts: list[dict]) -> None:
    if not starts:
        raise ValueError("evaluation needs at least one start")
    for index, entry in enumerate(starts):
        if entry.get("canonical"):
            continue
        cell = entry.get("cell")
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            raise ValueError(
                "start-pool evaluation supports canonical or cell/state "
                f"starts; start[{index}] has no cell (prefix starts are not "
                "supported by the evaluator)"
            )


def _spawn_for(entry: dict, agent: int) -> dict | None:
    if entry.get("canonical"):
        return None
    spawn = {
        "agent": int(agent),
        "cell": [int(entry["cell"][0]), int(entry["cell"][1])],
        "facing": 1 if int(entry.get("facing", 1)) >= 0 else -1,
    }
    for key in RESET_STATE_KEYS:
        if key in entry:
            spawn[key] = entry[key]
    return spawn


def _finish_time(info: dict, steps: int, action_repeat: int) -> float:
    value = finite_float(info.get("run_time"))
    if value is None or value <= 0.0:
        return steps * action_repeat / 60.0
    return value


def _start_fields(entry: dict) -> dict[str, Any]:
    return {
        "start_kind": entry.get("kind")
        or entry.get("stratum")
        or ("canonical" if entry.get("canonical") else "state"),
        "source_route": entry.get("source_route", ""),
        "source_time": entry.get("source_time", ""),
        "run_time_at_state": entry.get("run_time_at_state", ""),
        "start_cell": entry.get("cell", ""),
        "start_facing": entry.get("facing", ""),
    }


def run_starts(
    model: Any,
    env: GodotEnv,
    starts: list[dict],
    *,
    preprocessor: Any | None,
    max_decisions: int,
    action_repeat: int,
    force_start_jump: bool = False,
) -> list[dict[str, Any]]:
    """Run one batched deterministic rollout and return one row per start."""
    _validate_starts(starts)
    if force_start_jump and any(not entry.get("canonical") for entry in starts):
        raise ValueError("force_start_jump is only valid for canonical starts")

    count = len(starts)
    spawns = [
        spawn
        for index, entry in enumerate(starts)
        if (spawn := _spawn_for(entry, index)) is not None
    ]
    obs, infos = env.reset(spawns=spawns) if spawns else env.reset()
    policy_obs = policy_observations(preprocessor, obs)

    done = np.zeros(count, dtype=bool)
    finished = np.zeros(count, dtype=bool)
    finish_times = np.full(count, np.nan, dtype=np.float64)
    rewards = np.zeros(count, dtype=np.float64)
    steps_run = np.zeros(count, dtype=np.int32)
    deaths = np.zeros(count, dtype=np.int32)
    spawn_ok = np.ones(count, dtype=bool)
    last_infos: list[dict] = [{} for _ in range(count)]

    for index, info in enumerate(infos[:count]):
        last_infos[index] = dict(info)
        expected = starts[index].get("cell")
        if expected is None:
            continue
        actual = info.get("spawn_cell")
        spawn_ok[index] = bool(
            isinstance(actual, (list, tuple))
            and len(actual) == 2
            and [int(actual[0]), int(actual[1])] == [
                int(expected[0]),
                int(expected[1]),
            ]
        )

    offset = 1 if force_start_jump else 0
    if force_start_jump:
        obs, step_rewards, step_dones, _timeouts, step_infos = env.step(
            np.ones((count, 1), dtype=np.int64)
        )
        policy_obs = policy_observations(preprocessor, obs)
        steps_run[:] = 1
        for index in range(count):
            info = dict(step_infos[index]) if index < len(step_infos) else {}
            last_infos[index] = info
            rewards[index] += float(step_rewards[index])
            if bool(step_dones[index]):
                done[index] = True
                finished[index] = bool(
                    info.get("finished", step_rewards[index] > 0.0)
                )
                if finished[index]:
                    finish_times[index] = _finish_time(
                        info, 1, action_repeat
                    )

    for step in range(max_decisions):
        if done.all():
            break
        active = np.flatnonzero(~done)
        actions = np.zeros((count, 1), dtype=np.int64)
        if active.size:
            actions[active, 0] = predict_actions(
                model, policy_obs[active]
            )
        obs, step_rewards, step_dones, _timeouts, step_infos = env.step(
            actions
        )
        policy_obs = policy_observations(preprocessor, obs)
        for index in active:
            info = dict(step_infos[index]) if index < len(step_infos) else {}
            last_infos[index] = info
            rewards[index] += float(step_rewards[index])
            steps_run[index] = step + 1 + offset
            if bool(step_dones[index]):
                done[index] = True
                finished[index] = bool(
                    info.get("finished", step_rewards[index] > 0.0)
                )
                if finished[index]:
                    finish_times[index] = _finish_time(
                        info, int(steps_run[index]), action_repeat
                    )
            elif "deaths" in info:
                try:
                    deaths[index] = int(info["deaths"])
                except (TypeError, ValueError):
                    pass

    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(starts):
        info = last_infos[index]
        if "deaths" in info:
            try:
                deaths[index] = int(info["deaths"])
            except (TypeError, ValueError):
                pass
        rows.append(
            {
                "agent": index,
                **_start_fields(entry),
                "spawn_ok": bool(spawn_ok[index]),
                "finished": bool(finished[index]),
                "timed_out": bool(
                    not finished[index]
                    and steps_run[index] >= max_decisions + offset
                ),
                "steps": int(steps_run[index]),
                "reward": float(rewards[index]),
                "run_time": (
                    float(finish_times[index])
                    if np.isfinite(finish_times[index])
                    else None
                ),
                "efficiency": finite_float(info.get("efficiency")),
                "flow": finite_float(info.get("flow")),
                "deaths": int(deaths[index]),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finish_times = [
        float(row["run_time"])
        for row in rows
        if row["finished"] and row["run_time"] is not None
    ]
    mean_time = float(np.mean(finish_times)) if finish_times else None
    efficiency = [
        float(row["efficiency"])
        for row in rows
        if row.get("efficiency") is not None
    ]
    flow = [
        float(row["flow"])
        for row in rows
        if row.get("flow") is not None
    ]
    return {
        "agents": len(rows),
        "finishes": sum(1 for row in rows if row["finished"]),
        "deaths": sum(int(row["deaths"]) for row in rows),
        "timeouts": sum(1 for row in rows if not row["finished"]),
        "best_time": min(finish_times) if finish_times else None,
        "median_time": (
            float(np.median(finish_times)) if finish_times else None
        ),
        "mean_time": mean_time,
        "time_cv": (
            float(np.std(finish_times) / mean_time)
            if finish_times and mean_time
            else None
        ),
        "efficiency_mean": float(np.mean(efficiency)) if efficiency else None,
        "flow_mean": float(np.mean(flow)) if flow else None,
        "mean_reward": (
            float(np.mean([row["reward"] for row in rows])) if rows else 0.0
        ),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "agent",
        "start_kind",
        "source_route",
        "source_time",
        "run_time_at_state",
        "start_cell",
        "start_facing",
        "spawn_ok",
        "finished",
        "timed_out",
        "steps",
        "run_time",
        "reward",
        "deaths",
        "efficiency",
        "flow",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        "" if row[key] is None else row[key]
                    )
                    for key in fieldnames
                }
            )


def load_model(algorithm: str, path: Path, device: str):
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(path), device=device)
    if algorithm == "dqn":
        from stable_baselines3 import DQN

        return DQN.load(str(path), device=device)
    raise ValueError(f"unknown algorithm {algorithm!r}")


def load_preprocessor(run: Run) -> Any | None:
    if run.obs_encoding != "lean" and run.preprocessing != "lean":
        return None
    if not run.preprocessing_path.exists():
        raise FileNotFoundError(
            f"lean evaluation needs the saved pipeline at "
            f"{run.preprocessing_path}"
        )
    return load_lean_preprocessor(run.preprocessing_path)


def _safe_tag(tag: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in tag
    )
    return cleaned.strip("._") or "eval"


def evaluate_canonical(
    model: Any,
    run: Run,
    env_path: str,
    preprocessor: Any | None,
    tag: str,
    *,
    level: str,
) -> dict[str, Any]:
    env = make_eval_env(
        run,
        env_path,
        level=level,
        num_agents=1,
        port=run.eval_port,
        log_name=f"godot-{tag}-{level}-canonical.log",
        home_name=f"eval_{tag}_{level}_canonical",
    )
    try:
        rows = run_starts(
            model,
            env,
            [{"canonical": True}],
            preprocessor=preprocessor,
            max_decisions=run.max_decisions,
            action_repeat=run.action_repeat,
            force_start_jump=bool(run.env.get("force_start_jump", False)),
        )
    finally:
        env.close()
    summary = summarize(rows)
    summary.update({"tag": "canonical", "level": level})
    write_rows(
        run.eval_dir / tag / f"canonical-{level}.csv", rows
    )
    print(
        f"[eval {level} canonical] finishes={summary['finishes']}/"
        f"{summary['agents']} best={summary['best_time'] or 'none'} "
        f"median={summary['median_time'] or 'none'} "
        f"deaths={summary['deaths']} timeouts={summary['timeouts']}",
        flush=True,
    )
    return summary


def evaluate_set(
    model: Any,
    run: Run,
    env_path: str,
    preprocessor: Any | None,
    name: str,
    path: Path,
    tag: str,
) -> dict[str, Any]:
    starts = load_start_pool(path)
    pool_levels = {
        str(entry.get("level"))
        for entry in starts
        if entry.get("level") is not None
    }
    if pool_levels and run.level not in pool_levels:
        raise ValueError(
            f"{path} is for level(s) {sorted(pool_levels)}, "
            f"not {run.level}"
        )
    env = make_eval_env(
        run,
        env_path,
        level=run.level,
        num_agents=len(starts),
        port=run.eval_port + 1,
        log_name=f"godot-{tag}-{name}.log",
        home_name=f"eval_{tag}_{name}",
    )
    try:
        rows = run_starts(
            model,
            env,
            starts,
            preprocessor=preprocessor,
            max_decisions=run.max_decisions,
            action_repeat=run.action_repeat,
            force_start_jump=False,
        )
    finally:
        env.close()
    summary = summarize(rows)
    summary.update(
        {
            "tag": name,
            "path": str(path),
            "spawn_ok": sum(1 for row in rows if row["spawn_ok"]),
        }
    )
    write_rows(run.eval_dir / tag / f"{name}.csv", rows)
    print(
        f"[eval {name}] finishes={summary['finishes']}/{summary['agents']} "
        f"spawn_ok={summary['spawn_ok']}/{summary['agents']} "
        f"median={summary['median_time'] or 'none'} "
        f"best={summary['best_time'] or 'none'}",
        flush=True,
    )
    return summary


def evaluate_run(
    run: Run,
    model_path: Path,
    tag: str | None = None,
) -> dict[str, Any]:
    tag = _safe_tag(tag or model_path.stem)
    run.eval_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(run.algorithm_name, model_path, run.device)
    preprocessor = load_preprocessor(run)
    env_path = resolve_env_path()

    summary: dict[str, Any] = {
        "tag": tag,
        "model": str(model_path),
        "config": str(run.config_path),
        "canonical": {},
        "sets": {},
    }
    if bool(run.evaluation.get("canonical", True)):
        for level in run.levels:
            summary["canonical"][level] = evaluate_canonical(
                model, run, env_path, preprocessor, tag, level=level
            )

    set_paths = list(run.eval_set_paths)
    if not set_paths:
        if bool(run.evaluation.get("train_pool", False)):
            value = run.data.get("train_pool")
            if value:
                set_paths.append(resolve_path(value))
        if bool(run.evaluation.get("holdout_pool", False)):
            value = run.data.get("holdout_pool")
            if value:
                set_paths.append(resolve_path(value))
    if set_paths and len(run.levels) > 1:
        raise ValueError(
            "start-pool evaluation currently supports one level per run; "
            "set [env] level to a single level"
        )
    for path in set_paths:
        summary["sets"][path.stem] = evaluate_set(
            model, run, env_path, preprocessor, path.stem, path, tag
        )

    output = run.eval_dir / tag / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"eval summary: {output}", flush=True)
    return summary
