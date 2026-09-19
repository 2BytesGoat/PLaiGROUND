"""
stream_server.py
Run inside the lean PLaiGROUND container. Drives the Godot game via the
existing env wrapper and writes one JSON frame per line to STDOUT so ML
Forge's embedded Live Visualizer can render it in real time on the host.

Deliberately torch-free: uses wrappers.godot_env.GodotEnv directly instead
of the stable-baselines wrapper (whose VecEnv base class pulls torch).

Frame format (one JSON object per line on stdout):
    {"obs": [...], "action": int, "reward": [...], "done": [bool], "step": int}

Protocol:
    - logs go to stderr (stdout stays frame-pure)
    - the server streams until the client sends "STOP\\n" on stdin, stdin
      closes, or --max-steps is reached.

Usage (usually launched by ML Forge's Docker Build node):
    python src/stream_server.py [--fps 30] [--max-steps N]
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from wrappers.godot_env import GodotEnv  # noqa: E402

from ml_forge.game.env import (  # noqa: E402
    read_env_settings,
    resolve_env_dir,
    resolve_godot_project_path,
)


def _log(msg: str) -> None:
    print(f"[stream_server] {msg}", file=sys.stderr, flush=True)


def make_env():
    """Build the Godot env from src/.config without the SB3/torch stack."""
    settings = read_env_settings()

    env_dir = os.getenv("ENV_DIR") or os.getenv("ENV_PATH")
    env_path = None
    if env_dir:
        resolved_env_dir = resolve_env_dir(env_dir, settings["config_path"])
        env_path = os.path.join(resolved_env_dir, settings["env_name"] or "")

    return GodotEnv(
        env_path=env_path,
        show_window=settings["show_window"],
        seed=settings["seed"],
        speedup=settings["speedup"],
        action_repeat=settings["action_repeat"],
        godot_path=settings["godot_path"],
        godot_project_path=resolve_godot_project_path(
            settings["godot_project_path"], settings["config_path"]
        ),
        # Matches the SB3 wrapper's action conversion (see
        # stable_baselines_wrapper.py) so actions are passed the same way.
        convert_action_space=True,
        nb_agents=settings["nb_agents"],
        level=settings["level"],
    )


def _stop_requested() -> bool:
    """Non-blocking stdin check - returns True when STOP or EOF arrives."""
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return False
    line = sys.stdin.readline()
    if not line:
        return True
    return line.strip().upper() == "STOP"


def stream_loop(fps_limit: int, max_steps: int) -> None:
    env = None
    try:
        env = make_env()
        obs_list, _ = env.reset()
        nb_agents = len(obs_list)

        n_heads = len(env.action_spaces[0])
        _log(f"Game running with {nb_agents} agent(s), {n_heads} action head(s).")

        step_count = 0
        while True:
            if _stop_requested():
                _log("Stop requested by visualizer.")
                break

            # Random agent: jump on the very first frame so the run starts.
            value = 1 if step_count == 0 else int(np.random.randint(0, 2))
            observation = obs_list[0]["obs"] if nb_agents else None

            actions = np.full((n_heads, nb_agents), value, dtype=np.int64)
            obs_list, reward, term, trunc, info = env.step(actions)
            done = term

            frame = {
                "obs": np.asarray(observation, dtype=float).tolist(),
                "action": int(value),
                "reward": np.asarray(reward).tolist(),
                "done": [bool(d) for d in done],
                "step": step_count,
            }
            try:
                sys.stdout.write(json.dumps(frame) + "\n")
                sys.stdout.flush()
            except (OSError, BrokenPipeError):
                _log("Visualizer disconnected.")
                break

            step_count += 1
            if 0 < max_steps <= step_count:
                break
            if any(done):
                obs_list, _ = env.reset()
            if fps_limit:
                time.sleep(1.0 / max(fps_limit, 1))

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        _log("Stream server exiting.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream game frames to ML Forge.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    stream_loop(args.fps, args.max_steps)


if __name__ == "__main__":
    main()