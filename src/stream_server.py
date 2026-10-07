"""
stream_server.py
Run inside the lean PLaiGROUND container. Drives the Godot game via the
existing env wrapper and speaks a duplex JSON protocol over stdio so ML
Forge's Playground loop can both watch and drive it:

    stdout: one JSON frame per line (pure - logs go to stderr)
    stdin:  one JSON action per line, or the literal line STOP

Modes (--policy):
    host    - wait for an action line before every env.step; the host
              (ML Forge agent block) is the brain. This closes the loop:
              the container is a pure environment server.
    random  - act randomly and free-run (self-driving demo / watch mode).

Frame format (one JSON object per line on stdout):
    {"obs": [...], "action": int, "reward": [...], "done": [bool], "step": int}

Usage (usually launched by ML Forge's Game block):
    python src/stream_server.py --policy host --fps 30
    python src/stream_server.py --policy random --fps 30
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


def _read_action_blocking(timeout: float) -> int | None:
    """
    Block up to `timeout` seconds for one action line on stdin.
    Returns the action value, or None on STOP/EOF/timeout.
    """
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if not readable:
        return None
    line = sys.stdin.readline()
    if not line:
        return None  # EOF - host closed the pipe
    line = line.strip()
    if line.upper() == "STOP":
        return None
    try:
        cmd = json.loads(line)
        action = cmd.get("action", 0)
        return int(action[0] if isinstance(action, (list, tuple)) else action)
    except (ValueError, TypeError, IndexError):
        return 0


def _emit(frame: dict) -> bool:
    """Write one frame to stdout. Returns False when the pipe is gone."""
    try:
        sys.stdout.write(json.dumps(frame) + "\n")
        sys.stdout.flush()
        return True
    except (OSError, BrokenPipeError):
        _log("Visualizer disconnected.")
        return False


def _run_loop(env, obs_list: dict, nb_agents: int, n_heads: int,
              policy: str, fps_limit: int, max_steps: int) -> None:
    step_count = 0

    def _value(fallback: int) -> int:
        if policy == "host":
            got = _read_action_blocking(60.0)
            return fallback if got is None else got
        return fallback

    while True:
        observation = obs_list[0]["obs"] if nb_agents else None

        value = _value(1 if step_count == 0 else int(np.random.randint(0, 2)))
        if value is None and policy == "host":
            _log("Host closed the stream.")
            break

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
        if not _emit(frame):
            break

        step_count += 1
        if 0 < max_steps <= step_count:
            break
        if any(done):
            obs_list, _ = env.reset()
        if fps_limit and policy != "host":
            time.sleep(1.0 / max(fps_limit, 1))

    _log(f"Stream server exiting after {step_count} steps.")


def stream_loop(fps_limit: int, max_steps: int, policy: str) -> None:
    env = None
    try:
        env = make_env()
        obs_list, _ = env.reset()
        nb_agents = len(obs_list)

        n_heads = len(env.action_spaces[0])
        _log(f"Game running with {nb_agents} agent(s), {n_heads} action head(s), policy={policy}.")

        _run_loop(env, obs_list, nb_agents, n_heads, policy, fps_limit, max_steps)

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
    ap.add_argument("--policy", choices=["random", "host"], default="random",
                    help="host: the host sends actions on stdin (closed loop). "
                         "random: self-driving demo (default).")
    args = ap.parse_args()

    stream_loop(args.fps, args.max_steps, args.policy)


if __name__ == "__main__":
    main()