"""PPO pilot for Dragon Jump.

The game's training mode exposes isolated worlds over the godot-rl socket.
This script runs one world per Godot process, wraps each world as a normal
Gymnasium environment, and lets Stable-Baselines3 PPO train one shared policy.

Why one process per world: the Python multi-agent wrapper is designed for
parallel inference/evaluation. It does not auto-reset individual agents, which
is required by SB3's vector-env API. One agent per process gives us the standard
reset/step contract without changing the game's training protocol. The game now
supports optional per-agent reset indices, but this pilot stays 8x1 until a
multi-agent VecEnv exists; 8 is the default, not a measured optimum.

The existing game reward is used unchanged:
progress + efficiency + time + finish + survival. Every episode resets idle;
the policy makes the first jump that starts the run, in training and evaluation.

Usage:
    python src/10_ppo_pilot.py --level 1-3 --n-envs 8 --timesteps 100000
    python src/10_ppo_pilot.py --level 1-3 --n-envs 1 --timesteps 2000
    python src/10_ppo_pilot.py --level 1-3 --eval-only \
        --resume runs/ppo_pilot/ppo_pilot.zip --show

The PPO action repeat is 4 by default: 15 decisions/s, 67 ms. `max_decisions`
counts PPO decisions, so 150 decisions = 600 physics ticks = 10 seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from dotenv import load_dotenv

from wrappers.godot_env import GodotEnv
from wrappers.legacy_autoencoder import FrozenLegacyAutoencoderExtractor
from wrappers.legacy_grid_extractor import LegacyGridExtractor
from wrappers.multi_agent_vec_env import MultiAgentGodotVecEnv
from wrappers.semantic_autoencoder import FrozenSemanticAutoencoderExtractor


DEFAULT_LEVEL = "1-3"
LEVELS = [f"1-{i}" for i in range(1, 17)]
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / "environments"


def tensorboard_available() -> bool:
    """Return whether SB3 can create a TensorBoard output format."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return False
    return True


def parse_levels(value: str) -> list[str]:
    if not value or value.lower() == "all":
        return list(LEVELS)
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [level for level in levels if level not in LEVELS]
    if unknown:
        raise ValueError(f"Unknown level(s): {', '.join(unknown)}")
    if not levels:
        raise ValueError("No levels selected")
    return levels


def resolve_env_path() -> str:
    """Resolve the exported game path from src/.config, like utils.py does."""
    config_path = Path(__file__).resolve().with_name(".config")
    load_dotenv(config_path, override=True)

    env_dir = os.getenv("ENV_PATH", str(DEFAULT_ENV_PATH))
    env_name = os.getenv("ENV_NAME", "DragonJump")
    config_dir = config_path.parent
    repo_dir = config_dir.parent

    if os.path.isabs(env_dir):
        resolved = Path(env_dir)
    else:
        resolved = None
        for base in (Path.cwd(), config_dir, repo_dir):
            candidate = (base / env_dir).resolve()
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            resolved = (Path.cwd() / env_dir).resolve()

    return str(resolved / env_name)


def load_spawn_pool(path: Path) -> list[dict]:
    """Load ``{"train": [...]}`` or a bare spawn list for PPO resets."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("train") or payload.get("spawns") or payload.get("cells")
    else:
        entries = None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} has no non-empty train/spawns/cells list")
    pool: list[dict] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path} spawn[{index}] must be an object")
        if entry.get("canonical"):
            pool.append({"canonical": True})
            continue
        cell = entry.get("cell")
        facing = int(entry.get("facing", 1))
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            raise ValueError(f"{path} spawn[{index}].cell must be [x, y]")
        pool.append({
            "cell": [int(cell[0]), int(cell[1])],
            "facing": 1 if facing >= 0 else -1,
        })
    return pool


def load_spawn_holdout(path: Path) -> list[dict]:
    """Load the held-out spawn list from a spawn-pool JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    entries = payload.get("holdout", [])
    if not isinstance(entries, list):
        return []
    holdout: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cell = entry.get("cell")
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            continue
        holdout.append({
            "cell": [int(cell[0]), int(cell[1])],
            "facing": 1 if int(entry.get("facing", 1)) >= 0 else -1,
        })
    return holdout


def load_start_pool(path: Path) -> list[dict]:
    """Load optional canonical/cell/NEAT-prefix start specifications."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("starts") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} has no non-empty start list")
    pool: list[dict] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path} start[{index}] must be an object")
        item = dict(entry)
        item["weight"] = max(0.0, float(item.get("weight", 1.0)))
        prefix = item.get("prefix_actions")
        if prefix is not None:
            if not isinstance(prefix, list):
                raise ValueError(f"{path} start[{index}].prefix_actions must be a list")
            item["prefix_actions"] = [int(action) for action in prefix]
        elif not item.get("canonical"):
            cell = item.get("cell")
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                raise ValueError(f"{path} start[{index}] needs cell or prefix_actions")
            item["cell"] = [int(cell[0]), int(cell[1])]
            item["facing"] = 1 if int(item.get("facing", 1)) >= 0 else -1
        if item["weight"] <= 0.0:
            continue
        pool.append(item)
    if not pool:
        raise ValueError(f"{path} has no positive-weight starts")
    return pool


class SingleAgentGodotEnv(gym.Env):
    """One Godot training world exposed as a standard Gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_path: str,
        level: str,
        port: int,
        seed: int,
        action_repeat: int,
        speedup: float,
        max_decisions: int,
        show_window: bool,
        log_path: str,
        child_env: dict[str, str],
        force_start_jump: bool,
        reward_profile: str,
        obs_encoding: str,
        training_no_cosmetics: bool = False,
        start_pool: list[dict] | None = None,
        start_seed: int = 0,
        grid_dropout: float = 0.0,
        sensor_dropout: float = 0.0,
        trajectory_dir: str | None = None,
        trajectory_max_time: float = 0.0,
    ) -> None:
        super().__init__()
        self.level = level
        self.max_decisions = max_decisions
        self.action_repeat = action_repeat
        self.force_start_jump = force_start_jump
        self.obs_encoding = obs_encoding
        self._steps = 0
        self.start_pool = [dict(item) for item in (start_pool or [])]
        self.start_rng = np.random.default_rng(start_seed)
        # ``force_start_jump`` is applied during reset (not as an action
        # override in step) so PPO stores only actions that the policy chose.
        # It is part of the reset contract for stationary cell/canonical starts.
        self.start_weights = np.asarray(
            [max(0.0, float(item.get("weight", 1.0))) for item in self.start_pool],
            dtype=np.float64,
        )
        if self.start_weights.size:
            total = float(self.start_weights.sum())
            if total <= 0.0:
                self.start_pool = []
                self.start_weights = np.asarray([], dtype=np.float64)
            else:
                self.start_weights /= total
        self.last_start: dict = {}
        self._start_info: dict = {}
        self.grid_dropout = max(0.0, min(1.0, float(grid_dropout)))
        self.sensor_dropout = max(0.0, min(1.0, float(sensor_dropout)))
        self.dropout_rng = np.random.default_rng(start_seed + 104729)
        self.trajectory_dir = (
            Path(trajectory_dir).expanduser().resolve()
            if trajectory_dir is not None
            else None
        )
        if self.trajectory_dir is not None:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_max_time = max(0.0, float(trajectory_max_time))
        self._trajectory_obs: list[np.ndarray] = []
        self._trajectory_actions: list[int] = []
        self._trajectory_id = 0
        self._last_obs: np.ndarray | None = None
        extra_env_kwargs = (
            {"training_no_cosmetics": "true"}
            if training_no_cosmetics
            else {}
        )
        self.game = GodotEnv(
            env_path=env_path,
            port=port,
            show_window=show_window,
            seed=seed,
            action_repeat=action_repeat,
            speedup=speedup,
            nb_agents=1,
            level=level,
            convert_action_space=True,
            log_path=log_path,
            env=child_env,
            reward_profile=reward_profile,
            obs_encoding=obs_encoding,
            **extra_env_kwargs,
        )
        self.observation_space = self.game.observation_space.spaces["obs"]
        self.action_space = spaces.Discrete(2)

    def _obs(self, obs_list):
        return np.asarray(obs_list[0]["obs"], dtype=np.float32)

    def _augment_obs(self, observation: np.ndarray) -> np.ndarray:
        """Training-only structured dropout for the sensor adaptation run."""
        if self.grid_dropout <= 0.0 and self.sensor_dropout <= 0.0:
            return observation
        observation = np.asarray(observation, dtype=np.float32).copy()
        # Base multi-hot grid: 7*7 cells * 15 values. Keep the player/route
        # tail intact so the policy still knows where it is and where it goes.
        grid_values = 7 * 7 * 15
        if self.grid_dropout > 0.0 and observation.size >= grid_values:
            for cell in range(7 * 7):
                if self.dropout_rng.random() < self.grid_dropout:
                    start = cell * 15
                    observation[start : start + 15] = 0.0
        sensor_start = observation.size - 40
        if self.sensor_dropout > 0.0 and sensor_start >= 0:
            for index in range(sensor_start, observation.size):
                if self.dropout_rng.random() < self.sensor_dropout:
                    observation[index] = 0.0
        return observation

    def _sample_start(self) -> dict:
        if not self.start_pool:
            return {"canonical": True}
        index = int(
            self.start_rng.choice(len(self.start_pool), p=self.start_weights)
        )
        return dict(self.start_pool[index])

    def _clear_trajectory(self) -> None:
        self._trajectory_obs = []
        self._trajectory_actions = []

    def _save_trajectory(self, info: dict) -> None:
        """Persist one successful, sufficiently fast episode for offline BC."""
        if self.trajectory_dir is None or not self._trajectory_obs:
            return
        run_time = info.get("run_time", float("inf"))
        try:
            run_time = float(run_time)
        except (TypeError, ValueError):
            return
        if not np.isfinite(run_time) or run_time > self.trajectory_max_time:
            return
        cell = info.get("start_cell")
        if cell is None:
            cell_array = np.empty((0,), dtype=np.int64)
        else:
            cell_array = np.asarray(cell, dtype=np.int64)
        distance = info.get("start_distance")
        distance_value = (
            float(distance)
            if isinstance(distance, (int, float, np.integer, np.floating))
            else np.nan
        )
        self._trajectory_id += 1
        path = self.trajectory_dir / (
            f"episode_{self._trajectory_id:06d}_{time.time_ns()}.npz"
        )
        np.savez_compressed(
            path,
            observations=np.stack(self._trajectory_obs),
            actions=np.asarray(self._trajectory_actions, dtype=np.int64),
            run_time=np.float32(run_time),
            level=np.array(self.level),
            start_source=np.array(str(info.get("start_source", ""))),
            start_kind=np.array(str(info.get("start_kind", ""))),
            start_cell=cell_array,
            start_facing=np.int64(int(info.get("start_facing") or 0)),
            start_distance=np.float32(distance_value),
            prefix_len=np.int64(int(info.get("prefix_len") or 0)),
            decision_steps=np.int64(int(info.get("decision_steps") or 0)),
        )
        print(f"trajectory saved: {path}", flush=True)

    def reset(self, *, seed=None, options=None):
        self._clear_trajectory()
        spec = self._sample_start()
        for _attempt in range(4):
            prefix = spec.get("prefix_actions") or []
            if prefix:
                obs, infos = self.game.reset()
                done = False
                for action in prefix:
                    obs, _rewards, dones, _timeouts, infos = self.game.step(
                        np.asarray([[int(action)]], dtype=np.int64)
                    )
                    if bool(dones[0]):
                        done = True
                        break
                if done:
                    spec = self._sample_start()
                    continue
            elif spec.get("cell") is not None:
                obs, infos = self.game.reset(
                    spawns=[
                        {
                            "agent": 0,
                            "cell": list(spec["cell"]),
                            "facing": int(spec.get("facing", 1)),
                        }
                    ]
                )
            else:
                obs, infos = self.game.reset()

            start_jump = bool(self.force_start_jump and not prefix)
            if start_jump:
                # The training controller starts the run on the first jump.
                # Press it as part of reset, then hand control to the policy;
                # no synthetic action is inserted into the PPO rollout.
                obs, _rewards, dones, _timeouts, infos = self.game.step(
                    np.asarray([[1]], dtype=np.int64)
                )
                if bool(dones[0]):
                    spec = self._sample_start()
                    continue

            self.last_start = dict(spec)
            self._steps = 0
            info = dict(infos[0] if infos else {})
            self._start_info = {
                "start_source": (
                    "prefix"
                    if prefix
                    else "canonical"
                    if spec.get("canonical")
                    else "cell"
                ),
                "start_kind": (
                    "prefix"
                    if prefix
                    else str(
                        spec.get(
                            "kind",
                            "canonical" if spec.get("canonical") else "cell",
                        )
                    )
                ),
                "prefix_len": len(prefix),
                "start_cell": spec.get("cell"),
                "start_facing": spec.get("facing"),
                "start_distance": spec.get("distance"),
                "start_jump": start_jump,
            }
            info.update(self._start_info)
            observation = self._augment_obs(self._obs(obs))
            self._last_obs = observation
            return observation, info

        # A malformed/all-terminal prefix should not wedge training.
        obs, infos = self.game.reset()
        start_jump = bool(self.force_start_jump)
        if start_jump:
            obs, _rewards, _dones, _timeouts, infos = self.game.step(
                np.asarray([[1]], dtype=np.int64)
            )
        self.last_start = {"canonical": True}
        self._steps = 0
        info = dict(infos[0] if infos else {})
        self._start_info = {
            "start_source": "canonical_fallback",
            "start_kind": "canonical",
            "prefix_len": 0,
            "start_jump": start_jump,
        }
        info.update(self._start_info)
        observation = self._augment_obs(self._obs(obs))
        self._last_obs = observation
        return observation, info

    def step(self, action):
        if (
            self.trajectory_dir is not None
            and self._last_obs is not None
        ):
            self._trajectory_obs.append(self._last_obs.copy())
            self._trajectory_actions.append(int(action))
        actions = np.asarray([[int(action)]], dtype=np.int64)
        # GodotEnv.step exposes the raw 5-tuple. The multi-agent SB3 wrapper
        # collapses terminated/truncated into a single done, but here we keep
        # them separate so Gymnasium/SB3 can handle timeout bootstrapping.
        obs, rewards, dones, _timeouts, infos = self.game.step(actions)
        obs = self._augment_obs(self._obs(obs))
        reward = float(rewards[0])
        done = bool(dones[0])
        info = dict(infos[0] if infos else {})
        if self._start_info:
            info.update(self._start_info)

        self._steps += 1
        truncated = False
        if not done and self._steps >= self.max_decisions:
            done = True
            truncated = True

        run_time = info.get("run_time", float("nan"))
        try:
            run_time = float(run_time)
        except (TypeError, ValueError):
            run_time = float("nan")

        info["finished"] = bool(done and not truncated and reward > 0.0)
        info["timed_out"] = bool(truncated)
        if info["finished"] and (
            not np.isfinite(run_time) or run_time <= 0.0
        ):
            # The training controller can zero its timer when the finish signal
            # is consumed. Derive the time from the decision count instead of
            # reporting a false 0.0 s finish.
            run_time = self._steps * self.action_repeat / 60.0
        info["run_time"] = run_time
        info["decision_steps"] = self._steps
        info["level"] = self.level
        info["finished"] = bool(info["finished"])
        self._last_obs = obs
        if info["finished"]:
            self._save_trajectory(info)
        return obs, reward, done, truncated, info

    def close(self):
        self.game.close()


def isolated_home(output_dir: Path, name: str) -> dict[str, str]:
    """Give each game process a private user:// tree.

    Training must never read or write the player's real save, settings,
    ghosts, medals, unlocks, or leaderboard data. Godot resolves user:// from
    HOME/XDG on this platform; a per-process HOME keeps those writes isolated.
    """
    root = (output_dir / "userdata" / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["HOME"] = str(root)
    child_env["XDG_DATA_HOME"] = str(root / "xdg_data")
    child_env["XDG_CONFIG_HOME"] = str(root / "xdg_config")
    return child_env


def make_env(rank: int, args: argparse.Namespace, env_path: str):
    level = args.levels[rank % len(args.levels)]
    log_path = args.output_dir / "godot" / f"train_{rank}_{level}.log"
    start_pool = [
        item
        for item in args.start_pool_data
        if item.get("level") in (None, level)
    ]
    return SingleAgentGodotEnv(
        env_path=env_path,
        level=level,
        port=args.base_port + rank,
        seed=args.seed + rank,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        max_decisions=args.max_decisions,
        show_window=args.show,
        log_path=str(log_path),
        child_env=isolated_home(args.output_dir, f"train_{rank}"),
        # Applied during reset, not by overriding a rollout action. This keeps
        # SB3's stored action and the action actually sent to Godot identical.
        force_start_jump=args.force_start_jump,
        reward_profile=args.reward_profile,
        obs_encoding=args.obs_encoding,
        training_no_cosmetics=args.training_no_cosmetics,
        start_pool=start_pool,
        start_seed=args.start_seed + rank,
        grid_dropout=args.grid_dropout,
        sensor_dropout=args.sensor_dropout,
        trajectory_dir=(
            str(args.trajectory_dir) if args.trajectory_dir else None
        ),
        trajectory_max_time=args.trajectory_max_time,
    )


def build_vec_env(args: argparse.Namespace, env_path: str):
    if args.agents_per_process > 1:
        if len(args.levels) != 1:
            raise ValueError(
                "--agents-per-process currently supports exactly one level"
            )
        vec_env = MultiAgentGodotVecEnv(
            env_path=env_path,
            level=args.levels[0],
            n_agents_per_process=args.agents_per_process,
            n_processes=args.n_envs,
            seed=args.seed,
            action_repeat=args.action_repeat,
            speedup=args.speedup,
            max_decisions=args.max_decisions,
            base_port=args.base_port,
            output_dir=args.output_dir,
            obs_encoding=args.obs_encoding,
            reward_profile=args.reward_profile,
            show_window=args.show,
            training_no_cosmetics=args.training_no_cosmetics,
            spawn_pool=args.spawn_pool_data,
            spawn_seed=args.spawn_seed,
        )
    else:
        env_fns = [
            partial(make_env, rank, args, env_path)
            for rank in range(args.n_envs)
        ]
        if args.n_envs == 1:
            vec_env = DummyVecEnv(env_fns)
        else:
            # spawn is the portable macOS start method; Godot sockets are not
            # pickled, only the env factory is.
            vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    return VecMonitor(
        vec_env,
        filename=str(args.output_dir / "monitor.csv"),
        info_keywords=(
            "level",
            "finished",
            "timed_out",
            "run_time",
            "start_source",
            "start_kind",
            "start_cell",
            "start_facing",
            "start_distance",
            "prefix_len",
        ),
    )


class EpisodeCSVCallback(BaseCallback):
    """Write one row per completed training episode and save checkpoints."""

    WINDOW = 200
    LOG_EVERY = 25

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.started = time.time()
        self.episodes = 0
        self.last_save = 0
        self.last_metrics_log = 0
        self.metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.WINDOW)
        )
        self.csv_file = open(args.output_dir / "episodes.csv", "w", newline="")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timesteps",
                "level",
                "spawn_cell",
                "start_source",
                "start_kind",
                "start_cell",
                "start_facing",
                "start_distance",
                "prefix_len",
                "episode_return",
                "episode_length",
                "finished",
                "timed_out",
                "run_time",
                "elapsed_s",
            ],
        )
        self.writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if not episode:
                continue
            self.episodes += 1
            timed_out = bool(info.get("timed_out", False))
            run_time = info.get("run_time", "")
            if timed_out:
                run_time = ""
            elif isinstance(run_time, (float, np.floating)) and not np.isfinite(run_time):
                run_time = ""
            row = {
                "timesteps": self.num_timesteps,
                "level": info.get("level", ""),
                "spawn_cell": info.get("spawn_cell", ""),
                "start_source": info.get("start_source", ""),
                "start_kind": info.get("start_kind", ""),
                "start_cell": info.get("start_cell", ""),
                "start_facing": info.get("start_facing", ""),
                "start_distance": info.get("start_distance", ""),
                "prefix_len": info.get("prefix_len", ""),
                "episode_return": float(episode["r"]),
                "episode_length": int(episode["l"]),
                "finished": bool(info.get("finished", False)),
                "timed_out": timed_out,
                "run_time": run_time,
                "elapsed_s": round(time.time() - self.started, 2),
            }
            self.writer.writerow(row)
            self.csv_file.flush()
            self._remember(row)
            print(
                f"[ep {self.episodes}] t={row['timesteps']} level={row['level']} "
                f"start={row['start_kind'] or row['start_source'] or '?'} "
                f"reward={row['episode_return']:.2f} steps={row['episode_length']} "
                f"finish={row['finished']} timeout={row['timed_out']} "
                f"time={row['run_time']}",
                flush=True,
            )

        if self.episodes - self.last_metrics_log >= self.LOG_EVERY:
            self.last_metrics_log = self.episodes
            self._log_metrics()

        if (
            self.args.save_freq > 0
            and self.num_timesteps - self.last_save >= self.args.save_freq
        ):
            self.last_save = self.num_timesteps
            path = self.args.output_dir / "checkpoints" / f"ppo-{self.num_timesteps}"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))
            print(f"checkpoint: {path}.zip", flush=True)
        return True

    def _remember(self, row: dict) -> None:
        level = row["level"]
        kind = row["start_kind"] or row["start_source"] or "unknown"
        self.metrics["all"].append(row)
        self.metrics[level].append(row)
        self.metrics[f"{level}/{kind}"].append(row)
        if kind == "canonical":
            self.metrics["canonical"].append(row)

    @staticmethod
    def _summarize(records: deque) -> tuple[int, float, float | None, float | None]:
        rows = list(records)
        if not rows:
            return 0, 0.0, None, None
        finished = [row for row in rows if row["finished"]]
        times = [
            float(row["run_time"])
            for row in finished
            if row["run_time"] not in ("", None)
        ]
        finish_rate = len(finished) / len(rows)
        if not times:
            return len(rows), finish_rate, None, None
        return (
            len(rows),
            finish_rate,
            float(statistics.median(times)),
            float(min(times)),
        )

    def _log_metrics(self) -> None:
        if self.logger is None or not self.metrics:
            return
        for key in sorted(self.metrics):
            count, finish_rate, median_time, best_time = self._summarize(
                self.metrics[key]
            )
            if count == 0:
                continue
            self.logger.record(f"{key}/episodes_window", count)
            self.logger.record(f"{key}/finish_rate", finish_rate)
            if median_time is not None:
                self.logger.record(f"{key}/median_time", median_time)
            if best_time is not None:
                self.logger.record(f"{key}/best_time", best_time)
        self.logger.record("train/episodes_total", self.episodes)
        self.logger.dump(self.num_timesteps)

    def _on_training_end(self) -> None:
        self._log_metrics()
        self.csv_file.flush()
        self.csv_file.close()


class EvalPatienceCallback(BaseCallback):
    """Stop on a deterministic evaluation plateau and keep the best model.

    The training reward is not the stopping signal: PPO can improve its return
    while the deployed argmax policy gets slower. The callback evaluates the
    current policy on the target level, uses median finish time as the primary
    metric (finish rate is a gate), and saves the best checkpoint separately.
    """

    def __init__(self, args: argparse.Namespace, env_path: str):
        super().__init__()
        self.args = args
        self.env_path = env_path
        self.last_eval = 0
        self.best_metric = float("inf")
        self.best_step = -1
        self.no_improve = 0
        self.best_path = args.output_dir / "best_model.zip"
        self.baseline_done = False

    def _metric(self, result: dict) -> float:
        finish_rate = (
            int(result.get("finishes", 0)) / max(1, self.args.eval_agents)
        )
        if finish_rate < self.args.min_finish_rate:
            return float("inf")
        for key in ("median_time", "best_time"):
            value = result.get(key, "")
            if value not in ("", None):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        return float("inf")

    def _on_training_start(self) -> None:
        if (
            not self.args.baseline_eval
            or self.baseline_done
            or self.args.eval_every <= 0
            or self.args.patience <= 0
        ):
            return
        self.baseline_done = True
        level = self.args.levels[0]
        result = evaluate_policy(self.model, self.args, self.env_path, level)
        metric = self._metric(result)
        finish_rate = int(result.get("finishes", 0)) / max(
            1, self.args.eval_agents
        )
        if metric != float("inf"):
            self.best_metric = metric
            self.best_step = 0
            self.model.save(str(self.best_path))
            print(
                f"[baseline] {level} median={metric:.3f}s "
                f"finish_rate={finish_rate:.2f}; saved as best",
                flush=True,
            )

    def _on_step(self) -> bool:
        if self.args.eval_every <= 0 or self.args.patience <= 0:
            return True
        if self.num_timesteps - self.last_eval < self.args.eval_every:
            return True

        self.last_eval = self.num_timesteps
        checkpoint = (
            self.args.output_dir
            / "checkpoints"
            / f"ppo-{self.num_timesteps}.zip"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(checkpoint))

        level = self.args.levels[0]
        result = evaluate_policy(self.model, self.args, self.env_path, level)
        metric = self._metric(result)
        finish_rate = int(result.get("finishes", 0)) / max(
            1, self.args.eval_agents
        )
        improved = metric < self.best_metric * (1.0 - self.args.min_improvement)
        if improved:
            self.best_metric = metric
            self.best_step = self.num_timesteps
            self.no_improve = 0
            self.model.save(str(self.best_path))
            print(
                f"[patience] best {level} median={metric:.3f}s "
                f"finish_rate={finish_rate:.2f} at {self.num_timesteps}",
                flush=True,
            )
        else:
            self.no_improve += 1
            print(
                f"[patience] no improvement {self.no_improve}/"
                f"{self.args.patience} at {self.num_timesteps} "
                f"(metric={metric:.3f}s finish_rate={finish_rate:.2f})",
                flush=True,
            )
        if self.no_improve >= self.args.patience:
            print(
                f"[patience] stopping; best median={self.best_metric:.3f}s "
                f"at {self.best_step}",
                flush=True,
            )
            return False
        return True


class SpawnEvalCallback(BaseCallback):
    """Keep the best checkpoint by default-spawn time plus spawn recovery.

    The canonical 1-1 time is a hard constraint: a checkpoint that clears the
    held-out spawns but regresses the default start is not selected.
    """

    def __init__(self, args: argparse.Namespace, env_path: str):
        super().__init__()
        self.args = args
        self.env_path = env_path
        self.holdout = (
            load_spawn_holdout(args.spawn_pool)
            if args.spawn_pool is not None
            else []
        )
        self.env: GodotEnv | None = None
        self.last_eval = 0
        self.baseline_canonical: tuple[bool, float, int] | None = None
        self.best_score: tuple | None = None
        self.best_path = args.output_dir / "spawn_best.zip"
        self.baseline_path = args.output_dir / "spawn_baseline.zip"
        self.best_canonical_path = args.output_dir / "spawn_best_canonical.zip"
        self.best_holdout_path = args.output_dir / "spawn_best_holdout.zip"
        self.best_canonical_time = float("inf")
        self.best_holdout: tuple[int, float] = (-1, float("inf"))
        self.csv_file = open(args.output_dir / "spawn_eval.csv", "w", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "phase",
                "timesteps",
                "canonical_finished",
                "canonical_time",
                "canonical_ok",
                "holdout_finished",
                "holdout_total",
                "holdout_time",
                "saved_best",
            ],
        )
        self.csv_writer.writeheader()

    def _open_env(self) -> None:
        self.env = GodotEnv(
            env_path=self.env_path,
            port=self.args.eval_port,
            show_window=False,
            seed=self.args.seed,
            action_repeat=self.args.action_repeat,
            speedup=self.args.speedup,
            nb_agents=1,
            level=self.args.levels[0],
            convert_action_space=True,
            log_path=str(self.args.output_dir / "godot" / "spawn_eval.log"),
            env=isolated_home(self.args.output_dir, "spawn_eval"),
            reward_profile=self.args.reward_profile,
            obs_encoding=self.args.obs_encoding,
        )

    def _run(self, spawn: dict | None = None) -> tuple[bool, float, int]:
        assert self.env is not None
        if spawn is None:
            obs, _infos = self.env.reset()
        else:
            obs, _infos = self.env.reset(
                spawns=[{**spawn, "agent": 0}]
            )
        if self.args.force_start_jump:
            obs, _rewards, dones, _timeouts, infos = self.env.step(
                np.asarray([[1]], dtype=np.int64)
            )
            if bool(dones[0]):
                info = infos[0] if infos else {}
                return (
                    bool(info.get("finished", False)),
                    float(info.get("run_time", 0.0)),
                    1,
                )
        for step in range(self.args.max_decisions):
            action, _ = self.model.predict(
                np.asarray(obs[0]["obs"], dtype=np.float32),
                deterministic=True,
            )
            obs, _rewards, dones, _timeouts, infos = self.env.step(
                np.asarray([[int(action)]], dtype=np.int64)
            )
            if dones[0]:
                info = infos[0] if infos else {}
                return (
                    bool(info.get("finished", False)),
                    float(info.get("run_time", 0.0)),
                    step + 1,
                )
        return False, 0.0, self.args.max_decisions

    def _holdout_result(self) -> tuple[int, float]:
        finished = 0
        times: list[float] = []
        for spawn in self.holdout:
            ok, time_s, _steps = self._run(spawn)
            if ok:
                finished += 1
                times.append(time_s)
        return finished, (sum(times) / len(times) if times else float("inf"))

    def _score(
        self,
        canonical: tuple[bool, float, int],
        holdout_finished: int,
        holdout_time: float,
    ) -> tuple:
        target = float(self.args.spawn_canonical_target)
        canonical_ok = canonical[0] and canonical[1] <= target
        return (
            1 if canonical_ok else 0,
            holdout_finished,
            -canonical[1] if canonical[0] else -1e9,
            -holdout_time,
        )

    def _canonical_ok(self, canonical: tuple[bool, float, int]) -> bool:
        return (
            canonical[0]
            and canonical[1] <= float(self.args.spawn_canonical_target)
        )

    def _update_bests(
        self,
        canonical: tuple[bool, float, int],
        holdout_finished: int,
        holdout_time: float,
    ) -> None:
        """Keep independent best-canonical and best-holdout checkpoints.

        ``spawn_best`` remains the promotable checkpoint (canonical gate AND
        recovery). These two extra files preserve the best half-result so a
        later stage can resume from the best speed or the best recovery even
        when no checkpoint passes both gates yet.
        """
        if canonical[0] and canonical[1] < self.best_canonical_time:
            self.best_canonical_time = canonical[1]
            self.model.save(str(self.best_canonical_path))
            print(
                f"[spawn-eval] saved best canonical: "
                f"{self.best_canonical_path} ({canonical[1]:.3f}s)",
                flush=True,
            )
        if holdout_finished > self.best_holdout[0] or (
            holdout_finished == self.best_holdout[0]
            and holdout_time < self.best_holdout[1]
        ):
            self.best_holdout = (holdout_finished, holdout_time)
            self.model.save(str(self.best_holdout_path))
            print(
                f"[spawn-eval] saved best holdout: "
                f"{self.best_holdout_path} "
                f"({holdout_finished}/{len(self.holdout)})",
                flush=True,
            )

    def _record_eval(
        self,
        phase: str,
        canonical: tuple[bool, float, int],
        holdout_finished: int,
        holdout_time: float,
        saved_best: bool,
    ) -> None:
        def finite(value: float) -> str:
            return f"{value:.3f}" if np.isfinite(value) else ""

        self.csv_writer.writerow(
            {
                "phase": phase,
                "timesteps": self.num_timesteps,
                "canonical_finished": int(canonical[0]),
                "canonical_time": finite(canonical[1]) if canonical[0] else "",
                "canonical_ok": int(self._canonical_ok(canonical)),
                "holdout_finished": holdout_finished,
                "holdout_total": len(self.holdout),
                "holdout_time": finite(holdout_time),
                "saved_best": int(saved_best),
            }
        )
        self.csv_file.flush()

    def _on_training_start(self) -> None:
        if self.args.spawn_eval_every <= 0 or not self.holdout:
            return
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self._open_env()
        self.baseline_canonical = self._run()
        holdout_finished, holdout_time = self._holdout_result()
        self.best_score = self._score(
            self.baseline_canonical,
            holdout_finished,
            holdout_time,
        )
        self.model.save(str(self.baseline_path))
        baseline_ok = self._canonical_ok(self.baseline_canonical)
        if baseline_ok:
            self.model.save(str(self.best_path))
        self._update_bests(
            self.baseline_canonical,
            holdout_finished,
            holdout_time,
        )
        self._record_eval(
            "baseline",
            self.baseline_canonical,
            holdout_finished,
            holdout_time,
            baseline_ok,
        )
        self.last_eval = self.num_timesteps
        print(
            f"[spawn-eval] baseline canonical={self.baseline_canonical[1]:.3f}s "
            f"holdout={holdout_finished}/{len(self.holdout)} "
            f"canonical_ok={baseline_ok}",
            flush=True,
        )

    def _on_step(self) -> bool:
        if self.args.spawn_eval_every <= 0 or not self.holdout:
            return True
        if self.num_timesteps - self.last_eval < self.args.spawn_eval_every:
            return True
        self.last_eval = self.num_timesteps

        checkpoint = (
            self.args.output_dir
            / "checkpoints"
            / f"spawn-{self.num_timesteps}.zip"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(checkpoint))
        canonical = self._run()
        holdout_finished, holdout_time = self._holdout_result()
        score = self._score(canonical, holdout_finished, holdout_time)
        print(
            f"[spawn-eval] step={self.num_timesteps} "
            f"canonical={canonical[1]:.3f}s "
            f"holdout={holdout_finished}/{len(self.holdout)} "
            f"holdout_time={holdout_time:.3f}s",
            flush=True,
        )
        canonical_ok = self._canonical_ok(canonical)
        saved_best = False
        if canonical_ok and score > self.best_score:
            self.best_score = score
            self.model.save(str(self.best_path))
            saved_best = True
            print(f"[spawn-eval] saved best: {self.best_path}", flush=True)
        elif not canonical_ok:
            print("[spawn-eval] rejected: canonical gate failed", flush=True)
        self._update_bests(canonical, holdout_finished, holdout_time)
        self._record_eval(
            "step",
            canonical,
            holdout_finished,
            holdout_time,
            saved_best,
        )
        return True

    def _on_training_end(self) -> None:
        self.csv_file.flush()
        self.csv_file.close()
        if self.env is not None:
            self.env.close()


def batch_predict_actions(
    model: PPO,
    observations: list[np.ndarray],
    deterministic: bool = True,
) -> np.ndarray:
    """Run one policy forward for all active evaluation agents."""
    if not observations:
        return np.empty(0, dtype=np.int64)
    obs_array = np.asarray(observations, dtype=np.float32)
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs_array)
        actions, _, _ = model.policy(
            obs_tensor,
            deterministic=deterministic,
        )
    return actions.detach().cpu().numpy().reshape(-1).astype(np.int64)


def evaluate_policy(
    model: PPO,
    args: argparse.Namespace,
    env_path: str,
    level: str,
) -> dict:
    """Run one policy on 100 agents in one evaluation process for one level."""
    run_id = f"{level}-{args.seed}"
    out_dir = args.output_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    child_env = isolated_home(args.output_dir, f"eval_{run_id}")
    env = GodotEnv(
        env_path=env_path,
        port=args.eval_port,
        show_window=args.show,
        seed=args.seed,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        nb_agents=args.eval_agents,
        level=level,
        convert_action_space=True,
        log_path=str(out_dir / f"godot-{run_id}.log"),
        env=child_env,
        reward_profile=args.reward_profile,
        obs_encoding=args.obs_encoding,
    )

    rows = []
    try:
        obs, _ = env.reset()
        n = args.eval_agents
        done_once = np.zeros(n, dtype=bool)
        finished = np.zeros(n, dtype=bool)
        finish_times = np.full(n, np.inf, dtype=np.float64)
        rewards_sum = np.zeros(n, dtype=np.float64)
        steps_run = np.zeros(n, dtype=np.int32)
        efficiency = np.full(n, np.nan, dtype=np.float64)
        flow = np.full(n, np.nan, dtype=np.float64)
        deaths = np.zeros(n, dtype=np.int32)
        last_infos: list[dict] = [{} for _ in range(n)]
        step_offset = 0

        if args.force_start_jump:
            # Same reset contract as training: press jump once, then let the
            # policy choose every subsequent action. An exit-adjacent spawn
            # may finish on this jump; keep that terminal state.
            obs, start_rewards, start_dones, _timeouts, start_infos = env.step(
                np.ones((n, 1), dtype=np.int64)
            )
            step_offset = 1
            for i in range(n):
                info = start_infos[i] if i < len(start_infos) else {}
                last_infos[i] = dict(info)
                rewards_sum[i] += float(start_rewards[i])
                steps_run[i] = 1
                if bool(start_dones[i]):
                    done_once[i] = True
                    finished[i] = bool(
                        info.get("finished", float(start_rewards[i]) > 0.0)
                    )
                    if finished[i]:
                        try:
                            finish_times[i] = float(info["run_time"])
                        except (KeyError, TypeError, ValueError):
                            finish_times[i] = 0.0

        for step in range(args.max_decisions):
            if done_once.all():
                break
            actions = np.zeros((n, 1), dtype=np.int64)
            active: list[int] = []
            for i in range(n):
                if done_once[i]:
                    continue
                if step < args.start_delay:
                    continue
                active.append(i)
            if active:
                if args.eval_legacy_per_agent:
                    for i in active:
                        action, _ = model.predict(
                            obs[i]["obs"],
                            deterministic=not args.eval_stochastic,
                        )
                        actions[i, 0] = int(action)
                else:
                    actions[active, 0] = batch_predict_actions(
                        model,
                        [obs[i]["obs"] for i in active],
                        deterministic=not args.eval_stochastic,
                    )

            obs, rewards, dones, _timeouts, infos = env.step(actions)
            for i in range(n):
                if done_once[i]:
                    continue
                last_infos[i] = dict(infos[i]) if infos[i] else {}
                rewards_sum[i] += float(rewards[i])
                steps_run[i] = step + 1 + step_offset
                if dones[i]:
                    done_once[i] = True
                    finished[i] = float(rewards[i]) > 0.0
                    if finished[i]:
                        try:
                            finish_times[i] = float(infos[i]["run_time"])
                        except (KeyError, TypeError, ValueError):
                            finish_times[i] = float("nan")
                        if (
                            not np.isfinite(finish_times[i])
                            or finish_times[i] <= 0.0
                        ):
                            finish_times[i] = (
                                (step + 1) * args.action_repeat / 60.0
                            )
            if done_once.all():
                break

        for i in range(n):
            info = last_infos[i]
            if "efficiency" in info:
                try:
                    efficiency[i] = float(info["efficiency"])
                except (TypeError, ValueError):
                    pass
            if "flow" in info:
                try:
                    flow[i] = float(info["flow"])
                except (TypeError, ValueError):
                    pass
            if "deaths" in info:
                try:
                    deaths[i] = int(info["deaths"])
                except (TypeError, ValueError):
                    pass
            rows.append(
                {
                    "agent": i,
                    "finished": bool(finished[i]),
                    "steps": int(steps_run[i]),
                    "reward": float(rewards_sum[i]),
                    "run_time": (
                        "" if not finished[i] else f"{finish_times[i]:.6f}"
                    ),
                    "efficiency": (
                        f"{efficiency[i]:.6f}"
                        if np.isfinite(efficiency[i])
                        else ""
                    ),
                    "flow": (
                        f"{flow[i]:.6f}"
                        if np.isfinite(flow[i])
                        else ""
                    ),
                    "deaths": int(deaths[i]),
                }
            )

        finish_count = int(finished.sum())
        death_count = int(deaths.sum())
        timeout_count = int((~done_once).sum())
        clean_finish_times = finish_times[np.isfinite(finish_times)]
        best_time = (
            float(np.min(clean_finish_times))
            if clean_finish_times.size
            else None
        )
        median_time = (
            float(np.median(clean_finish_times))
            if clean_finish_times.size
            else None
        )
        mean_time = (
            float(np.mean(clean_finish_times))
            if clean_finish_times.size
            else None
        )
        time_cv = (
            float(np.std(clean_finish_times) / mean_time)
            if clean_finish_times.size and mean_time
            else None
        )
        efficiency_mean = (
            float(np.nanmean(efficiency))
            if np.isfinite(efficiency).any()
            else None
        )
        flow_mean = (
            float(np.nanmean(flow))
            if np.isfinite(flow).any()
            else None
        )
        print(
            f"[eval {level}] finishes={finish_count}/{n} "
            f"deaths={death_count} timeouts={timeout_count} "
            f"best={best_time if best_time is not None else 'none'} "
            f"median={median_time if median_time is not None else 'none'} "
            f"eff={efficiency_mean if efficiency_mean is not None else 'none'} "
            f"flow={flow_mean if flow_mean is not None else 'none'} "
            f"mean_reward={float(rewards_sum.mean()):.2f}",
            flush=True,
        )

        with open(out_dir / f"results-{run_id}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "agent",
                    "finished",
                    "steps",
                    "reward",
                    "run_time",
                    "efficiency",
                    "flow",
                    "deaths",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return {
            "level": level,
            "agents": n,
            "finishes": finish_count,
            "deaths": death_count,
            "timeouts": timeout_count,
            "best_time": f"{best_time:.6f}" if best_time is not None else "",
            "median_time": (
                f"{median_time:.6f}" if median_time is not None else ""
            ),
            "mean_time": f"{mean_time:.6f}" if mean_time is not None else "",
            "time_cv": f"{time_cv:.6f}" if time_cv is not None else "",
            "efficiency_mean": (
                f"{efficiency_mean:.6f}" if efficiency_mean is not None else ""
            ),
            "flow_mean": (
                f"{flow_mean:.6f}" if flow_mean is not None else ""
            ),
            "mean_reward": f"{float(rewards_sum.mean()):.2f}",
        }
    finally:
        env.close()


def evaluate_levels(
    model: PPO,
    args: argparse.Namespace,
    env_path: str,
) -> None:
    summaries = [
        evaluate_policy(model, args, env_path, level)
        for level in args.levels
    ]
    summary_path = args.output_dir / "eval" / "summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "level",
                "agents",
                "finishes",
                "deaths",
                "timeouts",
                "best_time",
                "median_time",
                "mean_time",
                "time_cv",
                "efficiency_mean",
                "flow_mean",
                "mean_reward",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)
    print(f"eval summary: {summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        help="Single-level shortcut. Overrides --levels when set.",
    )
    parser.add_argument(
        "--levels",
        default="all",
        help="'all' or comma-separated campaign ids (default: all).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Parallel worlds. Defaults to one per selected level.",
    )
    parser.add_argument(
        "--agents-per-process",
        type=int,
        default=1,
        help="Godot agents per process; 1 keeps the existing one-world-per-process path.",
    )
    parser.add_argument(
        "--spawn-pool",
        type=Path,
        help="JSON spawn pool for per-episode randomized start cells.",
    )
    parser.add_argument(
        "--spawn-seed",
        type=int,
        default=0,
        help="Seed for sampling from --spawn-pool.",
    )
    parser.add_argument(
        "--start-pool",
        type=Path,
        help="JSON weighted start pool with canonical/cell/prefix entries.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Seed for sampling from --start-pool.",
    )
    parser.add_argument(
        "--spawn-eval-every",
        type=int,
        default=25_000,
        help="Evaluate canonical + holdout spawns every N training steps; 0 disables.",
    )
    parser.add_argument(
        "--spawn-canonical-target",
        type=float,
        default=1.75,
        help="Absolute default-spawn time a spawn checkpoint must meet.",
    )
    parser.add_argument(
        "--spawn-learning-rate",
        type=float,
        default=None,
        help="Override the loaded PPO learning rate for spawn fine-tuning.",
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument(
        "--additional-timesteps",
        action="store_true",
        help="Compatibility flag for the lineage runner; SB3 already offsets resumed counters.",
    )
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-port", type=int, default=11008)
    parser.add_argument("--eval-port", type=int, default=12000)
    parser.add_argument("--eval-agents", type=int, default=100)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-stochastic", action="store_true")
    parser.add_argument(
        "--eval-legacy-per-agent",
        action="store_true",
        help="Diagnostic only: run model.predict once per active agent instead of batching.",
    )
    parser.add_argument(
        "--start-delay",
        type=int,
        default=0,
        help="Evaluation only: hold idle for N decisions after the reset start "
        "jump before the policy acts.",
    )
    parser.add_argument(
        "--grid-dropout",
        type=float,
        default=0.0,
        help="Training-only probability of zeroing each 7x7 multi-hot cell.",
    )
    parser.add_argument(
        "--sensor-dropout",
        type=float,
        default=0.0,
        help="Training-only probability of zeroing each appended sensor value.",
    )
    parser.add_argument(
        "--reward-profile",
        default="default",
        choices=("default", "time", "time_speed"),
        help="Training-only game reward profile.",
    )
    parser.add_argument(
        "--obs-encoding",
        default="legacy",
        choices=("legacy", "multi_hot", "multi_hot_sensor"),
        help="Game observation contract; multi_hot and multi_hot_sensor require "
        "a matching new model. multi_hot_sensor appends head-mounted ray features.",
    )
    parser.add_argument(
        "--grid-encoder",
        default="mlp",
        choices=("mlp", "legacy_cnn", "legacy_autoencoder", "semantic_autoencoder"),
        help="Legacy 58 only: tile embedding + 7x7 CNN instead of a flat MLP. "
        "legacy_autoencoder uses a frozen pretrained legacy grid encoder; "
        "semantic_autoencoder uses the frozen 754-value semantic MAE.",
    )
    parser.add_argument(
        "--autoencoder-path",
        type=Path,
        help="Checkpoint for a frozen autoencoder grid encoder.",
    )
    parser.add_argument(
        "--init-head",
        type=Path,
        help="Initialize the PPO policy/action head from a behavior-cloned "
        "frozen-encoder head; the value head remains random.",
    )
    parser.add_argument(
        "--force-start-jump",
        dest="force_start_jump",
        action="store_true",
        help="Press jump once during reset, then hand control to the policy "
        "(default: on).",
    )
    parser.add_argument(
        "--no-force-start-jump",
        dest="force_start_jump",
        action="store_false",
        help="Leave the first start action policy-owned (legacy behavior).",
    )
    parser.set_defaults(force_start_jump=True)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--training-no-cosmetics",
        action="store_true",
        help="Training-only A/B: suppress effect scenes, player particles, and animations.",
    )
    parser.add_argument(
        "--baseline-eval",
        action="store_true",
        help="Evaluate the initial policy and seed eval patience with it; "
        "implied by --init-student.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate every N timesteps (0 disables patience evaluation).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop after N evaluations without a meaningful improvement (0 disables).",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.01,
        help="Relative median-time improvement required to reset patience.",
    )
    parser.add_argument(
        "--min-finish-rate",
        type=float,
        default=0.90,
        help="Evaluation finish-rate gate for a candidate to count as an improvement.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--expand-from",
        type=Path,
        help="Warm-start a wider observation policy (e.g. multi_hot -> "
        "multi_hot_sensor) by expanding the first MLP input layer. "
        "Cannot be combined with --resume.",
    )
    parser.add_argument(
        "--expand-init",
        choices=("zero", "sample"),
        default="zero",
        help="Initialization for newly added input columns: zero (preserve old "
        "policy) or sample from the existing input columns.",
    )
    parser.add_argument(
        "--expand-init-scale",
        type=float,
        default=0.5,
        help="Scale applied to sampled new input columns.",
    )
    parser.add_argument(
        "--freeze-old-input",
        action="store_true",
        help="With --expand-from, freeze gradients on the original input "
        "columns so only appended sensor columns adapt.",
    )
    parser.add_argument(
        "--init-student",
        type=Path,
        help="Initialize PPO policy/action head from a StudentPolicy checkpoint; "
        "the value head remains random.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "ppo_pilot",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=None,
        help="TensorBoard log directory; defaults to <output-dir>/tensorboard.",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Opt-in directory for successful fast episode .npz trajectories.",
    )
    parser.add_argument(
        "--trajectory-max-time",
        type=float,
        default=5.5,
        help="Only save successful trajectories at or below this finish time.",
    )

    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def initialize_policy_from_student(
    model: PPO,
    student_path: Path,
    args: argparse.Namespace,
) -> None:
    """Warm-start PPO's policy/action head from a distillation checkpoint.

    The value head stays randomly initialized. This lets a multi-hot PPO
    teacher start from a legacy-teacher behavior clone instead of relearning
    movement and finish behavior from scratch.
    """
    payload = torch.load(student_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    student_encoding = str(payload.get("obs_encoding", "legacy"))
    if student_encoding != args.obs_encoding:
        raise ValueError(
            f"{student_path} uses obs_encoding={student_encoding}, "
            f"not {args.obs_encoding}"
        )
    input_dim = int(payload["input_dim"])
    expected_size = int(model.observation_space.shape[0])
    if input_dim != expected_size:
        raise ValueError(
            f"{student_path} has input_dim={input_dim}, "
            f"but this PPO env observes {expected_size}"
        )
    student_action_repeat = int(payload.get("action_repeat", 4))
    if student_action_repeat != args.action_repeat:
        raise ValueError(
            f"{student_path} was trained with action_repeat="
            f"{student_action_repeat}, not {args.action_repeat}"
        )
    hidden_sizes = [
        int(state["trunk.0.weight"].shape[0]),
        int(state["trunk.2.weight"].shape[0]),
    ]
    if hidden_sizes != list(args.net_arch):
        raise ValueError(
            f"{student_path} hidden sizes {hidden_sizes} do not match "
            f"--net-arch {list(args.net_arch)}"
        )

    device = model.policy.device
    policy_net = model.policy.mlp_extractor.policy_net
    with torch.no_grad():
        policy_net[0].weight.copy_(state["trunk.0.weight"].to(device))
        policy_net[0].bias.copy_(state["trunk.0.bias"].to(device))
        policy_net[2].weight.copy_(state["trunk.2.weight"].to(device))
        policy_net[2].bias.copy_(state["trunk.2.bias"].to(device))
        model.policy.action_net.weight.copy_(state["heads.weight"].to(device))
        model.policy.action_net.bias.copy_(state["heads.bias"].to(device))
    print(
        f"initialized policy from {student_path} "
        f"({input_dim} -> {hidden_sizes} -> 2); value head random",
        flush=True,
    )


def restore_frozen_encoder(model: PPO, context: str) -> None:
    """Reload a frozen autoencoder after SB3 rebuilds a policy."""
    extractor = getattr(model.policy, "features_extractor", None)
    if extractor is not None and hasattr(extractor, "reload_encoder"):
        extractor.reload_encoder()
        print(f"restored frozen autoencoder weights ({context})", flush=True)


def initialize_expanded_policy(
    model: PPO,
    old_model_path: Path,
    args: argparse.Namespace,
) -> None:
    """Warm-start a wider observation by zero-padding its first MLP layer.

    The sensor encoding appends features after the existing 754 values. Copying
    the old input columns preserves the old policy exactly at zero sensor input;
    the new columns start at zero and are learned during fine-tuning.
    """
    old_model = PPO.load(str(old_model_path), device=args.device)
    old_size = int(np.prod(old_model.observation_space.shape))
    new_size = int(np.prod(model.observation_space.shape))
    if new_size <= old_size:
        raise ValueError(
            f"--expand-from requires a wider observation: old={old_size}, "
            f"new={new_size}"
        )
    old_state = old_model.policy.state_dict()
    new_state = model.policy.state_dict()
    copied = 0
    expanded = 0
    for key, new_tensor in new_state.items():
        old_tensor = old_state.get(key)
        if old_tensor is None:
            continue
        if old_tensor.shape == new_tensor.shape:
            new_tensor.copy_(old_tensor)
            copied += 1
            continue
        if (
            old_tensor.ndim == 2
            and new_tensor.ndim == 2
            and old_tensor.shape[0] == new_tensor.shape[0]
            and new_tensor.shape[1] >= old_tensor.shape[1]
        ):
            new_tensor.zero_()
            new_tensor[:, : old_tensor.shape[1]].copy_(old_tensor)
            new_width = new_tensor.shape[1] - old_tensor.shape[1]
            if args.expand_init == "sample" and new_width > 0:
                generator = torch.Generator(device=old_tensor.device)
                generator.manual_seed(args.seed)
                sampled = torch.randint(
                    0,
                    old_tensor.shape[1],
                    (new_width,),
                    generator=generator,
                    device=old_tensor.device,
                )
                new_tensor[:, old_tensor.shape[1]:] = (
                    old_tensor[:, sampled] * float(args.expand_init_scale)
                )
            copied += 1
            expanded += 1
            continue
        print(
            f"expand-from: leaving {key} at initialization "
            f"({tuple(old_tensor.shape)} -> {tuple(new_tensor.shape)})",
            flush=True,
        )
    model.policy.load_state_dict(new_state)
    print(
        f"expanded policy {old_size} -> {new_size} observations "
        f"({copied} tensors copied, {expanded} input layers padded, "
        f"init={args.expand_init}, scale={args.expand_init_scale})",
        flush=True,
    )
    if args.baseline_eval:
        print("expand-from: using fine-tune, not from-scratch baseline", flush=True)
    return old_size


def freeze_expanded_input_columns(model: PPO, old_size: int) -> None:
    """Keep the original observation columns fixed during the canary."""
    def mask_old_columns(gradient: torch.Tensor) -> torch.Tensor:
        gradient[:, :old_size] = 0.0
        return gradient

    modules = (
        model.policy.mlp_extractor.policy_net[0],
        model.policy.mlp_extractor.value_net[0],
    )
    for module in modules:
        module.weight.register_hook(mask_old_columns)
    print(
        f"froze first {old_size} input columns; only appended features adapt",
        flush=True,
    )


def initialize_policy_head_from_checkpoint(
    model: PPO,
    head_path: Path,
    args: argparse.Namespace,
) -> None:
    """Warm-start the PPO policy/action head from a frozen-encoder BC run.

    The input dimension is ``latent_dim + 9``, so this is intentionally
    separate from ``--init-student`` (which expects the raw 58-value input).
    The value head stays randomly initialized.
    """
    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    if str(payload.get("obs_encoding", "legacy")) != args.obs_encoding:
        raise ValueError(
            f"{head_path} uses obs_encoding={payload.get('obs_encoding')}, "
            f"not {args.obs_encoding}"
        )
    features_dim = int(model.policy.features_dim)
    if int(payload["features_dim"]) != features_dim:
        raise ValueError(
            f"{head_path} expects features_dim={payload['features_dim']}, "
            f"but the policy produces {features_dim}"
        )
    if list(payload.get("net_arch", [])) != list(args.net_arch):
        raise ValueError(
            f"{head_path} hidden sizes {payload.get('net_arch')} do not match "
            f"--net-arch {list(args.net_arch)}"
        )
    if str(payload.get("activation", "tanh")).lower() != "tanh":
        raise ValueError(
            f"{head_path} was trained with {payload.get('activation')} activation; "
            "the frozen-autoencoder PPO head uses tanh"
        )

    state = payload["state_dict"]
    policy_net = model.policy.mlp_extractor.policy_net
    device = model.policy.device
    with torch.no_grad():
        policy_net[0].weight.copy_(state["0.weight"].to(device))
        policy_net[0].bias.copy_(state["0.bias"].to(device))
        policy_net[2].weight.copy_(state["2.weight"].to(device))
        policy_net[2].bias.copy_(state["2.bias"].to(device))
        model.policy.action_net.weight.copy_(state["4.weight"].to(device))
        model.policy.action_net.bias.copy_(state["4.bias"].to(device))
    print(
        f"initialized policy head from {head_path} "
        f"({features_dim} -> {list(args.net_arch)} -> 2); value head random",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.levels = (
        [args.level] if args.level else parse_levels(args.levels)
    )
    if args.init_head is not None:
        args.init_head = args.init_head.expanduser().resolve()
    if args.n_envs is None:
        args.n_envs = len(args.levels) if len(args.levels) > 1 else 8
    if len(args.levels) > 1 and args.n_envs < len(args.levels):
        raise ValueError(
            "--n-envs must cover every selected level; "
            f"got {args.n_envs} for {len(args.levels)} levels"
        )

    args.output_dir = args.output_dir.resolve()
    (args.output_dir / "godot").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval").mkdir(parents=True, exist_ok=True)

    if tensorboard_available():
        tensorboard_log = args.tensorboard_log
        if tensorboard_log is None:
            tensorboard_log = args.output_dir / "tensorboard"
        args.tensorboard_log = tensorboard_log.expanduser().resolve()
        args.tensorboard_log.mkdir(parents=True, exist_ok=True)
        print(f"tensorboard log: {args.tensorboard_log}", flush=True)
    else:
        args.tensorboard_log = None
        print(
            "tensorboard not installed; training will continue without it",
            flush=True,
        )

    args.spawn_pool_data: list[dict] = []
    if args.spawn_pool is not None:
        if args.agents_per_process < 2:
            raise ValueError("--spawn-pool requires --agents-per-process >= 2")
        args.spawn_pool_data = load_spawn_pool(args.spawn_pool.expanduser().resolve())
        print(
            f"spawn pool: {len(args.spawn_pool_data)} checkpoints from "
            f"{args.spawn_pool}",
            flush=True,
        )

    args.start_pool_data: list[dict] = []
    if args.start_pool is not None:
        if args.agents_per_process != 1:
            raise ValueError("--start-pool requires --agents-per-process 1")
        args.start_pool_data = load_start_pool(
            args.start_pool.expanduser().resolve()
        )
        print(
            f"start pool: {len(args.start_pool_data)} weighted starts from "
            f"{args.start_pool}",
            flush=True,
        )

    env_path = resolve_env_path()
    model_path = args.output_dir / (
        "ppo_all.zip" if len(args.levels) > 1 else "ppo_pilot.zip"
    )
    total_agents = args.n_envs * args.agents_per_process
    print(
        f"PPO levels={','.join(args.levels)} processes={args.n_envs} "
        f"agents_per_process={args.agents_per_process} "
        f"total_agents={total_agents} timesteps={args.timesteps} "
        f"action_repeat={args.action_repeat} grid_encoder={args.grid_encoder}",
        flush=True,
    )

    if args.eval_only:
        path = args.resume or model_path
        if not path.exists():
            raise FileNotFoundError(path)
        model = PPO.load(str(path), device=args.device)
        restore_frozen_encoder(model, "eval-only")
        evaluate_levels(model, args, env_path)
        return

    vec_env = build_vec_env(args, env_path)
    if args.resume:
        model = PPO.load(str(args.resume), env=vec_env, device=args.device)
        restore_frozen_encoder(model, "resume")
        if args.tensorboard_log is not None:
            model.set_logger(
                configure_logger(
                    args.verbose,
                    str(args.tensorboard_log),
                    tb_log_name="PPO",
                    reset_num_timesteps=True,
                )
            )
        # Saved PPO hyperparameters win over CLI defaults on load. Repair runs
        # deliberately change the exploration pressure and learning rate to
        # sharpen or re-explore a stalled per-level teacher.
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _progress: args.learning_rate
        # Resume overrides: SB3 restores the saved n_epochs, so a resumed run
        # would otherwise ignore --epochs. This is what lets a checkpoint be
        # extended with more PPO passes without rebuilding the policy.
        model.n_epochs = args.epochs
        for group in model.policy.optimizer.param_groups:
            group["lr"] = args.learning_rate
        print(
            f"resume overrides: n_epochs={model.n_epochs} "
            f"lr={args.learning_rate} ent_coef={args.ent_coef}",
            flush=True,
        )
        reset_num_timesteps = False
        # SB3 adds the loaded counter when reset_num_timesteps=False.
        # Passing an already-offset target would double-count the offset.
        target_timesteps = args.timesteps
    else:
        # StudentPolicy uses ReLU. A warm-started PPO net must use the same
        # activation or the copied weights mean a different function.
        activation_fn = (
            torch.nn.ReLU
            if args.init_student is not None
            else torch.nn.Tanh
        )
        if args.grid_encoder == "legacy_cnn":
            if args.obs_encoding != "legacy":
                raise ValueError("--grid-encoder=legacy_cnn requires --obs-encoding=legacy")
            if args.init_student is not None:
                raise ValueError("--grid-encoder=legacy_cnn cannot use --init-student")
            if args.init_head is not None:
                raise ValueError("--grid-encoder=legacy_cnn cannot use --init-head")
        if args.grid_encoder == "legacy_autoencoder":
            if args.obs_encoding != "legacy":
                raise ValueError(
                    "--grid-encoder=legacy_autoencoder requires --obs-encoding=legacy"
                )
            if args.autoencoder_path is None:
                raise ValueError(
                    "--grid-encoder=legacy_autoencoder requires --autoencoder-path"
                )
            if args.init_student is not None:
                raise ValueError(
                    "--grid-encoder=legacy_autoencoder cannot use --init-student"
                )
            args.autoencoder_path = args.autoencoder_path.expanduser().resolve()
        elif args.grid_encoder == "semantic_autoencoder":
            if args.obs_encoding != "multi_hot":
                raise ValueError(
                    "--grid-encoder=semantic_autoencoder requires "
                    "--obs-encoding=multi_hot"
                )
            if args.autoencoder_path is None:
                raise ValueError(
                    "--grid-encoder=semantic_autoencoder requires --autoencoder-path"
                )
            if args.init_student is not None:
                raise ValueError(
                    "--grid-encoder=semantic_autoencoder cannot use --init-student"
                )
            args.autoencoder_path = args.autoencoder_path.expanduser().resolve()
        elif args.init_head is not None:
            raise ValueError(
                "--init-head requires a frozen autoencoder grid encoder"
            )
        policy_kwargs = {
            "net_arch": dict(pi=args.net_arch, vf=args.net_arch),
            "activation_fn": activation_fn,
        }
        if args.grid_encoder == "legacy_cnn":
            policy_kwargs["features_extractor_class"] = LegacyGridExtractor
            policy_kwargs["features_extractor_kwargs"] = {
                "features_dim": 128,
                "embedding_dim": 8,
            }
        elif args.grid_encoder == "legacy_autoencoder":
            policy_kwargs["features_extractor_class"] = FrozenLegacyAutoencoderExtractor
            policy_kwargs["features_extractor_kwargs"] = {
                "checkpoint_path": str(args.autoencoder_path),
            }
        elif args.grid_encoder == "semantic_autoencoder":
            policy_kwargs["features_extractor_class"] = FrozenSemanticAutoencoderExtractor
            policy_kwargs["features_extractor_kwargs"] = {
                "checkpoint_path": str(args.autoencoder_path),
            }
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device=args.device,
            tensorboard_log=(
                str(args.tensorboard_log) if args.tensorboard_log else None
            ),
            verbose=args.verbose,
        )
        reset_num_timesteps = True
        target_timesteps = args.timesteps
        restore_frozen_encoder(model, "new policy")

    if args.expand_from is not None:
        if args.resume is not None:
            raise ValueError("--expand-from cannot be combined with --resume")
        if args.init_student is not None or args.init_head is not None:
            raise ValueError("--expand-from cannot be combined with --init-*")
        args.expand_from = args.expand_from.expanduser().resolve()
        old_size = initialize_expanded_policy(model, args.expand_from, args)
        if args.freeze_old_input:
            freeze_expanded_input_columns(model, old_size)
    elif args.freeze_old_input:
        raise ValueError("--freeze-old-input requires --expand-from")

    if args.init_student is not None:
        if args.resume is not None:
            raise ValueError("--init-student cannot be combined with --resume")
        initialize_policy_from_student(model, args.init_student, args)
        args.baseline_eval = True

    if args.init_head is not None:
        if args.resume is not None:
            raise ValueError("--init-head cannot be combined with --resume")
        initialize_policy_head_from_checkpoint(model, args.init_head, args)
        args.baseline_eval = True

    if args.spawn_pool is not None and args.spawn_learning_rate is not None:
        model.learning_rate = args.spawn_learning_rate
        if hasattr(model, "_setup_lr_schedule"):
            model._setup_lr_schedule()
        print(
            f"spawn fine-tune learning rate: {args.spawn_learning_rate}",
            flush=True,
        )

    callbacks = [EpisodeCSVCallback(args)]
    spawn_callback: SpawnEvalCallback | None = None
    if args.spawn_pool is not None and args.spawn_eval_every > 0:
        spawn_callback = SpawnEvalCallback(args, env_path)
        callbacks.append(spawn_callback)
    if args.eval_every > 0 and args.patience > 0:
        callbacks.append(EvalPatienceCallback(args, env_path))
    callback = CallbackList(callbacks)
    try:
        model.learn(
            total_timesteps=target_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        best_path = args.output_dir / "best_model.zip"
        if args.eval_every > 0 and args.patience > 0 and best_path.exists():
            model = PPO.load(str(best_path), device=args.device)
            restore_frozen_encoder(model, "best checkpoint")
            print(f"using best eval checkpoint: {best_path}", flush=True)
        spawn_best = args.output_dir / "spawn_best.zip"
        if spawn_callback is not None and spawn_best.exists():
            model = PPO.load(
                str(spawn_best),
                env=vec_env,
                device=args.device,
            )
            restore_frozen_encoder(model, "spawn best checkpoint")
            print(f"using best spawn checkpoint: {spawn_best}", flush=True)
        model.save(str(model_path))
        print(f"saved model: {model_path}", flush=True)
    finally:
        vec_env.close()

    if not args.skip_eval:
        evaluate_levels(model, args, env_path)


if __name__ == "__main__":
    main()
