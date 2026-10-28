"""One Godot training world exposed as a standard Gymnasium environment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .godot import GodotEnv


RESET_STATE_KEYS = (
    "position",
    "velocity",
    "state_name",
    "powerup_stack",
    "run_time",
    "best_run_time",
    "run_active",
    "started_walking",
    "world_state",
)


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
        preprocessor: Any | None = None,
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
        self.preprocessor = preprocessor
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
        raw_space = self.game.observation_space.spaces["obs"]
        if self.preprocessor is None:
            self.observation_space = raw_space
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(int(self.preprocessor.output_size),),
                dtype=np.float32,
            )
        self.action_space = spaces.Discrete(2)

    def _obs(self, obs_list):
        observation = np.asarray(obs_list[0]["obs"], dtype=np.float32)
        if self.preprocessor is not None:
            observation = np.asarray(
                self.preprocessor.transform(observation), dtype=np.float32
            )
        return observation

    def _augment_obs(self, observation: np.ndarray) -> np.ndarray:
        """Training-only structured dropout for the sensor adaptation run."""
        if self.preprocessor is not None:
            return observation
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

    @staticmethod
    def _state_start(spec: dict) -> bool:
        return any(key in spec for key in RESET_STATE_KEYS)

    @staticmethod
    def _spawn_from_start(spec: dict) -> dict:
        spawn = {
            "agent": 0,
            "cell": list(spec["cell"]),
            "facing": int(spec.get("facing", 1)),
        }
        for key in RESET_STATE_KEYS:
            if key in spec:
                spawn[key] = spec[key]
        return spawn

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
            state_start = not prefix and self._state_start(spec)
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
            elif state_start:
                obs, infos = self.game.reset(
                    spawns=[self._spawn_from_start(spec)]
                )
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

            start_jump = bool(
                self.force_start_jump and not prefix and not state_start
            )
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
                    else "state"
                    if state_start
                    else "canonical"
                    if spec.get("canonical")
                    else "cell"
                ),
                "start_kind": (
                    "prefix"
                    if prefix
                    else "state"
                    if state_start
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
                "state_start": state_start,
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

    def request_info(self, keys, agents=None):
        """Forward an on-demand info request to the single Godot world."""
        return self.game.request_info(keys, agents=agents)

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

        info["finished"] = bool(
            done and not truncated and info.get("finished", False)
        )
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
