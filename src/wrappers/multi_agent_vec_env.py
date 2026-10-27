"""SB3 vector environment for N agents per Godot process.

One Godot process can host many isolated TrainingWorlds. This wrapper exposes
those agents as independent virtual environments, including per-agent
auto-reset, while keeping one shared PPO policy in the parent process.

The game's reset protocol supports an optional `agents` index list. That is
what makes per-agent auto-reset possible; the older wrapper only reset every
world together.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from wrappers.godot_env import GodotEnv


class MultiAgentGodotVecEnv(VecEnv):
    """Expose ``n_agents_per_process * n_processes`` Godot agents as SB3 envs."""

    def __init__(
        self,
        env_path: str,
        level: str,
        n_agents_per_process: int,
        n_processes: int,
        seed: int = 42,
        action_repeat: int = 4,
        speedup: float = 8.0,
        max_decisions: int = 150,
        base_port: int = 13008,
        output_dir: Path | str = "runs/topology_bench",
        obs_encoding: str = "multi_hot",
        reward_profile: str = "time",
        show_window: bool = False,
        profile_training: bool = False,
        training_no_cosmetics: bool = False,
        spawn_pool: Optional[list[dict]] = None,
        spawn_seed: int = 0,
    ) -> None:
        if n_agents_per_process < 1 or n_processes < 1:
            raise ValueError("n_agents_per_process and n_processes must be >= 1")

        self.env_path = env_path
        self.level = level
        self.n_agents_per_process = int(n_agents_per_process)
        self.n_processes = int(n_processes)
        self.max_decisions = int(max_decisions)
        self.obs_encoding = obs_encoding
        self.reward_profile = reward_profile
        self._steps = [0] * (self.n_agents_per_process * self.n_processes)
        self._actions: Optional[np.ndarray] = None
        self._spawns: dict[int, dict] = {}
        self._spawn_pool: list[dict] = []
        self._spawn_rng = np.random.default_rng(spawn_seed)
        self.reset_count = 0

        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / "godot").mkdir(parents=True, exist_ok=True)
        extra_env_kwargs = {}
        if profile_training:
            extra_env_kwargs["profile_training"] = "true"
        if training_no_cosmetics:
            extra_env_kwargs["training_no_cosmetics"] = "true"
        self._envs: list[GodotEnv] = []
        for process in range(self.n_processes):
            child_env = _isolated_env(root, f"topology_train_{process}")
            env = GodotEnv(
                env_path=env_path,
                port=base_port + process,
                show_window=show_window,
                seed=seed + process,
                action_repeat=action_repeat,
                speedup=speedup,
                nb_agents=self.n_agents_per_process,
                level=level,
                convert_action_space=True,
                log_path=str(root / "godot" / f"topology_train_{process}.log"),
                env=child_env,
                reward_profile=reward_profile,
                obs_encoding=obs_encoding,
                **extra_env_kwargs,
            )
            self._envs.append(env)

        self.num_envs = self.n_agents_per_process * self.n_processes
        observation_space = self._envs[0].observation_space
        if isinstance(observation_space, spaces.Dict) and "obs" in observation_space.spaces:
            observation_space = observation_space.spaces["obs"]
        self.observation_space = observation_space
        self.action_space = _single_binary_action_space(self._envs[0].action_space)
        super().__init__(self.num_envs, self.observation_space, self.action_space)
        if spawn_pool:
            self.set_spawn_pool(spawn_pool, spawn_seed)

    @property
    def profiles(self) -> list[dict]:
        """Latest cumulative Godot profile block from each process."""
        return [env.last_profile for env in self._envs]

    def set_spawn_pool(self, spawns: list[dict], seed: int = 0) -> None:
        """Sample a fresh checkpoint for every episode reset.

        Entries use ``{"cell": [x, y], "facing": 1}``. The pool is sampled
        independently for initial resets and per-agent auto-resets; fixed
        ``set_spawns`` maps remain available for evaluation.
        """
        self._spawn_pool = [dict(spawn) for spawn in spawns]
        self._spawn_rng = np.random.default_rng(seed)
        self._spawns = {}

    def _sample_spawns(self, indices: range | list[int]) -> None:
        if not self._spawn_pool:
            return
        for index in indices:
            spawn = self._spawn_pool[
                int(self._spawn_rng.integers(0, len(self._spawn_pool)))
            ]
            if spawn.get("canonical"):
                self._spawns.pop(int(index), None)
            else:
                self._spawns[int(index)] = dict(spawn)

    def set_spawns(self, spawns: dict[int, dict]) -> None:
        """Set optional route checkpoints by global environment index.

        Entries use ``{"cell": [x, y], "facing": 1}``; indices not present use
        the level's canonical start. The low-level wrapper attaches the agent
        index when sending each process batch.
        """
        self._spawns = {
            int(index): dict(spawn) for index, spawn in spawns.items()
        }
        self._spawn_pool = []

    def clear_spawns(self) -> None:
        self._spawns = {}

    def _spawns_for_process(self, process: int, agents: Optional[list[int]] = None) -> list[dict]:
        per_process = self.n_agents_per_process
        selected = range(per_process) if agents is None else agents
        entries: list[dict] = []
        for agent in selected:
            global_index = process * per_process + agent
            spawn = self._spawns.get(global_index)
            if spawn is None:
                continue
            entries.append({**spawn, "agent": agent})
        return entries

    # ------------------------------------------------------------------
    # VecEnv API

    def reset(self) -> np.ndarray:
        observations: list[np.ndarray] = []
        self._sample_spawns(range(self.num_envs))
        for process, env in enumerate(self._envs):
            obs, _info = env.reset(
                spawns=self._spawns_for_process(process)
            )
            observations.extend(_flatten_obs(obs))
        self._steps = [0] * self.num_envs
        self._actions = None
        return np.asarray(observations, dtype=np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.int64).reshape(-1, 1)
        if actions.shape[0] != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} actions, got {actions.shape[0]}"
            )
        per_process = self.n_agents_per_process
        for process, env in enumerate(self._envs):
            start = process * per_process
            env.step_send(actions[start : start + per_process])
        self._actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        if self._actions is None:
            raise RuntimeError("step_wait called before step_async")

        all_obs: list[np.ndarray] = []
        all_rewards: list[float] = []
        all_dones: list[bool] = []
        all_infos: list[dict] = []

        per_process = self.n_agents_per_process
        for process, env in enumerate(self._envs):
            obs, rewards, dones, _timeouts, infos = env.step_recv()
            done_flags = [bool(value) for value in dones]

            for agent in range(per_process):
                index = process * per_process + agent
                self._steps[index] += 1
                info = dict(infos[agent]) if agent < len(infos) else {}
                truncated = False
                if not done_flags[agent] and self._steps[index] >= self.max_decisions:
                    done_flags[agent] = True
                    truncated = True
                if truncated:
                    info["TimeLimit.truncated"] = True
                elif done_flags[agent]:
                    info.setdefault("TimeLimit.truncated", False)
                info["level"] = self.level
                info["timed_out"] = bool(truncated)
                info["decision_steps"] = self._steps[index]
                info.setdefault("finished", False)
                infos[agent] = info

            reset_agents = [
                agent for agent, done in enumerate(done_flags) if done
            ]
            if reset_agents:
                self._sample_spawns([
                    process * per_process + agent for agent in reset_agents
                ])
                reset_obs, _reset_info = env.reset(
                    agents=reset_agents,
                    spawns=self._spawns_for_process(process, reset_agents),
                )
                for agent in reset_agents:
                    obs[agent] = reset_obs[agent]
                    self._steps[process * per_process + agent] = 0
                self.reset_count += len(reset_agents)

            all_obs.extend(_flatten_obs(obs))
            all_rewards.extend(float(value) for value in rewards)
            all_dones.extend(done_flags)
            all_infos.extend(infos)

        self._actions = None
        return (
            np.asarray(all_obs, dtype=np.float32),
            np.asarray(all_rewards, dtype=np.float32),
            np.asarray(all_dones, dtype=bool),
            all_infos,
        )

    def close(self) -> None:
        for env in self._envs:
            env.close()

    # ------------------------------------------------------------------
    # Optional VecEnv surface

    def env_is_wrapped(
        self, wrapper_class: type, indices: Optional[list[int]] = None
    ) -> list[bool]:
        return [False] * self.num_envs

    def get_attr(
        self, attr_name: str, indices: Optional[list[int]] = None
    ) -> list[Any]:
        if attr_name == "render_mode":
            return [None] * self.num_envs
        raise AttributeError(attr_name)

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: Optional[list[int]] = None,
    ) -> None:
        raise AttributeError(attr_name)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Optional[list[int]] = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        raise NotImplementedError(method_name)

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        return [seed] * self.num_envs


def _flatten_obs(obs_list: list[dict]) -> list[np.ndarray]:
    return [
        np.asarray(item["obs"], dtype=np.float32)
        for item in obs_list
    ]


def _single_binary_action_space(space: gym.Space) -> gym.Space:
    """PPO normally wants a scalar discrete action, not MultiDiscrete([2])."""
    if isinstance(space, spaces.MultiDiscrete) and len(space.nvec) == 1:
        # The game reports the jump head as size=1 (a button), which the
        # generic processor turns into MultiDiscrete([1]); the actual action
        # is binary idle/jump, so expose Discrete(2) to PPO.
        if int(space.nvec[0]) in (1, 2):
            return spaces.Discrete(2)
    if isinstance(space, spaces.Discrete) and int(space.n) <= 2:
        return spaces.Discrete(2)
    return space


def _isolated_env(root: Path, name: str) -> dict[str, str]:
    home = root / "userdata" / name
    home.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["HOME"] = str(home)
    child_env["XDG_DATA_HOME"] = str(home / "xdg_data")
    child_env["XDG_CONFIG_HOME"] = str(home / "xdg_config")
    return child_env
