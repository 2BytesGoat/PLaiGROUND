"""Shared environment construction for PPO and DQN.

Both trainers use one Godot process per world. The factory owns ports, the
private HOME trees, the start-pool filter, and the VecMonitor metadata so the
algorithm modules only deal with their model and training loop.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from plaiground.experiments.runner import Run

from .godot import GodotEnv
from .runtime import isolated_home
from .single_agent import SingleAgentGodotEnv


INFO_KEYWORDS = (
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
)


def make_env(
    run: Run,
    env_path: str,
    rank: int,
    level: str,
    start_pool: list[dict[str, Any]],
    preprocessor: Any | None,
) -> SingleAgentGodotEnv:
    env_config = run.env
    level_pool = [
        item
        for item in start_pool
        if item.get("level") in (None, level)
    ]
    GodotEnv.DEFAULT_TIMEOUT = run.timeout
    return SingleAgentGodotEnv(
        env_path=env_path,
        level=level,
        port=run.base_port + rank,
        seed=run.seed + rank,
        action_repeat=run.action_repeat,
        speedup=run.speedup,
        max_decisions=run.max_decisions,
        show_window=run.show,
        log_path=str(
            run.output_dir / "godot" / f"train_{rank}_{level}.log"
        ),
        child_env=isolated_home(run.output_dir, f"train_{rank}"),
        force_start_jump=bool(env_config.get("force_start_jump", False)),
        reward_profile=str(env_config.get("reward_profile", "default")),
        obs_encoding=run.obs_encoding,
        preprocessor=preprocessor,
        start_pool=level_pool,
        start_seed=int(env_config.get("start_seed", 0)) + rank,
    )


def build_vec_env(
    run: Run,
    env_path: str,
    start_pool: list[dict[str, Any]] | None = None,
    preprocessor: Any | None = None,
):
    levels = run.levels
    if run.n_envs < 1:
        raise ValueError("n_envs must be at least 1")
    if len(levels) > 1 and run.n_envs < len(levels):
        raise ValueError(
            "n_envs must cover every selected level; "
            f"got {run.n_envs} for {len(levels)} levels"
        )
    pool = start_pool or []
    if len(levels) > 1 and pool:
        if any(item.get("level") is None for item in pool):
            raise ValueError(
                "multi-level training needs a level-tagged start pool; "
                "build one pool per level or run one level at a time"
            )
        tagged_levels = {str(item.get("level")) for item in pool}
        missing = [level for level in levels if level not in tagged_levels]
        if missing:
            raise ValueError(
                "start pool has no entries for level(s): "
                f"{', '.join(missing)}"
            )

    (run.output_dir / "godot").mkdir(parents=True, exist_ok=True)
    run.monitor_path.parent.mkdir(parents=True, exist_ok=True)
    env_fns = [
        partial(
            make_env,
            run,
            env_path,
            rank,
            levels[rank % len(levels)],
            pool,
            preprocessor,
        )
        for rank in range(run.n_envs)
    ]
    if run.n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        # spawn is the portable macOS start method; only the env factory is
        # pickled, not the Godot socket.
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    return VecMonitor(
        vec_env,
        filename=str(run.monitor_path),
        info_keywords=INFO_KEYWORDS,
    )


def make_eval_env(
    run: Run,
    env_path: str,
    *,
    level: str,
    num_agents: int,
    port: int,
    log_name: str,
    home_name: str,
) -> GodotEnv:
    run.eval_dir.mkdir(parents=True, exist_ok=True)
    GodotEnv.DEFAULT_TIMEOUT = run.timeout
    return GodotEnv(
        env_path=env_path,
        port=port,
        show_window=run.show,
        seed=run.seed,
        action_repeat=run.action_repeat,
        speedup=run.speedup,
        nb_agents=num_agents,
        level=level,
        convert_action_space=True,
        log_path=str(run.eval_dir / log_name),
        env=isolated_home(run.output_dir, home_name),
        reward_profile=str(run.env.get("reward_profile", "default")),
        obs_encoding=run.obs_encoding,
    )
