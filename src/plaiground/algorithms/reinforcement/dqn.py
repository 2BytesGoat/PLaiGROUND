"""Train a DQN policy from an experiment manifest.

Route extraction and evaluation are separate tools:

    python tools/extract_route_starts.py --config experiments/0003_...
    python tools/evaluate.py --config experiments/0003_... --model <checkpoint>

The manifest owns the level, environment shape, observation encoding, DQN
hyperparameters, checkpoint cadence, and data paths.
"""

from __future__ import annotations

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from plaiground.envs.factory import build_vec_env
from plaiground.envs.runtime import resolve_env_path
from plaiground.experiments.runner import (
    Run,
    activation_module,
    parse_train_args,
    run_from_args,
    train_start_pool,
)


def _make_model(run: Run, vec_env):
    algorithm = run.algorithm
    policy_kwargs = {
        "net_arch": run.hidden,
        "activation_fn": activation_module(run.activation),
    }
    return DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=float(algorithm.get("learning_rate", 1e-4)),
        buffer_size=int(algorithm.get("buffer_size", 100_000)),
        learning_starts=int(algorithm.get("learning_starts", 2_000)),
        batch_size=int(algorithm.get("batch_size", 256)),
        tau=float(algorithm.get("tau", 1.0)),
        gamma=float(algorithm.get("gamma", 0.99)),
        train_freq=int(algorithm.get("train_freq", 4)),
        gradient_steps=int(algorithm.get("gradient_steps", 1)),
        target_update_interval=int(
            algorithm.get("target_update_interval", 1_000)
        ),
        exploration_fraction=float(
            algorithm.get("exploration_fraction", 0.25)
        ),
        exploration_initial_eps=float(
            algorithm.get("exploration_initial_eps", 1.0)
        ),
        exploration_final_eps=float(
            algorithm.get("exploration_final_eps", 0.05)
        ),
        policy_kwargs=policy_kwargs,
        tensorboard_log=(
            str(run.tensorboard_log) if run.tensorboard_log else None
        ),
        verbose=run.verbose,
        device=run.device,
        seed=run.seed,
    )


def main() -> None:
    run = run_from_args(parse_train_args(__doc__ or "Train DQN."))
    if run.resume is not None:
        raise SystemExit(
            "DQN resume is not supported yet; start a new run or load the "
            "checkpoint manually"
        )
    run.ensure_dirs()
    env_path = resolve_env_path()
    start_pool = train_start_pool(run, required=True)
    vec_env = build_vec_env(run, env_path, start_pool, preprocessor=None)

    print(
        f"DQN level={run.level} envs={run.n_envs} "
        f"timesteps={run.timesteps} action_repeat={run.action_repeat} "
        f"obs={run.obs_encoding} starts={len(start_pool)} "
        f"output={run.output_dir}",
        flush=True,
    )
    if run.tensorboard_log is not None:
        print(f"tensorboard log: {run.tensorboard_log}", flush=True)
    else:
        print("tensorboard not installed; training continues without it", flush=True)

    try:
        model = _make_model(run, vec_env)
        checkpoint = CheckpointCallback(
            save_freq=max(1, run.checkpoint_freq // run.n_envs),
            save_path=str(run.checkpoint_dir),
            name_prefix="dqn",
        )
        model.learn(
            total_timesteps=run.timesteps,
            callback=checkpoint,
            progress_bar=False,
        )
        model.save(str(run.model_path))
        print(f"saved model: {run.model_path}", flush=True)
    finally:
        vec_env.close()

    print(
        "evaluate with: python tools/evaluate.py "
        f"--config {run.config_path} --model {run.model_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
