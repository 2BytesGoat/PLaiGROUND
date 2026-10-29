"""Train a PPO policy from an experiment manifest.

The manifest owns the level, environment shape, observation encoding, PPO
hyperparameters, checkpoint cadence, and output paths. The entry point only
exposes run-level overrides such as ``--config``, ``--resume``, and
``--timesteps``.

Evaluation is a separate step:

    python tools/evaluate.py --config experiments/0002_... --model <checkpoint>
"""

from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import configure_logger

from plaiground.envs.factory import build_vec_env
from plaiground.envs.runtime import resolve_env_path
from plaiground.experiments.runner import (
    Run,
    activation_module,
    parse_train_args,
    run_from_args,
    train_start_pool,
)
from plaiground.observations.lean_io import prepare_lean_preprocessor


def _make_model(run: Run, vec_env):
    algorithm = run.algorithm
    policy_kwargs = {
        "net_arch": {"pi": run.hidden, "vf": run.hidden},
        "activation_fn": activation_module(run.activation),
    }
    return PPO(
        "MlpPolicy",
        vec_env,
        n_steps=int(algorithm.get("n_steps", 150)),
        batch_size=int(algorithm.get("batch_size", 120)),
        n_epochs=int(algorithm.get("epochs", 10)),
        learning_rate=float(algorithm.get("learning_rate", 3e-4)),
        gamma=float(algorithm.get("gamma", 0.99)),
        gae_lambda=float(algorithm.get("gae_lambda", 0.95)),
        clip_range=float(algorithm.get("clip_range", 0.2)),
        ent_coef=float(algorithm.get("ent_coef", 0.01)),
        vf_coef=float(algorithm.get("vf_coef", 0.5)),
        max_grad_norm=float(algorithm.get("max_grad_norm", 0.5)),
        target_kl=float(algorithm.get("target_kl", 0.03)),
        policy_kwargs=policy_kwargs,
        seed=run.seed,
        device=run.device,
        tensorboard_log=(
            str(run.tensorboard_log) if run.tensorboard_log else None
        ),
        verbose=run.verbose,
    )


def _load_resumed_model(run: Run, vec_env):
    model = PPO.load(str(run.resume), env=vec_env, device=run.device)
    if run.tensorboard_log is not None:
        model.set_logger(
            configure_logger(
                run.verbose,
                str(run.tensorboard_log),
                tb_log_name="PPO",
                reset_num_timesteps=False,
            )
        )
    # Explicit run-manifest values override the checkpoint. This lets a
    # continuation test a different rollout/batch shape without rebuilding
    # the policy or preprocessing from scratch.
    algorithm = run.algorithm
    learning_rate = float(
        algorithm.get("learning_rate", model.learning_rate)
    )
    model.n_steps = int(algorithm.get("n_steps", model.n_steps))
    model.batch_size = int(algorithm.get("batch_size", model.batch_size))
    model.n_epochs = int(algorithm.get("epochs", model.n_epochs))
    model.ent_coef = float(algorithm.get("ent_coef", model.ent_coef))
    model.learning_rate = learning_rate
    target_kl = algorithm.get("target_kl")
    model.target_kl = None if target_kl is None else float(target_kl)
    # Close over the float, not the model: a lambda that reads model attributes
    # makes the saved checkpoint contain a self-reference and fails to pickle.
    model.lr_schedule = lambda _progress: learning_rate
    # The loaded rollout buffer has the old n_steps/n_envs shape. Recreate it
    # with the continuation shape; SB3 does not do this automatically on load.
    buffer_class = model.rollout_buffer_class or RolloutBuffer
    model.rollout_buffer = buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=vec_env.num_envs,
        **(model.rollout_buffer_kwargs or {}),
    )
    # Force the next learn() to reset against the new environment shape.
    model._last_obs = None
    model._last_episode_starts = None
    for group in model.policy.optimizer.param_groups:
        group["lr"] = learning_rate
    print(
        f"resume overrides: n_steps={model.n_steps} "
        f"batch_size={model.batch_size} n_epochs={model.n_epochs} "
        f"lr={model.learning_rate} ent_coef={model.ent_coef} "
        f"target_kl={model.target_kl}",
        flush=True,
    )
    return model


def main() -> None:
    run = run_from_args(parse_train_args(__doc__ or "Train PPO."))
    run.ensure_dirs()
    env_path = resolve_env_path()
    preprocessor = prepare_lean_preprocessor(run, env_path)
    start_pool = train_start_pool(run)
    vec_env = build_vec_env(run, env_path, start_pool, preprocessor)

    print(
        f"PPO levels={','.join(run.levels)} envs={run.total_envs} "
        f"(processes={run.n_envs}, agents_per_process={run.agents_per_process}) "
        f"timesteps={run.timesteps} action_repeat={run.action_repeat} "
        f"obs={run.obs_encoding} output={run.output_dir}",
        flush=True,
    )
    if run.tensorboard_log is not None:
        print(f"tensorboard log: {run.tensorboard_log}", flush=True)
    else:
        print("tensorboard not installed; training continues without it", flush=True)

    try:
        if run.resume is not None:
            model = _load_resumed_model(run, vec_env)
            reset_num_timesteps = False
        else:
            model = _make_model(run, vec_env)
            reset_num_timesteps = True

        remaining = max(0, run.timesteps - int(model.num_timesteps))
        if remaining == 0:
            print(
                f"target already reached: {model.num_timesteps} >= "
                f"{run.timesteps}",
                flush=True,
            )
        else:
            checkpoint = CheckpointCallback(
                save_freq=max(1, run.checkpoint_freq // run.total_envs),
                save_path=str(run.checkpoint_dir),
                name_prefix="ppo",
            )
            model.learn(
                total_timesteps=remaining,
                callback=checkpoint,
                reset_num_timesteps=reset_num_timesteps,
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
