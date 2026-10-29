"""Train a DQN policy from an experiment manifest.

Route extraction and evaluation are separate tools:

    python tools/extract_route_starts.py --config experiments/0003_...
    python tools/evaluate.py --config experiments/0003_... --model <checkpoint>

The manifest owns the level, environment shape, observation encoding, DQN
hyperparameters, checkpoint cadence, and data paths.
"""

from __future__ import annotations

import pickle

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from plaiground.envs.factory import build_vec_env
from plaiground.envs.runtime import resolve_env_path
from plaiground.experiments.runner import (
    Run,
    activation_module,
    parse_train_args,
    resolve_path,
    run_from_args,
    train_start_pool,
)
from plaiground.observations.lean_io import (
    load_lean_preprocessor,
    prepare_lean_preprocessor,
)
from plaiground.observations.lean_stick import LeanStickPreprocessor
from plaiground.algorithms.reinforcement.replay import (
    ProtectedReplayBuffer,
    RareReplayBuffer,
    pad_anchor_observations,
)


def make_model(run: Run, vec_env, model_cls=DQN):
    algorithm = run.algorithm
    policy_kwargs = {
        "net_arch": run.hidden,
        "activation_fn": activation_module(run.activation),
    }
    return model_cls(
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
        n_steps=int(algorithm.get("n_steps", 1)),
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


def _copy_expanded_policy(old_model, new_model) -> None:
    """Copy an old policy into a wider input layer, zero-filling new columns."""
    old_state = old_model.policy.state_dict()
    new_state = new_model.policy.state_dict()
    for name, old_tensor in old_state.items():
        target = new_state.get(name)
        if target is None:
            continue
        if tuple(target.shape) == tuple(old_tensor.shape):
            target.copy_(old_tensor)
            continue
        if (
            target.ndim == 2
            and old_tensor.ndim == 2
            and target.shape[0] == old_tensor.shape[0]
            and target.shape[1] > old_tensor.shape[1]
        ):
            target.zero_()
            target[:, : old_tensor.shape[1]].copy_(old_tensor)
            continue
        raise ValueError(
            f"cannot expand policy tensor {name!r}: "
            f"{tuple(old_tensor.shape)} -> {tuple(target.shape)}"
        )
    new_model.policy.load_state_dict(new_state)


def load_resumed_model(run: Run, vec_env, model_cls=DQN):
    checkpoint = model_cls.load(str(run.resume), env=None, device=run.device)
    old_shape = getattr(checkpoint.observation_space, "shape", None)
    new_shape = getattr(vec_env.observation_space, "shape", None)
    if old_shape != new_shape:
        model = make_model(run, vec_env, model_cls)
        _copy_expanded_policy(checkpoint, model)
        reset_timesteps = bool(
            run.config.get("run", {}).get("reset_timesteps", True)
        )
        model.num_timesteps = 0 if reset_timesteps else checkpoint.num_timesteps
        print(
            f"expanded {model_cls.__name__} input "
            f"{old_shape} -> {new_shape}; copied old weights, "
            f"zero-initialized {int(new_shape[0]) - int(old_shape[0])} new inputs "
            f"(timesteps={model.num_timesteps})",
            flush=True,
        )
        return model
    model = model_cls.load(str(run.resume), env=vec_env, device=run.device)
    print(
        f"resumed {model_cls.__name__} checkpoint: {run.resume} "
        f"(timesteps={model.num_timesteps})",
        flush=True,
    )
    return model


def run_training(
    run: Run,
    *,
    model_cls=DQN,
    checkpoint_prefix: str = "dqn",
) -> None:
    run.ensure_dirs()
    env_path = resolve_env_path()
    start_pool = train_start_pool(run, required=True)
    if run.preprocessing in {"lean", "lean_stick"}:
        if run.preprocessing_path.exists():
            preprocessor = load_lean_preprocessor(run.preprocessing_path)
            if run.preprocessing == "lean_stick" and not isinstance(
                preprocessor, LeanStickPreprocessor
            ):
                preprocessor = LeanStickPreprocessor(preprocessor)
            print(
                f"preprocessing: loaded {run.preprocessing} pipeline "
                f"{run.preprocessing_path} ({preprocessor.output_size} inputs)",
                flush=True,
            )
        else:
            preprocessor = prepare_lean_preprocessor(run, env_path)
    else:
        preprocessor = None
    vec_env = build_vec_env(
        run, env_path, start_pool, preprocessor=preprocessor
    )

    print(
        f"{model_cls.__name__} levels={','.join(run.levels)} "
        f"envs={run.total_envs} "
        f"(processes={run.n_envs}, agents_per_process={run.agents_per_process}) "
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
        if run.resume is not None:
            model = load_resumed_model(run, vec_env, model_cls)
        else:
            model = make_model(run, vec_env, model_cls)

        buffer_value = run.config.get("run", {}).get("replay_buffer_from")
        anchor_value = run.config.get("run", {}).get("anchor_buffer_from")
        if buffer_value and anchor_value:
            raise ValueError(
                "use either replay_buffer_from or anchor_buffer_from, not both"
            )
        if buffer_value:
            buffer_path = resolve_path(buffer_value)
            with buffer_path.open("rb") as handle:
                model.replay_buffer = pickle.load(handle)
            print(
                f"loaded replay buffer: {buffer_path} "
                f"({model.replay_buffer.size() * model.replay_buffer.n_envs} transitions)",
                flush=True,
            )
        if anchor_value:
            anchor_path = resolve_path(anchor_value)
            with anchor_path.open("rb") as handle:
                anchor = pickle.load(handle)
            if not isinstance(anchor, RareReplayBuffer):
                raise ValueError(
                    f"anchor buffer must be a RareReplayBuffer: {anchor_path}"
                )
            anchor = pad_anchor_observations(
                anchor, vec_env.observation_space
            )
            model.replay_buffer = ProtectedReplayBuffer(
                online_size=int(
                    run.algorithm.get(
                        "online_buffer_size",
                        run.algorithm.get("buffer_size", 400_000),
                    )
                ),
                observation_space=vec_env.observation_space,
                action_space=vec_env.action_space,
                device=model.device,
                n_envs=run.total_envs,
                anchor_buffer=anchor,
                anchor_fraction=float(
                    run.algorithm.get("anchor_fraction", 0.25)
                ),
            )
            stats = model.replay_buffer.stats()
            print(
                f"installed protected replay: anchor {anchor_path} "
                f"({anchor.size() * anchor.n_envs} transitions), "
                f"online_size={model.replay_buffer.online.buffer_size * model.replay_buffer.online.n_envs}, "
                f"anchor_fraction={stats['anchor_fraction']}",
                flush=True,
            )

        # When reset_timesteps is false, SB3 treats total_timesteps as the
        # additional budget and adds the checkpoint's current step count.
        reset_timesteps = bool(
            run.config.get("run", {}).get("reset_timesteps", True)
        )
        checkpoint = CheckpointCallback(
            save_freq=max(1, run.checkpoint_freq // run.total_envs),
            save_path=str(run.checkpoint_dir),
            name_prefix=checkpoint_prefix,
        )
        model.learn(
            total_timesteps=run.timesteps,
            callback=checkpoint,
            reset_num_timesteps=reset_timesteps,
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


# Backward-compatible private names used by older callers/tests.
_make_model = make_model
_load_resumed_model = load_resumed_model


def main() -> None:
    run = run_from_args(parse_train_args(__doc__ or "Train DQN."))
    run_training(run)


if __name__ == "__main__":
    main()
