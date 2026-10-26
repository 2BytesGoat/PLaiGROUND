"""PPO pilot for Dragon Jump.

The game's training mode exposes isolated worlds over the godot-rl socket.
This script runs one world per Godot process, wraps each world as a normal
Gymnasium environment, and lets Stable-Baselines3 PPO train one shared policy.

Why one process per world: the multi-agent wrapper is designed for parallel
inference/evaluation. It does not auto-reset individual agents, which is
required by SB3's vector-env API. One agent per process gives us the standard
reset/step contract without changing the game's training protocol.

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
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from dotenv import load_dotenv

from wrappers.godot_env import GodotEnv


DEFAULT_LEVEL = "1-3"
LEVELS = [f"1-{i}" for i in range(1, 17)]
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / "environments"


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
    ) -> None:
        super().__init__()
        self.level = level
        self.max_decisions = max_decisions
        self.action_repeat = action_repeat
        self.force_start_jump = force_start_jump
        self._steps = 0
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
        )
        self.observation_space = self.game.observation_space.spaces["obs"]
        self.action_space = spaces.Discrete(2)

    def _obs(self, obs_list):
        return np.asarray(obs_list[0]["obs"], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, infos = self.game.reset()
        self._steps = 0
        return self._obs(obs), (infos[0] if infos else {})

    def step(self, action):
        # Diagnostic-only override. Normal training and evaluation leave the
        # start action to the policy: the first jump starts the run.
        if self.force_start_jump and self._steps == 0:
            action = 1
        actions = np.asarray([[int(action)]], dtype=np.int64)
        # GodotEnv.step exposes the raw 5-tuple. The multi-agent SB3 wrapper
        # collapses terminated/truncated into a single done, but here we keep
        # them separate so Gymnasium/SB3 can handle timeout bootstrapping.
        obs, rewards, dones, _timeouts, infos = self.game.step(actions)
        obs = self._obs(obs)
        reward = float(rewards[0])
        done = bool(dones[0])
        info = dict(infos[0] if infos else {})

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
        # Do not override actions inside PPO rollouts: SB3 stores the sampled
        # action, so forcing a different env action would corrupt that sample.
        force_start_jump=False,
    )


def build_vec_env(args: argparse.Namespace, env_path: str):
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
        info_keywords=("level", "finished", "timed_out", "run_time"),
    )


class EpisodeCSVCallback(BaseCallback):
    """Write one row per completed training episode and save checkpoints."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.started = time.time()
        self.episodes = 0
        self.last_save = 0
        self.csv_file = open(args.output_dir / "episodes.csv", "w", newline="")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timesteps",
                "level",
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
                "episode_return": float(episode["r"]),
                "episode_length": int(episode["l"]),
                "finished": bool(info.get("finished", False)),
                "timed_out": timed_out,
                "run_time": run_time,
                "elapsed_s": round(time.time() - self.started, 2),
            }
            self.writer.writerow(row)
            self.csv_file.flush()
            print(
                f"[ep {self.episodes}] t={row['timesteps']} level={row['level']} "
                f"reward={row['episode_return']:.2f} steps={row['episode_length']} "
                f"finish={row['finished']} timeout={row['timed_out']} "
                f"time={row['run_time']}",
                flush=True,
            )

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

    def _on_training_end(self) -> None:
        self.csv_file.flush()
        self.csv_file.close()


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

        for step in range(args.max_decisions):
            actions = []
            for i in range(n):
                if done_once[i]:
                    actions.append([0])
                    continue
                if args.force_start_jump and step == 0:
                    actions.append([1])
                    continue
                action, _ = model.predict(
                    obs[i]["obs"],
                    deterministic=not args.eval_stochastic,
                )
                actions.append([int(action)])

            obs, rewards, dones, _timeouts, infos = env.step(
                np.asarray(actions, dtype=np.int64)
            )
            for i in range(n):
                if done_once[i]:
                    continue
                rewards_sum[i] += float(rewards[i])
                steps_run[i] = step + 1
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
            rows.append(
                {
                    "agent": i,
                    "finished": bool(finished[i]),
                    "steps": int(steps_run[i]),
                    "reward": float(rewards_sum[i]),
                    "run_time": (
                        "" if not finished[i] else f"{finish_times[i]:.6f}"
                    ),
                }
            )

        finish_count = int(finished.sum())
        death_count = int((done_once & ~finished).sum())
        timeout_count = int((~done_once).sum())
        best_time = float(np.nanmin(finish_times)) if finish_count else None
        print(
            f"[eval {level}] finishes={finish_count}/{n} "
            f"deaths={death_count} timeouts={timeout_count} "
            f"best={best_time if best_time is not None else 'none'} "
            f"mean_reward={float(rewards_sum.mean()):.2f}",
            flush=True,
        )

        with open(out_dir / f"results-{run_id}.csv", "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["agent", "finished", "steps", "reward", "run_time"],
            )
            writer.writeheader()
            writer.writerows(rows)

        return {
            "level": level,
            "finishes": finish_count,
            "deaths": death_count,
            "timeouts": timeout_count,
            "best_time": f"{best_time:.6f}" if best_time is not None else "",
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
                "finishes",
                "deaths",
                "timeouts",
                "best_time",
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
        "--force-start-jump",
        dest="force_start_jump",
        action="store_true",
        help="Diagnostic only: force the first action to jump. Default is policy-owned.",
    )
    parser.add_argument(
        "--no-force-start-jump",
        dest="force_start_jump",
        action="store_false",
        help="Explicitly keep the first action policy-owned (default).",
    )
    parser.set_defaults(force_start_jump=False)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "ppo_pilot",
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


def main() -> None:
    args = parse_args()
    args.levels = (
        [args.level] if args.level else parse_levels(args.levels)
    )
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

    env_path = resolve_env_path()
    model_path = args.output_dir / (
        "ppo_all.zip" if len(args.levels) > 1 else "ppo_pilot.zip"
    )
    print(
        f"PPO levels={','.join(args.levels)} n_envs={args.n_envs} "
        f"timesteps={args.timesteps} action_repeat={args.action_repeat}",
        flush=True,
    )

    if args.eval_only:
        path = args.resume or model_path
        if not path.exists():
            raise FileNotFoundError(path)
        model = PPO.load(str(path), device=args.device)
        evaluate_levels(model, args, env_path)
        return

    vec_env = build_vec_env(args, env_path)
    if args.resume:
        model = PPO.load(str(args.resume), env=vec_env, device=args.device)
        # Saved PPO hyperparameters win over CLI defaults on load. Repair runs
        # deliberately change the exploration pressure and learning rate to
        # sharpen or re-explore a stalled per-level teacher.
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _progress: args.learning_rate
        for group in model.policy.optimizer.param_groups:
            group["lr"] = args.learning_rate
        reset_num_timesteps = False
        # SB3 adds the loaded counter when reset_num_timesteps=False.
        # Passing an already-offset target would double-count the offset.
        target_timesteps = args.timesteps
    else:
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
            policy_kwargs={"net_arch": dict(pi=args.net_arch, vf=args.net_arch)},
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )
        reset_num_timesteps = True
        target_timesteps = args.timesteps

    callback = EpisodeCSVCallback(args)
    try:
        model.learn(
            total_timesteps=target_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        model.save(str(model_path))
        print(f"saved model: {model_path}", flush=True)
    finally:
        vec_env.close()

    if not args.skip_eval:
        evaluate_levels(model, args, env_path)


if __name__ == "__main__":
    main()
