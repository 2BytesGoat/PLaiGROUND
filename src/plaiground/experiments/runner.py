"""Resolve an experiment manifest into a flat run object.

The committed TOML manifests are the source of truth for an experiment. The
entry points intentionally expose only run-level overrides: which config to
use, the timestep budget, a resume checkpoint, the output directory, and the
usual display/device switches. Algorithm and environment details stay in the
manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import REPO, load_experiment


def resolve_path(value: str | Path, *, base: Path = REPO) -> Path:
    """Resolve a manifest path relative to the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def tensorboard_available() -> bool:
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class Run:
    """One resolved experiment run."""

    config: dict[str, Any]
    config_path: Path
    output_dir: Path
    timesteps: int
    resume: Path | None
    preprocessing_path: Path
    show: bool
    device: str
    verbose: int

    @property
    def env(self) -> dict[str, Any]:
        return self.config.get("env", {})

    @property
    def algorithm(self) -> dict[str, Any]:
        return self.config.get("algorithm", {})

    @property
    def algorithm_name(self) -> str:
        name = self.algorithm.get("name") or self.config.get(
            "workflow", {}
        ).get("name", "")
        return str(name).rsplit(".", 1)[-1]

    @property
    def model(self) -> dict[str, Any]:
        return self.config.get("model", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.config.get("data", {})

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.config.get("evaluation", {})

    @property
    def output(self) -> dict[str, Any]:
        return self.config.get("output", {})

    @property
    def levels(self) -> list[str]:
        value = self.env.get("levels", self.env.get("level", "1-1"))
        if isinstance(value, str):
            levels = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple)):
            levels = [str(item) for item in value]
        else:
            raise ValueError(f"{self.config_path}: [env] level must be a string or list")
        if not levels:
            raise ValueError(f"{self.config_path}: no levels selected")
        return levels

    @property
    def level(self) -> str:
        return self.levels[0]

    @property
    def n_envs(self) -> int:
        return int(self.env.get("n_envs", 8))

    @property
    def agents_per_process(self) -> int:
        return int(self.env.get("agents_per_process", 1))

    @property
    def total_envs(self) -> int:
        return self.n_envs * self.agents_per_process

    @property
    def action_repeat(self) -> int:
        return int(self.env.get("action_repeat", 4))

    @property
    def max_decisions(self) -> int:
        return int(self.env.get("max_decisions", 150))

    @property
    def speedup(self) -> float:
        return float(self.env.get("speedup", 8.0))

    @property
    def timeout(self) -> int:
        return int(self.env.get("timeout", 60))

    @property
    def seed(self) -> int:
        return int(
            self.env.get("seed", self.config.get("run", {}).get("seed", 42))
        )

    @property
    def base_port(self) -> int:
        return int(self.env.get("base_port", 11008))

    @property
    def eval_port(self) -> int:
        return int(self.env.get("eval_port", 12000))

    @property
    def obs_encoding(self) -> str:
        return str(self.config.get("observation", {}).get("encoding", "legacy"))

    @property
    def reassert_held_jump(self) -> bool:
        """Whether repeated jump inputs must re-press the held button.

        Held-stick encodings need this by default so training, evaluation, and
        previews cannot silently use legacy edge-triggered semantics. Old
        encodings keep the legacy behavior unless the config explicitly opts in.
        """
        value = self.env.get("reassert_held_jump")
        if value is None:
            return self.obs_encoding in {"legacy_stick", "lean_stick"}
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def preprocessing(self) -> str:
        return str(self.config.get("observation", {}).get("preprocessing", "none"))

    @property
    def hidden(self) -> list[int]:
        return [int(value) for value in self.model.get("hidden", [64, 64])]

    @property
    def activation(self) -> str:
        return str(self.model.get("activation", "tanh"))

    @property
    def checkpoint_freq(self) -> int:
        return int(self.config.get("checkpointing", {}).get("save_freq", 25_000))

    def output_path(self, key: str, default: str) -> Path:
        return self.output_dir / str(self.output.get(key, default))

    @property
    def model_path(self) -> Path:
        default = (
            "ppo_pilot.zip"
            if self.algorithm_name == "ppo"
            else "dqn/dqn_final.zip"
        )
        return self.output_path("model_file", default)

    @property
    def monitor_path(self) -> Path:
        default = (
            "monitor.csv"
            if self.algorithm_name == "ppo"
            else "dqn/monitor.csv"
        )
        return self.output_path("monitor_file", default)

    @property
    def checkpoint_dir(self) -> Path:
        default = (
            "checkpoints"
            if self.algorithm_name == "ppo"
            else "dqn/checkpoints"
        )
        return self.output_path("checkpoint_dir", default)

    @property
    def eval_dir(self) -> Path:
        return self.output_dir / "eval"

    @property
    def tensorboard_log(self) -> Path | None:
        if not tensorboard_available():
            return None
        return self.output_dir / "tensorboard"

    @property
    def start_pool_path(self) -> Path | None:
        value = self.data.get("train_pool")
        return resolve_path(value) if value else None

    @property
    def eval_set_paths(self) -> list[Path]:
        values = self.evaluation.get("sets", [])
        if isinstance(values, (str, Path)):
            values = [values]
        return [resolve_path(value) for value in values]

    def ensure_dirs(self) -> None:
        for path in (
            self.output_dir / "godot",
            self.checkpoint_dir,
            self.eval_dir,
            self.monitor_path.parent,
            self.model_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_run(
    config_path: str | Path,
    *,
    timesteps: int | None = None,
    resume: str | Path | None = None,
    output_dir: str | Path | None = None,
    show: bool = False,
    device: str | None = None,
    verbose: int | None = None,
) -> Run:
    config = load_experiment(config_path)
    resolved_config_path = Path(config["_path"])
    algorithm = config.get("algorithm", {})
    output = config.get("output", {})
    run_config = config.get("run", {})

    output_value = output_dir if output_dir is not None else output.get("dir")
    if not output_value:
        raise ValueError(f"{resolved_config_path}: missing [output] dir")
    resolved_output = resolve_path(output_value)

    timestep_value = timesteps if timesteps is not None else algorithm.get("timesteps")
    if timestep_value is None:
        raise ValueError(f"{resolved_config_path}: missing [algorithm] timesteps")
    if int(timestep_value) <= 0:
        raise ValueError(f"{resolved_config_path}: timesteps must be positive")

    resume_value = resume if resume is not None else run_config.get("resume_from")
    resolved_resume = resolve_path(resume_value) if resume_value else None

    preprocessing_value = run_config.get("preprocessing_from")
    resolved_preprocessing = (
        resolve_path(preprocessing_value)
        if preprocessing_value
        else resolved_output / "preprocessing.pkl"
    )

    return Run(
        config=config,
        config_path=resolved_config_path,
        output_dir=resolved_output,
        timesteps=int(timestep_value),
        resume=resolved_resume,
        preprocessing_path=resolved_preprocessing,
        show=bool(show),
        device=str(device or run_config.get("device", "auto")),
        verbose=int(verbose if verbose is not None else run_config.get("verbose", 0)),
    )


def parse_train_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment directory or config.toml path.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override [algorithm].timesteps (target total timesteps).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Override [run].resume_from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override [output].dir.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", type=int, default=None)
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> Run:
    return load_run(
        args.config,
        timesteps=args.timesteps,
        resume=args.resume,
        output_dir=args.output_dir,
        show=args.show,
        device=args.device,
        verbose=args.verbose,
    )


def parse_eval_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment directory or config.toml path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Checkpoint to evaluate; defaults to [output].model_file.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Evaluation subdirectory name; defaults to the model stem.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", type=int, default=None)
    return parser.parse_args()


def activation_module(name: str):
    import torch

    modules = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}
    try:
        return modules[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported activation {name!r}; use tanh or relu"
        ) from exc


def train_start_pool(run: Run, *, required: bool = False) -> list[dict[str, Any]]:
    path = run.start_pool_path
    if path is None:
        if required:
            raise ValueError(f"{run.config_path}: missing [data] train_pool")
        return []
    if not path.exists():
        raise FileNotFoundError(f"start pool not found: {path}")
    from plaiground.data.pools import load_start_pool

    return load_start_pool(path)
