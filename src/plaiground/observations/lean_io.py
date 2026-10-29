"""Load or create the fitted lean observation pipeline for a run."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from plaiground.envs.godot import GodotEnv
from plaiground.envs.runtime import isolated_home
from plaiground.experiments.runner import Run

from . import build_lean_pipeline, build_lean_stick_pipeline
from .lean_stick import LeanStickPreprocessor


def load_lean_preprocessor(path: Path) -> Any:
    with path.open("rb") as handle:
        preprocessor = pickle.load(handle)
    if not hasattr(preprocessor, "transform") or not hasattr(
        preprocessor, "output_size"
    ):
        raise TypeError(f"{path} does not contain a fitted state pipeline")
    return preprocessor


def save_lean_preprocessor(
    path: Path,
    preprocessor: Any,
    tile_names: list[str],
    *,
    encoding: str = "lean",
    obs_size: int = 188,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(preprocessor, handle)
    metadata = {
        "preprocessing": encoding,
        "obs_encoding": encoding,
        "obs_size": int(obs_size),
        "model_input_size": int(preprocessor.output_size),
        "tile_names": list(tile_names),
    }
    (path.parent / "preprocessing.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"preprocessing: {encoding} {metadata['obs_size']} -> "
        f"{metadata['model_input_size']} values, "
        f"{len(tile_names)} tile IDs; saved {path}",
        flush=True,
    )


def probe_lean_tile_names(run: Run, env_path: str) -> list[str]:
    """Read the game's tile vocabulary without starting a training world."""
    encoding_name = run.obs_encoding
    expected_size = (
        191 if encoding_name == "lean_stick" else 188
    )
    port = run.base_port + run.n_envs + 50
    log_path = run.output_dir / "godot" / "lean_probe.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"probing {encoding_name} static info on port {port} "
        f"(level {run.level})",
        flush=True,
    )
    GodotEnv.DEFAULT_TIMEOUT = run.timeout
    env = GodotEnv(
        env_path=env_path,
        port=port,
        show_window=False,
        seed=run.seed,
        action_repeat=run.action_repeat,
        speedup=run.speedup,
        nb_agents=1,
        level=run.level,
        convert_action_space=True,
        log_path=str(log_path),
        env=isolated_home(run.output_dir, "lean_probe"),
        obs_encoding=encoding_name,
    )
    try:
        names = list(env.static_info.get("tile_names") or [])
        advertised = str(env.static_info.get("obs_encoding", ""))
        size = int(env.static_info.get("obs_size", 0))
    finally:
        env.close()
    if advertised != encoding_name or size != expected_size:
        raise RuntimeError(
            f"game did not advertise {encoding_name}: "
            f"obs_encoding={advertised!r} obs_size={size}"
        )
    if not names:
        raise RuntimeError("game did not advertise static_info.tile_names")
    print(f"{encoding_name} tile vocabulary: {len(names)} tiles", flush=True)
    return names


def prepare_lean_preprocessor(run: Run, env_path: str) -> Any | None:
    """Return the fitted pipeline for a run, creating it when needed."""
    if run.preprocessing == "none":
        if run.obs_encoding in {"lean", "lean_stick"}:
            raise ValueError(
                f"--obs-encoding {run.obs_encoding} requires "
                "--preprocessing lean or lean_stick so the model sees "
                "encoded features, not raw values"
            )
        return None
    if run.preprocessing not in {"lean", "lean_stick"}:
        raise ValueError(f"unknown preprocessing {run.preprocessing!r}")
    if run.preprocessing == "lean" and run.obs_encoding != "lean":
        raise ValueError("lean preprocessing requires observation encoding 'lean'")
    if run.preprocessing == "lean_stick" and run.obs_encoding != "lean_stick":
        raise ValueError(
            "lean_stick preprocessing requires observation encoding 'lean_stick'"
        )

    path = run.preprocessing_path
    if run.resume is not None or path.exists():
        if not path.exists():
            raise FileNotFoundError(
                f"lean resume/eval needs the saved pipeline at {path}; "
                "set [run].preprocessing_from in the experiment config"
            )
        preprocessor = load_lean_preprocessor(path)
        if run.preprocessing == "lean_stick" and not isinstance(
            preprocessor, LeanStickPreprocessor
        ):
            preprocessor = LeanStickPreprocessor(preprocessor)
        print(
            f"preprocessing: loaded {run.preprocessing} pipeline {path} "
            f"({preprocessor.output_size} inputs)",
            flush=True,
        )
        return preprocessor

    tile_names = probe_lean_tile_names(run, env_path)
    if run.preprocessing == "lean_stick":
        preprocessor = build_lean_stick_pipeline(tile_names=tile_names)
        # Fit the underlying lean pipeline; the flags are passthrough scalars.
        preprocessor.fit(np.zeros((1, 191), dtype=np.float32))
        save_lean_preprocessor(
            path,
            preprocessor,
            tile_names,
            encoding="lean_stick",
            obs_size=191,
        )
    else:
        preprocessor = build_lean_pipeline(tile_names=tile_names)
        # Fit on a neutral observation: the vocabulary and fixed categories,
        # not the first rollout batch, determine the model width.
        preprocessor.fit(np.zeros((1, 188), dtype=np.float32))
        save_lean_preprocessor(path, preprocessor, tile_names)
    return preprocessor
