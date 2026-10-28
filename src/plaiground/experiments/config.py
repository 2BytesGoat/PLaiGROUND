"""Load committed experiment manifests.

Experiment records live in ``experiments/<id>_<slug>/config.toml``. The file is
the source of truth for the question, workflow, environment, data, model, and
training budget. Generated artifacts stay in ``runs/``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO / "experiments"


def resolve_experiment(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if candidate.is_dir():
        candidate = candidate / "config.toml"
    if not candidate.exists():
        raise FileNotFoundError(f"experiment config not found: {candidate}")
    return candidate


def load_experiment(path: str | Path) -> dict[str, Any]:
    config_path = resolve_experiment(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"experiment must be a TOML object: {config_path}")
    payload["_path"] = str(config_path)
    payload.setdefault("experiment", {})
    payload["experiment"].setdefault("id", config_path.parent.name.split("_", 1)[0])
    return payload


def find_experiments(root: Path = EXPERIMENTS_DIR) -> list[Path]:
    return sorted(path.parent for path in root.glob("*/config.toml"))
