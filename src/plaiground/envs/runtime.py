"""Shared Godot process/runtime helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = REPO / "environments"


def resolve_env_path() -> str:
    """Resolve the exported game path from src/.config, like utils.py does."""
    config_path = REPO / "src" / ".config"
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

def isolated_home(output_dir: Path, name: str) -> dict[str, str]:
    """Give each game process a private user:// tree."""
    root = (output_dir / "userdata" / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["HOME"] = str(root)
    child_env["XDG_DATA_HOME"] = str(root / "xdg_data")
    child_env["XDG_CONFIG_HOME"] = str(root / "xdg_config")
    return child_env
