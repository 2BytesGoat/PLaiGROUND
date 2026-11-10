"""Shared Godot process/runtime helpers."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = REPO / "environments"

#: Godot reads ``override.cfg`` from the executable's directory. This is the
#: A41 training setting: it serializes the global worker pool, which removes
#: thread-pool contention when several training processes step worlds at once.
#: Never copy this into ``project.godot`` -- the game uses the pool for resource
#: loading, and the cap only pays in multi-process training.
TRAINING_OVERRIDE = "[threading]\n\nworker_pool/max_threads=1\n"

#: Set to ``1`` for an A/B baseline arm (or any run that must not write the cap).
DISABLE_TRAINING_OVERRIDE = "PLAIGROUND_DISABLE_TRAINING_OVERRIDE"


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


def training_override_paths(executable_path: str | Path) -> list[Path]:
    """Return the ``override.cfg`` destinations for one game executable.

    Exported builds read the file next to the binary (``Contents/MacOS`` inside
    a macOS ``.app``). A source-project shim also drops a ``project_path``
    marker next to its binary, so the setting can be written into the project
    root, where ``--path`` runs read it.
    """
    executable = Path(executable_path)
    if executable.suffix == ".app":
        executable = executable / "Contents" / "MacOS" / executable.stem
    paths = [executable.parent / "override.cfg"]
    marker = executable.parent / "project_path"
    if marker.is_file():
        project = Path(marker.read_text(encoding="utf-8").strip())
        if project:
            candidate = project / "override.cfg"
            if candidate not in paths:
                paths.append(candidate)
    return paths


def ensure_training_override(executable_path: str | Path) -> list[Path]:
    """Write the training override next to (and for shims, inside) the game.

    Idempotent and safe to call from every process launch: an up-to-date file
    is left alone, a stale or missing one is replaced atomically.
    """
    if os.getenv(DISABLE_TRAINING_OVERRIDE) == "1":
        return []

    written: list[Path] = []
    for target in training_override_paths(executable_path):
        if target.is_file() and target.read_text(encoding="utf-8") == TRAINING_OVERRIDE:
            written.append(target)
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            temporary.write_text(TRAINING_OVERRIDE, encoding="utf-8")
            os.replace(temporary, target)
        except OSError as exc:
            warnings.warn(
                f"could not write training override {target}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            temporary.unlink(missing_ok=True)
            continue
        written.append(target)
    return written


def remove_training_override(executable_path: str | Path) -> list[Path]:
    """Delete only overrides this harness wrote, for an A/B baseline arm."""
    removed: list[Path] = []
    for target in training_override_paths(executable_path):
        if not target.is_file():
            continue
        if target.read_text(encoding="utf-8") != TRAINING_OVERRIDE:
            continue
        target.unlink()
        removed.append(target)
    return removed


def isolated_home(output_dir: Path, name: str) -> dict[str, str]:
    """Give each game process a private user:// tree."""
    root = (output_dir / "userdata" / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["HOME"] = str(root)
    child_env["XDG_DATA_HOME"] = str(root / "xdg_data")
    child_env["XDG_CONFIG_HOME"] = str(root / "xdg_config")
    return child_env
