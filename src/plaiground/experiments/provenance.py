"""Capture code and game provenance for a run.

The experiment config records the expected game version. The run metadata
records what actually executed, including a diff hash and, when dirty, the raw
game diff so the result remains reproducible.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_metadata(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    status = _git(repo, "status", "--porcelain")
    diff = _git(repo, "diff", "--binary")
    dirty = bool(status)
    return {
        "path": str(repo),
        "commit": commit or None,
        "branch": branch or None,
        "dirty": dirty,
        "status": status.splitlines() if status else [],
        "diff_sha256": (
            hashlib.sha256(diff.encode("utf-8")).hexdigest() if dirty else None
        ),
        "diff": diff if dirty else "",
    }


def capture_provenance(
    *,
    game_repo: str | Path,
    plai_repo: str | Path = REPO,
) -> dict[str, Any]:
    game = git_metadata(Path(game_repo))
    plai = git_metadata(Path(plai_repo))
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "plai": {key: value for key, value in plai.items() if key != "diff"},
        "game": {key: value for key, value in game.items() if key != "diff"},
        "game_diff": game["diff"],
    }


def write_run_metadata(
    run_dir: str | Path,
    *,
    experiment: dict[str, Any],
    game_repo: str | Path,
    command: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = capture_provenance(game_repo=game_repo)
    game_diff = provenance.pop("game_diff", "")
    payload = {
        "experiment": experiment,
        "command": command,
        **provenance,
    }
    if extra:
        payload.update(extra)
    if game_diff:
        (run_dir / "game.diff").write_text(game_diff, encoding="utf-8")
        payload["game"]["diff_file"] = "game.diff"
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
