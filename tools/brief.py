#!/usr/bin/env python3
"""Print a compact, read-only brief of the repo or one run.

``brief`` answers the questions that otherwise turn into ad-hoc ``ls``,
``find``, ``ps``, pickle, and JSON commands:

* What branch/commit is the repo and the game on?
* What is training or evaluating right now?
* Which expert is selected for each level?
* Which runs changed most recently, and where is their latest checkpoint?
* For one run: checkpoints, eval summaries, NEAT archive stats, and log tail.

Usage:
    .venv/bin/python tools/brief.py
    .venv/bin/python tools/brief.py --run ddqn_1_9_single_route
    .venv/bin/python tools/brief.py --run 1_9 --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "reference" / "level-ledger.md"

PROCESS_MARKERS = (
    "scripts/reinforcement/",
    "scripts/genetic/",
    "tools/evaluate.py",
    "tools/collect_",
    "tools/build_",
    "tools/rebalance_neat.py",
    "tools/extract_route_starts.py",
    "godot --headless",
)


def _run(
    command: list[str], cwd: Path, timeout: float = 10.0, strip: bool = True
) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() if strip else result.stdout


def git_state(root: Path) -> dict[str, Any]:
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
    commit = _run(["git", "rev-parse", "--short", "HEAD"], root)
    subject = _run(["git", "log", "-1", "--pretty=%s"], root)
    status = _run(["git", "status", "--porcelain"], root, strip=False) or ""
    dirty = [line for line in status.splitlines() if line.strip()]
    return {
        "branch": branch,
        "commit": commit,
        "subject": subject,
        "dirty_count": len(dirty),
        "dirty_files": [line[3:].strip() for line in dirty[:5]],
    }


def running_jobs() -> list[dict[str, str]]:
    output = _run(["ps", "-axo", "pid=,etime=,command="], ROOT)
    if not output:
        return []
    jobs: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid, elapsed, command = parts
        if "brief.py" in command or command.startswith("ps -axo"):
            continue
        if not any(marker in command for marker in PROCESS_MARKERS):
            continue
        jobs.append({"pid": pid, "elapsed": elapsed, "command": command})
    return jobs


def _clean_cell(value: str) -> str:
    return value.replace("`", "").replace("**", "").strip()


def parse_ledger(path: Path = LEDGER) -> list[dict[str, str]]:
    """Read the level table from ``docs/reference/level-ledger.md``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict[str, str]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Level |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7 or not re.fullmatch(r"\d+-\d+", cells[0]):
            continue
        rows.append(
            {
                "level": cells[0],
                "par": _clean_cell(cells[1]),
                "route": _clean_cell(cells[2]),
                "selected": _clean_cell(cells[3]),
                "canonical": _clean_cell(cells[4]),
                "coverage": _clean_cell(cells[5]),
                "status": _clean_cell(cells[6]),
            }
        )
    return rows


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _step(value: str) -> int:
    numbers = re.findall(r"\d+", value)
    return int(numbers[-1]) if numbers else -1


def checkpoints(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in (
        "dqn/checkpoints/*.zip",
        "ppo/checkpoints/*.zip",
        "checkpoints/*.zip",
        "dqn/dqn_final.zip",
        "ppo/ppo_final.zip",
    ):
        paths.extend(path for path in run_dir.glob(pattern) if path.is_file())
    unique = {path.resolve(): path for path in paths}
    return sorted(unique.values(), key=_checkpoint_key)


def _checkpoint_key(path: Path) -> tuple[float, str]:
    step = _step(path.name)
    # ``dqn_final.zip`` has no step number; sort it after numbered checkpoints.
    return (step if step >= 0 else float("inf"), path.name)


def _compact_stats(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = ("agents", "finishes", "deaths", "timeouts", "best_time", "median_time")
    return {key: value.get(key) for key in keys if key in value}


def eval_summaries(run_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted((run_dir / "eval").glob("*/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summaries.append(
            {
                "tag": str(data.get("tag", path.parent.name)),
                "path": path.relative_to(ROOT).as_posix()
                if path.is_relative_to(ROOT)
                else str(path),
                "mtime": path.stat().st_mtime,
                "canonical": {
                    level: _compact_stats(stats)
                    for level, stats in (data.get("canonical") or {}).items()
                    if isinstance(stats, dict)
                },
                "sets": {
                    name: _compact_stats(stats)
                    for name, stats in (data.get("sets") or {}).items()
                    if isinstance(stats, dict)
                },
            }
        )
    return summaries


def neat_state(run_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    checkpoints_found = sorted(
        run_dir.glob("neat-checkpoint-*"), key=lambda path: _natural_key(path.name)
    )
    if checkpoints_found:
        info["latest_checkpoint"] = checkpoints_found[-1].name
    routes_path = run_dir / "archive" / "routes.json"
    if routes_path.exists():
        try:
            data = json.loads(routes_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        routes = list(data.values()) if isinstance(data, dict) else data
        if isinstance(routes, list):
            clean = [
                route
                for route in routes
                if isinstance(route, dict) and route.get("clean")
            ]
            times = [
                float(route["run_time"])
                for route in clean
                if route.get("run_time") is not None
            ]
            info["routes"] = len(routes)
            info["clean"] = len(clean)
            info["best_clean"] = min(times, default=None)
    best_clean = run_dir / "best_clean.json"
    if best_clean.exists():
        try:
            data = json.loads(best_clean.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict):
            info["best_clean_time"] = data.get("time")
            info["best_clean_generation"] = data.get("generation")
    return info


def scan_runs(root: Path = ROOT) -> list[dict[str, Any]]:
    directory = root / "runs"
    if not directory.exists():
        return []
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(directory.iterdir()):
        if not run_dir.is_dir():
            continue
        stat = run_dir.stat()
        runs.append(
            {
                "name": run_dir.name,
                "path": run_dir.relative_to(root).as_posix(),
                "mtime": stat.st_mtime,
                "checkpoints": checkpoints(run_dir),
                "evals": eval_summaries(run_dir),
                "neat": neat_state(run_dir),
            }
        )
    return runs


def experiment_index(root: Path = ROOT) -> dict[str, list[dict[str, Any]]]:
    """Map ``[output].dir`` to the experiment manifests that own it."""
    mapping: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((root / "experiments").glob("*/config.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        output_dir = str((data.get("output") or {}).get("dir", "")).strip().rstrip("/")
        if not output_dir:
            continue
        experiment = data.get("experiment") or {}
        algorithm = data.get("algorithm") or {}
        mapping.setdefault(output_dir, []).append(
            {
                "path": path.relative_to(root).as_posix(),
                "id": experiment.get("id"),
                "name": experiment.get("name"),
                "status": experiment.get("status"),
                "selected_checkpoint": (data.get("result") or {}).get(
                    "selected_checkpoint"
                ),
                "levels": (data.get("env") or {}).get("levels")
                or (data.get("env") or {}).get("level"),
                "algorithm": algorithm.get("name"),
                "timesteps": algorithm.get("timesteps"),
            }
        )
    return mapping


def _fmt_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _fmt_age(timestamp: float) -> str:
    seconds = max(0, int(datetime.now().timestamp() - timestamp))
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _fmt_seconds(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}s"
    except (TypeError, ValueError):
        return "—"


def _fmt_stats(stats: dict[str, Any]) -> str:
    agents = stats.get("agents")
    finishes = stats.get("finishes")
    best = stats.get("best_time")
    if agents is None and finishes is None and best is None:
        return "—"
    head = f"{finishes if finishes is not None else '?'}/{agents if agents is not None else '?'}"
    if best is None:
        return f"{head} timeout"
    return f"{head} {_fmt_seconds(best)}"


def _fmt_coverage(sets: dict[str, Any]) -> str:
    parts: list[str] = []
    for name, label in (
        ("train_start_pool", "train"),
        ("holdout_start_pool", "holdout"),
        ("clean", "clean"),
        ("all_success", "all"),
    ):
        stats = sets.get(name)
        if stats:
            parts.append(f"{label} {_fmt_stats(stats)}")
    if not parts:
        remaining = list(sets.items())[:2]
        parts = [f"{name} {_fmt_stats(stats)}" for name, stats in remaining]
    return ", ".join(parts) if parts else "—"


def _short(value: str, limit: int) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def render_repo_brief(
    root: Path = ROOT, *, jobs: list[dict[str, str]] | None = None
) -> str:
    repo = git_state(root)
    game_root = root.parent / "dragon-jump-remaster"
    game = git_state(game_root) if game_root.exists() else {}
    jobs = running_jobs() if jobs is None else jobs
    ledger = parse_ledger(root / "docs" / "reference" / "level-ledger.md")
    runs = scan_runs(root)
    recent = sorted(runs, key=lambda run: run["mtime"], reverse=True)[:8]

    lines: list[str] = []
    lines.append(f"PLaiGROUND brief — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(
        "repo  "
        + _repo_label(repo)
    )
    if game:
        lines.append("game  " + _repo_label(game))

    lines.append("")
    if jobs:
        lines.append(f"Running ({len(jobs)})")
        for job in jobs[:6]:
            lines.append(
                f"  {job['pid']:>7}  {job['elapsed']:>10}  {_short(job['command'], 100)}"
            )
    else:
        lines.append("Running  none")

    lines.append("")
    lines.append("Selected levels (docs/reference/level-ledger.md)")
    if ledger:
        lines.append(
            f"  {'Level':<5} {'Selected':<24} {'Canonical':<9} {'Coverage':<46} Status"
        )
        for row in ledger:
            lines.append(
                f"  {row['level']:<5} {_short(row['selected'], 24):<24} "
                f"{row['canonical']:<9} {_short(row['coverage'], 46):<46} "
                f"{_short(row['status'], 38)}"
            )
    else:
        lines.append("  (ledger table not found)")

    lines.append("")
    lines.append("Recent runs (by mtime)")
    if recent:
        for run in recent:
            latest = "—"
            detail = ""
            if run["checkpoints"]:
                numbered = [
                    path for path in run["checkpoints"] if _step(path.name) >= 0
                ]
                latest = (numbered or run["checkpoints"])[-1]
                detail = f"{_short(latest.name, 28):<28} {len(run['evals'])} eval(s)"
            elif run["neat"].get("latest_checkpoint"):
                latest = run["neat"]["latest_checkpoint"]
                routes = run["neat"].get("routes")
                clean = run["neat"].get("clean")
                detail = f"{_short(latest, 28):<28} "
                detail += (
                    f"{routes} routes / {clean} clean"
                    if routes is not None
                    else "NEAT"
                )
            elif run["evals"]:
                detail = f"{'—':<28} {len(run['evals'])} eval(s)"
            lines.append(
                f"  {_short(run['name'], 36):<36} {_fmt_time(run['mtime'])}  {detail}"
            )
    else:
        lines.append("  (no runs/)")

    lines.append("")
    lines.append("Details: make run RUN=<name>  |  make context q=\"...\"")
    return "\n".join(lines)


def _repo_label(state: dict[str, Any]) -> str:
    if not state.get("branch"):
        return "(not a git repo)"
    label = f"{state['branch']} @ {state.get('commit', '?')}"
    if state.get("dirty_count"):
        label += f" (dirty: {state['dirty_count']} file(s))"
    else:
        label += " (clean)"
    if state.get("subject"):
        label += f" — {_short(state['subject'], 44)}"
    return label


def _read_tail(path: Path, lines: int = 6, max_bytes: int = 8192) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    entries = [line for line in text.splitlines() if line.strip()]
    noise = (
        "ObjectDB instances leaked",
        "resources still in use",
        "run with --verbose",
        "close message sent",
    )
    filtered = [
        line
        for line in entries
        if not line.lstrip().startswith("at:")
        and not any(token in line for token in noise)
    ]
    deduped: list[str] = []
    for line in filtered:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return (deduped or entries)[-lines:]


def resolve_run(root: Path, name: str) -> Path | None:
    directory = root / "runs"
    exact = directory / name
    if exact.is_dir():
        return exact
    matches = [path for path in directory.glob(f"*{name}*") if path.is_dir()]
    if len(matches) == 1:
        return matches[0]
    return None


def render_run_brief(root: Path, run_dir: Path) -> str:
    experiments = experiment_index(root)
    run = next(
        (item for item in scan_runs(root) if item["path"] == run_dir.relative_to(root).as_posix()),
        None,
    )
    if run is None:
        return f"Run not found: {run_dir}"

    lines: list[str] = []
    lines.append(f"Run {run_dir.name}")
    lines.append(f"  path      {run['path']}")
    lines.append(f"  modified  {_fmt_time(run['mtime'])} ({_fmt_age(run['mtime'])} ago)")
    size = _run(["du", "-sh", str(run_dir)], root)
    if size:
        lines.append(f"  size      {size.split()[0]}")

    experiment = experiments.get(run["path"], [])
    if experiment:
        primary = max(
            experiment,
            key=lambda item: item.get("timesteps") or 0,
        )
        label = f"{primary['path']}"
        if primary.get("status"):
            label += f"  [{primary['status']}]"
        lines.append(f"  manifest  {label}")
        if len(experiment) > 1:
            lines.append(
                f"            +{len(experiment) - 1} eval manifest(s) share this run"
            )
        if primary.get("selected_checkpoint"):
            lines.append(f"  selected  {primary['selected_checkpoint']}")

    checkpoints_list = run["checkpoints"]
    if checkpoints_list:
        latest = checkpoints_list[-1]
        step = _step(latest.name)
        step_label = f"{step:,} steps" if step >= 0 else "final"
        lines.append(
            f"  checkpoint {len(checkpoints_list)} total; latest {latest.name} "
            f"({step_label})"
        )
    else:
        lines.append("  checkpoint none")

    neat = run["neat"]
    if neat:
        lines.append(
            "  NEAT       "
            + ", ".join(
                part
                for part in (
                    neat.get("latest_checkpoint"),
                    (
                        f"{neat.get('routes')} routes / {neat.get('clean')} clean"
                        if neat.get("routes") is not None
                        else None
                    ),
                    (
                        f"best clean {_fmt_seconds(neat.get('best_clean'))}"
                        if neat.get("best_clean") is not None
                        else None
                    ),
                )
                if part
            )
        )

    lines.append("")
    lines.append("Evals")
    if run["evals"]:
        lines.append(
            f"  {'tag':<28} {'level':<5} {'canonical':<20} {'sets'}"
        )
        rows = 0
        for summary in sorted(run["evals"], key=lambda item: item["mtime"], reverse=True):
            canonical = summary["canonical"]
            if not canonical:
                continue
            for level, stats in sorted(canonical.items()):
                lines.append(
                    f"  {_short(summary['tag'], 28):<28} {level:<5} "
                    f"{_fmt_stats(stats):<20} {_short(_fmt_coverage(summary['sets']), 46)}"
                )
                rows += 1
            if rows >= 12:
                break
        if rows == 0:
            lines.append("  (no canonical summaries)")
        if len(run["evals"]) > 12:
            lines.append(f"  ... {len(run['evals']) - 12} more")
    else:
        lines.append("  (none)")

    logs = sorted(
        [
            path
            for path in list(run_dir.glob("*.log")) + list(run_dir.glob("*/*.log"))
            if path.is_file()
        ],
        key=lambda path: (
            0 if path.parent == run_dir else 1,
            -path.stat().st_mtime,
        ),
    )
    if logs:
        lines.append("")
        lines.append("Logs")
        for path in logs[:2]:
            lines.append(f"  {path.relative_to(run_dir).as_posix()} (tail)")
            for entry in _read_tail(path):
                lines.append(f"    {_short(entry, 160)}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print a compact repo or run brief.",
    )
    parser.add_argument(
        "run",
        nargs="?",
        help="run name or unique substring (default: repo brief)",
    )
    parser.add_argument("--run", dest="run_flag", help="run name or unique substring")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    name = args.run_flag or args.run
    if name:
        run_dir = resolve_run(ROOT, name)
        if run_dir is None:
            print(f"Run not found or ambiguous: {name}")
            print("Available runs are listed by `make brief`.")
            return 2
        if args.json:
            run = next(
                (
                    item
                    for item in scan_runs(ROOT)
                    if item["path"] == run_dir.relative_to(ROOT).as_posix()
                ),
                None,
            )
            payload = {
                "run": run,
                "experiments": experiment_index(ROOT).get(
                    run_dir.relative_to(ROOT).as_posix(), []
                ),
            }
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(render_run_brief(ROOT, run_dir))
        return 0

    if args.json:
        game_root = ROOT.parent / "dragon-jump-remaster"
        payload = {
            "repo": git_state(ROOT),
            "game": git_state(game_root) if game_root.exists() else {},
            "running": running_jobs(),
            "ledger": parse_ledger(),
            "runs": scan_runs(ROOT),
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(render_repo_brief(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
