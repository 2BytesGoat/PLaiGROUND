#!/usr/bin/env python3
"""Build a deterministic search index for the PLaiGROUND repository.

The index has two consumers:

* ``docs/code-map.md`` is committed and answers "where does this live?" It is
  generated only from tracked-style source, experiment, and docs files, so it
  is stable across machines.
* ``.cache/repo-index.json`` is local and also covers generated ``runs/``
  artifacts. ``tools/context.py`` uses it to answer "what should I read first?"
  without a repo-wide grep.

Usage:
    .venv/bin/python tools/repo_index.py
    .venv/bin/python tools/repo_index.py --check
    .venv/bin/python tools/repo_index.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "repo-index.json"
MAP = ROOT / "docs" / "code-map.md"

CODE_ROOTS = ("src/plaiground", "scripts", "tools", "tests")
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".cache",
    "runs",
    "archive",
    "environments",
    "notebooks",
}
MAX_DOC_BYTES = 1_500_000
MAX_SUMMARY = 180


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def first_sentence(text: str, limit: int = MAX_SUMMARY) -> str:
    """Return a compact plain-text summary from markdown/docstring text."""
    text = text.strip()
    if not text:
        return ""
    lines = []
    in_frontmatter = False
    for index, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if index == 0 and line == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if line == "---":
                in_frontmatter = False
            continue
        if not line:
            if lines:
                break
            continue
        if line.startswith("#"):
            continue
        if line.startswith(("Status:", "**Status:**")):
            continue
        line = re.sub(r"^[>*\-+]\s*", "", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"[`*_]", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
        if len(" ".join(lines)) >= limit:
            break
    summary = " ".join(lines).strip()
    if len(summary) > limit:
        summary = summary[: limit - 1].rstrip() + "…"
    return summary


def docstring_summary(node: ast.AST) -> str:
    value = ast.get_docstring(node, clean=True)
    if not value:
        return ""
    line = value.strip().splitlines()[0].strip()
    if len(line) > MAX_SUMMARY:
        line = line[: MAX_SUMMARY - 1].rstrip() + "…"
    return line


def argument_names(args: ast.arguments) -> list[str]:
    names = [item.arg for item in args.posonlyargs]
    names += [item.arg for item in args.args]
    if args.vararg is not None:
        names.append("*" + args.vararg.arg)
    names += [item.arg for item in args.kwonlyargs]
    if args.kwarg is not None:
        names.append("**" + args.kwarg.arg)
    return names


def base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def scan_python(path: Path, root: Path) -> dict[str, Any] | None:
    text = read_text(path)
    if not text:
        return None
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return {
            "path": rel(path, root),
            "area": rel(path, root).split("/", 1)[0],
            "kind": "source",
            "lines": text.count("\n") + 1,
            "doc": "",
            "classes": [],
            "functions": [],
            "constants": [],
            "imports": [],
            "parse_error": True,
        }

    path_rel = rel(path, root)
    classes: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    constants: list[dict[str, Any]] = []
    imports: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                {
                    "name": child.name,
                    "line": child.lineno,
                    "args": argument_names(child.args),
                    "doc": docstring_summary(child),
                }
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "bases": [base_name(base) for base in node.bases],
                    "doc": docstring_summary(node),
                    "methods": methods,
                }
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "args": argument_names(node.args),
                    "doc": docstring_summary(node),
                }
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants.append({"name": target.id, "line": node.lineno})
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            constants.append({"name": node.target.id, "line": node.lineno})
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")

    return {
        "path": path_rel,
        "area": path_rel.split("/", 1)[0],
        "kind": "test" if path_rel.startswith("tests/") else "source",
        "lines": text.count("\n") + 1,
        "doc": first_sentence(ast.get_docstring(tree, clean=True) or ""),
        "classes": classes,
        "functions": functions,
        "constants": constants,
        "imports": sorted(imports),
        "parse_error": False,
    }


def scan_modules(root: Path) -> dict[str, dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {}
    for code_root in CODE_ROOTS:
        base = root / code_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            info = scan_python(path, root)
            if info is not None:
                modules[info["path"]] = info
    return modules


def flatten_symbols(modules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    for path, module in modules.items():
        for item in module["classes"]:
            symbols.append(
                {
                    "name": item["name"],
                    "kind": "class",
                    "path": path,
                    "line": item["line"],
                    "parent": None,
                    "doc": item["doc"],
                    "args": [],
                }
            )
            for method in item["methods"]:
                symbols.append(
                    {
                        "name": method["name"],
                        "kind": "method",
                        "path": path,
                        "line": method["line"],
                        "parent": item["name"],
                        "doc": method["doc"],
                        "args": method["args"],
                    }
                )
        for item in module["functions"]:
            symbols.append(
                {
                    "name": item["name"],
                    "kind": "function",
                    "path": path,
                    "line": item["line"],
                    "parent": None,
                    "doc": item["doc"],
                    "args": item["args"],
                }
            )
    symbols.sort(key=lambda item: (item["path"], item["line"], item["name"]))
    return symbols


def _config_levels(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _readme_summary(text: str) -> tuple[str, str]:
    title = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break
    return title, first_sentence(text)


def scan_experiments(root: Path) -> list[dict[str, Any]]:
    directory = root / "experiments"
    if not directory.exists():
        return []
    experiments: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_dir():
            continue
        config_path = path / "config.toml"
        if not config_path.exists():
            continue
        text = read_text(config_path)
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            experiments.append(
                {
                    "id": path.name.split("_", 1)[0],
                    "name": path.name,
                    "status": "parse-error",
                    "workflow": "",
                    "levels": [],
                    "selected_checkpoint": "",
                    "result": str(exc),
                    "output_dir": "",
                    "config": rel(config_path, root),
                    "readme": rel(path / "README.md", root)
                    if (path / "README.md").exists()
                    else "",
                    "title": path.name,
                    "summary": "",
                    "search_text": f"{path.name} {exc}",
                }
            )
            continue

        experiment = data.get("experiment", {})
        workflow = data.get("workflow", {})
        env = data.get("env", {})
        result = data.get("result", {})
        output = data.get("output", {})
        readme_path = path / "README.md"
        readme_text = read_text(readme_path) if readme_path.exists() else ""
        title, summary = _readme_summary(readme_text)
        levels = _config_levels(env.get("levels", env.get("level")))
        selected = str(result.get("selected_checkpoint", ""))
        result_note = str(result.get("note", result.get("status", "")))
        search_text = " ".join(
            [
                str(experiment.get("id", "")),
                str(experiment.get("name", path.name)),
                str(experiment.get("status", "")),
                str(workflow.get("name", "")),
                " ".join(levels),
                selected,
                result_note,
                title,
                summary,
                readme_text,
                text,
            ]
        )
        experiments.append(
            {
                "id": str(experiment.get("id", path.name.split("_", 1)[0])),
                "name": str(experiment.get("name", path.name)),
                "status": str(experiment.get("status", "")),
                "workflow": str(workflow.get("name", "")),
                "levels": levels,
                "selected_checkpoint": selected,
                "result": result_note,
                "output_dir": str(output.get("dir", "")),
                "config": rel(config_path, root),
                "readme": rel(readme_path, root) if readme_path.exists() else "",
                "title": title,
                "summary": summary,
                "search_text": search_text,
            }
        )
    experiments.sort(key=lambda item: (item["id"], item["name"]))
    return experiments


def scan_docs(root: Path) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for name in ("AGENTS.md", "README.md", "src/README.md"):
        path = root / name
        if path.exists():
            paths.append(path)
    docs_root = root / "docs"
    if docs_root.exists():
        for path in sorted(docs_root.rglob("*.md")):
            if path.name == "code-map.md":
                continue
            if path.stat().st_size > MAX_DOC_BYTES:
                continue
            paths.append(path)

    docs: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        text = read_text(path)
        title = ""
        headings: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if not match:
                continue
            level = len(match.group(1))
            heading = match.group(2).strip()
            headings.append({"level": level, "text": heading, "line": line_number})
            if level == 1 and not title:
                title = heading
        status = ""
        for line in text.splitlines():
            match = re.match(r"^\**Status\**\s*:\s*\**\s*(.+)$", line.strip())
            if match:
                status = match.group(1).strip().strip("*").strip()
                break
        docs.append(
            {
                "path": rel(path, root),
                "title": title or path.name,
                "status": status,
                "headings": headings,
                "summary": first_sentence(text),
                "text": text,
            }
        )
    docs.sort(key=lambda item: item["path"])
    return docs


def _compact_canonical(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value.get(key)
        for key in ("agents", "finishes", "deaths", "timeouts", "best_time", "median_time")
        if key in value
    }


def _compact_sets(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for name, summary in value.items():
        if isinstance(summary, dict):
            result[name] = {
                key: summary.get(key)
                for key in ("agents", "finishes", "deaths", "timeouts", "best_time", "median_time")
                if key in summary
            }
    return result


def _scan_eval_summaries(run_dir: Path, root: Path) -> list[dict[str, Any]]:
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*/summary.json")):
        try:
            data = json.loads(read_text(path))
        except json.JSONDecodeError:
            continue
        summaries.append(
            {
                "tag": str(data.get("tag", path.parent.name)),
                "canonical": _compact_canonical(data.get("canonical", {})),
                "sets": _compact_sets(data.get("sets", {})),
            }
        )
    return summaries


def _scan_archive(run_dir: Path, root: Path) -> dict[str, Any]:
    archive = run_dir / "archive"
    info: dict[str, Any] = {}
    routes_path = archive / "routes.json"
    if routes_path.exists():
        try:
            data = json.loads(read_text(routes_path))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            routes = list(data.values())
        elif isinstance(data, list):
            routes = data
        else:
            routes = []
        clean = [route for route in routes if isinstance(route, dict) and route.get("clean")]
        generations = [
            int(route.get("generation", 0))
            for route in routes
            if isinstance(route, dict)
        ]
        info.update(
            {
                "routes": len(routes),
                "clean": len(clean),
                "best_clean": min(
                    (
                        float(route.get("run_time", float("inf")))
                        for route in clean
                        if route.get("run_time") is not None
                    ),
                    default=None,
                ),
                "max_generation": max(generations, default=None),
            }
        )
    best_clean_path = run_dir / "best_clean.json"
    if best_clean_path.exists():
        try:
            info["best_clean_time"] = json.loads(read_text(best_clean_path)).get("time")
        except json.JSONDecodeError:
            pass
    history_path = run_dir / "history.csv"
    if history_path.exists():
        lines = [line for line in read_text(history_path).splitlines() if line.strip()]
        if len(lines) >= 2:
            header = lines[0].split(",")
            last = lines[-1].split(",")
            row = dict(zip(header, last))
            info["history"] = {
                key: row.get(key)
                for key in ("generation", "finishes", "deaths", "best_time")
                if key in row
            }
    return info


def _natural_number(name: str) -> int:
    numbers = re.findall(r"\d+", name)
    return int(numbers[-1]) if numbers else -1


def scan_runs(
    root: Path, experiments: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    directory = root / "runs"
    if not directory.exists():
        return []
    experiment_levels: dict[str, set[str]] = {}
    for experiment in experiments or []:
        output_dir = str(experiment.get("output_dir", "")).strip().rstrip("/")
        if not output_dir:
            continue
        experiment_levels.setdefault(output_dir, set()).update(
            experiment.get("levels", [])
        )
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(directory.iterdir()):
        if not run_dir.is_dir():
            continue
        key_files: list[str] = []
        for name in (
            "monitor.csv",
            "history.csv",
            "best_clean.json",
            "best_time.json",
            "dqn/dqn_final.zip",
            "archive/routes.json",
            "archive/finishers.jsonl",
        ):
            path = run_dir / name
            if path.exists():
                key_files.append(name)
        key_files.extend(
            item.name
            for item in sorted(run_dir.glob("expert_buffer*.pkl"))[:2]
        )
        key_files.extend(
            item.name
            for item in sorted(run_dir.glob("replay_buffer*.pkl"))[:2]
        )
        key_files.extend(
            item.name
            for item in sorted(
                run_dir.glob("neat-checkpoint-*"), key=lambda path: _natural_number(path.name)
            )[-3:]
        )
        key_files.extend(
            item.name
            for item in sorted(
                run_dir.glob("dqn/checkpoints/*.zip"),
                key=lambda path: _natural_number(path.name),
            )[-8:]
        )
        evals = _scan_eval_summaries(run_dir, root)
        archive = _scan_archive(run_dir, root)
        run_path = rel(run_dir, root)
        levels = sorted(experiment_levels.get(run_path, set()))
        search_parts = [
            run_dir.name,
            " ".join(key_files),
            " ".join(item["tag"] for item in evals),
            " ".join(levels),
        ]
        if archive:
            search_parts.append(json.dumps(archive, sort_keys=True))
        runs.append(
            {
                "name": run_dir.name,
                "path": run_path,
                "levels": levels,
                "key_files": sorted(set(key_files)),
                "evals": evals,
                "archive": archive,
                "search_text": " ".join(search_parts),
            }
        )
    return runs


def build_index(root: Path = ROOT, *, include_runs: bool = True) -> dict[str, Any]:
    modules = scan_modules(root)
    symbols = flatten_symbols(modules)
    experiments = scan_experiments(root)
    docs = scan_docs(root)
    runs = scan_runs(root, experiments) if include_runs else []
    stats = {
        "modules": len(modules),
        "source_modules": sum(1 for item in modules.values() if item["kind"] == "source"),
        "test_modules": sum(1 for item in modules.values() if item["kind"] == "test"),
        "classes": sum(1 for item in symbols if item["kind"] == "class"),
        "functions": sum(1 for item in symbols if item["kind"] == "function"),
        "methods": sum(1 for item in symbols if item["kind"] == "method"),
        "experiments": len(experiments),
        "docs": len(docs),
        "runs": len(runs),
    }
    return {
        "root": str(root),
        "stats": stats,
        "modules": modules,
        "symbols": symbols,
        "experiments": experiments,
        "docs": docs,
        "runs": runs,
    }


def md_code(value: Any) -> str:
    text = str(value) if value is not None else ""
    return f"`{text}`" if text else "—"


def short_list(items: list[str], limit: int = 3) -> str:
    if not items:
        return "—"
    shown = ", ".join(f"`{item}`" for item in items[:limit])
    if len(items) > limit:
        shown += f" +{len(items) - limit}"
    return shown


def truncate(value: str, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > limit:
        return value[: limit - 1].rstrip() + "…"
    return value


def render_markdown(index: dict[str, Any]) -> str:
    stats = index["stats"]
    modules = index["modules"]
    experiments = index["experiments"]
    docs = index["docs"]

    lines: list[str] = []
    lines.append("# Code map")
    lines.append("")
    lines.append(
        "> Generated by `make index` (`tools/repo_index.py`). Do not edit by hand."
    )
    lines.append("")
    lines.append(
        "This is the *where* layer: Python modules, symbols, experiment manifests, "
        "and docs. The *now* layer is [`status.md`](status.md); the task front door "
        "is [`index.md`](index.md); traps are in [`known-issues.md`](known-issues.md)."
    )
    lines.append("")
    lines.append("## Stats")
    lines.append("")
    lines.append(
        f"- {stats['source_modules']} source modules, {stats['test_modules']} test "
        f"modules, {stats['classes']} classes, {stats['functions']} top-level "
        f"functions, {stats['methods']} methods"
    )
    lines.append(
        f"- {stats['experiments']} experiment manifests, {stats['docs']} docs"
    )
    lines.append(
        "- Local run artifacts are indexed in `.cache/repo-index.json`; use "
        "`make context q=\"...\"` or [`reference/run-artifacts.md`](reference/run-artifacts.md)."
    )
    lines.append("")

    areas = Counter(item["area"] for item in modules.values())
    lines.append("## Source by area")
    lines.append("")
    lines.append("| Area | Modules |")
    lines.append("|---|---:|")
    for area in sorted(areas):
        lines.append(f"| `{area}/` | {areas[area]} |")
    lines.append("")

    for area in sorted(areas):
        lines.append(f"### {area}/")
        lines.append("")
        lines.append("| Module | Classes | Top-level functions | Lines | Summary |")
        lines.append("|---|---|---|---:|---|")
        area_modules = sorted(
            (item for item in modules.values() if item["area"] == area),
            key=lambda item: item["path"],
        )
        for module in area_modules:
            classes = [item["name"] for item in module["classes"]]
            functions = [item["name"] for item in module["functions"]]
            error = " **(parse error)**" if module.get("parse_error") else ""
            lines.append(
                "| %s | %s | %s | %d | %s%s |"
                % (
                    md_code(module["path"]),
                    short_list(classes),
                    short_list(functions),
                    module["lines"],
                    truncate(module.get("doc", ""), 110),
                    error,
                )
            )
        lines.append("")

    lines.append("## Experiments")
    lines.append("")
    lines.append("| ID | Name | Status | Workflow | Levels | Selected checkpoint |")
    lines.append("|---|---|---|---|---|---|")
    for item in experiments:
        checkpoint = item["selected_checkpoint"].split("/")[-1] if item["selected_checkpoint"] else ""
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                md_code(item["id"]),
                md_code(item["name"]),
                md_code(truncate(item["status"], 40)),
                md_code(item["workflow"]),
                md_code(", ".join(item["levels"])),
                md_code(checkpoint),
            )
        )
    lines.append("")

    lines.append("## Docs")
    lines.append("")
    lines.append("| Doc | Title | Headings | Status | Summary |")
    lines.append("|---|---|---:|---|---|")
    for item in docs:
        lines.append(
            "| %s | %s | %d | %s | %s |"
            % (
                md_code(item["path"]),
                md_code(truncate(item["title"], 40)),
                len(item["headings"]),
                md_code(truncate(item["status"], 30)),
                truncate(item["summary"], 100),
            )
        )
    lines.append("")

    lines.append("## Query this map")
    lines.append("")
    lines.append("```bash")
    lines.append('make context q="1-5 expert replay pool"   # code + experiments + docs + runs')
    lines.append("make index                              # regenerate this map and .cache/repo-index.json")
    lines.append("make index-check                        # fail if this map is stale")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if docs/code-map.md is stale.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the machine-readable index instead of writing files.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    index = build_index(ROOT, include_runs=not args.check)
    if args.json:
        print(json.dumps(index, indent=2, sort_keys=True))
        return 0

    markdown = render_markdown(index)
    if args.check:
        current = read_text(MAP) if MAP.exists() else ""
        if current != markdown:
            print(
                "docs/code-map.md is stale. Run `make index` and include the result.",
                flush=True,
            )
            return 1
        if not args.quiet:
            print("docs/code-map.md is up to date", flush=True)
        return 0

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    MAP.parent.mkdir(parents=True, exist_ok=True)
    MAP.write_text(markdown, encoding="utf-8")
    if not args.quiet:
        stats = index["stats"]
        print(
            f"indexed {stats['modules']} Python modules, "
            f"{stats['experiments']} experiments, {stats['docs']} docs, "
            f"{stats['runs']} runs -> {rel(MAP, ROOT)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
