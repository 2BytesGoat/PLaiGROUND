#!/usr/bin/env python3
"""Bounded text search for agents and humans.

This is the repo-native replacement for ``grep -R`` over the whole tree. It
prunes generated directories, skips binaries and oversized files, and caps the
number of lines printed. The goal is to get the matching lines without dumping
a 50 KB tool result that the model has to carry for the rest of the session.

Usage:
    .venv/bin/python tools/search.py "target_time"
    .venv/bin/python tools/search.py "eval_clean" --glob '*.py' --glob '*.md'
    .venv/bin/python tools/search.py "1-9" --dir ../dragon-jump-remaster/godot --glob '*.gd'
    .venv/bin/python tools/search.py "replay_buffer" --files-only
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".venv",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "__pycache__",
    "node_modules",
    "runs",
    "data",
    "archive",
    "environments",
    "notebooks",
    "dist",
    "build",
    "graphify-out",
}

SKIP_SUFFIXES = {
    ".7z",
    ".avi",
    ".bin",
    ".bmp",
    ".db",
    ".dmg",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".npz",
    ".onnx",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".wav",
    ".webp",
    ".zip",
}

MAX_LINE = 240
DEFAULT_MAX_FILES = 40


def _matches_glob(relative: str, name: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(
        fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(relative, pattern)
        for pattern in patterns
    )


def iter_files(
    root: Path,
    *,
    globs: list[str] | None = None,
    include_runs: bool = False,
    max_bytes: int = 2_000_000,
) -> Iterator[Path]:
    patterns = globs or []
    if root.is_file():
        yield root
        return
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in SKIP_DIRS
            or (include_runs and name == "runs")
        ]
        for name in filenames:
            path = current_path / name
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = path.name
            if not _matches_glob(relative, name, patterns):
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            yield path


def _compile(
    pattern: str, *, ignore_case: bool = False, fixed_strings: bool = False
) -> re.Pattern[str]:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        return (
            re.compile(re.escape(pattern), flags)
            if fixed_strings
            else re.compile(pattern, flags)
        )
    except re.error as error:
        raise ValueError(f"invalid regular expression: {error}") from error


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def search_lines(
    pattern: str,
    root: Path = ROOT,
    *,
    globs: list[str] | None = None,
    include_runs: bool = False,
    max_bytes: int = 2_000_000,
    max_matches: int = 40,
    max_per_file: int = 3,
    ignore_case: bool = False,
    fixed_strings: bool = False,
) -> dict[str, Any]:
    regex = _compile(
        pattern, ignore_case=ignore_case, fixed_strings=fixed_strings
    )
    matches: list[dict[str, Any]] = []
    files_with_matches: set[str] = set()
    files_scanned = 0
    global_limit_hit = False
    per_file_truncated = False

    for path in iter_files(
        root,
        globs=globs,
        include_runs=include_runs,
        max_bytes=max_bytes,
    ):
        text = _read_text(path)
        if text is None:
            continue
        files_scanned += 1
        relative = _relative(path, root)
        per_file = 0
        for number, line in enumerate(text.splitlines(), start=1):
            if not regex.search(line):
                continue
            files_with_matches.add(relative)
            if max_matches and len(matches) >= max_matches:
                global_limit_hit = True
                break
            if max_per_file and per_file >= max_per_file:
                per_file_truncated = True
                break
            text_line = line.strip()
            if len(text_line) > MAX_LINE:
                text_line = text_line[: MAX_LINE - 1] + "…"
            matches.append({"path": relative, "line": number, "text": text_line})
            per_file += 1
        if global_limit_hit:
            break

    return {
        "pattern": pattern,
        "root": str(root),
        "matches": matches,
        "files_with_matches": sorted(files_with_matches),
        "files_scanned": files_scanned,
        "limit_hit": global_limit_hit,
        "truncated": global_limit_hit or per_file_truncated,
    }


def search_files(
    pattern: str,
    root: Path = ROOT,
    *,
    globs: list[str] | None = None,
    include_runs: bool = False,
    max_bytes: int = 2_000_000,
    max_files: int = DEFAULT_MAX_FILES,
    ignore_case: bool = False,
    fixed_strings: bool = False,
) -> dict[str, Any]:
    regex = _compile(
        pattern, ignore_case=ignore_case, fixed_strings=fixed_strings
    )
    paths: list[str] = []
    files_scanned = 0
    limit_hit = False

    for path in iter_files(
        root,
        globs=globs,
        include_runs=include_runs,
        max_bytes=max_bytes,
    ):
        text = _read_text(path)
        if text is None:
            continue
        files_scanned += 1
        if not any(regex.search(line) for line in text.splitlines()):
            continue
        if max_files and len(paths) >= max_files:
            limit_hit = True
            break
        paths.append(_relative(path, root))

    return {
        "pattern": pattern,
        "root": str(root),
        "paths": paths,
        "files_scanned": files_scanned,
        "limit_hit": limit_hit,
        "truncated": limit_hit,
    }


def render_lines(result: dict[str, Any]) -> str:
    lines = [f'Search: {result["pattern"]!r} in {result["root"]}']
    if not result["matches"]:
        lines.append(f'No matches ({result["files_scanned"]} files scanned).')
        return "\n".join(lines)
    for match in result["matches"]:
        lines.append(f'{match["path"]}:{match["line"]}: {match["text"]}')
    suffix = " (stopped at the cap)" if result["limit_hit"] else ""
    lines.append(
        f'Showing {len(result["matches"])} match(es) in '
        f'{len(result["files_with_matches"])} file(s); '
        f'{result["files_scanned"]} files scanned{suffix}.'
    )
    if result["limit_hit"]:
        lines.append("Narrow with --glob, --dir, or --max-matches.")
    elif result["truncated"]:
        lines.append(
            "Some files have more matches; raise --max-per-file to see them."
        )
    return "\n".join(lines)


def render_files(result: dict[str, Any]) -> str:
    lines = [f'Search: {result["pattern"]!r} in {result["root"]}']
    if not result["paths"]:
        lines.append(f'No matches ({result["files_scanned"]} files scanned).')
        return "\n".join(lines)
    lines.extend(result["paths"])
    suffix = " (stopped at the cap)" if result["limit_hit"] else ""
    lines.append(
        f'{len(result["paths"])} file(s) with matches; '
        f'{result["files_scanned"]} files scanned{suffix}.'
    )
    if result["limit_hit"]:
        lines.append("Narrow with --glob, --dir, or --max-files.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded text search with generated-directory and binary pruning.",
    )
    parser.add_argument("pattern", help="regular expression (or literal with -F)")
    parser.add_argument("--dir", default=".", help="directory or file to search")
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="only search paths matching this glob (repeatable)",
    )
    parser.add_argument(
        "--include-runs",
        action="store_true",
        help="include runs/ (still skips binaries and oversized files)",
    )
    parser.add_argument(
        "--files-only",
        action="store_true",
        help="print unique matching paths instead of every line",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="disable the match caps",
    )
    parser.add_argument("--max-matches", type=int, default=40)
    parser.add_argument("--max-per-file", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    parser.add_argument("-i", "--ignore-case", action="store_true")
    parser.add_argument("-F", "--fixed-strings", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.dir).expanduser().resolve()
    if not root.exists():
        print(f"Search path not found: {root}", file=sys.stderr)
        return 2

    try:
        if args.files_only:
            result = search_files(
                args.pattern,
                root,
                globs=args.glob,
                include_runs=args.include_runs,
                max_bytes=args.max_bytes,
                max_files=0 if args.all else args.max_files,
                ignore_case=args.ignore_case,
                fixed_strings=args.fixed_strings,
            )
            output = render_files(result)
        else:
            result = search_lines(
                args.pattern,
                root,
                globs=args.glob,
                include_runs=args.include_runs,
                max_bytes=args.max_bytes,
                max_matches=0 if args.all else args.max_matches,
                max_per_file=0 if args.all else args.max_per_file,
                ignore_case=args.ignore_case,
                fixed_strings=args.fixed_strings,
            )
            output = render_lines(result)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
