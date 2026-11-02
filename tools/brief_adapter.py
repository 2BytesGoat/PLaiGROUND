"""Adapter for PLaiGROUND.

The generic adapter knows docs and a top-level backlog. This repo's living
record is different: open work is `## <title> — <severity>` sections in
`docs/known-issues.md` and the numbered "Next 3" list in `docs/now.md`.

Two trees must never be walked: `runs/` and `archive/runs/` are tens of
gigabytes of generated experiment output, and `environments/` is a binary
shim. They are artifacts of running the lab, not part of the repository, so
they are skipped rather than indexed and hidden.

The kernel does every bit of ranking. Return structure here, never scores.
"""
from __future__ import annotations

import re
from pathlib import Path

from repobrief.adapters.generic import GenericAdapter
from repobrief.contract import Item, Source
from repobrief.detect import SKIP_DIRS

# `runs` also covers `archive/runs`, which is the 6 GB tail of the old tree.
SKIP = SKIP_DIRS | {"runs", "environments"}

SOURCE_GLOBS = (
    "src/plaiground/**/*.py",
    "scripts/**/*.py",
    "tools/*.py",
    "tests/**/*.py",
    "experiments/**/*.md",
    "experiments/**/config.toml",
    "docs/**/*.md",
    "*.md",
)

KNOWN_ISSUES = "docs/known-issues.md"
NOW = "docs/now.md"

ISSUE_RE = re.compile(r"^## (.+?)\s+—\s+(critical|high|medium|low|deferred)\s*$", re.M)
NEXT_RE = re.compile(r"^## Next 3\s*$(.*?)(?=^## |\Z)", re.M | re.S)
NUMBERED_RE = re.compile(r"^(\d+)\.\s+(.+)$", re.M)


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


class Adapter(GenericAdapter):
    name = "plaiground"
    skip_dirs = SKIP

    def sources(self, root: Path) -> list[Source]:
        found = []
        seen = set()
        for pattern in SOURCE_GLOBS:
            for path in sorted(root.glob(pattern)):
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                if rel in seen or set(Path(rel).parts) & SKIP:
                    continue
                seen.add(rel)
                if rel.startswith("tests/"):
                    kind = "test"
                elif rel.endswith(".md"):
                    kind = "docs"
                else:
                    kind = "source"
                found.append(Source(path=rel, kind=kind))
        return found

    def canonical(self, root: Path) -> list[str]:
        # Generic appends docs/meta/writing.md when it exists.
        return [
            "docs/index.md",
            "docs/now.md",
            "docs/status.md",
            "docs/known-issues.md",
            "docs/reference/level-ledger.md",
            "docs/project/decisions.md",
        ]

    def dated(self, root: Path) -> list[str]:
        return [NOW, "docs/status.md"]

    def open_work(self, root: Path) -> list[Item]:
        items = super().open_work(root)
        items.extend(self._known_issues(root))
        items.extend(self._next_three(root))
        return items

    def _known_issues(self, root: Path) -> list[Item]:
        path = root / KNOWN_ISSUES
        if not path.is_file():
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(ISSUE_RE.finditer(text))
        items = []
        for position, match in enumerate(matches):
            end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
            body = text[match.end():end]
            items.append(
                Item(
                    path=KNOWN_ISSUES,
                    line=_line(text, match.start()),
                    text=match.group(1).strip(),
                    body=re.sub(r"\s+", " ", body).strip(),
                    kind="open",
                    group="Known issues",
                    label=match.group(2),
                )
            )
        return items

    def _next_three(self, root: Path) -> list[Item]:
        path = root / NOW
        if not path.is_file():
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
        section = NEXT_RE.search(text)
        if not section:
            return []
        body = section.group(1)
        matches = list(NUMBERED_RE.finditer(body))
        items = []
        for position, match in enumerate(matches):
            end = matches[position + 1].start() if position + 1 < len(matches) else len(body)
            title = match.group(2).replace("**", "").strip()
            detail = re.sub(r"\s+", " ", body[match.end():end]).strip()
            items.append(
                Item(
                    path=NOW,
                    line=_line(text, section.start(1) + match.start()),
                    text=title,
                    body=detail,
                    kind="open",
                    group="Next 3",
                )
            )
        return items
