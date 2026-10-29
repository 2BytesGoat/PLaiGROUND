#!/usr/bin/env python3
"""Return a compact context pack for a free-text task.

``tools/repo_index.py`` builds the index; this tool ranks the relevant code
symbols, experiments, docs, and local run artifacts. It is read-only and is the
first search call for an agent, before a repo-wide grep.

Usage:
    .venv/bin/python tools/context.py "1-5 expert replay pool"
    .venv/bin/python tools/context.py --brief "ddqn expert buffer"
    .venv/bin/python tools/context.py --json "time_to_exit start pool"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "repo-index.json"

DOC_BOOSTS = {
    "docs/status.md": 30,
    "docs/index.md": 20,
    "docs/known-issues.md": 15,
    "docs/reference/run-artifacts.md": 10,
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "how",
    "does",
    "do",
    "is",
    "are",
    "was",
    "were",
    "what",
    "where",
    "why",
    "when",
    "which",
    "who",
    "this",
    "that",
    "it",
    "its",
    "from",
    "into",
    "by",
    "as",
    "at",
    "be",
    "can",
    "should",
    "would",
    "could",
    "we",
    "i",
    "you",
    "they",
    "our",
    "your",
    "their",
    "not",
    "no",
    "if",
    "then",
    "than",
    "about",
    "code",
    "file",
    "files",
    "thing",
    "things",
    "please",
    "help",
    "need",
    "want",
}

IDENT_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(query: str) -> tuple[list[str], list[str]]:
    raw = query.lower()
    tokens: list[str] = []
    phrases: list[str] = []
    for match in IDENT_RE.finditer(raw):
        value = match.group(0).strip("-_")
        if not value or value in STOPWORDS:
            continue
        tokens.append(value)
        if "-" in value or "_" in value or any(ch.isdigit() for ch in value):
            phrases.append(value)
        if "-" in value or "_" in value:
            for part in re.split(r"[-_]", value):
                if len(part) >= 3 and part not in STOPWORDS:
                    tokens.append(part)
    seen: set[str] = set()
    unique_tokens = [item for item in tokens if not (item in seen or seen.add(item))]
    unique_phrases = [
        item for item in dict.fromkeys(phrases) if item not in STOPWORDS
    ]
    return unique_tokens, unique_phrases


def count_hits(text: str, token: str) -> int:
    return min(text.count(token), 5)


def score_text(
    text: str,
    tokens: list[str],
    phrases: list[str],
    *,
    name: str = "",
    name_weight: int = 6,
    body_weight: int = 1,
) -> int:
    lower = text.lower()
    score = 0
    for token in tokens:
        score += count_hits(lower, token) * body_weight
    for phrase in phrases:
        score += count_hits(lower, phrase) * body_weight * 4
    if name:
        name_lower = name.lower()
        for token in tokens:
            if token in name_lower:
                score += name_weight
        for phrase in phrases:
            if phrase in name_lower:
                score += name_weight * 3
    return score


def truncate(value: str, limit: int = 180) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > limit:
        return value[: limit - 1].rstrip() + "…"
    return value


def load_index() -> dict[str, Any]:
    if not CACHE.exists():
        raise SystemExit(
            "missing .cache/repo-index.json; run `make index` first"
        )
    return json.loads(CACHE.read_text(encoding="utf-8"))


def best_doc_section(
    doc: dict[str, Any], tokens: list[str], phrases: list[str]
) -> dict[str, Any] | None:
    text = doc.get("text", "")
    lines = text.splitlines()
    headings = doc.get("headings", [])
    candidates: list[tuple[int, str, int, str]] = []
    for index, heading in enumerate(headings):
        start = max(0, int(heading["line"]) - 1)
        end = (
            max(start + 1, int(headings[index + 1]["line"]) - 1)
            if index + 1 < len(headings)
            else len(lines)
        )
        section = "\n".join(lines[start:end])
        score = score_text(
            section,
            tokens,
            phrases,
            name=heading["text"],
            name_weight=4,
            body_weight=1,
        )
        if score > 0:
            excerpt = ""
            for line in lines[start + 1 : end]:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    excerpt = stripped
                    break
            candidates.append(
                (score, heading["text"], int(heading["line"]), excerpt)
            )
    if not candidates:
        score = score_text(text, tokens, phrases, body_weight=1)
        if score <= 0:
            return None
        return {
            "score": score,
            "heading": doc.get("title", ""),
            "line": 1,
            "excerpt": doc.get("summary", ""),
        }
    score, heading, line, excerpt = max(candidates, key=lambda item: item[0])
    return {
        "score": score,
        "heading": heading,
        "line": line,
        "excerpt": excerpt,
    }


def search_docs(
    index: dict[str, Any],
    tokens: list[str],
    phrases: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    results = []
    for doc in index.get("docs", []):
        section = best_doc_section(doc, tokens, phrases)
        if section is None:
            continue
        path_score = score_text(
            doc["path"], tokens, phrases, name=doc["path"], name_weight=3
        )
        score = section["score"] + path_score
        score += DOC_BOOSTS.get(doc["path"], 0)
        results.append(
            {
                "path": doc["path"],
                "title": doc.get("title", ""),
                "line": section["line"],
                "heading": section["heading"],
                "excerpt": truncate(section["excerpt"], 220),
                "score": score,
            }
        )
    results.sort(key=lambda item: (-item["score"], item["path"]))
    return results[:limit]


def search_code(
    index: dict[str, Any],
    tokens: list[str],
    phrases: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for symbol in index.get("symbols", []):
        name = symbol["name"]
        parent = symbol.get("parent") or ""
        display_name = f"{parent}.{name}" if parent else name
        score = score_text(
            " ".join(
                [
                    display_name,
                    symbol["path"],
                    symbol.get("doc", ""),
                    " ".join(symbol.get("args", [])),
                ]
            ),
            tokens,
            phrases,
            name=display_name,
            name_weight=10,
            body_weight=2,
        )
        if score <= 0:
            continue
        candidates.append(
            {
                "path": symbol["path"],
                "line": symbol["line"],
                "name": display_name,
                "kind": symbol["kind"],
                "summary": truncate(symbol.get("doc", ""), 160),
                "score": score,
            }
        )

    # Module-level entries keep a query like "expert replay buffer" useful even
    # when the matching function has no docstring; the module docstring and file
    # name still identify the right tool.
    for module in index.get("modules", {}).values():
        score = score_text(
            " ".join(
                [
                    module["path"],
                    module.get("doc", ""),
                    " ".join(item["name"] for item in module["classes"]),
                    " ".join(item["name"] for item in module["functions"]),
                ]
            ),
            tokens,
            phrases,
            name=module["path"],
            name_weight=14,
            body_weight=2,
        )
        if score <= 0:
            continue
        candidates.append(
            {
                "path": module["path"],
                "line": 1,
                "name": module["path"],
                "kind": "module",
                "summary": truncate(module.get("doc", ""), 160),
                "score": score,
            }
        )

    candidates.sort(key=lambda item: (-item["score"], item["path"], item["line"]))
    deduped = []
    seen = set()
    for item in candidates:
        key = (item["path"], item["line"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def search_experiments(
    index: dict[str, Any],
    tokens: list[str],
    phrases: list[str],
    limit: int,
    levels: list[str] | None = None,
) -> list[dict[str, Any]]:
    results = []
    level_tokens = levels or []
    for item in index.get("experiments", []):
        score = score_text(
            item.get("search_text", ""),
            tokens,
            phrases,
            name=f"{item['id']} {item['name']}",
            name_weight=8,
            body_weight=1,
        )
        if level_tokens:
            item_levels = {str(level).replace("-", "_") for level in item.get("levels", [])}
            normalized_name = f"{item['id']}_{item['name']}".replace("-", "_")
            for level in level_tokens:
                normalized_level = level.replace("-", "_")
                if normalized_level in item_levels:
                    score += 40
                if normalized_level in normalized_name:
                    score += 20
        if score <= 0:
            continue
        results.append(
            {
                "id": item["id"],
                "name": item["name"],
                "status": item.get("status", ""),
                "workflow": item.get("workflow", ""),
                "levels": item.get("levels", []),
                "selected_checkpoint": item.get("selected_checkpoint", ""),
                "result": truncate(item.get("result", ""), 240),
                "readme": item.get("readme", ""),
                "config": item.get("config", ""),
                "score": score,
            }
        )
    results.sort(key=lambda item: (-item["score"], item["id"]))
    return results[:limit]


def _run_excerpt(item: dict[str, Any]) -> str:
    parts = []
    archive = item.get("archive", {})
    if archive:
        pieces = []
        if archive.get("routes") is not None:
            pieces.append(f"routes={archive['routes']}")
        if archive.get("clean") is not None:
            pieces.append(f"clean={archive['clean']}")
        if archive.get("best_clean") is not None:
            pieces.append(f"best_clean={archive['best_clean']:.4f}")
        history = archive.get("history", {})
        if history.get("generation") is not None:
            pieces.append(f"generation={history['generation']}")
        if history.get("finishes") is not None:
            pieces.append(f"finishes={history['finishes']}")
        if pieces:
            parts.append("archive " + ", ".join(pieces))
    evals = item.get("evals", [])
    if evals:
        latest = evals[-1]
        canonical = latest.get("canonical", {})
        canonical_bits = []
        for level, summary in sorted(canonical.items()):
            if not isinstance(summary, dict):
                continue
            canonical_bits.append(
                f"{level} {summary.get('finishes', 0)}/{summary.get('agents', 0)}"
                f" {summary.get('best_time')}"
            )
        if canonical_bits:
            parts.append(f"eval {latest['tag']}: " + "; ".join(canonical_bits))
    if item.get("key_files"):
        parts.append("files: " + ", ".join(item["key_files"][:6]))
    return truncate(" | ".join(parts), 260)


def search_runs(
    index: dict[str, Any],
    tokens: list[str],
    phrases: list[str],
    limit: int,
    levels: list[str] | None = None,
) -> list[dict[str, Any]]:
    results = []
    level_tokens = levels or []
    for item in index.get("runs", []):
        score = score_text(
            item.get("search_text", ""),
            tokens,
            phrases,
            name=item["name"],
            name_weight=8,
            body_weight=1,
        )
        if level_tokens:
            normalized = f"{item['name']} {item.get('search_text', '')}".replace("-", "_")
            item_levels = {
                str(level).replace("-", "_") for level in item.get("levels", [])
            }
            for level in level_tokens:
                normalized_level = level.replace("-", "_")
                if normalized_level in item_levels:
                    score += 40
                if normalized_level in normalized:
                    score += 30
        if score <= 0:
            continue
        results.append(
            {
                "name": item["name"],
                "path": item["path"],
                "excerpt": _run_excerpt(item),
                "score": score,
            }
        )
    results.sort(key=lambda item: (-item["score"], item["name"]))
    return results[:limit]


def build_context(query: str, limit: int) -> dict[str, Any]:
    index = load_index()
    tokens, phrases = tokenize(query)
    levels = [
        token for token in tokens if re.fullmatch(r"\d+-\d+", token)
    ]
    return {
        "query": query,
        "tokens": tokens,
        "phrases": phrases,
        "levels": levels,
        "stats": index.get("stats", {}),
        "code": search_code(index, tokens, phrases, limit),
        "experiments": search_experiments(
            index, tokens, phrases, limit, levels=levels
        ),
        "docs": search_docs(index, tokens, phrases, limit),
        "runs": search_runs(index, tokens, phrases, limit, levels=levels),
    }


def next_steps(context: dict[str, Any]) -> list[str]:
    steps = []
    for item in context["experiments"][:2]:
        if item.get("readme"):
            steps.append(f"read {item['readme']}")
        if item.get("selected_checkpoint"):
            steps.append(f"evaluate {item['selected_checkpoint']}")
    for item in context["code"][:2]:
        steps.append(f"read {item['path']}:{item['line']}")
    for item in context["runs"][:2]:
        steps.append(f"inspect {item['path']}")
    if not steps:
        steps.append('try a broader query, or run `make index` if the tree changed')
    return steps[:4]


def print_context(context: dict[str, Any], brief: bool) -> None:
    stats = context["stats"]
    print(f'Context: "{context["query"]}"')
    print(
        f'Index: {stats.get("modules", 0)} modules, '
        f'{stats.get("experiments", 0)} experiments, '
        f'{stats.get("docs", 0)} docs, {stats.get("runs", 0)} runs'
    )
    print()

    sections = (
        ("Code", "code", ("path", "line", "name", "summary")),
        ("Experiments", "experiments", ("id", "name", "status", "detail")),
        ("Docs", "docs", ("path", "line", "heading", "excerpt")),
        ("Runs", "runs", ("path", "excerpt")),
    )
    for title, key, fields in sections:
        items = context[key]
        print(title)
        if not items:
            print("  (no strong matches)")
            print()
            continue
        for item in items:
            if key == "code":
                print(
                    f"  {item['path']}:{item['line']}  {item['name']}"
                    f"  ({item['kind']})"
                )
                if item["summary"] and not brief:
                    print(f"      {item['summary']}")
            elif key == "experiments":
                print(
                    f"  {item['id']}  {item['name']}  [{item['status']}]"
                )
                detail = []
                if item["workflow"]:
                    detail.append(item["workflow"])
                if item["levels"]:
                    detail.append("levels=" + ",".join(item["levels"]))
                if item["selected_checkpoint"]:
                    detail.append(
                        "selected=" + item["selected_checkpoint"].split("/")[-1]
                    )
                if detail:
                    print("      " + "  ".join(detail))
                if item["result"] and not brief:
                    print(f"      {truncate(item['result'], 220)}")
            elif key == "docs":
                print(
                    f"  {item['path']}:{item['line']}  {item['heading']}"
                )
                if item["excerpt"] and not brief:
                    print(f"      {item['excerpt']}")
            elif key == "runs":
                print(f"  {item['path']}")
                if item["excerpt"] and not brief:
                    print(f"      {item['excerpt']}")
        print()

    print("Next")
    for step in next_steps(context):
        print(f"  {step}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="free-text task or symbol query")
    parser.add_argument("--brief", action="store_true", help="one-screen output")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="maximum results per section (default: 5; --brief uses 3)",
    )
    args = parser.parse_args()

    limit = min(args.limit, 3) if args.brief else args.limit
    context = build_context(" ".join(args.query), limit)
    if args.json:
        print(json.dumps(context, indent=2, sort_keys=True))
    else:
        print_context(context, args.brief)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
