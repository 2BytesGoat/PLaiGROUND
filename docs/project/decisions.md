# Decisions

Dated decisions for PLaiGROUND, newest last. Earlier decisions live in the
dated audits under `docs/audit/`; this log starts with the repo-brief
migration. Each entry says what changed and links the evidence it came from.

## 2026-11-02 — Adopt repo-brief as the map and context layer

**Context.** The repo had two local tools of its own:
`tools/repo_index.py` regenerated `docs/code-map.md` plus a gitignored
`.cache/repo-index.json`, and `tools/context.py` ranked that cache, including
`runs/`. Both Godot repos now vendor `repo-brief` at
`tools/vendor/repo-brief`, so this repo was maintaining a second implementation
of the same idea.

**Decision.** `repo-brief` is the only map and context layer here.
`make index` writes the committed `.repo-brief/map.md`; `make context` ranks
committed code, experiments, docs and decisions; `make preflight` verifies the
map, profile, front doors, dates and references. The old indexer, the old
context tool, their test and the generated `docs/code-map.md` move to
`archive/2026-11-02-repo-brief/`.

**Tradeoff.** `make context` no longer ranks `runs/`. That tree is local,
gitignored, and about 43 GB; a committed map cannot include it without
becoming machine-specific. `make brief` and `make run` remain the run view.

**Evidence.** [C-2026-11-02-01](../meta/capture/2026-11-02.md#c-2026-11-02-01)
— the owner asked for repo-brief here and chose the full swap.
