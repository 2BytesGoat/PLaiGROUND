# AGENTS.md — PLaiGROUND

Guidance for coding agents and future maintainers working in this repository.

## Start here

Read these in order before changing anything:

1. [`docs/status.md`](docs/status.md) — current branch, active experiment, verified results, next work.
2. [`docs/runbooks/new-level.md`](docs/runbooks/new-level.md) — the full
   NEAT → expert → evaluate → repair procedure for a new level.
3. [`docs/reference/level-ledger.md`](docs/reference/level-ledger.md) — the
   selected expert and gate for each level.
4. [`docs/index.md`](docs/index.md) — task-based front door and repo map.
5. [`docs/architecture.md`](docs/architecture.md) — how the game ↔ Python stack fits together.
6. [`docs/known-issues.md`](docs/known-issues.md) — current traps, cross-repo drift, unfinished work.

The `README.md` and older experiment docs contain historical sections. When they
disagree with `docs/status.md` or `docs/index.md`, the docs win.

## Search and context

- Start a task with `make context q="<task>"` (or
  `.venv/bin/python tools/context.py "<task>"`). It ranks code symbols,
  experiments, docs, and local run artifacts before a repo-wide grep.
- `make index` regenerates `docs/code-map.md` (committed) and
  `.cache/repo-index.json` (local, gitignored). `make index-check` fails if the
  map is stale; run it after adding or renaming modules, experiments, or docs.
- Use `docs/code-map.md` for **where**, `docs/status.md` for **now**,
  `docs/index.md` for task routing, and `docs/known-issues.md` for traps. Grep
  only to confirm or fill a gap; prefer `grep -n` with a path/glob and `read`
  with `offset`/`limit` over `cat` or a repo-wide grep.
- `runs/` is local and gitignored. The index records top-level run metadata and
  eval summaries; use `docs/reference/run-artifacts.md` for the curated
  inventory.

## What this repo is

PLaiGROUND is the external Python/RL lab for **Dragon Jump**. It talks to the
game's training mode over a local TCP socket. It is **not** part of the shipped
game and is not linked into the Godot runtime.

The game source lives in a sibling repository:

```text
../dragon-jump-remaster/godot       # open this in the Godot editor
../dragon-jump-remaster/docs         # game-side architecture/training docs
```

## Where things live

| Path | What it is |
|---|---|
| `src/plaiground/` | Importable package (see [`src/README.md`](src/README.md)). |
| `scripts/` | Thin training and evaluation entry points. |
| `tools/` | Data, provenance, reporting, and repo search utilities. |
| `experiments/` | Committed experiment manifests and result notes. |
| `src/plaiground/envs/` | Game socket client, Gymnasium wrappers, shared env factory, SB3 VecEnv. |
| `src/plaiground/models/extractors/` | Frozen legacy feature extractors. |
| `src/plaiground/observations/spec.py` | Python mirror of the game's observation contract. |
| `src/plaiground/observations/lean_io.py` | Load/create the fitted lean preprocessing pipeline. |
| `src/plaiground/data/` | Route filtering, level route tiers, start-pool loading, and DQN route extraction. |
| `src/plaiground/data/level_data.py` | Strict/near/all-success classification, filtered routes, and train/holdout/eval pools. |
| `src/plaiground/evaluation/rollout.py` | Shared deterministic canonical/start-set evaluator. |
| `src/plaiground/experiments/runner.py` | Experiment config -> run paths and the common CLI. |
| `src/plaiground/models/` | Declared lean v1 model: `LeanMLP` 512×512 + SB3 policy kwargs. |
| `tools/level.py` | `data` (route tiers/pools) and `status` (per-level view). |
| `tools/evaluate.py` | Shared PPO/DQN evaluation entry point. |
| `tools/extract_route_starts.py` | NEAT checkpoint -> clean-start DQN pool. |
| `tests/` | Unit tests for the socket/info protocol, preprocessing, models, runner, evaluation, and route/pool handling. |
| `runs/` | Generated run artifacts. **Gitignored; do not delete or rewrite without asking.** |
| `archive/2026-10-27-cleanup/` | Historical scripts, runs, and docs moved out of the active tree. |
| `docs/experiments/` | Historical/deep-dive experiment documents. New records start in `experiments/`. |
| `docs/runbooks/` | Living procedures such as the new-level pipeline. |
| `docs/reference/` | Contracts, glossary, deployed model, level ledger, run-artifact inventory. |
| `docs/audit/` | Dated investigations and reviews. |
| `environments/` | Game binaries / macOS source shim. Gitignored contents except `.gitkeep`. |
| `notebooks/` | Tutorial notebooks, not the active training path. |
| `data/` | Small cheatsheet data used by tutorials. |

## Working rules

- Use `.venv/bin/python` (Python 3.13). Do not install into the system Python.
- Install the package editable with
  `uv pip install --python .venv/bin/python -e . --no-deps`, then run entry
  points from the repository root, e.g.
  `.venv/bin/python scripts/reinforcement/01_ppo.py --config experiments/<id>_<name>`.
- Reinforcement entry points are config-driven. Put experiment details in
  `experiments/<id>_<name>/config.toml`; the CLI only overrides run-level
  values (`--timesteps`, `--resume`, `--output-dir`, `--show`, `--device`,
  `--verbose`). Train and evaluate are separate: `tools/evaluate.py`.
- `runs/` is generated and gitignored. Record conclusions in `docs/status.md`,
  `docs/known-issues.md`, or a dated `docs/audit/` note instead of committing
  run artifacts.
- `archive/` is history. Do not edit archived scripts or runs; restore by
  copying a file back to its original path if needed.
- Keep `docs/status.md` current when an experiment starts, stops, or produces a
  result. Add a dated audit for a deep investigation instead of bloating status.
- **NEAT transfer rule:** when adding a new level N, seed route collection from
  the evolved population of the previous level (N-1), not from the original
  1-1 population. Evaluate/evolve on N with `--record-paths` until the clean
  route target is met, then filter/cluster and build the N start pool.
  Zero-shot transfer is a diagnostic only; a canonical-only pool is a fallback.
- **New-level pipeline:** after NEAT, run
  `.venv/bin/python tools/level.py data --level N --run-dir <run> ...` to
  create strict/near/all-success tiers, filtered routes, train/holdout pools,
  and the deduplicated eval union. Follow
  [`docs/runbooks/new-level.md`](docs/runbooks/new-level.md); record the
  selected model in [`docs/reference/level-ledger.md`](docs/reference/level-ledger.md).
- Do not move or rename scripts/docs without updating `README.md`,
  `docs/index.md`, `AGENTS.md`, and any import paths.
- Preserve user changes. Never run `git reset --hard`, `git checkout --`, or
  clean untracked files unless the user explicitly asks.
- Run `.venv/bin/python -m unittest discover -s tests` after changing the
  socket client, info decoding, wrappers, or level data tooling.

## Environment and game-side footguns

- On macOS, `environments/DragonJump.app` is a **source shim** that execs
  `/Applications/Godot.app/Contents/MacOS/Godot` against
  `../dragon-jump-remaster/godot`. It is not an exported app.
- `src/.config` is shared by the old notebook/wrapper path and ships
  `ACTION_REPEAT=0`, `LEVEL=1-3`, `SPEEDUP=1`, `SHOW_WINDOW=TRUE`. Active drivers
  override these; do not assume the file reflects the current experiment.
- The game's training mode is triggered by `--port`; do **not** put `--` before
  game args. Arguments after `--` are ignored by the parser.
- Each Godot process needs a unique port. Drivers use `base_port + rank`;
  evaluation uses `eval_port + index`. Port collisions look like hangs.
- Training processes get private `HOME`/`XDG` trees under the run directory so
  they never touch the player's real save, settings, ghosts, medals, or
  leaderboards. Keep that isolation.
- The game repo may be on a feature branch with a dirty tree. Record the exact
  branch/commit/dirty state when a result matters; docs/status tracks the
  current known state.
