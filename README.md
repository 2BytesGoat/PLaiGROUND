# PLaiGROUND

PLaiGROUND (name may change) is the external Python/RL lab for **Dragon Jump**.
It connects to the game's training mode over a local TCP socket and trains
policies with NEAT, PPO, supervised distillation, or DQN. It is a
development/research repo, not part of the shipped game.

> **Current state (2026-10-28):** see [`docs/status.md`](docs/status.md).
> **Front door:** [`docs/index.md`](docs/index.md).
> **Known traps:** [`docs/known-issues.md`](docs/known-issues.md).
> **Script map:** [`src/README.md`](src/README.md).
> **Agent guidance:** [`AGENTS.md`](AGENTS.md).
>
> The model actually installed in Godot is documented in
> [`docs/reference/deployed-model-58.md`](docs/reference/deployed-model-58.md).
> The rest of this README is setup and orientation; older experiment sections
> have moved into `docs/` and `archive/`.

## Repository map

```text
src/plaiground/      importable training/data/evaluation package (see src/README.md)
scripts/             thin baselines, genetic, supervised, reinforcement entry points
tools/               route filtering, start-pool building, checks, reporting
experiments/         committed experiment manifests and results
docs/                index, status, architecture, reference, audit
runs/                generated artifacts (gitignored)
archive/             historical scripts/runs/docs (start at its README.md)
environments/        game binary / macOS source shim (contents gitignored)
notebooks/           tutorial notebooks (not the active training path)
data/                small tutorial dataset
```

## Start here

| I want to… | Go to |
|---|---|
| See what is active and what has been verified | [`docs/status.md`](docs/status.md) |
| Find a script or run artifact | [`src/README.md`](src/README.md) → [`docs/reference/run-artifacts.md`](docs/reference/run-artifacts.md) |
| See what experiments exist and which game commit they used | [`experiments/README.md`](experiments/README.md) |
| Understand the game ↔ Python protocol | [`docs/architecture.md`](docs/architecture.md) |
| Read the DQN plan and review | [`docs/experiments/dqn-route-spawns.md`](docs/experiments/dqn-route-spawns.md) → [`docs/audit/2026-10-28-dqn-pipeline-review.md`](docs/audit/2026-10-28-dqn-pipeline-review.md) |
| Read the PPO/distillation docs | [`docs/experiments/`](docs/experiments/) |
| Read a term I don't know | [`docs/reference/glossary.md`](docs/reference/glossary.md) |

## Setup

The upstream project documents **Linux/Windows** local setup and Docker. This
checkout also has a macOS source shim, so local macOS experiments work without
a container.

### Local (Linux/Windows)

1. Install Python 3.13.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
   ```
3. Install the package and dependencies in editable mode:
   ```bash
   uv pip install --python .venv/bin/python -e .
   # or: python -m pip install poetry && poetry install
   ```
4. Put the exported game binary in `environments/` (see `src/.config` for
   `ENV_PATH` / `ENV_NAME`).

### macOS (source shim)

`environments/DragonJump.app/Contents/MacOS/DragonJump` is a zsh shim that
execs `/Applications/Godot.app/Contents/MacOS/Godot` with
`--path ../dragon-jump-remaster/godot`. It lets the local `.venv` train against
the game source. It is **not** an exported app and does not ship.

### Docker (portable)

```bash
chmod +x run_docker.sh
./run_docker.sh
```

Then open <http://127.0.0.1:8888/lab?token=plaiground>. Put the Linux game
binary in `environments/` as described in the upstream setup.

### Shared config footgun

`src/.config` is shared by the older notebook/wrapper path and ships
`ACTION_REPEAT=0`, `LEVEL=1-3`, `SPEEDUP=1`, `SHOW_WINDOW=TRUE`. Active drivers
override these; a new script that trusts the file will hit the game's `% 0`
crash. Prefer explicit CLI flags.

## Run

Smoke test the game/socket setup:

```bash
.venv/bin/python scripts/baselines/00_random.py
```

NEAT, 1-1 (the current reference path uses the 8×24 persistent evaluator):

```bash
.venv/bin/python scripts/genetic/01_neat.py \
  --generations 2 --population 192 --parallel 8 \
  --max-steps 150 --action-repeat 4 --speedup 8 \
  --record-paths \
  --output-dir runs/neat_1_1_smoke --port 15500
```

PPO, 1-1 (the manifest owns levels, env shape, PPO hyperparameters, and paths):

```bash
.venv/bin/python scripts/reinforcement/01_ppo.py \
  --config experiments/0001_ppo_clean_start_1_1

# Smoke override:
.venv/bin/python scripts/reinforcement/01_ppo.py \
  --config experiments/0001_ppo_clean_start_1_1 \
  --timesteps 10000 --output-dir runs/ppo_1_1_smoke
```

DQN clean-start (1-1):

```bash
.venv/bin/python tools/extract_route_starts.py \
  --config experiments/0003_dqn_clean_start_1_1
.venv/bin/python scripts/reinforcement/02_dqn.py \
  --config experiments/0003_dqn_clean_start_1_1
.venv/bin/python tools/evaluate.py \
  --config experiments/0003_dqn_clean_start_1_1 \
  --model runs/dqn_clean_start_1_1/dqn/dqn_final.zip
```

Do not use the old 45-flag PPO command or `02_dqn.py extract/train/eval`; the
active entry points are config-driven.

## Current results (selected)

| Track | Level(s) | Result |
|---|---|---|
| NEAT 8×24 | 1-1 / 1-2 / 1-3 | 2.15 / 3.0167 / 3.5667 s |
| 754 PPO multilevel, 200k | 1-1 / 1-2 / 1-3 | 1.6667 / 2.9667 / 5.2333 s; full 242-start coverage |
| 794 sensor PPO | 1-1 / 1-2 | 2.1333 / 3.0667 s |
| 794 sensor PPO | 1-3 | not solved (canonical 0/1) |
| 794 distilled | 1-3 | 3.9 s canonical; 43/46 medium, 25/27 hard |
| Deployed 58/256 | 1-1…1-16 | 16/16, 100/100, 78.583 s sum; 1-1 2.333 s |

These are baselines/reference results, not installed models. The game ships the
58/256 JSON student; 754/794 models need a runtime/export path and the full
16-level gate. See [`docs/status.md`](docs/status.md) for the current state.

## Notebooks and tutorials

`notebooks/` and `src/plaiground/legacy/src/plaiground/legacy/processing/` are the tutorial/notebook path (data
exploration, decision trees, random forests, early neural nets). They are not
the active training pipeline. The active scripts are in `src/` and documented in
[`src/README.md`](src/README.md).

## Historical work

Historical scripts, runs, and the experiment log are in
[`archive/2026-10-27-cleanup/`](archive/2026-10-27-cleanup/README.md). The
`archive/runs/` tree is gitignored and large; do not add it to git.

## Credits

This repo is an amalgamation of information from many people. Special thanks to:

- **Ed Beeching** — [GodotRL](https://github.com/edbeeching/godot_rl_agents);
  the socket client is derived from that project.
- **Heartbeast** — Godot tutorials.
- **Sentdex** — the first ML/vision tutorials that got this started.

## Contact

2BytesGoat — [Discord](https://discord.gg/FsKQPupcVs)
