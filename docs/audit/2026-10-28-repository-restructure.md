# Repository restructure

Date: 2026-10-28
Status: implemented
Scope: PLaiGROUND Python tree only; game repo untouched

## Why

The active tree mixed chronological course scripts, reusable library code,
data post-processing, training logic, and generated artifacts. The new layout
separates:

- **library code** in `src/plaiground/`;
- **thin entry points** in `scripts/`;
- **data/evaluation utilities** in `tools/`;
- **committed experiment records** in `experiments/`;
- **generated artifacts** in `runs/` (gitignored).

## Target layout

```text
src/plaiground/
  baselines/          random + course heuristic
  envs/               Godot socket, SB3 adapters, vector env
  observations/       spec, lean decoder, preprocessing pipeline
  data/               route filtering and start pools
  models/             LeanMLP and legacy frozen extractors
  algorithms/
    genetic/neat/     population, all-levels, lineage
    supervised/       BC/DAgger/distillation (reserved)
    reinforcement/    PPO and DQN
  evaluation/         strata evaluation
  experiments/        TOML loading and provenance
  legacy/             notebook/course compatibility

scripts/
  baselines/00_random.py
  baselines/01_heuristic.py
  genetic/01_neat.py
  genetic/02_neat_all_levels.py
  genetic/03_neat_lineage.py
  supervised/         reserved
  reinforcement/01_ppo.py
  reinforcement/02_dqn.py
  evaluation/01_evaluate_strata.py

tools/                route filter, start pool, checks, reports
experiments/          committed manifests and results
```

## Migration map

| Old | New |
|---|---|
| `src/00_random_agent.py` | `scripts/baselines/00_random.py` |
| `src/07_neat_agent.py` | `src/plaiground/algorithms/genetic/neat/trainer.py` + `scripts/genetic/01_neat.py` |
| `src/08_neat_all_levels.py` | `src/plaiground/algorithms/genetic/neat/all_levels.py` + `scripts/genetic/02_neat_all_levels.py` |
| `src/09_neat_lineage.py` | `src/plaiground/algorithms/genetic/neat/lineage.py` + `scripts/genetic/03_neat_lineage.py` |
| `src/10_ppo_pilot.py` | `src/plaiground/algorithms/reinforcement/ppo.py` + `scripts/reinforcement/01_ppo.py` |
| `src/41_dqn_route_spawns.py` | `src/plaiground/algorithms/reinforcement/dqn.py` + `scripts/reinforcement/02_dqn.py` |
| `src/obs_spec.py` | `src/plaiground/observations/spec.py` |
| `src/preprocessing/` | `src/plaiground/observations/` |
| `src/models/` | `src/plaiground/models/` |
| `src/postprocessing/` | `src/plaiground/data/` |
| `src/wrappers/` | `src/plaiground/envs/` and `src/plaiground/models/extractors/` |
| `src/42_filter_clean_routes.py` | `tools/filter_routes.py` |
| `src/44_build_clean_start_pool.py` | `tools/build_start_pool.py` |
| `src/43_check_raw_obs.py` | `tools/check_observation.py` |
| `src/36_episode_stats.py` | `tools/summarize_episodes.py` |
| `src/32_clean_archive_coverage.py` | `tools/clean_archive_coverage.py` |
| `src/33_start_strata.py` | `tools/start_strata.py` |
| `src/34_make_ppo_start_pool.py` | `tools/make_ppo_start_pool.py` |
| `src/35_eval_ppo_strata.py` | `src/plaiground/evaluation/strata.py` + `scripts/evaluation/01_evaluate_strata.py` |

## Packaging

`pyproject.toml` now declares the `plaiground` package from `src/`. The local
venv was installed with:

```bash
uv pip install --python .venv/bin/python -e . --no-deps
```

Tests import the installed package instead of inserting `src/` into
`sys.path`.

## Experiments

New records live in:

```text
experiments/0000_neat_clean_450/
experiments/0001_ppo_clean_start_1_1/
```

Each records the exact game branch, commit, dirty state, configuration, data
inputs, command, and result status. Generated outputs remain under `runs/`.

## Verification

```text
Python tests:       28/28
Package imports:    passed
PPO entry point:    64-step lean smoke passed on separate ports
Neat entry point:   checkpoint play smoke passed
Filter/pool tools:  reproduced 216 routes and 13/10 states
Link check:         0 broken links
```

The active 1M PPO run uses the process that was already running before the
move; new invocations use the package entry point.

## Remaining work

- Extract `SingleAgentGodotEnv` and shared env factories out of the PPO module
  into `src/plaiground/envs/`.
- Add the supervised workflow implementations and their scripts.
- Add per-run provenance capture to the generic experiment runner.
- Remove the remaining legacy `.config` path only after the notebook path is
  retired.
