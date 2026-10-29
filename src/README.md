# Source layout

`src/` is the installable Python package root. Entry points live outside it.

```text
scripts/        thin user-facing CLIs
tools/          data/evaluation utilities
src/plaiground/ importable library
experiments/    committed experiment manifests and result notes
runs/           generated artifacts (gitignored)
```

The package is installed editable once:

```bash
uv pip install --python .venv/bin/python -e . --no-deps
```

## Package map

| Path | Responsibility |
|---|---|
| `src/plaiground/envs/` | `GodotEnv`, Gymnasium/SB3 adapters, shared factory, runtime/launcher, legacy wrapper. |
| `src/plaiground/models/extractors/` | Frozen legacy grid/autoencoder feature extractors. |
| `src/plaiground/observations/` | Observation contract (`spec.py`), lean decoder, sklearn-style preprocessing, `lean_io.py`. |
| `src/plaiground/data/` | Route filtering, strict/near/all route tiers, clean-start pool construction, route extraction, and shared start/pool loaders. |
| `src/plaiground/models/` | Model architectures (`LeanMLP`) and frozen legacy extractors. |
| `src/plaiground/algorithms/genetic/` | NEAT population, all-level, and lineage trainers. |
| `src/plaiground/algorithms/reinforcement/` | Thin PPO/DQN/Double-DQN adapters (`ppo.py`, `dqn.py`, `ddqn.py`). |
| `src/plaiground/algorithms/supervised/` | Reserved for behavior cloning, DAgger, and distillation workflows. |
| `src/plaiground/evaluation/` | Shared deterministic evaluator (`rollout.py`). |
| `src/plaiground/experiments/` | Experiment TOML loading, run resolution (`runner.py`), and provenance capture. |
| `src/plaiground/legacy/` | Notebook/course compatibility path; not used by clean-start training. |

## Entry points

| Script | Purpose |
|---|---|
| `scripts/baselines/00_random.py` | Random-action smoke test. |
| `scripts/baselines/01_heuristic.py` | Course-era if/else heuristic. |
| `scripts/genetic/01_neat.py` | NEAT population train/evaluate/play. |
| `scripts/genetic/02_neat_all_levels.py` | Independent population per level. |
| `scripts/genetic/03_neat_lineage.py` | One population carried across levels. |
| `scripts/reinforcement/01_ppo.py` | PPO train from an experiment manifest. |
| `scripts/reinforcement/02_dqn.py` | DQN train from an experiment manifest. |
| `scripts/reinforcement/03_ddqn.py` | Double-DQN train from an experiment manifest. |
| `tools/level.py` | Route tiers, filtered pools, eval union, and per-level status. |
| `tools/evaluate.py` | Shared PPO/DQN canonical + start-set evaluation. |
| `tools/repo_index.py` | Generate `docs/code-map.md` and `.cache/repo-index.json`. |
| `tools/context.py` | Rank code, experiments, docs, and runs for a task query. |
| `tools/extract_route_starts.py` | NEAT checkpoint -> clean-start DQN pool. |
| `tools/filter_routes.py` | Route dedupe/clustering/holdout split. |
| `tools/build_start_pool.py` | Validated train/holdout start pools. |
| `tools/check_observation.py` | Raw observation smoke check. |
| `tools/watch.py` | Preview a PPO or DQN checkpoint in the game window. |
| `tools/summarize_episodes.py` | Episode CSV summaries. |
| `tools/start_strata.py` | Legacy start stratification. |
| `tools/make_ppo_start_pool.py` | Legacy prefix/strata pool builder. |

## Rules

- Reinforcement entry points are config-driven: pass `--config` and let the
  manifest own model, environment, data, and evaluation details.
- Do not add new numbered scripts under `src/`.
- Scripts are thin; reusable logic belongs in `src/plaiground/`.
- New experiments go in `experiments/<id>_<name>/` with `README.md` and
  `config.toml`; generated outputs go in `runs/`.
- Every experiment records the game branch, commit, dirty state, and any dirty
  diff needed to reproduce it.
- Run tests with `.venv/bin/python -m unittest discover -s tests`.
- Do not edit `archive/`; it is history.
