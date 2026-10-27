# `src/` — script map

Active drivers are numbered. The numbering is historical; gaps are archived
experiments. Run from the repository root with `.venv/bin/python`.

## Active drivers

| Script | Status | What it does | Typical entry point |
|---|---|---|---|
| `00_random_agent.py` | onboarding | Random-action smoke test through the legacy `StableBaselinesGodotEnv` path. Use it to verify the game/socket setup, not for training. | `.venv/bin/python src/00_random_agent.py` |
| `07_neat_agent.py` | active | NEAT population trainer/evaluator. Uses one Godot process with N agents; fitness is the game reward. Supports resume, persistent evaluator, `--record-paths`, `--target-time`, and play/visualise modes. | `... src/07_neat_agent.py --generations 2 --population 192 --parallel 8` |
| `08_neat_all_levels.py` | available | Thin batch layer over `07`: one independent population per campaign level, then a visible checkpoint replay. | `... src/08_neat_all_levels.py --levels 1-1,1-2 --workers 2` |
| `09_neat_lineage.py` | available | One population carried through levels in order, with zero-shot/post-stage evals and stage snapshots. | `... src/09_neat_lineage.py --levels all` |
| `10_ppo_pilot.py` | active | The main PPO pipeline: multi-level, multi-agent/process, start pools, prefix replay, eval patience, 754/794 encodings, sensor expansion, frozen encoders, trajectory saving. ~83 KB monolith. | `... src/10_ppo_pilot.py --level 1-1 --n-envs 8 --timesteps 100000` |
| `32_clean_archive_coverage.py` | active | Collapses NEAT finishers into clean routes, enumerates standable starts, runs top genomes from every start, writes coverage. | `... src/32_clean_archive_coverage.py` |
| `33_start_strata.py` | active | Converts standable starts into canonical/medium/hard/easy strata by BFS distance. | `... src/33_start_strata.py --level 1-1` |
| `34_make_ppo_start_pool.py` | active | Builds a weighted PPO start pool from strata + clean-route prefixes; optional focus cells. | `... src/34_make_ppo_start_pool.py --level 1-1` |
| `35_eval_ppo_strata.py` | active | Deterministic evaluation of a PPO model on canonical/medium/hard/focus starts. Reuses one env; this is the pattern DQN eval should follow. | `... src/35_eval_ppo_strata.py --model runs/.../best.zip` |
| `36_episode_stats.py` | active | Summarises `episodes.csv` by start source; `--canonical-only` is the required speed view. | `... src/36_episode_stats.py runs/.../episodes.csv --canonical-only` |
| `41_dqn_route_spawns.py` | **active, no result** | Fresh 58-value Double DQN from NEAT route spawns. `extract` / `train` / `eval` / `all`. The first extract did not produce a pool; read the audit before running. | `... src/41_dqn_route_spawns.py extract` |

## Shared modules

| Path | Status | What it does |
|---|---|---|
| `obs_spec.py` | active | Python mirror of the game's observation schema; validates widths and labels fields. |
| `wrappers/godot_env.py` | active | Low-level TCP client; launches the game, encodes reset/spawns/state, decodes the columnar info schema and binary observation frames. |
| `wrappers/multi_agent_vec_env.py` | active | SB3 `VecEnv` for N agents in one Godot process, with per-agent auto-reset. |
| `wrappers/legacy_autoencoder.py` | experimental | Frozen 7×7 grid autoencoder for the legacy 58 observation. |
| `wrappers/legacy_grid_extractor.py` | experimental | Joint tile-embedding + CNN feature extractor for the legacy 58 observation. |
| `wrappers/semantic_autoencoder.py` | experimental | Frozen semantic MAE for the 754 observation. |
| `wrappers/stable_baselines_wrapper.py` | legacy | Older Gym wrapper used by `00_random_agent.py` / notebooks. Not used by active drivers. |
| `utils.py` | legacy | `setup_environment()` for the legacy wrapper/notebook path. |
| `processing/` | legacy | Notebook-era data processing/visualisation helpers. |

## Archived drivers

Historical scripts moved to
`archive/2026-10-27-cleanup/src/` include:

- `00_data_exploration.py`, `01_if_else_agent.py`, `02_record_gameplay.py`,
  `03_decision_tree.py`, `04_random_forest.py`, `playground.py` — tutorials;
- `11_ppo_lineage.py`, `12_ppo_distill.py`, `13_ppo_retrain_all.py`,
  `14_select_teachers.py`, `15_recovery_test.py`, `16_ppo_seed_restarts.py`,
  `17_multi_hot_teachers.py`, `18_state_size_matrix.py`,
  `19_agent_topology_bench.py`, `20_train_all_levels_8x8.py`,
  `21_curriculum_8x8.py` — old PPO lineage/teacher/topology/curriculum work;
- `22_train_legacy_autoencoder.py`, `23_generate_level_windows.py`,
  `24_train_semantic_autoencoder.py`, `25_train_semantic_head.py` — encoder
  experiments (wrappers remain in `src/wrappers/`);
- `30_trajectory_holdout.py` — full-state trajectory holdout; contains the
  `state_from_info()` helper used by the state-snapshot path;
- `38_self_distill.py`, `39_merge_distill.py`, `40_train_trajectories.py` —
  distillation/merge/trajectory drivers.

Do not edit archived scripts in place. Copy the file back to `src/` if you need
it, and document why.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

Current coverage is the socket/info protocol and binary observation frame
decoding. Add pure tests for pool construction/route selection before another
long DQN run.

## Adding a new experiment

1. Add `src/NN_<name>.py` with a module docstring that states the question,
   level, observation, reward, and output directory.
2. Reuse `GodotEnv` / `SingleAgentGodotEnv`; do not duplicate the socket client.
3. Add `docs/experiments/<name>.md` with status, plan, command, and results.
4. Add a row to this file and to `docs/index.md`; update `docs/status.md` when
   it starts/produces a result.
5. Keep generated output in `runs/<name>/`; do not commit it.
