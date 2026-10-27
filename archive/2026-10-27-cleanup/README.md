# 2026-10-27 experiment cleanup

This folder contains experiment scripts, models, run outputs, logs, and
documentation moved out of the active tree. Nothing was deleted.

## Active pipeline

> **Current state:** see `docs/status.md`. This section describes the active
> pipeline **at cleanup time (2026-10-27)**. The DQN route-spawn experiment was
> added on 2026-10-28 and is documented in
> `docs/experiments/dqn-route-spawns.md`; it has no result yet.

The active experiment branch at cleanup time was the 794-value sensor PPO route:

```text
NEAT routes
  -> coverage / clean routes
  -> start strata
  -> weighted PPO start pool
  -> PPO training
  -> deterministic stratified evaluation
```

Active scripts:

- `07_neat_agent.py`, `08_neat_all_levels.py`, `09_neat_lineage.py`
- `10_ppo_pilot.py`
- `32_clean_archive_coverage.py`
- `33_start_strata.py`
- `34_make_ppo_start_pool.py`
- `35_eval_ppo_strata.py`
- `36_episode_stats.py`

Keep `00_random_agent.py` as the README onboarding smoke test.

## Active runs

- `runs/ppo_multilevel_1_1_1_2_1_3/` — 754 north-star reference
- `runs/ppo_fresh_1_1_sensor_794/` — 794 1-1 reference
- `runs/ppo_sensor_1_2_from900k/` — accepted 794 1-2 transfer
- `runs/ppo_sensor_1_3_from1_2/` — 794 1-3 PPO lineage
- `runs/distill_1_3_merged_available/` — best 1-3 distilled candidate
- `runs/neat_1_1_8x24_archive/` — 1-1 NEAT archive, coverage, start pool
- `runs/neat_1_2_transfer_8x24/` — 1-2 NEAT archive, coverage, start pool
- `runs/neat_1_3_transfer_8x24/` — 1-3 NEAT archive, coverage, start pool

Added after cleanup (2026-10-28): `src/41_dqn_route_spawns.py` and
`runs/dqn_route_spawns_1_1/` (incomplete; see `docs/status.md`).

## Archived

- `src/` — superseded training drivers, topology/encoder/curriculum sweeps,
  sequential PPO lineage, old distillation drivers, and `playground.py`.
- `runs/` — 146 run directories, including all old PPO lineage, teacher,
  encoder, topology, spawn, smoke, and failed-distillation runs. The blind
  1-4 PPO run and its geometry-only 1-4 pool are here too.
- `runs/_logs/` — old root-level training/evaluation logs.
- `docs/model-improvement-experiments.md` — historical experiment log.

`archive/runs/` is ignored by git because it is large. The rest of this
archive remains visible to git.

## Restore

Move a file or run back to its original path, for example:

```bash
mv archive/2026-10-27-cleanup/src/12_ppo_distill.py src/
mv archive/2026-10-27-cleanup/runs/neat_lineage_r4b runs/
```
