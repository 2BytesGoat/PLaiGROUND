# Audit — DQN route-spawn pipeline review

**Date:** 2026-10-28
**Scope:** `src/41_dqn_route_spawns.py`, `docs/experiments/dqn-route-spawns.md`,
`runs/dqn_route_spawns_1_1/`, and the game-side reset/reward contract.
**Status:** review only; no training run, no game-code change.

## Why this exists

The DQN experiment was committed at `d7e173d` with a plan and a driver, but the
first extraction did not produce a spawn pool. This note records what is actually
on disk, what the driver does, and which design issues need to be fixed before a
100k-step run is worth starting.

## What is actually on disk

- Repo at review time: branch `neat`, clean at `d7e173d`
  (`checkpointing before i lose my mind`). The docs restructure in this pass is
  uncommitted.
- `runs/dqn_route_spawns_1_1/`:
  - `routes/1-1_clean_routes.json` — 460 clean routes (~6 MB);
  - `routes/1-1_finishers.jsonl` — raw finisher archive;
  - `godot/route_replay.log`, `godot/debug_window.log`;
  - `spawns/` — **empty**;
  - `userdata/route_replay/` — isolated HOME from the attempted run;
  - no `dqn/`, no `eval/`.
- `godot/route_replay.log` stops after:

  ```text
  Godot Engine v4.6.3.stable.official.7d41c59c4
  ...
  [Telemetry] menu_opened: { "menu_name": "practice", ... }
  TRAINING_SPEEDUP x8.0 (480 physics ticks/s)
  ```

  No error, no completion line.
- Timestamps: run dir created 00:14:50; `src/41_dqn_route_spawns.py` modified
  00:16:17; log last written 00:16:21; commit 00:19:19. The process was stopped
  while iterating, not after a completed extract.
- No Godot/Python process was running when checked.

**Conclusion:** the pipeline is written, the route data is staged, and there is
**no spawn pool, no DQN checkpoint, and no evaluation**.

## What the driver does

`extract()`:

1. Copies `clean_routes.json` and `finishers.jsonl` into the run dir.
2. Sorts routes by `run_time`, keeps the fastest 10% (46 of 460).
3. Greedy-clusters them by Jaccard similarity of `path_cells` at 0.80.
4. Keeps the fastest route per cluster; holds out 20% of clusters (excluding the
   fastest cluster).
5. Replays each selected route from the canonical start through the game,
   recording `cell`, `facing`, `on_floor`, `run_time`, `route_signature`,
   `route_time`, and `cluster` at each step.
6. Keeps a state if `on_floor`, `run_time > 0.02`, and
   `time_to_exit >= 0.5`; dedupes globally by `(cell, facing)`.
7. Writes a weighted pool: canonical 15%, route checkpoints share 85%.

`train()`:

- SB3 `DQN` (`MlpPolicy`, `net_arch=[256, 256]`), fresh init.
- Replay buffer 100k, learning starts 2k, batch 256, gamma 0.99, tau 1.0,
  target update 1k, train freq 4, one gradient step.
- `DummyVecEnv` with 4 envs, `VecMonitor`, checkpoints every 25k.
- Each env uses `SingleAgentGodotEnv` with `force_start_jump=True`,
  `reward_profile="time"`, `obs_encoding="legacy"`, and the route start pool.

`evaluate()`:

- Loads the final model; runs canonical plus every holdout spawn.
- Creates a new `GodotEnv` per spawn and manually presses jump once after reset.
- Writes `eval/summary.json`.

## Findings

### F1 — The `time` reward is not comparable across reset forms

The game's `time` finish term is:

```text
20 + 80 * clamp(gold_time / run_time, 0, 2.5)
```

`reset()` sets `_run_time = 0.0`; a cell/facing reset does not restore the
clock. Examples for 1-1 (gold 2.2944 s):

| Run | `run_time` at finish | Finish reward |
|---|---:|---:|
| Full canonical 2.15 s | 2.15 | ~105 |
| Full canonical 2.40 s | 2.40 | ~96 |
| Mid-route spawn, 0.5 s to exit | 0.5 | 220 (cap) |
| Mid-route spawn, 1.0 s to exit | 1.0 | ~203 |

With 85% route-spawn weight, the agent is rewarded for finishing quickly from
late checkpoints, not for the full-level time the gate measures. This is the
same failure mode that made the PPO start-pool experiments regress canonical
while looking good on sampled starts.

**Options:**

1. **Preferred:** record a grounded `state` snapshot at each checkpoint and
   reset with it. The game restores position, velocity, `run_time`,
   `best_run_time`, `run_active`, and `started_walking` for grounded,
   powerup-free states. The finish reward then uses the full run clock.
2. **Alternative:** prefix replay from canonical. The env already supports
   `prefix_actions`; this preserves the clock but uses recorded NEAT actions
   during reset. The DQN plan currently says “no NEAT action replay”; if that
   constraint is about labels, prefix replay is still compatible in spirit, but
   the doc must say so.
3. **Last resort:** wrap the env and replace the terminal reward using
   `run_time_at_state + partial_run_time`. This duplicates reward logic and is
   easy to drift.

Do not run 100k with cell-only spawns and `time` reward.

### F2 — Extraction has no incremental output and no Python log

The only artifact is written at the end. A 46-route replay is a few minutes of
work; losing it to a Ctrl-C or a terminal close is avoidable.

**Recommendation:** write `spawns/route_<signature>.json` after each route,
merge into the final pool at the end, and tee stdout/stderr to
`runs/<run>/extract.log`. Add a `--resume` that skips completed routes.

### F3 — Spawn acceptance is never checked

The game validates `cell` (standable, not exit) and falls back to canonical on a
malformed/invalid entry. It returns `spawn_cell` in reset info and the wrapper
records `last_reset_spawns`. The driver doesn't inspect either.

It also doesn't filter `powerup_count > 0`, which the state-snapshot path
explicitly does not support, or dissolve cells, which are not restored.

**Recommendation:** for every pool entry, reset once from it and require the
returned `spawn_cell` to match. Filter `powerup_count == 0`; reject cells whose
symbols are dissolve/bounce/timed objects (use the game's exported analysis, not
a reimplementation).

### F4 — The holdout is checkpoint-level, not route-level

The plan says “hold out complete route clusters,” but the dedupe key is
`(cell, facing)`. A quick offline pass over the actual route file (path cells,
`time_to_exit >= 0.5`, standable symbols) gave:

```text
460 routes
top 10%                 46 routes
Jaccard clusters        34 (27 train / 7 holdout)
pre-dedupe holdout      45 checkpoints
  also on train routes  31
holdout route cells     97
  also on train routes  78
final pool estimate     43 train + 23 holdout
```

The final pool keys are disjoint, so the checkpoint holdout is valid. But most
holdout route cells are available in training under another checkpoint or
facing, so this is **not** an unseen-route holdout. The gate can measure
checkpoint generalization; it cannot claim route generalization.

**Recommendation:** if route holdout is the question, remove train checkpoints
that lie on holdout route cells or split connected route segments. If checkpoint
holdout is enough, say so explicitly in the doc.

### F5 — Evaluation pays one cold game launch per spawn

`run_eval_episode()` creates a new `GodotEnv` for each holdout spawn. Cold launch
on macOS can take ~20 s; the current 23–45 holdout entries imply 8–15+ minutes
just in startup, plus a new `userdata/eval_<port>` tree each time.

**Recommendation:** one persistent env, reset with `spawns` per start, as
`35_eval_ppo_strata.py` does. Keep the same reset/start-jump contract across
training and eval.

### F6 — Training VecEnv is sequential

`DummyVecEnv` steps 4 envs one at a time. The PPO pilot uses `SubprocVecEnv` for
`n_envs > 1`. For a smoke, 1 env is fine; for 100k, use `SubprocVecEnv` or the
multi-agent VecEnv.

### F7 — The driver depends on the PPO pilot

`41` does:

```python
PPO = load_ppo_module()   # importlib on 10_ppo_pilot.py
PPO.SingleAgentGodotEnv
PPO.resolve_env_path()
PPO.isolated_home()
```

`10_ppo_pilot.py` is ~83 KB and had 484 uncommitted lines before `d7e173d`.
This is a fragile dependency for a new algorithm.

**Recommendation:** extract `SingleAgentGodotEnv`, `isolated_home`, and
`resolve_env_path` into `src/wrappers/` or `src/env_factory.py`; import them
normally from both PPO and DQN. Then `41` can shrink to the DQN-specific parts.

### F8 — The default analysis path is machine-specific

`DEFAULT_ANALYSIS` is an absolute path to
`/Users/giani.statie/personal/dragon-jump-remaster/runs/level_analysis/1-1.json`.
The same pattern appears in `32`, `33`, `35`, and the archive scripts. This
will fail for any other checkout.

**Recommendation:** resolve the game repo from an env var or `../dragon-jump-remaster`,
and raise a clear error with the export command when missing.

### F9 — Doc/code drift in the plan

- Doc: “infer facing from the next path cell.” Code: records actual
  `facing_direction` from the game.
- Doc: “validate the checkpoint in the game.” Code: no `spawn_ok` check.
- Doc: “reset … using the start-jump convention.” For mid-route states, the
  forced jump changes the state; per-start control is needed.

These are not fatal, but the doc should describe the chosen design before the
next run.

### F10 — Scope conflict with the game vault

Game `docs/project/decisions.md` 2026-10-27: no DDQN before the convention.
Game `docs/project/active-backlog.md` A32/A33: DQN/DDQN deferred; the frozen
58/256 model is the ship baseline. This repo's doc: “Status: active experiment.”

**Recommendation:** record the side-experiment status in the game vault (A35 or
a decision note) and keep “no install” explicit. Do not present DQN as a
convention deliverable.

### F11 — The game-side state-snapshot docs are stale

Game `docs/technical/training.md` says only position/facing are restored. The
committed controller supports `state` for grounded/no-powerup states. The
feature is used by the archived `30_trajectory_holdout.py`. The game docs should
be corrected before DQN relies on it; otherwise the next reader will think the
fix is unavailable.

### F12 — Baseline number drift

The DQN doc lists “Deployed 58 student 2.333 s”; the game status sometimes says
2.35 s. Both are close; use 2.333 s from the deployed-model doc and re-run one
clean bench before quoting a ship number. Do not compare a partial-run time to a
full-run time.

## Recommended order

1. Decide scope (side experiment vs game-vault task).
2. Add state snapshots + powerup/dissolve/spawn validation to the extractor.
3. Add incremental writes and a run log; run `extract` to completion.
4. Inspect the pool (splits, counts, overlaps) before training.
5. Smoke: 1 env, 10k steps, check episode returns and finish rate.
6. Scale: 4 envs / `SubprocVecEnv`, 100k steps.
7. Evaluate with one persistent env; report canonical + holdout separately.
8. Only then consider a game install; the frozen 58/256 model stays the ship
   baseline until the full 16-level gate passes.

## What this audit does not claim

- It does not measure the DQN policy; no model was trained.
- It does not run the route replay; the holdout overlap numbers are an offline
  estimate from `clean_routes.json` + the exported level analysis, not a game
  run.
- It does not change any game code or install any model.
