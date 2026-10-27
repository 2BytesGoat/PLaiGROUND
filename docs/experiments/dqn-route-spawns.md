# DQN from NEAT route spawns

Status: **active experiment, no result**
Started: 2026-10-28
Last updated: 2026-10-28
First level: 1-1
Driver: [`../../src/41_dqn_route_spawns.py`](../../src/41_dqn_route_spawns.py)
Detailed review: [`../audit/2026-10-28-dqn-pipeline-review.md`](../audit/2026-10-28-dqn-pipeline-review.md)

## Goal

Train a fresh 58-value DQN that starts from spawn checkpoints discovered by
NEAT routes.

NEAT is used only as a route/spawn generator. It is not used as an action
teacher, and NEAT actions are not imitation labels.

## Scope

- One shared, level-agnostic 58-value policy.
- Start with **1-1 only**.
- No old PPO checkpoints, start pools, or trajectory datasets.
- No 754/794 sensor observations.
- 1-2 and 1-3 are deferred until 1-1 works.

## Inputs and network

```text
58 observations
  -> 256 ReLU
  -> 256 ReLU
  -> 2 Q-values (idle, jump)
```

Observation contract: legacy 58 values. Action repeat: 4 physics frames.

## Route and spawn pipeline

1. Copy the 1-1 NEAT clean routes into the new run directory.
2. Filter routes to the fastest 10%, clustered by path similarity.
3. For each selected route:
   - take path cells,
   - keep standable cells,
   - infer facing from the next path cell (current code records the game's
     actual facing instead),
   - validate the checkpoint in the game (current code does **not** check
     spawn acceptance),
   - reject cells that are not valid grounded spawns.
4. Deduplicate `(cell, facing)` checkpoints.
5. Hold out complete route clusters for evaluation (see caveat below).
6. Build a weighted spawn pool: canonical + route checkpoints.
7. Reset DQN episodes from the pool using the start-jump convention.

## DQN training

- Double DQN (SB3 `DQN`, `double_q=True` by default).
- Fresh initialization.
- Replay buffer, target network.
- Reward profile: `time`.
- The agent collects its own transitions from route spawns.
- No NEAT action replay during DQN training.

## Evaluation

Canonical 1-1 and held-out route spawns.

Baselines:

| Model | 1-1 |
|---|---:|
| NEAT teacher | 2.150 s |
| Deployed 58 student | 2.333 s (some game docs say 2.35) |
| 754 PPO teacher | 1.633 s |

The first gate is:

- canonical finishes,
- canonical time competitive with the NEAT teacher,
- held-out spawn coverage does not collapse.

## Current status (2026-10-28)

- Script and this plan are committed (`d7e173d`).
- `runs/dqn_route_spawns_1_1/` has the 460 clean routes and two Godot logs.
- `spawns/` is empty; there is no `dqn/` or `eval/`.
- The first extract stopped right after the game booted (`TRAINING_SPEEDUP`)
  and before the pool was written. No training or evaluation has run.
- Treat every number above as a **baseline**, not a DQN result.

## Before the next run

Do not run 100k with cell-only spawns and the `time` reward. The reset zeroes
`run_time`, so a mid-route spawn that finishes in 0.5 s gets ~220 terminal
reward while a full 2.15 s canonical run gets ~105. The agent is rewarded for
partial-run shortcuts. Use grounded `state` snapshots (run time, velocity,
`run_active`, `started_walking`) for mid-route checkpoints, or prefix replay
from canonical. Keep the forced start jump for canonical starts only.

Also fix:

1. Incremental extract output + a Python run log.
2. `powerup_count == 0` and dissolve-cell filtering.
3. Spawn acceptance check (`last_reset_spawns` / `spawn_cell`).
4. Persistent-env evaluation instead of one Godot launch per spawn.
5. `SubprocVecEnv` for multi-env training (or smoke with 1 env).
6. The checkpoint-vs-route holdout distinction.

See [`../known-issues.md`](../known-issues.md) and the
[audit](../audit/2026-10-28-dqn-pipeline-review.md) for details.

## Outputs

```text
runs/dqn_route_spawns_1_1/
  routes/        copied clean routes/finishers
  spawns/        spawn pool(s)
  dqn/           checkpoints, monitor.csv, tensorboard
  eval/          summary.json / results.json
  godot/         raw Godot logs
  userdata/      isolated HOME/XDG per process
```
