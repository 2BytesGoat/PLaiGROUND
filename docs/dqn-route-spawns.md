# DQN from NEAT route spawns

Status: active experiment  
Started: 2026-10-28  
First level: 1-1

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
   - infer facing from the next path cell,
   - validate the checkpoint in the game,
   - reject cells that are not valid grounded spawns.
4. Deduplicate `(cell, facing)` checkpoints.
5. Hold out complete route clusters for evaluation.
6. Build a weighted spawn pool: canonical + route checkpoints.
7. Reset DQN episodes from the pool using the start-jump convention.

## DQN training

- Double DQN.
- Fresh initialization.
- Replay buffer.
- Target network.
- Reward profile: time.
- The agent collects its own transitions from route spawns.
- No NEAT action replay during DQN training.

## Evaluation

Canonical 1-1 and held-out route spawns.

Baselines:

| Model | 1-1 |
|---|---:|
| NEAT teacher | 2.150 s |
| Deployed 58 student | 2.333 s |
| 754 PPO teacher | 1.633 s |

The first gate is:

- canonical finishes,
- canonical time competitive with the NEAT teacher,
- held-out spawn coverage does not collapse.

## Outputs

```text
runs/dqn_route_spawns_1_1/
  routes/
  spawns/
  dqn/
  eval/
```
