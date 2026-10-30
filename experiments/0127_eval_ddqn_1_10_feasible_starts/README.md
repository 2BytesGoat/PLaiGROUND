# 0127 — evaluate the 1-10 coverage expert

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

What are the deterministic canonical time and eval-pool coverage of each 0126
1-10 checkpoint?

## Setup

```text
config:      experiments/0127_eval_ddqn_1_10_feasible_starts/config.toml
canonical:   true
eval set:    runs/neat_transfer_1_10_from_1_9/pool_clean_g131/eval_clean_start_pool.json (25 states)
n_envs:      1
action repeat: 4
speedup:     12
```

The sweep covers all eight 0126 checkpoints. `selected_repeat` re-runs the
selected **2.122840M** checkpoint exactly.

## Result

Selected repeat:

```text
checkpoint:  runs/ddqn_1_10_feasible_starts/dqn/checkpoints/ddqn_2122840_steps.zip
canonical:   1/1, 5.3667 s, 0 deaths/timeouts
eval pool:   22/25, 3 deaths, 0 timeouts, spawn 25/25
eval best:   4.8333 s; median 7.8333 s
```

The other checkpoints are listed in the 0126 README. Best canonical among the
100k checkpoints is 5.3000 s (2.085352M), but coverage there is only 17/25.

## Output

```text
runs/ddqn_1_10_feasible_starts/eval/<tag>/summary.json
runs/ddqn_1_10_feasible_starts/eval/selected_repeat/summary.json
```
