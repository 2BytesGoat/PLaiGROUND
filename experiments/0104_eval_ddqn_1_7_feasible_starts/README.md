# 0104 — evaluate the 1-7 feasible-start expert

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Purpose

Evaluate 0103 checkpoints on 1-7 canonical plus the accepted 10-state
near-clean start pool; also report the 20-state all-success sensitivity set.

## Setup

```text
level:          1-7
encoding:       lean_stick 191 -> 957
action repeat:  4
canonical:      one deterministic episode
accepted set:   runs/neat_clean_1_7_from_0101_attempt2/pool_near_clean_g30/near_clean_start_pool.json
accepted count: 10
diagnostic set: runs/neat_clean_1_7_from_0101_attempt2/pool_all_success_g30/all_success_start_pool.json
diagnostic count: 20
holdout:        near-clean holdout has 5 route-cluster states, all shared with train keys
```

## Gate

```text
canonical finish; provisional <= 5.2728 s
10/10 near-clean starts finish
18/20 all-success starts is diagnostic; the 2 failures are on a high-revisit route
report deaths and timeouts separately
report spawn_ok for every start
```

The provisional gate is par 4.9728 s + 0.30 s. The user has not fixed a final
1-7 target yet.

## Run

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0104_eval_ddqn_1_7_feasible_starts \
  --model runs/ddqn_1_7_feasible_starts/dqn/checkpoints/<checkpoint>.zip \
  --tag <checkpoint>
```

## Result

Selected 1.877432M, repeated twice with identical output:

```text
canonical:         1/1, 5.2333 s, 0 deaths/timeouts
near-clean:        10/10, 0 deaths/timeouts, median 7.9583, best 5.7667
all-success:       18/20, 0 deaths, 2 timeouts, median 9.9000
spawn_ok:          10/10 near-clean, 20/20 all-success
```

The all-success set is diagnostic and includes two late facing-right starts
from a high-revisit route (`8f687...`, revisit 0.43). The near-clean set is the
accepted 1-7 coverage gate. The canonical time passes the provisional
**5.2728 s** gate by **0.0395 s**.
