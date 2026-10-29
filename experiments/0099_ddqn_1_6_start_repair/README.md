# 0099 — 1-6 start repair

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

The 0097 sweep produced an any-start model at 1.7675M (32/32, 4.65 s) and a
faster model at 1.805M (31/32, 4.55 s). Only `[21,16]` facing `-1` failed on
the faster model. Can a targeted expert-suffix fine-tune repair that start
without sacrificing canonical speed or the other 31 starts?

## Data

```text
base:           runs/ddqn_1_6_feasible_starts/.../ddqn_1804968_steps.zip
start:          [21,16], facing -1, run_time 6.6667
source route:   62569ffe... (source 10.6333 s; stable expert replay)
expert suffix:  transitions from step 100 to the route finish, repeated 4x
buffer:         232 transitions
starts:         canonical + 31 state starts (0.7/0.3 pool)
budget:         10,000 steps; checkpoints every 2.5k
lr:             5e-5 (manifest override now applied on resume)
```

The suffix buffer is built by
`tools/build_start_expert_buffer.py --suffix`; the start pool is
`runs/ddqn_1_6_start_repair/repair_start_pool.json`.

## Result

```text
checkpoint   canonical   32-state coverage   deaths  timeouts
1.8075M      4.9000      31/32              1       0
1.8100M      4.8167      28/32              1       3
1.8125M      4.5500      30/32              2       0
1.8150M      4.6000      32/32              0       0
```

The selected model is **1.815M**:

```text
runs/ddqn_1_6_start_repair/dqn/checkpoints/ddqn_1814952_steps.zip
canonical:   1/1, 4.6000 s, 0 deaths, 0 timeouts
32 states:   32/32 finishes, 0 deaths, 0 timeouts
median:      6.0417 s
spawn_ok:    32/32
repeat:      canonical 4.6000, 32/32 (same)
```

The repair fixed the remaining start but moved canonical from 4.55 to 4.60 s.
That is the selected **any-start 1-6 teacher**. It is not a 4.28 s speed model;
the stable route remains 4.50 s and the 4.28 gate is still open.

## Run

```bash
.venv/bin/python tools/build_start_expert_buffer.py \
  --config experiments/0097_ddqn_1_6_feasible_starts \
  --expert-data runs/ddqn_1_6_retrain/expert_1_6/expert_routes.npz \
  --start-pool runs/ddqn_1_6_start_repair/repair_start_pool.json \
  --output runs/ddqn_1_6_start_repair/expert_1_6/repair_suffix_buffer.pkl \
  --repeat 4 --suffix

.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0099_ddqn_1_6_start_repair
```
