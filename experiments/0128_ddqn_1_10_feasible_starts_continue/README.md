# 0128 — 1-10 coverage continuation

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

Does the 22/25-coverage 1-10 checkpoint improve with 200k more steps at a lower
learning rate, or is 5.3667 s canonical / 22/25 coverage the plateau?

## Source

```text
base checkpoint:  runs/ddqn_1_10_feasible_starts/dqn/checkpoints/ddqn_2122840_steps.zip
train pool:       runs/neat_transfer_1_10_from_1_9/pool_clean_g131/train_start_pool.json
expert buffer:    runs/ddqn_1_10_feasible_starts/expert_1_10/stable_expert_buffer.pkl
learning rate:    2e-5
expert TD weight: 0.75
exploration:      0.05 -> 0.01 over 20%
budget:           200,000 steps, checkpoints every 12,500
wall time:        239 s
```

## Result

16 checkpoints from 2.135336M to 2.322776M:

| Range | Canonical | Eval pool |
|---|---:|---:|
| 2.135336M-2.210312M | 5.3667 | 21/25 |
| 2.222808M-2.232776M best speed | 5.3000 | 21/25 |
| 2.247800M-2.272792M | 5.3000 | 20/25 |
| 2.285288M-2.322776M | 5.3000 | 19/25 |

The continuation gains **0.0667 s** canonical but never exceeds the base run's
22/25 coverage, and coverage decays late in training. No checkpoint is selected.
The 22/25 **2.122840M** checkpoint from 0126 remains the coverage best.

## Output

```text
runs/ddqn_1_10_feasible_starts_continue/
  dqn/checkpoints/ddqn_2222808_steps.zip  # fastest 5.3000, 21/25
  dqn/dqn_final.zip
  train.console.log
```
