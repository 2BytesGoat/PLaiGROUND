# 0126 — 1-10 coverage DDQN expert

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

Can the selected 1-9 expert be fine-tuned into an any-start 1-10 expert on the
24 grounded train states from the 0125 NEAT archive, and how close can it get to
the 4.70 s clean route?

## Source

```text
NEAT run:       0125, generation 131
accepted tier:  strict clean, 311 routes
filtered:       109 routes / 22 clusters
train/holdout:  81 / 28 routes
state starts:   24 train / 6 holdout
eval union:     25 states
best train:     4.7000 s (1778c800...)
best holdout:   4.7833 s
base model:     runs/ddqn_1_9_single_route/dqn/checkpoints/ddqn_2060360_steps.zip
par / gate:     3.0336 s / 3.3336 s provisional
```

## Data

```text
train pool:    runs/neat_transfer_1_10_from_1_9/pool_clean_g131/train_start_pool.json
holdout pool:  runs/neat_transfer_1_10_from_1_9/pool_clean_g131/holdout_start_pool.json
eval pool:     runs/neat_transfer_1_10_from_1_9/pool_clean_g131/eval_clean_start_pool.json
expert routes: runs/neat_transfer_1_10_from_1_9/filtered_clean_g131/train_clean_routes.json
expert data:   runs/ddqn_1_10_feasible_starts/expert_1_10/expert_routes.npz
expert buffer: runs/ddqn_1_10_feasible_starts/expert_1_10/stable_expert_buffer.pkl
```

Route replay: **79/81** train routes finished, **7,178** transitions. The
stable buffer kept **6,834 transitions / 78 routes** after the 0.25 s replay
gain filter.

## Training

```text
algorithm:          DDQN
timesteps:          100,000
n_envs:             8
learning rate:      5e-5
expert TD weight:   0.5
behavior margin:    1.0 x 0.5
exploration:        0.10 -> 0.02 over 50%
checkpoints:        every 12,500 steps
wall time:          108 s
```

## Result

All eight checkpoints finish canonical 1-10, but coverage and speed trade off:

| Checkpoint | Canonical | Eval pool | Deaths | Timeouts |
|---|---:|---:|---:|---:|
| 2.072856M | 5.6000 | 12/25 | 4 | 9 |
| 2.085352M | 5.3000 | 17/25 | 8 | 0 |
| 2.097848M | 5.3667 | 20/25 | 5 | 0 |
| 2.110344M | 5.3667 | 20/25 | 4 | 1 |
| **2.122840M** | **5.3667** | **22/25** | **3** | **0** |
| 2.135336M | 5.3667 | 21/25 | 4 | 0 |
| 2.147832M | 5.3667 | 21/25 | 4 | 0 |
| 2.160328M | 5.3667 | 20/25 | 4 | 1 |

Selected checkpoint:

```text
runs/ddqn_1_10_feasible_starts/dqn/checkpoints/ddqn_2122840_steps.zip
```

Repeat-stable selection:

```text
canonical:          1/1, 5.3667 s, 0 deaths/timeouts
eval pool:          22/25, 3 deaths, 0 timeouts, spawn 25/25
eval best/median:   4.8333 / 7.8333 s
failures:           [8,25] f1 (death, 11 steps)
                    [9,25] f1 (death, 11 steps)
                    [26,20] f1 (death, 31 steps)
```

The 5.3667 s canonical is 0.67 s slower than the 4.70 s NEAT route. The three
missing states are early/mid-route train starts that also fail the later
continuation and repair runs. The provisional 3.3336 s gate is open; the game's
bundled AI also finishes at 4.48 s, so 1-10 is a known slow level rather than a
single-checkpoint defect.

The follow-up runs are 0128 (200k low-LR continuation; canonical 5.3000 but
coverage fell to 19-21/25) and 0129 (targeted suffix repair; canonical 4.7000
but coverage 17-20/25). Neither is selected.

## Output

```text
runs/ddqn_1_10_feasible_starts/
  dqn/checkpoints/ddqn_2122840_steps.zip
  dqn/dqn_final.zip
  expert_1_10/
  eval/selected_repeat/summary.json
```
