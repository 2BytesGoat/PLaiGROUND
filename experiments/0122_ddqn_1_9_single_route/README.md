# 0122 — 1-9 single-route DDQN expert

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

Can the selected 1-8 expert learn the single fastest clean 1-9 route with a
canonical-only reset distribution and a one-route expert buffer?

## Source

```text
NEAT run:      0121 (stopped at generation 89)
route:         926883585ecb0f7469bba7968907f4014e23b235
route time:    2.9500 s
route cells:   38 compressed cells
actions:       43 decisions at action repeat 4
par / gate:    2.7648 s / 3.0648 s provisional
base model:    runs/ddqn_1_8_canonical_nudge/dqn/checkpoints/ddqn_1960392_steps.zip
```

The user confirmed 1-9 is a single-route precision map, so this run does not
train on the other clean starts. The broad 11-state pool remains a diagnostic
artifact in `runs/neat_transfer_1_9_from_1_8/pool_clean_g89/`.

## Data

```text
train pool:    runs/neat_transfer_1_9_from_1_8/single_route/canonical_only_start_pool.json
expert routes: runs/neat_transfer_1_9_from_1_8/single_route/train_clean_routes.json
expert data:   runs/ddqn_1_9_single_route/expert_1_9/expert_routes.npz
expert buffer: runs/ddqn_1_9_single_route/expert_1_9/stable_expert_buffer.pkl
```

## Training

```text
algorithm:          DDQN
timesteps:          100,000
n_envs:             8
learning rate:      5e-5
expert TD weight:   1.0
behavior margin:    1.0 x 0.5
exploration:        0.10 -> 0.02 over 25%
checkpoints:        every 12,500 steps
```

## Result

Training completed 100k steps in about **295 s** (2.060360M total). The route
replayed exactly during collection: **43/43 decisions, 2.9500 s**.

Training monitor:

```text
episodes:          2,446
last 200:          170/200 finishes (85%)
last 1000:         808/1000 finishes (81%)
best time:         2.9000 s
median finish:     2.9500 s
```

Every saved checkpoint was evaluated deterministically on canonical 1-9:

```text
1.972888M  2.9500 s   finish
1.985384M  2.9500 s   finish
1.997880M  2.9500 s   finish
2.010376M  2.9500 s   finish
2.022872M  2.9500 s   finish
2.035368M  2.9500 s   finish
2.047864M  2.9500 s   finish
2.060360M  2.9500 s   finish
```

Selected checkpoint:

```text
runs/ddqn_1_9_single_route/dqn/checkpoints/ddqn_2060360_steps.zip
```

The final model is equivalent and is the one used for the retention check.

## Retention check

Deterministic canonical 1-1..1-9 with the selected 1-9 checkpoint:

| Level | Result | Time | Gate | Verdict |
|---|---|---:|---:|---|
| 1-1 | finish | 2.5167 s | 2.5944 s | pass |
| 1-2 | timeout | — | 3.2376 s | fail |
| 1-3 | finish | 6.0167 s | 3.3816 s | finish, gate fail |
| 1-4 | timeout | — | 4.4856 s | fail |
| 1-5 | death | — | 3.2088 s | fail |
| 1-6 | death | — | 4.28 s (explicit) | fail |
| 1-7 | timeout | — | 5.2728 s | fail |
| 1-8 | timeout | — | 4.5144 s | fail |
| 1-9 | finish | 2.9500 s | 3.0648 s | pass |

This is a 1-9-only fine-tune. It retains the ability to finish 1-1 and 1-3,
but only 1-1 and 1-9 are under gate; it is not a multi-level retention model.

## Output

```text
runs/ddqn_1_9_single_route/
  dqn/checkpoints/
  dqn/dqn_final.zip
  expert_1_9/
  eval/
```
