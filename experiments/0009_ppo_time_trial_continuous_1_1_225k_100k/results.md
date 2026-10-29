# Results — 0009

Status: complete; reward ranking fixed, route not learned  
Completed: 2026-10-28

## Outcome

The continuous under-gold bonus made the fast route score higher than the
stable route, but the 100k continuation did not turn that one sample into the
deterministic policy default. The 600k stable model remains selected.

## Deterministic evaluation

| Model | Canonical | Train pool | Holdout pool |
|---|---:|---:|---:|
| 600k stable selected | 1/1, 2.1500 s | 12/12, 3.3250 s | 8/8, 3.8583 s |
| 325k flat-bonus (0008) | 1/1, 2.1500 s | 12/12, 3.2917 s | 8/8, 3.8583 s |
| 325k continuous (0009) | 1/1, 2.1500 s | 12/12, 3.5583 s | 8/8, 4.9500 s |

All 0009 pool runs finished and all spawns were applied.

The continuous run is worse on both pools and still does not lower canonical
time. It is not selected.

## Training monitor

Canonical-only training, 100k continuation:

```text
episodes:          3,114
median:            2.150 s
best:              1.633 s
≤2.2 s:            71.7%
timeouts:          0.06%
episodes <1.9 s:   1
episodes <1.75 s:  1
```

The reward ordering is now correct:

```text
1.633 s: reward 556.8
2.133 s: reward 546.4
```

But the fast route appeared only **once in 3,114 episodes**. The source policy
is not exploring it enough for PPO to reinforce it, and the 8×1 canonical-only
setup still converges to the stable 2.13/2.15 route.

## Decision

Keep `runs/ppo_clean_start_1_1_best_100k/ppo_pilot_600k.zip` selected.

The continuous under-gold term is the right reward shape, but it needs more
exploration or a better starting distribution to work. The next experiment
should increase exploration/topology — for example 8×8 with a proportionally
larger batch — rather than repeating another 100k of 8×1 canonical-only.
