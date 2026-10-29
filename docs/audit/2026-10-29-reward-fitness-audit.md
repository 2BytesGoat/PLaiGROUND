# Reward / NEAT fitness audit — 2026-10-29

## Question

Why did the 17.6 s, backtracking 1-4 genome beat the 4.9833 s clean genome in
NEAT fitness?

```text
best_time genome:     4.9833 s
best_fitness genome: 17.6000 s
```

## Method

Replayed both genomes on canonical 1-4 with:

```text
observation:   legacy_stick (61)
action:        reassert_held_jump
action repeat: 4
speedup:       12
horizon:       300 decisions
```

The game reward was decomposed from `info` counters using the same formula as
`PlayerAITrainingController.get_reward()`:

```text
progress   = C_prev - 0.99 * C_curr
efficiency = -0.02 * delta(wasted_air_frames)
time       = -5.0 * delta(run_time) * (1 - gold_ratio)
finish     = +500 + under-gold bonus
death      = -100
```

The per-step components were summed and compared with the environment reward.

## Result

```text
                        4.9833 s          17.6000 s
steps                  73                263
path cells             73                263
unique cells           59                134
revisit ratio          0.192             0.490
backtracks             2                 28
clean                  yes               no
environment reward     600.64            628.72

progress               66.04             137.82
efficiency             0.00              -7.28
time                   +28.38            -1.82
finish                 500.00            500.00
death/timeout           0.00              0.00
component total        594.42            628.72
delta vs env           6.22              ~0.00
```

The small 6.22 discrepancy on the fast route is from first-step/finish timing;
the slow-route decomposition matches the environment reward to numerical
precision.

## Finding

The 17.6 s genome did **not** win because of the finish bonus alone. It earned:

```text
progress advantage:    +71.78
time disadvantage:     -30.20
efficiency disadvantage: -7.28
net advantage:         +34.30
```

NEAT fitness is an **undiscounted sum** of per-step rewards. The progress term
uses `C_prev - 0.99 * C_curr`; with a finite episode, that does not telescope
exactly. For a longer path around cells with positive flow cost, each step
contributes a small positive `0.01 * C` bias. The slow route also has no
explicit revisit/backtracking penalty.

`clean` is computed **after** the episode, not in the reward:

```text
revisit_ratio <= 0.10
backtracks <= 2
action_flip_rate <= 0.75
```

The reward therefore permits a slow, loopy finish to outscore a fast clean one.

## DDQN implication

The same game reward is used by DDQN, but DDQN's Q target is discounted:

```text
r_t + 0.99 * max Q(s')
```

For a discounted return, the `C_prev - 0.99 * C_curr` shaping term largely
telescopes, so DDQN is much less exposed to the exact path-length bias. It is
still exposed to:

- no revisit/backtrack penalty;
- weak time pressure relative to the fixed finish bonus;
- a fixed `+500` finish term;
- a 10 s training horizon, which permits slow but non-timeout loops.

The 1-4 DDQN failure (`0/1` canonical, `1/9` train, `1/3` holdout) is more
consistent with observation/sample-efficiency limits (lean 188, no ceiling
feature, sparse pool) than with the NEAT-specific fitness bias alone.

## Decision

Do not silently change the shipped reward contract. Keep the existing clean/
time evaluation gate for deployment and the 1-5 seed. If a reward v2 is tested,
version it and audit it separately:

1. Make progress shaping telescope (`gamma = 1` or a terminal correction).
2. Add a revisit/backtrack penalty.
3. Make time dominate the fixed finish bonus.
4. Keep `clean` route filtering as an independent gate.
