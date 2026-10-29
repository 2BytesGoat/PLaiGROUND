# 0030 — 1-1/1-2 anchor replay collection

Status: complete

Completed: 2026-10-29

## Purpose

Collect a fixed old-policy replay anchor from the selected 2.15M model for
protected DDQN training.

```text
levels:        1-1 + 1-2
starts:        canonical + all 12 + 31 clean starts
model:         ddqn_2150000_steps.zip
epsilon:       0.02
transitions:   200,000
buffer:        RareReplayBuffer
```

Each transition is bucketed by its **current state**:

```text
level | current cell | facing | run-time bin | powerup stack
```

Each transition records its return-to-go and the old model’s Q(s,a). The
collector computes:

```text
relative_gap = max(0, return_to_go - Q_old) / (abs(return_to_go) + 1)
```

and keeps positive gaps above the 0.90 quantile as profit-weighted anchor
states. Rare state buckets still receive inverse-frequency weight, so the
anchor covers both valuable and uncommon states.

## Output

```text
runs/ddqn_anchor_1_1_1_2/replay_buffer_200k.pkl
runs/ddqn_anchor_1_1_1_2/replay_buffer_200k.json
```

The earlier start-bucketed buffer is superseded by this state-bucketed version.

## Result

```text
transitions:      200,000
state buckets:    3,675
positive gaps:    141,249
gap quantile:     0.90
gap threshold:    0.12649
priority max:     0.48469
```

The protected training smoke test loaded the anchor and trained two thousand
steps successfully, so experiment 0031 is ready for the full run.
