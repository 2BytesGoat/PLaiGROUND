# Protected replay for multi-level DDQN — 2026-10-29

## Problem

The 0025 1-3 DDQN run solved 1-3 but regressed 1-1/1-2 at some checkpoints:

```text
2.30M selected
1-1: 1.6167 canonical; 12/12 train 3.2750; 8/8 holdout 3.8000
1-2: 3.8667 canonical; 31/31 train 4.0000; 20/20 holdout 3.1667
1-3: 3.1000 canonical; 45/45 train 5.2000; 23/23 holdout 5.4500

2.15M prior model
1-2 canonical: 2.9667
```

The 1-2 coverage did not collapse; its canonical route slowed down. The current
start pools already include the longest clean routes, so adding more start
states does not address the drift. The issue is replay turnover and
catastrophic forgetting.

## Design

Use two replay partitions:

```text
online
  current mixed-level transitions
  ring buffer, older transitions may be evicted

anchor
  fixed transitions from the selected prior model
  canonical + longest/rare 1-1 and 1-2 starts
  never evicted during a run
```

Sample every minibatch as:

```text
75% online
25% anchor
```

The anchor is not just uniform old data, and it is not start-bucketed. It
records the **current state** at each transition:

```text
level
current cell
facing
run-time bin
powerup stack
```

The collector stores both the episode’s **return-to-go** and the old model’s
**Q_old(s, a)** for each transition. It computes the positive gap:

```text
gap = return_to_go - Q_old(s, a)
relative_gap = max(0, gap) / (abs(return_to_go) + 1)
```

and keeps only positive gaps above a quantile threshold (default 0.90). Anchor
sampling combines:

```text
inverse state-bucket frequency × capped positive relative gap
```

with a cap so a single rare or extremely profitable state cannot dominate. This
keeps high-value mid-route states, not only canonical/longest episode starts.
States below the gap threshold still receive rarity weight but no extra profit
weight.

## Implementation

```text
src/plaiground/algorithms/reinforcement/replay.py
  RareReplayBuffer
  ProtectedReplayBuffer

tools/collect_dqn_buffer.py
  --rare-buffer
  collects a rare-bucket anchor buffer

src/plaiground/algorithms/reinforcement/dqn.py
  loads an anchor buffer from [run].anchor_buffer_from
  installs ProtectedReplayBuffer before model.learn()

config.toml

[run]
anchor_buffer_from = "runs/.../anchor_buffer.pkl"

[algorithm]
anchor_fraction = 0.25
```

`ProtectedReplayBuffer` keeps the online and anchor buffers at the same
`n_envs`; samples both with the same shape and concatenates them.

The anchor is fixed during training, so `RareReplayBuffer` caches its sampling
probability vector after the first minibatch. The cache is invalidated whenever
stored data or priorities change. Without the cache, recomputing 25k anchor
weights on every gradient step cut throughput from about 720 fps to 160 fps.

## Collection

Collect the anchor from the selected 2.15M model, not from the new 1-3 policy:

```text
levels:       1-1 + 1-2
starts:       canonical + all existing 1-1/1-2 train starts
              (the longest clean routes are already present)
epsilon:      0.02
transitions:  ~200k
```

This gives fixed old-policy trajectories for both previous levels. The collector
records current-state buckets, return-to-go, Q_old, and positive gap priorities;
the earlier start-bucketed and state-only 200k buffers are superseded and should
not be used for training.

## Training

```text
source:          2.15M selected model
levels:          4 × 1-3 + 2 × 1-2 + 2 × 1-1
online buffer:   400k
anchor buffer:   200k
sample mix:      75% online / 25% anchor
budget:          300k
checkpoint every: 25k
```

Evaluate every checkpoint on:

```text
1-1 canonical + train + holdout
1-2 canonical + train + holdout
1-3 canonical + train + holdout
```

Select the Pareto checkpoint, not automatically the final one.

## Success criteria

```text
1-3 train/holdout remain fully covered
1-1 train/holdout remain fully covered
1-2 canonical recovers toward 2.9667 s
1-2 train/holdout medians stay near the 2.15M reference
```

## Risks

- The anchor can overfit old behavior if the fraction is too high. Start at 25%.
- Rare buckets can overrepresent a single start. Cap inverse-frequency weights.
- Anchor transitions from the old model may be stale as the new policy changes;
  this is intentional for retention, but monitor 1-3 learning.
- Do not use an anchor from the 1-3-trained model when evaluating retention.

## Result

Experiment 0031 completed 300k additional steps from 2.15M in about seven
minutes. All checkpoints were evaluated on 1-3; the full 1-1/1-2 gates were run
on the candidates. The selected checkpoint is **2.225M**:

```text
runs/ddqn_protected_1_1_1_2_1_3/dqn/checkpoints/ddqn_2225000_steps.zip
```

| Level | Canonical | Train | Holdout |
|---|---:|---:|---:|
| 1-1 | 1.6167 s | 12/12, 3.2667 s | 8/8, 3.7917 s |
| 1-2 | **2.9667 s** | 31/31, 4.3500 s | 20/20, 3.1333 s |
| 1-3 | 3.2500 s | 45/45, 5.2667 s | 23/23, 5.3333 s |

Old 0025 2.30M control:

```text
1-1: 1.6167 / 12/12 3.2750 / 8/8 3.8000
1-2: 3.8667 / 31/31 4.0000 / 20/20 3.1667
1-3: 3.1000 / 45/45 5.2000 / 23/23 5.4500
```

The anchor solved the retention problem: 1-2 canonical returned to the 2.15M
baseline exactly, with full coverage, and 1-1 remained stable. It did **not**
improve 1-3 speed: the selected model is 0.15 s slower canonically and has a
slightly slower train median, with a faster holdout median.

The 25% anchor allocates 37.5% of samples to 1-3 instead of the 50% online share.
That is the likely source of the small 1-3 slowdown. If 1-3 speed is the next
objective, use a lower anchor fraction (10–15%), or build a 1-3 high-gap anchor
and keep the old-level anchor smaller. Do not increase the old-level anchor.

Other checkpoints and gates:

```text
2.300M / 2.325M / 2.350M: 1-1 holdout 4.050 s > 3.89 s
2.375M:                   1-2 train coverage 30/31
2.400M / 2.425M:           1-3 canonical 5.633-5.767 s
2.450M:                   1-2 canonical 3.800 s, 1-3 canonical 3.183 s
```

The final 2.45M checkpoint was not selected. The old 2.30M control was
re-evaluated under the current game state and reproduced its recorded metrics.
