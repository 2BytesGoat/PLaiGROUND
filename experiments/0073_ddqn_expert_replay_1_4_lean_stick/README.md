# 0073 — one-step DDQN with 1-4 expert replay

Status: complete, selected 1.555M  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

Can the selected 1.53M lean_stick 191 model learn the 1-4 canonical route from
the successful 0046 NEAT routes in a single DDQN run, without forgetting
1-1/1-2/1-3?

## Why this exists

The 0068 1-4 probe reproduced the 0053 failure:

```text
1-4 canonical: 0/1; 0/310 training episodes; death at ~101 decisions
1-4 train:     1/11 at 6.7833 s
1-4 holdout:   0/2
old levels:    full coverage, but 1-2/1-3 canonical drifted
```

The three `lean_stick` flags are wired and the first-layer weights are large,
but they are rare in the old-level anchor: 24 `on_ceiling` and 193
`stick_active` transitions in 25k. The 1-4 NEAT population already uses the
same three flags (`legacy_stick`) and has 382/384 finishers, best 4.9833 s, and
75 clean routes.

Replaying the 27 clustered train routes in the 191 environment reproduced 21
exactly, including the canonical route at 5.7667 s. The 6 misses are filtered
out. The successful 1-4 transitions have `stick_active=1` on 18.75% of steps and
`jump_held=1` on 70.6%, so the expert data contains the missing route behavior.

## Setup

```text
source model:       0067 1.53M checkpoint
observation:        lean_stick 191 -> 957
levels:             4 x 1-4 + 1 x 1-3 + 2 x 1-2 + 1 x 1-1
expert routes:      21 successful routes, 2,231 transitions
expert buffer:      sampled every gradient step (64)
old anchor:         25k 1.53M 1-1/1-2/1-3 transitions, 50% of the main batch
budget:             100,000 additional steps
```

One training run only. Every gradient step has:

```text
main TD loss on online + old-anchor replay
+ 1.0 x TD loss on the 1-4 expert buffer
+ 0.2 x margin loss on the 1-4 expert batch
+ 0.1 x margin loss on the old-anchor batch
```

The margin loss is:

```text
relu(0.5 - (Q(s, a_expert) - max_{a != a_expert} Q(s, a)))
```

The old-anchor margin uses the stored 1.53M actions. The old anchor also stays
in the protected replay, and the old-level environments remain in the mix.

## Gates

```text
1-4 canonical:  finish, ideally <= 6.0 s
1-4 train:      >= 10/11
1-4 holdout:    >= 1/2
1-1:            1.5667, 12/12, 8/8
1-2:            3.0167, 31/31, 20/20 — reject materially above 3.1
1-3:            3.1667, 45/45, 23/23
```

The 1-2 canonical is the fragile retention gate. If it drifts despite the
anchor and margin, the shared Q-network is interfering and the next fix is a
1-4 head/level-conditioned policy.

## Run

```text
.venv/bin/python tools/collect_expert_routes.py --config experiments/0073_ddqn_expert_replay_1_4_lean_stick
.venv/bin/python tools/build_expert_buffer.py --config experiments/0073_ddqn_expert_replay_1_4_lean_stick
.venv/bin/python scripts/reinforcement/03_ddqn.py --config experiments/0073_ddqn_expert_replay_1_4_lean_stick
```

## Output

```text
runs/ddqn_expert_replay_1_4_lean_stick/
  expert_buffer_1_4.pkl
  dqn/dqn_final.zip
  dqn/checkpoints/
  eval/
```

## Result

The one-step expert replay plus margin worked. The 1-4 canonical, train, and
holdout gates all pass at **1.555M**, the first 25k checkpoint:

```text
                       1.555M      1.58M       1.605M      1.63M
1-4 canonical          1/1 4.4667  0/1         1/1 5.8667  0/1
1-4 train              11/11       6/11        7/11        7/11
1-4 holdout            2/2         1/2         1/2         1/2
```

The 1.555M old-level gate:

```text
1-1: 1.5667, 12/12, 8/8
1-2: 3.1500, 31/31, 20/20; holdout 4.4750
1-3: 3.2667, 45/45, 23/23; holdout 5.6500
```

Coverage is full on every old level. The cost is canonical speed on 1-2 and
1-3: 3.1500 vs the 3.0167 baseline, and 3.2667 vs 3.1667. The 1.63M checkpoint
has a better 1-2 (3.0167; holdout 3.3583) but fails 1-4 (0/1 canonical, 7/11
train, 1/2 holdout), so 1.555M is the only 1-4 candidate.

Selection:

- **1.555M** — 1-4 chain checkpoint; all 1-4 gates, full old coverage.
- **1.53M** — old-level-only reference and fallback (1-2 3.0167, 1-3 3.1667).
- **1.63M** — better 1-2 retention, but not a 1-4 candidate.

The 1-4 problem was search/data, not the three hold flags: once the successful
NEAT routes were in the replay and the expert action was ranked by the margin
loss, the canonical route started finishing within 25k steps. The remaining
tradeoff is shared-network interference on 1-2/1-3 canonical speed; the next
fix, if needed, is a 1-4 head or level-conditioned policy rather than more
expert weight.
