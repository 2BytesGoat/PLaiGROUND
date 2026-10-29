# 0091 — 1-6 DDQN from the 4-frame NEAT clean routes

Status: complete  
Started: 2026-10-29

## Question

Can the selected 1-5 DDQN learn 1-6 from the 4.55 s NEAT routes and reach the
user's 3.98 s pace without losing the levels needed by the chain?

## Expert data

0090 seeded the 4-frame 1-5 population on 1-6. Generation 51 produced 31 unique
routes / 25 clean; the fastest clean route is **4.55 s**. Filtering at Jaccard
0.5, seed 0:

```text
clean routes:   24
clusters:       9
train:          22 (4.55 s fastest)
holdout:        2 (6.62 s, 11.72 s)
replay:         20/22 finished
transitions:    1,696
expert buffer:  1,559 transitions
```

The two replay misses are 4.65 s and 4.72 s routes; the fast 4.55 s pair and the
4.58–4.75 s cluster replayed exactly.

## Setup

```text
source model:   runs/ddqn_expert_replay_1_5_4frame/dqn/dqn_final.zip (1.655M)
levels:         4 x 1-6 + 1 x 1-5 + 2 x 1-4 + 1 x 1-3
action repeat:  4
expert:         1-6 22-route train set (1,559 transitions)
old anchor:     0087 1.555M all-level 4-frame RareReplayBuffer
budget:         100,000 steps, checkpoints every 25k
```

The 1-6 model is intended to be the 1-6 member of the per-level chain. The 1-5
and old-level models remain available if the shared backbone drifts.

## Gates

```text
1-6:       finish, <= 4.28 s (user 3.98 + 0.30)
1-5:       finish, <= 3.41 s if retained by this model
1-4/1-3:   finish, within 0.30 s of the old chain
```

## Run

```bash
.venv/bin/python tools/collect_expert_routes.py \
  --config experiments/0091_ddqn_expert_replay_1_6_lean_stick
.venv/bin/python tools/build_expert_buffer.py \
  --config experiments/0091_ddqn_expert_replay_1_6_lean_stick
.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0091_ddqn_expert_replay_1_6_lean_stick
```

## Output

```text
runs/ddqn_expert_replay_1_6_lean_stick/
  expert_1_6/expert_routes.npz
  expert_buffer_1_6.pkl
  dqn/dqn_final.zip
  dqn/checkpoints/
  eval/
```

## Result

Deterministic eval at 4-frame control. A full checkpoint sweep (the first pass
missed 1.705M) gives:

```text
              1.68M      1.705M     1.73M     1.755M     final
1-6           4.6833 s   4.5667 s   8.6000 s  8.3167 s   7.2500 s
```

The best 1-6 result is **4.5667 s** at **1.705M**, still **0.29 s above** the
4.28 s acceptance gate. The 1.68M checkpoint is 4.6833 s; the final degraded to
7.25 s. The 1.705M model also lost 1-5 (death) and 1-4 (timeout), so it is a
1-6-only model and not a chain member. On the new 18-entry 1-6 state pool it
finishes 13/18 (5 deaths, median 5.8667 s), confirming that 4.5667 is a
canonical-only result:
`runs/ddqn_expert_replay_1_6_lean_stick/eval/state_pool_1705/summary.json`.
The 4.55 s expert was the bottleneck: the DDQN could not improve on it to the
gate. 0093 extended the 1-6 NEAT speed search; the 1.705M checkpoint is the
reference 1-6 model for any retrain.
