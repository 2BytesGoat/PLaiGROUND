# 0087 — 1.555M all-level 4-frame anchor

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

Can the 1.555M chain model provide a 4-frame anchor that covers 1-1..1-4,
including the 1-3 and 1-4 states missing from the older 1.53M anchor?

## Setup

```text
model:         runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip
levels:        2 x 1-1, 2 x 1-2, 2 x 1-3, 2 x 1-4
observation:   lean_stick 191 -> 957
action repeat: 4
policy:        epsilon 0.02
transitions:   200,000
buffer:        RareReplayBuffer (state buckets + return/Q gaps)
```

Command:

```bash
.venv/bin/python tools/collect_dqn_buffer.py \
  --config experiments/0087_ddqn_anchor_1555_all_levels_4frame \
  --model runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip \
  --transitions 200000 --epsilon 0.02 --rare-buffer --gap-quantile 0.9 \
  --output runs/ddqn_anchor_1555_all_levels_4frame/rare
```

## Result

```text
episodes:     5,105
canonical:    3,571
median:       3.00 s
best:         1.5667 s
< 1.75 s:     1,458
timeouts:     6
buffer:       runs/ddqn_anchor_1555_all_levels_4frame/rare/replay_buffer_200k.pkl
```

This is the anchor used by the 0088 retention fine-tune. It covers all four old
levels at the correct control rate and has good canonical coverage; it is not
the degraded 2-frame anchor from 0083.
