# 0083 — 2-frame old-level anchor

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

What does the selected 1.555M DDQN do on 1-1..1-4 when the control rate is
changed from `action_repeat=4` to `action_repeat=2`?

## Setup

```text
model:        runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip
levels:       2 x 1-1, 2 x 1-2, 2 x 1-3, 2 x 1-4
observation:  lean_stick 191 -> 957
action repeat: 2 (30 Hz)
policy:       epsilon 0.02
transitions:  200,000
buffer:       RareReplayBuffer (state buckets + return/Q gaps)
```

Command:

```bash
.venv/bin/python tools/collect_dqn_buffer.py \
  --config experiments/0083_ddqn_anchor_1555_2frame \
  --model runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip \
  --transitions 200000 --epsilon 0.02 --rare-buffer --gap-quantile 0.9 \
  --output runs/ddqn_anchor_1555_2frame/rare
```

## Result

```text
transitions:  200,000
episodes:     2,229
canonical:    1,568
median:       3.25 s
best:         1.6167 s
timeouts:     409
buffer:       runs/ddqn_anchor_1555_2frame/rare/replay_buffer_200k.pkl (1.54 GB)
```

The 1.555M model was trained at 4 frames; at 2 frames its old-level behavior is
noticeably degraded. This anchor is still the correct 2-frame baseline for the
0082 run, but it is not evidence of 4-frame retention. Evaluate the old levels
at 2 frames after 25k steps; if the 0.3 s gate fails, fine-tune 1-1..1-4 at 2
frames or use a 1-5-specific head.
