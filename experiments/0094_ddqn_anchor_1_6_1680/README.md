# 0094 — 1-6-only anchor from the 1.68M checkpoint

Status: complete  
Started: 2026-10-29

## Question

Can a 1-6-only RareReplayBuffer from the 1.68M 1-6 model stabilize the
1-6-only retrain without the old-level anchor?

## Setup

```text
model:          runs/ddqn_expert_replay_1_6_lean_stick/dqn/checkpoints/ddqn_1680000_steps.zip
levels:         8 x 1-6
action repeat:  4
transitions:    200,000
epsilon:        0.02
buffer:         RareReplayBuffer (gap-quantile 0.9)
```

Command:

```bash
.venv/bin/python tools/collect_dqn_buffer.py \
  --config experiments/0094_ddqn_anchor_1_6_1680 \
  --model runs/ddqn_expert_replay_1_6_lean_stick/dqn/checkpoints/ddqn_1680000_steps.zip \
  --transitions 200000 --epsilon 0.02 --rare-buffer --gap-quantile 0.9 \
  --output runs/ddqn_anchor_1_6_1680/rare
```

## Output

```text
runs/ddqn_anchor_1_6_1680/rare/replay_buffer_200k.pkl
runs/ddqn_anchor_1_6_1680/rare/replay_buffer_200k.json
```

## Result

```text
transitions:    200,000
episodes:       3,224
canonical:      3,224
finished:       1,697
deaths:         1,412
timeouts:       115
median finish:  4.6833 s
best finish:    3.0667 s (stochastic; 3.2/3.3333 modes also present)
finishes <4.28: 155
best death:     2.4333 s  (finished=False, length 35, reward -48.49)
state buckets:  4,325
positive gaps:  52,268 (threshold 0.10455)
```

The 2.4333 s row is a death, not a fast finish. The **best deterministic**
1-6 model is 4.5667 s at 1.705M; the anchor's 3.0667/3.2/3.3333 s rows are real
stochastic finishes from the 2% exploration policy, not deterministic
checkpoints. The 4.07 s expert replay is the best replayed clean route.
