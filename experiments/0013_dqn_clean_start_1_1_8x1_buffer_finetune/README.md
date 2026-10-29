# 0013 — DQN 1M → 1.85M with saved replay buffer

Status: complete  
Started: 2026-10-29

## Question

Can the saved 200k replay buffer — containing 41 canonical 1.633 s episodes —
turn the fast route into the deterministic policy with a short continuation?

## Setup

```text
source:            runs/dqn_clean_start_1_1_8x1_1m/dqn/dqn_final.zip (1M)
replay buffer:     runs/dqn_clean_start_1_1_8x1_1m/dqn/replay_buffer_200k.pkl
buffer contents:   200k transitions; 41 canonical 1.633 s episodes
intended budget:   +100,000 (1.1M total)
actual run:        +853,000 (stopped at 1.853M)
selected:          1.85M checkpoint
epsilon:           continued from the 1M schedule, ending at 0.02
gradient_steps:    4
timesteps reset:   false
topology:          8×1
inputs:            lean 188 → 954
```

The buffer was loaded into the model before `learn`, so the first updates could
replay the fast episodes instead of waiting for a fresh FIFO refill.

The planned 100k continuation was too short. SB3 treats `total_timesteps` as
**additional** steps when `reset_num_timesteps=False`; the 1.1M request therefore
ran toward 2.1M. It was stopped at 1.853M and the best checkpoint was selected.

## Output

```text
runs/dqn_clean_start_1_1_8x1_buffer_finetune/
```

Evaluate with:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0013_dqn_clean_start_1_1_8x1_buffer_finetune \
  --model runs/dqn_clean_start_1_1_8x1_buffer_finetune/dqn/dqn_final.zip
```

## Result

Selected model: **1.85M** checkpoint, saved as `dqn/dqn_final.zip`.

```text
canonical:    1/1, 1.6167 s
train pool:   12/12, median 3.2500 s, best 1.6167 s
holdout pool: 8/8, median 3.7917 s, best 3.3167 s
```

This is a **0.2500 s canonical improvement** over the 1M model's 1.8667 s and
matches the fastest route observed in the 1M training monitor.

Checkpoint trajectory:

| Checkpoint | Canonical | Train median | Holdout median |
|---:|---:|---:|---:|
| 1.025M | 1.8667 | 3.2917 | 4.5333 |
| 1.050M | 1.8667 | 3.2833 | 3.8583 |
| 1.075M | 1.8667 | 3.2833 | 4.5333 |
| 1.100M | 1.8667 | 3.2750 | 4.5333 |
| 1.800M | **1.6167** | 3.2583 | **3.7917** |
| 1.825M | **1.6167** | **3.2500** | 3.8000 |
| 1.850M | **1.6167** | **3.2500** | **3.7917** |

The fast route did not become the deterministic argmax during the first 100k.
It appeared by 1.8M and stayed stable through 1.85M. The original 200k buffer was
fully replaced well before then; the late gain came from continued optimization,
not only from replaying the original fast episodes.

Evaluation artifacts:

```text
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/dqn_1025000_steps/summary.json
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/dqn_1050000_steps/summary.json
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/dqn_1075000_steps/summary.json
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/dqn_1100000_steps/summary.json
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/dqn_1800000_steps/summary.json
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/dqn_1825000_steps/summary.json
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/dqn_1850000_steps/summary.json
```
