# 0018 — Double DQN multi-level 1-1/1-2, 300k

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

Does replacing the vanilla DQN target with a Double-DQN target reduce the
jittery 1-2 canonical policy while preserving 1-1 behavior?

## Setup

```text
source:          runs/dqn_multilevel_1_1_1_2_100k/dqn/dqn_final.zip (2.05M)
levels:          1-1, 1-2
processes:       4 × 1-1 + 4 × 1-2
budget:          300,000 additional steps (2.35M total)
algorithm:       Double DQN
n_steps/grads:   3 / 4
buffer:          200,000
epsilon:         0.02
learning rate:   0.0001
```

The only algorithmic change from 0017 is the TD target:

```text
online net selects next action
target net evaluates that action
```

Everything else is held constant so the DDQN effect is measurable.

## Baseline at 2.05M

```text
1-1 canonical: 1.6167 s
1-1 train:     12/12, median 3.2667 s
1-1 holdout:   8/8, median 3.8000 s

1-2 canonical: 3.7833 s
1-2 train:     31/31, median 4.0167 s, best 3.1000 s
1-2 holdout:   20/20, median 3.3500 s, best 3.0833 s

1-2 canonical action trace:
  55 decisions, 17 flips, mean Q-margin 19.1
```

## Output

```text
runs/dqn_multilevel_1_1_1_2_ddqn_300k/
```

## Result

The final 2.35M checkpoint did **not** pass the 1-1 gate: canonical stayed
1.6167 s and train was 3.2667 s, but one 1-1 holdout start regressed enough to
move the holdout median to 4.1833 s (limit 3.8900 s). Its 1-2 canonical was
3.0333 s.

The **2.15M checkpoint** is the selected DDQN model:

```text
checkpoint:  dqn/checkpoints/ddqn_2150000_steps.zip
1-1 canonical: 1.6167 s
1-1 train:      12/12, median 3.2750 s
1-1 holdout:    8/8, median 3.7917 s   PASS
1-2 canonical: 2.9667 s
1-2 train:      31/31, median 3.9167 s, best 2.9667 s
1-2 holdout:    20/20, median 3.1417 s, best 3.0500 s
```

No deterministic timeouts or deaths.

This ties the best PPO 1-2 canonical reference (2.9667 s), beats the old NEAT
transfer (3.0167 s), and preserves the selected 1-1 behavior. The DDQN
canonical action trace also improved from 17 flips / 55 decisions to
10 flips / 44 decisions.

## Evaluation artifacts

```text
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/ddqn_2150k_1_1/summary.json
runs/dqn_zero_shot_1_2/eval/ddqn_2150k_1_2/summary.json
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/ddqn_300k_1_1/summary.json
runs/dqn_zero_shot_1_2/eval/ddqn_300k_1_2/summary.json
```
