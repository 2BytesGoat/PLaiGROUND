# 0039 — Protected DDQN 1-1/1-2/1-3/1-4

Status: complete; 1-4 coverage negative, no 1-4 checkpoint selected

Started: 2026-10-29

## Question

Can the selected 2.225M 1-1/1-2/1-3 DDQN absorb 1-4 while keeping the old
levels fully covered?

This is the first 1-4 training stage. It uses the clean 1-4 start pool produced
by 0038; it does not use the rough 0037 archive as training states.

## Inputs

```text
checkpoint:    runs/ddqn_protected_1_1_1_2_1_3/dqn/checkpoints/ddqn_2225000_steps.zip
anchor:        runs/ddqn_anchor_1_1_1_2/replay_buffer_200k.pkl
preprocessing: runs/ppo_clean_start_1_1_fresh/preprocessing.pkl
train starts:  runs/neat_seeded_1_4_stick/pool/multilevel_train_start_pool.json
1-4 holdout:   runs/neat_seeded_1_4_stick/pool/holdout_start_pool.json
```

The 1-4 pool is:

```text
5 clean train routes / 8 train states / canonical
5 clean holdout routes / 3 holdout states
```

The fastest 1-4 route (5.7667 s) is held out; it is not in the training starts.

## Training

```text
levels:          4 × 1-4 + 2 × 1-3 + 1 × 1-2 + 1 × 1-1
online buffer:   400k transitions
anchor buffer:   200k 1-1/1-2 state/Q-gap transitions
sampling:        75% online / 25% anchor
action semantics: reassert_held_jump = true
budget:          300k additional steps
```

The level mix follows the project rule: current level dominant, previous level
represented, and the oldest levels rotated. The protected replay is the
retention mechanism; the 1-4 start pool is the discovery/coverage mechanism.

## Intermediate check

A 2.35M checkpoint (125k steps into this stage) was evaluated on the 1-4 pools:

```text
canonical:   0/1, timeout
train pool:  1/9, best 6.7833 s, 9/9 spawns accepted
holdout:     1/3, best 6.7833 s, 3/3 spawns accepted
```

This is early-stage behavior, not a selection result. It confirms the pool and
held-input path run end-to-end; the final checkpoint and 1-4/old-level matrix
still decide the stage.

## Final result

The run completed at **2.525M** steps. The deterministic final checkpoint was
identical to the 2.35M intermediate on 1-4:

```text
1-4 canonical:   0/1, timeout
1-4 train:       1/9, best 6.7833 s, 9/9 spawns accepted
1-4 holdout:     1/3, best 6.7833 s, 3/3 spawns accepted
```

The extra 175k steps did not improve deterministic 1-4 behavior. The training
monitor had **39/1,139** 1-4 finishes (best 6.65 s), so the level mix and held
input path work, but the policy is far below the coverage bar.

Old-level retention at the final checkpoint:

```text
1-1: 1/1 canonical 1.6167 s; 12/12 train 3.2667 s; 8/8 holdout 4.1583 s
1-2: 1/1 canonical 3.0333 s; 30/31 train 4.8333 s; 19/20 holdout 3.4333 s
1-3: 1/1 canonical 3.2333 s; 45/45 train 5.3667 s; 23/23 holdout 5.4333 s
```

Retention is mostly intact, with two 1-2 start misses. There is no 1-4
checkpoint worth promoting from this run. **No 1-4 checkpoint selected.**

The next step is not another blind DDQN continuation. The 1-4 NEAT population
must first be made finisher-rich and diverse with finisher-preserving
rebalancing (0044), then a stronger pool can be built and, if needed, a
`lean_stick` DDQN branch can be tested.

## Selection

Evaluate at least every 25k checkpoint on:

```text
1-4 canonical + train + holdout
1-3 canonical + train + holdout
1-2 canonical + train + holdout
1-1 canonical + train + holdout
```

Select the Pareto checkpoint, not the final checkpoint by default. Gate:

```text
1-4 train/holdout coverage improves
1-1 remains full (12/12, 8/8)
1-2 remains full (31/31, 20/20) and canonical stays near 2.9667 s
1-3 remains full (45/45, 23/23)
```

## Run

```bash
caffeinate -dimsu .venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0039_ddqn_protected_1_1_1_2_1_3_1_4
```
