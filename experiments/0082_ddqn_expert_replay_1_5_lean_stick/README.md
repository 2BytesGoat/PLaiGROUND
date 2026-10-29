# 0082 — one-step DDQN with 1-5 expert replay

Status: stopped  
Started: 2026-10-29

## Question

Can the selected 1.555M lean_stick 191 model learn 1-5 from the 2-frame
**4.40 s** NEAT routes without forgetting 1-1..1-4?

## Why this setup

The 4-frame 1-5 pool was not good enough: its train split was 6.45–7.58 s and
the only 4.68 s route was held out. Experiment 0081 produced a much better
2-frame pool:

```text
clean routes:       114 (best 4.4000 s)
Jaccard 0.5:        6 clusters
seed 0 split:       32 train / 4 holdout routes
expert replay:      28/32 finished, 4,405 transitions
expert buffer:      4,191 transitions, 28 finishes
```

The 4.40 s route reproduces exactly at `action_repeat=2`. The train split also
contains the 4.43–5.55 s routes, so the expert is not dominated by slow routes.

## Setup

```text
source model:       1.555M 1-4 chain checkpoint
observation:        lean_stick 191 -> 957
levels:             4 x 1-5 + 1 x 1-4 + 2 x 1-3 + 1 x 1-2
action repeat:      2 (30 Hz)
expert routes:      32 train clean routes
expert buffer:      4,191 transitions, sampled 64 per gradient step
old anchor:         200k 2-frame transitions from the 1.555M model on 1-1..1-4
budget:             100,000 steps, checkpoints every 25k
```

The expert TD and margin terms are the same as 0073:

```text
main TD on online + old-anchor replay
+ 1.0 x TD on the 1-5 expert buffer
+ 0.2 x margin on the 1-5 expert batch
+ 0.1 x margin on the old-anchor batch
```

## Gates

```text
1-5 canonical:   finish, ideally <= 3.41 s (user 3.11 + 0.30)
1-5 expert:      the 4.40 s route should be the first learned behavior
1-1..1-4:        within 0.30 s of the 1.555M 4-frame times
```

## 2-frame anchor caveat

The 1.555M model was trained at `action_repeat=4`; at 2 frames its old-level
behavior is degraded. The 200k anchor stats are:

```text
episodes:     2229
canonical:    1568
median:       3.25 s
best:         1.6167 s
timeouts:     409
```

That is expected for a control-rate change, and it means the old-level gate
must be checked at 2 frames. If old-level retention fails, the next step is a
short 1-1..1-4 fine-tune at 2 frames or a 1-5-specific head — not a rollback to
the 4-frame expert.

## Expert refresh

This round uses a staged refresh, not a live mid-gradient swap:

1. train 25k steps;
2. evaluate 1-5 canonical and the old levels;
3. if a better clean route exists, rebuild the expert buffer and resume from
   the checkpoint with the new `expert_buffer_from`;
4. keep the online replay and the model weights across the refresh.

That keeps the expert target stationary inside each 25k block. A file-watch
callback at the checkpoint boundary is the natural automation later.

## Run

```bash
.venv/bin/python tools/collect_expert_routes.py \
  --config experiments/0082_ddqn_expert_replay_1_5_lean_stick
.venv/bin/python tools/build_expert_buffer.py \
  --config experiments/0082_ddqn_expert_replay_1_5_lean_stick
.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0082_ddqn_expert_replay_1_5_lean_stick
```

## Output

```text
runs/ddqn_expert_replay_1_5_lean_stick/
  expert_1_5_2frame/expert_routes.npz
  expert_buffer_1_5_2frame.pkl
  dqn/dqn_final.zip
  dqn/checkpoints/
  eval/
```

## Result (2-frame, stopped at 1.58M)

Deterministic canonical eval of the 1.58M checkpoint at `action_repeat=2`:

```text
1-5: 0/1 (death)
1-4: timeout
1-3: 3.2833 s
1-2: 3.0333 s
1-1: 2.9333 s
```

The 3.20 s training episode was not robust. Changing the control rate to 2
frames degraded 1-1 and 1-4 and did not produce a stable 1-5 policy, so 0082 was
stopped. Experiment 0085 keeps the proven 4-frame anchor and uses the 4.68 s
1-5 expert cluster.
