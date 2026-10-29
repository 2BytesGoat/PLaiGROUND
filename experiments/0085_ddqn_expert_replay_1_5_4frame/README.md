# 0085 — 4-frame DDQN with the fastest 1-5 expert cluster

Status: complete  
Started: 2026-10-29

## Question

Can the 1.555M chain model learn 1-5 from the clean 4.68 s NEAT routes while
keeping the proven 4-frame 1-1..1-4 behavior?

## Why the 4-frame retry

The 2-frame run (0082) failed its deterministic 1.58M eval:

```text
1-5: 0/1 (death)
1-4: timeout
1-3: 3.2833 s
1-2: 3.0333 s
1-1: 2.9333 s
```

The 1.555M model was trained at 4 frames; changing the control rate broke the
old-level policy and did not give a robust 1-5 route. This run keeps
`action_repeat=4`, the 1.53M 4-frame anchor, and the 1.555M resume.

## Expert

The clean 4-frame archive has 16 routes. All 16 replay exactly in the 191
environment, including the fastest cluster:

```text
4.6833 s  (two routes)
4.7166 s  (two routes)
5.1666 s
...
7.5833 s
```

The expert data is 1,510 transitions, 16 finishes. The 4.68 s cluster is
included in the expert buffer, so the run is not limited to the 6.45–7.58 s
train split that caused the earlier concern.

## Setup

```text
source model:  1.555M 1-4 chain checkpoint
observation:   lean_stick 191 -> 957
levels:        4 x 1-5 + 1 x 1-4 + 2 x 1-3 + 1 x 1-2
action repeat: 4
expert buffer: 1,510 transitions, sampled 64 per gradient step
old anchor:    200k 4-frame 1.53M transitions
budget:        100,000 steps, checkpoints every 25k
```

## Gates

```text
1-5 canonical:  finish, ideally <= 3.41 s
1-5 expert:     4.68 s cluster learned first
1-1..1-4:       within 0.30 s of the 1.555M 4-frame times
```

## Run

```bash
.venv/bin/python tools/collect_expert_routes.py \
  --config experiments/0085_ddqn_expert_replay_1_5_4frame
.venv/bin/python tools/build_expert_buffer.py \
  --config experiments/0085_ddqn_expert_replay_1_5_4frame
.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0085_ddqn_expert_replay_1_5_4frame
```

## Expert refresh

The refresh is staged at the 25k checkpoints. If a better clean route appears,
rebuild the expert buffer and resume from the checkpoint with the new
`expert_buffer_from`; keep the online replay and model weights. This avoids
changing the expert target inside a 25k block.

## Output

```text
runs/ddqn_expert_replay_1_5_4frame/
  expert_1_5/expert_routes.npz
  expert_buffer_1_5.pkl
  dqn/dqn_final.zip
  dqn/checkpoints/
  eval/
```

## Result

Final deterministic eval (1.655M, 4-frame control), repeated twice:

```text
1-5: 3.1667 s  (finish, efficiency 1.0)
1-3: 3.1667 s
1-1: 1.5667 s
1-4: fail (one death, one timeout)
1-2: 5.3500 s
```

The 1-5 result is stable and is **+0.057 s** versus the user's 3.11 s, well
inside the 3.41 s acceptance band. It beat the 4.68 s expert by 1.51 s. The
same model retained 1-1 and 1-3 but lost 1-4 and late 1-2.

The old 1.555M chain checkpoint still holds the old levels (0079: 1.5667 /
3.1500 / 3.2667 / 4.4667). The practical chain is therefore two models:

```text
1-5:       runs/ddqn_expert_replay_1_5_4frame/dqn/dqn_final.zip
1-1..1-4:  runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip
```

The 0088 single-model retention fine-tune did not work: its final gave 1-5
3.5000 s and still timed out on 1-4. The two-model chain above remains the
selected deliverable.
