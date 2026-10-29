# 0095 — 1-6-only DDQN retrain from the 1.68M checkpoint

Status: complete  
Started: 2026-10-29

## Question

Does the 1-6 policy improve when the old-level environments and old anchor are
removed, using 8×1-6 plus a 1-6-only anchor?

## Expert

0093's filtered archive (seed 0):

```text
filtered:      133 clean routes / 33 clusters
train/holdout: 104 / 29
fastest:       4.5000 s (train)
replay:        89/104 finished
transitions:   8,645
buffer:        7,586 transitions / 89 finishes
replay best:   4.0667 s (inside the 4.28 gate)
```

The replay produced a 4.0667 s finish from a route whose source was 6.2667 s;
that faster route is in the expert buffer.

## Setup

```text
start:         1.68M 1-6 checkpoint
levels:        8 x 1-6
action repeat: 4
anchor:        1-6-only 200k RareReplayBuffer from the 1.68M model
expert:        1-6 7,586-transition buffer
budget:        50,000 steps
lr:            5e-5
exploration:   0.10 -> 0.05
```

The 1-6 model is the 1-6 member of the per-level chain; 1-5 and 1-4..1-1 stay
with their existing checkpoints.

## Gates

```text
1-6:  finish, <= 4.28 s (user 3.98 + 0.30)
goal: <= 3.98 s
```

## Run

```bash
.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0095_ddqn_1_6_retrain_1680
```

## Output

```text
runs/ddqn_1_6_retrain/
  expert_1_6/expert_routes.npz
  expert_buffer_1_6.pkl
  dqn/dqn_final.zip
  dqn/checkpoints/
  eval/
```

## Result

Deterministic 1-6-only eval:

```text
1.6925M: timeout
1.7050M: 8.5000 s
1.7175M: death + timeout
1.7300M final: 6.3833 s
```

The retrain did **not** make the 4.07 s route deterministic. The training monitor
also showed 3.08-3.20 s finishes, but those were stochastic exploration episodes;
the final deterministic result is 6.3833 s. The 1-6 baseline is the 0091 1.705M
checkpoint at 4.5667 s (the 1.68M anchor source is 4.6833 s). See
[`../../docs/audit/2026-10-29-1-6-ddqn-investigation.md`](../../docs/audit/2026-10-29-1-6-ddqn-investigation.md).
