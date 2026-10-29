# 0109 — canonical repair for the 1-8 expert

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

The 0106 sweep produced an any-start 1-8 checkpoint at **1.939912M** with
canonical **4.5167 s**, 39/40 clean states, and 106/107 all-success states. The
canonical time misses the provisional 4.5144 s gate by only **0.0023 s**. Can a
short canonical-heavy continuation close that gap without losing state coverage?

## Data

```text
base:            runs/ddqn_1_8_feasible_starts/.../ddqn_1939912_steps.zip
training pool:   runs/neat_transfer_1_8_from_1_7/pool_clean_g46_canonical_heavy/train_start_pool.json
start weights:   canonical 0.70 / state 0.30
expert data:     runs/ddqn_1_8_feasible_starts/expert_1_8/expert_routes.npz
expert buffer:   runs/ddqn_1_8_feasible_starts/expert_1_8/stable_expert_buffer.pkl
budget:          25,000 steps; checkpoints every 5k
```

The pool still contains all 37 strict-clean train states; only the sampling
weight shifts toward canonical. The 40-state evaluation gate is unchanged.

## Setup

```text
levels:         8 x 1-8
action repeat:  4
lr:             5e-5
exploration:    0.05 -> 0.02
expert TD:      0.5
behavior margin: 1.0 x 0.5
old anchor:     none
```

## Gates

```text
canonical:      finish, provisional <= 4.5144 s (target <= 4.51)
coverage:       40/40 clean states
safety:         no deaths; report timeouts separately
```

## Run

```bash
.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0109_ddqn_1_8_canonical_repair
```

## Result

All five checkpoints cover all 40 strict-clean states:

```text
checkpoint      canonical   clean states   deaths   timeouts   best
1.944912M       9.5000      40/40          0        0          4.5167
1.949912M       4.5167      40/40          0        0          4.5167
1.954912M       4.5167      40/40          0        0          4.5167
1.959912M       4.5167      40/40          0        0          4.5167
1.964912M       4.5833      40/40          0        0          4.5167
```

The selected repair checkpoint is **1.949912M** (earliest full-coverage
checkpoint):

```text
canonical:   4.5167 s
clean pool:  40/40, 0 deaths, 0 timeouts, median 7.3917, best 4.5167
spawn_ok:    40/40
```

This fixes the 0106 `[32,18]/-1` timeout. Canonical remains **4.5167 s**,
still 0.0023 s above the provisional gate. The final 0112 suffix fine-tune
tests whether a targeted fast-route repair can remove the last frame.

## Artifacts

```text
runs/ddqn_1_8_canonical_repair/dqn/checkpoints/
runs/ddqn_1_8_canonical_repair/eval/
```
