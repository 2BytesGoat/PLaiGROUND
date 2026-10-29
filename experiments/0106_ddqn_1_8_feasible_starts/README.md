# 0106 — 1-8 expert from NEAT feasible starts

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-30

## Question

Can a 1-8-only DDQN expert learn the level from the canonical start plus every
grounded strict-clean NEAT start, initialized from the selected 1-7 teacher and
without old-level environments or an old-level anchor?

## Data

```text
source level:       1-8
source route run:   0105
training pool:      runs/neat_transfer_1_8_from_1_7/pool_clean_g46/train_start_pool.json
holdout pool:       runs/neat_transfer_1_8_from_1_7/pool_clean_g46/holdout_start_pool.json
accepted eval pool: runs/neat_transfer_1_8_from_1_7/pool_clean_g46/eval_clean_start_pool.json
expert routes:      runs/neat_transfer_1_8_from_1_7/filtered_clean_g46/train_clean_routes.json
expert data:        runs/ddqn_1_8_feasible_starts/expert_1_8/expert_routes.npz
stable buffer:      runs/ddqn_1_8_feasible_starts/expert_1_8/stable_expert_buffer.pkl
start weights:      canonical 0.50 / state 0.50
```

The route archive has 26 strict-clean routes in 11 clusters. After Jaccard 0.5
filtering and a 25% cluster holdout, 16 train routes and 10 holdout routes
remain. They expand to 37 train states and 14 holdout states; the union for
evaluation is 40 unique states.

## Setup

```text
base model:      runs/ddqn_1_7_feasible_starts/.../ddqn_1877432_steps.zip
levels:          8 x 1-8
action repeat:   4
budget:          100,000 steps; checkpoints every 12.5k
lr:              5e-5 (manifest LR applied after resume)
exploration:     0.10 -> 0.02 over 50% declared; resumed SB3 schedule reports
                 current 0.02 because the timestep counter continues
expert:          stable successful strict-clean NEAT routes
expert TD:       0.5
behavior margin: 1.0 x 0.5
old anchor:      none
```

## Gates

```text
canonical:      finish, provisional <= 4.5144 s
coverage:       every clean train and holdout state must finish
safety:         no spawn fallback; report deaths and timeouts separately
```

The provisional gate is level par 4.2144 s + 0.30 s. It is not final until the
user sets a 1-8 target.

## Run

```bash
.venv/bin/python tools/collect_expert_routes.py \
  --config experiments/0106_ddqn_1_8_feasible_starts

.venv/bin/python tools/build_expert_buffer.py \
  --config experiments/0106_ddqn_1_8_feasible_starts \
  --max-replay-gain 0.25

.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0106_ddqn_1_8_feasible_starts
```

## Result

The 16 clean train routes replayed **16/16** successfully; the stable filter
kept **1,800 transitions**. The 100k run produced eight checkpoints:

```text
checkpoint      canonical   clean states   all-success   note
1.889928M       4.6000      25/40           -             early
1.902424M       timeout     29/40           -             not selected
1.914920M       4.6833      26/40           -             early
1.927416M      10.0667      40/40          106/107        stuck canonical
1.939912M       4.5167      39/40          106/107        initial selection
1.952408M       4.5167      39/40          102/107        worse all-success
1.964904M       4.5833      36/40           -             regression
1.977400M       9.8667      37/40           -             final, not selected
```

The selected **1.939912M**:

```text
runs/ddqn_1_8_feasible_starts/dqn/checkpoints/ddqn_1939912_steps.zip
canonical:          1/1, 4.5167 s, 0 deaths/timeouts (repeat exact)
clean pool:         39/40, 0 deaths, 1 timeout, median 7.2667, best 4.5167
all-success pool:   106/107, 0 deaths, 1 timeout, median 9.5500, best 4.4833
spawn_ok:           40/40 clean, 107/107 all-success
```

The clean timeout is `[32,18]` facing `-1` from route `9537113b...`; the
all-success timeout is `[22,18]` facing `-1` from the high-revisit route
`c00e6aad...`. The 1.927416M checkpoint covers all 40 clean states but its
canonical policy is stuck at 10.0667 s, so 1.939912M is the any-start selection.

The canonical time **4.5167 s** misses the provisional 4.5144 s gate by
**0.0023 s**. The follow-up 0109 canonical-heavy repair targets that gap.

## Artifacts

```text
runs/ddqn_1_8_feasible_starts/dqn/checkpoints/
runs/ddqn_1_8_feasible_starts/eval/
runs/neat_transfer_1_8_from_1_7/pool_clean_g46/
```
