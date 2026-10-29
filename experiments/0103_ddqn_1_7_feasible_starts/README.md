# 0103 — 1-7 expert from NEAT feasible starts

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

Can a 1-7-only DDQN expert learn the level from the canonical start plus every
grounded NEAT-discovered feasible start, initialized from the selected 1-6
teacher and without old-level environments or an old-level anchor?

## Data

```text
source level:       1-7
source route run:   0102
training pool:      runs/neat_clean_1_7_from_0101_attempt2/pool_all_success_g30/
accepted gate pool: runs/neat_clean_1_7_from_0101_attempt2/pool_near_clean_g30/
expert routes:      runs/neat_clean_1_7_from_0101_attempt2/filtered_all_success_g30/train_expert_routes.json
expert data:        runs/ddqn_1_7_feasible_starts/expert_1_7/expert_routes.npz
stable buffer:      runs/ddqn_1_7_feasible_starts/expert_1_7/stable_expert_buffer.pkl
start weights:      canonical 0.50 / state 0.50
```

The start/holdout split is route-cluster based. Training used the 20-state
all-success pool (canonical plus 20). Strict clean routes were not available
for 1-7, so the accepted coverage gate uses **near-clean** routes
(`revisit_ratio <= 0.20`, `backtracks <= 2`, action flips <= 0.75), a 10-state
subset. The all-success pool is also the diagnostic sensitivity set; it includes
two late facing-right states from a high-revisit route (`8f687...`, revisit
0.43).

## Setup

```text
base model:     runs/ddqn_1_6_start_repair/.../ddqn_1814952_steps.zip
levels:         8 x 1-7
action repeat:  4
budget:         100,000 steps; checkpoints every 12.5k
lr:             5e-5 (manifest LR applied after resume)
exploration:    0.10 -> 0.02 over 50% declared; resumed SB3 schedule reports
                current 0.02 because the timestep counter continues
expert:         stable successful NEAT routes
expert TD:      0.5
behavior margin: 1.0 x 0.5
old anchor:     none
```

## Gates

```text
canonical:      finish, provisional <= 5.2728 s
coverage:       10/10 near-clean starts; 18/20 all-success diagnostic
holdout:        all route-cluster holdout starts must finish
safety:         no spawn fallback; report deaths and timeouts separately
```

The provisional gate is level par 4.9728 s + 0.30 s. It is not final until the
user sets a 1-7 target.

## Run

```bash
# accepted near-clean pool (10 grounded states)
.venv/bin/python tools/filter_routes.py \
  --routes runs/neat_clean_1_7_from_0101_attempt2/near_clean_routes_g30.json \
  --output-dir runs/neat_clean_1_7_from_0101_attempt2/filtered_near_clean_g30 \
  --level 1-7 --jaccard 0.5 --max-per-cluster 8 \
  --holdout-fraction 0.25 --seed 0

.venv/bin/python tools/build_start_pool.py \
  --filtered-dir runs/neat_clean_1_7_from_0101_attempt2/filtered_near_clean_g30 \
  --archive-dir runs/neat_clean_1_7_from_0101_attempt2/archive \
  --analysis ../dragon-jump-remaster/runs/level_analysis/1-7.json \
  --output-dir runs/neat_clean_1_7_from_0101_attempt2/pool_near_clean_g30 \
  --level 1-7 --canonical-weight 0.50 --state-weight 0.50

# diagnostic all-success pool (20 grounded states)
# same commands with all_success_routes_g30.json -> filtered_all_success_g30
# -> pool_all_success_g30.

.venv/bin/python tools/collect_expert_routes.py \
  --config experiments/0103_ddqn_1_7_feasible_starts

.venv/bin/python tools/build_expert_buffer.py \
  --config experiments/0103_ddqn_1_7_feasible_starts \
  --max-replay-gain 0.25

.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0103_ddqn_1_7_feasible_starts
```

## Baseline

The selected 1.6 teacher at 1.815M is a canonical-only 1-6 model:

```text
1-7 canonical:  timeout (0/1)
20 feasible starts: 20/20 finishes, 0 deaths, 0 timeouts
median 11.4917 s, best 7.7500 s, spawn_ok 20/20
```

It generalizes to the late NEAT starts but does not know the 1-7 canonical
route. This is the baseline the 0103 expert must beat.

## Result

The 100k run produced eight checkpoints. The selected **1.877432M**:

```text
runs/ddqn_1_7_feasible_starts/dqn/checkpoints/ddqn_1877432_steps.zip
canonical:          1/1, 5.2333 s, 0 deaths/timeouts (repeat exact)
near-clean pool:   10/10, 0 deaths/timeouts, median 7.9583, best 5.7667
all-success pool:  18/20, 0 deaths, 2 timeouts (diagnostic)
spawn_ok:          10/10 near-clean, 20/20 all-success
```

The two all-success timeouts are late facing-right states from `8f687...`, a
high-revisit route (revisit 0.43) that is not in the near-clean gate. The
1.889928M and 1.914920M checkpoints have a faster canonical **5.2167 s** but
cover only 15/16 of 20 all-success starts, so 1.877432M is the selected
any-start model.

The provisional gate is par 4.9728 + 0.30 = **5.2728 s**; 1.877432M passes by
**0.0395 s**. A final 1-7 target is still open.

## Artifacts

```text
runs/ddqn_1_7_feasible_starts/dqn/checkpoints/ddqn_1877432_steps.zip
runs/ddqn_1_7_feasible_starts/eval/selected_repeat/summary.json
runs/neat_clean_1_7_from_0101_attempt2/pool_near_clean_g30/
runs/neat_clean_1_7_from_0101_attempt2/pool_all_success_g30/
```
