# 0097 — 1-6 expert from all NEAT feasible starts

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

Can a 1-6-only DDQN expert learn the level from **every feasible NEAT start**,
not only the canonical start, without the old-level environments, anchor, or
gates?

This is the first run under the
[expert-portfolio decision](../../docs/audit/2026-10-29-expert-portfolio-and-student-decision.md):
NEAT defines the feasible starts; the per-level expert is trained and evaluated
on those starts; the teacher phase remains single-level.

## Feasible starts

The 0093 archive has 376 routes / 263 clean routes. Expanding all 263 clean
routes against the 1-6 level analysis gives **32 unique grounded state starts**
(cell, facing, powerup stack), plus canonical.

```text
archive clean routes:   263
unique state starts:     32
train states:            31  (route-cluster split)
holdout state key:        1  ([22,16], facing -1)
shared state keys:       15  (train/holdout overlap; see caveat)
canonical weight:       0.50
state weight:           0.50
```

The training pool is
`runs/neat_speed_1_6_from_0090/pool_state_50/train_start_pool.json`. The
evaluation set is the full 32-state union:
`runs/neat_speed_1_6_from_0090/pool_state_50/eval_all_start_pool.json`.

Caveat: the existing route-cluster holdout still shares 15 cell/facing/stack
keys with train. Only `[22,16] / -1` is a true unseen state. The full 32-state
set is used as the coverage gate; the one genuine holdout is reported
separately.

## Fast-route audit

The 0095 replay reported 3.1833 / 3.8667 / 3.9833 / 4.0667 s fast routes. They
were replayed again here before being used as teacher data:

```text
0095 replay       0097 batch replay      0097 standalone
3.1833 s          4.7167 s               —
3.8667 s          3.9833 s               5.0500 s
3.9833 s          5.0000 s               —
4.0667 s          death at 2.7167 s      —
```

The fast results depend on route order / carried RNG or world state; they are
not reproducible clean-start routes. They are therefore **excluded** from the
expert buffer. Only routes whose replay is not more than 0.25 s faster than the
source route are kept.

```text
expert source:   runs/ddqn_1_6_retrain/expert_1_6/expert_routes.npz
filter:          route_success and source_time - replay_time <= 0.25
kept:            85 routes / 7,365 transitions
```

This is a correction to the earlier 0095 claim that the 4.0667 s replay was a
usable clean route.

## Setup

```text
start:          runs/ddqn_expert_replay_1_6_lean_stick/.../ddqn_1705000_steps.zip
levels:         8 x 1-6
action repeat:  4
starts:         canonical + 31 NEAT state starts (50/50 weight)
expert:         85 stable successful routes, 7,365 transitions
anchor:         none (no old-level mixing)
budget:         100,000 steps, checkpoints every 12.5k
lr:             1e-4 actual (the manifest declared 5e-5, but the pre-fix
                resume path kept the checkpoint LR; config now records 1e-4
                for reproducibility. 0099 uses the fixed 5e-5 override.)
exploration:    0.10 -> 0.02 over 50%
margin:         1.0 x action margin, 0.5 x expert TD
```

## Baseline

The selected 1.705M 1-6 model is canonical-only:

```text
canonical:      1/1, 4.5667 s
all 32 states:  11/32 finishes, 20 deaths, 1 timeout
                median 6.1167 s, best 4.6667 s
                spawn_ok 32/32
```

## Gates

```text
canonical:      finish, <= 4.28 s (user 3.98 + 0.30)
coverage:       finish from every one of the 32 feasible starts
holdout:        [22,16], facing -1 must finish
safety:         no spawn fallback; report deaths and timeouts separately
```

## Run

```bash
.venv/bin/python tools/collect_expert_routes.py \
  --config experiments/0097_ddqn_1_6_feasible_starts

.venv/bin/python tools/build_expert_buffer.py \
  --config experiments/0097_ddqn_1_6_feasible_starts \
  --expert-data runs/ddqn_1_6_retrain/expert_1_6/expert_routes.npz \
  --output runs/ddqn_1_6_feasible_starts/expert_1_6/stable_expert_buffer.pkl \
  --max-replay-gain 0.25

.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0097_ddqn_1_6_feasible_starts
```

## Output

```text
runs/ddqn_1_6_feasible_starts/
  expert_1_6/fast_expert_routes.npz
  expert_1_6/stable_expert_buffer.pkl
  dqn/checkpoints/
  dqn/monitor.csv
  eval/
```

## Result

The 100k run was evaluated at every saved checkpoint (canonical + all 32
states):

```text
checkpoint   canonical   32-state coverage   deaths  timeouts
1.7175M      5.3833      26/32              6       0
1.7300M      4.6000      26/32              6       0
1.7425M      4.5500      27/32              5       0
1.7550M      4.6500      31/32              0       1
1.7675M      4.6500      32/32              0       0
1.7800M      4.6500      31/32              1       0
1.7925M      4.5500      31/32              1       0
1.8050M      4.5500      31/32              1       0
```

The first full row is **1.7675M: 32/32**, but canonical is 4.6500 s. The best
canonical/time trade-off before repair is **1.8050M: 4.5500 s, 31/32**, with
only `[21,16]` facing `-1` failing.

0099 repaired that single start with a targeted suffix expert buffer and
selected **1.815M**:

```text
runs/ddqn_1_6_start_repair/dqn/checkpoints/ddqn_1814952_steps.zip
canonical: 4.6000 s
32 states: 32/32, 0 deaths, 0 timeouts, median 6.0417 s
spawn_ok:  32/32
```

The selected any-start expert is 0099 1.815M. It still misses the 4.28 s speed
gate by 0.32 s; the stable NEAT route is 4.50 s. Do not use the 3.3/4.07 replay
as a deterministic target.

## Artifacts

```text
runs/ddqn_1_6_feasible_starts/dqn/checkpoints/ddqn_1767480_steps.zip  # first 32/32
runs/ddqn_1_6_feasible_starts/eval/check_*/                           # sweep
runs/ddqn_1_6_start_repair/dqn/checkpoints/ddqn_1814952_steps.zip     # selected
runs/ddqn_1_6_start_repair/eval/selected_repeat/summary.json
```

See [`../0099_ddqn_1_6_start_repair/README.md`](../0099_ddqn_1_6_start_repair/README.md).
