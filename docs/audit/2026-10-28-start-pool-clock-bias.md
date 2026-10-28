# Audit — start-pool resets, the zeroed run clock, and the PPO canonical regression

**Date:** 2026-10-28
**Status:** investigation + design decision; no new training result.
**Scope:** `scripts/reinforcement/01_ppo.py` start-pool resets, `tools/make_ppo_start_pool.py`,
`scripts/reinforcement/02_dqn.py`, the game reset/reward contract, and the PPO
754/794 results.
**Game state at writing:** `../../dragon-jump-remaster` HEAD `360b055`
(`idk anymore`), clean. PLaiGROUND HEAD `b9dd103` (`update docs`), with an
uncommitted WIP in `scripts/reinforcement/02_dqn.py`.

> **2026-10-28 update:** the `time`/`time_speed` finish ratios described below
> were replaced by the simplified fixed `+500` finish and clamped `gold_ratio`
> time term. The clock-bias mechanism still applies: a cell-only reset zeroes
> `run_time`, so `gold_ratio` reads 3 and the dense time term is wrong. See
> [`../reference/reward-contract.md`](../reference/reward-contract.md). The
> measured finish-term table is historical.

> This note records the reasoning behind the DQN start-pool pivot. It does not
> claim that cell starts caused any specific PPO regression; §5 separates the
> mechanism from the evidence and lists the A/B that would settle it. The
> chosen follow-up contract is in
> [`../experiments/dqn-clean-start.md`](../experiments/dqn-clean-start.md) and
> the decision record
> [`2026-10-28-clean-start-state-capture.md`](2026-10-28-clean-start-state-capture.md),
> and the observation path is moving to
> [`2026-10-28-raw-state-observation-decision.md`](2026-10-28-raw-state-observation-decision.md).

## TL;DR

- The start pool mixes three reset forms: canonical, cell+facing, and NEAT
  prefix replay.
- A cell+facing reset teleports the player and zeroes `run_time`. From the
  game's point of view, that cell is the start of a fresh run.
- The `time` finish reward is `20 + 80 * clamp(gold / run_time, 0, 2.5)`. A
  late cell start therefore gets a much larger finish reward than a full run.
- The observation has no elapsed clock and no "I was reset here" flag. The same
  state can have two different return scales.
- Measured in the 794 1-1 run: canonical finish term ~97, focus cell starts
  ~161, medium/hard ~103–106, prefix ~73.
- This is a systematic bias, not noise. In DQN the replay buffer and TD
  bootstrapping keep and propagate it.
- The bias is a plausible contributor to the PPO canonical regressions, but it
  is **not proven**. The 794 run hit 2.40 s at 400k and 2.1333 s at 600k; the
  754 multilevel model hit 1.6667 s on 1-1 with only a 3% canonical share. Run
  the A/B before rewriting the PPO story.
- The flow field is a BFS distance, not a time. `time_to_exit` is `-1` on the
  floor rows. The usable time-to-cell prior comes from replaying the clean
  routes.

## 1. Architecture

### 1.1 Reset forms

| Form | Pool entry | What the game restores | Run clock |
|---|---|---|---|
| Canonical | no spawn, or `{"canonical": true}` | level start | 0 at the true start |
| Cell + facing | `{"cell": [x,y], "facing": ±1}` | position + facing | **0 at the teleported cell** |
| State snapshot | `{"cell", "facing", "state": {...}}` | position, velocity, `run_time`, `best_run_time`, `run_active`, `started_walking` | restored from the snapshot |
| Prefix replay | `{"prefix_actions": [...]}` | canonical reset, then replay actions | real elapsed time |

The canonical definition lives in the game controller:
`../../dragon-jump-remaster/godot/src/player/controllers/player_ai_training_controller.gd`.

The “State snapshot” row describes the current nested-`state` compatibility
format. The clean-start decision supersedes it with one reset dict: physical
fields and `run_time` top-level, nested `world_state`, and NEAT-side candidate
validation. See
[`2026-10-28-clean-start-state-capture.md`](2026-10-28-clean-start-state-capture.md).

### 1.2 Wrapper path

`SingleAgentGodotEnv.reset()` (`scripts/reinforcement/01_ppo.py:359`) samples a start by
weight and branches in this order:

1. `prefix_actions`: reset canonical, step the recorded actions, discard the
   prefix rewards.
2. `cell`: reset with `spawns=[{"agent": 0, "cell": ..., "facing": ...}]`.
3. canonical: plain reset.

For non-prefix starts the wrapper presses jump once during reset
(`start_jump`, line 390). The jump is part of the reset contract and is not
stored as a policy action. `step()` truncates at `max_decisions` and exposes
`finished`, `timed_out`, `run_time`, `start_source`, `start_kind`,
`start_cell`, and `prefix_len` (lines 455–502).

`load_start_pool()` (line 159) accepts a bare list or `{"starts": [...]}`,
keeps weights, `prefix_actions`, and `cell`/`facing`. `build_vec_env()` uses
`SubprocVecEnv(start_method="spawn")` for `n_envs > 1` (lines 560–611).

The DQN driver currently loads the PPO module through `importlib` to reuse
`SingleAgentGodotEnv`, `resolve_env_path()`, and `isolated_home()`
(`scripts/reinforcement/02_dqn.py:37–47`). That coupling is known issue #7.

### 1.3 Reward contract

`time` profile in the game controller:

- finish: `20 + 80 * clamp(gold / run_time, 0, 2.5)`
- per-step time penalty: `-0.10`
- efficiency: `-0.03`
- progress: potential-based `C(s) - 0.99 * C(s')`
- death: `-50`

1-1 gold (`par_time`) is **2.2944 s**. The finish term is the dominant reward
and assumes `run_time` is the full-level clock.

### 1.4 Observation contract

| Encoding | Size | Tail / features | Elapsed `run_time`? |
|---|---:|---|---|
| `legacy` | 58 | 7×7 tile IDs, exit direction, flow distance, velocity direction, floor/wall, jump phase, powerup flag | No |
| `multi_hot` | 754 | 7×7×15 semantic grid + 19-value tail (exit direction, time-to-exit, route gradient, velocity, floor/wall, jump phase, powerups) | No; `time-to-exit` is spatial |
| `multi_hot_sensor` | 794 | 754 + 20 ray distances + 20 hit flags | No |

The 794 1-1 model is `794 → 512 → 512 → 2` (verified in the saved PPO
checkpoint under `runs/ppo_fresh_1_1_sensor_794/run_1m_resume400k/`).

### 1.5 Start-pool provenance

`tools/make_ppo_start_pool.py` builds:

- one canonical entry;
- medium/hard strata cells from `tools/start_strata.py`;
- optional focus cells (one facing in train, the mirror held out);
- prefix entries: 5–80% of a clean route's actions, replayed from canonical;
- weights normalized to 1.0.

```text
NEAT clean routes ─┐
                   ├─▶ tools/make_ppo_start_pool.py ─▶ start_pool JSON
tools/start_strata.py ┘                               │
                                                      ├─ canonical
                                                      ├─ cell + facing
                                                      └─ prefix_actions
                                                              │
                                          SingleAgentGodotEnv.reset()
                                                              │
                                                     SB3 PPO / DQN
```

Pool files in use:

| File | Canonical | Cell/focus | Prefix | Notes |
|---|---:|---:|---:|---|
| `runs/neat_1_1_8x24_archive/ppo/start_pool.json` | 0.15 | 0.60 | 0.25 | default builder output |
| `.../start_pool_canonical.json` | 0.70 | 0.20 | 0.10 | intermediate |
| `.../start_pool_focus.json` | 0.65 | 0.25 | 0.10 | 794 1-1 run |
| `runs/ppo_multilevel_1_1_1_2_1_3/ppo/start_pool.json` | 0.030 / 0.036 / 0.069 per level | 0.12 / 0.155 / 0.367 | 0.05 / 0.06 / 0.115 | global weights; per-level canonical share collapses |

The multilevel pool allocates 20% of resets to 1-1, 25% to 1-2, and 55% to
1-3. For 1-3, only 6.9% of resets are canonical and 36.7% are cell/focus
starts.

### 1.6 DQN target

Fresh SB3 Double DQN:

- input: 794 `multi_hot_sensor`
- `512 → 512` ReLU, then 2 Q-values (idle, jump)
- replay buffer 100k, learning starts 2k, batch 256
- gamma 0.99, tau 1.0, target update 1k, `train_freq=4`, one gradient step
- epsilon 1.0 → 0.05 over 25% of the steps
- 4 envs via `SubprocVecEnv` for the full run; 1 env for the smoke

The 58-value fallback is `58 → 256 → 256 → 2`.

## 2. The zero-clock mechanism

### 2.1 Game code

`player_ai_training_controller.gd`:

```gdscript
func reset(spawn: Dictionary = {}) -> void:
    ...
    _run_time = 0.0                 # line 179 — unconditional
    _run_active = false
    ...
    _apply_spawn_or_default(spawn)  # line 190
    ...
    if not state.is_empty():
        _apply_state_after_reset(state)   # lines 195–196
```

`_apply_spawn_or_default()` only sets position and facing:

```gdscript
player.starting_facing_direction = 1 if facing > 0 else -1
player.starting_position = level.get_cell_global_position(cell)   # line 245
```

The clock is restored only when the spawn carries a `state` dictionary:

```gdscript
_run_time = float(state.get("run_time", 0.0))          # line 286
_best_run_time = float(state.get("best_run_time", INF))
_run_active = bool(state.get("run_active", true))
```

`_on_player_run_started()` arms the clock when the run starts (line 870).
After a cell reset, the wrapper's forced start jump is what arms it at the
teleported cell.

### 2.2 Reward arithmetic

| Actual covered segment | `run_time` at finish | Finish term |
|---|---:|---:|
| Full level | 2.15 s | ~105 |
| Last 1.0 s | 1.0 s | ~204 |
| Last 0.5 s | 0.5 s | 220 (cap) |

The per-step penalty is only `-0.10`; finishing 15 decisions early saves
~1.5, far less than the +100 finish difference.

## 3. Measured effect in the 794 run

Source: `runs/ppo_fresh_1_1_sensor_794/run_1m/episodes.csv`, 13,298 episodes
over the first 477,584 steps. The finish term is recomputed with gold 2.2944.

| `start_kind` | n | finish | median `run_time` | median finish term | median episode return |
|---|---:|---:|---:|---:|---:|
| canonical | 8,604 | 99.5% | 2.383 | 97.0 | 127.7 |
| medium | 1,300 | 99.7% | 2.200 | 103.4 | 135.0 |
| hard | 1,399 | 99.6% | 2.133 | 106.0 | 140.0 |
| focus | 650 | 99.7% | 1.300 | 161.2 | 204.0 |
| prefix | 1,345 | 99.6% | 3.467 | 72.9 | 96.8 |

Interpretation:

- The late-level **focus** starts are the worst: +64 finish reward and +76
  return vs canonical.
- Medium/hard cell starts are only +6/+9 because they are earlier in the
  level; the remaining run is long.
- Prefix starts are clock-correct but slower; the policy is not as good from
  mid-run handoffs. Prefix coverage is not free either.
- The 25% cell weight is not uniform damage. Most of it is in four focus
  entries (5% total weight) and the late starts.

## 4. Bias, not noise

If the same observation `S` appears:

- at 1.8 s in a canonical run, finishing at 2.15 s → finish term ~105;
- at 0 s in a cell episode at `S`, finishing at 0.35 s → finish term ~220.

The network sees the same input and is asked to fit both. The value converges
to a mixture. With Q-learning:

```text
Q(S_prev, a) = r + gamma * max_a' Q(S, a')
```

the inflation propagates backward into states that are only reached from the
canonical start.

PPO is on-policy: the bias is present in the batch, but old batches are
discarded. DQN keeps the biased transitions in the replay buffer for the whole
run, so the exposure is larger. That is why this matters more for the DQN
experiment than it did for PPO.

The observation has no `run_time` or reset flag, so the agent cannot separate
the two tasks. This is a POMDP from the learner's perspective. Adding the
clock to the observation would fix the observability but changes the
observation contract and invalidates existing models.

## 5. Did it poison PPO?

### 5.1 Evidence for

- The default builder pool is **60% cell starts** (canonical 0.15). The 794
  run used **25%** (canonical 0.65) and reached **2.1333 s**.
- `docs/known-issues.md` already describes the 1-1 PPO 512-wide plateau
  (2.40 s) as a cell/start-pool result.
- The multilevel pool is 55% 1-3 by weight, with only 6.9% canonical and 36.7%
  cell+focus for 1-3. 1-3 is the unsolved level (canonical 5.2333 s).
- The mechanism is measured in the 794 run (focus +64 finish term).

### 5.2 Evidence against / confounders

- The 794 run itself hit 2.40 s canonical at 400k and 2.1333 s at 600k; the
  plateau was temporary, not a permanent ceiling.
- The 754 multilevel 200k model reached 1.6667 s on 1-1 with only a 3%
  canonical share for 1-1. If cell starts were a strong poison, that result is
  hard to explain.
- The 794/754 runs differ in observation (794 vs 754), net size, steps, reward
  fixes, and curriculum. There is no controlled comparison.
- The 1-3 failure could be level difficulty and sample budget, not the pool.

### 5.3 The clean test

Run 1-1 PPO, 794 sensor, 512×512, 8 envs, same seed, `time` reward, AR4,
200k–400k:

| Arm | Pool | Cell starts |
|---|---|---:|
| A | `start_pool.json` | 60% |
| B | canonical 0.7 + prefix 0.3 | 0% |
| C | canonical 0.65 + route-prior cells 0.25 + prefix 0.10 | 0% clock-zero |

Metric: canonical-only deterministic time at 100k/200k/400k, plus per-start
finish rates. Do not select on average training return.

If B/C beat A on canonical, the hypothesis is supported. If not, look at the
other confounders. Do not rewrite the PPO history from the correlation alone.

## 6. The time-prior idea

### 6.1 The flow field is not time

`flow` is a BFS distance to the exit in cells. `time_to_exit` is the
physics-aware seconds-to-exit grid.

Examples from the current 1-1 analysis (re-exported at game HEAD `360b055`):

| Cell | `flow` | Flow-based prior `par * (start - flow) / start` | Route time to cell (approx.) |
|---|---:|---:|---:|
| (4,7) focus | 11 | 0.19 s | 1.08–3.67 s |
| (5,7) focus | 10 | 0.38 s | 0.98–3.83 s |
| (20,14) focus | 11 | 0.19 s | 0.95–1.25 s |
| (7,14) focus | 15 | negative | 1.48–1.71 s |

`time_to_exit` would give `par - tte(cell)`, which is sensible for platform
cells, but the current exporter writes `-1` for the entire floor rows 12–14,
including the canonical start (10,14). Re-exporting with the current game code
does not fix it. There is no complete time-to-cell grid from the game.

### 6.2 Why the prior is frozen, not the current policy's average

A live per-cell table from the current policy changes the reward function during
training. In DQN, old replay transitions were scored with a different
`run_time` prior, so the same `(state, action)` gets a different target; the
buffer would have to be rescored or flushed. In PPO, returns and advantages
shift between iterations even when the policy did not change. The prior is
therefore fixed for the run: route minimum (or a percentile), not the policy's
live average. A reference-policy table is acceptable if it is frozen at the
start of the run.

### 6.3 Route-time table

Every one of the 43 cell entries in `start_pool_focus.json` is visited by at
least one clean route. During the route replay:

1. record `run_time` at each cell (exact, not interpolated);
2. take the minimum (fastest route that visits the cell) or a percentile;
3. use it as the synthetic `run_time` at that cell.

Approximate values from linear interpolation of `path_cells` by index (rough;
the exact replay will refine them):

| Cell | min | median |
|---|---:|---:|
| (4,7) | 1.08 s | 3.67 s |
| (5,7) | 0.98 s | 3.83 s |
| (20,14) | 0.95 s | 1.25 s |
| (7,14) | 1.48 s | 1.71 s |
| (8,7) | 0.93 s | 2.08 s |

Use the minimum (or the 25th percentile). The median over-penalizes: the
(4,7) median 3.67 s is longer than the full par time, so a cell start there
would get a finish term of ~68 and be learned as a bad state.

### 6.4 Synthetic state

```json
{
  "cell": [4, 7],
  "facing": 1,
  "state": {
    "velocity": [0.0, 0.0],
    "state_name": "Idle",
    "run_time": 1.08,
    "best_run_time": 1e9,
    "run_active": true,
    "started_walking": true
  }
}
```

No `position`: the game uses the cell placement. No forced start jump: the run
is already active. The game restores the clock in `_apply_state_after_reset()`.

Validation per entry:

- reset once from it;
- check `spawn_cell == cell`;
- check reset `run_time` is within ~0.05 s of the prior;
- check `on_floor` and `powerup_count == 0`.

Caveats:

- The prior is a lower bound; a slower student gets a slightly inflated
  reward. The direction is the same as the bug, but the magnitude is ~10
  points instead of ~115.
- The player starts from standstill, not arrival velocity; the remaining time
  is slightly longer, so the prior is conservative in the other direction.
- It does not give the student practice on the early level.
- Real state snapshots are strictly better where available; the prior is for
  cells with no usable snapshot.

## 7. Decisions

1. **Do not use the 60% cell pool for new work.** The default builder output
   is not a safe `time`-reward pool.
2. **First DQN smoke:** 1-1, 794 `multi_hot_sensor`, 512×512, canonical+prefix
   pool (0% cell) or the route-prior pool. New run directory
   `runs/dqn_startpool_1_1`, separate from the extractor run.
3. **Proper pool:** replay the clean routes, record exact time-to-cell and
   optional state snapshots, build canonical + route-prior cells + prefix.
4. **PPO A/B:** run the three arms above before claiming the poisoning. Record
   the result in this note or a follow-up.
5. **Do not install** any DQN/PPO/754/794 model. The deployed 58/256 JSON
   remains the ship baseline.
6. **Log start metadata:** the multilevel `episodes.csv` has no `start_kind`;
   add it before the next long run.

Operational note: `tools/make_ppo_start_pool.py` writes
`start_pool_summary.json` and `eval_starts_heldout.json` next to its
`--output`. Use a new output directory when generating an experimental pool so
the focus pool's sidecars are not overwritten.

## 8. Open questions

- Does the A/B confirm the canonical regression? Which arm wins?
- Is the route-time minimum or the 25th percentile the better prior?
- Is the prior worth it vs. real state snapshots or prefix-only?
- How should multilevel pools allocate canonical weight per level? The current
  global weights give 1-3 6.9% canonical and 36.7% cell+focus.
- Should `run_time` or a start-context flag be added to the observation? That
  is a game-side contract change and breaks existing models.
- Can DQN learn 1-1 with 794/512 within 200k–500k steps?

## 9. References

- `scripts/reinforcement/01_ppo.py` — `SingleAgentGodotEnv`, `load_start_pool`,
  `build_vec_env`.
- `tools/make_ppo_start_pool.py` — pool construction.
- `scripts/evaluation/01_evaluate_strata.py` — persistent-env evaluation pattern.
- `scripts/reinforcement/02_dqn.py` — DQN extractor/train/eval.
- `../../dragon-jump-remaster/godot/src/player/controllers/player_ai_training_controller.gd`
  — reset and reward contract.
- `../../dragon-jump-remaster/godot/src/level/level.gd` — flow field and
  time-to-exit.
- `runs/neat_1_1_8x24_archive/ppo/start_pool*.json` — pool weights.
- `runs/ppo_fresh_1_1_sensor_794/run_1m/episodes.csv` — measured start-kind
  table.
- `runs/ppo_multilevel_1_1_1_2_1_3/ppo/start_pool.json` — multilevel weights.
- `docs/experiments/dqn-route-spawns.md` — DQN experiment doc.
- `docs/audit/2026-10-28-dqn-pipeline-review.md` — extractor pipeline review.
- `docs/known-issues.md` — living issue list.
