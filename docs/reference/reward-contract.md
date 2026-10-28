# Reward contract

**Status:** shipped 2026-10-28  
**Owner:** `godot/src/player/controllers/player_ai_training_controller.gd`  
**Raw telemetry:** `get_dynamic_info()` and the synchronizer's columnar `info`

The default training reward is computed in Godot. `info` also carries the raw
counters so PLaiGROUND can build a different reward without changing the game.

## Formula

```text
r_t = progress + efficiency + time + finish + survival

progress    = 1.0 * (C(s_prev) - 0.99 * C(s))       if both finite
efficiency  = -0.02 * (wasted_t - wasted_prev)
time        = -3.75 * (run_time_t - run_time_prev) * (1 - gold_ratio)
finish      = +500                                   once, on the finish edge
survival    = -50                                    once, on the death edge
```

### Progress

`C(s)` is the level's deterministic flow-field distance-to-exit. Progress is
the discounted cost potential `C(s) - gamma*C(s')`; `gamma = 0.99` matches PPO.
Leaving the level (`C = INF`) costs `-1.0` and resets the previous value; the
first step back in gets no spurious `INF` jump. A canonical reset seeds
`C(s_prev) = INF`, so the first decision has no progress term. A `state` reset
seeds `C(s_prev) = C(s)`, so its first transition can earn progress.

### Efficiency

`wasted_air_frames` is cumulative in the jump-arc tracker. The reward uses the
**delta** since the previous decision, not the cumulative value. Bounce-pad arcs
are excluded by the tracker.

### Time / gold ratio

```text
gold_ratio = clamp(gold_time / effective_time, 0.0, 3.0)
effective_time = max(run_time, gold_time / 3.0, 0.001)
```

- `gold_ratio = 1.0` at the level's gold time.
- `> 1.0` faster than gold; capped at `3.0` after `gold_time / 3`.
- `< 1.0` slower than gold; approaches `0.0` on very slow runs.
- `gold_time` is `target_time`: the campaign gold time, or the 10 s fallback for
  custom/unknown levels.

The time term is **per second of run time**, not per decision. At
`action_repeat=4` and 15 decisions/s, `-3.75/s` is equivalent to `-0.25` per
decision; it stays consistent for NEAT's `action_repeat=1` runs. At gold pace
the term is zero; below gold it is negative; above gold it is positive (bounded
by the cap). Force-start means the run clock is active immediately, so there is
no idle/start penalty.

### Finish and survival

- Finish: fixed `+500`. It does not depend on the agent's previous best or the
  run time; time is handled by `gold_ratio` and the discount. `best_run_time`
  remains telemetry only.
- Death: `-50` once, on the spike/hazard edge.

## Terminal death semantics

In training, spike contact is terminal at the hazard:

```text
spike -> Die state frozen at the hazard
      -> death_started -> controller: done = true, reward -50
      -> synchronizer returns the terminal observation at the spike
      -> next env.reset() samples the next start normally
```

The player is **not** reset to its starting position inside the death sequence.
The next episode uses the normal reset/start sample; it is not forced to the
canonical start. Normal (non-training) play keeps the existing death/reset
cinematic.

## Force-start reset

Training reset starts the run for the agent:

- `started_walking = true`
- `run_active = true`
- zero velocity, idle/grounded state
- clean-start `run_time` offset if supplied

For clean starts, `run_time` is the **source checkpoint time from the same
route/run**, not a mean or minimum across routes. The reset is one dict: the
physical fields and `run_time` are top-level, and the dynamic `world_state` is
nested; none of that restore metadata enters the observation. See
[`../audit/2026-10-28-clean-start-state-capture.md`](../audit/2026-10-28-clean-start-state-capture.md).

The player auto-runs; the agent's first decision is a normal jump/idle action.
There is no forced first jump and no idle phase. `--force-start-jump` /
`--start-jump` remain only as compatibility flags for old checkpoints.

## Raw telemetry for Python rewards

`info` adds:

| Field | Type | Meaning |
|---|---|---|
| `gold_ratio` | float | Clamped `[0, 3]` ratio above |
| `flow_cost` | float | `C(s)`, or `0.0` when out of bounds |
| `flow_in_bounds` | bool | `false` when `C(s) = INF` |
| `wasted_air_frames` | int | Cumulative wasted air frames |
| `run_time` | float | Run clock; includes the clean-start offset |
| `best_run_time` | float | Telemetry only |
| `finished` | bool | Finish edge for this episode |
| `deaths` | int | Death count for this episode |

A Python custom reward must recompute progress with its own previous `C(s)`
and use the same `gold_ratio` clamp. Static `info` also exposes `gold_time`,
`gold_ratio_cap`, and the default weights (`reward_time_per_second`,
`reward_finish_bonus`, `reward_death_penalty`, `reward_progress_weight`,
`reward_efficiency_weight`, `reward_shaping_gamma`) so a custom reward can
match the game or vary one term deliberately.

## Horizon and suicide tradeoff

The time term is bounded to `[-3.75/s, +7.5/s]`. With a 10 s episode horizon,
the worst-case time cost is about `-37.5`, still less negative than `-50` death.
That keeps death the worst outcome.

- PPO/DQN: `--max-decisions 150` at `action_repeat=4` = 10 s.
- NEAT `07`: 600 decisions at `action_repeat=1` = 10 s.
- NEAT `08`: default max steps 600 at `action_repeat=1` = 10 s.

If a caller uses a longer horizon, keep `time_cost <= 50` or raise the death
penalty; otherwise dying early can become better than timing out.

## Compatibility

- The old `reward_profile` values (`default`, `time`, `time_speed`) are gone;
  `--reward_profile` is ignored by the training controller.
- Old NEAT/PPO checkpoints trained with the forced start jump were trained on a
  different first-step distribution. Do not assume transfer under force-start;
  retrain or keep `--force-start-jump`/`--start-jump` as an eval compatibility
  flag. A 1-1 smoke of the archived 8×24 NEAT best genome timed out without the
  forced jump, which is expected for a pre-change checkpoint.
- Reward scale changed: old episode returns (finish ~100–220) are not
  comparable to new runs (finish 500 plus dense terms).
- `best_run_time` is still tracked for evaluation and telemetry, not reward.

## Validation

- `godot/tests/ai/test_ai_training_reward.gd` pins force-start, fixed finish,
  gold-ratio clamping, the per-second time term, and terminal death.
- `src/43_check_raw_obs.py` checks the raw observation/info `gold_ratio`
  agreement and the 342-value layout.
