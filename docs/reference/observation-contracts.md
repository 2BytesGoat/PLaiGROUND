# Observation and reset contracts

Python-side summary of the game's training contract. The canonical definitions
live in the game repo:

- `godot/src/ai/ai_observation_spec.gd`
- `godot/src/player/controllers/player_ai_training_controller.gd`
- `godot/src/training/synchronizer.gd`
- `docs/technical/training.md` in the game vault

`src/obs_spec.py` is the Python mirror used by drivers and debugging tools.

## Observation encodings

### `legacy` — 58 values

| Index | Count | Feature | Source |
|---|---:|---|---|
| 0–48 | 49 | 7×7 terrain tile IDs around the player (row-major) | `Level.get_surrounding_cells(..., 3)` |
| 49–50 | 2 | Normalized direction to exit (x, y) | `Level.exit_global_position` |
| 51 | 1 | Normalized flow-field distance to exit, `clamp(flow / 200, 0, 1)` | `Level.get_flowfield_value` |
| 52–53 | 2 | Normalized velocity direction (x, y); magnitude is not observed | `Player.velocity` |
| 54 | 1 | On floor (0/1) | `Player.on_floor()` |
| 55 | 1 | On wall (0/1) | `Player.on_wall()` |
| 56 | 1 | Jump-peak percentage | `Player.perc_to_peak` |
| 57 | 1 | Has powerup (0/1) | `Player.has_powerups()` |

Limits: no velocity magnitude, no flow gradient, no dissolve timer/collision
state, no powerup type/count, no bounce facing. Decisions are 15 Hz at
`action_repeat=4`.

### `multi_hot` — 754 values

- 735 = 7×7 cells × 15 values, row-major.
- Per cell: type multi-hot (8: solid, hazard, exit, bounce, dissolve, powerup,
  pickup, secret), powerup kind (2), dissolve state (2), dissolve timer (1),
  bounce orientation (2).
- 19-value tail: exit direction (2), time-to-exit (1), route gradient (2),
  velocity (2), floor/wall (2), jump phase (1), powerup counts (4), top-of-stack
  type (4), powerup total (1).

Spec version 2. Models saved against a different encoding are incompatible by
design.

### `multi_hot_sensor` — 794 values

754 base + 20 head-mounted ray distances + 20 hit flags. Training-only
experiment; no installed runtime. The game creates the ray sensor lazily for
this encoding.

## Actions and cadence

- Discrete binary action: `0 = idle`, `1 = jump`.
- `action_repeat=4` by default: one decision per four physics frames, 15
  decisions/s at 60 Hz.
- `max_decisions=150` = 600 physics ticks = 10 s of game time.
- `speedup` scales wall-clock physics ticks and `Engine.time_scale`; per-tick
  simulation and run times are unchanged.

## Info fields exposed to Python

The synchronizer sends a static block once and columnar dynamic fields per step
(schema 2); `GodotEnv._decode_info()` reconstructs the per-agent dict. Useful
fields:

| Field | Meaning |
|---|---|
| `cell` | `[x, y]` grid cell of the player |
| `facing_direction` | ±1 |
| `on_floor`, `on_wall` | contact booleans |
| `velocity` | raw `[vx, vy]` |
| `perc_to_peak` | jump-phase percentage |
| `powerup_count` | number of held powerups |
| `state` | player state-machine state name |
| `run_time` | elapsed run clock (seconds) |
| `best_run_time` | agent's best full run this episode/controller lifetime |
| `run_active` | whether the run clock is running |
| `started_walking` | whether the run has started |
| `finished`, `deaths` | terminal/episode counters |
| `efficiency`, `flow` | eval telemetry (air-arc usefulness, cell coverage) |
| `spawn_cell` | cell the controller actually reset to |
| `global_position` | raw world position |

`spawn_cell` and the wrapper's `last_reset_spawns` are how you verify that a
requested spawn was accepted. The controller silently falls back to the
canonical start for invalid cells (non-standable, exit, malformed).

## Reset forms

`GodotEnv.reset(agents=..., spawns=...)` sends a reset message. Supported spawn
entries:

### 1. Canonical

No `spawns` entry, or `{"canonical": true}` in a PLaiGROUND start pool. The
level's normal start.

### 2. Cell + facing

```json
{"agent": 0, "cell": [10, 14], "facing": 1}
```

Only position and facing are restored. `run_time` is zeroed. Use the start-jump
convention for stationary starts: press jump once during reset, then hand
control to the policy.

### 3. Grounded state snapshot

```json
{"agent": 0, "cell": [10, 14], "facing": 1,
 "state": {"position": [x, y], "velocity": [vx, vy], "state_name": "Move",
           "run_time": 1.2, "best_run_time": 2.3, "run_active": true,
           "started_walking": true}}
```

Restores the run clock and velocity after `Player.reset()`. Supported for
grounded, powerup-free states only. Jump states, powerup stacks, and dissolve
timers are not supported; filter them in the collector. Do not apply the start
jump to a snapshot whose run is already active.

### 4. Prefix replay (PLaiGROUND wrapper)

`SingleAgentGodotEnv` start-pool entries can carry `prefix_actions`. The wrapper
resets to canonical, replays the actions, then hands control to the policy. This
preserves the full run clock without a state snapshot. It uses recorded actions
during reset, not as imitation labels.

## Reward profiles

The game reward is a five-term sum: flow progress, jump-arc efficiency, per-step
time pressure, finish bonus, death penalty. PLaiGROUND can select a training-only
profile:

| Profile | Finish term | Per-step time | Efficiency |
|---|---|---|---|
| `default` | `+100 + 50 * previous_best / run_time` | −0.05 | −0.02 |
| `time` | `+20 + 80 * clamp(gold_time / run_time, 0, 2.5)` | −0.10 | −0.03 |
| `time_speed` | `+20 + 160 * clamp(1.75 / run_time, 0, 2.0)` | −0.10 | −0.03 |

Notes:

- Progress is potential-based shaping:
  `+1.0 * (C(s) - 0.99 * C(s'))`, with gamma 0.99 matching PPO.
- The finish bonus uses the previous best for `default`; the fixed gold/1.75
  target for `time`/`time_speed`.
- A death costs −50.
- The `time`/`time_speed` formulas assume a **full run clock**. Cell-only resets
  zero it; a short partial run looks extremely fast. Use state snapshots or
  prefix replay for mid-route training.

See [`../known-issues.md`](../known-issues.md) § DQN for the concrete bias and
the audit for the math.
