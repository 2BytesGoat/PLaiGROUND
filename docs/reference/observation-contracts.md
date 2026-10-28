# Observation and reset contracts

Python-side summary of the game's training contract. The canonical definitions
live in the game repo:

> **Raw-state change (2026-10-28):** the `raw` encoding is implemented
> (342 values) and validated. Godot also emits the lean **188-value** encoding;
> Python decodes it and one-hots categorical fields. Card/object phase and
> driver wiring are still pending. The 58/754/794 layouts below remain the
> current runtime/legacy contracts. See
> [`../audit/2026-10-28-raw-state-observation-decision.md`](../audit/2026-10-28-raw-state-observation-decision.md)
> and [`../audit/2026-10-28-lean-state-encoding.md`](../audit/2026-10-28-lean-state-encoding.md).

- `godot/src/ai/ai_observation_spec.gd`
- `godot/src/player/controllers/player_ai_training_controller.gd`
- `godot/src/training/synchronizer.gd`
- `docs/technical/training.md` in the game vault

`src/plaiground/observations/spec.py` is the Python mirror used by drivers and debugging tools.

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

### `raw` — 342 values (implemented 2026-10-28; Python encoders available)

Godot sends primitive local state; Python builds the training features:

- 49 tile IDs (7×7 window, symbol indices).
- 49 × 5 tile states: powerup kind, dissolve state, dissolve timer,
  bounce orientation x/y.
- 16 player values: cell x/y, normalized velocity, facing, state ID,
  on-floor/on-wall, jump phase, held powerup stack (3), `gold_ratio`
  (clamped 0–3), run-active/started/finished flags.
- 8 × 3 collisions: normalized proximity, hit, object type ID.
- 8 flow deltas: `(flow[neighbor] - flow[player]) / 4`, clamped to [-1, 1].

`obs_spec.RAW_*` defines the offsets; `obs_spec.decode_raw()` splits the
vector. `tools/check_observation.py` validates the path. This is the input to the
Python feature encoders; it is not yet a trained model input.

### `lean` — 188 values (Godot-emitted)

```text
49 primary tile IDs
49 rotation values (radians)
49 signed phase values
9 player values: velocity x/y, facing, state ID, jump phase,
  powerup stack (3), gold_ratio
24 collision values
8 flow deltas
= 188
```

Tile priority is `terrain → secret → object/static`. A cell has at most one
non-terrain tile, which wins over secret/terrain.

Phase semantics:

```text
+1.0   solid / present / fully interactable
+1..0  active transition (dissolving)
-1..0  inactive transition (disabled/repairing)
 0.0   transition boundary
```

Python decodes the packet and one-hots categoricals (49 tile IDs, player
`state_id`, powerup slots, collision object types). The resulting model input
size depends on the vocabularies; 188 is the wire size, not the model width.
`build_lean_pipeline()` requires the game's tile vocabulary.

Implemented in `AIObservationSpec` / `_get_obs_lean()` and
`src/plaiground/observations/lean.py`. Card respawn and moving-platform phase, plus
PPO/DQN driver wiring, are still pending.

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
| `run_time` | elapsed run clock (seconds; includes clean-start offset) |
| `gold_ratio` | clamped gold-time ratio in `[0, 3]` |
| `flow_cost` | absolute flow-field cost `C(s)` |
| `flow_in_bounds` | false when `C(s)` is `INF` |
| `wasted_air_frames` | cumulative wasted air frames |
| `best_run_time` | agent's best full run this episode/controller lifetime (telemetry) |
| `run_active` | whether the run clock is running |
| `started_walking` | whether the run has started |
| `finished`, `deaths` | terminal/episode counters |
| `efficiency`, `flow` | eval telemetry (air-arc usefulness, cell coverage) |
| `spawn_cell` | cell the controller actually reset to |
| `global_position` | raw world position |

`spawn_cell` and the wrapper's `last_reset_spawns` are how you verify that a
requested spawn was accepted. The controller silently falls back to the
canonical start for invalid cells (non-standable, exit, malformed).

Static info also exposes `gold_time`, `gold_ratio_cap`, and the default reward
weights (`reward_time_per_second`, `reward_finish_bonus`,
`reward_death_penalty`, `reward_progress_weight`, `reward_efficiency_weight`,
`reward_shaping_gamma`) for Python custom rewards. See
[`reward-contract.md`](reward-contract.md).

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

Only position and facing are restored. `run_time` is zeroed. Training reset
force-starts the run; there is no idle state and no forced first jump. Cell-only
starts are still clock-biased for mid-route positions because `run_time` is zero;
use a state snapshot with a route-time offset for mid-route training.

### 3. Grounded state snapshot

Current game support:

```json
{"agent": 0, "cell": [10, 14], "facing": 1,
 "state": {"position": [x, y], "velocity": [vx, vy], "state_name": "Move",
           "run_time": 1.2, "best_run_time": 2.3, "run_active": true,
           "started_walking": true}}
```

Restores the run clock and velocity after `Player.reset()`. Supported for
grounded, powerup-free states only. Jump states, powerup stacks, and dissolve
timers are not supported; filter them in the collector. Training reset
force-starts the run after restoring the state; do not inject a start jump. This
nested `state` shape is the current compatibility format; the clean-start target
below uses one reset dict.

### 3a. Clean-start single reset dict

The clean-start contract uses one reset dict. Restore metadata sits alongside
the physical fields; only `world_state` is nested:

```json
{"agent": 0, "cell": [12, 18], "facing": 1,
 "position": [x, y], "velocity": [0.0, 0.0],
 "state_name": "Move", "powerup_stack": [0],
 "run_active": true, "started_walking": true,
 "run_time": 1.08, "best_run_time": 1e9,
 "world_state": {"version": 1, "objects": {}, "cells": {}}}
```

- The observation remains a single dict; `run_time`, `best_run_time`, and
  `world_state` are ignored by the encoder.
- `run_time` is the source-checkpoint time; `gold_ratio` is computed from it.
- `world_state` restores dynamic cells/objects and held pickups.
- The NEAT extraction/pool builder rejects jump states or unsupported dynamic
  objects; Godot applies the reset or returns an error, not a canonical
  fallback.
- Reset order: `Player.reset()` → restore `world_state` → restore player
  state/clock → rebuild held powerups → force-start.

The game implements this extension. The reset reply advances one physics tick,
so a source `run_time` of 1.2667 s reports roughly 1.3333 s on the first
observation; measure that drift in the clean-start gates. See the
[clean-start decision record](../audit/2026-10-28-clean-start-state-capture.md).

### 3b. World-state request

The clean-start capture path requests a separate optional block after a step,
at a checkpoint transition:

```json
{"type": "info_request", "request": ["world_state"], "agents": [0]}
```

```json
{"type": "info", "request": ["world_state"],
 "world_state": {"0": {"version": 1, "objects": {}, "cells": {}}}}
```

The game is paused between the step reply and the next action, so the snapshot
matches the observation and `run_time`. This is used during route capture, not
during normal training; training resets carry the stored snapshot as the
top-level `world_state` field. The per-step `info` remains columnar and does not
include the world-state block. The `info_request` protocol is implemented.

### 4. Prefix replay (PLaiGROUND wrapper)

`SingleAgentGodotEnv` start-pool entries can carry `prefix_actions`. The wrapper
resets to canonical, replays the actions, then hands control to the policy. This
preserves the full run clock without a state snapshot. It uses recorded actions
during reset, not as imitation labels. Training reset now force-starts the run,
so prefix replay is legacy/clock-preserving and should be re-recorded; clean
state starts are the preferred next path.

## Reward

The training reward is the simplified five-term sum documented in
[`reward-contract.md`](reward-contract.md):

```text
progress + efficiency + time + finish + survival
+1.0*(C(s)-0.99*C(s')) -0.02*dwasted
-3.75*delta_run_time*(1-gold_ratio) +500 on finish -50 on death
```

`gold_ratio` is clamped to `[0, 3]`; the time term is per second of run time.
`best_run_time` is telemetry only. The old `default`/`time`/`time_speed`
profiles and `--reward_profile` are gone/ignored.

A cell-only reset still zeroes `run_time`; a mid-route start will therefore look
faster than it is. Use a state snapshot with a route-time offset or prefix
replay for mid-route training. See [`../known-issues.md`](../known-issues.md)
§ DQN and the clock-bias audit.
