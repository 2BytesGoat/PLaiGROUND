# Decision — raw-state observation contract and Python encoders

**Date:** 2026-10-28
**Status:** observation packet and Godot/Python lean encoding implemented and
validated; card phase and driver wiring pending
**Scope:** the Godot ↔ Python observation path for PPO/DQN training; the
shipped 58-value runtime model stays in Godot.
**Related:** [`../reference/observation-contracts.md`](../reference/observation-contracts.md),
[`../experiments/dqn-clean-start.md`](../experiments/dqn-clean-start.md),
[`2026-10-28-start-pool-clock-bias.md`](2026-10-28-start-pool-clock-bias.md).

> **Decision.** Godot sends **local raw state** every step. Python builds the
> training observation from it. Godot stops owning multi-hot/one-hot feature
> engineering; it keeps only the 58-value encoder needed by the shipped
> runtime model. Normalization stays in Godot because it is a unit/physics
> concern. One-hot and feature composition move to Python.

## 1. Why

The current path pre-processes everything in Godot:

- 58 legacy: 49 tile IDs + 9 tail values.
- 754 multi-hot: 7×7×15 semantic grid + 19-value tail.
- 794: 754 + 20 rays × (distance, hit).

That puts feature engineering in the game and sends a large, fixed vector.

The 754/794 layouts are overengineered for the information the agent needs:

- 7 of the 15 values per cell are powerup/dissolve/bounce state. On
  1-1/1-2/1-3 those are always zero; on other levels they are sparse.
- The 8 type bits are mostly one-hot. For most cells only one bit is set;
  a tile ID plus a lookup conveys the same thing with far less data.
- The 19-value tail spends 9 values on powerup counts/top/total. Those are
  zero on 1-1/1-2/1-3 and sparse elsewhere.
- `time_to_exit` and the 2D route gradient are static per cell. Sending them
  every step duplicates data that can be derived from the level grid.
- The 20-ray fan gives distance + hit but no collision type. It has 18°
  angular resolution, but the 7×7 grid already covers the nearby angles with
  exact tile data. For a game whose actions are left/right/jump, 8 directions
  at 45° plus the grid is enough.

A rough first-layer weight check on the deployed 58 model supports keeping the
local geometry, flow, velocity, exit direction, and jump phase; the powerup
flag is low-use on 1-1. The 794 model's weights are not a clean per-feature
ranking because the grid is a dense layer, so this decision is based on the
information design, not on a precise attribution study.

The practical problems with the current approach:

- The packet is large: 794 float32 = 3.2 KB per agent per step.
- New training encodings require game changes and a new observation-space
  version.
- Python cannot freely combine or ablate features.
- The 7×7×15 expansion is redundant with a tile ID grid.

## 2. Responsibility boundary

| Concern | Godot | Python |
|---|---|---|
| Physics, collision, raycasts | yes | no |
| Local window extraction | yes | no |
| Normalization (units, ray length, flow scale) | yes | no |
| Tile IDs / tile states | send raw | one-hot / type bits / compose |
| Player pose/velocity/facing/state | send normalized | use / one-hot state |
| Powerup stack | send kind IDs | counts, top, one-hot |
| 8-way collisions | send distance/hit/type | compose / one-hot type |
| 8-way flow | send normalized delta | use |
| Runtime 58 encoder | yes | no (export path only) |

Normalization is a physical-unit concern, so Godot owns it. Representation is
a learning concern, so Python owns it.

## 3. Per-step packet

Every step, for the local 7×7 window:

```text
tile_id_window       7×7 raw tile IDs
tile_state_window    7×7 dynamic state:
                       dissolve state + normalized timer
                       bounce orientation
                       moving-platform state/phase
                       powerup kind (if a pickup is present)

player               cell x/y
                     normalized velocity (x, y)
                     facing (±1)
                     state ID
                     on_floor, on_wall
                     jump phase (0..1)
                     held powerup stack (kind IDs, bottom → top)
                     gold_ratio (clamp 0..3)
                     run_active, started_walking, finished

collisions           8 directions × (normalized distance, hit, object type ID)

flow                 8 directions × normalized delta
                     delta = flow[neighbor] - flow[player]
```

No full static level grid is sent every step. The window is the observation.

## 4. Static header

Sent once per level/reset:

```text
max_speed, max_fall_speed
ray_length
flow_delta_scale
dissolve_timer_scale
par_time
level_id
exit cell
```

Python can invert normalized values for debugging or new encoders.

## 5. Normalization

| Value | Formula |
|---|---|
| Collision distance | `distance / ray_length`, clamped 0..1 |
| Flow delta | `clamp((flow[neighbor] - flow[player]) / FLOW_DELTA_SCALE, -1, 1)`; `FLOW_DELTA_SCALE` is a fixed constant, e.g. 4 cells |
| Velocity x | `clamp(vx / max_speed, -3, 3)` (current 754 convention) |
| Velocity y | `clamp(vy / max_fall_speed, -1.5, 1.5)` |
| Gold ratio | `clamp(gold_time / max(run_time, gold_time/3, 0.001), 0, 3)` |
| Jump phase | already 0..1 |
| Dissolve timer | `time_to_next_state / dissolve_timer_scale`, clamped 0..1 |
| Facing | ±1 |
| Tile IDs, tile states, powerup kinds, collision types, player state | categorical, sent raw |

`FLOW_DELTA_SCALE` must be a fixed constant, not the per-level maximum flow:
adjacent deltas are usually −1/0/+1, and dividing by a 12-cell max flow would
crush them to ±0.08.

## 6. What we drop

- The 7×7×15 one-hot grid.
- Per-step `time_to_exit` and 2D route gradient.
- The 9-value powerup tail (counts/top/total). The stack remains in player
  state; Python can encode counts if it wants.
- The 20-ray fan. Replaced by 8 directions with collision type.
- Sending the full static level grid every step.
- Velocity direction without magnitude. Send normalized velocity components.

## 7. What we keep

- The 7×7 local window: tile IDs + dynamic tile states.
- Player raw state: velocity, facing, state, floor/wall, jump phase,
  powerup stack, run clock.
- 8-way collision sensors with distance, hit, and object type.
- 8-way flow deltas.
- The 58 runtime encoder in Godot for the deployed model.
- NEAT can stay on the game-side 58 until it is migrated separately.

## 8. Compatibility

- The old 794 model expects the 20-ray fan. It is **not** a drop-in target for
  the lean packet. Keep it as a reference baseline; the lean observation is a
  new model/encoding.
- The 58 deployed model stays compatible because Godot keeps its runtime
  encoder.
- PPO/DQN training switches to the raw packet + Python encoder after parity
  and throughput checks.
- If a reproduced encoding is needed for comparison, build it in Python from
  the raw packet and add a parity test; do not add it back to the game.

## 9. Migration plan

1. Define the binary/JSON packet and static header.
2. Add a raw-state mode to the Godot training synchronizer; keep the encoded
   path as the default until cutover.
3. Add the Python decoder and encoder modules:
   - `legacy58` (for parity/debugging),
   - `multi_hot754` (if needed for comparison),
   - `multi_hot_sensor794` (only if we keep the 20-ray fan),
   - `lean` (tile IDs + states + player + 8-way collisions + 8-way flow).
4. Parity tests for any reproduced encoding; throughput benchmark at 1/4/8
   envs; compare decisions/s and packet size.
5. Cut over PPO/DQN training to the raw packet + lean encoder.
6. Remove the game-side 754/794 training encoders; keep the 58 runtime encoder.
7. Migrate NEAT later if desired.

## 10. Implementation and validation (2026-10-28)

Implemented:

- `AIObservationSpec.ENCODING_RAW = "raw"`, size **342**.
- Layout: 49 tile IDs, 49×5 tile states, 16 player values, 8×3 collisions,
  8 flow deltas.
- `_get_obs_raw()` in the training controller.
- `_get_raw_collisions()` casts 8 rays at 45° with
  `PhysicsRayQueryParameters2D`, returns normalized proximity, hit, type.
- `_get_raw_flow_deltas()` returns `(flow[neighbor] - flow[player]) / 4`,
  clamped to [-1, 1].
- `obs_spec.decode_raw()` decodes the flat vector.
- `tools/check_observation.py` validation script.
- `_register_agent()` split out of `_ensure_sensor()` so raw/legacy controllers
  join the `AGENT` group without instantiating the 20-ray sensor.

Validation:

| Check | Result |
|---|---|
| 1-1, 10 steps | 342 values, 10/10 player-cell matches, center tile `Player`, no powerup/dissolve/bounce state, collision type 0 (solid), flow in [-0.5, 1.0] |
| 1-8, 20 steps | 342 values, 20/20 player-cell matches, one powerup-kind cell in the window; random agent did not pick it up |
| 794 regression | 1 env × 200 steps, observation space `[794]`, no errors |
| 58 legacy regression | 1 env × 200 steps, observation space `[58]`, no errors |

Implemented after this note:

- Game-side **primary-symbol** query: `Level.get_primary_symbol()` /
  `get_surrounding_primary_cells()` use terrain → secret → object/static.
- Godot **lean 188-value** encoding (`AIObservationSpec.ENCODING_LEAN`) with
  nine player fields.
- `BouncePad.get_ai_state()` and `DisolveBlock.get_ai_state()` for rotation and
  signed phase.
- Python `lean` decoder plus categorical one-hot encoder
  (`src/plaiground/observations/lean.py`), covered by
  [`2026-10-28-lean-state-encoding.md`](2026-10-28-lean-state-encoding.md).

Still pending:

- generic card/pickup respawn phase; those currently default to `phase=+1`;
- moving-platform state/phase;
- collision type coverage for hazard/exit/powerup hits (the short checks only
  exercised solid);
- driver wiring for `--preprocessing lean`; PPO/DQN still use raw 342 directly.

## 11. Impact on the clean-start/reset plan

The reset payload and the raw player state share fields, but the reset is one
dict:

```text
reset: velocity, facing, state_name, powerup_stack, run_active,
       started_walking, run_time, best_run_time, world_state
```

- The physical fields and `run_time` are top-level; `world_state` is nested.
  There is no separate `state`/`info` reset split.
- The game restores the powerup stack from the world state/held entries; Python
  records the NN stack from the raw state and the held IDs from `world_state`.
- No separate `powerup_stack` info field is needed for the NN; the dynamic
  `world_state` carries held pickup IDs/order for restoration.
- The clean-start pool records the player state at each checkpoint and emits
  entries with zero velocity, the source-checkpoint `run_time`, the held
  powerup stack, and the dynamic `world_state`.
- The source time and world state must come from the same run; no mean/min
  across routes.
- The clean-start target is raw/lean state, not 794. See
  [`2026-10-28-clean-start-state-capture.md`](2026-10-28-clean-start-state-capture.md).

## 12. Open questions

- Packet format: compact binary frame + JSON header vs JSON per step.
- `FLOW_DELTA_SCALE`: start at 4; revisit if the 8-way deltas saturate.
- Collision type mapping: which node groups/classes map to which IDs.
- Moving-platform state: phase/offset per cell; include only in the window.
- Dissolve timer: keep the existing 3.0 scale.
- Throughput: Python encoding cost at 8 envs.
- Whether any level needs 12 rays instead of 8; add only after an ablation.

## 13. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-28 | Godot sends raw local state, Python encodes | feature engineering belongs to the learning side |
| 2026-10-28 | Send the 7×7 window every step | it is the observation; no full static grid in the packet |
| 2026-10-28 | Normalize in Godot, one-hot in Python | units/physics vs representation |
| 2026-10-28 | 8 collision rays, not 20 | the 7×7 grid covers nearby angles; 45° is enough for left/right/jump |
| 2026-10-28 | Add collision type per ray | the 20-ray fan only had distance/hit |
| 2026-10-28 | 8-way flow deltas instead of scalar/time-to-exit | explicit local direction; static data not resent |
| 2026-10-28 | Drop 7×7×15 one-hot and per-step time_to_exit/gradient | redundant with tile IDs + static level data |
| 2026-10-28 | Keep the 58 runtime encoder in Godot | the shipped model runs in-game |
| 2026-10-28 | Old 794 is a separate baseline | lean observation is a new model, not a drop-in |
| 2026-10-28 | Raw observation implemented and validated | size 342; 1-1/1-8 checks pass; 794 path unchanged |
