# Audit — lean state encoding

**Date:** 2026-10-28
**Status:** Godot lean 188-value encoding and Python decoder/one-hot encoder
implemented and tested. PPO/DQN driver wiring is still pending.
**Scope:** model-facing observation layout for the raw/lean training path.

## Decision

Godot emits a **lean 188-value** observation:

```text
49 primary tile IDs
49 rotations (radians)
49 signed phases
 9 player values
24 collision values
 8 flow deltas
= 188
```

Python does **not** derive lean from the raw 342 packet anymore. Python decodes
the lean packet and one-hots categorical fields for the model:

```text
Godot lean 188
    → Python decode
    → one-hot tile IDs, player state, powerup slots, collision types
    → model input D
```

The raw 342 encoding remains available as the validation/debug path.

## Tile identity

One primary symbol per cell, not a full multi-hot grid.

Priority:

```text
terrain → secret → non-terrain tile (object/static)
```

Implementation: `Level.get_primary_symbol()` / `get_surrounding_primary_cells()`.
A cell may contain terrain, a secret marker, and at most one non-terrain tile;
the non-terrain tile wins. Secret is a fallback when no non-terrain tile exists.

## Per-cell rotation and phase

### Rotation

Radians. Bounce-pad orientation is exposed by
`BouncePad.get_ai_state().rotation`; static/other cells use `0`.

### Phase

Signed availability/progress:

```text
+1.0   solid / present / fully interactable
+1..0  active transition (dissolving)
-1..0  inactive transition (disabled/repairing)
 0.0   transition boundary
```

Implemented in `DisolveBlock.get_ai_state()`:

| State | Phase |
|---|---|
| solid | `+1.0` |
| dissolving | `+clip(time_to_disable / 0.7, 0, 1)` |
| disabled | `-clip(time_to_next / 3.0, 0, 1)` |
| repairing | `-clip(time_to_solid / 0.5, 0, 1)` |

Static cells default to `phase=+1`, `rotation=0`.

Objects can own their state through:

```text
get_ai_state() -> { rotation: float, phase: float }
```

`BouncePad` and `DisolveBlock` implement it. Card/pickup respawn and moving
platforms still need their own implementation.

## Player block

Pruned to 9 values:

```text
velocity_x
velocity_y
facing
state_id
jump_phase
powerup_0
powerup_1
powerup_2
gold_ratio
```

Removed:

- `cell_x`, `cell_y` — absolute coordinates are a level shortcut;
- `on_floor`, `on_wall` — covered by `state_id` and collision rays;
- `run_active`, `started_walking`, `finished` — lifecycle belongs in `info`.

`gold_ratio` stays because it is a real optimization signal.

## Python model input

`src/plaiground/observations/lean.py` now provides:

- `DecodeLeanState` — validates the 188-value packet and splits it into tile
  IDs, rotations, phases, player, collisions, and flow;
- `LeanFeatures` — one-hot encodes:
  - 49 tile IDs over a supplied tile vocabulary;
  - player `state_id` over 12 states + unknown;
  - three powerup slots over `-1` + four kinds;
  - eight collision object types over a supplied/inferred vocabulary.

Numeric fields remain numeric. The model input is therefore larger than 188
and tile-vocabulary-dependent. Example: six tile names and the fixed eight
collision types produce 513 inputs.

```python
from plaiground.observations import build_lean_pipeline

pre = build_lean_pipeline(tile_names=game_tile_names)
model_x = pre.fit_transform(lean_batch)  # 188 -> D
```

## Implementation

Game:

- `AIObservationSpec.ENCODING_LEAN` / `LEAN_OBSERVATION_SIZE=188`;
- `PlayerAITrainingController._get_obs_lean()`;
- `Level.get_primary_symbol()` / `get_surrounding_primary_cells()`;
- `DisolveBlock.get_ai_state()`, `BouncePad.get_ai_state()`;
- AI training tests cover the 188 size and pruned player fields.

Python:

- `src/plaiground/observations/spec.py` — lean constants, offsets, validation, `decode_lean()`;
- `src/plaiground/observations/lean.py` — `DecodeLeanState`, `LeanFeatures`;
- `src/plaiground/observations/defaults.py` — `build_lean_pipeline()`;
- `tests/test_preprocessing.py` — 188 decode, one-hot shape, state/powerup/
  collision encodings, single/batch and pickle round-trip.

## Lean policy baseline

The first model is intentionally simple and declared in `src/plaiground/models/lean_mlp.py`:

```text
Godot lean 188
  → Python one-hot features
  → LeanMLP 512 × 512
  → action/value heads
```

`lean_policy_kwargs()` returns the matching SB3 `net_arch`. No custom embedding
extractor is used. This keeps save/resume/evaluation on
standard SB3 tooling. An embedding ablation can be added later if the one-hot
baseline has a sample-efficiency or generalization problem.

## Current limitations

- Card/pickup respawn phase is not exposed yet; those objects currently default
  to `phase=+1`.
- Moving-platform state/phase is not in the lean object contract yet.
- Collision-type and tile vocabularies must be stable and supplied by static
  game info for training/evaluation.
- The PPO/DQN drivers still use raw 342 directly; `--obs-encoding lean` /
  `--preprocessing lean` wiring remains pending.
- NEAT remains on the game-side 58 observation.

## References

- [`../../src/plaiground/observations/lean.py`](../../src/plaiground/observations/lean.py)
- [`../reference/observation-contracts.md`](../reference/observation-contracts.md)
- [`2026-10-28-raw-state-observation-decision.md`](2026-10-28-raw-state-observation-decision.md)
- [`../experiments/ppo-clean-start.md`](../experiments/ppo-clean-start.md)
