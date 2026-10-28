# `src/plaiground/observations/` — observations and preprocessing

This package turns the game's **188-value lean observation** into the flat
vector consumed by a learning model.

## Contract

- Input: one lean observation `(188,)` or a batch `(N, 188)`.
- Decode: `LeanStateBatch` with tile IDs, rotations, phases, player,
  collisions, and flow.
- Output: one float vector `(D,)` or `(N, D)`.
- `fit` follows the sklearn convention.

## Usage

```python
from plaiground.observations import build_lean_pipeline

pre = build_lean_pipeline(tile_names=game_tile_names)
train_x = pre.fit_transform(lean_batch)
model_x = pre.transform(lean_observation)
```

The pipeline one-hots:

- 49 tile IDs over the supplied tile vocabulary;
- player `state_id` over 12 states + unknown;
- three powerup slots over `-1` + four kinds;
- eight collision object types.

The numeric fields stay numeric. The model input width is vocabulary-dependent
and larger than 188.

## Lean layout

```text
49 primary tile IDs
49 rotations (radians)
49 signed phases
 9 player values
24 collision values
 8 flow deltas
= 188
```

Per-cell phase:

```text
+1.0   solid / present / fully interactable
+1..0  active transition (dissolving)
-1..0  inactive transition (disabled/repairing)
 0.0   transition boundary
```

The game-side primary symbol, rotation, and phase are the source of truth.
