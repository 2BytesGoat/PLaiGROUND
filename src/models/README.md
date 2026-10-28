# `src/models/` — lean policy architecture

The lean v1 baseline is declared here:

```text
Godot lean 188
    → Python LeanFeatures one-hot/categorical encoder
    → LeanMLP
    → 512 → 512 → action/value
```

`src/models/lean_mlp.py` exposes:

- `LeanMLP(input_dim)` — basic 512×512 MLP;
- `build_lean_mlp(input_dim)` — constructor helper;
- `lean_policy_kwargs()` — SB3 `net_arch` config.

The input width is vocabulary-dependent because tile IDs, `state_id`, powerup
slots, and collision types are one-hot encoded by `preprocessing.lean`.

```python
from models import LeanMLP, lean_policy_kwargs

policy_kwargs = lean_policy_kwargs()
model = LeanMLP(input_dim=pre.output_size)
```

Trained artifacts stay under `runs/`, not here.
