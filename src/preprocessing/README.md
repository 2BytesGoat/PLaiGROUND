# `src/preprocessing/` — observation transforms

This package turns the game's raw observation into the flat vector consumed by
a learning model. It is deliberately separate from `src/processing/`, which
contains notebook-era tabular helpers.

## Contract

- Input: one raw observation `(342,)` or a batch `(N, 342)`.
- Decode: `RawStateBatch` with named tile/player/collision/flow arrays.
- Output: one float vector `(D,)` or `(N, D)`.
- `fit` follows the sklearn convention. Most steps are stateless; `Standardize`
  is fitted on training data.

## Usage

```python
from preprocessing import build_raw_state_pipeline

pre = build_raw_state_pipeline("features")
train_x = pre.fit_transform(train_raw_batch)
model_x = pre.transform(raw_observation)
```

The default `"features"` pipeline currently rebuilds the 342-value contract in
an explicit, testable order:

```text
tile IDs (49)
tile dynamic states (49 * 5)
player fields (16)
collisions (8 * 3)
flow (8)
= 342
```

`tile_one_hot=True` replaces the 49 tile IDs with one-hot tile IDs and therefore
grows the output. `tile_names` should come from the game's `static_info`.

## Chaining

```python
from preprocessing import (
    Clip, DecodeRawState, FeatureUnion, FlowFeatures,
    PlayerFeatures, StatePipeline, Standardize,
)

pre = StatePipeline([
    ("decode", DecodeRawState()),
    ("features", FeatureUnion([
        ("player", PlayerFeatures()),
        ("flow", FlowFeatures()),
    ])),
    ("standardize", Standardize()),
])
```

`StatePipeline` and `FeatureUnion` expose `fit`, `transform`,
`fit_transform`, `get_params`, `set_params`, named steps, pickle support, and
`sklearn.base.clone` compatibility. They do not use `sklearn.pipeline.Pipeline`
because intermediate values are structured rather than always 2-D arrays.

`build_raw_state_pipeline("raw")` is the safe identity preset. The `"lean"`
preset raises until the new feature layout is explicitly defined.
