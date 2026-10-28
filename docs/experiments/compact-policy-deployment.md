# Compact policy options for JSON deployment

Status: proposed; no compact student trained yet
Date: 2026-10-28
Related: [`../../experiments/0001_ppo_clean_start_1_1/README.md`](../../experiments/0001_ppo_clean_start_1_1/README.md)

## Question

How do we ship the clean-start PPO policy as a small Godot JSON model without
losing the 225k teacher's behavior?

The current 225k teacher is:

```text
lean 188 → Python one-hot 954 → 512 → 512 → 2
```

Canonical 1-1: **100/100 finishes, 2.3167 s**.

## Constraint

The existing game JSON runtime is a flat MLP:

```text
input → Linear → ReLU → Linear → ReLU → Linear → argmax
```

The deployed 58-value model is:

```text
58 → 256 → 256 → 2
81,410 parameters → about 1.7 MB JSON
```

The PPO checkpoint is 17 MB, but that includes optimizer state and the value
network. Only the policy is needed for deployment.

## Approximate policy-only sizes

JSON size estimates use the deployed model's roughly 21 bytes per parameter.
Re-export before treating these as exact.

| Architecture | Parameters | Estimated JSON |
|---|---:|---:|
| `954 → 512 → 512 → 2` current teacher | 752,642 | ~15.8 MB |
| `954 → 256 → 256 → 2` flat | 310,786 | ~6.5 MB |
| `954 → 192 → 192 → 2` flat | 220,802 | ~4.6 MB |
| `954 → 128 → 128 → 2` flat | 139,010 | ~2.9 MB |
| `embedding-8 → 256 → 256` | 223,098 | ~4.7 MB |
| `embedding-8 → 128 → 128` | 95,226 | ~2.0 MB |
| `embedding-4 → 128 → 128` | 70,078 | ~1.5 MB |

The flat models keep the existing JSON format. The embedding models replace
the 735 tile one-hots with a shared `15 × d` tile table:

```text
49 × 8 = 392 tile features
392 + 219 other features = 611 model inputs
```

That requires a JSON-schema and Godot runtime change, so embeddings should be a
second-stage option, not the first deployment attempt.

## Recommended order

### 1. Distill the 225k PPO teacher into a flat MLP

Do not shrink and retrain PPO from scratch first. Use the working teacher as
the label source.

Collect deterministic teacher action probabilities on:

- canonical starts;
- train start pool;
- holdout start pool;
- DAgger states where the student disagrees or visits different states.

Train with cross-entropy against teacher probabilities.

Candidate order:

1. `954 → 256 → 256 → 2` — likely best size/quality starting point.
2. `954 → 192 → 192 → 2` — middle ground.
3. `954 → 128 → 128 → 2` — aggressive; only if quality holds.

### 2. Export policy only

The runtime JSON should contain:

- first linear weights/bias;
- second linear weights/bias;
- action head weights/bias;
- input contract;
- activation;
- action repeat;
- source checkpoint metadata.

It should not contain the value network, optimizer, or PPO buffer data.

### 3. Round exported weights

Store 6–7 significant decimal digits, not full float precision. The current
JSON format stores decimal numbers, so rounding can meaningfully reduce file
size without changing inference behavior much.

### 4. Only then consider embeddings

If a flat 128×128 student is still too large, test:

- `embedding-8 → 256 → 256`;
- `embedding-8 → 128 → 128`;
- `embedding-4 → 128 → 128`.

The embedding table is tiny; the savings come from replacing the 735-value
tile block with `49 × d` values.

This requires runtime support for:

```text
tile IDs → lookup table → flatten → concatenate numeric features
```

### 5. Do not use binary tile IDs

Packing 15 categories into four bits creates false ordinal structure. A
learned embedding is the principled compact alternative.

## Evaluation gates

A compact student is not a candidate unless:

- canonical 1-1 is 100/100 finishes;
- canonical median is close to the 225k teacher's 2.3167 s;
- train-pool and holdout-pool finish rates stay at 100%;
- holdout median does not materially regress;
- JSON forward-pass parity against PyTorch passes on randomized observations;
- argmax agreement with the teacher is measured and reported;
- no model is installed until the full game gate passes.

The current holdout is route-cluster-level only. It is not state-unseen.

## Open questions

- Is the 225k teacher's deterministic policy stable enough to distill cleanly?
- Does a 128-wide flat student preserve 1-1 behavior, or is 192/256 needed?
- Does the game runtime need model-size limits or a maximum JSON load time?
- Should the exported JSON store floats rounded to 6 decimals or use a compact
  binary/base64 encoding with a runtime decoder?
