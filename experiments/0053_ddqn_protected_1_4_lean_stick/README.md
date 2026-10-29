# 0053 — DDQN 1-4 with lean_stick 191

Status: running

Started: 2026-10-29

## Contract

```text
lean_stick 191 = lean 188
               + on_ceiling
               + jump_held
               + stick_active
```

The first 188 values remain byte-for-byte the lean observation. The Python
preprocessor one-hots the lean values to 954 model inputs and appends the three
raw flags, producing **957 model inputs**.

## Warm start

The run loads the selected 2.225M lean-188 checkpoint and expands the first
linear layer:

```text
old model input: 954
new model input: 957
```

The first 954 columns are copied from the old model; the three new columns are
zero-initialized, so the initial policy is exactly the old behavior when the new
flags are zero. The exploration schedule is reset.

The 200k anchor buffer is padded with zeros for the three new features. This is
correct for the old 1-1/1-2 transitions, which have no stick state.

## Training

```text
levels:       4×1-4 + 2×1-3 + 1×1-2 + 1×1-1
starts:       99 (11 × 1-4 + 88 old-level)
budget:       600k additional
checkpoints:  every 25k
anchor:       25%
```

## Target

```text
1-4 canonical: < 4.9833 s clean
1-4 train:     full coverage
1-4 holdout:   high coverage
1-1/1-2/1-3:  retain the 2.225M baseline
```

## 100k probe result

The branch was stopped at 100k for the decision point.

```text
1-4 canonical: 0/1, death
1-4 train:     2/11 at 6.7833 s
1-4 holdout:   0/2
spawns:        11/11 train, 2/2 holdout
```

Old-level retention at 100k:

```text
1-1: 1/1 1.6167; 12/12 3.3667; 8/8 3.9083
1-2: 1/1 4.3333; 31/31 5.0833; 20/20 4.8500
1-3: 1/1 6.5000; 41/45 6.6333; 19/23 6.3167
```

1-4 improved only slightly (1/11 -> 2/11) and the old levels regressed,
especially 1-3. The 100k checkpoint is not promotable. This result does not
justify retraining the whole backbone yet; the next decision is whether to
continue the 191 fine-tune with more old-level protection or freeze part of the
backbone.
