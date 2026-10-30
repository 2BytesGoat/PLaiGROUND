# 0130 — shipping candidate (Python-side)

Status: **16/16 canonical finish; 12/16 strict**  
Selected: 2026-10-30  
Campaign: [`README.md`](README.md)  
Checkpoint: `runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/student.pt`
Stable copy: `runs/distill_campaign_student/students/selected_shipping/student.pt`

This is the first shared student in the campaign that finishes every canonical
level. It is a Python-side `lean_stick` model, not yet a drop-in replacement for
the game's bundled legacy/multi-hot JSON policy.

## Model

```text
input:        lean_stick 191 -> fitted 957 one-hot
architecture: 957 -> 512 -> 512 -> 2, ReLU
data:         dataset_verified_false_ddqn (29,081 transitions, 216 routes)
level weight: 1-11 = 2
seed:         2
best epoch:   171
checkpoint:   student.pt (2.9 MB)
sha256:       f562929ab0cd3294c23ce91a6599f60301919679da8e2d78e1d6efd00333f4d2
```

Accuracy:

```text
train       98.19%
validation  95.13%
holdout     97.67%
```

## Canonical gate

All times are deterministic canonical single-agent runs. 1-14 uses
`reassert_held_jump=false`; the other levels use true held-jump semantics.

| Level | Target | Result | Strict |
|---|---:|---:|---|
| 1-1 | 1.5667 | 1.5667 | pass |
| 1-2 | 3.1500 | 3.1500 | pass |
| 1-3 | 3.2667 | 5.2333 | **miss +1.9666** |
| 1-4 | 4.4667 | 4.4667 | pass |
| 1-5 | 3.1667 | 3.1667 | pass |
| 1-6 | 4.6000 | 4.6000 | pass |
| 1-7 | 5.2333 | 5.2333 | pass |
| 1-8 | 4.4667 | 4.5000 | **miss +0.0333** |
| 1-9 | 2.9500 | 2.9500 | pass |
| 1-10 | 5.3667 | 5.3667 | pass |
| 1-11 | 9.8167 | 9.8167 | pass |
| 1-12 | 5.9667 | 6.0167 | **miss +0.0500** |
| 1-13 | 6.2667 | 6.2667 | pass |
| 1-14 | 6.9333 | 8.1833 | **miss +1.2500** |
| 1-15 | 5.8500 | 5.8500 | pass |
| 1-16 | 6.1333 | 6.1167 | pass |

Summary:

```text
finish:    16 / 16
strict:    12 / 16
sum:       82.4833 s
target sum: 79.2002 s
```

The 1-8 and 1-12 misses are one action-repeat frame. The material timing
improvements are 1-3 and 1-14.

## Reproduction

```bash
.venv/bin/python scripts/supervised/03_distill_student.py \
  --data-dir runs/distill_campaign_student/dataset_verified_false_ddqn \
  --output-dir runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2 \
  --input-kind onehot957 --hidden 512 512 \
  --seed 2 --level-weight 1-11=2 --device mps
```

Evaluate canonical 1-14 under the required semantics:

```bash
.venv/bin/python tools/evaluate_student.py \
  --checkpoint runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/student.pt \
  --level 1-14 --canonical --max-decisions 180 \
  --no-reassert-held-jump \
  --output runs/distill_campaign_student/eval/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/1-14_false_canonical.json
```

Use the same command without `--no-reassert-held-jump` for the other levels.

## Deployment status

The game's `MLPPolicy` runtime currently accepts only:

- `legacy` 58-value observations, or `multi_hot` semantic grid observations;
- two dense ReLU hidden layers with JSON `w1/b1`, `w2/b2`, `w3/b3` weights.

It does **not** accept `lean_stick` 191 or the fitted 957 one-hot input. The
bundled `student_obs58_256.json` remains the safe production model (16/16,
78.583 s sum). This candidate is staged for the next runtime step, not installed.

To ship this candidate in-game, choose one:

1. add a `lean_stick` + one-hot spec to `MLPPolicy` and export the transform
   parameters with the weights;
2. distill this model's canonical/recovery actions into a `multi_hot` or legacy
   58 student that the existing JSON runtime already accepts.

## Deferred improvement

The 7x7 grid-encoder diagnostic is negative and deferred:

```text
holdout:      96.79% (flat candidate: 96.89%)
1-9:          death
1-10:         timeout
1-11:         timeout
1-14:         finish at 11.9833 s (slower than flat 8.1833 s)
```

The encoder did fix strict 1-3 timing (3.2667 s), but losing 1-10 and 1-11 is
not an acceptable trade. Revisit only if the flat canonical model plateaus, or
if strict 1-3 timing becomes the primary objective. It also requires a new
game-side convolution runtime.
