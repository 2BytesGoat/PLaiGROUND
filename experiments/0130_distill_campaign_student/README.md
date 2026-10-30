# 0130 — campaign distillation data capture

Status: **shipping candidate selected; 16/16 canonical finish**  
Date: 2026-10-30  
Manifest: [`config.toml`](config.toml)  
Shipping card: [`SHIPPING_CANDIDATE.md`](SHIPPING_CANDIDATE.md)  
Local report: `runs/distill_campaign_student/DATA_REPORT.md`

> **Selected candidate:** `cand_verified_onehot957_512_false_ddqn2_w11x2_s2`
> (one-hot 957, 512×512, seed 2). It finishes **16/16** canonical levels and
> passes **12/16** strict times. The game runtime still needs a `lean_stick`
> one-hot spec or a `multi_hot`/legacy distillation pass; the bundled 58-value
> JSON model remains the production fallback.

## What was captured

- Clean NEAT routes for 1-1 → 1-10, including path cells/facings, the shortest
  successful action sequence, and the selected expert's finish time from every
  deduplicated route start.
- Selected-expert rollouts for 1-1 → 1-10 in `lean_stick` 191 with the fitted
  957 one-hot input and the teacher action.
- Canonical teacher paths for 1-11 → 1-16. The archived 58-input PPO teachers
  were used where they finished; the deployed 58-input student was the fallback.
- Paired 191/957 replays for those 58-input paths.
- One DAgger round for 1-11, 1-12, and 1-14.

## Canonical times at capture

| Level | Teacher | Canonical |
|---|---|---:|
| 1-1 | `0073` chain | 1.5667 |
| 1-2 | `0073` chain | 3.1500 |
| 1-3 | `0073` chain | 3.2667 |
| 1-4 | `0073` chain | 4.4667 |
| 1-5 | `0085` final | 3.1667 |
| 1-6 | `0099` 1.815M | 4.6000 |
| 1-7 | `0103` 1.877432M | 5.2333 |
| 1-8 | `0118` 1.960392M | 4.4667 |
| 1-9 | `0122` 2.060360M | 2.9500 |
| 1-10 | `0126` 2.122840M | 5.3667 |
| 1-11 | PPO teacher, true semantics | 9.8167 |
| 1-12 | PPO teacher, true semantics | 5.9667 |
| 1-13 | PPO teacher, true semantics | 6.2667 |
| 1-14 | deployed student, legacy semantics | 6.9333 |
| 1-15 | PPO teacher, true semantics | 5.8500 |
| 1-16 | PPO teacher, true semantics | 6.1333 |

1-11 is the only level where the true-semantics 191 path is materially slower
than the legacy 58 path: **9.8167 s** versus **8.2500 s**. 1-14 has no 58 teacher
that finishes with true held-jump semantics, so its recorded path is the
deployed model under legacy semantics.

## Dataset

Candidate index:

```text
runs/distill_campaign_student/dataset_final2/index.json
```

It contains the expert rollouts, the true-semantics late-level paths, the 1-14
legacy path, and the DAgger corrections. Expert holdout routes remain marked
`holdout`.

## Holdout

- 1-1 → 1-10: existing route-cluster holdout. 1-5 has no mid-route holdout.
- 1-11/1-14/1-15: the slower alternative path is retained in
  `dataset`/`dataset_true` as a path-level holdout.
- 1-12/1-13/1-16: archived PPO and deployed canonical paths are identical, so
  there is no independent path holdout. Use one whole level as an unseen-level
  diagnostic if needed.

## Preliminary result

`students/final_raw191_wide` is raw 191 → 1024 → 512 → 2 with level-balanced
loss. It finishes **13/16** canonical levels:

```text
passes 1-1 .. 1-10 at the selected-expert times
passes 1-13, 1-15, 1-16
fails 1-11 (timeout), 1-12 (death), 1-14 (timeout)
```

The late-level data volume is still the bottleneck. Another DAgger round from
this candidate is the next step if training does not close 1-11/1-12/1-14.

## Training handoff

```bash
.venv/bin/python scripts/supervised/03_distill_student.py \
  --data-dir runs/distill_campaign_student/dataset_final2 \
  --output-dir runs/distill_campaign_student/students/candidate_raw191_wide \
  --input-kind raw191-wide --hidden 1024 512 --epochs 350 --seed 0
```

Evaluate with `tools/evaluate_student.py`; see the local report for the full
command block.
