# Results — 0008

Status: complete; no canonical speed gain, stable 600k retained  
Completed: 2026-10-28

## Outcome

The 100k time-trial from the 225k checkpoint reached **325,800 timesteps**. It
preserved deterministic performance but did not make the 1.6–1.75 s route the
policy default.

## Deterministic evaluation

| Model | Canonical | Train pool | Holdout pool |
|---|---:|---:|---:|
| 500k baseline | 1/1, 2.1500 s | 12/12, 3.3333 s | 8/8, 3.8583 s |
| 600k stable selected | 1/1, 2.1500 s | 12/12, 3.3250 s | 8/8, 3.8583 s |
| 325k time-trial | 1/1, 2.1500 s | **12/12, 3.2917 s** | 8/8, 3.8583 s |

All 325k pool runs finished and all spawns were applied.

The 325k model has the best train median by a small margin, but no canonical
speed improvement. It is not a replacement for the stable 600k model unless we
want that small train-pool difference.

## Training monitor

Canonical-only training, 100k continuation:

```text
episodes:          3,035
median:            2.150 s
best:              1.633 s
≤2.2 s:            69.8%
timeouts:          0.3%
episodes <1.9 s:   1
episodes <1.75 s:  1
```

Reward bands:

```text
1.50–1.75 s:  n=1, mean reward 575.9
2.13–2.16 s:  n=2,081, mean reward 577.7
2.16–3.00 s:  n=858, mean reward 577.1
```

## Why the flat bonus did not work

The new reward was:

```text
time:             5.0/s
finish:           500 + 50 if run_time < gold
death:            -100
timeout:          -100
```

At gold ≈ 2.15, both **1.63 s and 2.13 s are under gold**, so both receive the
same flat `+50`. The 2.13 route still gets more progress/efficiency reward and
remains the higher-return path. The fast route was sampled once and was not
reinforced into the deterministic policy.

## Decision

Keep the **600k stable model** selected. The 325k time-trial is a useful
negative result: the terminal threshold bonus removed the “under-gold is
unrewarded” problem but did not create a gradient between two under-gold times.

A follow-up should use a continuous under-gold term, for example:

```text
finish += K * max(0, gold_time - run_time)
```

with `K` around 20–50, or reduce the progress/efficiency dominance. The flat
+50 should not be repeated as the only speed change.
