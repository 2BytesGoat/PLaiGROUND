# 0068 — fresh lean_stick 191, 1-4 probe

Status: running  
Started: 2026-10-29

## Question

Can the selected 1.53M 1-1/1-2/1-3 policy make progress on 1-4 with the agreed
`4×current + 1×previous + 3 retention` schedule, without regressing the old
levels?

## Setup

```text
source:            0067 1.53M checkpoint
observation:       lean_stick 191 -> 957
levels:            4 x 1-4 + 1 x 1-3 + 2 x 1-2 + 1 x 1-1
starts:            99 tagged starts (12 x 1-1 + 31 x 1-2 + 45 x 1-3 + 11 x 1-4)
additional steps:  100,000 (decision-point probe)
reset timesteps:   false
protected replay:  400k online + 200k 1-1/1-2/1-3 anchor, 25% anchor
```

The 1-4 start pool is the validated 0046 artifact:

```text
runs/neat_seeded_1_4_stick_full_population/
  382/384 evaluated, best 4.9833 s, 75 clean routes
  11 train states / 2 holdout states
```

The previous 0053 100k 1-4 probe was negative:

```text
1-4 canonical: 0/1 (death)
1-4 train:     2/11 at 6.7833 s
1-4 holdout:   0/2
old levels:    1-3 regressed to 41/45 and 19/23
```

This probe uses the stronger fresh 1.53M base and the agreed retention split.

## Probe gate

Do not continue past 100k unless 1-4 shows real progress:

```text
1-4 canonical:  finish (not death/timeout)
1-4 train:      >= 2/11, preferably more
1-4 holdout:    >= 1/2
1-3:            full 45/45 and 23/23
1-2:            31/31 and 20/20
1-1:            12/12 and 8/8
```

The 1-4 canonical and train/holdout gates are evaluated with 0069; old-level
retention is 0070 (1-3), 0071 (1-2), and 0072 (1-1).

## Output

```text
runs/ddqn_fresh_1_4_lean_stick/
```

## Result

Stopped at the 100k decision point. The probe is negative at every checkpoint:

```text
1-4 canonical: 0/1 (timeout; 1.63M also death)
1-4 train:     1/11 at 6.7833 s
1-4 holdout:   0/2
```

Training monitor over the 100k probe:

```text
1-4 canonical: 0/310 finishes
1-4 state:     13/136 finishes, median 6.7833, best 6.1833
1-2 canonical: 3.1667 (4 timeouts)
1-3 canonical: 3.3167 (2 no-finish)
1-1 canonical: 1.5667
```

This reproduces the 0053 failure pattern: the 1-3 policy does not transfer to
1-4, and the 1-4 training starts to cost old-level coverage.

Deterministic old-level retention at the final 1.63M checkpoint:

```text
1-1: 1.5667 canonical; 12/12, 3.2833; 8/8, 3.8000
1-2: 3.1833 canonical; 31/31, 4.5833; 20/20, 3.9083
1-3: 3.2333 canonical; 45/45, 5.4333; 23/23, 5.4333
```

Coverage is intact, but 1-2/1-3 canonical are slower than the 1.53M baseline.
The 0046 NEAT population is strong on 1-4, so the next attempt should use a
different method (NEAT-route distillation, a 1-4-specific curriculum/pool, or a
separate 1-4 head) rather than a longer DDQN continuation.
