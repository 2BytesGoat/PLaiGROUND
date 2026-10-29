# 0046 — Full finisher-preserving 1-4 population

Status: complete

Started: 2026-10-29

Completed: 2026-10-29

## Question

Can the 0038 seed produce a genuinely finisher-rich 1-4 population for the
1-5 transfer seed, without the rebalancing losing slow finishers?

## What changed

0044 and 0045 exposed two rebalancing bugs:

- `Population.run` evaluates the parents and then reproduces, so evaluated
  finishers that were not retained as parents disappear before rebalancing.
- `time_bin()` labels >10 s routes as `exploration`; the first live-finisher
  selection therefore dropped 50 of 173 finishers.

0046 fixes both:

```text
finisher_genomes:  evaluated finishers are copied back after reproduction
is_finisher:       finisher status is separate from the time bin
preserve_finishers: every finisher is selected before discovery genomes
```

The 61-value `legacy_stick` contract and `reassert_held_jump` are unchanged.

## Run

```text
seed:              runs/neat_seeded_1_4_stick/seed-checkpoint-0
level:             1-4
population:        384
parallel:          8
generations:       30
action repeat:     4
speedup:           12
rebalance:         every 2 generations + final
checkpoint:        every generation
clean target:      none (population quality is the gate)
```

## Training result

```text
final generation:  380/384 finishes, 2 deaths
peak:              383/384 at generations 12 and 26
best finish time:  4.9833 s
clean routes:      75
archive routes:    498
```

Generation 30 rebalance:

```text
source finishers:  751
selected finishers: 384
deployment:        384
```

The final population is composed of finisher genomes; the 17.6 s high-fitness
but non-clean route was not used as a seed.

## Independent 1-4 evaluation

Checkpoint 30 evaluated as 384 genomes on canonical 1-4:

```text
finishes:   382/384
deaths:     2
best time:  4.9833 s
```

That clears the finisher-rate seed gate. The best clean route remains
**4.9833 s**.

## Next

Use checkpoint 30 as the 1-4 source population for the 1-5 transfer probe. Do
not merge 1-4 route states or archives into 1-5; only the genome population is
transferred.
