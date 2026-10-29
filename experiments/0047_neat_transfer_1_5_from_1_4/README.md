# 0047 — NEAT 1-4 → 1-5 transfer

Status: complete; diagnostic, not the active training path

Started: 2026-10-29

Completed: 2026-10-29

## Protocol

```text
source population: runs/neat_seeded_1_4_stick_full_population/neat-checkpoint-30
source level:      1-4
target level:      1-5
observation:       legacy_stick (61)
action:            reassert_held_jump
zero-shot:         384 genomes, record paths
seeded phase:      2 route genomes, high mutation, 384 population
```

1-4 route states and archives were not merged into 1-5.

## Results

Zero-shot transfer:

```text
finishes:     6/384
best time:    6.05 s
routes:       2
clean routes: 0
```

The seeded high-mutation phase then ran 20 generations:

```text
generation 20: 384/384 finishes, 0 deaths
best clean:    4.6833 s
clean routes:  9
```

The 1-5 result is strong enough to keep as a diagnostic and future seed, but it
was not pursued into a DDQN stage because the priority is the 1-4 DDQN gate.

## Output

```text
runs/neat_transfer_1_5_from_1_4/
  zero_shot/
  archive/
  seed-checkpoint-0
  neat-checkpoint-20
  train.console.log
```
