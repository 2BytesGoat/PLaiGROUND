# 0020 — NEAT 1-2 → 1-3 route collection

Status: running  
Started: 2026-10-29

## Question

Can the evolved 1-2 NEAT population produce clean 1-3 routes and a usable start
pool?

## Setup

```text
source level:      1-2
source checkpoint: runs/neat_clean_450_1_2_transfer/neat-checkpoint-31
target level:      1-3
population:        192
parallel:          8
observation:       legacy 58
action repeat:     4
speedup:           8
max steps:         300
target:            25 clean route signatures
budget:            20 generations (32–51)
```

The 1-1 and 1-2 populations both produced **0 finishes** on 1-3 in zero-shot
evaluation. This run evolves the closer 1-2 population on 1-3 instead of keeping
it frozen or starting a fresh population.

## Output

```text
runs/neat_clean_450_1_3_from_1_2/
  archive/routes.json
  archive/finishers.jsonl
  archive/checkpoints/
  archive/genomes/
  best_clean_genome.pkl
  best_time_genome.pkl
```

## Result

Completed 20 generations in **532.8 s** with no finishers:

```text
best fitness:       74.67 (plateaued from generation 32)
mean fitness:       ~72–73
finishes:           0 / 192
deaths:             0 / 192
clean routes:       0
checkpoint files:   0
```

The best genome ended around `y≈424`; the exit is at `y=10` and the player
starts around `y=416`. The population never reached the exit, so a longer
episode horizon would not help. This is a wall-jump capability gap, not a
timeout near the finish.

Follow-up: experiment 0021 is a fresh 384-genome, higher-mutation exploratory
search on 1-3.
