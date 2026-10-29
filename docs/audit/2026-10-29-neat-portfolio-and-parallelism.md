# NEAT portfolio and parallelism plan — 2026-10-29

## 1-3 discovery result

The 384-genome exploratory run (experiment 0021) and its 40-generation
continuation (0022) found substantially more 1-3 behavior than the zero-finisher
192-genome transfer:

```text
generations:         70 total (0021: 30, 0022: 40)
final population:    384
final finishes:      220 / 384  (~57%)
best fitness:        582.49
fastest route:       4.2333 s  (unclean)
unique routes:       304
finisher records:    6,361
checkpoint files:    6,361
strict clean routes: 53
clean best:          5.4667 s
clean median:        5.9833 s
route clusters:      13 at Jaccard 0.5
                     26 at Jaccard 0.7
```

This is enough data for a 1-3 start pool and a 1-4 seed portfolio.

## Decision: do not force a Gaussian on the live population

Keep NEAT as the discovery algorithm. The deployment portfolio is a separate
curation layer.

```text
Discovery population
  large, high mutation
  non-finishers allowed
  fitness + novelty / route diversity
  raw archive preserved

Deployment portfolio
  finishers only
  best genome per route cluster × finish-time bin
```

Suggested deployment shape:

```text
10–15%  speedrunners       < 4.5 s
60–70%  median finishers   4.5–6.0 s, different route clusters
10–15%  slow/robust        6.0–10.0 s, only if behaviorally distinct
5–10%   exploration        excluded from the final seed
```

“All must finish” is a final portfolio filter, not an early training constraint.

## Checkpoint rebalancing, not a live watcher

A live reporter can observe the population but cannot reliably stop NEAT from
culling a protected genome during reproduction. The first implementation should
therefore rebalance at checkpoints:

```text
run NEAT for K generations
analyze the checkpoint
select bin/cluster representatives
rebuild the population with protected genomes + mutations/random fill
resume from the rebalanced checkpoint
```

Target modules:

```text
src/plaiground/algorithms/genetic/neat/portfolio.py
  PortfolioWatcher
  classify_genome()
  select_portfolio()
  rebalance_checkpoint()

tools/rebalance_neat.py
```

A custom `PortfolioReproduction` is a later option only if checkpoint
rebalancing proves too coarse.

## Hardware and parallelism

Current machine:

```text
Mac17,3 (MacBook Air class)
Apple M5
10 physical/logical cores
32 GB RAM
```

Current NEAT configuration uses 8 processes × 48 agents (384 population).
Parallelism must divide the population; 10 is invalid for 384.

Battery benchmark, 2 generations per combo:

```text
parallel  speedup  gen mean  peak RAM
4         8        53.59 s   2.33 GB
4         12       57.72 s   2.34 GB
6         8        48.94 s   2.77 GB
6         12       50.43 s   2.85 GB
8         8        46.59 s   3.32 GB
8         12       42.11 s   3.34 GB
12        8        59.44 s   3.33 GB
12        12       55.81 s   2.84 GB
```

AC-power rerun, 3 generations, Low Power Mode off:

```text
8 × 12:   28.81 s/generation, 2.64 GB peak
12 × 12:  40.53 s/generation, 3.54 GB peak
```

Recommendation for this laptop: **8 processes × speedup 12 on AC**. AC power
gave a large improvement over the battery measurement; 12 processes still
oversubscribes the 10-core CPU. Re-run on the future PC before adopting a new
default.

## Benchmark harness

Add a reusable script:

```text
tools/benchmark_neat.py
```

Responsibilities:

```text
detect OS/CPU/RAM/free disk
run short NEAT generations for each parallel × speedup combination
skip route recording during throughput tests
record wall time, generation time, finish rate, CPU/memory samples
write JSON + Markdown + logs under runs/benchmarks/<date>_<host>/
recommend the best stable configuration
```

The same script should be reusable on the future PC.

## Implementation status

Added:

```text
tools/benchmark_neat.py
src/plaiground/algorithms/genetic/neat/portfolio.py
tools/rebalance_neat.py
tests/test_portfolio.py
```

The first rebalancer run used checkpoint 66 and the full 0022 archive:

```text
checkpoint genomes:  384
archive genomes:     294
candidates:          678
finishers:           213
clusters:            46
selected:            384
  speed:             3
  median:            56
  slow:              154
  exploration:       171
deployment pool:     213 finishers
```

The source does not contain enough fast/median finishers to fill the target
quota, so the rebalancer preserves all finishers and fills the remaining
discovery slots with exploration genomes. The rebalanced checkpoint loads as a
valid 384-genome population.

The first rebalanced checkpoint inherited a low innovation counter and produced
one innovation-collision warning during the continuation. The rebalancer now
advances the tracker to the highest archived connection innovation and writes
`runs/neat_explore_1_3_long/rebalanced_fixed/rebalanced-66`; use that as the
seed going forward.

## Validation: 0023 ten-generation continuation

The rebalanced population ran for 10 generations on 1-3 with 8×12 on AC. It
finished in **257.8 s**:

```text
final finishes:   238 / 384  (62.0%)   source final: 220 / 384 (57.3%)
best time:        4.2333 s
best clean:       5.4667 s
archive routes:   364                    source: 304
new routes:       154
clean routes:     60                     source: 53
new clean routes: 25
clusters @ 0.5:   54                     source: 46
clusters @ 0.7:   137                    source: 118
```

Archive bins:

```text
speed: 3 / median: 65 / slow: 187 / exploration: 109
```

This is a successful **discovery** result: the archive became larger and more
diverse. But the live final population did not preserve the deployment shape:

```text
source final (70):  5 speed / 4 median / 207 slow / 168 exploration
0023 final (76):    0 speed / 7 median / 224 slow / 153 exploration
```

NEAT selection culled the speed tail and moved the live population toward slow
finishers. One rebalance at checkpoint 66 is therefore insufficient for a
deployment portfolio. The next run should rebalance every 2–3 generations or
protect the distribution during reproduction.

## Validation: 0024 frequent rebalancing

Experiment 0024 merged the 0022/0023 archives (458 routes/genomes), started
from the fixed rebalanced checkpoint 66, and rebalanced the live population
every 2 generations. The successful resumed segment ran generations 69–76 in
**230.0 s**.

```text
final evaluated generation:  384 / 384 finishes (100%)
best time:                   4.2333 s
archive best clean:          4.7667 s
archive routes:              655
clean routes:                105
clusters @ 0.5:              60
clusters @ 0.7:              162
final evaluated bins:        3 speed / 78 median / 227 slow / 76 exploration
final protected population:  3 speed / 78 median / 230 slow / 73 exploration
deployment genomes:          311
```

Comparison with one-shot 0023:

```text
                    one-shot 0023    frequent 0024
final finishes      238 / 384        384 / 384
speed                0                3
median               7                78
slow                224               227
exploration         153               76
archive routes      364               655
clean routes         60               105
clusters @ 0.5       46                60
archive best clean   5.4667 s          4.7667 s
```

The hypothesis is supported: frequent live rebalancing prevents the live
speed/median collapse and expands the discovery archive. The final checkpoint
`runs/neat_frequent_rebalance_1_3/neat-checkpoint-76` is the 1-4 seed. A separate
evaluation of that checkpoint finished **384/384**, with 3 speed / 78 median /
228 slow / 75 >10 s, so the seed is validated. The rebound is visible in the
alternating finish rate: a freshly rebalanced generation finishes 384/384,
then selection drops it to roughly half, and the next rebalance restores the
portfolio.

## Checkpointing and population-size issues

The interval checkpointer saved generation 72 but missed the final generation
76. The trainer now always saves the final checkpoint. A separate 4-generation
replay from checkpoint 72 produced a valid checkpoint 76 but a different
trajectory (219 finishes, 1 speed / 9 median / 205 slow / 169 exploration); it
is a diagnostic, not the 0023 final.

NEAT also reproduced 383 and 381 genomes in 0024, which cannot split across 8
processes. The trainer now repairs the population to `config.pop_size` after
restore and after every generation, then re-speciates. The successful 0024 run
used the repaired population.

The 1-3 start pool was built from the strict clean routes:

```text
clean routes:  53
clusters:      13
filtered:      46 (35 train / 11 holdout)
pool:          44 train entries (43 state + canonical) / 26 holdout states
```

## Next steps

1. ~~Finish the NEAT parallelism benchmark and record the winning default.~~
   Complete: 8×12 on AC is the laptop default.
2. ~~Rebalance every 2–3 generations and verify that the live speed/median bins
   survive.~~ Complete: 0024 preserved 3 speed / 78 median and produced 311
   deployable genomes.
3. Use `runs/neat_frequent_rebalance_1_3/neat-checkpoint-76` as the 1-4 seed.
4. Build the DQN 4+2+2 1-3 training config using the new 1-3 pool.
5. Re-run the benchmark on the new PC before adopting a different default.
