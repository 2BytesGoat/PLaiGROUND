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
Parallelism must divide the population. Promising untested combinations:

```text
parallel: 4, 6, 8, 12
speedup:  8, 12
```

10 processes is invalid for population 384. 12 processes oversubscribes the
10-core CPU; it may help or may throttle. This needs measurement.

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

## Next steps

1. Implement and run `tools/benchmark_neat.py` on this laptop.
2. Record the winning NEAT parallelism/speedup in `docs/status.md` and the
   experiment manifests.
3. Implement checkpoint rebalancing (`portfolio.py`, `tools/rebalance_neat.py`).
4. Build the 1-3 start pool and 1-4 seed portfolio from the 0022 archive.
5. Re-run the benchmark on the new PC before adopting a different default.
