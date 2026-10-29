# The Utility of the B-Team

Status: idea
Updated: 2026-10-29

## Hook

> "From a utilitarian standpoint, we need diversity to keep evolving."

A population that selects only its current geniuses looks rational. It is
maximizing the score in front of it. But it is also throwing away the variants
that might solve the next level, recover from a perturbation, or discover a
different route family.

The best genome is not the best population.

## Thesis

Diversity is not the opposite of utility. It is how utility survives
uncertainty.

If "utility" means this generation's fitness, the strongest genome wins and
everything else is waste. If "utility" means expected capability over future
levels, perturbations, and recovery cases, diversity has **option value**. It is
capital we are choosing not to spend today.

The useful split:

```text
deployment:  the best current finishers and routes
discovery:   a population broad enough to keep finding new basins

geniuses win deployment
diversity wins discovery
```

## Evidence from the Dragon Jump experiments

### 1. One-shot rebalancing preserved the archive, not the live population

On 1-3, a single rebalance kept good routes in the archive but the live
population lost its speed tail:

```text
one-shot rebalance final:
  0 speed / 7 median / 224 slow / 153 exploration
```

Frequent rebalancing, every two generations, kept the portfolio in the breeding
population:

```text
frequent rebalance final:
  3 speed / 78 median / 227 slow / 76 exploration
  384/384 finishes
  655 routes / 105 clean
```

The archive was not enough. The **selection mechanism** had to keep diversity
alive in the population that actually reproduced.

### 2. A rough finisher is valuable even when it is not clean

On 1-4, the 1-3 transfer produced only a few rough finishers:

```text
transfer evolution:  6/384 finishes, best 10.5333 s, 4 routes, 0 clean
fresh high mutation: 0/384, 0 routes, 0 clean
```

Seeding a high-mutation population from those rough finishers immediately found
a much better route:

```text
seeded generation 1: 40/384 finishes, best 5.9167 s
```

The “less good” genomes were not noise. They were the bridge to a new behavior.
Without them, the fresh population never reached the exit.

### 3. High exploration without selection is not search

A high-epsilon DDQN with learning disabled is a fixed random-ish policy. It has
no mechanism to keep a promising partial route or compound progress across
episodes. It is random sampling, not discovery.

```text
method                         selection pressure     route discovery
high-epsilon DDQN, no learning  none                   effectively random
NEAT                            fitness + selection     good
DDQN with learning              gradients               possible, expensive
```

Turning off learning removes the only thing that makes the search improve.

## The utilitarian argument

A strict act-utilitarian selection rule — “always breed the best genome” — is
locally correct and globally myopic. It treats fitness as the territory, when it
is only a map. The “genius” is usually the best exploiter of the current basin,
not the thing most likely to find the next basin.

The portfolio view is still utilitarian; it just uses a longer time horizon:

```text
expected future capability =
    exploit(current best)
  + option value(preserved diversity)
```

Diversity has costs: compute, diluted selection, slower convergence. This is not
an argument to keep every weak genome. It is an argument to protect **meaningful
behavioral diversity**: route clusters, time bins, recovery behaviors, and
different route families.

## What NEAT should protect

```text
route/time portfolio
  speed     -> current best
  median    -> robust finishers
  slow      -> alternative mechanics / recovery
  exploration -> unfinished but structurally different

speciation
  protects innovation long enough to be evaluated

archive
  remembers history

live rebalancing
  keeps the archive in the breeding population
```

The last one is the part that is easy to miss. An archive is a museum; a
population is a search.

## Possible outline

1. Open with the quote: from a utilitarian standpoint, we need diversity.
2. Define the two utilities: current score vs future capability.
3. Explain why fitness-only selection overfits the current level and seed.
4. Show the 1-3 one-shot vs frequent-rebalance result.
5. Show the 1-4 rough-finisher → seeded population result.
6. Explain why no-learning random exploration is not a substitute for search.
7. Distinguish discovery population from deployment portfolio.
8. End with the rule: keep the B-team in the breeding population, not just in
   the archive.

## Caveats

- “Utilitarian” here means long-run expected capability, not fairness or
  intrinsic value. A short-horizon score maximizer would rationally discard
  diversity.
- Diversity is not automatically useful; genetic noise is cheap, behavioral
  diversity is the target.
- The 1-4 seeded run is now validated: 0038 found 10 clean routes in one
  generation after the hold/ceiling contract fix. The earlier 0037 run found a
  5.9167 s route but no clean routes because the action/observation contract and
  rebalancing were incomplete; this is a contract lesson, not only a diversity
  lesson.
- The 1-3 numbers are from the current committed experiment records; the
  portfolio quotas are a design choice, not a proven universal optimum.

## Working titles

- *The Utility of the B-Team*
- *Keep the Losers*
- *Diversity Is a Utility Function, Not a Vibe*
- *From Leaderboard to Portfolio*
- *Why the Best Genome Is Not the Best Population*
- *The Genius Gets You Today; the B-Team Gets You Tomorrow*
