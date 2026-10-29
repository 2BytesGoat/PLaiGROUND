# Level-transfer playbook — 2026-10-29

This is the repeatable NEAT path for adding level N after N-1. It prevents the
"try a random idea" loop and makes each stage produce a go/no-go artifact.

## Pipeline

```text
1. TRANSFER PROBE
   resume the protected N-1 population on N
   record paths, no evolution
   report finishers, best time, unique routes, clean routes

2. IF CLEAN ROUTES >= TARGET
   filter routes -> build start pool -> train policy

3. IF FINISHERS > 0 BUT CLEAN ROUTES < TARGET
   seed a high-mutation population from the best archived route genomes
   (tools/seed_neat_checkpoint.py)
   run 30 discovery generations with record_paths
   extend if finishers exist but clean count is still low

4. IF FINISHERS == 0
   run a fresh high-mutation discovery population
   if it still has zero finishers, change the representation/reward/curriculum
   before spending more generations on the same search

5. PROTECT THE ARCHIVE
   once finishers exist, rebalance every 2 generations
   preserve route/time portfolio; never return to one-shot rebalancing

6. BUILD THE POOL
   filter clean routes at Jaccard 0.5, cap 8 per cluster, 25% holdout
   build train/holdout starts
   only then start multi-level DQN/DDQN training
```

## Seed tool

`tools/seed_neat_checkpoint.py` creates a fresh population from archived route
genomes:

- one intact elite per source route;
- high-mutation copies for the rest;
- innovation tracker advanced past all source innovations;
- no level identity or level-conditioned input.

Example:

```bash
.venv/bin/python tools/seed_neat_checkpoint.py \
  --config experiments/<new>/neat_config.txt \
  --genome runs/<source>/archive/genomes/<route>.pkl \
  --genome runs/<source>/archive/genomes/<route2>.pkl \
  --population 384 \
  --output-prefix runs/<new>/seed-checkpoint-
```

## Go/no-go rules

```text
probe finishers == 0
  -> fresh high mutation; if still zero, change representation/reward/curriculum

probe finishers > 0, clean == 0
  -> seed high mutation from the best finishers

seeded search finishers > 0, clean == 0
  -> extend the seeded search; do not start another random population

any search clean >= target
  -> filter, build pool, train
```

## Why not a no-learning, high-exploration DDQN?

A high-epsilon DDQN with learning disabled is a **fixed random-ish policy**.
It has no selection pressure: it cannot keep a promising partial route, improve
its behavior, or compound progress across episodes. It is random sampling, not
search.

```text
method                         uses reward to improve?   route discovery
high-epsilon DDQN, no learning  no                        effectively random
NEAT                            yes, via population       good
DDQN with learning              yes, via gradients        possible, expensive
```

On a long wall-jump level such as 1-4, the probability of randomly producing the
required sequence is tiny. Turning off learning removes the only mechanism that
makes the search improve.

A **learning-enabled** high-exploration phase with random standable starts is a
different possibility: it is online route mining. It is worth testing as a
DDQN discovery channel, but it is not equivalent to the no-learning idea.

The practical split remains:

```text
route discovery:  NEAT or learning-enabled exploration
route retention:  protected DDQN replay
route training:   protected DDQN
```

## Action/observation contract gate

Before blaming mutation or fitness, verify that the new level's mechanics are
representable by the training contract:

- Does the mechanic require a **held** button, not just an edge press?
- Does the observation expose the relevant contact/state?
- Does the action adapter preserve hold across state transitions?

1-4 exposed both sides of this. The level is **STICK**, whose ceiling-stick
mechanic needs `wants_to_jump` held while touching a ceiling. The legacy
58-value observation has `on_wall` but **no `on_ceiling`**. The training
controller also deduplicates jump input and only calls `JumpCommand` when the
bit changes; `JumpState.exit()` clears `wants_to_jump`, so a held `1` is not
reasserted. The heuristic controller deliberately reasserts the held input every
frame.

This is a **contract mismatch**, not a mutation problem. Fix the action hold
semantics (and add ceiling state in a new encoding if needed) before more
search. Existing 58-input models must not silently change meaning.

The fix is now versioned:

```text
legacy_stick        legacy 58 + on_ceiling + jump_held + stick_active (61)
reassert_held_jump  opt-in action mode; old 58-input runs keep edge semantics
```

The game-side controller reads both flags only for new runs. Existing agents
are not re-encoded or retrained.

## Current 1-4 evidence

```text
0035 transfer probe:       1/384 zero-shot at 10.5333 s
0035 transfer evolution:   6/384, best 10.5333 s, 4 routes, 0 clean
0036 fresh high mutation:  0/384, 0 routes, 0 clean
0037 seeded high mutation: stopped at generation 25; archive 5.9167 s, 0 clean
0038 hold-aware seeded:    gen 1: 173/384 finishes, 5.7667 s, 10 clean routes
                           => 5 train / 5 holdout routes; 8 / 3 state pools
```

The 0035 transfer found the only real 1-4 finishers. 0037 is the first test of
the seeding branch. Its first generation reached **40/384** finishes and
**5.9167 s**, but the live population regressed toward the old 10.5 s route
because 0037 did not enable archive rebalancing. The fast route is still in the
archive. 0038 tested the combination that 0037 lacked: the versioned hold/ceiling
observation, reasserted held input, and archive rebalancing every two
generations from the start. It found **10 clean routes in generation 1** and
was stopped on the clean target. The resulting pool is valid and is now used by
0039 for protected DDQN training.

## Non-negotiable checks

- Do not merge an N-1 route archive into N: route states are level-specific.
- Do not select by best time alone: clean-route count and full start coverage
  are the gates.
- Do not build a pool from an archive with zero clean routes.
- Record the game commit/dirty state for every result.
