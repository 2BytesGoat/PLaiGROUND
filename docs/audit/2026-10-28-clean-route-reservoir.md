# Audit — clean-route reservoir and post-processing filter

**Date:** 2026-10-28
**Status:** decision recorded; collection target and filter implemented;
filter smoke-tested.
**Scope:** clean-route collection target, path filtering, route-level holdout,
and the handoff from NEAT to the clean-start pool.

## Decision

The next 1-1 NEAT collection targets **450 unique clean route signatures with
checkpoint snapshots**.

That number is a **data reservoir**, not a claim that there are 450 independent
route families. Route diversity is bounded by the small 1-1 map; many exact
signatures differ only by small path variations. We will therefore:

1. collect until 450 unique clean signatures with checkpoint snapshots exist;
2. filter and cluster the paths **after collection**;
3. cap near-duplicate routes per cluster;
4. build start states from the filtered routes;
5. judge the pool by actual state coverage, not by the raw signature count.

We will **not** require 450 independent route clusters. That would turn a
bounded-map problem into an open-ended collection target.

## Why

The old 8×24 archive gives the reference scale:

```text
65 generations × 192 agents
4,243 finishers
1,417 clean finishers
460 unique clean route signatures
```

The 460 exact signatures are highly redundant. Connected-component clustering
at 0.5 path-cell Jaccard collapses them to 33 clusters, and the largest cluster
contains 397 routes. At 0.7 Jaccard there are 209 clusters. The 450 target is
therefore useful as volume and opportunity, but it cannot be used as a
diversity metric.

On a small map, many routes can still contribute useful **state snapshots**
(different cells, facings, times, velocities, and powerup states) even when the
final routes are near-duplicates. The right acceptance test is the filtered
start pool, not the route signature count.

## Collection target implementation

`07_neat_agent.py` now supports:

```text
--record-paths
--min-clean-routes 450
```

It stops after the generation in which the route archive contains at least 450
unique clean route signatures with checkpoint snapshots. The target does not
count raw finishers, and it does not count route clusters. Use a hard
`--generations` cap and leave `--target-time` unset for a route-collection run.

## Filtering policy

Implemented in
[`../../src/postprocessing/route_filter.py`](../../src/postprocessing/route_filter.py),
with the numbered CLI in
[`../../src/42_filter_clean_routes.py`](../../src/42_filter_clean_routes.py).

- Input: `clean_routes.json` (or a route mapping).
- Exact duplicate signatures keep the fastest route.
- Near-duplicate paths are grouped with **greedy representative clustering**:
  - similarity is Jaccard overlap of visited level cells;
  - a route joins the most similar cluster representative above the threshold;
  - otherwise it starts a new cluster.
- Default threshold: `0.5`.
- Keep at most `8` routes per cluster by default.
- Within a cluster, routes are selected across the run-time range rather than
  only the eight fastest.
- Optional holdout: reserve a fraction of **whole clusters**, never individual
  routes, so train and holdout never share a cluster.
- Outputs:
  - `filtered_clean_routes.json`
  - `route_clusters.json`
  - `summary.json`
  - `train_clean_routes.json` / `holdout_clean_routes.json` when split

If filtering leaves only one cluster, a route-level holdout is not meaningful.
Use a time-segment or cell/facing holdout and label the result accordingly; do
not claim unseen-route generalization.

## Smoke results

On the old 460-route archive, with `--jaccard 0.5 --max-per-cluster 8`:

```text
460 input routes
113 clusters (greedy representative clustering)
354 filtered routes
0 checkpoint files (old archive predates clean-start capture)
```

That run validates filtering only. On the current 8-route capture, where every
route has a checkpoint snapshot, with a 25% cluster holdout:

```text
8 input routes
6 clusters
8 filtered routes
6 train / 2 holdout routes
8 routes with checkpoint files
```

The difference between the 113 greedy clusters and the earlier 33 connected
components is intentional: greedy representative clustering avoids chaining
two dissimilar routes through an intermediate path.

## Consequences

- The NEAT collection run stays simple: stop at 450 clean signatures with
  checkpoints.
- The route filter is deterministic and runs offline.
- The filter does not yet build state pools or feed the PPO/DQN drivers. The
  existing extractor still needs to consume `filtered_clean_routes.json` and
  preserve the cluster split when expanding checkpoint snapshots.
- No 794/prefix data is introduced by this decision.

## References

- [`../experiments/ppo-clean-start.md`](../experiments/ppo-clean-start.md)
- [`../known-issues.md`](../known-issues.md)
- [`2026-10-28-clean-start-state-capture.md`](2026-10-28-clean-start-state-capture.md)
- [`2026-10-28-raw-state-observation-decision.md`](2026-10-28-raw-state-observation-decision.md)
