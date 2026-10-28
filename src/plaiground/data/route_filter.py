#!/usr/bin/env python3
"""Pure post-processing for clean NEAT routes.

The NEAT collection target is a reservoir of unique clean route signatures;
this module decides which of those routes are useful for the start pool. It is
kept separate from the numbered CLI so later pool-building and evaluation code
can reuse the same deterministic filtering logic.

Pipeline:

1. load clean routes from ``clean_routes.json`` (or ``routes.json``);
2. deduplicate exact route signatures, keeping the fastest one;
3. greedily cluster near-duplicate paths by Jaccard similarity of visited
   level cells;
4. keep up to ``max_per_cluster`` routes per cluster, spread across the
   cluster's run-time range;
5. optionally split whole clusters into train/holdout route files.

The output keeps the original route fields and adds cluster/selection metadata.
It does not build state pools; feed the filtered route files to the clean-start
pool extractor.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DEFAULT_JACCARD = 0.5
DEFAULT_MAX_PER_CLUSTER = 8
DEFAULT_HOLDOUT_FRACTION = 0.0
DEFAULT_SEED = 42


def load_routes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        routes = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("routes"), list):
            routes = payload["routes"]
        else:
            # The NEAT archive is keyed by route signature. Preserve the key
            # when the value itself does not repeat the field.
            routes = []
            for signature, route in payload.items():
                if not isinstance(route, dict):
                    raise ValueError(f"{path}: route {signature!r} must be an object")
                item = dict(route)
                item.setdefault("route_signature", str(signature))
                routes.append(item)
    else:
        raise ValueError(f"{path}: expected a route list or mapping")
    if not all(isinstance(route, dict) for route in routes):
        raise ValueError(f"{path}: every route must be an object")
    return routes


def _cells(route: dict[str, Any]) -> list[tuple[int, int]]:
    cells = route.get("path_cells")
    if not isinstance(cells, list):
        return []
    result: list[tuple[int, int]] = []
    for cell in cells:
        if not isinstance(cell, (list, tuple)) or len(cell) < 2:
            continue
        try:
            result.append((int(cell[0]), int(cell[1])))
        except (TypeError, ValueError):
            continue
    return result


def _path_key(route: dict[str, Any]) -> str:
    return str(route.get("route_signature", "")).strip()


def _run_time(route: dict[str, Any]) -> float:
    try:
        return float(route.get("run_time", float("inf")))
    except (TypeError, ValueError):
        return float("inf")


def _sort_key(route: dict[str, Any]) -> tuple[float, str]:
    return (_run_time(route), _path_key(route))


def _prepare(
    routes: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: dict[str, dict[str, Any]] = {}
    stats = Counter()
    for route in routes:
        stats["input_routes"] += 1
        if route.get("clean") is False:
            stats["skipped_unclean"] += 1
            continue
        signature = _path_key(route)
        cells = _cells(route)
        if not signature or len(set(cells)) < 2:
            stats["skipped_invalid"] += 1
            continue
        previous = accepted.get(signature)
        if previous is not None:
            stats["duplicate_signatures"] += 1
            if _sort_key(route) >= _sort_key(previous):
                continue
        accepted[signature] = route
    prepared = sorted(accepted.values(), key=_sort_key)
    stats["unique_signatures"] = len(prepared)
    return prepared, dict(stats)


def _jaccard(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


class RouteCluster:
    def __init__(self, representative: dict[str, Any]) -> None:
        self.representative = representative
        self.representative_cells = set(_cells(representative))
        self.routes: list[dict[str, Any]] = []

    def add(self, route: dict[str, Any]) -> None:
        self.routes.append(route)

    @property
    def signature(self) -> str:
        return _path_key(self.representative)

    @property
    def run_time(self) -> float:
        return _run_time(self.representative)

    def sorted_routes(self) -> list[dict[str, Any]]:
        return sorted(self.routes, key=_sort_key)


def cluster_routes(
    routes: list[dict[str, Any]],
    threshold: float,
) -> list[RouteCluster]:
    """Greedy representative clustering.

    A route joins the most similar existing representative whose Jaccard
    similarity is at least ``threshold``; otherwise it starts a new cluster.
    This avoids the chaining problem of connected-component clustering while
    remaining deterministic for a sorted input list.
    """

    clusters: list[RouteCluster] = []
    for route in routes:
        cells = set(_cells(route))
        best: RouteCluster | None = None
        best_score = -1.0
        for cluster in clusters:
            score = _jaccard(cells, cluster.representative_cells)
            if score > best_score:
                best = cluster
                best_score = score
        if best is not None and best_score >= threshold:
            best.add(route)
        else:
            cluster = RouteCluster(route)
            cluster.add(route)
            clusters.append(cluster)

    clusters.sort(key=lambda cluster: (cluster.run_time, cluster.signature))
    return clusters


def _evenly_spaced_indices(length: int, limit: int) -> list[int]:
    if limit <= 0 or length <= limit:
        return list(range(length))
    if limit == 1:
        return [0]
    indices = {
        round(index * (length - 1) / (limit - 1)) for index in range(limit)
    }
    return sorted(indices)


def select_cluster_routes(
    cluster: RouteCluster,
    max_per_cluster: int,
) -> list[dict[str, Any]]:
    routes = cluster.sorted_routes()
    indices = _evenly_spaced_indices(len(routes), max_per_cluster)
    selected: list[dict[str, Any]] = []
    for rank in indices:
        route = dict(routes[rank])
        route["cluster_id"] = None
        route["cluster_size"] = len(routes)
        route["cluster_rank"] = rank
        route["cluster_representative"] = rank == 0
        selected.append(route)
    return selected


def _assign_splits(
    filtered: list[dict[str, Any]],
    clusters: list[RouteCluster],
    holdout_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    if not clusters or holdout_fraction <= 0.0:
        return set(), set()
    if len(clusters) == 1:
        return set(), {clusters[0].signature}

    holdout_count = max(1, round(len(clusters) * holdout_fraction))
    holdout_count = min(holdout_count, len(clusters) - 1)
    candidates = [cluster.signature for cluster in clusters]
    rng = random.Random(seed)
    holdout = set(rng.sample(candidates, holdout_count))
    train = set(candidates) - holdout
    for route in filtered:
        route["split"] = "holdout" if route["_cluster_signature"] in holdout else "train"
    return holdout, train


def filter_routes(
    routes: Iterable[dict[str, Any]],
    *,
    jaccard: float = DEFAULT_JACCARD,
    max_per_cluster: int = DEFAULT_MAX_PER_CLUSTER,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    seed: int = DEFAULT_SEED,
    level: str = "1-1",
) -> dict[str, Any]:
    if not 0.0 <= jaccard <= 1.0:
        raise ValueError("jaccard must be between 0 and 1")
    if not 0.0 <= holdout_fraction <= 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    prepared, stats = _prepare(routes)
    clusters = cluster_routes(prepared, jaccard)
    filtered: list[dict[str, Any]] = []
    cluster_payload: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters):
        cluster_id = f"c{index:03d}"
        selected = select_cluster_routes(cluster, max_per_cluster)
        for route in selected:
            route["cluster_id"] = cluster_id
            route["_cluster_signature"] = cluster.signature
            route["_cluster_size"] = len(cluster.routes)
            route["_cluster_kept"] = len(selected)
            filtered.append(route)
        cluster_payload.append(
            {
                "cluster_id": cluster_id,
                "representative_signature": cluster.signature,
                "representative_run_time": cluster.run_time,
                "route_count": len(cluster.routes),
                "kept_count": len(selected),
                "route_signatures": [route["route_signature"] for route in cluster.sorted_routes()],
            }
        )

    holdout, train = _assign_splits(filtered, clusters, holdout_fraction, seed)
    for route in filtered:
        route["split"] = route.get("split", "train")
        route.pop("_cluster_signature", None)
        route.pop("_cluster_size", None)
        route.pop("_cluster_kept", None)
    for cluster in cluster_payload:
        if holdout or train:
            cluster["split"] = (
                "holdout" if cluster["representative_signature"] in holdout else "train"
            )
        else:
            cluster["split"] = None

    filtered.sort(key=lambda route: (route.get("split", "train"), _sort_key(route)))
    cluster_payload.sort(key=lambda cluster: cluster["cluster_id"])
    routes_with_checkpoint = sum(bool(route.get("checkpoint_file")) for route in filtered)
    summary = {
        "level": level,
        "jaccard_threshold": jaccard,
        "max_per_cluster": max_per_cluster,
        "holdout_fraction": holdout_fraction,
        "seed": seed,
        **stats,
        "clusters": len(clusters),
        "filtered_routes": len(filtered),
        "filtered_routes_with_checkpoint": routes_with_checkpoint,
        "holdout_clusters": len(holdout),
        "train_clusters": len(train) if holdout else len(clusters),
        "holdout_routes": sum(route.get("split") == "holdout" for route in filtered),
        "train_routes": sum(route.get("split") == "train" for route in filtered),
        "cluster_sizes": [cluster["route_count"] for cluster in cluster_payload],
    }
    return {
        "summary": summary,
        "filtered_routes": filtered,
        "clusters": cluster_payload,
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "filtered_clean_routes.json").write_text(
        json.dumps(result["filtered_routes"], indent=2), encoding="utf-8"
    )
    (output_dir / "route_clusters.json").write_text(
        json.dumps(result["clusters"], indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )
    train = [route for route in result["filtered_routes"] if route.get("split") == "train"]
    holdout = [route for route in result["filtered_routes"] if route.get("split") == "holdout"]
    if result["summary"]["holdout_clusters"]:
        (output_dir / "train_clean_routes.json").write_text(
            json.dumps(train, indent=2), encoding="utf-8"
        )
        (output_dir / "holdout_clean_routes.json").write_text(
            json.dumps(holdout, indent=2), encoding="utf-8"
        )



