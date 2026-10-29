"""Quality-diversity portfolio helpers for NEAT checkpoints.

Discovery and deployment are separate concerns:

* the discovery population keeps high mutation and may contain non-finishers;
* the deployment portfolio is finishers only, binned by route cluster and time;
* checkpoint rebalancing protects representative genomes while preserving
  NEAT's mutation/innovation state.

The module deliberately does not give the policy a level identity.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import neat

from plaiground.data.route_filter import cluster_routes, load_routes

TIME_BINS = ("speed", "median", "slow", "exploration")

# Fractions of a deployment portfolio.
DEFAULT_QUOTAS = {
    "speed": 0.15,
    "median": 0.70,
    "slow": 0.10,
    "exploration": 0.05,
}


@dataclass
class PortfolioEntry:
    genome_id: int
    run_time: float | None
    clean: bool
    cluster_id: str | None
    bin: str
    route_signature: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "run_time": self.run_time,
            "clean": self.clean,
            "cluster_id": self.cluster_id,
            "bin": self.bin,
            "route_signature": self.route_signature,
        }


def time_bin(run_time: float | None) -> str:
    if run_time is None or not math.isfinite(run_time):
        return "exploration"
    if run_time < 4.5:
        return "speed"
    if run_time < 6.0:
        return "median"
    if run_time < 10.0:
        return "slow"
    return "exploration"


def _route_clusters(
    routes: list[dict[str, Any]], jaccard: float
) -> tuple[dict[str, tuple[str, int]], dict[str, dict[str, Any]]]:
    """Return route-signature -> (cluster id, cluster size) and best route map."""
    unique: dict[str, dict[str, Any]] = {}
    for route in routes:
        signature = str(route.get("route_signature", "")).strip()
        if not signature:
            continue
        cells = route.get("path_cells")
        if not isinstance(cells, list) or len(cells) < 2:
            continue
        previous = unique.get(signature)
        if previous is None or float(route.get("run_time", 1e30)) < float(
            previous.get("run_time", 1e30)
        ):
            unique[signature] = route
    clusters = cluster_routes(list(unique.values()), jaccard)
    mapping: dict[str, tuple[str, int]] = {}
    for index, cluster in enumerate(clusters):
        cluster_id = f"c{index:03d}"
        for route in cluster.routes:
            signature = str(route.get("route_signature", "")).strip()
            if signature:
                mapping[signature] = (cluster_id, len(cluster.routes))
    return mapping, unique


def build_entries(
    population: dict[int, Any],
    routes: list[dict[str, Any]],
    *,
    jaccard: float = 0.5,
) -> tuple[list[PortfolioEntry], dict[str, Any]]:
    """Attach the best route for each genome to its portfolio metadata."""
    cluster_map, _unique = _route_clusters(routes, jaccard)
    best_by_genome: dict[int, dict[str, Any]] = {}
    for route in routes:
        try:
            genome_id = int(route.get("genome_id"))
        except (TypeError, ValueError):
            continue
        if genome_id not in population:
            continue
        previous = best_by_genome.get(genome_id)
        if previous is None or float(route.get("run_time", 1e30)) < float(
            previous.get("run_time", 1e30)
        ):
            best_by_genome[genome_id] = route

    entries: list[PortfolioEntry] = []
    for genome_id in sorted(population):
        route = best_by_genome.get(genome_id)
        if route is None:
            entries.append(
                PortfolioEntry(
                    genome_id=genome_id,
                    run_time=None,
                    clean=False,
                    cluster_id=None,
                    bin="exploration",
                    route_signature=None,
                )
            )
            continue
        signature = str(route.get("route_signature", "")).strip() or None
        cluster = cluster_map.get(signature or "")
        entries.append(
            PortfolioEntry(
                genome_id=genome_id,
                run_time=float(route["run_time"])
                if route.get("run_time") is not None
                else None,
                clean=bool(route.get("clean")),
                cluster_id=cluster[0] if cluster else None,
                bin=time_bin(
                    float(route["run_time"])
                    if route.get("run_time") is not None
                    else None
                ),
                route_signature=signature,
            )
        )
    counts = {name: 0 for name in TIME_BINS}
    for entry in entries:
        counts[entry.bin] += 1
    return entries, {
        "population": len(entries),
        "finishers": sum(entry.bin != "exploration" for entry in entries),
        "bins": counts,
        "clusters": len({entry.cluster_id for entry in entries if entry.cluster_id}),
    }


def select_portfolio(
    entries: list[PortfolioEntry],
    *,
    total: int,
    quotas: dict[str, float] | None = None,
    max_per_cluster: int = 32,
) -> list[PortfolioEntry]:
    """Select a quota-based portfolio, preserving cluster diversity."""
    quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    quota_total = sum(max(0.0, float(value)) for value in quotas.values())
    if quota_total <= 0:
        raise ValueError("portfolio quotas must sum to a positive value")
    quotas = {
        key: max(0.0, float(value)) / quota_total
        for key, value in quotas.items()
    }

    by_bin = {name: [] for name in TIME_BINS}
    for entry in entries:
        by_bin[entry.bin].append(entry)
    for name in by_bin:
        by_bin[name].sort(
            key=lambda entry: (
                entry.run_time is None,
                entry.run_time if entry.run_time is not None else float("inf"),
                entry.genome_id,
            )
        )

    selected: list[PortfolioEntry] = []
    selected_ids: set[int] = set()
    cluster_counts: dict[str, int] = {}
    for name in TIME_BINS:
        quota = int(round(total * quotas.get(name, 0.0)))
        candidates = by_bin[name]
        # First pass respects the cluster cap.
        for entry in candidates:
            if len(selected) >= total or quota <= 0:
                break
            if entry.genome_id in selected_ids:
                continue
            if (
                entry.cluster_id is not None
                and cluster_counts.get(entry.cluster_id, 0) >= max_per_cluster
            ):
                continue
            selected.append(entry)
            selected_ids.add(entry.genome_id)
            if entry.cluster_id is not None:
                cluster_counts[entry.cluster_id] = (
                    cluster_counts.get(entry.cluster_id, 0) + 1
                )
            quota -= 1
        # Second pass fills a short bin from its remaining candidates.
        for entry in candidates:
            if len(selected) >= total or quota <= 0:
                break
            if entry.genome_id in selected_ids:
                continue
            selected.append(entry)
            selected_ids.add(entry.genome_id)
            quota -= 1

    # Fill any remaining slots with the best remaining finishers, then
    # exploration genomes. This keeps the population at the requested size.
    remaining = sorted(
        (entry for entry in entries if entry.genome_id not in selected_ids),
        key=lambda entry: (
            entry.run_time is None,
            entry.run_time if entry.run_time is not None else float("inf"),
            entry.genome_id,
        ),
    )
    for entry in remaining:
        if len(selected) >= total:
            break
        selected.append(entry)
        selected_ids.add(entry.genome_id)
    return selected


def _archive_candidates(
    checkpoint_population: dict[int, Any],
    routes: list[dict[str, Any]],
    archive_dir: Path,
) -> tuple[dict[int, Any], list[dict[str, Any]], dict[str, int]]:
    """Add archived route genomes to the candidate pool.

    A checkpoint contains only the live population, while the route archive
    contains the best genome from every generation. Long runs usually have many
    more finishers in the archive than in the final live population.
    """
    import pickle

    candidates = {
        genome_id: copy.deepcopy(genome)
        for genome_id, genome in checkpoint_population.items()
    }
    route_copies: list[dict[str, Any]] = []
    added = 0
    next_key = max(candidates) + 1 if candidates else 1
    seen_genomes: dict[int, int] = {}
    for route in routes:
        item = dict(route)
        genome_file = item.get("genome_file")
        if not genome_file:
            continue
        path = archive_dir / str(genome_file)
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                genome = pickle.load(handle)
        except (OSError, pickle.UnpicklingError):
            continue
        original_key = getattr(genome, "key", item.get("genome_id"))
        try:
            original_key = int(original_key)
        except (TypeError, ValueError):
            original_key = None
        if original_key is not None and original_key in seen_genomes:
            candidate_key = seen_genomes[original_key]
        else:
            candidate_key = (
                original_key
                if original_key is not None and original_key not in candidates
                else next_key
            )
            if candidate_key == next_key:
                next_key += 1
            if original_key is not None:
                seen_genomes[original_key] = candidate_key
            genome.key = candidate_key
            candidates[candidate_key] = genome
            added += 1
        item["genome_id"] = candidate_key
        route_copies.append(item)
    return candidates, route_copies, {
        "checkpoint_genomes": len(checkpoint_population),
        "archive_genomes": added,
    }


def rebalance_population(
    checkpoint: Path,
    routes_path: Path,
    *,
    output_prefix: Path,
    quotas: dict[str, float] | None = None,
    jaccard: float = 0.5,
    max_per_cluster: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Rebuild a checkpoint population around a protected portfolio."""
    import random

    checkpoint = Path(checkpoint)
    routes_path = Path(routes_path)
    output_prefix = Path(output_prefix)
    random.seed(seed)
    population = neat.Checkpointer.restore_checkpoint(str(checkpoint))
    config = population.config
    original_generation = int(getattr(population, "generation", 0))
    routes = load_routes(routes_path)
    candidates, routes, candidate_stats = _archive_candidates(
        population.population, routes, routes_path.parent
    )
    entries, stats = build_entries(candidates, routes, jaccard=jaccard)
    stats["archive_candidates"] = candidate_stats
    target_size = len(population.population)
    selected = select_portfolio(
        entries,
        total=target_size,
        quotas=quotas,
        max_per_cluster=max_per_cluster,
    )
    selected_ids = {entry.genome_id for entry in selected}

    max_key = max(population.population) if population.population else 0
    next_key = max_key + 1
    new_population = {
        genome_id: copy.deepcopy(population.population[genome_id])
        for genome_id in selected_ids
        if genome_id in population.population
    }
    parents = list(new_population.values())
    while len(new_population) < target_size and parents:
        parent = copy.deepcopy(random.choice(parents))
        parent.key = next_key
        next_key += 1
        parent.fitness = None
        parent.mutate(config.genome_config)
        new_population[parent.key] = parent

    while len(new_population) < target_size:
        genome = neat.DefaultGenome(next_key)
        next_key += 1
        genome.configure_new(config.genome_config)
        new_population[genome.key] = genome

    # Rebuild a clean species set from the rebalanced population.
    rebalanced = neat.Population(config)
    rebalanced.population = new_population
    rebalanced.generation = original_generation
    species_config = config.species_set_config
    rebalanced.species = neat.DefaultSpeciesSet(species_config, rebalanced.reporters)
    rebalanced.species.speciate(config, rebalanced.population, original_generation)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = neat.Checkpointer(1, filename_prefix=str(output_prefix))
    checkpointer.save_checkpoint(
        config, rebalanced.population, rebalanced.species, original_generation
    )
    output_checkpoint = Path(f"{output_prefix}{original_generation}")
    portfolio = {
        "source_checkpoint": str(checkpoint),
        "source_generation": original_generation,
        "output_checkpoint": str(output_checkpoint),
        "target_size": target_size,
        "selected": [entry.as_dict() for entry in selected],
        "portfolio_stats": {
            "selected": len(selected),
            "selected_by_bin": {
                name: sum(1 for entry in selected if entry.bin == name)
                for name in TIME_BINS
            },
            "selected_by_cluster": len(
                {entry.cluster_id for entry in selected if entry.cluster_id}
            ),
        },
        "source_stats": stats,
    }
    portfolio_path = output_prefix.with_suffix(".portfolio.json")
    portfolio_path.write_text(json.dumps(portfolio, indent=2), encoding="utf-8")
    return portfolio
