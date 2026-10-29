#!/usr/bin/env python3
"""Seed a NEAT population from one or more archived finisher genomes.

This is the reusable recovery path for a new level:

1. transfer the previous level's population;
2. if it produces finishers but no clean routes, pick the best route genomes;
3. seed a high-mutation population around those genomes;
4. continue with ``--record-paths`` and build the route archive.

Only genomes are copied; no level identity or level-conditioned input is added.
"""

from __future__ import annotations

import argparse
import copy
import pickle
from itertools import count
from pathlib import Path

import neat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="NEAT config file for the new level/search.",
    )
    parser.add_argument(
        "--genome",
        type=Path,
        action="append",
        required=True,
        help="Archived genome pickle; repeat to seed from multiple routes.",
    )
    parser.add_argument("--population", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Checkpoint prefix; writes <prefix>0.",
    )
    return parser.parse_args()


def max_innovation(genomes: list[neat.DefaultGenome]) -> int:
    highest = 0
    for genome in genomes:
        for connection in genome.connections.values():
            highest = max(highest, int(getattr(connection, "innovation", 0)))
    return highest


def main() -> int:
    args = parse_args()
    if args.population < 1:
        raise SystemExit("--population must be positive")

    genomes = []
    for path in args.genome:
        with path.open("rb") as handle:
            genome = pickle.load(handle)
        if not isinstance(genome, neat.DefaultGenome):
            raise SystemExit(f"not a NEAT genome: {path}")
        genomes.append(genome)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(args.config),
    )
    config.pop_size = int(args.population)
    population = neat.Population(config, seed=args.seed)

    genome_config = config.genome_config
    tracker = population.reproduction.innovation_tracker
    tracker.global_counter = max(
        int(getattr(tracker, "global_counter", 0)),
        max_innovation(genomes),
    )
    tracker.reset_generation()
    genome_config.innovation_tracker = tracker

    next_key = 1
    seeded: dict[int, neat.DefaultGenome] = {}
    for index in range(int(args.population)):
        source = genomes[index % len(genomes)]
        genome = copy.deepcopy(source)
        genome.key = next_key
        genome.fitness = None
        next_key += 1
        # Keep one intact elite per source route, mutate the rest.
        if index >= len(genomes):
            genome.mutate(genome_config)
        seeded[genome.key] = genome

    species = neat.DefaultSpeciesSet(config.species_set_config, population.reporters)
    species.speciate(config, seeded, 0)
    population.population = seeded
    population.species = species
    population.generation = 0
    population.best_genome = None
    population.reproduction.genome_indexer = count(next_key)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = neat.Checkpointer(1, filename_prefix=str(args.output_prefix))
    checkpointer.save_checkpoint(config, population.population, species, 0)

    print(
        f"seeded {len(seeded)} genomes from {len(genomes)} source route(s) "
        f"at {args.output_prefix}0 (max innovation {max_innovation(genomes)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
