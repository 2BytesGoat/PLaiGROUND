"""Prepare one level's route tiers and clean-start pools from a NEAT archive.

This module is the repeatable bridge between NEAT route discovery and a
single-level DDQN expert. It deliberately does not start training or choose a
checkpoint; it only turns an archive into auditable, deterministic artifacts:

1. classify every finisher as strict-clean, near-clean, or all-success;
2. filter the selected quality tier into train/holdout route clusters;
3. build the train/holdout start pools and their deduplicated eval union;
4. optionally build the same artifacts for the all-success diagnostic tier.

The strict/near-clean thresholds mirror the NEAT trainer. They are recomputed
here rather than trusting the archive's ``clean`` flag so old archives with
different metadata remain usable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .route_filter import (
    DEFAULT_JACCARD,
    DEFAULT_MAX_PER_CLUSTER,
    filter_routes,
    load_routes,
    write_outputs,
)
from .start_pool import build_start_pools

STRICT_CLEAN = "strict_clean"
NEAR_CLEAN = "near_clean"
ALL_SUCCESS = "all_success"
QUALITY_CHOICES = (STRICT_CLEAN, NEAR_CLEAN, ALL_SUCCESS)

DEFAULT_PIPELINE_SEED = 0


class LevelDataError(ValueError):
    """A level archive cannot produce the requested route/pool artifacts."""


@dataclass(frozen=True)
class QualitySpec:
    """Route-quality thresholds used by the NEAT trainer and this pipeline."""

    name: str
    max_revisit_ratio: float | None
    max_backtracks: int | None
    max_action_flip_rate: float | None
    min_path_length: int = 5


QUALITY_SPECS: dict[str, QualitySpec] = {
    STRICT_CLEAN: QualitySpec(
        name=STRICT_CLEAN,
        max_revisit_ratio=0.10,
        max_backtracks=2,
        max_action_flip_rate=0.75,
    ),
    NEAR_CLEAN: QualitySpec(
        name=NEAR_CLEAN,
        max_revisit_ratio=0.20,
        max_backtracks=2,
        max_action_flip_rate=0.75,
    ),
    ALL_SUCCESS: QualitySpec(
        name=ALL_SUCCESS,
        max_revisit_ratio=None,
        max_backtracks=None,
        max_action_flip_rate=None,
        min_path_length=0,
    ),
}

QUALITY_TAGS = {
    STRICT_CLEAN: "clean",
    NEAR_CLEAN: "near_clean",
    ALL_SUCCESS: "all_success",
}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _route_time(route: dict[str, Any]) -> float:
    value = _finite_float(route.get("run_time"))
    return value if value is not None else float("inf")


def _route_signature(route: dict[str, Any]) -> str:
    return str(route.get("route_signature", "")).strip()


def _path_length(route: dict[str, Any]) -> int:
    value = route.get("path_length")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    cells = route.get("path_cells")
    return len(cells) if isinstance(cells, list) else 0


def _dedupe_routes(
    routes: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep the fastest route per signature, matching the NEAT archive rule."""
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    skipped = 0
    for route in routes:
        if not isinstance(route, dict):
            skipped += 1
            continue
        signature = _route_signature(route)
        if not signature:
            skipped += 1
            continue
        run_time = _route_time(route)
        previous = best.get(signature)
        if previous is None or run_time < previous[0]:
            best[signature] = (run_time, route)
    ordered = [
        route
        for _run_time, route in sorted(
            best.values(),
            key=lambda item: (item[0], _route_signature(item[1])),
        )
    ]
    return ordered, skipped


def _matches_quality(route: dict[str, Any], spec: QualitySpec) -> bool:
    if spec.max_revisit_ratio is None:
        return True
    if _path_length(route) < spec.min_path_length:
        return False
    revisit = _finite_float(route.get("revisit_ratio"))
    if revisit is None or revisit > spec.max_revisit_ratio:
        return False
    backtracks = _finite_float(route.get("backtracks"))
    if backtracks is None or backtracks > float(spec.max_backtracks or 0):
        return False
    flips = _finite_float(route.get("action_flip_rate"))
    if flips is None or flips > float(spec.max_action_flip_rate or 0.0):
        return False
    return True


def _quality_route(route: dict[str, Any], quality: str) -> dict[str, Any]:
    item = dict(route)
    item.pop("clean", None)
    item["quality"] = quality
    return item


def _quality_best(routes: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    if not routes:
        return None, None
    best = min(routes, key=lambda route: (_route_time(route), _route_signature(route)))
    return _finite_float(best.get("run_time")), _route_signature(best)


def classify_all_routes(
    routes: Iterable[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Classify routes into strict-clean, near-clean, and all-success tiers."""
    route_list = list(routes)
    deduped, skipped = _dedupe_routes(route_list)
    tiers: dict[str, list[dict[str, Any]]] = {}
    qualities: dict[str, Any] = {}
    for quality, spec in QUALITY_SPECS.items():
        selected = [
            _quality_route(route, quality)
            for route in deduped
            if _matches_quality(route, spec)
        ]
        selected.sort(key=lambda route: (_route_time(route), _route_signature(route)))
        tiers[quality] = selected
        best_time, best_signature = _quality_best(selected)
        qualities[quality] = {
            "routes": len(selected),
            "with_checkpoint": sum(
                bool(route.get("checkpoint_file")) for route in selected
            ),
            "best_time": best_time,
            "best_signature": best_signature,
            "thresholds": {
                "max_revisit_ratio": spec.max_revisit_ratio,
                "max_backtracks": spec.max_backtracks,
                "max_action_flip_rate": spec.max_action_flip_rate,
                "min_path_length": spec.min_path_length,
            },
        }
    summary = {
        "input_routes": len(route_list),
        "unique_routes": len(deduped),
        "skipped_invalid": skipped,
        "qualities": qualities,
    }
    return tiers, summary


def write_route_tiers(
    routes: Iterable[dict[str, Any]],
    *,
    output_dir: Path,
    level: str,
    generation: int | None = None,
    source: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Path]]:
    """Write the three route-tier JSON files plus a quality summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tiers, summary = classify_all_routes(routes)
    suffix = f"_g{generation}" if generation is not None else ""
    paths: dict[str, Path] = {}
    for quality, tier_routes in tiers.items():
        path = output_dir / f"{quality}_routes{suffix}.json"
        path.write_text(json.dumps(tier_routes, indent=2), encoding="utf-8")
        paths[quality] = path
    summary.update(
        {
            "level": level,
            "generation": generation,
            "source": source,
            "files": {quality: str(path) for quality, path in paths.items()},
        }
    )
    summary_path = output_dir / f"route_quality{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary"] = summary_path
    return tiers, summary, paths


def select_quality(
    quality_summary: dict[str, Any],
    *,
    requested: str = "auto",
    min_routes: int = 25,
) -> str:
    """Choose the accepted route tier.

    ``auto`` prefers strict-clean routes. If the strict tier has fewer than
    ``min_routes``, it falls back to near-clean. All-success is never selected
    automatically because it is a diagnostic tier, not a clean-start gate.
    """
    if requested not in ("auto", *QUALITY_CHOICES):
        raise LevelDataError(
            f"unknown quality {requested!r}; use auto, {', '.join(QUALITY_CHOICES)}"
        )
    counts = {
        quality: int(quality_summary["qualities"][quality]["routes"])
        for quality in QUALITY_CHOICES
    }
    if requested != "auto":
        return requested
    for quality in (STRICT_CLEAN, NEAR_CLEAN):
        if counts[quality] >= int(min_routes):
            return quality
    raise LevelDataError(
        "no clean tier meets --min-routes "
        f"({min_routes}): strict={counts[STRICT_CLEAN]}, "
        f"near={counts[NEAR_CLEAN]}, all_success={counts[ALL_SUCCESS]}. "
        "Extend the NEAT search or seed a high-mutation route-discovery run "
        "before building an expert pool."
    )


def _tag_for(quality: str) -> str:
    try:
        return QUALITY_TAGS[quality]
    except KeyError as exc:
        raise LevelDataError(f"unknown quality tier: {quality}") from exc


def _fastest_cluster(filter_result: dict[str, Any]) -> dict[str, Any] | None:
    clusters = filter_result.get("clusters", [])
    if not clusters:
        return None
    fastest = min(
        clusters,
        key=lambda cluster: (
            _finite_float(cluster.get("representative_run_time")) or float("inf"),
            str(cluster.get("cluster_id", "")),
        ),
    )
    return {
        "cluster_id": fastest.get("cluster_id"),
        "split": fastest.get("split"),
        "run_time": _finite_float(fastest.get("representative_run_time")),
        "route_count": fastest.get("route_count"),
        "kept_count": fastest.get("kept_count"),
    }


def _best_time_for_split(
    routes: list[dict[str, Any]], split: str | None
) -> float | None:
    if split is None:
        return None
    times = []
    for route in routes:
        if route.get("split") != split:
            continue
        value = _finite_float(route.get("run_time"))
        if value is not None:
            times.append(value)
    return min(times) if times else None


def _filter_and_pool(
    routes: list[dict[str, Any]],
    *,
    level: str,
    tag: str,
    generation: int | None,
    archive_dir: Path,
    analysis_path: Path,
    output_root: Path,
    jaccard: float,
    max_per_cluster: int,
    holdout_fraction: float,
    seed: int,
    canonical_weight: float,
    state_weight: float,
    min_time_to_exit: float,
) -> dict[str, Any]:
    suffix = f"_g{generation}" if generation is not None else ""
    filtered_dir = output_root / f"filtered_{tag}{suffix}"
    pool_dir = output_root / f"pool_{tag}{suffix}"
    filter_result = filter_routes(
        routes,
        jaccard=jaccard,
        max_per_cluster=max_per_cluster,
        holdout_fraction=holdout_fraction,
        seed=seed,
        level=level,
    )
    write_outputs(filter_result, filtered_dir)
    eval_pool_name = f"eval_{tag}_start_pool.json"
    pool_result = build_start_pools(
        filtered_dir=filtered_dir,
        archive_dir=archive_dir,
        analysis_path=analysis_path,
        output_dir=pool_dir,
        level=level,
        min_time_to_exit=min_time_to_exit,
        canonical_weight=canonical_weight,
        state_weight=state_weight,
        eval_pool_name=eval_pool_name,
    )
    filtered_routes = filter_result["filtered_routes"]
    fastest_cluster = _fastest_cluster(filter_result)
    warnings: list[str] = []
    if fastest_cluster and fastest_cluster.get("split") == "holdout":
        run_time = fastest_cluster.get("run_time")
        time_text = f" ({run_time:.4f} s)" if isinstance(run_time, float) else ""
        warnings.append(
            f"fastest cluster {fastest_cluster.get('cluster_id')}{time_text} "
            "is holdout-only; expert training will not see its actions"
        )
    return {
        "quality": tag,
        "tag": tag,
        "routes": len(routes),
        "filtered_dir": str(filtered_dir),
        "pool_dir": str(pool_dir),
        "eval_pool": str(pool_dir / eval_pool_name),
        "filter": filter_result["summary"],
        "pool": pool_result["summary"],
        "best_time": _quality_best(routes)[0],
        "best_time_train": _best_time_for_split(filtered_routes, "train"),
        "best_time_holdout": _best_time_for_split(filtered_routes, "holdout"),
        "fastest_cluster": fastest_cluster,
        "warnings": warnings,
    }


def prepare_level_data(
    *,
    level: str,
    run_dir: Path,
    generation: int | None,
    analysis_path: Path,
    output_root: Path | None = None,
    archive_dir: Path | None = None,
    quality: str = "auto",
    min_routes: int = 25,
    tag: str | None = None,
    build_all_success: bool = True,
    jaccard: float = DEFAULT_JACCARD,
    max_per_cluster: int = DEFAULT_MAX_PER_CLUSTER,
    holdout_fraction: float = 0.25,
    seed: int = DEFAULT_PIPELINE_SEED,
    canonical_weight: float = 0.5,
    state_weight: float = 0.5,
    min_time_to_exit: float = 0.5,
) -> dict[str, Any]:
    """Create route tiers, filtered route sets, pools, and the eval union."""
    run_dir = Path(run_dir).resolve()
    output_root = Path(output_root or run_dir).resolve()
    archive_dir = Path(archive_dir or (run_dir / "archive")).resolve()
    analysis_path = Path(analysis_path).resolve()
    routes_path = archive_dir / "routes.json"
    if not routes_path.exists():
        raise LevelDataError(f"missing NEAT route archive: {routes_path}")
    if not analysis_path.exists():
        raise LevelDataError(f"missing level analysis: {analysis_path}")

    routes = load_routes(routes_path)
    tiers, route_summary, tier_paths = write_route_tiers(
        routes,
        output_dir=output_root,
        level=level,
        generation=generation,
        source=str(routes_path),
    )
    selected_quality = select_quality(
        route_summary,
        requested=quality,
        min_routes=min_routes,
    )
    if not tiers[selected_quality]:
        raise LevelDataError(
            f"selected quality {selected_quality!r} has no routes; "
            "extend the NEAT search or choose another tier"
        )
    strict_count = int(route_summary["qualities"][STRICT_CLEAN]["routes"])
    near_count = int(route_summary["qualities"][NEAR_CLEAN]["routes"])
    if selected_quality == STRICT_CLEAN:
        selection_reason = (
            f"strict-clean routes {strict_count} >= min {min_routes}"
        )
    elif selected_quality == NEAR_CLEAN:
        selection_reason = (
            f"near-clean fallback: strict-clean routes {strict_count} "
            f"< min {min_routes}; near-clean routes {near_count}"
        )
    else:
        selection_reason = "all-success explicitly requested"
    selected_tag = tag or _tag_for(selected_quality)
    accepted = _filter_and_pool(
        tiers[selected_quality],
        level=level,
        tag=selected_tag,
        generation=generation,
        archive_dir=archive_dir,
        analysis_path=analysis_path,
        output_root=output_root,
        jaccard=jaccard,
        max_per_cluster=max_per_cluster,
        holdout_fraction=holdout_fraction,
        seed=seed,
        canonical_weight=canonical_weight,
        state_weight=state_weight,
        min_time_to_exit=min_time_to_exit,
    )

    diagnostic = None
    if build_all_success and selected_quality != ALL_SUCCESS:
        diagnostic = _filter_and_pool(
            tiers[ALL_SUCCESS],
            level=level,
            tag="all_success",
            generation=generation,
            archive_dir=archive_dir,
            analysis_path=analysis_path,
            output_root=output_root,
            jaccard=jaccard,
            max_per_cluster=max_per_cluster,
            holdout_fraction=holdout_fraction,
            seed=seed,
            canonical_weight=canonical_weight,
            state_weight=state_weight,
            min_time_to_exit=min_time_to_exit,
        )

    summary = {
        "level": level,
        "generation": generation,
        "source_archive": str(archive_dir),
        "analysis": str(analysis_path),
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "quality": route_summary,
        "tier_files": {
            quality_name: str(path)
            for quality_name, path in tier_paths.items()
            if quality_name != "summary"
        },
        "quality_summary": str(tier_paths["summary"]),
        "selected_quality": selected_quality,
        "selected_tag": selected_tag,
        "selection_reason": selection_reason,
        "min_routes": min_routes,
        "filter": {
            "jaccard": jaccard,
            "max_per_cluster": max_per_cluster,
            "holdout_fraction": holdout_fraction,
            "seed": seed,
        },
        "pool": {
            "canonical_weight": canonical_weight,
            "state_weight": state_weight,
            "min_time_to_exit": min_time_to_exit,
        },
        "accepted": accepted,
        "all_success": diagnostic,
        "warnings": list(accepted["warnings"]),
    }
    summary_path = output_root / "level_data_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if generation is not None:
        generation_path = output_root / f"level_data_summary_g{generation}.json"
        generation_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return summary
