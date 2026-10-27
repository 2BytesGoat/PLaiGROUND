"""Select per-level teacher champions after the time-profile retrain.

The old post-fix teacher is the fallback. A new teacher is promoted only when
it keeps the finish gate and beats the old median time. The chosen model is
copied to ``runs/teacher_champions/<level>/ppo_pilot.zip``, which is the first
root used by the distillation collector.

Usage:
    .venv/bin/python src/14_select_teachers.py --levels all
    .venv/bin/python src/14_select_teachers.py --levels 1-11 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


LEVELS = [f"1-{i}" for i in range(1, 17)]
OLD_TEACHER_ROOTS = (
    Path("runs/teacher_repair/final"),
    Path("runs/teacher_repair_post"),
    Path("runs/ppo_lineage/snapshots"),
    Path("runs/ppo_lineage_cont/snapshots"),
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "runs" / "teacher_champions"
MIN_FINISH_RATE = 0.90
DEFAULT_TIME_TOLERANCE = 0.01
DEFAULT_METRIC_TOLERANCE = 0.02


def parse_levels(value: str) -> list[str]:
    if not value or value.lower() == "all":
        return list(LEVELS)
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [level for level in levels if level not in LEVELS]
    if unknown:
        raise ValueError(f"Unknown level(s): {', '.join(unknown)}")
    return levels


def old_teacher_path(level: str, root: Path | None = None) -> Path | None:
    roots = (root,) if root is not None else OLD_TEACHER_ROOTS
    for candidate_root in roots:
        candidate = candidate_root / level / "ppo_pilot.zip"
        if candidate.exists():
            return candidate
    return None


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def as_float(value, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def eligible(summary: dict) -> bool:
    if not summary:
        return False
    agents = as_float(summary.get("agents"), 100.0)
    finish_rate = as_float(summary.get("finishes"), 0.0) / max(agents, 1.0)
    if finish_rate < MIN_FINISH_RATE:
        return False
    metric = as_float(summary.get("median_time"))
    if metric == float("inf"):
        metric = as_float(summary.get("best_time"))
    return metric != float("inf")


def selection_metric(summary: dict) -> float:
    median = as_float(summary.get("median_time"))
    return median if median != float("inf") else as_float(summary.get("best_time"))


def choose(
    old_eval: dict,
    new_eval: dict,
    old_ok: bool,
    new_ok: bool,
    time_tolerance: float,
    metric_tolerance: float,
) -> tuple[bool, str]:
    """Return (choose_new, reason).

    Speed is primary. Efficiency/flow only break near-ties; consistency is a
    separate recovery gate and is not folded into this scalar decision.
    """
    if not old_ok and not new_ok:
        return False, "no-eligible-teacher"
    if not old_ok:
        return True, "old-ineligible"
    if not new_ok:
        return False, "new-ineligible"

    old_time = selection_metric(old_eval)
    new_time = selection_metric(new_eval)
    if new_time < old_time * (1.0 - time_tolerance):
        return True, "faster"
    if old_time < new_time * (1.0 - time_tolerance):
        return False, "slower"

    old_eff = as_float(old_eval.get("efficiency_mean"), 0.0)
    new_eff = as_float(new_eval.get("efficiency_mean"), 0.0)
    if new_eff > old_eff + metric_tolerance:
        return True, "efficiency-tiebreak"
    if old_eff > new_eff + metric_tolerance:
        return False, "efficiency-tiebreak"

    old_flow = as_float(old_eval.get("flow_mean"), 0.0)
    new_flow = as_float(new_eval.get("flow_mean"), 0.0)
    if new_flow > old_flow + metric_tolerance:
        return True, "flow-tiebreak"
    if old_flow > new_flow + metric_tolerance:
        return False, "flow-tiebreak"
    return False, "equal-keep-old"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="all")
    parser.add_argument("--old-eval-root", type=Path, default=Path("runs/teacher_eval"))
    parser.add_argument("--new-eval-root", type=Path, default=Path("runs/teacher_time"))
    parser.add_argument("--new-model-root", type=Path, default=Path("runs/teacher_time"))
    parser.add_argument(
        "--old-model-root",
        type=Path,
        default=None,
        help="Override old-teacher roots, e.g. runs/teacher_champions for a merge pass.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=DEFAULT_TIME_TOLERANCE,
        help="Relative time difference treated as a tie before metric tie-breaks.",
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=DEFAULT_METRIC_TOLERANCE,
        help="Efficiency/flow difference required to break a time tie.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    rows = []
    for level in parse_levels(args.levels):
        old_model = old_teacher_path(level, args.old_model_root)
        new_model = args.new_model_root / level / "ppo_pilot.zip"
        old_eval_path = args.old_eval_root / level / "eval" / "summary.csv"
        new_eval_path = args.new_eval_root / level / "eval" / "summary.csv"
        old_eval = read_summary(old_eval_path)
        new_eval = read_summary(new_eval_path)

        old_ok = old_model is not None and old_model.exists() and eligible(old_eval)
        new_ok = new_model.exists() and eligible(new_eval)
        choose_new, reason = choose(
            old_eval,
            new_eval,
            old_ok,
            new_ok,
            args.time_tolerance,
            args.metric_tolerance,
        )
        chosen_model = new_model if choose_new else old_model
        chosen_eval = new_eval if choose_new else old_eval
        chosen_source = "new" if choose_new else "old"

        status = "chosen" if chosen_model is not None and chosen_model.exists() else "missing"
        if status == "chosen" and not args.dry_run:
            target_dir = args.output_dir / level
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(chosen_model, target_dir / "ppo_pilot.zip")
            chosen_eval_path = new_eval_path if choose_new else old_eval_path
            if chosen_eval_path.exists():
                target_eval = target_dir / "eval" / "summary.csv"
                target_eval.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(chosen_eval_path, target_eval)

        row = {
            "level": level,
            "status": status,
            "chosen": chosen_source,
            "reason": reason,
            "old_median": old_eval.get("median_time", ""),
            "old_finishes": old_eval.get("finishes", ""),
            "old_efficiency": old_eval.get("efficiency_mean", ""),
            "old_flow": old_eval.get("flow_mean", ""),
            "new_median": new_eval.get("median_time", ""),
            "new_finishes": new_eval.get("finishes", ""),
            "new_efficiency": new_eval.get("efficiency_mean", ""),
            "new_flow": new_eval.get("flow_mean", ""),
            "chosen_median": chosen_eval.get("median_time", ""),
            "chosen_efficiency": chosen_eval.get("efficiency_mean", ""),
            "chosen_flow": chosen_eval.get("flow_mean", ""),
            "model": str(args.output_dir / level / "ppo_pilot.zip"),
        }
        rows.append(row)
        print(
            f"[{level}] {row['chosen']} ({reason}) "
            f"old={row['old_median'] or '?'} "
            f"new={row['new_median'] or '?'} "
            f"eff old/new={row['old_efficiency'] or '?'}/"
            f"{row['new_efficiency'] or '?'} "
            f"flow old/new={row['old_flow'] or '?'}/"
            f"{row['new_flow'] or '?'}",
            flush=True,
        )

    summary_path = args.output_dir / "summary.csv"
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[select] summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
