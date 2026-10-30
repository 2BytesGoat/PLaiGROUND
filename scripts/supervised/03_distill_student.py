#!/usr/bin/env python3
"""Train one level-agnostic student from captured teacher rollouts.

Supported input variants:

* ``raw191``  — 191 raw ``lean_stick`` values.
* ``onehot957`` — the fitted 957-value lean one-hot input.
* ``raw191-wide`` — raw 191 with a wider first hidden layer.
* ``grid191`` — raw 191 with a 7x7 convolutional grid encoder.

All variants use the same trajectories and action labels. The script writes a
``student.pt`` checkpoint, training history, and a small metadata JSON.

Example:
    .venv/bin/python scripts/supervised/03_distill_student.py \
      --data-dir runs/distill_campaign_student \
      --output-dir runs/distill_campaign_student/students/raw191_512 \
      --input-kind raw191 --hidden 512 512
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from plaiground.models.student import GridStudent, StudentMLP


def load_dataset(data_dir: Path, input_kind: str) -> dict:
    obs_key = (
        "obs_raw"
        if input_kind.startswith("raw191") or input_kind == "grid191"
        else "obs_onehot"
    )
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    levels: list[np.ndarray] = []
    splits: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    route_indices: list[np.ndarray] = []
    index_path = data_dir / "index.json"
    entries: list[tuple[Path, str | None]] = []
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for item in index:
            if isinstance(item, str):
                entries.append((data_dir / item / "transitions.npz", None))
            else:
                entries.append(
                    (
                        data_dir / str(item["path"]) / "transitions.npz",
                        item.get("split"),
                    )
                )
    else:
        entries = [(path, None) for path in sorted(data_dir.glob("**/transitions.npz"))]
    if not entries:
        raise SystemExit(f"no transitions.npz under {data_dir}")
    for path, split_override in entries:
        if not path.exists():
            raise SystemExit(f"missing dataset file: {path}")
        with np.load(path, allow_pickle=False) as data:
            if obs_key not in data:
                raise SystemExit(f"{path}: missing {obs_key}")
            success = (
                np.asarray(data["route_success"], dtype=bool)
                if "route_success" in data
                else np.ones(len(data["actions"]), dtype=bool)
            )
            if not success.any():
                print(f"[skip] {path}: no successful teacher routes")
                continue
            features.append(np.asarray(data[obs_key][success], dtype=np.float32))
            actions.append(np.asarray(data["actions"][success], dtype=np.int64))
            levels.append(np.asarray(data["level"][success]))
            if split_override is not None:
                splits.append(
                    np.full(int(success.sum()), str(split_override), dtype=object)
                )
            else:
                splits.append(
                    np.asarray(data["split"][success])
                    if "split" in data
                    else np.full(int(success.sum()), "train", dtype=object)
                )
            signatures.append(
                np.asarray(data["route_signature"][success])
                if "route_signature" in data
                else np.array([path.parent.name] * int(success.sum()), dtype=object)
            )
            route_indices.append(
                np.asarray(data["route_index"][success], dtype=np.int64)
                if "route_index" in data
                else np.zeros(int(success.sum()), dtype=np.int64)
            )
    if not features:
        raise SystemExit("no successful transitions in the dataset")
    result = {
        "features": np.concatenate(features, axis=0),
        "actions": np.concatenate(actions, axis=0),
        "levels": np.concatenate(levels, axis=0),
        "splits": np.concatenate(splits, axis=0),
        "signatures": np.concatenate(signatures, axis=0),
        "route_indices": np.concatenate(route_indices, axis=0),
    }
    return result


def split_arrays(dataset: dict, seed: int) -> tuple[dict, dict, dict]:
    """Return train, validation, and holdout views.

    The route split is the primary holdout. If no holdout routes were captured,
    reserve 20% of route signatures as validation so the training loop still
    has an early-stopping signal; that is not a substitute for a holdout path.
    """
    splits = np.asarray(dataset["splits"], dtype=object)
    train_mask = splits == "train"
    holdout_mask = splits == "holdout"
    if not holdout_mask.any():
        signatures = sorted(set(str(value) for value in dataset["signatures"]))
        rng = np.random.default_rng(seed)
        rng.shuffle(signatures)
        held = set(signatures[: max(1, len(signatures) // 5)])
        holdout_mask = np.asarray(
            [str(value) in held for value in dataset["signatures"]], dtype=bool
        )
        train_mask = ~holdout_mask
        print(
            "[warn] no explicit holdout routes; using a route-signature "
            "validation split instead"
        )
    train = {key: value[train_mask] for key, value in dataset.items()}
    holdout = {key: value[holdout_mask] for key, value in dataset.items()}
    # Carve a small validation slice out of the training routes for early
    # stopping only. The holdout remains untouched until evaluation.
    train_signatures = sorted(set(str(value) for value in train["signatures"]))
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(train_signatures)
    val_signatures = set(train_signatures[: max(1, len(train_signatures) // 10)])
    val_mask = np.asarray(
        [str(value) in val_signatures for value in train["signatures"]], dtype=bool
    )
    validation = {key: value[val_mask] for key, value in train.items()}
    fit = {key: value[~val_mask] for key, value in train.items()}
    return fit, validation, holdout


def summarize(dataset: dict) -> dict:
    levels = {}
    for level in sorted(set(str(value) for value in dataset["levels"])):
        mask = dataset["levels"] == level
        levels[level] = {
            "transitions": int(mask.sum()),
            "routes": int(len(set(str(v) for v in dataset["signatures"][mask]))),
            "jump_fraction": float(dataset["actions"][mask].mean()),
        }
    return {
        "transitions": int(len(dataset["actions"])),
        "routes": int(len(set(str(v) for v in dataset["signatures"]))),
        "jump_fraction": float(dataset["actions"].mean()),
        "levels": levels,
    }


def evaluate(model: nn.Module, dataset: dict, device: str) -> dict:
    if len(dataset["actions"]) == 0:
        return {"transitions": 0, "accuracy": None, "loss": None}
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(dataset["features"], dtype=torch.float32, device=device)
        y = torch.as_tensor(dataset["actions"], dtype=torch.long, device=device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y).item()
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()
    return {
        "transitions": int(len(y)),
        "accuracy": float(accuracy),
        "loss": float(loss),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-kind",
        choices=["raw191", "onehot957", "raw191-wide", "grid191"],
        required=True,
    )
    parser.add_argument("--hidden", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-level-balance",
        action="store_true",
        help="Use raw transition frequencies instead of equal level weights.",
    )
    parser.add_argument(
        "--level-weight",
        action="append",
        default=[],
        metavar="LEVEL=WEIGHT",
        help=(
            "Multiply one level's balanced sample weight, e.g. "
            "--level-weight 1-3=4 --level-weight 1-14=2. May be repeated."
        ),
    )
    parser.add_argument(
        "--init-student",
        type=Path,
        default=None,
        help="Optional compatible student.pt to fine-tune from.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_dataset(args.data_dir, args.input_kind)
    fit, validation, holdout = split_arrays(dataset, args.seed)
    summary = summarize(dataset)
    print(json.dumps(summary, indent=2))

    input_dim = int(dataset["features"].shape[1])
    if args.input_kind == "onehot957" and input_dim != 957:
        raise SystemExit(f"expected 957 inputs, got {input_dim}")
    if args.input_kind in {"raw191", "raw191-wide", "grid191"} and input_dim != 191:
        raise SystemExit(f"expected 191 inputs, got {input_dim}")
    hidden = args.hidden or (
        [1024, 512] if args.input_kind == "raw191-wide" else [512, 512]
    )
    if args.input_kind == "grid191":
        model = GridStudent(input_dim, hidden).to(args.device)
    else:
        model = StudentMLP(input_dim, hidden).to(args.device)
    if args.init_student is not None:
        source = torch.load(
            args.init_student, map_location="cpu", weights_only=False
        )
        for key, expected in (
            ("input_dim", input_dim),
            ("hidden", hidden),
            ("input_kind", args.input_kind),
        ):
            if source.get(key) != expected:
                raise SystemExit(
                    f"{args.init_student}: incompatible {key} "
                    f"({source.get(key)!r} != {expected!r})"
                )
        model.load_state_dict(source["state_dict"])
        print(f"[init] fine-tuning from {args.init_student}")

    action_counts = np.bincount(fit["actions"], minlength=2).astype(np.float64)
    class_weights = action_counts.sum() / (2.0 * np.maximum(action_counts, 1.0))
    print(f"[data] class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=args.device),
        label_smoothing=0.02,
        reduction="none",
    )
    sample_weights = np.ones(len(fit["actions"]), dtype=np.float32)
    if not args.no_level_balance:
        level_counts = {
            str(level): int((fit["levels"] == level).sum())
            for level in set(str(value) for value in fit["levels"])
        }
        level_overrides: dict[str, float] = {}
        for spec in args.level_weight:
            level, separator, value = spec.partition("=")
            if not separator or not level or not value:
                raise SystemExit(f"invalid --level-weight {spec!r}; expected LEVEL=WEIGHT")
            try:
                level_overrides[level] = float(value)
            except ValueError as error:
                raise SystemExit(f"invalid --level-weight {spec!r}") from error
            if level not in level_counts:
                print(f"[warn] --level-weight {level} did not match any training level")
        raw_weights = np.asarray(
            [
                level_overrides.get(str(level), 1.0)
                / max(level_counts[str(level)], 1)
                for level in fit["levels"]
            ],
            dtype=np.float32,
        )
        sample_weights = raw_weights / raw_weights.mean()
        print(
            "[data] level-balanced sample weights: "
            + ", ".join(
                f"{level}={sample_weights[fit['levels'] == level][0]:.2f}"
                for level in sorted(level_counts)
            )
        )
        if level_overrides:
            print(
                "[data] level weight overrides: "
                + ", ".join(
                    f"{level}={value:g}"
                    for level, value in sorted(level_overrides.items())
                )
            )
    sample_weight_tensor = torch.as_tensor(
        sample_weights, dtype=torch.float32, device=args.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    fit_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(fit["features"], dtype=torch.float32),
            torch.as_tensor(fit["actions"], dtype=torch.long),
            torch.arange(len(fit["actions"]), dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    validation_x = torch.as_tensor(
        validation["features"], dtype=torch.float32, device=args.device
    )
    validation_y = torch.as_tensor(
        validation["actions"], dtype=torch.long, device=args.device
    )

    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch_x, batch_y, batch_indices in fit_loader:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            batch_indices = batch_indices.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            per_sample = criterion(logits, batch_y)
            weights = sample_weight_tensor[batch_indices]
            loss = (per_sample * weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total += len(batch_y)
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_x)
            validation_loss = nn.functional.cross_entropy(
                validation_logits, validation_y
            ).item()
            validation_accuracy = (
                (validation_logits.argmax(dim=1) == validation_y)
                .float()
                .mean()
                .item()
            )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "val_loss": validation_loss,
            "val_accuracy": validation_accuracy,
        }
        history.append(row)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"[epoch {epoch:03d}] train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_acc={validation_accuracy:.4f}"
            )
        if stale >= args.patience and epoch >= 50:
            print(f"[early-stop] no improvement for {args.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {
        "train": evaluate(model, fit, args.device),
        "validation": evaluate(model, validation, args.device),
        "holdout": evaluate(model, holdout, args.device),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden": hidden,
        "input_kind": args.input_kind,
        "levels": sorted(set(str(value) for value in dataset["levels"])),
        "metrics": metrics,
        "data_summary": summary,
        "seed": args.seed,
        "init_student": (
            str(args.init_student) if args.init_student is not None else None
        ),
    }
    torch.save(checkpoint, args.output_dir / "student.pt")
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "input_dim": input_dim,
                "hidden": hidden,
                "input_kind": args.input_kind,
                "levels": checkpoint["levels"],
                "metrics": metrics,
                "data_summary": summary,
                "seed": args.seed,
                "init_student": (
                    str(args.init_student)
                    if args.init_student is not None
                    else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
