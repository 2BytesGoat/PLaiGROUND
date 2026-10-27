"""Train a masked semantic-grid autoencoder on the 754-value observations.

The encoder sees a 7x7x15 semantic cell grid with random cells masked. The
decoder reconstructs all 15 channels. The 19 exact player/route values are
never passed through the autoencoder; they are concatenated after the latent
vector for downstream heads.

This script supports a clean held-out-level evaluation:

    python src/24_train_semantic_autoencoder.py --holdout-levels 1-11
    python src/24_train_semantic_autoencoder.py --holdout-levels ""

The first run trains on the other 15 levels and validates on 1-11. The second
trains the final encoder on all levels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import obs_spec
from wrappers.semantic_autoencoder import (
    BINARY_FEATURES,
    CELL_FEATURES,
    GRID_SIZE,
    GRID_VALUES,
    TAIL_VALUES,
    SemanticGridAutoencoder,
)

REPO_DIR = Path(__file__).resolve().parent.parent
LEVELS = [f"1-{index}" for index in range(1, 17)]


def parse_holdout(value: str) -> set[int]:
    if not value.strip():
        return set()
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in LEVELS:
            raise ValueError(f"unknown level {item}")
        result.add(int(item.split("-")[1]) - 1)
    return result


def load_semantic_grids(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return unique semantic grids, tails, level masks, and source metadata."""
    seen: dict[bytes, int] = {}
    grids: list[np.ndarray] = []
    tails: list[np.ndarray] = []
    levels: list[set[int]] = []
    sources = 0
    rows = 0
    for path in sorted(data_root.glob("**/dataset/*.npz")):
        try:
            payload = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            continue
        if "obs" not in payload.files:
            continue
        obs = payload["obs"]
        if obs.ndim != 2 or obs.shape[1] != obs_spec.MULTI_HOT_OBSERVATION_SIZE:
            continue
        source_levels = (
            payload["level"]
            if "level" in payload.files
            else np.zeros(len(obs), dtype=np.int64)
        )
        source_levels = np.asarray(source_levels, dtype=np.int64)
        sources += 1
        rows += len(obs)
        for index, row in enumerate(obs):
            grid = np.ascontiguousarray(
                row[:GRID_VALUES].reshape(GRID_SIZE, GRID_SIZE, CELL_FEATURES)
            )
            key = grid.tobytes()
            level = int(source_levels[index])
            existing = seen.get(key)
            if existing is None:
                seen[key] = len(grids)
                grids.append(grid)
                tails.append(row[GRID_VALUES:].copy())
                levels.append({level})
            else:
                levels[existing].add(level)
        print(f"loaded {path}: {len(obs):,} rows", flush=True)
    if not grids:
        raise FileNotFoundError(f"no 754-value datasets under {data_root}")
    metadata = {"sources": sources, "rows": rows, "unique": len(grids)}
    return (
        np.stack(grids),
        np.stack(tails),
        np.asarray([sorted(item) for item in levels], dtype=object),
        metadata,
    )


def make_loader(
    grids: np.ndarray,
    tails: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    # The model only uses the grid, but keeping the tail here keeps the state
    # layout identical to the game observation and makes future heads easy.
    observations = np.zeros(
        (len(grids), obs_spec.MULTI_HOT_OBSERVATION_SIZE), dtype=np.float32
    )
    observations[:, :GRID_VALUES] = grids.reshape(len(grids), -1)
    observations[:, GRID_VALUES:] = tails
    dataset = TensorDataset(
        torch.from_numpy(observations),
        torch.from_numpy(
            np.moveaxis(grids, 3, 1).astype(np.float32)
        ),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def class_pos_weight(grids: np.ndarray) -> torch.Tensor:
    """Sqrt inverse-frequency positive weights for the 12 binary channels."""
    flat = grids.reshape(-1, CELL_FEATURES)[:, :BINARY_FEATURES]
    pos = flat.sum(axis=0)
    neg = len(flat) - pos
    weights = np.sqrt(np.maximum(neg, 1.0) / np.maximum(pos, 1.0))
    return torch.from_numpy(np.clip(weights, 0.5, 10.0).astype(np.float32))


def loss_and_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    # reconstruction: [B, 15, 7, 7]; target: same
    cell_weight = (1.0 + 2.0 * mask[:, :, :, 0]).unsqueeze(1)  # [B, 1, 7, 7]
    binary = F.binary_cross_entropy_with_logits(
        reconstruction[:, :BINARY_FEATURES],
        target[:, :BINARY_FEATURES],
        pos_weight=(
            pos_weight.view(1, -1, 1, 1) if pos_weight is not None else None
        ),
        reduction="none",
    )
    continuous = F.mse_loss(
        reconstruction[:, BINARY_FEATURES:],
        target[:, BINARY_FEATURES:],
        reduction="none",
    )
    loss = (
        (binary * cell_weight).mean()
        + (continuous * cell_weight).mean()
    )

    with torch.no_grad():
        predicted = (reconstruction[:, :BINARY_FEATURES] > 0).float()
        truth = (target[:, :BINARY_FEATURES] > 0.5).float()
        bit_accuracy = float((predicted == truth).float().mean().item())
        exact_grid = float(
            (predicted == truth)
            .reshape(len(target), -1)
            .all(dim=1)
            .float()
            .mean()
            .item()
        )
        continuous_mae = float(
            (reconstruction[:, BINARY_FEATURES:] - target[:, BINARY_FEATURES:])
            .abs()
            .mean()
            .item()
        )
    return loss, {
        "bit_accuracy": bit_accuracy,
        "exact_grid": exact_grid,
        "continuous_mae": continuous_mae,
    }


def evaluate(
    model: SemanticGridAutoencoder,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "bit_accuracy": 0.0, "exact_grid": 0.0, "continuous_mae": 0.0}
    count = 0
    with torch.no_grad():
        for observations, target in loader:
            observations = observations.to(device)
            target = target.to(device)
            reconstruction, _latent, mask = model(observations, mask_prob=0.0)
            loss, metrics = loss_and_metrics(
                reconstruction, target, mask, pos_weight
            )
            batch = len(observations)
            totals["loss"] += float(loss.item()) * batch
            for key, value in metrics.items():
                totals[key] += value * batch
            count += batch
    return {key: value / max(1, count) for key, value in totals.items()}


def train(
    grids: np.ndarray,
    tails: np.ndarray,
    levels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    holdout = parse_holdout(args.holdout_levels)
    if holdout:
        val_mask = np.asarray(
            [bool(set(row) & holdout) for row in levels], dtype=bool
        )
        train_mask = ~val_mask
    else:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(grids))
        val_count = max(1, int(round(len(grids) * args.val_fraction)))
        val_mask = np.zeros(len(grids), dtype=bool)
        val_mask[order[:val_count]] = True
        train_mask = ~val_mask
    if not train_mask.any() or not val_mask.any():
        raise ValueError("empty train or validation split")

    train_loader = make_loader(
        grids[train_mask], tails[train_mask], args.batch_size, True
    )
    val_loader = make_loader(
        grids[val_mask], tails[val_mask], args.batch_size, False
    )
    print(
        f"train grids={int(train_mask.sum()):,} "
        f"val grids={int(val_mask.sum()):,} "
        f"holdout={sorted(holdout)}",
        flush=True,
    )

    model = SemanticGridAutoencoder(
        latent_dim=args.latent_dim,
        conv_channels=tuple(args.conv_channels),
    ).to(device)
    pos_weight = class_pos_weight(grids[train_mask]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    history_path = args.output_dir / "history.csv"
    history_file = history_path.open("w", newline="")
    history_writer = csv.DictWriter(
        history_file,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "bit_accuracy",
            "exact_grid",
            "continuous_mae",
        ],
    )
    history_writer.writeheader()
    best_loss = float("inf")
    best_state: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None = None
    best_metrics: dict[str, float] = {}
    stale = 0
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total = 0
            for observations, target in train_loader:
                observations = observations.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, _latent, mask = model(
                    observations, mask_prob=args.mask_prob
                )
                loss, _metrics = loss_and_metrics(
                    reconstruction, target, mask, pos_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                total_loss += float(loss.item()) * len(observations)
                total += len(observations)
            scheduler.step()
            train_loss = total_loss / max(1, total)
            metrics = evaluate(model, val_loader, pos_weight, device)
            history_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{metrics['loss']:.6f}",
                    "bit_accuracy": f"{metrics['bit_accuracy']:.6f}",
                    "exact_grid": f"{metrics['exact_grid']:.6f}",
                    "continuous_mae": f"{metrics['continuous_mae']:.6f}",
                }
            )
            history_file.flush()
            print(
                f"[mae {epoch:03d}] train={train_loss:.4f} "
                f"val={metrics['loss']:.4f} bit={metrics['bit_accuracy']:.4f} "
                f"exact={metrics['exact_grid']:.4f} "
                f"cont_mae={metrics['continuous_mae']:.4f}",
                flush=True,
            )
            if metrics["loss"] < best_loss - 1e-6:
                best_loss = metrics["loss"]
                best_metrics = metrics
                best_state = (
                    {
                        key: value.detach().cpu().clone()
                        for key, value in model.encoder.state_dict().items()
                    },
                    {
                        key: value.detach().cpu().clone()
                        for key, value in model.decoder.state_dict().items()
                    },
                )
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(
                        f"[mae] early stop at epoch {epoch}; best={best_loss:.4f}",
                        flush=True,
                    )
                    break
    finally:
        history_file.close()

    if best_state is None:
        raise RuntimeError("semantic autoencoder produced no checkpoint")
    encoder_state, decoder_state = best_state
    checkpoint = {
        "encoder": encoder_state,
        "decoder": decoder_state,
        "latent_dim": args.latent_dim,
        "conv_channels": list(args.conv_channels),
        "input_encoding": "semantic_multi_hot",
        "grid_size": GRID_SIZE,
        "cell_features": CELL_FEATURES,
        "mask_prob": args.mask_prob,
        "holdout_levels": sorted(holdout),
        "train_grids": int(train_mask.sum()),
        "val_grids": int(val_mask.sum()),
        "seed": args.seed,
        **best_metrics,
    }
    torch.save(checkpoint, args.output_dir / "autoencoder.pt")
    np.save(args.output_dir / "semantic_grids.npy", grids)
    np.save(args.output_dir / "semantic_tails.npy", tails)
    summary = {
        "autoencoder": str(args.output_dir / "autoencoder.pt"),
        "latent_dim": args.latent_dim,
        "input_encoding": "semantic_multi_hot",
        "holdout_levels": sorted(holdout),
        "train_grids": int(train_mask.sum()),
        "val_grids": int(val_mask.sum()),
        **best_metrics,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=REPO_DIR / "runs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR / "runs" / "legacy_autoencoder" / "semantic_mae64_holdout_1-11_s42",
    )
    parser.add_argument(
        "--holdout-levels",
        default="1-11",
        help='Comma-separated held-out levels, or "" for a random split.',
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--conv-channels", type=int, nargs=2, default=[32, 64])
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"device={device} output={args.output_dir}", flush=True)
    grids, tails, levels, metadata = load_semantic_grids(args.data_root)
    print(json.dumps(metadata, indent=2), flush=True)
    with (args.output_dir / "coverage.json").open("w") as handle:
        json.dump(
            {
                **metadata,
                "level_grid_counts": {
                    LEVELS[level]: int(
                        sum(level in set(level_set) for level_set in levels)
                    )
                    for level in range(len(LEVELS))
                },
            },
            handle,
            indent=2,
        )
    train(
        grids,
        tails,
        levels,
        args,
        device,
    )


if __name__ == "__main__":
    main()
