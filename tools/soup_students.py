#!/usr/bin/env python3
"""Average compatible distilled-student checkpoints into one model soup.

All inputs must share the same input kind, input dimension, and hidden sizes.
This is useful when two training runs specialize on different levels: a uniform
or weighted weight-space average can retain both behaviours without adding a
level ID or a second network at deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more student.pt checkpoints, in soup order.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-checkpoint weights; normalized automatically.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.weights is None:
        weights = [1.0] * len(args.checkpoint)
    else:
        if len(args.weights) != len(args.checkpoint):
            raise SystemExit("--weights must match --checkpoint")
        weights = [float(value) for value in args.weights]
    total = sum(weights)
    if total <= 0:
        raise SystemExit("weights must sum to a positive value")
    weights = [value / total for value in weights]

    checkpoints = [torch.load(path, map_location="cpu") for path in args.checkpoint]
    first = checkpoints[0]
    for path, checkpoint in zip(args.checkpoint[1:], checkpoints[1:]):
        for key in ("input_dim", "hidden", "input_kind"):
            if checkpoint.get(key) != first.get(key):
                raise SystemExit(
                    f"{path}: incompatible {key} "
                    f"({checkpoint.get(key)} != {first.get(key)})"
                )

    soup_state = {}
    for key, value in first["state_dict"].items():
        if not torch.is_floating_point(value):
            soup_state[key] = value.clone()
            continue
        merged = torch.zeros_like(value, dtype=torch.float32)
        for weight, checkpoint in zip(weights, checkpoints):
            merged += weight * checkpoint["state_dict"][key].float()
        soup_state[key] = merged.to(value.dtype)

    metadata = {
        "input_dim": first["input_dim"],
        "hidden": first["hidden"],
        "input_kind": first["input_kind"],
        "levels": first.get("levels", []),
        "soup_sources": [
            {"checkpoint": str(path), "weight": weight}
            for path, weight in zip(args.checkpoint, weights)
        ],
        "source_metrics": [
            checkpoint.get("metrics", {}) for checkpoint in checkpoints
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": soup_state,
            "input_dim": metadata["input_dim"],
            "hidden": metadata["hidden"],
            "input_kind": metadata["input_kind"],
            "levels": metadata["levels"],
            "metrics": {},
            "soup_sources": metadata["soup_sources"],
        },
        args.output_dir / "student.pt",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
