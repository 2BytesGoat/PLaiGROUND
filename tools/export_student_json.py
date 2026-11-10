#!/usr/bin/env python3
"""Export a lean_stick one-hot student to the game's versioned JSON runtime.

The game runtime consumes a flat ReLU MLP plus enough preprocessing metadata to
reproduce the Python ``lean_stick`` pipeline in GDScript. This tool verifies the
NumPy/GDScript-equivalent encoder against the fitted PLaiGROUND preprocessor and
the JSON forward pass against PyTorch before writing the artifact.

Example:
    .venv/bin/python tools/export_student_json.py \
      --checkpoint runs/distill_campaign_student/students/selected_shipping/student.pt \
      --preprocessing runs/ddqn_fresh_1_1_lean_stick/preprocessing.pkl \
      --output ../dragon-jump-remaster/godot/src/resources/ai/student_lean_stick_512.json \
      --golden-data-dir runs/distill_campaign_student/dataset_verified_false_ddqn \
      --golden-output ../dragon-jump-remaster/godot/tests/ai/fixtures/student_lean_stick_512_golden.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from plaiground.models.student import StudentMLP
from plaiground.observations import spec
from plaiground.observations.lean_io import load_lean_preprocessor
from plaiground.observations.lean_stick import LeanStickPreprocessor

FORMAT = "dragon-jump-mlp"
POWERUP_CATEGORIES = (-1, 0, 1, 2, 3)
LEAN_STICK_SIZE = spec.LEAN_STICK_OBSERVATION_SIZE
LEAN_SIZE = spec.LEAN_OBSERVATION_SIZE


def _json_float(value: float) -> float:
    """Shortest decimal that round-trips through float32 (9 significant digits)."""
    return float(f"{float(value):.9g}")


def _json_floats(array: np.ndarray) -> list[float]:
    return [_json_float(value) for value in array.reshape(-1)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_arrays(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise SystemExit("checkpoint has no state_dict")
    required = {
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
        "net.4.weight",
        "net.4.bias",
    }
    missing = required.difference(state)
    if missing:
        raise SystemExit(f"checkpoint is missing state keys: {sorted(missing)}")
    return {
        "w1": state["net.0.weight"].detach().cpu().numpy().astype(np.float32),
        "b1": state["net.0.bias"].detach().cpu().numpy().astype(np.float32),
        "w2": state["net.2.weight"].detach().cpu().numpy().astype(np.float32),
        "b2": state["net.2.bias"].detach().cpu().numpy().astype(np.float32),
        "w3": state["net.4.weight"].detach().cpu().numpy().astype(np.float32),
        "b3": state["net.4.bias"].detach().cpu().numpy().astype(np.float32),
    }


def _feature_step(preprocessor: LeanStickPreprocessor) -> Any:
    pipeline = preprocessor.lean_pipeline
    try:
        return pipeline.named_steps["features"]
    except (AttributeError, KeyError) as error:
        raise SystemExit(
            "preprocessing must be a LeanStickPreprocessor with a fitted "
            "lean feature step"
        ) from error


def preprocessing_metadata(preprocessor: LeanStickPreprocessor) -> dict[str, Any]:
    """Describe the fitted lean_stick -> one-hot transform for the game runtime."""
    features = _feature_step(preprocessor)
    tile_names = getattr(features, "tile_names", None)
    if not tile_names:
        raise SystemExit("the fitted pipeline has no explicit tile_names vocabulary")
    tile_names = [str(name) for name in tile_names]
    tile_ids = list(range(len(tile_names)))
    state_names = [str(name) for name in features.state_names]
    state_count = int(getattr(features, "n_states_", len(state_names) + 1))
    collision_categories = [
        int(value) for value in getattr(features, "collision_categories_", range(spec.TYPE_BITS))
    ]
    flow_scale = float(getattr(features, "flow_scale", 1.0))

    tile_offset = 0
    rotation_offset = tile_offset + spec.GRID_CELLS * len(tile_names)
    phase_offset = rotation_offset + spec.LEAN_ROTATION_COUNT
    player_numeric_offset = phase_offset + spec.LEAN_PHASE_COUNT
    state_offset = player_numeric_offset + 5
    powerup_offset = state_offset + state_count
    collision_numeric_offset = powerup_offset + 3 * len(POWERUP_CATEGORIES)
    collision_type_offset = (
        collision_numeric_offset + spec.RAW_DIRECTIONS * 2
    )
    flow_offset = (
        collision_type_offset
        + spec.RAW_DIRECTIONS * len(collision_categories)
    )
    flags_offset = flow_offset + spec.RAW_FLOW_COUNT
    output_size = flags_offset + 3

    if output_size != int(preprocessor.output_size):
        raise SystemExit(
            f"layout output_size {output_size} does not match fitted pipeline "
            f"output_size {preprocessor.output_size}"
        )

    layout = {
        "tile_offset": tile_offset,
        "tile_cells": spec.GRID_CELLS,
        "tile_categories": len(tile_names),
        "rotation_offset": rotation_offset,
        "rotation_count": spec.LEAN_ROTATION_COUNT,
        "phase_offset": phase_offset,
        "phase_count": spec.LEAN_PHASE_COUNT,
        "player_numeric_offset": player_numeric_offset,
        "player_numeric_count": 5,
        "state_offset": state_offset,
        "state_count": state_count,
        "powerup_offset": powerup_offset,
        "powerup_slots": 3,
        "powerup_categories": len(POWERUP_CATEGORIES),
        "collision_numeric_offset": collision_numeric_offset,
        "collision_numeric_count": spec.RAW_DIRECTIONS * 2,
        "collision_type_offset": collision_type_offset,
        "collision_type_count": spec.RAW_DIRECTIONS * len(collision_categories),
        "flow_offset": flow_offset,
        "flow_count": spec.RAW_FLOW_COUNT,
        "flags_offset": flags_offset,
        "flags_count": 3,
        "output_size": output_size,
    }
    return {
        "kind": "lean_one_hot",
        "flow_scale": flow_scale,
        "tile_names": tile_names,
        "tile_ids": tile_ids,
        "state_names": state_names,
        "state_count": state_count,
        "powerup_categories": list(POWERUP_CATEGORIES),
        "collision_categories": collision_categories,
        "feature_layout": layout,
    }


def _category_index(value: Any, categories: Iterable[int]) -> int:
    values = [int(item) for item in categories]
    target = int(value)
    try:
        return values.index(target)
    except ValueError:
        return len(values) - 1


def encode_lean_stick(
    observations: np.ndarray,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    """NumPy reference for MLPPolicy's lean_stick one-hot encoder."""
    array = np.asarray(observations, dtype=np.float32)
    single = array.ndim == 1
    batch = array.reshape(1, -1) if single else array
    if batch.ndim != 2 or batch.shape[1] != LEAN_STICK_SIZE:
        raise ValueError(
            f"expected (*, {LEAN_STICK_SIZE}) lean_stick observations; "
            f"got {array.shape}"
        )

    tile_names = [str(name) for name in preprocessing["tile_names"]]
    tile_ids = [int(value) for value in preprocessing["tile_ids"]]
    if len(tile_ids) != len(tile_names):
        raise ValueError("tile_ids and tile_names have different lengths")
    tile_index = {tile_id: index for index, tile_id in enumerate(tile_ids)}
    state_count = int(preprocessing["state_count"])
    powerup_categories = [int(value) for value in preprocessing["powerup_categories"]]
    collision_categories = [
        int(value) for value in preprocessing["collision_categories"]
    ]
    flow_scale = float(preprocessing.get("flow_scale", 1.0))
    layout = preprocessing["feature_layout"]
    count = int(batch.shape[0])
    output = np.zeros((count, int(layout["output_size"])), dtype=np.float32)

    tile_offset = int(layout["tile_offset"])
    for cell in range(spec.GRID_CELLS):
        ids = batch[:, cell].astype(np.int64)
        indices = np.asarray(
            [tile_index.get(int(value), 0) for value in ids], dtype=np.int64
        )
        output[np.arange(count), tile_offset + cell * len(tile_ids) + indices] = 1.0

    rotation_offset = int(layout["rotation_offset"])
    output[:, rotation_offset:rotation_offset + spec.LEAN_ROTATION_COUNT] = batch[
        :, spec.LEAN_ROTATION_OFFSET:spec.LEAN_PHASE_OFFSET
    ]
    phase_offset = int(layout["phase_offset"])
    output[:, phase_offset:phase_offset + spec.LEAN_PHASE_COUNT] = batch[
        :, spec.LEAN_PHASE_OFFSET:spec.LEAN_PLAYER_OFFSET
    ]

    numeric_offset = int(layout["player_numeric_offset"])
    player = spec.LEAN_PLAYER_OFFSET
    output[:, numeric_offset:numeric_offset + 5] = batch[:, [
        player + 0,
        player + 1,
        player + 2,
        player + 4,
        player + 8,
    ]]

    state_offset = int(layout["state_offset"])
    state_indices = np.asarray(
        [
            int(value) if 0 <= int(value) < state_count - 1 else state_count - 1
            for value in batch[:, player + 3]
        ],
        dtype=np.int64,
    )
    output[np.arange(count), state_offset + state_indices] = 1.0

    powerup_offset = int(layout["powerup_offset"])
    for slot in range(3):
        indices = np.asarray(
            [
                _category_index(value, powerup_categories)
                for value in batch[:, player + 5 + slot]
            ],
            dtype=np.int64,
        )
        width = len(powerup_categories)
        output[
            np.arange(count), powerup_offset + slot * width + indices
        ] = 1.0

    collision_numeric_offset = int(layout["collision_numeric_offset"])
    collision = spec.LEAN_COLLISION_OFFSET
    for direction in range(spec.RAW_DIRECTIONS):
        base = collision + direction * spec.RAW_COLLISION_FLOATS
        output[
            :, collision_numeric_offset + direction * 2
        ] = batch[:, base]
        output[
            :, collision_numeric_offset + direction * 2 + 1
        ] = batch[:, base + 1]

    collision_type_offset = int(layout["collision_type_offset"])
    width = len(collision_categories)
    for direction in range(spec.RAW_DIRECTIONS):
        base = collision + direction * spec.RAW_COLLISION_FLOATS
        indices = np.asarray(
            [
                _category_index(value, collision_categories)
                for value in batch[:, base + 2]
            ],
            dtype=np.int64,
        )
        output[
            np.arange(count), collision_type_offset + direction * width + indices
        ] = 1.0

    flow_offset = int(layout["flow_offset"])
    output[:, flow_offset:flow_offset + spec.RAW_FLOW_COUNT] = (
        batch[:, spec.LEAN_FLOW_OFFSET:LEAN_SIZE] * flow_scale
    )

    flags_offset = int(layout["flags_offset"])
    output[:, flags_offset:flags_offset + 3] = batch[:, LEAN_SIZE:LEAN_STICK_SIZE]
    return output[0] if single else output


def json_forward(
    inputs: np.ndarray,
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    hidden1 = np.maximum(
        inputs @ weights["w1"].T + weights["b1"],
        0.0,
    )
    hidden2 = np.maximum(
        hidden1 @ weights["w2"].T + weights["b2"],
        0.0,
    )
    return hidden2 @ weights["w3"].T + weights["b3"]


def _load_observations(data_dir: Path) -> list[tuple[str, np.ndarray]]:
    index_path = data_dir / "index.json"
    entries: list[tuple[Path, str]] = []
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for item in index:
            relative = item if isinstance(item, str) else str(item["path"])
            entries.append((data_dir / relative / "transitions.npz", relative))
    else:
        entries = [
            (path, path.parent.name)
            for path in sorted(data_dir.glob("**/transitions.npz"))
        ]
    result: list[tuple[str, np.ndarray]] = []
    for path, name in entries:
        if not path.exists():
            raise SystemExit(f"missing dataset file: {path}")
        with np.load(path, allow_pickle=False) as data:
            if "obs_raw" not in data:
                continue
            result.append((name, np.asarray(data["obs_raw"], dtype=np.float32)))
    if not result:
        raise SystemExit(f"no obs_raw arrays under {data_dir}")
    return result


def _golden_case(data_dir: Path) -> tuple[str, np.ndarray]:
    best_name = ""
    best_row = np.zeros(LEAN_STICK_SIZE, dtype=np.float32)
    best_score = -1
    for name, rows in _load_observations(data_dir):
        if rows.size == 0:
            continue
        scores = np.count_nonzero(rows, axis=1)
        index = int(np.argmax(scores))
        score = int(scores[index])
        if score > best_score:
            best_score = score
            best_name = name
            best_row = rows[index].copy()
    return best_name, best_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/distill_campaign_student/students/selected_shipping/student.pt"
        ),
    )
    parser.add_argument(
        "--preprocessing",
        type=Path,
        default=Path("runs/ddqn_fresh_1_1_lean_stick/preprocessing.pkl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument(
        "--reassert-held-jump",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reassert-held-jump-false-level",
        action="append",
        default=[],
        metavar="LEVEL",
        help="Level where the exported policy needs edge-triggered jump input.",
    )
    parser.add_argument(
        "--auto-start-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start the run after reset (the student was trained this way).",
    )
    parser.add_argument("--verify-samples", type=int, default=32)
    parser.add_argument("--verify-seed", type=int, default=7)
    parser.add_argument("--golden-data-dir", type=Path, default=None)
    parser.add_argument("--golden-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    input_kind = str(checkpoint.get("input_kind", ""))
    if input_kind != "onehot957":
        raise SystemExit(
            f"{args.checkpoint} is {input_kind!r}; this exporter needs onehot957"
        )
    arrays = _checkpoint_arrays(checkpoint)
    input_dim = int(checkpoint["input_dim"])
    hidden = [int(value) for value in checkpoint["hidden"]]
    if input_dim != 957:
        raise SystemExit(f"expected a 957-input student, got {input_dim}")
    if len(hidden) != 2:
        raise SystemExit(f"game runtime needs two hidden layers, got {hidden}")
    if arrays["w1"].shape != (hidden[0], input_dim):
        raise SystemExit(f"w1 shape mismatch: {arrays['w1'].shape}")
    if arrays["w2"].shape != (hidden[1], hidden[0]):
        raise SystemExit(f"w2 shape mismatch: {arrays['w2'].shape}")
    if arrays["w3"].shape != (2, hidden[1]):
        raise SystemExit(f"w3 shape mismatch: {arrays['w3'].shape}")

    loaded = load_lean_preprocessor(args.preprocessing)
    if not isinstance(loaded, LeanStickPreprocessor):
        loaded = LeanStickPreprocessor(loaded)
    preprocessor = loaded
    if int(preprocessor.output_size) != input_dim:
        raise SystemExit(
            f"preprocessing output {preprocessor.output_size} != checkpoint "
            f"input {input_dim}"
        )
    metadata = preprocessing_metadata(preprocessor)
    layout = metadata["feature_layout"]
    if int(layout["output_size"]) != input_dim:
        raise SystemExit(
            f"preprocessing layout {layout['output_size']} != input {input_dim}"
        )

    # 1. Encoder parity against the fitted Python pipeline.
    verification_raw: list[np.ndarray] = []
    if args.golden_data_dir is not None:
        verification_raw.extend(rows for _name, rows in _load_observations(args.golden_data_dir))
    rng = np.random.default_rng(args.verify_seed)
    synthetic = rng.standard_normal((args.verify_samples, LEAN_STICK_SIZE)).astype(
        np.float32
    )
    synthetic[:, : spec.GRID_CELLS] = rng.integers(
        0, len(metadata["tile_names"]), size=(args.verify_samples, spec.GRID_CELLS)
    )
    synthetic[:, spec.LEAN_PLAYER_OFFSET + 3] = rng.integers(
        0, metadata["state_count"], size=args.verify_samples
    )
    synthetic[:, spec.LEAN_PLAYER_OFFSET + 5: spec.LEAN_PLAYER_OFFSET + 8] = (
        rng.integers(-1, 4, size=(args.verify_samples, 3))
    )
    collision_end = (
        spec.LEAN_COLLISION_OFFSET
        + spec.RAW_DIRECTIONS * spec.RAW_COLLISION_FLOATS
    )
    synthetic[:, spec.LEAN_COLLISION_OFFSET + 2:collision_end:3] = rng.integers(
        0,
        len(metadata["collision_categories"]),
        size=(args.verify_samples, spec.RAW_DIRECTIONS),
    )
    verification_raw.append(synthetic)
    all_raw = np.concatenate(verification_raw, axis=0)
    expected = np.asarray(preprocessor.transform(all_raw), dtype=np.float32)
    actual = np.asarray(encode_lean_stick(all_raw, metadata), dtype=np.float32)
    encoder_max_abs = float(np.max(np.abs(expected - actual)))
    encoder_match = float(np.mean(expected == actual))
    if encoder_max_abs > 1e-6 or encoder_match < 1.0:
        raise SystemExit(
            "encoder verification failed: "
            f"max_abs={encoder_max_abs:.6g} exact_match={encoder_match:.6f}"
        )

    # 2. JSON forward parity against PyTorch after float32-safe formatting.
    weights = {key: np.asarray(_json_floats(value), dtype=np.float32).reshape(value.shape)
               for key, value in arrays.items()}
    for key, value in arrays.items():
        if not np.array_equal(weights[key], value):
            raise SystemExit(f"JSON float formatting lost precision in {key}")
    model = StudentMLP(input_dim, hidden)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    sample_count = min(int(args.verify_samples), len(all_raw))
    model_inputs = actual[:sample_count]
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(model_inputs)).numpy()
    exported_logits = json_forward(model_inputs, weights)
    forward_max_abs = float(np.max(np.abs(torch_logits - exported_logits)))
    argmax_match = float(
        (torch_logits.argmax(axis=1) == exported_logits.argmax(axis=1)).mean()
    )
    if forward_max_abs > 1e-4 or argmax_match < 1.0:
        raise SystemExit(
            "forward verification failed: "
            f"max_abs={forward_max_abs:.6g} argmax_match={argmax_match:.6f}"
        )

    model_name = args.model_name or args.checkpoint.stem
    document = {
        "format": FORMAT,
        "spec_version": spec.SPEC_VERSION_LEAN_STICK,
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": args.checkpoint.name,
        "source_checkpoint_sha256": _sha256(args.checkpoint),
        "obs_encoding": spec.ENCODING_LEAN_STICK,
        "observation_size": LEAN_STICK_SIZE,
        "input_size": input_dim,
        "hidden_sizes": hidden,
        "activation": "relu",
        "output_size": 2,
        "action_names": ["idle", "jump"],
        "action_repeat": int(args.action_repeat),
        "level_conditioning": "none",
        "trained_levels": checkpoint.get("levels", []),
        "missing_levels": [],
        "reassert_held_jump": bool(args.reassert_held_jump),
        "reassert_held_jump_false_levels": list(
            args.reassert_held_jump_false_level
        ),
        "auto_start_run": bool(args.auto_start_run),
        "training": checkpoint.get("metrics", {}),
        "preprocessing": metadata,
        "weights": {key: _json_floats(value) for key, value in arrays.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, separators=(",", ":")),
        encoding="utf-8",
    )

    if args.golden_output is not None:
        if args.golden_data_dir is None:
            raise SystemExit("--golden-output needs --golden-data-dir")
        source_name, raw = _golden_case(args.golden_data_dir)
        model_input = np.asarray(encode_lean_stick(raw, metadata), dtype=np.float32)
        logits = json_forward(model_input[None, :], weights)[0]
        golden = {
            "source": source_name,
            "observation": _json_floats(raw),
            "model_input": _json_floats(model_input),
            "logits": _json_floats(logits),
            "action": int(logits.argmax()),
        }
        args.golden_output.parent.mkdir(parents=True, exist_ok=True)
        args.golden_output.write_text(
            json.dumps(golden, indent=2),
            encoding="utf-8",
        )
        print(
            f"[export] golden case from {source_name}: "
            f"action={golden['action']} logits={logits.tolist()}",
            flush=True,
        )

    print(
        f"[export] wrote {args.output} "
        f"({args.output.stat().st_size} bytes, obs={LEAN_STICK_SIZE}, "
        f"input={input_dim}, hidden={hidden}, "
        f"encoder_max_abs={encoder_max_abs:.3g}, "
        f"forward_max_abs={forward_max_abs:.3g}, "
        f"argmax_match={argmax_match:.3f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
