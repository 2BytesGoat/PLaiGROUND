"""Generate 7x7 tile-grid windows by sliding over the campaign levels.

The legacy 58-value observation is a 7x7 terrain window around the player plus
nine scalar tail values. The autoencoder only needs the 7x7 grid, so the static
campaign levels can provide much more coverage than walking an agent over the
same few routes.

This script parses the committed CampaignLevelData codes, reproduces the
game's border fill and cell-symbol stacking, and saves:

- every unique static window;
- the unique real windows from the saved 58-value agent datasets;
- their union;
- the real grids that are not covered by any static window (a clean holdout).

Usage:
    python src/23_generate_level_windows.py
    python src/23_generate_level_windows.py --center-mode non_wall
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
GAME_DIR = REPO_DIR.parent / "dragon-jump-remaster"
DEFAULT_LEVEL_DIR = GAME_DIR / "godot" / "src" / "resources" / "level_data"
DEFAULT_OUTPUT = REPO_DIR / "runs" / "legacy_autoencoder" / "static_windows_s42"
DEFAULT_DATA_ROOT = REPO_DIR / "runs"

# TileRegistry.get_symbols() insertion order. Keep this in lockstep with
# godot/src/level/tile_registry.gd; the legacy observation stores this index.
SYMBOLS = "EWPYQRBKIOJSDGM"
SYMBOL_INDEX = {symbol: index for index, symbol in enumerate(SYMBOLS)}
VOID_INDEX = 16
GRID_SIZE = 7

RE_LEVEL_ID = re.compile(r'^level_id = "([^"]*)"', re.M)
RE_CODE = re.compile(r'^code = "([^"]*)"', re.M)


def parse_level_code(code: str) -> tuple[list[list[set[str]]], int]:
    """Decode a level code into rows of cell-symbol sets.

    The game parser treats a run like ``WM3`` as three cells carrying both W
    and M; a run like ``E7`` as seven empty cells. Counts are decimal and may
    span multiple digits.
    """
    rows: list[list[set[str]]] = []
    width = 0
    for encoded_row in code.split("|"):
        row: list[set[str]] = []
        index = 0
        while index < len(encoded_row):
            symbols: list[str] = []
            while index < len(encoded_row) and encoded_row[index] in SYMBOL_INDEX:
                symbols.append(encoded_row[index])
                index += 1
            digits: list[str] = []
            while index < len(encoded_row) and encoded_row[index].isdigit():
                digits.append(encoded_row[index])
                index += 1
            if not symbols:
                continue
            count = int("".join(digits)) if digits else 1
            # Empty is the absence of a layer cell, not a symbol in
            # populated_cells. Keeping E here would turn a border-filled
            # empty cell into a stacked E+W cell (-1) instead of a wall.
            cell_symbols = {symbol for symbol in symbols if symbol != "E"}
            row.extend(cell_symbols for _ in range(count))
        rows.append(row)
        width = max(width, len(row))

    for row in rows:
        row.extend(set() for _ in range(width - len(row)))

    # Ice is an over-wall object: the game writes a wall into the terrain layer
    # under it, so a legacy cell containing ice becomes a stacked W+I cell.
    for row in rows:
        for symbols in row:
            if "I" in symbols:
                symbols.add("W")

    # _fill_rectangle_with_walls: scan inward from both edges until the first
    # terrain cell, adding walls to the empty edge cells.
    for row in rows:
        x = 0
        while x < width and "W" not in row[x]:
            row[x].add("W")
            x += 1
        x = width - 1
        while x >= 0 and "W" not in row[x]:
            row[x].add("W")
            x -= 1

    return rows, width


def cell_value(symbols: set[str]) -> int:
    if not symbols:
        return 0
    if len(symbols) == 1:
        return SYMBOL_INDEX[next(iter(symbols))]
    # get_cell_symbol_index looks up the concatenated symbol string, which is
    # not in the registry when a cell stacks two symbols; the observation gets
    # -1 in that case.
    return VOID_INDEX


def level_to_values(code: str) -> tuple[list[list[int]], int]:
    rows, width = parse_level_code(code)
    return [[cell_value(symbols) for symbols in row] for row in rows], width


def windows_for_level(
    values: list[list[int]],
    width: int,
    cells: list[list[set[str]]],
    center_mode: str,
) -> set[tuple[int, ...]]:
    windows: set[tuple[int, ...]] = set()
    height = len(values)
    for y in range(height):
        for x in range(width):
            if center_mode == "non_wall" and "W" in cells[y][x]:
                continue
            cells_out: list[int] = []
            for dy in range(-3, 4):
                yy = y + dy
                for dx in range(-3, 4):
                    xx = x + dx
                    cells_out.append(
                        values[yy][xx]
                        if 0 <= yy < height and 0 <= xx < width
                        else 0
                    )
            windows.add(tuple(cells_out))
    return windows


def read_levels(level_dir: Path) -> dict[str, str]:
    levels: dict[str, str] = {}
    for path in sorted(level_dir.glob("*.tres")):
        if path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        level_id = RE_LEVEL_ID.search(text)
        code = RE_CODE.search(text)
        if level_id and code:
            levels[level_id.group(1)] = code.group(1)
    if not levels:
        raise FileNotFoundError(f"no campaign level resources under {level_dir}")
    return levels


def real_windows(data_root: Path) -> tuple[np.ndarray, dict[str, int]]:
    arrays: list[np.ndarray] = []
    sources = 0
    for path in sorted(data_root.glob("**/dataset/*.npz")):
        try:
            payload = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            continue
        if "obs" not in payload.files:
            continue
        obs = payload["obs"]
        if obs.ndim != 2 or obs.shape[1] != 58:
            continue
        grid = np.rint(obs[:, :49]).astype(np.int16)
        grid = np.where(grid < 0, VOID_INDEX, grid).astype(np.uint8)
        arrays.append(grid)
        sources += 1
    if not arrays:
        raise FileNotFoundError(f"no 58-value observation datasets under {data_root}")
    real = np.unique(np.concatenate(arrays, axis=0), axis=0)
    return real, {"sources": sources, "unique": int(len(real))}


def array_keys(array: np.ndarray) -> set[bytes]:
    return {row.tobytes() for row in array}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-dir", type=Path, default=DEFAULT_LEVEL_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--center-mode",
        choices=("all", "non_wall"),
        default="all",
        help="Include all cell centers or only centers not inside a wall.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    levels = read_levels(args.level_dir)
    static_rows: list[np.ndarray] = []
    level_summary: dict[str, dict[str, int]] = {}
    for level_id, code in sorted(
        levels.items(), key=lambda item: int(item[0].split("-")[1])
    ):
        values, width = level_to_values(code)
        cells, _ = parse_level_code(code)
        windows = windows_for_level(values, width, cells, args.center_mode)
        level_summary[level_id] = {
            "width": width,
            "height": len(values),
            "windows": len(windows),
        }
        static_rows.append(
            np.asarray(list(windows), dtype=np.uint8)
        )
        print(
            f"{level_id}: {width}x{len(values)} cells, "
            f"{len(windows):,} unique windows",
            flush=True,
        )

    static = np.unique(np.concatenate(static_rows, axis=0), axis=0)
    real, real_meta = real_windows(args.data_root)
    real_keys = array_keys(real)
    static_keys = array_keys(static)
    holdout_mask = np.asarray(
        [row.tobytes() not in static_keys for row in real], dtype=bool
    )
    holdout = real[holdout_mask]
    combined = np.unique(np.concatenate((static, real), axis=0), axis=0)

    np.save(args.output_dir / "static_grids.npy", static)
    np.save(args.output_dir / "real_grids.npy", real)
    np.save(args.output_dir / "combined_grids.npy", combined)
    np.save(args.output_dir / "real_holdout_grids.npy", holdout)

    coverage = {
        "level_dir": str(args.level_dir),
        "center_mode": args.center_mode,
        "levels": level_summary,
        "static_unique": int(len(static)),
        "real_unique": int(len(real)),
        "real_covered_by_static": int(len(real_keys & static_keys)),
        "real_holdout": int(len(holdout)),
        "combined_unique": int(len(combined)),
        "real_sources": real_meta,
    }
    with (args.output_dir / "coverage.json").open("w") as handle:
        json.dump(coverage, handle, indent=2)
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == "__main__":
    main()
