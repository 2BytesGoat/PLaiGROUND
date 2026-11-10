#!/usr/bin/env python3
"""Create a macOS ``.app`` shim that runs a Godot project from source.

The training launcher writes ``override.cfg`` next to the executable. For a
source-project run the setting has to live in the project root instead, so the
shim also drops a ``project_path`` marker next to its binary; the launcher
follows that marker and writes the override in both places.

Example:
    python tools/make_godot_source_app.py \
        --project ../dragon-jump-remaster/godot \
        --out /tmp/GodotSource.app
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_GODOT = Path("/Applications/Godot4.7.app/Contents/MacOS/Godot")
FALLBACK_GODOT = Path("/Applications/Godot.app/Contents/MacOS/Godot")

INFO_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>{name}</string>
    <key>CFBundleIdentifier</key>
    <string>com.2bytesgoat.godotsource</string>
    <key>CFBundleName</key>
    <string>{name}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
"""


def resolve_godot(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Godot binary not found: {path}")
        return path
    for candidate in (DEFAULT_GODOT, FALLBACK_GODOT):
        if candidate.is_file():
            return candidate
    raise SystemExit("No Godot binary found; pass --godot /path/to/Godot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Project dir (contains project.godot)")
    parser.add_argument("--out", required=True, help="Destination .app bundle")
    parser.add_argument("--name", default="GodotSource", help="Bundle/executable name")
    parser.add_argument("--godot", default=None, help="Godot editor binary")
    parser.add_argument("--force", action="store_true", help="Replace an existing bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    if not (project / "project.godot").is_file():
        raise SystemExit(f"Not a Godot project: {project}")
    out = Path(args.out).expanduser().resolve()
    if out.suffix != ".app":
        raise SystemExit(f"--out must end in .app: {out}")
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)

    godot = resolve_godot(args.godot)
    macos = out / "Contents" / "MacOS"
    macos.mkdir(parents=True)
    executable = macos / args.name
    executable.write_text(
        "#!/bin/sh\n"
        f'exec "{godot}" --path "{project}" "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    (macos / "project_path").write_text(f"{project}\n", encoding="utf-8")
    (out / "Contents" / "Info.plist").write_text(
        INFO_PLIST.format(name=args.name), encoding="utf-8"
    )
    print(f"created {out}")
    print(f"  project: {project}")
    print(f"  godot:   {godot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
