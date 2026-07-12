"""Verify every game-data file against the committed manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "game-data"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    entries = manifest.get("files")
    if not isinstance(entries, list):
        print("manifest files must be an array")
        return 1
    expected: set[str] = set()
    for entry in entries:
        relative = str(entry.get("path") or "") if isinstance(entry, dict) else ""
        if not relative or relative in expected:
            errors.append(f"invalid or duplicate manifest path: {relative!r}")
            continue
        expected.add(relative)
        path = (DATA_ROOT / relative).resolve()
        try:
            path.relative_to(DATA_ROOT.resolve())
        except ValueError:
            errors.append(f"manifest path escapes game-data: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        if path.stat().st_size != entry["bytes"]:
            errors.append(f"size mismatch: {relative}")
        if digest(path) != entry["sha256"]:
            errors.append(f"hash mismatch: {relative}")

    actual = {
        path.relative_to(DATA_ROOT).as_posix()
        for folder in (DATA_ROOT / "raw", DATA_ROOT / "generated")
        if folder.exists()
        for path in folder.rglob("*")
        if path.is_file()
    }
    for extra in sorted(actual - expected):
        errors.append(f"unmanifested file: {extra}")
    for missing in sorted(expected - actual):
        if f"missing: {missing}" not in errors:
            errors.append(f"manifest entry has no file: {missing}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"game-data ok: {len(entries)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
