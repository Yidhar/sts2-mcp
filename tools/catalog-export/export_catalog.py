#!/usr/bin/env python3
"""Publish the canonical game-data package as a verified offline catalog.

This tool intentionally does not connect to the in-game Bridge. The former
/static/export surface accepted arbitrary output paths on the game main thread
and has been retired. Refresh game-data through the dedicated offline generation
pipeline, then use this command to validate and publish it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.artifacts.path_policy import resolve_artifact_path  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_source(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing canonical manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("game-data/manifest.json must contain a non-empty files array")

    validated: list[dict[str, Any]] = []
    source_resolved = source.resolve()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("Every game-data manifest file entry must be an object")
        relative = Path(str(raw_entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        path = (source / relative).resolve()
        if path == source_resolved or source_resolved not in path.parents:
            raise ValueError(f"Manifest path escapes game-data: {relative}")
        if not path.is_file():
            raise ValueError(f"Missing game-data file: {relative}")
        expected_bytes = int(raw_entry.get("bytes", -1))
        expected_hash = str(raw_entry.get("sha256", "")).lower()
        actual_bytes = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_bytes != expected_bytes or actual_hash != expected_hash:
            raise ValueError(
                f"Manifest mismatch for {relative}: expected {expected_bytes}/{expected_hash}, "
                f"got {actual_bytes}/{actual_hash}"
            )
        validated.append({"path": relative.as_posix(), "bytes": actual_bytes, "sha256": actual_hash})
    return manifest, validated


def validate_destination(source: Path, output: Path) -> None:
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if output_resolved == source_resolved or source_resolved in output_resolved.parents:
        raise ValueError("Output must not be game-data or a directory inside game-data")
    if output_resolved in source_resolved.parents:
        raise ValueError("Output must not be an ancestor of game-data")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("Output already exists and is not an empty directory")
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)


def export_catalog(
    source: Path,
    output: str | os.PathLike[str],
    *,
    artifact_root_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    output = resolve_artifact_path(output, root=artifact_root_path)
    validate_destination(source, output)
    source_manifest, entries = load_and_validate_source(source)

    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=False, exist_ok=False)
        for entry in entries:
            relative = Path(entry["path"])
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
            if sha256_file(destination) != entry["sha256"]:
                raise RuntimeError(f"Post-copy checksum mismatch: {relative}")

        shutil.copyfile(source / "manifest.json", staging / "source-game-data.manifest.json")
        export_manifest = {
            "schema_version": "1.0.0",
            "source_manifest_sha256": sha256_file(source / "manifest.json"),
            "source_schema_version": source_manifest.get("schema_version"),
            "source_commit": source_manifest.get("source_commit"),
            "upstream_sts2_ai_commit": source_manifest.get("upstream_sts2_ai_commit"),
            "files": entries,
        }
        manifest_bytes = (json.dumps(export_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        manifest_temp = staging / ".catalog-export.manifest.json.tmp"
        manifest_temp.write_bytes(manifest_bytes)
        os.replace(manifest_temp, staging / "catalog-export.manifest.json")
        staging.replace(output)
        return export_manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPOSITORY_ROOT / "game-data",
        help="Canonical game-data root (default: repository game-data)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New or empty directory below STS2_ARTIFACT_ROOT",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        output = resolve_artifact_path(args.output)
        result = export_catalog(args.source, output)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"catalog-export: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "output": str(output), "files": len(result["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
