"""Inventory and compare repository artifact trees without following links.

The externalization map is one-to-one. Artifact-layout inventories reserve any
nested destination namespaces while scanning a parent entry, keeping every byte
in exactly one mapping and source/post-move inventories directly comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_TOOL_PATH = Path(__file__).resolve()
MOVE_MAP_PATH = Path(__file__).with_name("move-map.json")
INVENTORY_SCHEMA_VERSION = "2.0.0"
MOVE_MAP_SCHEMA_VERSION = "1.0.0"
REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
KEY_METADATA_NAMES = frozenset({"metadata.json", "manifest.json", "config.json", "latest.json"})


def _safe_relative(value: object, *, label: str) -> str:
    """Validate one manifest path under both Windows and POSIX path rules."""

    # Import lazily so this file remains directly executable from the checkout.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.artifacts.path_policy import _relative_path  # noqa: PLC0415

    try:
        path = _relative_path(str(value or ""), label=label)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"{label} must be a non-empty confined relative path, got {value!r}")
    return normalized


def _safe_pattern(value: object, *, label: str) -> str:
    """Validate a repository-relative glob without interpreting it as a path."""

    text = str(value or "").replace("\\", "/")
    # Validate a wildcard-free surrogate with the shared cross-platform policy.
    surrogate = text.replace("*", "wildcard").replace("?", "wildcard")
    _safe_relative(surrogate, label=label)
    if "[" in text or "]" in text:
        raise ValueError(f"{label} must not use character-class glob syntax, got {value!r}")
    if any("*" in part or "?" in part for part in text.split("/")[:-1]):
        raise ValueError(f"{label} may use wildcards only in its final component, got {value!r}")
    return text


def load_move_map(path: Path = MOVE_MAP_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MOVE_MAP_SCHEMA_VERSION or not isinstance(payload.get("entries"), list):
        raise ValueError(f"unsupported or malformed move map: {path}")

    normalized_entries: list[dict[str, str]] = []
    for index, raw in enumerate(payload["entries"]):
        if not isinstance(raw, dict):
            raise ValueError(f"move-map entry {index} must be an object")
        source = _safe_relative(raw.get("source"), label=f"move-map source #{index}")
        destination = _safe_relative(raw.get("destination"), label=f"move-map destination #{index}")
        category = str(raw.get("category") or "").strip()
        if not category:
            raise ValueError(f"move-map entry {index} requires a non-empty category")
        normalized_entries.append(
            {"source": source, "destination": destination, "category": category}
        )

    for key in ("source", "destination"):
        values = [entry[key] for entry in normalized_entries]
        folded = [value.casefold() for value in values]
        if len(folded) != len(set(folded)):
            raise ValueError(f"move-map {key} paths must be unique under Windows case rules")

    source_paths = [PurePosixPath(entry["source"]) for entry in normalized_entries]
    for index, path_value in enumerate(source_paths):
        for other in source_paths[index + 1 :]:
            if path_value.is_relative_to(other) or other.is_relative_to(path_value):
                raise ValueError(f"move-map sources must not overlap: {path_value} and {other}")

    residue = payload.get("forbidden_residue_patterns", [])
    if not isinstance(residue, list):
        raise ValueError("move-map forbidden_residue_patterns must be a list")
    normalized_patterns = [
        _safe_pattern(value, label=f"forbidden residue pattern #{index}")
        for index, value in enumerate(residue)
    ]
    return {
        "schema_version": MOVE_MAP_SCHEMA_VERSION,
        "entries": normalized_entries,
        "forbidden_residue_patterns": normalized_patterns,
    }


def move_map_paths(layout: str) -> list[str]:
    if layout not in {"source", "artifact"}:
        raise ValueError(f"unsupported inventory layout: {layout}")
    key = "source" if layout == "source" else "destination"
    return [entry[key] for entry in load_move_map()["entries"]]


def remove_overlapping_paths(paths: list[str]) -> list[str]:
    """Compatibility helper for callers that need a non-overlapping root set."""

    selected: list[PurePosixPath] = []
    for path in sorted(
        (PurePosixPath(_safe_relative(value, label="inventory path")) for value in paths),
        key=lambda item: (len(item.parts), item.as_posix().casefold()),
    ):
        if any(path == parent or path.is_relative_to(parent) for parent in selected):
            continue
        selected.append(path)
    return [path.as_posix() for path in selected]


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & REPARSE_POINT_ATTRIBUTE)


def _entry_kind(metadata: os.stat_result) -> str:
    if stat.S_ISLNK(metadata.st_mode):
        return "symlink"
    if _is_reparse(metadata):
        return "reparse"
    if stat.S_ISREG(metadata.st_mode):
        return "file"
    if stat.S_ISDIR(metadata.st_mode):
        return "directory"
    return "other"


def _record_fingerprint(hasher: Any, *parts: object) -> None:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _canonical_inventory_root(value: Path) -> Path:
    """Resolve an existing directory only after proving its path has no links."""

    expanded = value.expanduser()
    lexical = expanded if expanded.is_absolute() else Path.cwd() / expanded
    lexical = Path(os.path.abspath(lexical))
    anchor = Path(lexical.anchor)
    cursor = anchor
    for part in lexical.parts[1:]:
        cursor /= part
        try:
            metadata = cursor.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"inventory root does not exist: {lexical}") from exc
        if _entry_kind(metadata) in {"symlink", "reparse"}:
            raise ValueError(f"inventory root contains a link or reparse component: {cursor}")
    if not lexical.is_dir():
        raise ValueError(f"inventory root is not a directory: {lexical}")
    return lexical.resolve(strict=True)


def _validate_candidate_parent(path: Path, root: Path) -> None:
    """Reject an existing intermediate link without following it."""

    try:
        relative_parent = path.parent.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"mapped path parent escapes inventory root: {path}") from exc
    cursor = root
    for part in relative_parent.parts:
        cursor /= part
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            # Remaining descendants cannot exist without this parent.
            return
        if _entry_kind(metadata) in {"symlink", "reparse"}:
            raise ValueError(f"mapped path parent contains a link or reparse point: {cursor}")


def _reserved_top_names(entry: dict[str, str], entries: Sequence[dict[str, str]]) -> set[str]:
    parent = PurePosixPath(entry["destination"])
    reserved: set[str] = set()
    for other in entries:
        if other is entry:
            continue
        child = PurePosixPath(other["destination"])
        if child != parent and child.is_relative_to(parent):
            reserved.add(child.relative_to(parent).parts[0].casefold())
    return reserved


def inventory_path(
    path: Path,
    *,
    root: Path,
    hash_limit: int,
    excluded_top_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Inventory one mapped file/directory without following any reparse point."""

    excluded = {name.casefold() for name in excluded_top_names}
    relative_root = _display_path(path, root)
    result: dict[str, Any] = {
        "path": relative_root,
        "exists": False,
        "root_kind": "missing",
        "files": 0,
        "directories": 0,
        "links": 0,
        "other_entries": 0,
        "bytes": 0,
        "largest": [],
        "hashes": [],
        "links_detail": [],
        "errors": [],
    }
    hasher = hashlib.sha256()
    largest: list[tuple[int, str]] = []

    try:
        root_metadata = path.lstat()
    except FileNotFoundError:
        _record_fingerprint(hasher, "missing")
        result["structure_fingerprint_sha256"] = hasher.hexdigest()
        return result
    except OSError as exc:
        result["errors"].append(f"lstat {relative_root}: {type(exc).__name__}: {exc}")
        _record_fingerprint(hasher, "error", type(exc).__name__)
        result["structure_fingerprint_sha256"] = hasher.hexdigest()
        return result

    root_kind = _entry_kind(root_metadata)
    result["exists"] = True
    result["root_kind"] = root_kind
    _record_fingerprint(hasher, "root", root_kind)

    def record_link(child: Path, relative: str, metadata: os.stat_result, kind: str) -> None:
        target: str | None = None
        if kind == "symlink":
            try:
                target = os.readlink(child)
            except OSError as exc:
                result["errors"].append(
                    f"readlink {_display_path(child, root)}: {type(exc).__name__}: {exc}"
                )
        detail = {
            "path": _display_path(child, root),
            "relative_path": relative,
            "kind": kind,
            "target": target,
            "reparse_tag": int(getattr(metadata, "st_reparse_tag", 0)),
            "file_attributes": int(getattr(metadata, "st_file_attributes", 0)),
        }
        result["links"] += 1
        result["links_detail"].append(detail)
        _record_fingerprint(
            hasher,
            kind,
            relative,
            target,
            detail["reparse_tag"],
            detail["file_attributes"],
        )

    def record_file(child: Path, relative: str, metadata: os.stat_result) -> None:
        size = int(metadata.st_size)
        digest: str | None = None
        should_hash = (hash_limit > 0 and size <= hash_limit) or child.name in KEY_METADATA_NAMES
        if should_hash:
            try:
                digest = file_hash(child)
            except OSError as exc:
                result["errors"].append(
                    f"hash {_display_path(child, root)}: {type(exc).__name__}: {exc}"
                )
        result["files"] += 1
        result["bytes"] += size
        display = _display_path(child, root)
        largest.append((size, display))
        if digest is not None:
            result["hashes"].append(
                {"path": display, "relative_path": relative, "bytes": size, "sha256": digest}
            )
        _record_fingerprint(hasher, "file", relative, size, digest)

    def scan_directory(directory: Path, relative_prefix: PurePosixPath | None = None) -> None:
        try:
            with os.scandir(directory) as stream:
                children = sorted(list(stream), key=lambda item: (item.name.casefold(), item.name))
        except OSError as exc:
            result["errors"].append(
                f"scandir {_display_path(directory, root)}: {type(exc).__name__}: {exc}"
            )
            return

        for child_entry in children:
            relative_path = (
                PurePosixPath(child_entry.name)
                if relative_prefix is None
                else relative_prefix / child_entry.name
            )
            relative = relative_path.as_posix()
            if relative_prefix is None and child_entry.name.casefold() in excluded:
                continue
            child = Path(child_entry.path)
            try:
                metadata = child_entry.stat(follow_symlinks=False)
            except OSError as exc:
                result["errors"].append(
                    f"lstat {_display_path(child, root)}: {type(exc).__name__}: {exc}"
                )
                continue
            kind = _entry_kind(metadata)
            if kind in {"symlink", "reparse"}:
                record_link(child, relative, metadata, kind)
            elif kind == "file":
                record_file(child, relative, metadata)
            elif kind == "directory":
                result["directories"] += 1
                _record_fingerprint(hasher, "directory", relative)
                scan_directory(child, relative_path)
            else:
                result["other_entries"] += 1
                result["errors"].append(
                    f"unsupported filesystem entry {_display_path(child, root)} (mode={metadata.st_mode:o})"
                )
                _record_fingerprint(hasher, "other", relative, int(metadata.st_mode))

    if root_kind in {"symlink", "reparse"}:
        record_link(path, ".", root_metadata, root_kind)
        result["errors"].append(f"mapped root must not be a link or reparse point: {relative_root}")
    elif root_kind == "file":
        record_file(path, ".", root_metadata)
    elif root_kind == "directory":
        scan_directory(path)
    else:
        result["other_entries"] = 1
        result["errors"].append(
            f"mapped root has unsupported filesystem type: {relative_root} (mode={root_metadata.st_mode:o})"
        )

    largest.sort(key=lambda item: (-item[0], item[1].casefold(), item[1]))
    result["largest"] = [{"path": name, "bytes": size} for size, name in largest[:20]]
    result["hashes"].sort(key=lambda item: str(item["relative_path"]).casefold())
    result["links_detail"].sort(key=lambda item: str(item["relative_path"]).casefold())
    result["structure_fingerprint_sha256"] = hasher.hexdigest()
    return result


def build_inventory(
    *,
    root: Path,
    layout: str,
    hash_limit: int,
    move_map_path: Path = MOVE_MAP_PATH,
) -> dict[str, Any]:
    if layout not in {"source", "artifact"}:
        raise ValueError(f"unsupported inventory layout: {layout}")
    if hash_limit < 0:
        raise ValueError("hash limit must be non-negative")
    move_map = load_move_map(move_map_path)
    entries = move_map["entries"]
    root = _canonical_inventory_root(root)
    snapshots: list[dict[str, Any]] = []
    for mapping in entries:
        selected = mapping["source"] if layout == "source" else mapping["destination"]
        candidate = root.joinpath(*PurePosixPath(selected).parts)
        # The mapped root itself is inspected with lstat below, but an existing
        # parent symlink/junction must not redirect traversal outside root.
        try:
            _validate_candidate_parent(candidate, root)
        except (OSError, ValueError) as exc:
            snapshot = {
                "path": selected,
                "exists": False,
                "root_kind": "error",
                "files": 0,
                "directories": 0,
                "links": 0,
                "other_entries": 0,
                "bytes": 0,
                "largest": [],
                "hashes": [],
                "links_detail": [],
                "errors": [f"resolve parent {selected}: {type(exc).__name__}: {exc}"],
                "structure_fingerprint_sha256": hashlib.sha256(b"parent-error").hexdigest(),
            }
        else:
            exclusions = _reserved_top_names(mapping, entries) if layout == "artifact" else set()
            snapshot = inventory_path(
                candidate,
                root=root,
                hash_limit=hash_limit,
                excluded_top_names=exclusions,
            )
            snapshot["excluded_destination_names"] = sorted(exclusions)
        snapshot["source_relative_path"] = mapping["source"]
        snapshot["destination_relative_path"] = mapping["destination"]
        snapshot["category"] = mapping["category"]
        snapshots.append(snapshot)

    totals = {
        "mapped_entries": len(snapshots),
        "existing_entries": sum(bool(entry["exists"]) for entry in snapshots),
        "files": sum(int(entry["files"]) for entry in snapshots),
        "directories": sum(int(entry["directories"]) for entry in snapshots),
        "links": sum(int(entry["links"]) for entry in snapshots),
        "other_entries": sum(int(entry["other_entries"]) for entry in snapshots),
        "bytes": sum(int(entry["bytes"]) for entry in snapshots),
        "errors": sum(len(entry["errors"]) for entry in snapshots),
    }
    aggregate = hashlib.sha256()
    for entry in snapshots:
        _record_fingerprint(
            aggregate,
            entry["source_relative_path"],
            entry["destination_relative_path"],
            entry["exists"],
            entry["root_kind"],
            entry["structure_fingerprint_sha256"],
        )
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "layout": layout,
        "hash_limit_bytes": hash_limit,
        "inventory_tool_sha256": file_hash(INVENTORY_TOOL_PATH),
        "move_map_sha256": file_hash(move_map_path),
        "complete": totals["errors"] == 0,
        "totals": totals,
        "aggregate_fingerprint_sha256": aggregate.hexdigest(),
        "entries": snapshots,
    }


def compare_inventories(source: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    if source.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        problems.append("source inventory schema is unsupported")
    if artifact.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        problems.append("artifact inventory schema is unsupported")
    if source.get("layout") != "source":
        problems.append("first inventory must use source layout")
    if artifact.get("layout") != "artifact":
        problems.append("second inventory must use artifact layout")
    if source.get("move_map_sha256") != artifact.get("move_map_sha256"):
        problems.append("inventories were generated from different move maps")
    if source.get("inventory_tool_sha256") != artifact.get("inventory_tool_sha256"):
        problems.append("inventories were generated by different inventory tool revisions")
    if source.get("hash_limit_bytes") != artifact.get("hash_limit_bytes"):
        problems.append("inventories used different content-hash limits")
    if not source.get("complete", False):
        problems.append("source inventory contains filesystem errors")
    if not artifact.get("complete", False):
        problems.append("artifact inventory contains filesystem errors")

    def key(entry: dict[str, Any]) -> tuple[str, str]:
        return str(entry.get("source_relative_path")), str(entry.get("destination_relative_path"))

    source_entries = {key(entry): entry for entry in source.get("entries", [])}
    artifact_entries = {key(entry): entry for entry in artifact.get("entries", [])}
    if len(source_entries) != len(source.get("entries", [])):
        problems.append("source inventory contains duplicate mapping entries")
    if len(artifact_entries) != len(artifact.get("entries", [])):
        problems.append("artifact inventory contains duplicate mapping entries")

    entry_reports: list[dict[str, Any]] = []
    for mapping in sorted(set(source_entries) | set(artifact_entries)):
        before = source_entries.get(mapping)
        after = artifact_entries.get(mapping)
        reasons: list[str] = []
        if before is None:
            reasons.append("mapping is missing from source inventory")
        if after is None:
            reasons.append("mapping is missing from artifact inventory")
        if before is not None and after is not None:
            if before.get("exists"):
                if not after.get("exists"):
                    reasons.append("destination is missing")
                for field in (
                    "root_kind",
                    "files",
                    "directories",
                    "links",
                    "other_entries",
                    "bytes",
                    "structure_fingerprint_sha256",
                ):
                    if before.get(field) != after.get(field):
                        reasons.append(f"{field} differs: {before.get(field)!r} != {after.get(field)!r}")
            else:
                # A destination parent may exist solely to contain another
                # mapping. Reserved child namespaces are excluded above.
                for field in ("files", "directories", "links", "other_entries", "bytes"):
                    if int(after.get(field, 0)) != 0:
                        reasons.append(f"absent source acquired unexpected {field}: {after.get(field)!r}")
            if before.get("errors"):
                reasons.append("source mapping contains inventory errors")
            if after.get("errors"):
                reasons.append("artifact mapping contains inventory errors")
        if reasons:
            problems.extend(f"{mapping[0]} -> {mapping[1]}: {reason}" for reason in reasons)
        entry_reports.append(
            {
                "source_relative_path": mapping[0],
                "destination_relative_path": mapping[1],
                "status": "ok" if not reasons else "mismatch",
                "reasons": reasons,
            }
        )

    return {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if not problems else "mismatch",
        "source_inventory_root": source.get("root"),
        "artifact_inventory_root": artifact.get("root"),
        "inventory_tool_sha256": source.get("inventory_tool_sha256"),
        "move_map_sha256": source.get("move_map_sha256"),
        "problems": problems,
        "entries": entry_reports,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.expanduser().resolve(strict=False)
    source = ROOT.resolve(strict=False)
    if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise ValueError(f"inventory output must be disjoint from source checkout {source}, got {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_inventory_output(path: Path, *, inventory_root: Path, layout: str) -> None:
    """Keep evidence out of the payload it is measuring."""

    resolved = path.expanduser().resolve(strict=False)
    root = inventory_root.resolve(strict=False)
    if layout == "source":
        if resolved == root or resolved.is_relative_to(root) or root.is_relative_to(resolved):
            raise ValueError(f"source inventory output must be disjoint from inventory root {root}, got {resolved}")
        return
    for destination in move_map_paths("artifact"):
        mapped = root.joinpath(*PurePosixPath(destination).parts).resolve(strict=False)
        if resolved == mapped or resolved.is_relative_to(mapped) or mapped.is_relative_to(resolved):
            raise ValueError(
                f"artifact inventory output must not overlap mapped destination {mapped}, got {resolved}"
            )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"inventory must be a JSON object: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--layout", choices=("source", "artifact"), default="source")
    parser.add_argument("--hash-limit-mib", type=int, default=8)
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("SOURCE_INVENTORY", "ARTIFACT_INVENTORY"),
        default=None,
    )
    args = parser.parse_args(argv)

    if args.compare is not None:
        source = _load_json(args.compare[0])
        artifact = _load_json(args.compare[1])
        result = compare_inventories(source, artifact)
        if args.output is not None:
            output = args.output.expanduser().resolve(strict=False)
            inputs = {path.expanduser().resolve(strict=False) for path in args.compare}
            if output in inputs:
                raise ValueError("comparison output must not overwrite either input inventory")
            _write_json(args.output, result)
        print(json.dumps({"status": result["status"], "problems": len(result["problems"])}))
        return 0 if result["status"] == "ok" else 1

    if args.output is None:
        parser.error("--output is required when generating an inventory")
    if args.hash_limit_mib < 0:
        parser.error("--hash-limit-mib must be non-negative")
    result = build_inventory(
        root=args.root,
        layout=args.layout,
        hash_limit=args.hash_limit_mib * 1024 * 1024,
    )
    _validate_inventory_output(
        args.output,
        inventory_root=Path(str(result["root"])),
        layout=args.layout,
    )
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": "ok" if result["complete"] else "error",
                "files": result["totals"]["files"],
                "bytes": result["totals"]["bytes"],
                "links": result["totals"]["links"],
                "errors": result["totals"]["errors"],
            }
        )
    )
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
