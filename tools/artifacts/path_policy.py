"""Shared path policy for repository tooling and runtime artifact outputs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT_ENV = "STS2_ARTIFACT_ROOT"
DEFAULT_ARTIFACT_DIRNAME = ".sts2-artifacts"


def _absolute(value: str | os.PathLike[str], *, label: str) -> Path:
    path = Path(os.path.expandvars(os.fspath(value))).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path, got {os.fspath(value)!r}")
    return path.resolve(strict=False)


def _relative_path(value: str | os.PathLike[str], *, label: str) -> Path:
    """Return a confined relative path under both POSIX and Windows rules.

    Tooling is exercised on both platforms and migration manifests may be
    produced on one platform then consumed on the other.  ``Path`` alone does
    not recognise ``C:\\...`` as absolute on POSIX, while ``PurePosixPath``
    does not recognise drive-relative Windows paths.  Reject both grammars
    before joining with a trusted root.
    """

    raw = os.fspath(value)
    if raw == "":
        raise ValueError(f"{label} must not be empty")
    normalized = raw.replace("\\", "/")
    # PurePath deliberately collapses repeated separators and ``.``.  Check
    # the original components first so a manifest cannot smuggle an ambiguous
    # spelling past validation and be interpreted differently by another OS.
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError(f"{label} must stay below its root, got {raw!r}")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(raw)
    if (
        posix.is_absolute()
        or bool(posix.root)
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
    ):
        raise ValueError(f"{label} must be relative, got {raw!r}")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError(f"{label} must stay below its root, got {raw!r}")
    return Path(*posix.parts)


def _validate_disjoint(path: Path, *, source_root: Path = REPOSITORY_ROOT, label: str) -> None:
    source = source_root.resolve(strict=False)
    if path == source or path.is_relative_to(source) or source.is_relative_to(path):
        raise ValueError(f"{label} must be disjoint from the source checkout {source}, got {path}")
    anchor = Path(path.anchor).resolve(strict=False) if path.anchor else None
    if anchor is not None and path == anchor:
        raise ValueError(f"{label} must not be a filesystem root, got {path}")


def artifact_root(
    *,
    environ: Mapping[str, str] | None = None,
    home: str | os.PathLike[str] | None = None,
    source_root: Path = REPOSITORY_ROOT,
) -> Path:
    values = os.environ if environ is None else environ
    configured = values.get(ARTIFACT_ROOT_ENV, "").strip()
    if configured:
        root = _absolute(configured, label=ARTIFACT_ROOT_ENV)
    else:
        home_path = _absolute(home if home is not None else Path.home(), label="home directory")
        root = (home_path / DEFAULT_ARTIFACT_DIRNAME).resolve(strict=False)
    _validate_disjoint(root, source_root=source_root, label="artifact root")
    return root


def resolve_artifact_path(
    value: str | os.PathLike[str] | None,
    *,
    default: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    source_root: Path = REPOSITORY_ROOT,
) -> Path:
    selected = default if value is None or os.fspath(value) == "" else value
    if selected is None or os.fspath(selected) == "":
        raise ValueError("artifact path requires a value or non-empty default")
    base = _absolute(root, label="artifact root") if root is not None else artifact_root(source_root=source_root)
    _validate_disjoint(base, source_root=source_root, label="artifact root")
    expanded = os.path.expandvars(os.fspath(selected))
    path = Path(expanded).expanduser()
    candidate = (
        path.resolve(strict=False)
        if path.is_absolute()
        else (base / _relative_path(expanded, label="artifact path")).resolve(strict=False)
    )
    _validate_disjoint(candidate, source_root=source_root, label="artifact path")
    if candidate != base and not candidate.is_relative_to(base):
        raise ValueError(f"artifact path must stay below {base}, got {candidate}")
    return candidate


def resolve_external_input_path(
    value: str | os.PathLike[str] | None,
    *,
    default: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    source_root: Path = REPOSITORY_ROOT,
) -> Path:
    selected = default if value is None or os.fspath(value) == "" else value
    if selected is None or os.fspath(selected) == "":
        raise ValueError("external input path requires a value or non-empty default")
    expanded = os.path.expandvars(os.fspath(selected))
    path = Path(expanded).expanduser()
    base = _absolute(root, label="input root") if root is not None else artifact_root(source_root=source_root)
    _validate_disjoint(base, source_root=source_root, label="input root")
    candidate = (
        path.resolve(strict=False)
        if path.is_absolute()
        else (base / _relative_path(expanded, label="external input")).resolve(strict=False)
    )
    _validate_disjoint(candidate, source_root=source_root, label="external input")
    if not path.is_absolute() and candidate != base and not candidate.is_relative_to(base):
        raise ValueError(f"relative input path must stay below {base}, got {candidate}")
    return candidate


__all__ = [
    "ARTIFACT_ROOT_ENV",
    "DEFAULT_ARTIFACT_DIRNAME",
    "REPOSITORY_ROOT",
    "artifact_root",
    "resolve_artifact_path",
    "resolve_external_input_path",
]
