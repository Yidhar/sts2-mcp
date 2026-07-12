"""Canonical filesystem boundary for mutable RL runtime artifacts.

Source modules and checked-in configuration are discovered from the package or
repository roots.  Logs, checkpoints, replay buffers, demonstrations, and
other mutable runtime output must instead pass through this module so a launch
from inside the checkout never writes into the source tree by accident.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

ARTIFACT_ROOT_ENV = "STS2_ARTIFACT_ROOT"
DEFAULT_ARTIFACT_DIRNAME = ".sts2-artifacts"
_SAFE_COMPONENT_PUNCTUATION = frozenset("._-+=@")
_WINDOWS_RESERVED_COMPONENTS = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
)


def _expanded_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.fspath(value))).expanduser()


def validate_artifact_component(value: object, *, label: str = "artifact path component") -> str:
    """Validate an operator-controlled single directory/file-name component."""

    text = str(value).strip()
    if not text or len(text) > 160:
        raise ValueError(f"{label} must contain between 1 and 160 characters")
    if text in {".", ".."} or any(not (char.isalnum() or char in _SAFE_COMPONENT_PUNCTUATION) for char in text):
        raise ValueError(f"{label} must be one safe path component, got {value!r}")
    if text.split(".", 1)[0].upper() in _WINDOWS_RESERVED_COMPONENTS:
        raise ValueError(f"{label} is reserved on Windows, got {value!r}")
    return text


def _discover_source_root() -> Path:
    """Find the checkout boundary containing this module."""

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / ".git").exists():
            return parent
        if parent.name == "rl-agent" and parent.parent.name == "packages":
            return parent.parent.parent
    # Installed/editable layouts still need to protect the package containing
    # this module, even when repository metadata is unavailable.
    return module_path.parents[1]


def _canonical_absolute(value: str | os.PathLike[str], *, label: str) -> Path:
    path = _expanded_path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path, got {os.fspath(value)!r}")
    return path.resolve(strict=False)


def _source_boundary(source_root: str | os.PathLike[str] | None) -> Path:
    if source_root is None:
        return _discover_source_root().resolve(strict=False)
    return _canonical_absolute(source_root, label="source_root")


def _reject_source_path(path: Path, *, source_root: str | os.PathLike[str] | None) -> None:
    source = _source_boundary(source_root)
    if path == source or path.is_relative_to(source) or source.is_relative_to(path):
        raise ValueError(
            f"artifact paths must be disjoint from the source checkout {source}, got {path}"
        )


def _reject_filesystem_root(path: Path, *, label: str) -> None:
    anchor = Path(path.anchor).resolve(strict=False) if path.anchor else None
    if anchor is not None and path == anchor:
        raise ValueError(f"{label} must not be a filesystem root, got {path}")


def artifact_root(
    *,
    environ: Mapping[str, str] | None = None,
    home: str | os.PathLike[str] | None = None,
    source_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the absolute root for mutable runtime artifacts.

    ``STS2_ARTIFACT_ROOT`` is deliberately required to be absolute.  Resolving
    a relative environment override against the process working directory
    would reintroduce the exact checkout-write hazard this boundary prevents.
    When the variable is unset, the root is ``~/.sts2-artifacts``.
    """

    values = os.environ if environ is None else environ
    configured = values.get(ARTIFACT_ROOT_ENV, "").strip()
    if configured:
        root = _canonical_absolute(configured, label=ARTIFACT_ROOT_ENV)
        _reject_filesystem_root(root, label=ARTIFACT_ROOT_ENV)
        _reject_source_path(root, source_root=source_root)
        return root

    home_value = home if home is not None else Path.home()
    home_dir = _canonical_absolute(home_value, label="home directory")
    root = (home_dir / DEFAULT_ARTIFACT_DIRNAME).resolve(strict=False)
    _reject_filesystem_root(root, label="artifact root")
    _reject_source_path(root, source_root=source_root)
    return root


def resolve_artifact_path(
    value: str | os.PathLike[str] | None,
    *,
    default: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    source_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a mutable-output path inside the canonical artifact root.

    Absolute values do not bypass the boundary: they must still be descendants
    of ``root``.  Operators who want a different output location must select it
    via ``STS2_ARTIFACT_ROOT`` (or the injected ``root`` argument in embedding
    code). ``default`` is used only when ``value`` is absent or empty.
    """

    selected = default if value is None or os.fspath(value) == "" else value
    if selected is None or os.fspath(selected) == "":
        raise ValueError("artifact path requires a value or a non-empty default")
    path = _expanded_path(selected)
    if not path.is_absolute() and (path.anchor or any(part == ".." for part in path.parts)):
        raise ValueError(f"relative artifact path must stay below the artifact root, got {os.fspath(selected)!r}")

    base = (
        _canonical_absolute(root, label="artifact root")
        if root is not None
        else artifact_root(source_root=source_root)
    )
    _reject_source_path(base, source_root=source_root)
    _reject_filesystem_root(base, label="artifact root")
    candidate = path.resolve(strict=False) if path.is_absolute() else (base / path).resolve(strict=False)
    _reject_source_path(candidate, source_root=source_root)
    if candidate != base and not candidate.is_relative_to(base):
        raise ValueError(f"mutable artifact path must stay below the artifact root {base}, got {candidate}")
    return candidate


def resolve_external_input_path(
    value: str | os.PathLike[str] | None,
    *,
    default: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    source_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a read-only runtime input while keeping it outside source.

    Relative inputs live below ``root`` (the canonical artifact root by
    default).  An explicit absolute input may live elsewhere, but never inside
    the source checkout.  This intentionally separate API prevents mutable
    outputs from acquiring the same broad exception.
    """

    selected = default if value is None or os.fspath(value) == "" else value
    if selected is None or os.fspath(selected) == "":
        raise ValueError("external input path requires a value or a non-empty default")
    path = _expanded_path(selected)
    if not path.is_absolute() and (path.anchor or any(part == ".." for part in path.parts)):
        raise ValueError(f"relative input path must stay below its input root, got {os.fspath(selected)!r}")

    base = (
        _canonical_absolute(root, label="input root")
        if root is not None
        else artifact_root(source_root=source_root)
    )
    _reject_source_path(base, source_root=source_root)
    _reject_filesystem_root(base, label="input root")
    candidate = path.resolve(strict=False) if path.is_absolute() else (base / path).resolve(strict=False)
    _reject_source_path(candidate, source_root=source_root)
    if not path.is_absolute() and candidate != base and not candidate.is_relative_to(base):
        raise ValueError(f"relative input path must stay below its input root {base}, got {candidate}")
    return candidate


__all__ = [
    "ARTIFACT_ROOT_ENV",
    "DEFAULT_ARTIFACT_DIRNAME",
    "artifact_root",
    "resolve_artifact_path",
    "resolve_external_input_path",
    "validate_artifact_component",
]
