"""Resolve canonical shared game data from the repository or an override."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def game_data_root() -> Path:
    """Return the canonical source/override root for versioned game data."""

    configured = os.environ.get("STS2_GAME_DATA_ROOT", "").strip()
    return (
        Path(configured).expanduser().resolve(strict=False)
        if configured
        else (repository_root() / "game-data").resolve(strict=False)
    )


def resolve_generated_game_data_output(value: str | os.PathLike[str] | None) -> Path:
    """Resolve a generator output strictly below ``game-data/generated``.

    Generated source assets are the one intentional write boundary inside a
    checkout.  Keeping this API separate from ``resolve_artifact_path`` makes
    that exception explicit and prevents a CLI path from escaping into other
    source directories.
    """

    generated_root = (game_data_root() / "generated").resolve(strict=False)
    if value is None or os.fspath(value) == "":
        candidate = generated_root
    else:
        candidate = Path(os.fspath(value)).expanduser()
    candidate = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (generated_root / candidate).resolve(strict=False)
    )
    if candidate != generated_root and not candidate.is_relative_to(generated_root):
        raise ValueError(
            f"generated game-data output must stay below {generated_root}, got {candidate}"
        )
    return candidate


def _root_variants(root: Path) -> Iterable[Path]:
    yield root
    yield root / "generated"
    yield root / "raw"


def game_data_search_roots() -> tuple[Path, ...]:
    """Return override roots followed by the canonical repository roots."""

    roots: list[Path] = []
    configured = os.environ.get("STS2_GAME_DATA_ROOT")
    if configured:
        roots.extend(_root_variants(game_data_root()))
    roots.extend(_root_variants((repository_root() / "game-data").resolve(strict=False)))

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve(strict=False)).casefold()
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return tuple(deduped)


def resolve_game_data_file(filename: str, *, required: bool = False) -> Path:
    """Resolve one data file without coupling callers to repository layout."""

    name = Path(filename)
    if name.is_absolute() or len(name.parts) != 1:
        raise ValueError("filename must be a single relative file name")
    candidates = tuple(root / name for root in game_data_search_roots())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if required:
        separator = os.linesep + "  - "
        searched = separator.join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"game data file {filename!r} not found; searched:" + separator + searched
        )
    return candidates[0]


__all__ = [
    "game_data_root",
    "game_data_search_roots",
    "repository_root",
    "resolve_game_data_file",
    "resolve_generated_game_data_output",
]
