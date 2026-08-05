"""Read-only localized display names for monitored game entities.

The training dashboard must never maintain a second, hand-written card/relic/
potion dictionary.  These helpers resolve titles from the canonical generated
game-data package, tolerate both transport IDs (``CARD.ANGER``) and compact
IDs (``ANGER``), and fail open to the original identifier when a future game
version introduces an entity that is not in the pinned catalog yet.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

from sts2_rl.game_data import resolve_game_data_file

EntityKind = Literal["card", "relic", "potion"]

_CATALOG_FILE: Final[dict[EntityKind, str]] = {
    "card": "cards.static.generated.json",
    "relic": "relics.static.generated.json",
    "potion": "potions.static.generated.json",
}
_PREFIX: Final[dict[EntityKind, str]] = {
    "card": "CARD.",
    "relic": "RELIC.",
    "potion": "POTION.",
}


def _normalized_identifier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    identifier = value.strip()
    if not identifier:
        return None
    # Bridge display keys occasionally arrive as ``ANGER.title`` or
    # ``CARD.ANGER.title+``.  The plus is presentation state, not identity.
    if identifier.endswith("+"):
        identifier = identifier[:-1]
    if identifier.casefold().endswith(".title"):
        identifier = identifier[:-6]
    return identifier or None


@lru_cache(maxsize=8)
def _load_titles(path_text: str, kind: EntityKind) -> dict[str, str]:
    """Load a bounded alias map from one canonical generated catalog."""

    path = Path(path_text)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    prefix = _PREFIX[kind]
    titles: dict[str, str] = {}
    for catalog_key, raw_record in payload.items():
        if not isinstance(catalog_key, str) or not isinstance(raw_record, Mapping):
            continue
        title = raw_record.get("title")
        record_id = raw_record.get("id")
        if not isinstance(title, str) or not title.strip():
            continue
        identifiers = [catalog_key]
        if isinstance(record_id, str):
            identifiers.append(record_id)
        for identifier in identifiers:
            normalized = _normalized_identifier(identifier)
            if normalized is None:
                continue
            normalized = normalized.upper()
            titles.setdefault(normalized, title.strip())
            if normalized.startswith(prefix):
                titles.setdefault(normalized[len(prefix) :], title.strip())
    return titles


def _titles(kind: EntityKind) -> dict[str, str]:
    path = resolve_game_data_file(_CATALOG_FILE[kind], required=False)
    return _load_titles(str(path.resolve(strict=False)), kind)


def localized_entity_name(
    identifier: object,
    *,
    kind: EntityKind | None = None,
    fallback: str | None = None,
) -> str:
    """Return the canonical Simplified-Chinese title or a safe fallback.

    ``kind`` should be supplied whenever the schema identifies the entity
    family.  Generic lookup is deliberately conservative: an unprefixed ID is
    translated only when it resolves in exactly one family, preventing a
    future suffix collision from displaying the wrong entity name.
    """

    normalized = _normalized_identifier(identifier)
    fallback_text = fallback if fallback is not None else (identifier if isinstance(identifier, str) else "")
    if normalized is None:
        return fallback_text
    lookup = normalized.upper()

    if kind is not None:
        return _titles(kind).get(lookup, fallback_text)

    prefixed_kind = next(
        (candidate_kind for candidate_kind, prefix in _PREFIX.items() if lookup.startswith(prefix)),
        None,
    )
    if prefixed_kind is not None:
        return _titles(prefixed_kind).get(lookup, fallback_text)

    matches = {title for candidate_kind in _CATALOG_FILE if (title := _titles(candidate_kind).get(lookup)) is not None}
    return next(iter(matches)) if len(matches) == 1 else fallback_text


__all__ = ["EntityKind", "localized_entity_name"]
