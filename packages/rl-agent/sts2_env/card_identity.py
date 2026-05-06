"""Stable card-instance identity helper (P0-6).

The bridge is being extended to expose a per-instance ``instance_uuid``
that survives transform / replay / copy / triggered-action lifecycles.
Until that lands, this helper exposes the canonical Python-side reader
with explicit confidence so consumers (lifecycle aux targets, future-
world heads) can refuse to make hard claims about card-destination
probabilities when only weak identifiers are available.

Resolution order (highest confidence first):

1. ``runtime_internal``  — bridge-supplied stable instance UUID.  Keys
   checked: ``instance_uuid``, ``combat_uuid``, ``uuid``, ``uid``,
   ``instance_id``, ``card_instance_id``.
2. ``static_export``     — card definition id (``id``).  Survives across
   instances of the same card definition; OK as a coarse cluster id but
   NOT a per-instance handle (two copies of Strike collide).
3. ``text_fallback``     — localized ``title`` / ``name`` string.  Worst
   confidence; e.g. transformed cards lose their original title and
   matching breaks.  Callers SHOULD treat ``text_fallback`` matches as
   diagnostics-only and never let them drive lifecycle aux Bernoulli
   targets.

The helper is the canonical reader.  ``aux_targets.compute_future_lifecycle_targets``
and ``card_lifecycle_tokens`` are wired through this so a card without a
typed instance UUID does not get a false-confidence destination prediction.
"""

from __future__ import annotations

from typing import Any


_INSTANCE_UUID_KEYS = (
    "instance_uuid",
    "combat_uuid",
    "uuid",
    "uid",
    "instance_id",
    "card_instance_id",
)


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def card_identity(card: Any) -> dict[str, Any]:
    """Return a typed identity dict for ``card``.

    Always returns a dict with keys ``key``, ``confidence``, ``source``::

        {"key": "uuid:...", "confidence": "runtime_internal", "source": "instance_uuid"}
        {"key": "id:CARD.STRIKE", "confidence": "static_export", "source": "id"}
        {"key": "title:Strike", "confidence": "text_fallback", "source": "title"}
        {"key": "", "confidence": "none", "source": "none"}

    The ``key`` is namespaced so ``instance_uuid:abc`` cannot accidentally
    compare equal to ``id:abc`` even though they share the raw string.
    """
    if not isinstance(card, dict):
        return {"key": "", "confidence": "none", "source": "none"}
    for source_name in _INSTANCE_UUID_KEYS:
        value = _safe_text(card.get(source_name))
        if value:
            return {
                "key": f"uuid:{value}",
                "confidence": "runtime_internal",
                "source": source_name,
            }
    card_id = _safe_text(card.get("id"))
    if card_id:
        return {
            "key": f"id:{card_id}",
            "confidence": "static_export",
            "source": "id",
        }
    title = _safe_text(card.get("title") or card.get("name"))
    if title:
        return {
            "key": f"title:{title}",
            "confidence": "text_fallback",
            "source": "title",
        }
    return {"key": "", "confidence": "none", "source": "none"}


def card_identity_key(card: Any) -> str:
    """Convenience: return just the namespaced identity key."""
    return card_identity(card).get("key", "") or ""


def card_identity_confidence(card: Any) -> str:
    return card_identity(card).get("confidence", "none") or "none"


def cards_match(card_a: Any, card_b: Any) -> bool:
    """Strict identity match: True only when both cards have *non-empty*
    identity AND the keys are equal.

    Returns False when either side has empty identity (avoids matching
    two unknown-identity cards as the "same").
    """
    key_a = card_identity_key(card_a)
    key_b = card_identity_key(card_b)
    return bool(key_a and key_b and key_a == key_b)


def cards_match_strict_uuid(card_a: Any, card_b: Any) -> bool:
    """Even stricter match used by hard-confidence consumers (lifecycle
    aux Bernoulli targets): both sides must carry a ``runtime_internal``
    instance UUID.  Returns False if either falls back to ``id`` or
    ``title``.
    """
    id_a = card_identity(card_a)
    id_b = card_identity(card_b)
    return (
        id_a.get("confidence") == "runtime_internal"
        and id_b.get("confidence") == "runtime_internal"
        and id_a.get("key") == id_b.get("key")
    )
