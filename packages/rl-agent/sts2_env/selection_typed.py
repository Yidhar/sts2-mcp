"""Typed card-selection / mutation semantics helper (P0-5).

The bridge can attach a typed selection block to any action carrying
selection/mutation metadata.  In compact action payloads the top-level
``selection`` key is still the legacy UI action string (for example
``"select"`` / ``"confirm"``), so the typed dict is emitted under
``typed_selection`` (or the alias ``selection_typed``).  Full payloads and
nested card payloads may still carry the typed dict under ``selection``::

    {
        "typed_selection": {
            "screen_type": "...",
            "operation_type": "discard | retain | exhaust | remove |
                               transform | upgrade | copy | add | replace |
                               enchant | afflict | unknown",
            "source": "card | event | rest | shop | reward | unknown",
            "source_zone": "hand | draw | discard | exhaust | deck | play | none",
            "destination_zone": "hand | draw | discard | exhaust | deck | none",
            "min_count": 0,
            "max_count": 0,
            "selection_required": true,
            "modifier_id": "...",
            "confidence": "runtime_internal | static_export | text_fallback"
        }
    }

This helper is the canonical Python-side reader.  Consumers MUST prefer
``operation_type`` from this dict over re-deriving from localized
``selection_prompt`` / ``description`` text (per P0-5 spec).  Text-fallback
strings are accepted only when the bridge's typed payload is missing AND
the resulting confidence is explicitly downgraded to ``text_fallback``.
"""

from __future__ import annotations

from typing import Any

from .card_effect_profile import card_effect_operations


_OPERATION_TYPE_VALUES = {
    "discard",
    "retain",
    "exhaust",
    "remove",
    "transform",
    "upgrade",
    "copy",
    "add",
    "replace",
    "enchant",
    "afflict",
    "unknown",
}

_OP_NAME_TO_OPERATION_TYPE: dict[str, str] = {
    "discard_card": "discard",
    "retain_card": "retain",
    "exhaust_card": "exhaust",
    "transform_card": "transform",
    "upgrade_card": "upgrade",
    "copy_card": "copy",
    "add_generated_card": "add",
    "add_modifier": "enchant",
    "add_keyword": "enchant",
    "set_replay": "enchant",
    "modify_cost": "enchant",
    "move_card": "replace",
}

_TEXT_OPERATION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("upgrade", ("upgrade", "smith")),
    ("transform", ("transform", "mutate", "metamorph")),
    ("remove", ("remove", "purge")),
    ("discard", ("discard",)),
    ("exhaust", ("exhaust", "consume", "void")),
    ("retain", ("retain",)),
    ("copy", ("copy", "duplicate")),
    ("add", ("add", "generate", "obtain", "draft")),
    ("enchant", ("enchant", "modifier", "imbue")),
    ("afflict", ("afflict", "curse")),
)


def _safe_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_operation_type(value: Any) -> str:
    text = _safe_text(value)
    if text in _OPERATION_TYPE_VALUES:
        return text
    if text in {"transform_card", "upgrade_card", "exhaust_card", "discard_card", "retain_card", "copy_card"}:
        return _OP_NAME_TO_OPERATION_TYPE[text]
    return ""


def _typed_block(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}

    # Compact bridge actions deliberately keep top-level ``selection`` as the
    # legacy string ("select", "confirm", ...).  Read typed aliases first, then
    # the full-payload ``selection`` dict, then nested card aliases for cards
    # that carry their own runtime selection block.
    for key in ("typed_selection", "selection_typed", "selection"):
        selection = action.get(key)
        if isinstance(selection, dict):
            return selection

    card = action.get("card")
    if isinstance(card, dict):
        for key in ("typed_selection", "selection_typed", "selection"):
            selection = card.get(key)
            if isinstance(selection, dict):
                return selection

    return {}


def _operation_from_card_profile(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    card = action.get("card") if isinstance(action.get("card"), dict) else {}
    operations = card_effect_operations(card)
    for op in operations:
        op_name = _safe_text(op.get("op"))
        mapped = _OP_NAME_TO_OPERATION_TYPE.get(op_name)
        if mapped:
            return mapped
    return ""


def _operation_from_text(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    fragments: list[str] = []
    for key in (
        "selection_prompt",
        "selection_action",
        "selection_semantics",
        "selection",
        "surface",
        "screen_type",
        "label",
        "title",
        "description",
        "action_id",
    ):
        value = action.get(key)
        text = _safe_text(value)
        if text:
            fragments.append(text)
    blob = " | ".join(fragments)
    if not blob:
        return ""
    for op_type, keywords in _TEXT_OPERATION_KEYWORDS:
        for keyword in keywords:
            if keyword in blob:
                return op_type
    return ""


def selection_view(action: Any) -> dict[str, Any]:
    """Return a typed selection view with explicit confidence.

    Resolution order (highest confidence first):
      1. Bridge-supplied typed block (``typed_selection`` /
         ``selection_typed`` / dict-valued ``selection``) — trusted typed
         source.  String-valued legacy ``selection`` is ignored here.
      2. Typed card-effect-profile operations (still typed, just inferred
         on the Python side from ``card_effect_profile.operations``).
      3. Localized text regex fallback — flagged as
         ``confidence="text_fallback"``; consumers MUST treat this as
         diagnostics-only and never let it drive hard safety / heavy
         training weights.
    """
    out: dict[str, Any] = {
        "operation_type": "",
        "screen_type": "",
        "source": "",
        "source_zone": "",
        "destination_zone": "",
        "min_count": 0,
        "max_count": 0,
        "selection_required": False,
        "modifier_id": "",
        "confidence": "none",
    }
    typed = _typed_block(action)
    if typed:
        op_type = _normalize_operation_type(typed.get("operation_type"))
        if op_type:
            out["operation_type"] = op_type
        for key in ("screen_type", "source", "source_zone", "destination_zone", "modifier_id"):
            value = typed.get(key)
            if value:
                out[key] = _safe_text(value)
        for key in ("min_count", "max_count"):
            value = typed.get(key)
            try:
                if value is not None:
                    out[key] = int(value)
            except (TypeError, ValueError):
                pass
        if "selection_required" in typed:
            out["selection_required"] = bool(typed.get("selection_required"))
        confidence = _safe_text(typed.get("confidence"))
        if confidence in {"runtime_internal", "static_export", "text_fallback", "none"}:
            out["confidence"] = confidence
        elif out["operation_type"]:
            out["confidence"] = "runtime_internal"
        if out["operation_type"]:
            return out
        # A bridge/card typed block with an empty or unrecognized
        # ``operation_type`` is metadata only (normal play-card payloads use
        # this shape).  Do not leak its ``runtime_internal`` confidence into
        # diagnostics, otherwise every ordinary play_card would falsely count
        # as a runtime-typed selection action.
        out = {
            "operation_type": "",
            "screen_type": "",
            "source": "",
            "source_zone": "",
            "destination_zone": "",
            "min_count": 0,
            "max_count": 0,
            "selection_required": False,
            "modifier_id": "",
            "confidence": "none",
        }

    # Step 2: typed card profile fallback (still typed, no localized text).
    # Always overwrite confidence to ``static_export`` here — the bridge
    # block (if any) was rejected because its operation_type was unknown,
    # so the resolved op_type is being served by the profile, not by the
    # bridge.  Mislabelling would let downstream consumers treat the
    # profile-derived op as if it had the bridge's higher confidence.
    op_from_profile = _operation_from_card_profile(action)
    if op_from_profile:
        out["operation_type"] = op_from_profile
        out["confidence"] = "static_export"
        return out

    # Step 3: localized text regex — last resort, must be flagged so callers
    # never treat it as hard safety.
    op_from_text = _operation_from_text(action)
    if op_from_text:
        out["operation_type"] = op_from_text
        out["confidence"] = "text_fallback"
    return out


def selection_operation_type(action: Any) -> str:
    """Convenience: return just the resolved operation_type ("" if unknown)."""
    return selection_view(action).get("operation_type", "") or ""


def selection_confidence(action: Any) -> str:
    """Convenience: return the source confidence label."""
    return selection_view(action).get("confidence", "none") or "none"


def is_text_fallback_only(action: Any) -> bool:
    """True when the only signal we have is localized text — caller should
    avoid letting this drive hard safety / heavy training weights."""
    return selection_confidence(action) == "text_fallback"
