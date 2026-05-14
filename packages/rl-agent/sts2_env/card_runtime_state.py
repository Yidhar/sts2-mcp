"""Runtime card-state typed reader (P2-1).

The bridge can attach a per-instance runtime state block to each card in
the legal-action payload. Many card effects modify "this instance for
this combat" rather than the static card definition, so consumers that
look only at ``card.id`` / ``card.title`` will misjudge value.

Fields exposed here are pulled directly from the bridge payload — the
helper does NOT regex localized text for any field that drives a hard
decision. Text fallback is allowed only for diagnostic provenance.

The four field groups (per spec §P2-1):

* identity: ``card_id`` / ``instance_uuid`` / ``upgraded`` / ``misc`` /
  ``magic_number`` / ``base_damage`` / ``base_block`` / ``cost`` /
  ``cost_for_turn``.
* zone: ``hand`` / ``draw`` / ``discard`` / ``exhaust`` / ``limbo`` /
  ``retained`` / ``generated`` / ``temporary``.
* runtime modifiers: ``exhaust_this_combat`` / ``ethereal_this_combat``
  / ``retain_this_turn`` / ``purge_on_use`` / ``autoplay`` /
  ``replay_count`` / ``copied_from`` / ``duplicated`` / ``transformed``
  / ``enchanted`` / ``cost_modified_this_turn`` / ``cost_modified_combat``
  / ``free_to_play_once`` / ``temporary_zero_cost`` /
  ``x_cost_effective_energy``.
* selection effect: ``requires_card_selection`` / ``min_select`` /
  ``max_select`` / ``valid_selection_zone`` / ``selection_target_type``
  / ``selection_can_cancel`` / ``selection_changes_cost`` /
  ``selection_changes_exhaust`` / ``selection_changes_retain`` /
  ``selection_transforms_cards`` / ``selection_duplicates_cards`` /
  ``selection_discards_cards`` / ``selection_exhausts_cards``.

Acceptance criteria (from spec §P2-1):

* ``card_identity_runtime_internal_selected_rate`` not stuck at 0.
* ``selection_runtime_internal_selected_rate`` > 0 when selection cards
  appear in the legal-action set.

The presence-rate counters this module exposes are the diagnostic side
of those acceptance gates.
"""
from __future__ import annotations

from typing import Any

from .card_identity import card_identity, card_identity_confidence


# --------------------------------------------------------------------- field groups

_IDENTITY_FIELDS: tuple[str, ...] = (
    "card_id",
    "instance_uuid",
    "upgraded",
    "misc",
    "magic_number",
    "base_damage",
    "base_block",
    "cost",
    "cost_for_turn",
)

_ZONE_FIELDS: tuple[str, ...] = (
    "hand",
    "draw_pile",
    "discard_pile",
    "exhaust_pile",
    "limbo",
    "retained",
    "generated",
    "temporary",
)

_RUNTIME_MODIFIER_FIELDS: tuple[str, ...] = (
    "exhaust_this_combat",
    "ethereal_this_combat",
    "retain_this_turn",
    "purge_on_use",
    "autoplay",
    "replay_count",
    "copied_from",
    "duplicated",
    "transformed",
    "enchanted",
    "cost_modified_this_turn",
    "cost_modified_combat",
    "free_to_play_once",
    "temporary_zero_cost",
    "x_cost_effective_energy",
)

_SELECTION_EFFECT_FIELDS: tuple[str, ...] = (
    "requires_card_selection",
    "min_select",
    "max_select",
    "valid_selection_zone",
    "selection_target_type",
    "selection_can_cancel",
    "selection_changes_cost",
    "selection_changes_exhaust",
    "selection_changes_retain",
    "selection_transforms_cards",
    "selection_duplicates_cards",
    "selection_discards_cards",
    "selection_exhausts_cards",
)


# Aliases the bridge / older payloads may use for the same field. The
# first key found wins; downstream consumers see the canonical name.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "card_id": ("card_id", "id"),
    "instance_uuid": (
        "instance_uuid",
        "combat_uuid",
        "uuid",
        "uid",
        "instance_id",
        "card_instance_id",
    ),
    "upgraded": ("upgraded", "is_upgraded", "upgrade_level"),
    "magic_number": ("magic_number", "misc", "magic"),
    "base_damage": ("base_damage", "damage_base"),
    "base_block": ("base_block", "block_base"),
    "cost": ("cost", "energy_cost", "canonical_energy_cost"),
    "cost_for_turn": ("cost_for_turn", "current_cost", "effective_cost"),
    "exhaust_this_combat": ("exhaust_this_combat", "exhausted", "exhausts"),
    "ethereal_this_combat": ("ethereal_this_combat", "ethereal", "is_ethereal"),
    "retain_this_turn": ("retain_this_turn", "retained_this_turn", "retain"),
    "x_cost_effective_energy": (
        "x_cost_effective_energy",
        "x_effective_energy",
        "x_energy_remaining",
    ),
    "free_to_play_once": ("free_to_play_once", "free_to_play"),
    "temporary_zero_cost": ("temporary_zero_cost", "temp_zero_cost"),
    "cost_modified_this_turn": ("cost_modified_this_turn", "cost_modified_turn"),
    "cost_modified_combat": ("cost_modified_combat", "cost_modified_combat_flag"),
    "requires_card_selection": (
        "requires_card_selection",
        "requires_selection",
        "selection_required",
    ),
    "min_select": ("min_select", "selection_min", "selection_min_count", "min_count"),
    "max_select": ("max_select", "selection_max", "selection_max_count", "max_count"),
}


def _first_present_value(card: dict[str, Any], canonical: str) -> Any:
    """Look up ``canonical`` on ``card`` honouring the alias chain.

    Returns the first non-None value found. Returns ``None`` when no
    alias is present so downstream presence-rate counters can tell
    "field absent" from "field present but falsy".
    """
    aliases = _FIELD_ALIASES.get(canonical, (canonical,))
    for key in aliases:
        value = card.get(key)
        if value is not None:
            return value
    return None


def _field_present(card: dict[str, Any], canonical: str) -> bool:
    return _first_present_value(card, canonical) is not None


def card_runtime_state(card: Any) -> dict[str, Any]:
    """Return a typed runtime-state view for ``card``.

    Fields that the bridge does not expose are returned as ``None``
    rather than absent so downstream consumers can tell "missing" from
    "explicitly false". Caller decides whether ``None`` should be treated
    as zero or as "skip this aux target".

    The ``confidence`` key reuses the ``card_identity`` resolution
    semantics: ``runtime_internal`` when an instance UUID is present,
    ``static_export`` when only a ``card_id`` is present, ``text_fallback``
    when only a localized title is present, ``none`` otherwise.
    """
    if not isinstance(card, dict):
        return {
            "identity": {f: None for f in _IDENTITY_FIELDS},
            "zone": {f: None for f in _ZONE_FIELDS},
            "modifiers": {f: None for f in _RUNTIME_MODIFIER_FIELDS},
            "selection_effect": {f: None for f in _SELECTION_EFFECT_FIELDS},
            "confidence": "none",
            "presence": {
                "identity_count": 0,
                "zone_count": 0,
                "modifier_count": 0,
                "selection_effect_count": 0,
            },
        }

    identity = {field: _first_present_value(card, field) for field in _IDENTITY_FIELDS}
    zone = {field: _first_present_value(card, field) for field in _ZONE_FIELDS}
    modifiers = {
        field: _first_present_value(card, field) for field in _RUNTIME_MODIFIER_FIELDS
    }
    selection_effect = {
        field: _first_present_value(card, field) for field in _SELECTION_EFFECT_FIELDS
    }
    confidence = card_identity_confidence(card)
    return {
        "identity": identity,
        "zone": zone,
        "modifiers": modifiers,
        "selection_effect": selection_effect,
        "confidence": confidence,
        "presence": {
            "identity_count": sum(1 for v in identity.values() if v is not None),
            "zone_count": sum(1 for v in zone.values() if v is not None),
            "modifier_count": sum(1 for v in modifiers.values() if v is not None),
            "selection_effect_count": sum(
                1 for v in selection_effect.values() if v is not None
            ),
        },
    }


def card_runtime_field_present(card: Any, field: str) -> bool:
    """Return True if ``field`` (canonical or alias) is present on ``card``."""
    if not isinstance(card, dict):
        return False
    if field in _FIELD_ALIASES:
        return _field_present(card, field)
    return card.get(field) is not None


def card_runtime_presence_flags(card: Any) -> dict[str, float]:
    """Return per-field 0.0/1.0 presence flags for diagnostic counters.

    Keys mirror the spec §P2-1 ``card_runtime/*`` TB tags:

    * ``instance_uuid_present``
    * ``modified_cost_present``
    * ``exhaust_flag_present``
    * ``ethereal_flag_present``
    * ``retain_flag_present``
    * ``enchantment_present``
    * ``replay_flag_present``
    * ``selection_effect_present``
    """
    if not isinstance(card, dict):
        return {
            "instance_uuid_present": 0.0,
            "modified_cost_present": 0.0,
            "exhaust_flag_present": 0.0,
            "ethereal_flag_present": 0.0,
            "retain_flag_present": 0.0,
            "enchantment_present": 0.0,
            "replay_flag_present": 0.0,
            "selection_effect_present": 0.0,
        }
    identity = card_identity(card)
    instance_uuid_present = identity.get("source") in {
        "instance_uuid",
        "combat_uuid",
        "uuid",
        "uid",
        "instance_id",
        "card_instance_id",
    }
    cost_for_turn = _first_present_value(card, "cost_for_turn")
    base_cost = _first_present_value(card, "cost")
    modified_cost = (
        cost_for_turn is not None
        and base_cost is not None
        and cost_for_turn != base_cost
    )
    selection_effect_present = any(
        _field_present(card, field) for field in _SELECTION_EFFECT_FIELDS
    )
    return {
        "instance_uuid_present": 1.0 if instance_uuid_present else 0.0,
        "modified_cost_present": 1.0 if modified_cost else 0.0,
        "exhaust_flag_present": 1.0 if _field_present(card, "exhaust_this_combat") else 0.0,
        "ethereal_flag_present": 1.0 if _field_present(card, "ethereal_this_combat") else 0.0,
        "retain_flag_present": 1.0 if _field_present(card, "retain_this_turn") else 0.0,
        "enchantment_present": 1.0 if (
            _field_present(card, "enchanted") or card.get("modifier_summary")
        ) else 0.0,
        "replay_flag_present": 1.0 if _field_present(card, "replay_count") else 0.0,
        "selection_effect_present": 1.0 if selection_effect_present else 0.0,
    }


def card_runtime_confidence(card: Any) -> str:
    """Convenience: return the resolved confidence level
    (``runtime_internal`` / ``static_export`` / ``text_fallback`` / ``none``).
    """
    return card_identity_confidence(card) or "none"
