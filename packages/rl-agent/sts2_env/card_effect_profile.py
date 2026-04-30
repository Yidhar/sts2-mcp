"""Typed card-effect profile helpers.

The bridge can attach ``card_effect_profile`` to every live card payload.  The
profile is generated from internal card ids/source facts/curated overrides and
contains typed operations such as ``gain_energy``, ``upgrade_card`` or
``modify_cost``.  These helpers keep that path independent from localized card
text regex so observation/action encoders can share the same semantics.
"""

from __future__ import annotations

from typing import Any


_PROFILE_FIELDS = (
    "card_effect_profile",
    "cardEffectProfile",
    "effect_profile",
    "effectProfile",
    "profile",
)


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    text = _safe_text(value).lower()
    return text not in {"", "0", "false", "none", "null", "no"}


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_safe_text(item) for item in value if _safe_text(item)]
    if isinstance(value, tuple):
        return [_safe_text(item) for item in value if _safe_text(item)]
    if isinstance(value, set):
        return [_safe_text(item) for item in value if _safe_text(item)]
    text = _safe_text(value)
    if not text:
        return []
    # Keep simple strings as one token; split obvious comma/pipe delimited
    # payloads that may come from compact JSON/debug dumps.
    if "," in text or "|" in text or ";" in text:
        out: list[str] = []
        for chunk in text.replace("|", ",").replace(";", ",").split(","):
            part = chunk.strip()
            if part:
                out.append(part)
        return out
    return [text]


def _operation_name(op: dict[str, Any]) -> str:
    return _safe_text(op.get("op") or op.get("operation") or op.get("type")).lower()


def _is_upgraded(card: dict[str, Any] | None) -> bool:
    if not isinstance(card, dict):
        return False
    for key in ("is_upgraded", "upgraded"):
        if key in card and _truthy(card.get(key)):
            return True
    for key in ("upgrade_level", "current_upgrade_level", "level"):
        if _float(card.get(key), 0.0) > 0.0:
            return True
    title = _safe_text(card.get("title") or card.get("name"))
    return title.endswith("+")


def card_effect_profile(card: dict[str, Any] | None) -> dict[str, Any]:
    """Return the best typed profile object attached to a card-like payload."""
    if not isinstance(card, dict):
        return {}
    for field in _PROFILE_FIELDS:
        profile = card.get(field)
        if isinstance(profile, dict):
            return profile
    # Some compact/debug paths may carry operations directly on the card.
    operations = card.get("operations")
    if isinstance(operations, list):
        return {
            "operations": operations,
            "semantic_tags": card.get("semantic_tags"),
            "training_tags": card.get("training_tags"),
            "source_facts": card.get("source_facts"),
        }
    return {}


def card_derived_view(card: dict[str, Any] | None) -> dict[str, Any]:
    """Return the structured ``derived_view`` block from a card's profile.

    The derived view exposes seven field groups (cost / lifecycle /
    hand_mutation / pile_mutation / combat_effect / mechanism_effect / source)
    derived from internal card ids, source facts and curated overrides.  It is
    JSON-stable and intended for observation/action token consumers that do not
    want to walk the raw operation list.
    """
    if not isinstance(card, dict):
        return {}
    profile = card_effect_profile(card)
    view = profile.get("derived_view") if isinstance(profile, dict) else None
    if isinstance(view, dict):
        return view
    direct = card.get("derived_view")
    if isinstance(direct, dict):
        return direct
    return {}


def _view_section(card: dict[str, Any] | None, name: str) -> dict[str, Any]:
    view = card_derived_view(card)
    section = view.get(name) if isinstance(view, dict) else None
    return section if isinstance(section, dict) else {}


def card_cost_view(card: dict[str, Any] | None) -> dict[str, Any]:
    return _view_section(card, "cost")


def card_lifecycle_view(card: dict[str, Any] | None) -> dict[str, Any]:
    return _view_section(card, "lifecycle")


def card_hand_mutation_view(card: dict[str, Any] | None) -> dict[str, Any]:
    return _view_section(card, "hand_mutation")


def card_pile_mutation_view(card: dict[str, Any] | None) -> dict[str, Any]:
    return _view_section(card, "pile_mutation")


def card_combat_effect_view(card: dict[str, Any] | None) -> dict[str, Any]:
    return _view_section(card, "combat_effect")


def card_mechanism_effect_view(card: dict[str, Any] | None) -> dict[str, Any]:
    return _view_section(card, "mechanism_effect")


def card_source_view(card: dict[str, Any] | None) -> dict[str, Any]:
    return _view_section(card, "source")


def card_effect_operations(card: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return typed operation dicts from ``card_effect_profile`` if present."""
    profile = card_effect_profile(card)
    operations = profile.get("operations") if isinstance(profile, dict) else None
    if not isinstance(operations, list):
        return []
    return [op for op in operations if isinstance(op, dict) and _operation_name(op)]


def card_effect_operation_names(card: dict[str, Any] | None, *, max_count: int = 8) -> list[str]:
    """Compact stable operation names for semantic text/signatures."""
    names: list[str] = []
    for op in card_effect_operations(card):
        name = _operation_name(op)
        if name and name not in names:
            names.append(name)
        if len(names) >= max_count:
            break
    return names


def _default_profile_semantics() -> dict[str, float]:
    return {
        "typed_ops_count": 0.0,
        "typed_modifies_hand": 0.0,
        "typed_upgrade_hand": 0.0,
        "typed_exhaust_cards": 0.0,
        "typed_discard_cards": 0.0,
        "typed_transform_cards": 0.0,
        "typed_copy_cards": 0.0,
        "typed_modify_cost": 0.0,
        "typed_set_replay": 0.0,
        "typed_retain_cards": 0.0,
        "typed_add_modifier": 0.0,
        "typed_add_keyword": 0.0,
        "typed_add_generated_card": 0.0,
        "typed_draw_cards": 0.0,
        "typed_draw_amount": 0.0,
        "typed_gain_energy": 0.0,
        "typed_gain_energy_amount": 0.0,
        "typed_hp_loss": 0.0,
        "typed_apply_power": 0.0,
        "typed_no_draw": 0.0,
        "typed_future_penalty": 0.0,
        "typed_requires_followup": 0.0,
        "typed_strategic_skip_if_no_followup": 0.0,
        "typed_not_x_cost_filter": 0.0,
        "typed_x_cost_filter": 0.0,
        "typed_hand_context_dependency": 0.0,
        "typed_discard_context_dependency": 0.0,
        "typed_exhaust_context_dependency": 0.0,
        "typed_draw_context_dependency": 0.0,
        "typed_deck_context_dependency": 0.0,
        "typed_consumes_future_resource": 0.0,
        "typed_once_or_exhaust_self": 0.0,
        "typed_selection_required": 0.0,
        "typed_all_scope": 0.0,
        "typed_card_rule_modifier": 0.0,
        "typed_card_state_mutation": 0.0,
        "typed_hand_context_needed": 0.0,
    }


def _flag(result: dict[str, float], key: str) -> None:
    result[key] = 1.0


def _add(result: dict[str, float], key: str, value: Any) -> None:
    val = _float(value, 0.0)
    if val != 0.0:
        result[key] = result.get(key, 0.0) + val


def _zone_flags(result: dict[str, float], *values: Any) -> None:
    zones = " ".join(part.lower() for value in values for part in _as_string_list(value))
    if not zones:
        return
    if "hand" in zones:
        _flag(result, "typed_hand_context_dependency")
    if "discard" in zones:
        _flag(result, "typed_discard_context_dependency")
    if "exhaust" in zones or "consume" in zones or "void" in zones:
        _flag(result, "typed_exhaust_context_dependency")
    if "draw" in zones:
        _flag(result, "typed_draw_context_dependency")
    if "deck" in zones or "library" in zones:
        _flag(result, "typed_deck_context_dependency")


def _profile_tags(profile: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    for field in ("semantic_tags", "training_tags", "tags"):
        for tag in _as_string_list(profile.get(field)):
            tags.add(tag.lower())
    return tags


def _apply_source_facts(result: dict[str, float], profile: dict[str, Any]) -> None:
    facts = profile.get("source_facts")
    if not isinstance(facts, dict):
        return
    if _truthy(facts.get("uses_hand_pile")):
        _flag(result, "typed_hand_context_dependency")
    if _truthy(facts.get("uses_discard_pile")):
        _flag(result, "typed_discard_context_dependency")
    if _truthy(facts.get("uses_exhaust_pile")):
        _flag(result, "typed_exhaust_context_dependency")
    if _truthy(facts.get("uses_draw_pile")):
        _flag(result, "typed_draw_context_dependency")
    commands = " ".join(_as_string_list(facts.get("commands"))).lower()
    powers = " ".join(_as_string_list(facts.get("powers"))).lower()
    if "nodrawpower" in powers or "no_draw" in powers or "no draw" in powers:
        _flag(result, "typed_no_draw")
    if any(token in commands for token in ("energycost", "basecost", "setcost", "cost")):
        _flag(result, "typed_card_rule_modifier")


def aggregate_card_effect_profile_semantics(card: dict[str, Any] | None) -> dict[str, float]:
    """Aggregate typed card operations into stable numeric semantics.

    This is intentionally broad: it exposes whether a card depends on hand /
    draw / discard / exhaust context, whether it mutates card state, and whether
    using it only makes sense with follow-up actions (energy refund/cost
    reducers/future penalties).  The exact model shape can consume a subset of
    these keys without changing this source of truth.
    """
    result = _default_profile_semantics()
    if not isinstance(card, dict):
        return result

    profile = card_effect_profile(card)
    operations = card_effect_operations(card)
    result["typed_ops_count"] = float(len(operations))

    if isinstance(profile, dict):
        tags = _profile_tags(profile)
        if any(tag in tags for tag in ("hand_mutation", "modify_cost", "upgrade_card", "set_replay", "card_rule_modifier")):
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_card_state_mutation")
        if "hand_mutation" in tags:
            _flag(result, "typed_hand_context_dependency")
        if "energy_gen" in tags or "gain_energy" in tags:
            _flag(result, "typed_gain_energy")
        if "exhaust" in tags:
            _flag(result, "typed_once_or_exhaust_self")
            _flag(result, "typed_consumes_future_resource")
        if "retain" in tags:
            _flag(result, "typed_retain_cards")
        if "card_rule_modifier" in tags:
            _flag(result, "typed_card_rule_modifier")
        _apply_source_facts(result, profile)

    upgraded = _is_upgraded(card)
    for op in operations:
        name = _operation_name(op)
        _zone_flags(
            result,
            op.get("source_zone"),
            op.get("target_zone"),
            op.get("zone"),
            op.get("from_zone"),
            op.get("to_zone"),
        )
        selection = _safe_text(op.get("selection")).lower()
        scope = _safe_text(op.get("scope")).lower()
        if selection in {"choice", "choose", "select", "target"}:
            _flag(result, "typed_selection_required")
        if scope in {"all", "aoe", "hand"}:
            _flag(result, "typed_all_scope")

        filters = {item.lower() for item in _as_string_list(op.get("target_filter"))}
        filters.update(item.lower() for item in _as_string_list(op.get("filter")))
        if filters.intersection({"not_x_cost", "not_cost_x", "non_x_cost"}):
            _flag(result, "typed_not_x_cost_filter")
        if filters.intersection({"x_cost", "cost_x", "costs_x", "is_x_cost"}):
            _flag(result, "typed_x_cost_filter")

        future_rule = _safe_text(op.get("future_rule")).lower()
        power_id = _safe_text(op.get("power_id") or op.get("power")).lower()
        timing = _safe_text(op.get("timing")).lower()
        if "no_draw" in future_rule or "nodraw" in power_id or "no_draw" in power_id:
            _flag(result, "typed_no_draw")
        if any(token in future_rule for token in ("future", "penalty", "extra_cost", "no_draw")):
            _flag(result, "typed_future_penalty")
            _flag(result, "typed_consumes_future_resource")
        if _truthy(op.get("strategic_skip_if_no_followup")):
            _flag(result, "typed_strategic_skip_if_no_followup")
            _flag(result, "typed_requires_followup")

        if name == "gain_energy":
            _flag(result, "typed_gain_energy")
            energy_key = "upgraded_energy" if upgraded and op.get("upgraded_energy") is not None else "energy"
            _add(result, "typed_gain_energy_amount", op.get(energy_key))
            _add(result, "typed_hp_loss", op.get("hp_loss"))
            # Same-turn energy conversion is only useful when the remaining
            # hand/deck can spend it.  This covers Production/Borrowed Time/
            # Bloodletting-like cards even if the curated flag is absent.
            if timing in {"same_turn_resource", "this_turn", "immediate"} or op.get("energy") is not None:
                _flag(result, "typed_requires_followup")
        elif name == "draw_card":
            _flag(result, "typed_draw_cards")
            _add(result, "typed_draw_amount", op.get("count") if op.get("count") is not None else 1.0)
            _flag(result, "typed_draw_context_dependency")
        elif name == "upgrade_card":
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_upgrade_hand")
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_hand_context_dependency")
        elif name == "exhaust_card":
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_exhaust_cards")
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_consumes_future_resource")
            _flag(result, "typed_hand_context_dependency")
        elif name == "discard_card":
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_discard_cards")
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_discard_context_dependency")
            _flag(result, "typed_hand_context_dependency")
        elif name == "transform_card":
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_transform_cards")
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_hand_context_dependency")
        elif name == "copy_card":
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_copy_cards")
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_hand_context_dependency")
        elif name == "modify_cost":
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_modify_cost")
            _flag(result, "typed_card_rule_modifier")
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_hand_context_dependency")
            # Cost reducers are also follow-up dependent: they need another
            # playable card to convert the setup into value.
            if op.get("set_cost") is not None or op.get("cost_delta") is not None:
                _flag(result, "typed_requires_followup")
        elif name == "add_generated_card":
            _flag(result, "typed_add_generated_card")
            _flag(result, "typed_card_state_mutation")
        elif name == "move_card":
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_modifies_hand")
        elif name == "add_modifier":
            _flag(result, "typed_add_modifier")
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_hand_context_dependency")
        elif name == "add_keyword":
            _flag(result, "typed_add_keyword")
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_hand_context_dependency")
            keyword = " ".join(_as_string_list(op.get("keyword") or op.get("keywords"))).lower()
            if "retain" in keyword:
                _flag(result, "typed_retain_cards")
            if "exhaust" in keyword:
                _flag(result, "typed_exhaust_cards")
            if "ethereal" in keyword or "void" in keyword:
                _flag(result, "typed_consumes_future_resource")
        elif name == "set_replay":
            _flag(result, "typed_set_replay")
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_hand_context_dependency")
        elif name == "retain_card":
            _flag(result, "typed_retain_cards")
            _flag(result, "typed_modifies_hand")
            _flag(result, "typed_card_state_mutation")
            _flag(result, "typed_hand_context_dependency")
        elif name == "apply_power":
            _flag(result, "typed_apply_power")
            if any(token in power_id for token in ("nodraw", "no_draw")):
                _flag(result, "typed_no_draw")
            if future_rule:
                _flag(result, "typed_consumes_future_resource")

    result["typed_hand_context_needed"] = float(
        result.get("typed_hand_context_dependency", 0.0) > 0.0
        or result.get("typed_modifies_hand", 0.0) > 0.0
        or result.get("typed_requires_followup", 0.0) > 0.0
    )
    return result


def compact_card_effect_profile_signature(card: dict[str, Any] | None) -> dict[str, Any]:
    """Small JSON-safe typed summary for action compact/history payloads."""
    sem = aggregate_card_effect_profile_semantics(card)
    names = card_effect_operation_names(card)
    out: dict[str, Any] = {}
    if names:
        out["card_ops"] = names
    bool_keys = (
        "typed_modifies_hand",
        "typed_upgrade_hand",
        "typed_modify_cost",
        "typed_set_replay",
        "typed_retain_cards",
        "typed_exhaust_cards",
        "typed_discard_cards",
        "typed_transform_cards",
        "typed_copy_cards",
        "typed_add_modifier",
        "typed_add_keyword",
        "typed_gain_energy",
        "typed_requires_followup",
        "typed_strategic_skip_if_no_followup",
        "typed_no_draw",
        "typed_future_penalty",
        "typed_not_x_cost_filter",
        "typed_x_cost_filter",
        "typed_hand_context_dependency",
        "typed_discard_context_dependency",
        "typed_exhaust_context_dependency",
        "typed_draw_context_dependency",
        "typed_deck_context_dependency",
        "typed_consumes_future_resource",
        "typed_card_state_mutation",
    )
    for key in bool_keys:
        if sem.get(key, 0.0) > 0.0:
            out[key] = True
    for source_key, out_key in (
        ("typed_gain_energy_amount", "typed_gain_energy_amount"),
        ("typed_hp_loss", "typed_hp_loss"),
        ("typed_draw_amount", "typed_draw_amount"),
    ):
        value = sem.get(source_key, 0.0)
        if value:
            out[out_key] = value
    return out
