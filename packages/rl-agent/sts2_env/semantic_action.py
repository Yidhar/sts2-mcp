"""Semantic action schema and binding helpers for grounded planning.

This module provides a stable action vocabulary that is independent from the
bridge's transient legal-action slot ordering.  The immediate goal is to expose
semantic action features to the model; later search layers can operate on these
semantic candidates and bind them back to live legal actions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from content_registry import get_card_metadata, get_potion_metadata, get_relic_metadata
from .card_effect_profile import (
    aggregate_card_effect_profile_semantics as _aggregate_card_effect_profile_semantics,
    card_effect_operation_names as _card_effect_operation_names,
    compact_card_effect_profile_signature as _compact_card_effect_profile_signature,
)

SEMANTIC_ACTION_FAMILIES = [
    "play_card",
    "use_potion",
    "discard_potion",
    "card_selection",
    "end_turn",
    "proceed",
    "map",
    "reward",
    "card_reward",
    "shop",
    "rest",
    "smith",
    "deck_upgrade",
    "event_option",
    "treasure_relic",
    "startup",
    "other",
]
_FAMILY_TO_IDX = {family: index for index, family in enumerate(SEMANTIC_ACTION_FAMILIES)}

SEMANTIC_TARGET_SCOPES = [
    "none",
    "self",
    "single_enemy",
    "all_enemies",
    "choice",
    "map",
    "shop",
    "event",
    "other",
]
_TARGET_SCOPE_TO_IDX = {scope: index for index, scope in enumerate(SEMANTIC_TARGET_SCOPES)}

SEMANTIC_ROLE_NAMES = [
    "attack",
    "block",
    "draw",
    "debuff",
    "buff",
    "heal",
    "aoe",
    "x_cost",
    "setup",
    "scaling",
    "resource",
    "terminal",
]
_ROLE_TO_IDX = {role: index for index, role in enumerate(SEMANTIC_ROLE_NAMES)}

_EXTRA_NUMERIC_DIM = 15
SEMANTIC_ACTION_DIM = (
    len(SEMANTIC_ACTION_FAMILIES)
    + len(SEMANTIC_TARGET_SCOPES)
    + len(SEMANTIC_ROLE_NAMES)
    + _EXTRA_NUMERIC_DIM
)

_LOG1P_200 = math.log1p(200.0)
_LOG1P_500 = math.log1p(500.0)
_LOG1P_100 = math.log1p(100.0)


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _log_norm(value: float, anchor: float) -> float:
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / anchor, 1.0)


def _nested_value(source: dict[str, Any] | None, *keys: str):
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _preview_metric(source: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(source, dict):
        return default
    key_aliases = {
        "damage": ("damage", "total_damage"),
        "block": ("block", "total_block"),
    }
    for candidate in key_aliases.get(key, (key,)):
        nested = _nested_value(source, "effect_preview", candidate)
        if nested is not None:
            return _float(nested, default)
    for candidate in key_aliases.get(key, (key,)):
        direct = source.get(candidate)
        if direct is not None:
            return _float(direct, default)
    return default


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _metadata_for_action(action: dict[str, Any]) -> dict[str, Any] | None:
    card = action.get("card") if isinstance(action.get("card"), dict) else None
    potion = action.get("potion") if isinstance(action.get("potion"), dict) else None
    relic = action.get("relic") if isinstance(action.get("relic"), dict) else None
    if isinstance(card, dict):
        return get_card_metadata(_safe_text(card.get("id")))
    if isinstance(potion, dict):
        return get_potion_metadata(_safe_text(potion.get("id")))
    if isinstance(relic, dict):
        return get_relic_metadata(_safe_text(relic.get("id")))
    return None


def _preferred_signature_title(
    source: dict[str, Any] | None,
    action: dict[str, Any],
    metadata: dict[str, Any] | None,
    stable_id: str,
) -> str:
    metadata_title = _safe_text((metadata or {}).get("title")) if isinstance(metadata, dict) else ""
    if metadata_title:
        return metadata_title
    return _safe_text(
        (source or {}).get("title")
        or action.get("title")
        or action.get("label")
        or stable_id
    )


def _infer_family(action: dict[str, Any]) -> str:
    kind = _safe_text(action.get("kind"))
    action_id = _safe_text(action.get("action_id"))
    selection = _safe_text(action.get("selection")).lower()
    reward = action.get("reward") if isinstance(action.get("reward"), dict) else {}
    shop_action = _safe_text(action.get("shop_action")).lower()
    if kind == "play_card" or action_id.startswith("play_card:"):
        return "play_card"
    if kind == "use_potion" or action_id.startswith("use_potion:"):
        return "use_potion"
    if kind == "discard_potion":
        return "discard_potion"
    if kind in {"card_selection", "combat_select_card", "combat_select"} or action_id.startswith("combat_select"):
        return "card_selection"
    if action_id == "end_turn":
        return "end_turn"
    if kind == "proceed" or action_id == "proceed":
        return "proceed"
    if kind == "map":
        return "map"
    if kind == "reward" or reward:
        return "reward"
    if kind == "card_reward" or selection in {"pick", "skip"}:
        return "card_reward"
    if kind == "shop" or shop_action:
        return "shop"
    if kind == "rest_site":
        return "rest"
    if kind == "deck_upgrade":
        return "deck_upgrade"
    if kind == "event_option":
        return "event_option"
    if kind == "treasure_relic":
        return "treasure_relic"
    if action_id == "embark" or action_id.startswith(("main_menu:", "run_mode:", "character_select:")):
        return "startup"
    return "other"


def _infer_target_scope(action: dict[str, Any]) -> str:
    if _infer_family(action) in {"map"}:
        return "map"
    if _infer_family(action) in {"shop"}:
        return "shop"
    if _infer_family(action) in {"event_option"}:
        return "event"
    if _infer_family(action) in {"card_reward", "reward", "deck_upgrade", "card_selection"}:
        return "choice"

    card = action.get("card") if isinstance(action.get("card"), dict) else None
    potion = action.get("potion") if isinstance(action.get("potion"), dict) else None
    target = ""
    if isinstance(card, dict):
        target = _safe_text(card.get("target_type") or card.get("target"))
    elif isinstance(potion, dict):
        target = _safe_text(potion.get("target_type") or potion.get("target"))
    if not target:
        target_obj = action.get("target")
        if isinstance(target_obj, dict):
            target = _safe_text(target_obj.get("side") or target_obj.get("name"))
        else:
            target = _safe_text(target_obj)

    lower = target.lower()
    if not lower:
        return "none"
    if "all" in lower:
        return "all_enemies"
    if "single" in lower or "enemy" in lower or "anyenemy" in lower:
        return "single_enemy"
    if "self" in lower or "player" in lower:
        return "self"
    return "other"


def _infer_roles(action: dict[str, Any], metadata: dict[str, Any] | None) -> list[str]:
    roles: set[str] = set()
    family = _infer_family(action)
    if family in {"end_turn", "proceed"}:
        roles.add("terminal")
    if family in {"reward", "card_reward", "shop", "rest", "smith", "deck_upgrade", "map", "treasure_relic"}:
        roles.add("resource")
    if family == "card_selection":
        semantics = _safe_text(action.get("selection_semantics")).lower()
        if any(token in semantics for token in ("upgrade", "smith", "transform", "remove", "purge", "reward", "discover", "draft")):
            roles.add("resource")
        if any(token in semantics for token in ("exhaust", "discard")):
            roles.add("setup")

    source = None
    if isinstance(action.get("card"), dict):
        source = action["card"]
    elif isinstance(action.get("potion"), dict):
        source = action["potion"]
    effect_sem = _aggregate_card_effect_profile_semantics(source) if isinstance(source, dict) else {}

    damage = _preview_metric(source, "damage")
    block = _preview_metric(source, "block")
    draw = max(_preview_metric(source, "draw"), effect_sem.get("typed_draw_amount", 0.0))
    heal = _preview_metric(source, "heal")
    weak = _preview_metric(source, "weak")
    vulnerable = _preview_metric(source, "vulnerable")
    poison = _preview_metric(source, "poison")
    strength = _preview_metric(source, "strength")
    dexterity = _preview_metric(source, "dexterity")
    hits = _preview_metric(source, "hits")

    if damage > 0:
        roles.add("attack")
    if block > 0:
        roles.add("block")
    if draw > 0:
        roles.add("draw")
    if heal > 0:
        roles.add("heal")
    if weak > 0 or vulnerable > 0 or poison > 0:
        roles.add("debuff")
    if strength > 0 or dexterity > 0:
        roles.add("buff")
    if hits > 1 or _infer_target_scope(action) == "all_enemies":
        roles.add("aoe")
    if isinstance(source, dict) and (
        bool(source.get("x_cost") or source.get("costs_x"))
        or _safe_text(source.get("cost")).strip().upper() == "X"
        or _safe_text(source.get("canonical_energy_cost")).strip().upper() == "X"
    ):
        roles.add("x_cost")
    if isinstance(source, dict):
        card_type = _safe_text(source.get("type")).lower()
        if card_type == "power" or strength > 0 or dexterity > 0:
            roles.add("scaling")
        if _preview_metric(source, "draw") > 0 and damage <= 0 and block <= 0:
            roles.add("setup")

    # Typed card-effect operations are generated from card ids/source facts.
    # Map them onto the existing role vocabulary so checkpoints/replay buffers
    # keep the same fixed action-vector shape while the policy can still see
    # hand mutation, cost mutation, replay/retain/exhaust, no-draw, and
    # same-turn resource conversion semantics.
    if effect_sem.get("typed_gain_energy", 0.0) > 0.0 or effect_sem.get("typed_hp_loss", 0.0) > 0.0:
        roles.add("resource")
    if effect_sem.get("typed_draw_cards", 0.0) > 0.0:
        roles.add("draw")
    if any(
        effect_sem.get(key, 0.0) > 0.0
        for key in (
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
            "typed_card_state_mutation",
            "typed_no_draw",
            "typed_future_penalty",
            "typed_requires_followup",
        )
    ):
        roles.add("setup")
    if effect_sem.get("typed_apply_power", 0.0) > 0.0:
        roles.add("setup")
        roles.add("scaling")

    tags = metadata.get("semantic_tags") if isinstance(metadata, dict) else None
    if isinstance(tags, list):
        joined = " ".join(_safe_text(tag).lower() for tag in tags)
        if "aoe" in joined:
            roles.add("aoe")
        if "scale" in joined or "power" in joined:
            roles.add("scaling")
        if "setup" in joined:
            roles.add("setup")
        if "draw" in joined:
            roles.add("draw")
        if "block" in joined:
            roles.add("block")
        if "attack" in joined:
            roles.add("attack")

    # Phase 3 of potion-timing-modeling-plan.md §7.4: when this action is a
    # potion use, lift effect_family from the potion profile (bridge live or
    # registry fallback) into roles. Map onto existing SEMANTIC_ROLE_NAMES so
    # action feature dim stays unchanged.
    if family == "use_potion" and isinstance(action.get("potion"), dict):
        potion_payload = action["potion"]
        effect_family = potion_payload.get("effect_family")
        if not isinstance(effect_family, list):
            effect_family = []
        if not effect_family:
            try:
                from .potion_profiles import get_potion_profile as _gpp
                pid = _safe_text(potion_payload.get("id"))
                if pid:
                    effect_family = list(_gpp(pid).get("effect_family") or [])
            except Exception:
                effect_family = []
        family_to_role = {
            "damage": "attack", "aoe": "aoe", "self_damage": "aoe",
            "block": "block", "intangible": "block", "prevent_damage": "block",
            "delayed_block": "block",
            "draw": "draw",
            "energy": "resource", "energy_gain": "resource",
            "generate_cards": "resource", "discover": "resource",
            "retrieve_from_discard": "resource",
            "weak": "debuff", "vulnerable": "debuff", "poison": "debuff",
            "debuff": "debuff",
            "strength": "buff", "dexterity": "buff", "focus": "buff",
            "scaling": "scaling", "ritual": "buff", "thorns": "buff",
            "plated": "buff", "regen": "buff", "buffer": "buff",
            "heal": "heal", "max_hp": "heal",
            "upgrade": "setup", "duplicate_next": "setup",
            "transform_hand": "setup", "exhaust_hand": "setup",
            "free_play": "setup", "snecko": "setup", "long_term": "setup",
        }
        for fam in effect_family:
            mapped = family_to_role.get(_safe_text(fam).lower())
            if mapped:
                roles.add(mapped)

    return [role for role in SEMANTIC_ROLE_NAMES if role in roles]


def _stable_entity_id(action: dict[str, Any]) -> str:
    for payload_key in ("card", "potion", "relic"):
        payload = action.get(payload_key)
        if isinstance(payload, dict):
            entity_id = _safe_text(payload.get("id"))
            if entity_id:
                return entity_id
            title = _safe_text(payload.get("title"))
            if title:
                return title.lower().replace(" ", "_")
    title = _safe_text(action.get("title") or action.get("label") or action.get("name"))
    if title:
        return title.lower().replace(" ", "_")
    action_id = _safe_text(action.get("action_id"))
    if action_id:
        return action_id
    return "unknown"


def semantic_action_signature(action: Any) -> dict[str, Any]:
    """Return a semantic signature for one live legal action."""
    if not isinstance(action, dict):
        return {}

    family = _infer_family(action)
    target_scope = _infer_target_scope(action)
    metadata = _metadata_for_action(action) or {}
    stable_id = _stable_entity_id(action)
    roles = _infer_roles(action, metadata)

    source = None
    if isinstance(action.get("card"), dict):
        source = action["card"]
    elif isinstance(action.get("potion"), dict):
        source = action["potion"]

    title = _preferred_signature_title(source, action, metadata, stable_id)
    price = _float(action.get("price") if action.get("price") is not None else action.get("cost"))
    if price <= 0 and isinstance(action.get("item"), dict):
        price = _float(action["item"].get("cost"))

    choice_index = action.get("index") if action.get("index") is not None else action.get("choice_index")
    target_index = action.get("target_index")
    if target_index is None and isinstance(action.get("target"), dict):
        target_index = action["target"].get("index")

    card_type = _safe_text((source or {}).get("type"))
    effect_summary = _safe_text(
        _nested_value(source, "effect_preview", "summary")
        or (source or {}).get("effect")
        or (source or {}).get("description")
    )
    effect_sem = _aggregate_card_effect_profile_semantics(source) if isinstance(source, dict) else {}
    card_ops = _card_effect_operation_names(source) if isinstance(source, dict) else []
    effect_compact = _compact_card_effect_profile_signature(source) if isinstance(source, dict) else {}
    draw_value = max(_preview_metric(source, "draw"), effect_sem.get("typed_draw_amount", 0.0))
    hp_loss_value = max(_preview_metric(source, "hp_loss"), effect_sem.get("typed_hp_loss", 0.0))
    semantic_key = "|".join(
        part
        for part in (
            family,
            stable_id,
            target_scope,
            _safe_text(action.get("surface")),
            _safe_text(action.get("selection")),
            _safe_text(action.get("selection_semantics")),
            _safe_text(action.get("shop_action")),
            _safe_text((action.get("reward") or {}).get("type")),
        )
        if part
    )

    return {
        "semantic_key": semantic_key,
        "family": family,
        "title": title,
        "stable_id": stable_id,
        "kind": _safe_text(action.get("kind")),
        "action_id": _safe_text(action.get("action_id")),
        "domain": (
            "route" if family in {"map"} else
            "selection" if family in {"card_selection"} else
            "build" if family in {"reward", "card_reward", "shop", "rest", "smith", "deck_upgrade", "event_option", "treasure_relic", "startup", "proceed"} else
            "combat"
        ),
        "selection_semantics": _safe_text(action.get("selection_semantics")),
        "target_scope": target_scope,
        "target_index": int(target_index) if isinstance(target_index, int) else None,
        "choice_index": int(choice_index) if isinstance(choice_index, int) else None,
        "roles": roles,
        "card_id": _safe_text(((action.get("card") or {}).get("id"))),
        "potion_id": _safe_text(((action.get("potion") or {}).get("id"))),
        "relic_id": _safe_text(((action.get("relic") or {}).get("id"))),
        "card_type": card_type,
        "cost": _float((source or {}).get("cost")),
        "star": _float((source or {}).get("star")),
        "upgrade_level": _float((source or {}).get("upgrade_level") or (source or {}).get("current_upgrade_level")),
        "price": price,
        "damage": _preview_metric(source, "damage"),
        "block": _preview_metric(source, "block"),
        "draw": draw_value,
        "heal": _preview_metric(source, "heal"),
        "hp_loss": hp_loss_value,
        "weak": _preview_metric(source, "weak"),
        "vulnerable": _preview_metric(source, "vulnerable"),
        "hits": _preview_metric(source, "hits"),
        "damage_per_hit": _preview_metric(source, "damage_per_hit"),
        "x_cost_value": _preview_metric(source, "x_cost_value"),
        "is_x_cost": bool(
            isinstance(source, dict)
            and (
                source.get("x_cost")
                or source.get("costs_x")
                or _safe_text(source.get("cost")).strip().upper() == "X"
                or _safe_text(source.get("canonical_energy_cost")).strip().upper() == "X"
            )
        ),
        # Static cost view: -1 sentinel for X-cost, raw int otherwise.  Diagnostic
        # fields below let the trainer detect "X-cost played at 0 energy without a
        # non-energy effect" without reparsing the bridge payload.
        "base_cost": (
            -1.0
            if isinstance(source, dict)
            and (
                source.get("x_cost")
                or source.get("costs_x")
                or _safe_text(source.get("cost")).strip().upper() == "X"
                or _safe_text(source.get("canonical_energy_cost")).strip().upper() == "X"
            )
            else _float((source or {}).get("cost"))
        ),
        # Non-energy effect: any structural side-effect that fires regardless of
        # energy spent (hand/pile/card-state mutation, retain, etc.).  Used to
        # gate the zero_energy_x_cost_selected offender so X-cost cards used as
        # pile-manipulation tools are not flagged as "wasted".
        "x_cost_has_non_energy_effect": bool(
            effect_sem.get("typed_modifies_hand", 0.0) > 0.0
            or effect_sem.get("typed_upgrade_hand", 0.0) > 0.0
            or effect_sem.get("typed_exhaust_cards", 0.0) > 0.0
            or effect_sem.get("typed_discard_cards", 0.0) > 0.0
            or effect_sem.get("typed_transform_cards", 0.0) > 0.0
            or effect_sem.get("typed_copy_cards", 0.0) > 0.0
            or effect_sem.get("typed_add_modifier", 0.0) > 0.0
            or effect_sem.get("typed_add_keyword", 0.0) > 0.0
            or effect_sem.get("typed_set_replay", 0.0) > 0.0
            or effect_sem.get("typed_retain_cards", 0.0) > 0.0
            or effect_sem.get("typed_card_state_mutation", 0.0) > 0.0
        ),
        "is_attack": card_type.lower() == "attack",
        "is_skill": card_type.lower() == "skill",
        "is_power": card_type.lower() == "power",
        "is_zero_cost": _float((source or {}).get("cost")) == 0.0 if isinstance(source, dict) else False,
        "is_terminal": family in {"end_turn", "proceed"},
        "effect_summary": effect_summary,
        "card_ops": card_ops,
        "typed_modifies_hand": bool(effect_sem.get("typed_modifies_hand", 0.0) > 0.0),
        "typed_upgrade_hand": bool(effect_sem.get("typed_upgrade_hand", 0.0) > 0.0),
        "typed_modify_cost": bool(effect_sem.get("typed_modify_cost", 0.0) > 0.0),
        "typed_set_replay": bool(effect_sem.get("typed_set_replay", 0.0) > 0.0),
        "typed_retain_cards": bool(effect_sem.get("typed_retain_cards", 0.0) > 0.0),
        "typed_exhaust_cards": bool(effect_sem.get("typed_exhaust_cards", 0.0) > 0.0),
        "typed_discard_cards": bool(effect_sem.get("typed_discard_cards", 0.0) > 0.0),
        "typed_transform_cards": bool(effect_sem.get("typed_transform_cards", 0.0) > 0.0),
        "typed_copy_cards": bool(effect_sem.get("typed_copy_cards", 0.0) > 0.0),
        "typed_add_modifier": bool(effect_sem.get("typed_add_modifier", 0.0) > 0.0),
        "typed_add_keyword": bool(effect_sem.get("typed_add_keyword", 0.0) > 0.0),
        "typed_gain_energy": effect_sem.get("typed_gain_energy_amount", 0.0),
        "typed_hp_loss": effect_sem.get("typed_hp_loss", 0.0),
        "typed_no_draw": bool(effect_sem.get("typed_no_draw", 0.0) > 0.0),
        "typed_future_penalty": bool(effect_sem.get("typed_future_penalty", 0.0) > 0.0),
        "typed_requires_followup": bool(effect_sem.get("typed_requires_followup", 0.0) > 0.0),
        "typed_strategic_skip_if_no_followup": bool(effect_sem.get("typed_strategic_skip_if_no_followup", 0.0) > 0.0),
        "typed_not_x_cost_filter": bool(effect_sem.get("typed_not_x_cost_filter", 0.0) > 0.0),
        "typed_x_cost_filter": bool(effect_sem.get("typed_x_cost_filter", 0.0) > 0.0),
        "typed_hand_context_dependency": bool(effect_sem.get("typed_hand_context_dependency", 0.0) > 0.0),
        "typed_discard_context_dependency": bool(effect_sem.get("typed_discard_context_dependency", 0.0) > 0.0),
        "typed_exhaust_context_dependency": bool(effect_sem.get("typed_exhaust_context_dependency", 0.0) > 0.0),
        "typed_draw_context_dependency": bool(effect_sem.get("typed_draw_context_dependency", 0.0) > 0.0),
        "typed_deck_context_dependency": bool(effect_sem.get("typed_deck_context_dependency", 0.0) > 0.0),
        "typed_consumes_future_resource": bool(effect_sem.get("typed_consumes_future_resource", 0.0) > 0.0),
        "typed_card_state_mutation": bool(effect_sem.get("typed_card_state_mutation", 0.0) > 0.0),
        "card_effect_profile": effect_compact,
    }


def semantic_action_text(signature: dict[str, Any] | None) -> str:
    if not isinstance(signature, dict):
        return ""
    parts: list[str] = []
    title = _safe_text(signature.get("title"))
    family = _safe_text(signature.get("family"))
    target_scope = _safe_text(signature.get("target_scope"))
    roles = signature.get("roles") if isinstance(signature.get("roles"), list) else []
    if title:
        parts.append(title)
    if family:
        parts.append(f"family={family}")
    if target_scope and target_scope != "none":
        parts.append(f"target={target_scope}")
    semantics = _safe_text(signature.get("selection_semantics"))
    if semantics:
        parts.append(f"selection={semantics}")
    if roles:
        parts.append(f"role={','.join(_safe_text(role) for role in roles if _safe_text(role))}")
    metrics: list[str] = []
    for key in ("damage", "block", "draw", "heal", "hits", "damage_per_hit", "x_cost_value", "price"):
        value = signature.get(key)
        if value not in (None, "", False):
            numeric = _float(value, 0.0)
            if numeric > 0:
                metrics.append(f"{key}={int(numeric) if float(numeric).is_integer() else round(numeric, 2)}")
    if metrics:
        parts.append("sig=" + ",".join(metrics))
    card_ops = signature.get("card_ops") if isinstance(signature.get("card_ops"), list) else []
    if card_ops:
        joined_ops = ",".join(_safe_text(op) for op in card_ops if _safe_text(op))
        if joined_ops:
            parts.append(f"ops={joined_ops}")
    typed_flags: list[str] = []
    for key, label in (
        ("typed_requires_followup", "requires_followup"),
        ("typed_strategic_skip_if_no_followup", "skip_without_followup"),
        ("typed_modifies_hand", "hand_mutation"),
        ("typed_upgrade_hand", "upgrade_hand"),
        ("typed_modify_cost", "modify_cost"),
        ("typed_set_replay", "replay"),
        ("typed_retain_cards", "retain"),
        ("typed_exhaust_cards", "exhaust_cards"),
        ("typed_discard_cards", "discard_cards"),
        ("typed_transform_cards", "transform_cards"),
        ("typed_copy_cards", "copy_cards"),
        ("typed_add_modifier", "modifier"),
        ("typed_add_keyword", "keyword"),
        ("typed_no_draw", "no_draw"),
        ("typed_future_penalty", "future_penalty"),
        ("typed_not_x_cost_filter", "not_x_cost"),
        ("typed_x_cost_filter", "x_cost_filter"),
        ("typed_consumes_future_resource", "future_resource"),
    ):
        if signature.get(key):
            typed_flags.append(label)
    if typed_flags:
        parts.append("typed=" + ",".join(typed_flags))
    typed_energy = _float(signature.get("typed_gain_energy"), 0.0)
    if typed_energy > 0:
        parts.append(f"typed_energy={int(typed_energy) if typed_energy.is_integer() else round(typed_energy, 2)}")
    typed_hp_loss = _float(signature.get("typed_hp_loss"), 0.0)
    if typed_hp_loss > 0:
        parts.append(f"typed_hp_loss={int(typed_hp_loss) if typed_hp_loss.is_integer() else round(typed_hp_loss, 2)}")
    return " | ".join(parts)


def encode_semantic_action_numeric(signature: dict[str, Any] | None) -> np.ndarray:
    """Encode a semantic signature into a fixed numeric vector."""
    vector = np.zeros(SEMANTIC_ACTION_DIM, dtype=np.float32)
    if not isinstance(signature, dict):
        return vector

    offset = 0
    family_idx = _FAMILY_TO_IDX.get(_safe_text(signature.get("family")))
    if family_idx is not None:
        vector[offset + family_idx] = 1.0
    offset += len(SEMANTIC_ACTION_FAMILIES)

    target_idx = _TARGET_SCOPE_TO_IDX.get(_safe_text(signature.get("target_scope")), _TARGET_SCOPE_TO_IDX["other"])
    vector[offset + target_idx] = 1.0
    offset += len(SEMANTIC_TARGET_SCOPES)

    roles = signature.get("roles") if isinstance(signature.get("roles"), list) else []
    for role in roles:
        role_idx = _ROLE_TO_IDX.get(_safe_text(role))
        if role_idx is not None:
            vector[offset + role_idx] = 1.0
    offset += len(SEMANTIC_ROLE_NAMES)

    vector[offset + 0] = min(_float(signature.get("cost")) / 5.0, 1.0)
    vector[offset + 1] = _log_norm(_float(signature.get("price")), _LOG1P_500)
    vector[offset + 2] = _log_norm(_float(signature.get("damage")), _LOG1P_200)
    vector[offset + 3] = _log_norm(_float(signature.get("block")), _LOG1P_200)
    vector[offset + 4] = min(_float(signature.get("draw")) / 5.0, 1.0)
    vector[offset + 5] = _log_norm(_float(signature.get("heal")), _LOG1P_100)
    vector[offset + 6] = _log_norm(_float(signature.get("hp_loss")), _LOG1P_100)
    vector[offset + 7] = min(_float(signature.get("hits")) / 10.0, 1.0)
    vector[offset + 8] = _log_norm(_float(signature.get("damage_per_hit")), _LOG1P_100)
    vector[offset + 9] = min(_float(signature.get("x_cost_value")) / 10.0, 1.0)
    vector[offset + 10] = min(max(_float(signature.get("target_index")), 0.0) / 5.0, 1.0)
    vector[offset + 11] = min(max(_float(signature.get("choice_index")), 0.0) / 20.0, 1.0)
    vector[offset + 12] = min(_float(signature.get("star")) / 5.0, 1.0)
    vector[offset + 13] = min(_float(signature.get("upgrade_level")) / 3.0, 1.0)
    vector[offset + 14] = 1.0 if signature.get("is_zero_cost") else 0.0
    return vector


def compact_semantic_signature(signature: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(signature, dict):
        return {}
    compact = {
        "semantic_key": signature.get("semantic_key"),
        "family": signature.get("family"),
        "domain": signature.get("domain"),
        "title": signature.get("title"),
        "stable_id": signature.get("stable_id"),
        "selection_semantics": signature.get("selection_semantics"),
        "target_scope": signature.get("target_scope"),
        "roles": signature.get("roles"),
        "damage": signature.get("damage"),
        "block": signature.get("block"),
        "draw": signature.get("draw"),
        "hp_loss": signature.get("hp_loss"),
        "hits": signature.get("hits"),
        "damage_per_hit": signature.get("damage_per_hit"),
        "x_cost_value": signature.get("x_cost_value"),
        "price": signature.get("price"),
        "card_ops": signature.get("card_ops"),
        "typed_modifies_hand": signature.get("typed_modifies_hand"),
        "typed_upgrade_hand": signature.get("typed_upgrade_hand"),
        "typed_modify_cost": signature.get("typed_modify_cost"),
        "typed_set_replay": signature.get("typed_set_replay"),
        "typed_retain_cards": signature.get("typed_retain_cards"),
        "typed_exhaust_cards": signature.get("typed_exhaust_cards"),
        "typed_discard_cards": signature.get("typed_discard_cards"),
        "typed_transform_cards": signature.get("typed_transform_cards"),
        "typed_copy_cards": signature.get("typed_copy_cards"),
        "typed_add_modifier": signature.get("typed_add_modifier"),
        "typed_add_keyword": signature.get("typed_add_keyword"),
        "typed_gain_energy": signature.get("typed_gain_energy"),
        "typed_hp_loss": signature.get("typed_hp_loss"),
        "typed_no_draw": signature.get("typed_no_draw"),
        "typed_future_penalty": signature.get("typed_future_penalty"),
        "typed_requires_followup": signature.get("typed_requires_followup"),
        "typed_strategic_skip_if_no_followup": signature.get("typed_strategic_skip_if_no_followup"),
        "typed_not_x_cost_filter": signature.get("typed_not_x_cost_filter"),
        "typed_x_cost_filter": signature.get("typed_x_cost_filter"),
        "typed_hand_context_dependency": signature.get("typed_hand_context_dependency"),
        "typed_discard_context_dependency": signature.get("typed_discard_context_dependency"),
        "typed_exhaust_context_dependency": signature.get("typed_exhaust_context_dependency"),
        "typed_draw_context_dependency": signature.get("typed_draw_context_dependency"),
        "typed_deck_context_dependency": signature.get("typed_deck_context_dependency"),
        "typed_consumes_future_resource": signature.get("typed_consumes_future_resource"),
        "typed_card_state_mutation": signature.get("typed_card_state_mutation"),
        "card_effect_profile": signature.get("card_effect_profile"),
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], False)}


def semantic_match_score(plan_signature: dict[str, Any] | None, candidate_signature: dict[str, Any] | None) -> float:
    if not isinstance(plan_signature, dict) or not isinstance(candidate_signature, dict):
        return float("-inf")

    score = 0.0
    if plan_signature.get("semantic_key") == candidate_signature.get("semantic_key"):
        score += 8.0
    if plan_signature.get("family") == candidate_signature.get("family"):
        score += 3.0
    if plan_signature.get("stable_id") == candidate_signature.get("stable_id"):
        score += 3.0
    if plan_signature.get("target_scope") == candidate_signature.get("target_scope"):
        score += 1.5
    if plan_signature.get("title") == candidate_signature.get("title"):
        score += 1.0

    plan_roles = set(plan_signature.get("roles") or [])
    candidate_roles = set(candidate_signature.get("roles") or [])
    score += 0.5 * len(plan_roles.intersection(candidate_roles))

    for key, weight in (
        ("damage", 0.02),
        ("block", 0.02),
        ("draw", 0.10),
        ("heal", 0.05),
        ("hits", 0.10),
        ("damage_per_hit", 0.05),
        ("x_cost_value", 0.10),
        ("price", 0.01),
    ):
        plan_value = _float(plan_signature.get(key))
        candidate_value = _float(candidate_signature.get(key))
        score -= abs(plan_value - candidate_value) * weight

    return score
