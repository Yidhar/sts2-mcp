"""Generic combat hand-mutation affordance extraction.

The primary contract is typed effect operations supplied by the bridge/static
card profile, e.g. ``card_effect_profile.operations`` entries such as
``upgrade_card`` + ``source_zone=hand`` + ``selection=choice``.  That path is
based on internal source IDs / command facts, not localized description text.

Description-text matching is intentionally kept only as a missing-profile
fallback so older observations and offline smoke tests still produce a weak
signal while we finish exporting profiles for every card.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import observation_common as obs_common


def _card_text(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    parts: list[str] = []
    for key in ("title", "name", "id", "description", "text", "canonical_text"):
        value = str(card.get(key) or "").strip()
        if value:
            parts.append(value)
    for key in ("keywords", "semantic_tags"):
        values = card.get(key)
        if isinstance(values, list):
            parts.extend(str(v or "").strip() for v in values if str(v or "").strip())
    signals = card.get("semantic_signals")
    if isinstance(signals, dict):
        parts.extend(str(k) for k, v in signals.items() if v)
    return " | ".join(parts).lower()


def _norm_count(value: float, denom: float = 12.0) -> float:
    return min(max(float(value), 0.0) / max(denom, 1.0), 1.0)


def _signed_norm(value: float, denom: float = 10.0) -> float:
    return max(min(float(value) / max(denom, 1.0), 1.0), -1.0)


def _cost(card: dict[str, Any] | None) -> float:
    if not isinstance(card, dict):
        return 0.0
    return max(obs_common._runtime_spend_cost(card), 0.0)


def _upgrade_level(card: dict[str, Any] | None) -> int:
    if not isinstance(card, dict):
        return 0
    return max(obs_common._infer_upgrade_level(card), 0)


def _preview(card: dict[str, Any] | None, key: str) -> float:
    if not isinstance(card, dict):
        return 0.0
    if key in {"damage", "block"}:
        bundle = obs_common._build_card_preview_bundle(card)
        return float(bundle.get(f"preview_{key}", 0.0))
    return obs_common._preview_metric(card, key)


def _card_type(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    return str(card.get("type") or "").strip().lower()


def _is_upgradeable(card: dict[str, Any] | None) -> bool:
    if not isinstance(card, dict):
        return False
    if bool(card.get("upgraded")) or bool(card.get("is_upgraded")):
        return False
    if _upgrade_level(card) > 0:
        return False
    if card.get("can_upgrade") is not None:
        return bool(card.get("can_upgrade"))
    text = _card_text(card)
    return "unupgradable" not in text and "cannot upgrade" not in text


def _is_x_cost_card(card: dict[str, Any] | None) -> bool:
    if not isinstance(card, dict):
        return False
    if bool(
        card.get("x_cost")
        or card.get("costs_x")
        or card.get("energy_cost_x")
        or card.get("is_x_cost")
        or card.get("cost_is_x")
    ):
        return True
    cost = str(card.get("cost") or card.get("energy_cost") or card.get("canonical_energy_cost") or "").strip().upper()
    if cost == "X":
        return True
    tags = card.get("tags")
    if isinstance(tags, list) and any(_norm_token(tag) in {"cost_x", "x_cost"} for tag in tags):
        return True
    return False


def _infer_scope(text: str, *, default_one: bool = False) -> tuple[str, bool, bool, bool, bool]:
    """Return scope plus one/all/random/choice flags."""
    all_scope = bool(re.search(r"\b(all|each|every)\b.*\b(hand|cards? in your hand)\b", text))
    if not all_scope:
        all_scope = "all cards in your hand" in text or "your hand" in text and "all" in text
    random_scope = "random" in text
    one_scope = bool(re.search(r"\b(a|one|1)\b.*\b(card|attack|skill|power)\b.*\b(hand|your hand)\b", text))
    choice = one_scope or "choose" in text or "select" in text
    if all_scope:
        return "all", False, True, random_scope, False
    if random_scope:
        return "random", False, False, True, False
    if one_scope or default_one:
        return "one", True, False, False, choice
    return "hand", False, False, False, choice


def _type_filter(text: str) -> set[str]:
    filters: set[str] = set()
    for kind in ("attack", "skill", "power"):
        if kind in text:
            filters.add(kind)
    return filters


def _matches_filter(card: dict[str, Any], filters: set[str]) -> bool:
    if not filters:
        return True
    return _card_type(card) in filters


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _norm_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.upper().startswith("CARD."):
        text = text.split(".", 1)[1]
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _profile_operations(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Return typed effect operations from bridge/static card payloads.

    Accepted shapes are deliberately broad so the Python observation stack can
    consume both bridge live payloads and generated offline fixtures:

    * ``card_effect_profile.operations`` / ``cardEffectProfile.operations``
    * ``effect_profile.operations`` / ``effectProfile.operations``
    * top-level ``operations`` (unit tests / generated artifacts)
    """

    def collect(container: Any) -> list[dict[str, Any]]:
        if isinstance(container, list):
            return [dict(v) for v in container if isinstance(v, dict)]
        if not isinstance(container, dict):
            return []
        for key in ("operations", "ops", "effects", "effect_operations"):
            ops = container.get(key)
            if isinstance(ops, list):
                return [dict(v) for v in ops if isinstance(v, dict)]
        return []

    for key in (
        "card_effect_profile",
        "cardEffectProfile",
        "effect_profile",
        "effectProfile",
        "profile",
    ):
        ops = collect(source.get(key))
        if ops:
            return ops
    return collect(source.get("operations"))


def _op_name(op: dict[str, Any]) -> str:
    for key in ("op", "operation", "operation_id", "type", "kind", "command", "command_id"):
        value = op.get(key)
        if value:
            token = _norm_token(value)
            break
    else:
        return ""
    aliases = {
        "upgrade": "upgrade_card",
        "cardcmd_upgrade": "upgrade_card",
        "card_cmd_upgrade": "upgrade_card",
        "exhaust": "exhaust_card",
        "cardcmd_exhaust": "exhaust_card",
        "card_cmd_exhaust": "exhaust_card",
        "discard": "discard_card",
        "transform": "transform_card",
        "cardcmd_transform": "transform_card",
        "card_cmd_transform": "transform_card",
        "copy": "copy_card",
        "clone": "copy_card",
        "create_clone": "copy_card",
        "createclone": "copy_card",
        "draw": "draw_card",
        "add": "add_generated_card",
        "add_card": "add_generated_card",
        "addgeneratedcardtocombat": "add_generated_card",
        "cardpilecmd_addgeneratedcardtocombat": "add_generated_card",
        "card_pile_cmd_add_generated_card_to_combat": "add_generated_card",
        "move": "move_card",
        "cardpilecmd_add": "move_card",
        "set_cost": "modify_cost",
        "energycost_set": "modify_cost",
        "energy_cost_set": "modify_cost",
        "energycost_setthisturnoruntilplayed": "modify_cost",
        "energycost_setthiscombat": "modify_cost",
        "modify_energy_cost": "modify_cost",
        "apply": "apply_power",
        "powercmd_apply": "apply_power",
        "power_cmd_apply": "apply_power",
        "retain": "retain_card",
        "set_retain": "retain_card",
        "replay": "set_replay",
    }
    return aliases.get(token, token)


def _num_field(op: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = op.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _has_num_field(op: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = op.get(key)
        if value is None:
            continue
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            continue
    return False


def _bool_field(op: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = op.get(key)
        if isinstance(value, bool):
            return value
        if value is None:
            continue
        token = _norm_token(value)
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    return False


def _merge_upgraded_override(op: dict[str, Any], source_level: int) -> dict[str, Any]:
    if source_level <= 0:
        return dict(op)
    merged = dict(op)
    override = op.get("upgraded_override") or op.get("upgradedOverride") or op.get("upgraded")
    if isinstance(override, dict):
        merged.update(override)
    # Also support generated flat fields such as upgraded_scope/upgraded_count.
    for key, value in op.items():
        if not key.startswith("upgraded_") or key == "upgraded_override":
            continue
        base_key = key[len("upgraded_") :]
        if base_key:
            merged[base_key] = value
    return merged


def _zone_token(value: Any) -> str:
    token = _norm_token(value)
    zone_aliases = {
        "piletype_hand": "hand",
        "hand_pile": "hand",
        "cards_in_hand": "hand",
        "piletype_draw": "draw_pile",
        "draw": "draw_pile",
        "draw_pile_top": "draw_pile_top",
        "piletype_discard": "discard_pile",
        "discard": "discard_pile",
        "piletype_exhaust": "exhaust_pile",
        "exhaust": "exhaust_pile",
        "exhausted": "exhaust_pile",
    }
    return zone_aliases.get(token, token)


def _zone_from(op: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = op.get(key)
        if value is not None:
            return _zone_token(value)
    return ""


def _is_hand_zone(value: Any) -> bool:
    return _zone_token(value) == "hand"


def _is_draw_zone(value: Any) -> bool:
    return _zone_token(value) in {"draw_pile", "draw_pile_top"}


def _is_discard_zone(value: Any) -> bool:
    return _zone_token(value) == "discard_pile"


def _is_exhaust_zone(value: Any) -> bool:
    return _zone_token(value) == "exhaust_pile"


def _duration_is_temporary(value: Any) -> bool:
    token = _norm_token(value)
    return token in {
        "this_turn",
        "turn",
        "temporary",
        "until_played",
        "this_turn_or_until_played",
        "until_end_of_turn",
        "end_of_turn",
    }


def _operation_filters(op: dict[str, Any]) -> set[str]:
    filters: set[str] = set()

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            token = _norm_token(value)
            if token:
                filters.add(token)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            # Boolean dictionaries: {"is_upgradable": true}
            for key, inner in value.items():
                if isinstance(inner, bool):
                    if inner:
                        filters.add(_norm_token(key))
                    continue
                key_token = _norm_token(key)
                if key_token in {"type", "card_type", "cardtype"}:
                    for item in _as_list(inner):
                        filters.add(f"type_{_norm_token(item)}")
                elif key_token in {"not_type", "exclude_type"}:
                    for item in _as_list(inner):
                        filters.add(f"not_type_{_norm_token(item)}")
                else:
                    visit(inner)

    for key in ("target_filter", "target_filters", "filters", "filter", "predicate", "target_type"):
        visit(op.get(key))
    card_type = op.get("card_type") or op.get("cardType")
    if card_type:
        filters.add(f"type_{_norm_token(card_type)}")
    return filters


def _matches_operation_filter(card: dict[str, Any], filters: set[str]) -> bool:
    if not filters:
        return True
    ctype = _card_type(card)
    positive_types: set[str] = set()
    for raw in filters:
        token = _norm_token(raw)
        if token in {"attack", "type_attack", "cardtype_attack"}:
            positive_types.add("attack")
        elif token in {"skill", "type_skill", "cardtype_skill"}:
            positive_types.add("skill")
        elif token in {"power", "type_power", "cardtype_power"}:
            positive_types.add("power")
        elif token in {"status", "type_status", "cardtype_status"}:
            positive_types.add("status")
        elif token in {"curse", "type_curse", "cardtype_curse"}:
            positive_types.add("curse")
        elif token in {"not_attack", "not_type_attack", "non_attack"} and ctype == "attack":
            return False
        elif token in {"not_skill", "not_type_skill", "non_skill"} and ctype == "skill":
            return False
        elif token in {"not_power", "not_type_power", "non_power"} and ctype == "power":
            return False
        elif token in {"not_status", "not_type_status", "non_status"} and ctype == "status":
            return False
        elif token in {"not_curse", "not_type_curse", "non_curse"} and ctype == "curse":
            return False
        elif token in {"is_upgradable", "upgradable", "can_upgrade"} and not _is_upgradeable(card):
            return False
        elif token in {"is_transformable", "transformable", "can_transform"} and card.get("can_transform") is False:
            return False
        elif token in {"playable", "is_playable", "can_play"} and not bool(card.get("can_play", card.get("is_playable", True))):
            return False
        elif token in {"not_x_cost", "not_cost_x", "non_x_cost"} and _is_x_cost_card(card):
            return False
        elif token in {"x_cost", "cost_x", "costs_x", "is_x_cost"} and not _is_x_cost_card(card):
            return False
        elif token in {"without_replay", "no_replay"}:
            replay = card.get("replay") or card.get("replay_count") or card.get("base_replay_count")
            if replay:
                return False
    if positive_types and ctype not in positive_types:
        return False
    return True


def _scope_flags(op: dict[str, Any], *, default_one: bool = False) -> tuple[str, bool, bool, bool, bool]:
    """Return scope plus one/all/random/choice flags for typed operations."""

    scope_token = _norm_token(op.get("scope") or op.get("target_scope") or op.get("targetScope"))
    selection_token = _norm_token(op.get("selection") or op.get("selection_mode") or op.get("selectionMode"))
    count = _num_field(op, "count", "target_count", "targetCount", "max_count", "maxCount", default=0.0)

    all_scope = scope_token in {
        "all",
        "all_hand",
        "whole_hand",
        "hand_all",
        "each",
        "every",
        "all_cards",
    } or selection_token in {"all", "each", "every"}
    random_scope = scope_token in {"random", "random_one", "random_card"} or selection_token in {"random", "rng"}
    one_scope = scope_token in {"one", "single", "a_card", "one_card", "target"} or (
        count == 1.0 and not all_scope and not random_scope
    )
    choice = selection_token in {
        "choice",
        "choose",
        "select",
        "manual",
        "target",
        "targeted",
        "from_hand",
        "card_select",
        "card_selection",
    } or _bool_field(op, "choice_required", "requires_choice", "optional")
    if all_scope:
        return "all", False, True, random_scope, choice
    if random_scope:
        return "random", False, False, True, False
    if one_scope or default_one:
        return "one", True, False, False, True if default_one else choice
    return scope_token or "hand", False, False, False, choice


def _operation_expected_count(op: dict[str, Any], *, default: float = 1.0) -> float:
    for key in (
        "count",
        "target_count",
        "targetCount",
        "copy_count",
        "copyCount",
        "draw_count",
        "drawCount",
        "max_count",
        "maxCount",
        "stacks",
    ):
        if _has_num_field(op, key):
            return max(_num_field(op, key, default=default), 0.0)
    return default


def _source_card_index(source: dict[str, Any], hand_cards: list[dict[str, Any]]) -> int | None:
    """Best-effort index of the card being played in the pre-action hand.

    Internal card effects that iterate ``PileType.Hand`` usually run after the
    played card has left hand, so typed operation targets should not include the
    source card unless a profile explicitly says ``include_source``.
    """

    for i, card in enumerate(hand_cards):
        if card is source:
            return i
    unique_keys = (
        "instance_id",
        "instanceId",
        "uuid",
        "guid",
        "card_uuid",
        "cardUuid",
        "combat_card_id",
        "combatCardId",
    )
    for key in unique_keys:
        source_value = source.get(key)
        if source_value is None:
            continue
        for i, card in enumerate(hand_cards):
            if card.get(key) == source_value:
                return i
    source_id = str(source.get("id") or "").strip()
    source_title = str(source.get("title") or source.get("name") or "").strip()
    for i, card in enumerate(hand_cards):
        if source_id and str(card.get("id") or "").strip() == source_id:
            return i
        if source_title and str(card.get("title") or card.get("name") or "").strip() == source_title:
            return i
    return None


@dataclass(slots=True)
class MutationTarget:
    index: int
    card: dict[str, Any]
    affected: bool = False
    upgradeable: bool = False
    would_upgrade: bool = False
    would_cost_modify: bool = False
    would_discard: bool = False
    would_exhaust: bool = False
    would_transform: bool = False
    would_copy: bool = False
    would_gain_modifier: bool = False
    damage_delta: float = 0.0
    block_delta: float = 0.0
    cost_delta: float = 0.0
    draw_delta: float = 0.0
    energy_delta: float = 0.0
    playable_now: bool = False
    playable_after: bool = False
    followup_value: float = 0.0
    uncertain: bool = False


@dataclass(slots=True)
class HandMutationPlan:
    will_mutate_hand: bool = False
    upgrade_hand: bool = False
    upgrade_one: bool = False
    upgrade_all: bool = False
    upgrade_random: bool = False
    cost_modify_hand: bool = False
    set_cost_zero: bool = False
    draw_to_hand: bool = False
    add_to_hand: bool = False
    return_to_hand: bool = False
    discard_from_hand: bool = False
    exhaust_from_hand: bool = False
    transform_hand: bool = False
    copy_hand: bool = False
    modifier_hand: bool = False
    random_target: bool = False
    choice_required: bool = False
    temporary: bool = False
    affected_count: float = 0.0
    upgradeable_count: float = 0.0
    playable_followup_count: float = 0.0
    current_hand_damage: float = 0.0
    expected_damage_delta: float = 0.0
    current_hand_block: float = 0.0
    expected_block_delta: float = 0.0
    expected_cost_delta: float = 0.0
    best_single_target_value: float = 0.0
    same_turn_followup_possible: bool = False
    source_upgraded_scope_bonus: bool = False
    post_hand_count: float = 0.0
    post_upgrade_count: float = 0.0
    post_total_damage: float = 0.0
    post_total_block: float = 0.0
    post_total_draw: float = 0.0
    post_playable_cards: float = 0.0
    post_zero_cost_cards: float = 0.0
    post_energy_spendable_value: float = 0.0
    mutation_uncertainty: float = 0.0
    targets: list[MutationTarget] = field(default_factory=list)
    text: str = ""


def _infer_from_operations(
    source: dict[str, Any],
    hand_cards: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    *,
    current_energy: float = 0.0,
) -> HandMutationPlan:
    """Infer hand affordances from typed internal effect operations.

    This is the goal-state path.  It does not inspect localized card text; it
    consumes operation IDs plus explicit zone/scope/filter/count/duration
    parameters exported from source analysis / bridge payloads.
    """

    plan = HandMutationPlan()
    source_level = _upgrade_level(source)
    ops = [_merge_upgraded_override(op, source_level) for op in operations if isinstance(op, dict)]
    op_names = [_op_name(op) for op in ops]
    plan.text = "ops:" + ",".join(name for name in op_names if name)
    plan.current_hand_damage = sum(_preview(c, "damage") for c in hand_cards)
    plan.current_hand_block = sum(_preview(c, "block") for c in hand_cards)
    plan.upgradeable_count = float(sum(1 for c in hand_cards if _is_upgradeable(c)))
    played_source_index = _source_card_index(source, hand_cards)

    per_target: list[dict[str, Any]] = [
        {
            "affected": False,
            "would_upgrade": False,
            "would_cost_modify": False,
            "would_discard": False,
            "would_exhaust": False,
            "would_transform": False,
            "would_copy": False,
            "would_gain_modifier": False,
            "cost_delta": 0.0,
            "uncertain": False,
        }
        for _ in hand_cards
    ]
    total_expected_mutations = 0.0
    expected_removed_from_hand = 0.0
    expected_added_to_hand = 0.0
    expected_draw_to_hand = 0.0
    any_choice_or_random = False

    for op in ops:
        name = _op_name(op)
        if not name:
            continue
        src_zone = _zone_from(op, "source_zone", "sourceZone", "from_zone", "fromZone", "zone")
        dst_zone = _zone_from(op, "destination_zone", "destinationZone", "dest_zone", "destZone", "to_zone", "toZone")
        target_zone = _zone_from(op, "target_zone", "targetZone")
        if not target_zone:
            target_zone = src_zone
        filters = _operation_filters(op)
        if name == "upgrade_card":
            filters = set(filters)
            filters.add("is_upgradable")

        hand_source = _is_hand_zone(src_zone)
        hand_dest = _is_hand_zone(dst_zone)
        hand_target = _is_hand_zone(target_zone) or hand_source
        source_or_target_hand = hand_source or hand_target
        include_source = _bool_field(op, "include_source", "includeSource", "can_target_source", "canTargetSource")
        candidates = [
            i
            for i, card in enumerate(hand_cards)
            if _matches_operation_filter(card, filters) and (include_source or played_source_index is None or i != played_source_index)
        ]

        default_one = name in {
            "upgrade_card",
            "exhaust_card",
            "discard_card",
            "transform_card",
            "copy_card",
            "modify_cost",
            "add_modifier",
            "add_keyword",
            "set_replay",
            "retain_card",
        }
        scope, one_scope, all_scope, random_scope, choice = _scope_flags(op, default_one=default_one)
        expected_count = _operation_expected_count(op, default=1.0)
        if all_scope:
            target_indices = set(candidates)
            op_expected_targets = float(len(target_indices))
        elif source_or_target_hand:
            target_indices = set(candidates)
            op_expected_targets = min(max(expected_count, 1.0), float(len(target_indices))) if target_indices else expected_count
        else:
            target_indices = set()
            op_expected_targets = expected_count

        if random_scope or choice or (one_scope and len(target_indices) > 1):
            any_choice_or_random = True
        for idx in target_indices:
            per_target[idx]["affected"] = True
            per_target[idx]["uncertain"] = bool(per_target[idx]["uncertain"] or random_scope or choice or one_scope)

        if name == "upgrade_card" and source_or_target_hand:
            plan.will_mutate_hand = True
            plan.upgrade_hand = True
            plan.upgrade_one = bool(plan.upgrade_one or (one_scope and not all_scope and not random_scope))
            plan.upgrade_all = bool(plan.upgrade_all or all_scope)
            plan.upgrade_random = bool(plan.upgrade_random or random_scope)
            plan.random_target = bool(plan.random_target or random_scope)
            plan.choice_required = bool(plan.choice_required or choice or one_scope)
            for idx in target_indices:
                per_target[idx]["would_upgrade"] = bool(_is_upgradeable(hand_cards[idx]))
            total_expected_mutations += op_expected_targets
            if source_level > 0 and (op.get("upgraded_override") or op.get("upgradedOverride") or op.get("upgraded")):
                plan.source_upgraded_scope_bonus = True

        elif name == "modify_cost" and source_or_target_hand:
            plan.will_mutate_hand = True
            plan.cost_modify_hand = True
            set_cost_present = _has_num_field(op, "set_cost", "setCost", "to_cost", "toCost", "cost")
            set_cost = _num_field(op, "set_cost", "setCost", "to_cost", "toCost", "cost", default=0.0)
            reduce_by = _num_field(op, "reduce_by", "reduceBy", "cost_delta", "costDelta", default=0.0)
            reduce_only = _bool_field(op, "reduce_only", "reduceOnly")
            if set_cost_present and set_cost <= 0.0:
                plan.set_cost_zero = True
            duration = op.get("duration") or op.get("timing") or op.get("until")
            plan.temporary = bool(plan.temporary or _duration_is_temporary(duration))
            plan.choice_required = bool(plan.choice_required or choice)
            plan.random_target = bool(plan.random_target or random_scope)
            for idx in target_indices:
                base_cost = _cost(hand_cards[idx])
                if set_cost_present:
                    delta = set_cost - base_cost
                    if reduce_only:
                        delta = min(delta, 0.0)
                elif reduce_by:
                    delta = -abs(reduce_by) if base_cost > 0 else 0.0
                else:
                    delta = -1.0 if base_cost > 0 else 0.0
                per_target[idx]["would_cost_modify"] = True
                per_target[idx]["cost_delta"] = float(per_target[idx]["cost_delta"]) + delta
            total_expected_mutations += op_expected_targets

        elif name == "draw_card":
            plan.will_mutate_hand = True
            plan.draw_to_hand = True
            draw_count = _operation_expected_count(op, default=1.0)
            expected_draw_to_hand += draw_count
            total_expected_mutations += draw_count

        elif name in {"add_generated_card", "add_card"}:
            add_count = _operation_expected_count(op, default=1.0)
            if hand_dest or not dst_zone:
                plan.will_mutate_hand = True
                plan.add_to_hand = True
                expected_added_to_hand += add_count
                total_expected_mutations += add_count

        elif name == "move_card":
            move_count = _operation_expected_count(op, default=1.0)
            if hand_source:
                plan.will_mutate_hand = True
                if _is_discard_zone(dst_zone):
                    plan.discard_from_hand = True
                if _is_exhaust_zone(dst_zone):
                    plan.exhaust_from_hand = True
                if _is_draw_zone(dst_zone):
                    plan.return_to_hand = False
                expected_removed_from_hand += op_expected_targets if target_indices else move_count
                total_expected_mutations += op_expected_targets
                for idx in target_indices:
                    if _is_discard_zone(dst_zone):
                        per_target[idx]["would_discard"] = True
                    if _is_exhaust_zone(dst_zone):
                        per_target[idx]["would_exhaust"] = True
            elif hand_dest:
                plan.will_mutate_hand = True
                plan.add_to_hand = True
                plan.return_to_hand = bool(_is_discard_zone(src_zone) or _is_exhaust_zone(src_zone) or _is_draw_zone(src_zone))
                expected_added_to_hand += move_count
                total_expected_mutations += move_count

        elif name in {"discard_card", "exhaust_card", "transform_card"} and source_or_target_hand:
            plan.will_mutate_hand = True
            if name == "discard_card":
                plan.discard_from_hand = True
            elif name == "exhaust_card":
                plan.exhaust_from_hand = True
            else:
                plan.transform_hand = True
            plan.choice_required = bool(plan.choice_required or choice)
            plan.random_target = bool(plan.random_target or random_scope)
            for idx in target_indices:
                per_target[idx]["would_discard"] = bool(per_target[idx]["would_discard"] or name == "discard_card")
                per_target[idx]["would_exhaust"] = bool(per_target[idx]["would_exhaust"] or name == "exhaust_card")
                per_target[idx]["would_transform"] = bool(per_target[idx]["would_transform"] or name == "transform_card")
            if name in {"discard_card", "exhaust_card"}:
                expected_removed_from_hand += op_expected_targets
            total_expected_mutations += op_expected_targets

        elif name == "copy_card" and (source_or_target_hand or hand_dest):
            plan.will_mutate_hand = True
            plan.copy_hand = True
            plan.choice_required = bool(plan.choice_required or choice or one_scope)
            plan.random_target = bool(plan.random_target or random_scope)
            copy_count = _operation_expected_count(op, default=1.0)
            if hand_dest or not dst_zone:
                plan.add_to_hand = True
                expected_added_to_hand += copy_count
            for idx in target_indices:
                per_target[idx]["would_copy"] = True
            total_expected_mutations += max(op_expected_targets, copy_count)

        elif name in {"add_modifier", "add_keyword", "set_replay", "retain_card"} and source_or_target_hand:
            plan.will_mutate_hand = True
            plan.modifier_hand = True
            duration = op.get("duration") or op.get("timing") or op.get("until")
            plan.temporary = bool(plan.temporary or _duration_is_temporary(duration))
            plan.choice_required = bool(plan.choice_required or choice)
            plan.random_target = bool(plan.random_target or random_scope)
            for idx in target_indices:
                per_target[idx]["would_gain_modifier"] = True
            total_expected_mutations += op_expected_targets

        elif name == "apply_power":
            # Powers like Corruption/FreeAttack do not mutate current hand
            # immediately, but they rewrite future card rules.  Expose a weak
            # hand-rule signal when the typed profile declares such a rule.
            future_rule = _norm_token(op.get("future_rule") or op.get("futureRule") or op.get("rule"))
            if any(token in future_rule for token in ("cost_zero", "free", "exhaust_on_play", "retain", "replay")):
                plan.will_mutate_hand = True
                plan.modifier_hand = True
                plan.cost_modify_hand = bool(plan.cost_modify_hand or "cost_zero" in future_rule or "free" in future_rule)
                plan.exhaust_from_hand = bool(plan.exhaust_from_hand or "exhaust_on_play" in future_rule)
                total_expected_mutations += 1.0

    if not plan.will_mutate_hand:
        return plan

    plan.affected_count = max(total_expected_mutations, 1.0 if any(t["affected"] for t in per_target) else 0.0)
    target_rows: list[MutationTarget] = []
    for i, card in enumerate(hand_cards):
        flags = per_target[i]
        affected = bool(flags["affected"])
        upg = _is_upgradeable(card)
        base_damage = _preview(card, "damage")
        base_block = _preview(card, "block")
        base_draw = _preview(card, "draw")
        base_energy = obs_common._get_card_extra_metrics(card)[2]
        base_cost = _cost(card)
        dmg_delta = 0.0
        block_delta = 0.0
        draw_delta = 0.0
        energy_delta = 0.0
        if affected and bool(flags["would_upgrade"]) and upg:
            dmg_delta = max(base_damage * 0.25, 3.0 if base_damage > 0 else 0.0)
            block_delta = max(base_block * 0.25, 2.0 if base_block > 0 else 0.0)
            draw_delta = 1.0 if "draw" in _card_text(card) else 0.0
        cost_delta = float(flags["cost_delta"])
        playable_now = base_cost <= max(current_energy, 0.0)
        playable_after = max(base_cost + cost_delta, 0.0) <= max(current_energy, 0.0)
        followup = (
            max(base_damage + dmg_delta, 0.0)
            + 0.75 * max(base_block + block_delta, 0.0)
            + 2.0 * max(base_draw + draw_delta, 0.0)
            + 2.0 * max(base_energy + energy_delta, 0.0)
            - max(base_cost + cost_delta, 0.0)
        )
        target_rows.append(
            MutationTarget(
                index=i,
                card=card,
                affected=affected,
                upgradeable=upg,
                would_upgrade=bool(flags["would_upgrade"] and upg),
                would_cost_modify=bool(flags["would_cost_modify"]),
                would_discard=bool(flags["would_discard"]),
                would_exhaust=bool(flags["would_exhaust"]),
                would_transform=bool(flags["would_transform"]),
                would_copy=bool(flags["would_copy"]),
                would_gain_modifier=bool(flags["would_gain_modifier"]),
                damage_delta=dmg_delta,
                block_delta=block_delta,
                cost_delta=cost_delta,
                draw_delta=draw_delta,
                energy_delta=energy_delta,
                playable_now=playable_now,
                playable_after=playable_after,
                followup_value=followup,
                uncertain=bool(flags["uncertain"]),
            )
        )

    affected_targets = [t for t in target_rows if t.affected]
    scale = 1.0 / max(len(affected_targets), 1) if any_choice_or_random else 1.0
    plan.expected_damage_delta = sum(t.damage_delta for t in affected_targets) * scale
    plan.expected_block_delta = sum(t.block_delta for t in affected_targets) * scale
    plan.expected_cost_delta = sum(t.cost_delta for t in affected_targets) * scale
    plan.best_single_target_value = max((t.followup_value for t in affected_targets), default=0.0)
    plan.playable_followup_count = sum(1.0 for t in target_rows if t.playable_after and not (t.would_discard or t.would_exhaust))
    plan.same_turn_followup_possible = any(t.affected and t.playable_after and not (t.would_discard or t.would_exhaust) for t in target_rows)
    plan.mutation_uncertainty = 0.25 if any_choice_or_random else 0.0

    plan.post_hand_count = max(float(len(hand_cards)) + expected_draw_to_hand + expected_added_to_hand - expected_removed_from_hand, 0.0)
    plan.post_upgrade_count = sum(1.0 for c in hand_cards if not _is_upgradeable(c)) + sum(1.0 for t in target_rows if t.would_upgrade) * scale
    plan.post_total_damage = max(plan.current_hand_damage + plan.expected_damage_delta, 0.0)
    plan.post_total_block = max(plan.current_hand_block + plan.expected_block_delta, 0.0)
    plan.post_total_draw = sum(_preview(c, "draw") for c in hand_cards) + sum(t.draw_delta for t in affected_targets) * scale
    plan.post_playable_cards = plan.playable_followup_count
    plan.post_zero_cost_cards = sum(1.0 for c in hand_cards if _cost(c) <= 0.0) + sum(
        1.0 for t in affected_targets if t.cost_delta < 0 and max(_cost(t.card) + t.cost_delta, 0.0) == 0.0
    ) * scale
    plan.post_energy_spendable_value = (
        sum(max(_preview(t.card, "damage"), _preview(t.card, "block")) for t in target_rows if t.playable_after) / 20.0
    )
    target_rows.sort(key=lambda t: (float(t.affected), float(t.would_upgrade or t.would_cost_modify), t.followup_value), reverse=True)
    plan.targets = target_rows
    return plan


def infer_hand_mutation(
    source: dict[str, Any] | None,
    hand: list[Any] | None,
    draw_pile: list[Any] | None = None,
    discard_pile: list[Any] | None = None,
    exhaust_pile: list[Any] | None = None,
    *,
    current_energy: float = 0.0,
) -> HandMutationPlan:
    plan = HandMutationPlan()
    if not isinstance(source, dict):
        return plan
    hand_cards = [c for c in (hand or []) if isinstance(c, dict)]
    typed_ops = _profile_operations(source)
    if typed_ops:
        return _infer_from_operations(source, hand_cards, typed_ops, current_energy=current_energy)

    text = _card_text(source)
    plan.text = text
    if not text:
        return plan

    source_level = _upgrade_level(source)
    mentions_hand = "hand" in text or "cards in your hand" in text or "your cards" in text
    upgrade = "upgrade" in text or "smith" in text
    cost_mod = any(t in text for t in ("cost 0", "costs 0", "cost to 0", "reduce the cost", "lower the cost", "set its cost", "set the cost"))
    set_zero = any(t in text for t in ("cost 0", "costs 0", "cost to 0", "cost is 0", "set its cost to 0", "set the cost to 0"))
    draw = bool(re.search(r"\bdraw\s+\d+", text) or "draw a card" in text or "draw cards" in text)
    add_to_hand = any(t in text for t in ("add", "create", "put")) and "hand" in text
    return_to_hand = any(t in text for t in ("return", "retrieve")) and "hand" in text
    discard = "discard" in text and mentions_hand
    exhaust = ("exhaust" in text or "consume" in text) and mentions_hand
    transform = ("transform" in text or "mutate" in text or "change" in text) and mentions_hand
    copy = ("copy" in text or "duplicate" in text) and mentions_hand
    modifier = any(t in text for t in ("retain", "ethereal", "bound", "temporary", "enchant", "afflict")) and mentions_hand
    temporary = "this turn" in text or "temporary" in text or "until played" in text

    if not any((upgrade and mentions_hand, cost_mod and mentions_hand, draw, add_to_hand, return_to_hand, discard, exhaust, transform, copy, modifier)):
        return plan

    # Upgraded cards often change one-card hand effects into all-hand effects.
    scope, one_scope, all_scope, random_scope, choice = _infer_scope(text, default_one=bool(mentions_hand and (upgrade or cost_mod or discard or exhaust or transform or copy or modifier)))
    if upgrade and mentions_hand and source_level > 0 and not random_scope and not all_scope:
        # Generic upgraded-scope exposure: if live text already says all, the text
        # path sets this.  If source only has title+/upgrade_level but stale text,
        # expose a soft "scope bonus" instead of hardcoding Armaments.
        plan.source_upgraded_scope_bonus = True

    filters = _type_filter(text)
    upgradeable_cards = [c for c in hand_cards if _is_upgradeable(c) and _matches_filter(c, filters)]
    candidate_cards = [c for c in hand_cards if _matches_filter(c, filters)] or hand_cards
    if all_scope:
        affected_indices = set(range(len(hand_cards))) if not filters else {i for i, c in enumerate(hand_cards) if _matches_filter(c, filters)}
    elif random_scope or one_scope:
        # For choice/random single-target effects, expose all plausible target
        # cards; aggregate count remains expected one target.
        affected_indices = {i for i, c in enumerate(hand_cards) if c in candidate_cards}
    else:
        affected_indices = {i for i, c in enumerate(hand_cards) if c in candidate_cards}

    plan.will_mutate_hand = True
    plan.upgrade_hand = bool(upgrade and mentions_hand)
    plan.upgrade_one = bool(plan.upgrade_hand and one_scope and not all_scope and not random_scope)
    plan.upgrade_all = bool(plan.upgrade_hand and all_scope)
    plan.upgrade_random = bool(plan.upgrade_hand and random_scope)
    plan.cost_modify_hand = bool(cost_mod and mentions_hand)
    plan.set_cost_zero = bool(plan.cost_modify_hand and set_zero)
    plan.draw_to_hand = bool(draw)
    plan.add_to_hand = bool(add_to_hand)
    plan.return_to_hand = bool(return_to_hand)
    plan.discard_from_hand = bool(discard)
    plan.exhaust_from_hand = bool(exhaust)
    plan.transform_hand = bool(transform)
    plan.copy_hand = bool(copy)
    plan.modifier_hand = bool(modifier)
    plan.random_target = bool(random_scope)
    plan.choice_required = bool(choice)
    plan.temporary = bool(temporary)

    plan.current_hand_damage = sum(_preview(c, "damage") for c in hand_cards)
    plan.current_hand_block = sum(_preview(c, "block") for c in hand_cards)
    plan.upgradeable_count = float(len(upgradeable_cards))

    effective_expected_count = 1.0 if (one_scope or random_scope) and not all_scope else float(len(affected_indices))
    if plan.draw_to_hand:
        m = re.search(r"draw\s+(\d+)", text)
        draw_count = float(m.group(1)) if m else 1.0
        effective_expected_count += draw_count
    if plan.add_to_hand or plan.return_to_hand:
        effective_expected_count += 1.0
    if plan.discard_from_hand or plan.exhaust_from_hand or plan.transform_hand:
        effective_expected_count = max(effective_expected_count, 1.0)
    plan.affected_count = effective_expected_count

    target_rows: list[MutationTarget] = []
    for i, card in enumerate(hand_cards):
        affected = i in affected_indices
        upg = _is_upgradeable(card)
        base_damage = _preview(card, "damage")
        base_block = _preview(card, "block")
        base_draw = _preview(card, "draw")
        base_energy = obs_common._get_card_extra_metrics(card)[2]
        base_cost = _cost(card)
        dmg_delta = 0.0
        block_delta = 0.0
        draw_delta = 0.0
        energy_delta = 0.0
        cost_delta = 0.0
        if affected and plan.upgrade_hand and upg:
            # Generic upgrade preview when exact upgraded payload is unavailable.
            dmg_delta = max(base_damage * 0.25, 3.0 if base_damage > 0 else 0.0)
            block_delta = max(base_block * 0.25, 2.0 if base_block > 0 else 0.0)
            draw_delta = 1.0 if "draw" in _card_text(card) else 0.0
        if affected and plan.cost_modify_hand:
            if plan.set_cost_zero:
                cost_delta = -base_cost
            else:
                cost_delta = -1.0 if base_cost > 0 else 0.0
        playable_now = base_cost <= max(current_energy, 0.0)
        playable_after = max(base_cost + cost_delta, 0.0) <= max(current_energy, 0.0)
        followup = max(base_damage + dmg_delta, 0.0) + 0.75 * max(base_block + block_delta, 0.0) + 2.0 * max(base_draw + draw_delta, 0.0) + 2.0 * max(base_energy + energy_delta, 0.0) - max(base_cost + cost_delta, 0.0)
        target_rows.append(
            MutationTarget(
                index=i,
                card=card,
                affected=affected,
                upgradeable=upg,
                would_upgrade=bool(affected and plan.upgrade_hand and upg),
                would_cost_modify=bool(affected and plan.cost_modify_hand),
                would_discard=bool(affected and plan.discard_from_hand),
                would_exhaust=bool(affected and plan.exhaust_from_hand),
                would_transform=bool(affected and plan.transform_hand),
                would_copy=bool(affected and plan.copy_hand),
                would_gain_modifier=bool(affected and plan.modifier_hand),
                damage_delta=dmg_delta,
                block_delta=block_delta,
                cost_delta=cost_delta,
                draw_delta=draw_delta,
                energy_delta=energy_delta,
                playable_now=playable_now,
                playable_after=playable_after,
                followup_value=followup,
                uncertain=bool(random_scope or (one_scope and len(affected_indices) > 1)),
            )
        )

    affected_targets = [t for t in target_rows if t.affected]
    if one_scope or random_scope:
        scale = 1.0 / max(len(affected_targets), 1)
    else:
        scale = 1.0
    plan.expected_damage_delta = sum(t.damage_delta for t in affected_targets) * scale
    plan.expected_block_delta = sum(t.block_delta for t in affected_targets) * scale
    plan.expected_cost_delta = sum(t.cost_delta for t in affected_targets) * scale
    plan.best_single_target_value = max((t.followup_value for t in affected_targets), default=0.0)
    plan.playable_followup_count = sum(1.0 for t in target_rows if t.playable_after and not (t.would_discard or t.would_exhaust))
    plan.same_turn_followup_possible = any(t.affected and t.playable_after and not (t.would_discard or t.would_exhaust) for t in target_rows)
    # Text is now a missing-profile fallback, so keep a non-zero uncertainty
    # floor even when the phrase looks deterministic.
    plan.mutation_uncertainty = max(0.35, 1.0 if (random_scope or one_scope or choice) else 0.0)

    post_hand_count = float(len(hand_cards))
    if plan.draw_to_hand:
        post_hand_count += max(plan.affected_count - (1.0 if (one_scope or random_scope) else float(len(affected_indices))), 1.0)
    if plan.discard_from_hand or plan.exhaust_from_hand:
        post_hand_count -= effective_expected_count if (one_scope or random_scope) else min(effective_expected_count, len(hand_cards))
    if plan.copy_hand or plan.add_to_hand or plan.return_to_hand:
        post_hand_count += 1.0
    plan.post_hand_count = max(post_hand_count, 0.0)
    plan.post_upgrade_count = sum(1.0 for c in hand_cards if not _is_upgradeable(c)) + sum(1.0 for t in target_rows if t.would_upgrade) * (scale if (one_scope or random_scope) else 1.0)
    plan.post_total_damage = max(plan.current_hand_damage + plan.expected_damage_delta, 0.0)
    plan.post_total_block = max(plan.current_hand_block + plan.expected_block_delta, 0.0)
    plan.post_total_draw = sum(_preview(c, "draw") for c in hand_cards) + sum(t.draw_delta for t in affected_targets) * scale
    plan.post_playable_cards = plan.playable_followup_count
    plan.post_zero_cost_cards = sum(1.0 for c in hand_cards if _cost(c) <= 0.0) + sum(1.0 for t in affected_targets if t.cost_delta < 0 and max(_cost(t.card) + t.cost_delta, 0.0) == 0.0) * scale
    plan.post_energy_spendable_value = sum(max(_preview(t.card, "damage"), _preview(t.card, "block")) for t in target_rows if t.playable_after) / 20.0

    # Keep valuable targets first; encoder can then budget the most useful rows.
    target_rows.sort(key=lambda t: (float(t.affected), float(t.would_upgrade or t.would_cost_modify), t.followup_value), reverse=True)
    plan.targets = target_rows
    return plan


def mutation_summary_numeric(plan: HandMutationPlan) -> list[float]:
    row = [0.0] * 96
    row[0] = float(plan.will_mutate_hand)
    row[1] = float(plan.upgrade_hand)
    row[2] = float(plan.upgrade_one)
    row[3] = float(plan.upgrade_all)
    row[4] = float(plan.upgrade_random)
    row[5] = float(plan.cost_modify_hand)
    row[6] = float(plan.set_cost_zero)
    row[7] = float(plan.draw_to_hand)
    row[8] = float(plan.add_to_hand)
    row[9] = float(plan.return_to_hand)
    row[10] = float(plan.discard_from_hand)
    row[11] = float(plan.exhaust_from_hand)
    row[12] = float(plan.transform_hand)
    row[13] = float(plan.copy_hand)
    row[14] = float(plan.modifier_hand)
    row[15] = float(plan.random_target)
    row[16] = float(plan.choice_required)
    row[17] = float(plan.temporary)
    row[18] = _norm_count(plan.affected_count)
    row[19] = _norm_count(plan.upgradeable_count)
    row[20] = _norm_count(plan.playable_followup_count)
    row[21] = min(plan.current_hand_damage / 80.0, 1.0)
    row[22] = _signed_norm(plan.expected_damage_delta, 30.0)
    row[23] = min(plan.current_hand_block / 80.0, 1.0)
    row[24] = _signed_norm(plan.expected_block_delta, 30.0)
    row[25] = _signed_norm(plan.expected_cost_delta, 5.0)
    row[26] = min(max(plan.best_single_target_value, 0.0) / 40.0, 1.0)
    row[27] = float(plan.same_turn_followup_possible)
    row[28] = float(plan.source_upgraded_scope_bonus)
    return row


def mutation_target_numeric(target: MutationTarget) -> list[float]:
    row = [0.0] * 96
    row[0] = float(target.affected)
    row[1] = float(target.upgradeable)
    row[2] = min(_upgrade_level(target.card) / 3.0, 1.0)
    row[3] = float(target.would_upgrade)
    row[4] = float(target.would_cost_modify)
    row[5] = float(target.would_discard)
    row[6] = float(target.would_exhaust)
    row[7] = float(target.would_transform)
    row[8] = float(target.would_copy)
    row[9] = float(target.would_gain_modifier)
    row[10] = _signed_norm(target.damage_delta, 30.0)
    row[11] = _signed_norm(target.block_delta, 30.0)
    row[12] = _signed_norm(target.cost_delta, 5.0)
    row[13] = _signed_norm(target.draw_delta, 5.0)
    row[14] = _signed_norm(target.energy_delta, 5.0)
    row[15] = min(_cost(target.card) / 5.0, 1.0)
    row[16] = float(target.playable_now)
    row[17] = float(target.playable_after)
    row[18] = min(max(target.followup_value, 0.0) / 40.0, 1.0)
    row[19] = float(target.uncertain)
    row[20] = _norm_count(target.index + 1)
    return row


def post_hand_preview_numeric(plan: HandMutationPlan) -> list[float]:
    row = [0.0] * 96
    row[0] = _norm_count(plan.post_hand_count)
    row[1] = _norm_count(plan.post_upgrade_count)
    row[2] = min(plan.post_total_damage / 80.0, 1.0)
    row[3] = min(plan.post_total_block / 80.0, 1.0)
    row[4] = min(plan.post_total_draw / 10.0, 1.0)
    row[5] = _norm_count(plan.post_playable_cards)
    row[6] = _norm_count(plan.post_zero_cost_cards)
    row[7] = min(max(plan.post_energy_spendable_value, 0.0), 1.0)
    row[8] = min(max(plan.expected_damage_delta, 0.0) / max(plan.current_hand_damage, 1.0), 1.0)
    row[9] = min(max(plan.expected_block_delta, 0.0) / max(plan.current_hand_block, 1.0), 1.0)
    row[10] = min(max(plan.mutation_uncertainty, 0.0), 1.0)
    return row
