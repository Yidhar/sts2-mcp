"""Encode raw environment state without domain rules or candidate leakage.

The old observation stack embedded card, boss, route and action-quality rules in
the input itself.  That made the learner inseparable from a hand-written
planner.  This module deliberately implements a much smaller contract:

* world state and action candidates are encoded by disjoint code paths;
* field names and categorical values are stable-hashed, never interpreted as
  card/boss mechanics;
* numeric values receive one generic, bounded transform;
* dispatch handles, bridge internals and retired engineered fields are never
  model inputs; and
* list order is represented, while candidate list order is not.

It is intentionally a structural encoder rather than a game strategy module.
The model must learn the meaning of entity IDs and numeric state from data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch

from sts2_rl.models.grounded_candidate import (
    MIN_TOKEN_FEATURE_DIM,
    GroundedCandidateBatch,
)

from .snapshot import (
    ENCODED_DECISION_SNAPSHOT_VERSION,
    EncodedDecisionSnapshot,
    GroundedEncodingConfig,
    collate_encoded_snapshots,
    sparse_token_table,
)

_DOMAIN_IDS: Final[dict[str, int]] = {
    "unknown": 0,
    "combat": 1,
    "build": 2,
    "route": 3,
    "terminal": 4,
    "chance": 5,
}
MODEL_ACTION_KIND_VOCABULARY: Final[frozenset[str]] = frozenset(
    {
        "card_reward",
        "card_selection",
        "character_select",
        "deck_upgrade",
        "discard_potion",
        "end_turn",
        "event_option",
        "game_over",
        "main_menu",
        "map",
        "play_card",
        "proceed",
        "rest_site",
        "reward",
        "run_mode_selection",
        "shop",
        "treasure",
        "treasure_relic",
        "use_potion",
    }
)

# These keys are transport identity, duplicated compatibility views, retired
# engineered targets, or candidate containers.  Excluding them is an
# architectural firewall, not an action-selection rule.
_WORLD_EXCLUDED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action_candidates",
        "available_actions",
        "legal_actions",
        "actions",
        "action_history",
        "canonical_text",
        "captured_at_utc",
        "combat_id",
        "episode_id",
        "instance_id",
        "instance_uuid",
        "request_id",
        "session_id",
        "state_version",
        "step_index",
        "state_hash",
        "semantic_state_hash",
        "route_summary",
        "effect_deltas",
        "effect_preview",
        "semantic",
        "action_semantic",
        "preview",
        "card_effect_profile",
        "effect_profile",
        "derived_view",
        "semantic_tags",
        "timing_tags",
        "training_tags",
        "static_traits",
        "reactive_triggers",
        "phase_rules",
        "combat_tags",
        "danger_profile",
        "target_priority_hints",
        "aux_targets",
        "objective_targets",
        "objective_context",
        "boss_mechanics",
        "combat_tactical",
        "potion_timing",
    }
)
_CANDIDATE_EXCLUDED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action_handle",
        "action_id",
        "action_index",
        "card_index",
        "target_id",
        "target_handle",
        "target_combat_id",
        "option_index",
        "selection_id",
        "slot_index",
        "slot",
        "idx",
        "index",
        "combat_id",
        "instance_id",
        "instance_uuid",
        "is_enabled",
        "enabled",
        "transport_kind",
        "semantic",
        "action_semantic",
        "preview",
        "canonical_text",
        "route_summary",
        "effect_deltas",
        "effect_preview",
        "card_effect_profile",
        "effect_profile",
        "derived_view",
        "semantic_tags",
        "timing_tags",
        "training_tags",
        "static_traits",
        "reactive_triggers",
        "phase_rules",
        "combat_tags",
        "danger_profile",
        "target_priority_hints",
        # Compatibility aliases duplicate canonical nested objects.
        "shop_item",
        "rest_site_option",
        "treasure_relic",
        "event_option",
    }
)
_IDENTITY_KEYS: Final[tuple[str, ...]] = (
    "model_id",
    "entity_id",
    "id",
    "card_id",
    "potion_id",
    "relic_id",
    "enemy_id",
    "character_id",
    "encounter_id",
    "event_id",
    "option_id",
    "text_key",
    "room_model_id",
    "room_model",
)
_ROLE_KEYS: Final[tuple[str, ...]] = (
    "kind",
    "type",
    "category",
    "phase",
    "status",
    "point_type",
    "selection",
)
_OWNER_KEYS: Final[tuple[str, ...]] = ("owner", "side", "controller", "team")

# Only primitive mechanics/state measurements enter numeric feature hashing.
# This positive contract prevents a newly-added ``quality_score`` or planner
# estimate from entering the policy merely because the Bridge emitted a number.
_FACT_NUMERIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "act",
        "amount",
        "amount_on_turn_start",
        "ascension",
        "block",
        "charges",
        "col",
        "column",
        "cost",
        "count",
        "current",
        "current_energy",
        "current_hp",
        "current_stars",
        "damage",
        "damage_per_hit",
        "dexterity",
        "discard",
        "display_amount",
        "draw",
        "energy",
        "evoke_val",
        "exhaust",
        "floor",
        "floor_added_to_deck",
        "focus",
        "gold",
        "height",
        "hits",
        "hp",
        "level",
        "max",
        "max_count",
        "max_energy",
        "max_hp",
        "max_stars",
        "max_select",
        "merchant_cost",
        "min_count",
        "min_select",
        "open_potion_slots",
        "orb_empty_slots",
        "orb_slots",
        "passive_val",
        "poison",
        "price",
        "progress",
        "quantity",
        "remaining_select",
        "repeats",
        "round",
        "row",
        "stars",
        "selected_count",
        "stack_count",
        "strength",
        "turn",
        "total_damage",
        "upgrade_level",
        "upgrades",
        "width",
        "x",
        "y",
    }
)
_CARD_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {
        "card",
        "cards",
        "card_reward",
        "deck",
        "deck_cards",
        "discard_pile",
        "draw_pile",
        "exhaust_pile",
        "hand",
        "play_pile",
        "selectable_cards",
        "selected_cards",
        "upgrade_preview",
    }
)
_CARD_FACT_NUMERIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "amount",
        "base_replay_count",
        "cost",
        "current_replay_count",
        "current_upgrade_level",
        "display_amount",
        "energy_cost",
        "floor_added_to_deck",
        "last_stars_spent",
        "max_upgrade_level",
        "price",
        "stack_count",
        "star_cost",
        "upgrade_level",
        "upgrades",
    }
)
_CARD_FACT_BOOLEAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "can_afflict_unplayable_cards",
        "has_been_removed_from_state",
        "has_extra_card_text",
        "has_overlay",
        "is_clone",
        "is_dupe",
        "is_in_combat",
        "is_selected",
        "is_playable",
        "is_removable",
        "is_retained",
        "is_sly_this_turn",
        "is_stackable",
        "is_transformable",
        "is_upgraded",
        "is_upgradable",
        "playable",
        "should_glow_gold",
        "should_glow_red",
        "should_start_at_bottom_of_draw_pile",
        "show_amount",
        "upgraded",
    }
)
_FACT_CATEGORICAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        *_IDENTITY_KEYS,
        *_OWNER_KEYS,
        *_ROLE_KEYS,
        "action",
        "character",
        "character_id",
        "class_name",
        "confirmation_mode",
        "decision_domain",
        "description_key",
        "facing",
        "family",
        "encounter_id",
        "event_id",
        "intent_type",
        "destination_zone",
        "layout_type",
        "mode",
        "next_move_id",
        "next_move_state_id",
        "operation_type",
        "owner_model_id",
        "owner_side",
        "applier_model_id",
        "applier_side",
        "pile",
        "prompt_id",
        "power_type",
        "rarity",
        "room_model",
        "room_model_id",
        "room_type",
        "run_mode_action",
        "screen",
        "selection_membership",
        "slot_name",
        "source_zone",
        "stack_type",
        "state_type",
        "status",
        "target_type",
        "target_model_id",
        "target_side",
        "text_key",
        "transport_kind",
        "type_for_current_amount",
        "usage",
        "value_props",
        "var_type",
        "visibility",
        "zone",
    }
)
_FACT_NUMERIC_SUFFIXES: Final[tuple[str, ...]] = (
    # Deliberately empty: suffix matching admitted arbitrary new engineered
    # fields such as ``predicted_damage``. New factual measurements must be
    # added to the explicit, reviewed key contract above.
)
_FACT_BOOLEAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "active",
        "adds_pet",
        "allow_negative",
        "can_afflict_unplayable_cards",
        "can_be_generated_in_combat",
        "can_receive_powers",
        "can_skip",
        "can_throw_at_ally",
        "can_transition_away",
        "can_use_in_combat",
        "cancelable",
        "confirm_ready",
        "exhaust",
        "exhausts",
        "ethereal",
        "has_been_removed_from_state",
        "has_extra_card_text",
        "has_overlay",
        "has_upon_pickup_effect",
        "in_combat",
        "in_dialogue",
        "in_progress",
        "is_allowed_in_shops",
        "is_alive",
        "is_deterministic",
        "is_finished",
        "is_hittable",
        "is_instanced",
        "is_chosen",
        "is_melted",
        "is_move",
        "is_performing_move",
        "is_pet",
        "is_playable",
        "is_primary_enemy",
        "is_queued",
        "is_secondary_enemy",
        "is_selected",
        "is_shared",
        "is_stackable",
        "is_stunned",
        "is_tradable",
        "is_upgraded",
        "is_used_up",
        "is_proceed",
        "is_visible",
        "is_wax",
        "must_perform_once_before_transitioning",
        "owner_is_secondary_enemy",
        "passes_custom_usability_check",
        "playable",
        "retain",
        "retained",
        "run_active",
        "should_glow_gold",
        "should_glow_red",
        "should_scale_in_multiplayer",
        "should_start_at_bottom_of_draw_pile",
        "show_amount",
        "show_counter",
        "shows_infinite_hp",
        "skip_next_duration_tick",
        "spawned_this_turn",
        "spawns_pets",
        "upgraded",
    }
)
_ENGINEERED_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "action_quality",
    "bias",
    "estimate",
    "guard",
    "heuristic",
    "objective_target",
    "planner",
    "prediction",
    "preview",
    "quality",
    "recommend",
    "root_prior",
    "safety",
    "score",
    "strategy",
)
_FACT_CONTAINER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "affliction",
        "afflictions",
        "allies",
        "card",
        "card_reward",
        "card_reward_selection",
        "card_selection",
        "cards",
        "character",
        "children",
        "combat",
        "coord",
        "creature",
        "deck",
        "deck_cards",
        "discard_pile",
        "draw_pile",
        "dynamic_vars",
        "enchantment",
        "enchantments",
        "enemies",
        "event",
        "exhaust_pile",
        "game_over",
        "hand",
        "hand_select",
        "hover_tips",
        "intents",
        "intent",
        "item",
        "items",
        "map",
        "map_node",
        "modifiers",
        "move_history",
        "next_options",
        "nodes",
        "option",
        "options",
        "orbs",
        "player",
        "player_powers",
        "players",
        "play_pile",
        "points",
        "potion",
        "potions",
        "power",
        "powers",
        "relic",
        "relics",
        "rest_site",
        "rewards",
        "run",
        "selected_cards",
        "selected_character",
        "selectable_cards",
        "selection",
        "shop",
        "target",
        "treasure",
        "upgrade_preview",
        "keywords",
        "tags",
        "traits",
        "decision",
    }
)
_SOURCE_KEYS: Final[tuple[str, ...]] = (
    "card",
    "character",
    "selected_character",
    "potion",
    "relic",
    "item",
    "option",
    "map_node",
)
_CANDIDATE_LOCAL_ROOTS: Final[frozenset[str]] = frozenset(
    {*_SOURCE_KEYS, "target", "coord", "selection"}
)

# Feature ABI v1.  Numeric state keys must never share a slot: doing so makes
# observations such as ``block=3,max_hp=80`` alias a value-swapped state.  The
# fixed table is sorted for deterministic versioning and occupies a region
# disjoint from categorical and arbitrary ``dynamic_vars`` hashing.
_SUMMARY_SLOT_COUNT: Final = 6
_FIXED_NUMERIC_KEYS: Final[tuple[str, ...]] = tuple(
    sorted(
        _FACT_NUMERIC_KEYS
        | _FACT_BOOLEAN_KEYS
        | _CARD_FACT_NUMERIC_KEYS
        | _CARD_FACT_BOOLEAN_KEYS
    )
)
_NUMERIC_SLOT_BY_KEY: Final[dict[str, int]] = {
    key: _SUMMARY_SLOT_COUNT + index
    for index, key in enumerate(_FIXED_NUMERIC_KEYS)
}
_CATEGORY_SLOT_COUNT: Final = 32
_CATEGORY_SLOT_START: Final = _SUMMARY_SLOT_COUNT + len(_FIXED_NUMERIC_KEYS)
_DYNAMIC_SLOT_COUNT: Final = 16
_DYNAMIC_SLOT_START: Final = _CATEGORY_SLOT_START + _CATEGORY_SLOT_COUNT
_DYNAMIC_VALUE_KEYS: Final[tuple[str, ...]] = (
    "base_value",
    "enchanted_value",
    "current_value",
    "int_value",
    "was_just_upgraded",
)
_DYNAMIC_VALUE_SLOT_BY_KEY: Final[dict[str, int]] = {
    key: _DYNAMIC_SLOT_START + index
    for index, key in enumerate(_DYNAMIC_VALUE_KEYS)
}
_DYNAMIC_HASH_SLOT_START: Final = _DYNAMIC_SLOT_START + len(_DYNAMIC_VALUE_KEYS)
_DYNAMIC_HASH_SLOT_COUNT: Final = _DYNAMIC_SLOT_COUNT - len(_DYNAMIC_VALUE_KEYS)
_FEATURE_ABI_END: Final = _DYNAMIC_SLOT_START + _DYNAMIC_SLOT_COUNT
GROUNDING_ENCODING_VERSION: Final = "grounded-runtime-mechanics-encoding-v6"

if _FEATURE_ABI_END > MIN_TOKEN_FEATURE_DIM:  # pragma: no cover - import invariant
    raise RuntimeError(
        "grounded feature ABI exceeds MIN_TOKEN_FEATURE_DIM: "
        f"{_FEATURE_ABI_END} > {MIN_TOKEN_FEATURE_DIM}"
    )


def grounding_encoding_identity() -> dict[str, Any]:
    """Return the compact semantic identity persisted in every checkpoint.

    Config dimensions alone cannot detect a changed allowlist, exclusion
    firewall, domain vocabulary, or feature-slot layout.  Hash those reviewed
    contracts so exact resume and cross-curriculum model initialization fail
    closed after an encoder change, even when tensor shapes still match.
    Algorithm changes that do not alter this payload must bump
    :data:`GROUNDING_ENCODING_VERSION`.
    """

    contract = {
        "version": GROUNDING_ENCODING_VERSION,
        "min_token_feature_dim": MIN_TOKEN_FEATURE_DIM,
        "feature_abi_end": _FEATURE_ABI_END,
        "domain_ids": _DOMAIN_IDS,
        "model_action_kinds": sorted(MODEL_ACTION_KIND_VOCABULARY),
        "entity_hash_namespaces": ["entity", "entity_aux"],
        "summary_slot_count": _SUMMARY_SLOT_COUNT,
        "numeric_slots": _NUMERIC_SLOT_BY_KEY,
        "category_slot_start": _CATEGORY_SLOT_START,
        "category_slot_count": _CATEGORY_SLOT_COUNT,
        "dynamic_slot_start": _DYNAMIC_SLOT_START,
        "dynamic_slot_count": _DYNAMIC_SLOT_COUNT,
        "dynamic_value_slots": _DYNAMIC_VALUE_SLOT_BY_KEY,
        "dynamic_hash_slot_start": _DYNAMIC_HASH_SLOT_START,
        "dynamic_hash_slot_count": _DYNAMIC_HASH_SLOT_COUNT,
        "world_excluded_keys": sorted(_WORLD_EXCLUDED_KEYS),
        "candidate_excluded_keys": sorted(_CANDIDATE_EXCLUDED_KEYS),
        "identity_keys": list(_IDENTITY_KEYS),
        "role_keys": list(_ROLE_KEYS),
        "owner_keys": list(_OWNER_KEYS),
        "categorical_keys": sorted(_FACT_CATEGORICAL_KEYS),
        "numeric_keys": sorted(_FACT_NUMERIC_KEYS),
        "boolean_keys": sorted(_FACT_BOOLEAN_KEYS),
        "card_numeric_keys": sorted(_CARD_FACT_NUMERIC_KEYS),
        "card_boolean_keys": sorted(_CARD_FACT_BOOLEAN_KEYS),
        "container_keys": sorted(_FACT_CONTAINER_KEYS),
        "source_keys": list(_SOURCE_KEYS),
        "candidate_local_roots": sorted(_CANDIDATE_LOCAL_ROOTS),
        "snapshot_version": ENCODED_DECISION_SNAPSHOT_VERSION,
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return {
        "version": GROUNDING_ENCODING_VERSION,
        "min_token_feature_dim": MIN_TOKEN_FEATURE_DIM,
        "feature_abi_end": _FEATURE_ABI_END,
        "fingerprint_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }


def _normalize_key(value: Any) -> str:
    text = str(value).strip().replace("-", "_")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()


@dataclass(frozen=True, slots=True)
class ActionReference:
    """Dispatch-only action identity, kept outside model tensors."""

    position: int
    handle: str | None
    enabled: bool


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    """One model batch plus the non-learned action dispatch table."""

    batch: GroundedCandidateBatch
    actions: tuple[ActionReference, ...]
    snapshot: EncodedDecisionSnapshot
    encoding_fingerprint: str

    def action(self, position: int) -> ActionReference:
        if position < 0 or position >= len(self.actions):
            raise IndexError(f"candidate position {position} is out of range")
        action = self.actions[position]
        if not action.enabled:
            raise ValueError(f"candidate position {position} is disabled")
        return action


@dataclass(frozen=True, slots=True)
class _Token:
    features: tuple[float, ...]
    type_id: int
    role_id: int
    owner_id: int
    entity_id: int
    entity_aux_id: int
    zone_id: int
    order_id: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    token: _Token
    target_owner_id: int
    target_entity_id: int
    target_entity_aux_id: int
    locals: tuple[_Token, ...]
    enabled: bool


@dataclass(frozen=True, slots=True)
class _WalkItem:
    value: Any
    path: tuple[str, ...]
    inherited_owner: str
    order: int
    depth: int


def _hash_id(namespace: str, value: Any, size: int) -> int:
    """Return a process-independent ID while reserving 0/1 for pad/unknown."""

    text = str(value).strip().lower()
    if not text:
        return 1
    digest = hashlib.sha256(f"{namespace}\0{text}".encode()).digest()
    return 2 + int.from_bytes(digest[:8], "big") % (size - 2)


def _bounded_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("numeric model input must be finite")
    return math.copysign(min(1.0, math.log1p(abs(number)) / 10.0), number)


def _string_field(value: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return str(candidate)
    return ""


def _owner_label(value: Mapping[str, Any], path: tuple[str, ...], inherited: str) -> str:
    direct = _string_field(value, _OWNER_KEYS)
    if direct:
        return direct
    lowered = {part.lower() for part in path}
    if lowered & {"player", "players", "self", "hero"}:
        return "player"
    if lowered & {"enemy", "enemies", "monster", "monsters"}:
        return "enemy"
    return inherited


def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _as_mapping_list(value: Any, *, label: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of mappings")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{label}[{index}] must be a mapping")
        result.append(item)
    return result


def _first_present(value: Mapping[str, Any], *keys: str) -> Any:
    normalized = {_normalize_key(key): item for key, item in value.items()}
    for key in keys:
        if key in normalized and normalized[key] is not None:
            return normalized[key]
    return None


def _canonical_card_labels(
    value: Any,
    *,
    label: str,
    fact_type: str,
) -> list[dict[str, Any]]:
    """Represent exact runtime enums as typed entity tokens.

    These are not inferred semantic tags: every label must be emitted by the
    game model (keyword, card tag, or factual lifecycle flag).
    """

    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        values = value
    else:
        raise TypeError(f"{label} must be a string or sequence")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, Mapping):
            identity = _first_present(item, "id", "name", "value", "type")
        else:
            identity = item
        text = str(identity or "").strip()
        if not text or text.lower() == "none" or text.lower() in seen:
            continue
        seen.add(text.lower())
        result.append({"id": text, "type": fact_type})
    return result


def _canonical_dynamic_var(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project one game-owned DynamicVar to a typed, collision-free value token."""

    result: dict[str, Any] = {}
    name = _first_present(value, "name", "id")
    if name is not None and str(name).strip():
        result["id"] = str(name).strip()
    var_type = _first_present(value, "var_type", "dynamic_var_type", "class_name")
    family = _first_present(value, "family", "effect_family")
    if family is None:
        family = var_type or "value"
    result["type"] = str(family)
    if var_type is not None and str(var_type).strip():
        result["var_type"] = str(var_type).strip()
    for key, semantic_aliases in {
        "power_type": ("power_type", "power_id"),
        "value_props": ("value_props", "props"),
    }.items():
        item = _first_present(value, *semantic_aliases)
        if item is not None and str(item).strip():
            result[key] = str(item).strip()
    for key, numeric_aliases in {
        "base_value": ("base_value",),
        "enchanted_value": ("enchanted_value",),
        # ``preview_value`` is a factual value calculated by CardModel hooks,
        # not the retired action-effect preview.  Rename it so the engineered
        # preview firewall remains closed outside this exact DynamicVar path.
        "current_value": ("current_value", "preview_value"),
        "int_value": ("int_value",),
        "was_just_upgraded": ("was_just_upgraded",),
    }.items():
        item = _first_present(value, *numeric_aliases)
        if item is not None:
            result[key] = item
    return result


def _canonical_dynamic_vars(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if _first_present(value, "name", "id", "base_value") is not None:
            raw_values: Sequence[Any] = [value]
        else:
            raw_values = [
                {"name": key, **dict(item)} if isinstance(item, Mapping) else {
                    "name": key,
                    "base_value": item,
                    "current_value": item,
                }
                for key, item in value.items()
            ]
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        raw_values = value
    else:
        raise TypeError("card.dynamic_vars must be a mapping or sequence")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw_values):
        if not isinstance(item, Mapping):
            raise TypeError(f"card.dynamic_vars[{index}] must be a mapping")
        normalized = _canonical_dynamic_var(item)
        if normalized.get("id"):
            result.append(normalized)
    return result


def _canonical_card_modifier(
    value: Mapping[str, Any],
    *,
    modifier_type: str,
) -> dict[str, Any]:
    """Project an exact native enchantment/affliction without text inference."""

    result: dict[str, Any] = {"type": modifier_type}
    for key, aliases in {
        "id": ("id", "model_id", f"{modifier_type}_id"),
        "class_name": ("class_name", "runtime_type"),
        "status": ("status",),
        "amount": ("amount",),
        "display_amount": ("display_amount",),
    }.items():
        item = _first_present(value, *aliases)
        if item is not None:
            result[key] = item
    for key in (
        "show_amount",
        "is_stackable",
        "should_start_at_bottom_of_draw_pile",
        "should_glow_gold",
        "should_glow_red",
        "has_extra_card_text",
        "can_afflict_unplayable_cards",
        "has_overlay",
    ):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
    dynamic_vars = _canonical_dynamic_vars(_first_present(value, "dynamic_vars"))
    if dynamic_vars:
        result["dynamic_vars"] = dynamic_vars
    return result


def _canonical_card_modifiers(
    value: Any,
    *,
    modifier_type: str,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        raw: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        raw = value
    else:
        raise TypeError(f"card.{modifier_type} must be a mapping or sequence")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise TypeError(f"card.{modifier_type}[{index}] must be a mapping")
        normalized = _canonical_card_modifier(item, modifier_type=modifier_type)
        if normalized.get("id"):
            result.append(normalized)
    result.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return result


def _canonical_card(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project a card to exact runtime facts shared by maintained backends."""

    result: dict[str, Any] = {}
    aliases = {
        "id": ("id", "card_id", "model_id"),
        "type": ("type", "card_type"),
        "target_type": ("target_type",),
        # Prefer the live resolved cost. ``base_cost``/``is_x_cost`` are used
        # by the pinned native catalog and remain useful for fail-closed
        # catalog/fixture audits where no combat owner exists yet.
        "cost": ("resolved_energy_cost", "cost", "energy_cost", "base_cost"),
        "star_cost": ("star_cost", "current_star_cost"),
        "upgrade_level": ("upgrade_level", "current_upgrade_level"),
        "max_upgrade_level": ("max_upgrade_level",),
        "base_replay_count": ("base_replay_count",),
        "current_replay_count": ("current_replay_count", "replay_count"),
        "last_stars_spent": ("last_stars_spent",),
        "floor_added_to_deck": ("floor_added_to_deck",),
        "is_playable": ("is_playable", "playable"),
        "is_upgraded": ("is_upgraded", "upgraded"),
        "exhaust": ("exhaust",),
        "ethereal": ("ethereal",),
        "retain": ("retain", "retained"),
        "rarity": ("rarity",),
        # Selection membership and the physical pile are orthogonal.  A card
        # remains in Discard/Hand/Deck while the prompt marks it selected.
        "pile": ("source_pile", "pile", "zone"),
        "selection_membership": ("selection_membership",),
        "is_selected": ("is_selected",),
    }
    for canonical, source_keys in aliases.items():
        item = _first_present(value, *source_keys)
        if item is not None:
            result[canonical] = item
    for field, fact_type in (
        ("keywords", "card_keyword"),
        ("tags", "card_tag"),
        ("hover_tips", "card_hover_tip"),
    ):
        source = (
            _first_present(value, "hover_tip_ids", "hover_tips")
            if field == "hover_tips"
            else _first_present(value, field)
        )
        labels = _canonical_card_labels(
            source,
            label=f"card.{field}",
            fact_type=fact_type,
        )
        if labels:
            result[field] = labels
    traits = _canonical_card_labels(
        _first_present(value, "traits"),
        label="card.traits",
        fact_type="card_trait",
    )
    for field, source_keys in (
        ("costs_x", ("costs_x", "is_x_cost")),
        ("has_star_cost_x", ("has_star_cost_x",)),
        ("gains_block", ("gains_block",)),
        ("has_turn_end_in_hand_effect", ("has_turn_end_in_hand_effect",)),
        ("has_on_draw_effect", ("has_on_draw_effect",)),
        ("exhaust_on_next_play", ("exhaust_on_next_play",)),
    ):
        if _first_present(value, *source_keys) is True:
            traits.extend(
                _canonical_card_labels(
                    [field],
                    label=f"card.{field}",
                    fact_type="card_trait",
                )
            )
    if traits:
        deduplicated = {str(item["id"]).lower(): item for item in traits}
        result["traits"] = list(deduplicated.values())
    dynamic_vars = _canonical_dynamic_vars(_first_present(value, "dynamic_vars"))
    if dynamic_vars:
        result["dynamic_vars"] = dynamic_vars
    for field, singular, modifier_type in (
        ("enchantments", "enchantment", "enchantment"),
        ("afflictions", "affliction", "affliction"),
    ):
        modifiers = _canonical_card_modifiers(
            _first_present(value, field, singular),
            modifier_type=modifier_type,
        )
        if modifiers:
            result[field] = modifiers
    for field in (
        "is_removable",
        "is_transformable",
        "is_in_combat",
        "is_upgradable",
        "is_sly_this_turn",
        "is_retained",
        "is_clone",
        "is_dupe",
        "has_been_removed_from_state",
    ):
        item = _first_present(value, field)
        if item is not None:
            result[field] = item
    return result


def _canonical_power(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    aliases = {
        "id": ("id", "power_id", "model_id"),
        "amount": ("amount", "stacks", "value"),
        "amount_on_turn_start": ("amount_on_turn_start",),
        "display_amount": ("display_amount",),
        "type": ("type", "power_type"),
        "stack_type": ("stack_type",),
        "type_for_current_amount": ("type_for_current_amount",),
        "class_name": ("class_name", "runtime_type"),
    }
    for canonical, source_keys in aliases.items():
        item = _first_present(value, *source_keys)
        if item is not None:
            result[canonical] = item
    for field in (
        "is_instanced",
        "is_visible",
        "allow_negative",
        "skip_next_duration_tick",
        "should_scale_in_multiplayer",
        "owner_is_secondary_enemy",
    ):
        item = _first_present(value, field)
        if item is not None:
            result[field] = item
    for field in (
        "owner_side",
        "owner_model_id",
        "applier_side",
        "applier_model_id",
        "target_side",
        "target_model_id",
    ):
        item = _first_present(value, field)
        if item is not None:
            result[field] = item
    dynamic_vars = _canonical_dynamic_vars(_first_present(value, "dynamic_vars"))
    if dynamic_vars:
        result["dynamic_vars"] = dynamic_vars
    return result


def _canonical_inventory_entity(
    value: Mapping[str, Any],
    *,
    id_keys: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    identity = _first_present(value, *id_keys)
    if identity is not None and str(identity).strip():
        result["id"] = identity
    for key in ("rarity", "charges", "target_type", "type", "status"):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
    return result


def _canonical_relic(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _canonical_inventory_entity(
        value,
        id_keys=("id", "relic_id", "model_id"),
    )
    for key, aliases in {
        "display_amount": ("display_amount", "counter"),
        "stack_count": ("stack_count",),
        "merchant_cost": ("merchant_cost",),
        "floor_added_to_deck": ("floor_added_to_deck",),
    }.items():
        item = _first_present(value, *aliases)
        if item is not None:
            result[key] = item
    for key in (
        "is_tradable",
        "is_allowed_in_shops",
        "is_used_up",
        "has_upon_pickup_effect",
        "spawns_pets",
        "is_stackable",
        "is_wax",
        "is_melted",
        "adds_pet",
        "show_counter",
        "has_been_removed_from_state",
    ):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
    dynamic_vars = _canonical_dynamic_vars(_first_present(value, "dynamic_vars"))
    if dynamic_vars:
        result["dynamic_vars"] = dynamic_vars
    return result


def _canonical_potion(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _canonical_inventory_entity(
        value,
        id_keys=("id", "potion_id", "model_id"),
    )
    for key in ("usage",):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
    for key in (
        "is_queued",
        "can_be_generated_in_combat",
        "passes_custom_usability_check",
        "has_been_removed_from_state",
        "can_use_in_combat",
        "can_throw_at_ally",
    ):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
    dynamic_vars = _canonical_dynamic_vars(_first_present(value, "dynamic_vars"))
    if dynamic_vars:
        result["dynamic_vars"] = dynamic_vars
    return result


def _canonical_enemy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop episode-local combat IDs and retain stable observable facts."""

    result: dict[str, Any] = {}
    stable_identity = _first_present(value, "model_id", "entity_id")
    raw_id = _first_present(value, "id")
    if stable_identity is None and isinstance(raw_id, str) and raw_id.strip():
        stable_identity = raw_id
    if stable_identity is not None:
        result["model_id"] = stable_identity
    for key in (
        "side",
        "hp",
        "max_hp",
        "block",
        "is_alive",
        "is_hittable",
        "is_primary_enemy",
        "is_secondary_enemy",
        "is_stunned",
        "is_pet",
        "shows_infinite_hp",
        "can_receive_powers",
        "spawned_this_turn",
        "is_performing_move",
        "intends_to_attack",
        "slot_name",
        "is_move",
        "must_perform_once_before_transitioning",
        "can_transition_away",
    ):
        item = _first_present(value, key, f"current_{key}")
        if item is not None:
            result[key] = item
    next_move = _first_present(value, "next_move_state_id", "next_move_id")
    if next_move is not None:
        result["next_move_state_id"] = next_move
    powers = _as_mapping_list(
        _first_present(value, "powers", "status"),
        label="enemy powers",
    )
    result["powers"] = [_canonical_power(item) for item in powers]
    result["powers"].sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    intents = _as_mapping_list(
        _first_present(value, "intents"),
        label="enemy intents",
    )
    normalized_intents: list[dict[str, Any]] = []
    for intent in intents:
        normalized: dict[str, Any] = {}
        for key, item in {
            "type": _first_present(intent, "type", "intent_type"),
            "damage": _first_present(intent, "damage", "total_damage"),
            "repeats": _first_present(intent, "repeats", "hits"),
        }.items():
            if item is not None:
                normalized[key] = item
        normalized_intents.append(normalized)
    result["intents"] = normalized_intents
    raw_history = _first_present(value, "move_history", "state_log")
    if raw_history is not None:
        result["move_history"] = _canonical_card_labels(
            raw_history,
            label="enemy.move_history",
            fact_type="move_state",
        )
    return result


def _pile_count(value: Any, *, label: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a number or card sequence")
    if isinstance(value, int | float):
        return value
    if isinstance(value, Mapping):
        explicit = _first_present(value, "count")
        if explicit is not None:
            if isinstance(explicit, bool) or not isinstance(explicit, int | float):
                raise TypeError(f"{label}.count must be numeric")
            return float(explicit) if isinstance(explicit, float) else int(explicit)
        cards = _first_present(value, "cards")
        if cards is None:
            return None
        return len(_as_mapping_list(cards, label=f"{label}.cards"))
    return len(_as_mapping_list(value, label=label))


def _pile_cards(value: Any, *, label: str) -> list[Mapping[str, Any]] | None:
    """Return visible pile cards, or ``None`` when composition is redacted."""

    if value is None or isinstance(value, bool | int | float):
        return None
    if isinstance(value, Mapping):
        cards = _first_present(value, "cards")
        if cards is None:
            return None
        return _as_mapping_list(cards, label=f"{label}.cards")
    return _as_mapping_list(value, label=label)


def _canonical_card_sequence(
    value: Any,
    *,
    label: str,
    pile: str | None = None,
    membership: str | None = None,
    sort_as_set: bool = False,
) -> list[dict[str, Any]] | None:
    cards = _pile_cards(value, label=label)
    if cards is None:
        return None
    result: list[dict[str, Any]] = []
    for card in cards:
        enriched = dict(card)
        if pile is not None and _first_present(enriched, "source_pile", "pile", "zone") is None:
            enriched["pile"] = pile
        if membership is not None:
            enriched["selection_membership"] = membership
            enriched["is_selected"] = membership == "selected"
        result.append(_canonical_card(enriched))
    if sort_as_set:
        result.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return result


def _canonical_event_option(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, aliases in {
        "text_key": ("text_key", "option_id", "id"),
        "title": ("title",),
        "description": ("description",),
        "text": ("text", "label"),
        "is_locked": ("is_locked",),
        "is_chosen": ("is_chosen", "was_chosen"),
        "is_proceed": ("is_proceed", "proceed"),
    }.items():
        item = _first_present(value, *aliases)
        if item is not None:
            result[key] = item
    raw_relic = _first_present(value, "relic")
    if raw_relic is not None:
        relic = _canonical_relic(_as_mapping(raw_relic, label="event.option.relic"))
        if relic.get("id"):
            result["relic"] = relic
    return result


def _canonical_event(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, aliases in {
        "event_id": ("event_id", "id", "model_id"),
        "layout_type": ("layout_type",),
        "description_key": ("description_key", "page_key"),
        "encounter_id": ("encounter_id", "canonical_encounter_id"),
        "is_deterministic": ("is_deterministic",),
        "is_shared": ("is_shared",),
        "is_finished": ("is_finished",),
        "in_dialogue": ("in_dialogue",),
    }.items():
        item = _first_present(value, *aliases)
        if item is not None:
            result[key] = item
    dynamic_vars = _canonical_dynamic_vars(_first_present(value, "dynamic_vars"))
    if dynamic_vars:
        result["dynamic_vars"] = dynamic_vars
    raw_options = _as_mapping_list(
        _first_present(value, "options", "current_options"),
        label="event options",
    )
    if raw_options:
        result["options"] = [_canonical_event_option(option) for option in raw_options]
    return result


def _canonical_model_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Project live/headless state to one deliberately shared model DTO.

    Inactive UI sections are not part of the model contract.  Public discard
    and exhaust composition, physical source piles, and exact multi-selection
    membership are retained because they change legal decisions.  Hidden draw
    order/composition remains represented by count only.
    """

    phase = str(observation.get("phase") or "unknown")
    domain = str(observation.get("decision_domain") or "unknown")
    raw_player = _as_mapping(
        observation.get("player"),
        label="observation.player",
    )
    raw_run = _as_mapping(observation.get("run"), label="observation.run")
    raw_combat = _as_mapping(
        observation.get("combat"),
        label="observation.combat",
    )

    deck_value = _first_present(raw_player, "deck_cards", "deck")
    if isinstance(deck_value, int | float) and not isinstance(deck_value, bool):
        deck_cards = _as_mapping_list(
            _first_present(raw_player, "deck_cards"),
            label="player.deck_cards",
        )
        deck_count: int | float = deck_value
    else:
        deck_cards = _as_mapping_list(deck_value, label="player deck")
        deck_count = len(deck_cards)

    canonical_deck_cards = _canonical_card_sequence(
        deck_cards,
        label="player.deck_cards",
        pile="Deck",
    ) or []
    player: dict[str, Any] = {
        "deck": deck_count,
        "deck_cards": canonical_deck_cards,
        "powers": [
            _canonical_power(power)
            for power in _as_mapping_list(
                _first_present(raw_player, "powers", "status"),
                label="player powers",
            )
        ],
        "relics": [],
        "potions": [],
    }
    player["powers"].sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    for relic in _as_mapping_list(
        _first_present(raw_player, "relics"),
        label="player relics",
    ):
        normalized = _canonical_relic(relic)
        if normalized.get("id"):
            player["relics"].append(normalized)
    for potion in _as_mapping_list(
        _first_present(raw_player, "potions"),
        label="player potions",
    ):
        normalized = _canonical_potion(potion)
        if normalized.get("id"):
            player["potions"].append(normalized)
    for field in ("relics", "potions"):
        player[field].sort(
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
    for key, aliases in {
        "character_id": ("character_id", "character"),
        "hp": ("hp", "current_hp"),
        "max_hp": ("max_hp",),
        "block": ("block",),
        "gold": ("gold",),
        "open_potion_slots": ("open_potion_slots",),
        "orb_slots": ("orb_slots",),
        "orb_empty_slots": ("orb_empty_slots",),
    }.items():
        item = _first_present(raw_player, *aliases)
        if item is not None and not isinstance(item, Mapping):
            player[key] = item
    raw_orbs = _as_mapping_list(_first_present(raw_player, "orbs"), label="player orbs")
    if raw_orbs:
        orbs: list[dict[str, Any]] = []
        for orb in raw_orbs:
            normalized_orb: dict[str, Any] = {}
            for key, aliases in {
                "id": ("id", "orb_id", "model_id"),
                "passive_val": ("passive_val",),
                "evoke_val": ("evoke_val",),
            }.items():
                item = _first_present(orb, *aliases)
                if item is not None:
                    normalized_orb[key] = item
            if normalized_orb.get("id"):
                orbs.append(normalized_orb)
        player["orbs"] = orbs

    run: dict[str, Any] = {}
    for key, aliases in {
        "active": ("active", "run_active"),
        "game_over": ("game_over",),
        "floor": ("floor", "total_floor"),
        "act": ("act", "act_index"),
        "ascension": ("ascension", "ascension_level"),
        "progress": ("progress",),
        "room_type": ("room_type",),
        "room_model_id": ("room_model_id", "room_model"),
    }.items():
        item = _first_present(raw_run, *aliases)
        if item is not None:
            run[key] = item
    raw_coord = _first_present(raw_run, "coord")
    if raw_coord is not None:
        coord = _as_mapping(raw_coord, label="run.coord")
        canonical_coord: dict[str, Any] = {}
        for key, item in {
            "x": _first_present(coord, "x", "col", "column"),
            "y": _first_present(coord, "y", "row"),
        }.items():
            if item is not None:
                canonical_coord[key] = item
        if canonical_coord:
            run["coord"] = canonical_coord

    canonical: dict[str, Any] = {
        "phase": phase,
        "decision_domain": domain,
        "run": run,
        "player": player,
    }
    if bool(observation.get("terminated", False)):
        canonical["terminated"] = True

    raw_enemies = _first_present(raw_combat, "enemies")
    combat_active = (
        phase == "combat"
        or _first_present(raw_combat, "in_progress") is True
        or bool(raw_enemies)
    )
    if combat_active:
        raw_hand = _first_present(raw_combat, "hand")
        if raw_hand is None:
            raw_hand = _first_present(raw_player, "hand")
        combat: dict[str, Any] = {
            "in_progress": True,
            "hand": _canonical_card_sequence(
                raw_hand,
                label="combat hand",
                pile="Hand",
            ) or [],
            "enemies": [
                _canonical_enemy(enemy)
                for enemy in _as_mapping_list(
                    raw_enemies,
                    label="combat enemies",
                )
            ],
        }
        for key, aliases in {
            "round": ("round", "turn"),
            "energy": ("energy", "current_energy"),
            "max_energy": ("max_energy",),
            "stars": ("stars", "current_stars"),
            "side": ("side",),
        }.items():
            item = _first_present(raw_combat, *aliases)
            if item is None:
                item = _first_present(raw_player, *aliases)
            if item is not None:
                combat[key] = item
        for key, aliases in {
            "draw": ("draw", "draw_pile"),
            "discard": ("discard", "discard_pile"),
            "exhaust": ("exhaust", "exhaust_pile"),
        }.items():
            item = _first_present(raw_combat, *aliases)
            if item is None:
                item = _first_present(raw_player, *aliases)
            count = _pile_count(item, label=f"combat {key}")
            if count is not None:
                combat[key] = count
        for key, aliases, pile in (
            ("discard_pile", ("discard_pile",), "Discard"),
            ("exhaust_pile", ("exhaust_pile",), "Exhaust"),
            ("play_pile", ("play_pile",), "Play"),
        ):
            item = _first_present(raw_combat, *aliases)
            if item is None:
                item = _first_present(raw_player, *aliases)
            cards = _canonical_card_sequence(
                item,
                label=f"combat {key}",
                pile=pile,
                sort_as_set=True,
            )
            if cards is not None:
                combat[key] = cards
        canonical["combat"] = combat

    raw_event = _as_mapping(observation.get("event"), label="observation.event")
    if raw_event:
        event = _canonical_event(raw_event)
        if event:
            canonical["event"] = event

    raw_decision = _as_mapping(
        observation.get("decision"),
        label="observation.decision",
    )
    explicit_selection = _as_mapping(
        observation.get("card_selection"),
        label="observation.card_selection",
    )
    # Live places the card option membership on top-level card_selection while
    # its compact decision block carries counts.  Headless places both in the
    # explicit selection DTO.  Merge the two instead of allowing the summary
    # block to hide card identity.
    raw_selection: dict[str, Any] = dict(explicit_selection)
    raw_selection.update(raw_decision)
    # Card/hand prompts can occur inside combat while the top-level phase stays
    # ``combat``.  Presence of the explicit selection DTO, rather than the
    # screen phase, determines whether selection state is model-visible.
    if raw_selection:
        selected_cards = _first_present(raw_selection, "selected_cards")
        selected_count = _first_present(raw_selection, "selected_count")
        if selected_count is None and selected_cards is not None:
            selected_count = len(
                _as_mapping_list(
                    selected_cards,
                    label="selection.selected_cards",
                )
            )
        requires_manual_confirmation = _first_present(
            raw_selection,
            "requires_manual_confirmation",
        )
        confirmation_mode = (
            "manual"
            if requires_manual_confirmation is True
            else "automatic"
            if requires_manual_confirmation is False
            else None
        )
        selection: dict[str, Any] = {}
        for key, item in {
            "selected_count": selected_count,
            "min_select": _first_present(raw_selection, "min_select"),
            "max_select": _first_present(raw_selection, "max_select"),
            "remaining_select": _first_present(
                raw_selection,
                "remaining_select",
                "remaining_picks",
            ),
            "confirm_ready": _first_present(
                raw_selection,
                "confirm_ready",
                "can_confirm",
            ),
            "can_skip": _first_present(raw_selection, "can_skip"),
            "cancelable": _first_present(
                raw_selection,
                "cancelable",
                "can_cancel",
            ),
            "confirmation_mode": confirmation_mode,
            "mode": _first_present(raw_selection, "mode", "screen_type"),
            "prompt_id": _first_present(raw_selection, "prompt_id"),
            "operation_type": _first_present(raw_selection, "operation_type"),
            "source_zone": _first_present(
                raw_selection,
                "source_zone",
                "source_pile",
            ),
            "destination_zone": _first_present(
                raw_selection,
                "destination_zone",
                "destination_pile",
            ),
        }.items():
            if item is not None:
                selection[key] = item
        selectable = _canonical_card_sequence(
            _first_present(raw_selection, "selectable_cards", "cards"),
            label="selection.selectable_cards",
            membership="selectable",
            sort_as_set=True,
        )
        selected = _canonical_card_sequence(
            selected_cards,
            label="selection.selected_cards",
            membership="selected",
            sort_as_set=True,
        )
        raw_options = _first_present(raw_selection, "options")
        if raw_options is not None:
            option_selectable: list[dict[str, Any]] = []
            option_selected: list[dict[str, Any]] = []
            for option in _as_mapping_list(raw_options, label="selection.options"):
                raw_card = _first_present(option, "card")
                if raw_card is None:
                    continue
                card = _as_mapping(raw_card, label="selection.options.card")
                membership = "selected" if option.get("is_selected") is True else "selectable"
                canonical_cards = _canonical_card_sequence(
                    [card],
                    label="selection.options.card",
                    membership=membership,
                ) or []
                (option_selected if membership == "selected" else option_selectable).extend(
                    canonical_cards
                )
            if selectable is None:
                selectable = option_selectable
            if selected is None:
                selected = option_selected
        if selectable is not None:
            selection["selectable_cards"] = selectable
        if selected is not None:
            selection["selected_cards"] = selected
        if selection:
            canonical["decision"] = {"selection": selection}
    return canonical


def _canonical_coord(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in {
        "x": _first_present(value, "x", "col", "column"),
        "y": _first_present(value, "y", "row"),
    }.items():
        if item is not None:
            result[key] = item
    return result


def _canonical_option(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, aliases in {
        "option_id": ("option_id", "text_key", "id", "model_id"),
        "option_type": ("option_type", "type"),
        "title": ("title",),
        "description": ("description",),
        "text": ("text", "label"),
        "is_locked": ("is_locked",),
        "is_proceed": ("is_proceed", "proceed"),
        "is_selected": ("is_selected",),
    }.items():
        item = _first_present(value, *aliases)
        if item is not None:
            result[key] = item
    raw_coord = _first_present(value, "coord")
    if raw_coord is not None:
        coord = _canonical_coord(_as_mapping(raw_coord, label="option.coord"))
        if coord:
            result["coord"] = coord
    return result


def _canonical_map_node(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    raw_coord = _first_present(value, "coord")
    coord_source = (
        _as_mapping(raw_coord, label="map_node.coord")
        if raw_coord is not None
        else value
    )
    coord = _canonical_coord(coord_source)
    if coord:
        result["coord"] = coord
    for key, aliases in {
        "point_type": ("point_type", "room_type", "type"),
        "room_model_id": ("room_model_id", "room_model", "model_id"),
    }.items():
        item = _first_present(value, *aliases)
        if item is not None:
            result[key] = item
    return result


def _canonical_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    requires_manual_confirmation = _first_present(
        value,
        "requires_manual_confirmation",
    )
    if requires_manual_confirmation is True:
        result["confirmation_mode"] = "manual"
    elif requires_manual_confirmation is False:
        result["confirmation_mode"] = "automatic"
    for key, aliases in {
        "selected_count": ("selected_count",),
        "min_select": ("min_select", "min_count"),
        "max_select": ("max_select", "max_count"),
        "remaining_select": ("remaining_select", "remaining_picks"),
        "confirm_ready": ("confirm_ready", "can_confirm"),
        "can_skip": ("can_skip",),
        "cancelable": ("cancelable", "can_cancel"),
        "is_selected": ("is_selected",),
        "operation_type": (
            "operation_type",
            "selection_operation",
            "model_action_variant",
        ),
        "mode": ("mode", "screen_type"),
        "prompt_id": ("prompt_id",),
        "source_zone": ("source_zone", "source_pile"),
        "destination_zone": ("destination_zone", "destination_pile"),
    }.items():
        item = _first_present(value, *aliases)
        if item is not None:
            result[key] = item
    return result


def _canonical_item(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, aliases in {
        "id": ("id", "model_id"),
        "type": ("type", "item_type", "kind"),
        "price": ("price", "cost"),
        "rarity": ("rarity",),
    }.items():
        item = _first_present(value, *aliases)
        if item is not None:
            result[key] = item
    for key, projector in (
        ("card", _canonical_card),
        (
            "potion",
            _canonical_potion,
        ),
        (
            "relic",
            _canonical_relic,
        ),
    ):
        raw = _first_present(value, key)
        if raw is not None:
            result[key] = projector(_as_mapping(raw, label=f"item.{key}"))
    return result


def _target_fact_lookup(
    observation: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    combat = _as_mapping(observation.get("combat"), label="observation.combat")
    enemies = _as_mapping_list(
        _first_present(combat, "enemies"),
        label="combat enemies",
    )
    result: dict[str, dict[str, Any]] = {}
    for enemy in enemies:
        canonical = _canonical_enemy(enemy)
        for key in ("combat_id", "id"):
            identity = _first_present(enemy, key)
            if identity is not None:
                result[str(identity)] = canonical
    return result


def _candidate_local_roots(
    action: Mapping[str, Any],
    *,
    model_kind: str,
    target_lookup: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    roots: dict[str, dict[str, Any]] = {}
    projectors = {
        "card": _canonical_card,
        "character": lambda value: {
            "id": _first_present(value, "id", "character_id", "model_id")
        },
        "selected_character": lambda value: {
            "id": _first_present(value, "id", "character_id", "model_id")
        },
        "potion": _canonical_potion,
        "relic": _canonical_relic,
        "item": _canonical_item,
        "option": _canonical_option,
        "map_node": _canonical_map_node,
        "coord": _canonical_coord,
        "selection": _canonical_selection,
        "typed_selection": _canonical_selection,
    }
    for key, projector in projectors.items():
        raw = action.get(key)
        if isinstance(raw, Mapping):
            projected = projector(raw)
            if projected:
                roots[key] = projected

    # The live map DTO exposes coord/point_type at the root, whereas the
    # simulator joins them under map_node. Normalize both to one local token.
    if model_kind == "map" and "map_node" not in roots:
        raw_coord = action.get("coord")
        if isinstance(raw_coord, Mapping):
            map_node = _canonical_map_node(
                {
                    "coord": raw_coord,
                    "point_type": action.get("point_type"),
                    "room_model_id": action.get("room_model_id"),
                }
            )
            if map_node:
                roots["map_node"] = map_node
        roots.pop("coord", None)

    raw_target = action.get("target")
    if isinstance(raw_target, Mapping):
        target = _canonical_enemy(raw_target)
        combat_id = _first_present(raw_target, "combat_id", "id")
        if combat_id is not None and str(combat_id) in target_lookup:
            target = dict(target_lookup[str(combat_id)])
        if target:
            roots["target"] = target

    selection = _canonical_selection(action)
    typed_selection = action.get("typed_selection")
    if isinstance(typed_selection, Mapping):
        selection = {
            **_canonical_selection(typed_selection),
            **selection,
        }
    if selection:
        roots["selection"] = {
            **roots.get("selection", {}),
            **selection,
        }
    return roots


class GroundedObservationEncoder:
    """Convert bridge observations to the grounded candidate tensor contract."""

    def __init__(self, config: GroundedEncodingConfig | None = None) -> None:
        self.config = config or GroundedEncodingConfig()

    def encode(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
        *,
        device: torch.device | str | None = None,
    ) -> EncodedDecision:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        if isinstance(legal_actions, str | bytes) or not isinstance(legal_actions, Sequence):
            raise TypeError("legal_actions must be a sequence of mappings")
        if len(legal_actions) > self.config.max_candidates:
            raise ValueError(
                "legal action count exceeds the grounded model capacity; refusing to "
                "silently hide dispatchable candidates: "
                f"count={len(legal_actions)} capacity={self.config.max_candidates}"
            )
        model_observation = _canonical_model_observation(observation)
        world_tokens = self._world_tokens(model_observation)
        target_lookup = _target_fact_lookup(observation)
        candidates: list[_Candidate] = []
        references: list[ActionReference] = []
        for position, action in enumerate(legal_actions):
            if not isinstance(action, Mapping):
                raise TypeError(f"legal action {position} must be a mapping")
            candidate = self._candidate_token(
                action,
                target_lookup=target_lookup,
            )
            candidates.append(candidate)
            handle_value = action.get("action_handle", action.get("action_id"))
            handle = str(handle_value) if handle_value is not None and str(handle_value) else None
            references.append(
                ActionReference(position=position, handle=handle, enabled=candidate.enabled)
            )
        domain_id = self._domain_id(model_observation)
        fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        snapshot = self._snapshot(
            world_tokens,
            candidates,
            domain_id=domain_id,
            encoding_fingerprint=fingerprint,
        )
        batch = self.collate_snapshots((snapshot,), device=device)
        return EncodedDecision(
            batch=batch,
            actions=tuple(references),
            snapshot=snapshot,
            encoding_fingerprint=fingerprint,
        )

    def stack(self, decisions: Sequence[EncodedDecision]) -> GroundedCandidateBatch:
        """Collate decision snapshots without repeating structural encoding."""

        if not decisions:
            raise ValueError("at least one decision is required")
        expected_fingerprint = decisions[0].encoding_fingerprint
        if any(
            item.encoding_fingerprint != expected_fingerprint
            for item in decisions[1:]
        ):
            raise ValueError("cannot stack decisions from different encoding contracts")
        if expected_fingerprint != grounding_encoding_identity()["fingerprint_sha256"]:
            raise ValueError("decision encoding contract differs from the active encoder")
        return self.collate_snapshots(
            tuple(item.snapshot for item in decisions),
            device=decisions[0].batch.world.features.device,
        )

    def collate_snapshots(
        self,
        snapshots: Sequence[EncodedDecisionSnapshot],
        *,
        device: torch.device | str | None = None,
    ) -> GroundedCandidateBatch:
        return collate_encoded_snapshots(
            tuple(snapshots),
            expected_config=self.config,
            expected_fingerprint=grounding_encoding_identity()["fingerprint_sha256"],
            device=device,
        )

    def _snapshot(
        self,
        world_tokens: Sequence[_Token],
        candidates: Sequence[_Candidate],
        *,
        domain_id: int,
        encoding_fingerprint: str,
    ) -> EncodedDecisionSnapshot:
        world = sparse_token_table(
            features=tuple(token.features for token in world_tokens),
            ids=tuple(
                (
                    token.type_id,
                    token.role_id,
                    token.owner_id,
                    token.entity_id,
                    token.entity_aux_id,
                    token.zone_id,
                    token.order_id,
                )
                for token in world_tokens
            ),
            feature_dim=self.config.feature_dim,
            id_width=7,
        )
        candidate_tokens = tuple(candidate.token for candidate in candidates)
        candidate_table = sparse_token_table(
            features=tuple(token.features for token in candidate_tokens),
            ids=tuple(
                (
                    token.type_id,
                    token.role_id,
                    token.owner_id,
                    token.entity_id,
                    token.entity_aux_id,
                    token.zone_id,
                    candidate.target_owner_id,
                    candidate.target_entity_id,
                    candidate.target_entity_aux_id,
                )
                for token, candidate in zip(candidate_tokens, candidates, strict=True)
            ),
            feature_dim=self.config.feature_dim,
            id_width=9,
        )
        flattened_locals = tuple(
            local for candidate in candidates for local in candidate.locals
        )
        local_table = sparse_token_table(
            features=tuple(token.features for token in flattened_locals),
            ids=tuple(
                (
                    token.type_id,
                    token.role_id,
                    token.owner_id,
                    token.entity_id,
                    token.entity_aux_id,
                    token.zone_id,
                    token.order_id,
                )
                for token in flattened_locals
            ),
            feature_dim=self.config.feature_dim,
            id_width=7,
        )
        local_offsets = [0]
        for candidate in candidates:
            local_offsets.append(local_offsets[-1] + len(candidate.locals))
        return EncodedDecisionSnapshot(
            config=self.config,
            encoding_fingerprint=encoding_fingerprint,
            world=world,
            candidates=candidate_table,
            locals=local_table,
            local_offsets=np.asarray(local_offsets, dtype=np.uint32),
            action_mask=np.asarray(
                [candidate.enabled for candidate in candidates],
                dtype=np.bool_,
            ),
            domain_id=domain_id,
        )

    def _domain_id(self, observation: Mapping[str, Any]) -> int:
        if bool(observation.get("terminated", False)):
            return _DOMAIN_IDS["terminal"]
        raw = str(
            observation.get("decision_domain")
            or observation.get("domain")
            or observation.get("phase")
            or "unknown"
        ).strip().lower()
        if raw in _DOMAIN_IDS:
            return _DOMAIN_IDS[raw]
        if raw in {"map", "navigation"}:
            return _DOMAIN_IDS["route"]
        return _DOMAIN_IDS["unknown"]

    def _world_tokens(self, observation: Mapping[str, Any]) -> tuple[_Token, ...]:
        queue: deque[_WalkItem] = deque([_WalkItem(observation, ("world",), "neutral", 0, 0)])
        result: list[_Token] = []
        while queue and len(result) < self.config.max_world_tokens:
            item = queue.popleft()
            token, children = self._tokenize_node(item, excluded=_WORLD_EXCLUDED_KEYS)
            result.append(token)
            queue.extend(children)
        if queue:
            raise ValueError(
                "world observation exceeds grounded token capacity; refusing "
                "lossy training input: "
                f"capacity={self.config.max_world_tokens} pending_nodes={len(queue)}"
            )
        return tuple(result)

    def _candidate_token(
        self,
        action: Mapping[str, Any],
        *,
        target_lookup: Mapping[str, dict[str, Any]],
    ) -> _Candidate:
        raw_model_kind = action.get("model_action_kind")
        if not isinstance(raw_model_kind, str) or not raw_model_kind.strip():
            raise ValueError("legal action is missing non-empty model_action_kind")
        kind = raw_model_kind.strip()
        if kind not in MODEL_ACTION_KIND_VOCABULARY:
            raise ValueError(f"unregistered model_action_kind: {kind!r}")
        root_variant = _string_field(
            action,
            (
                "model_action_variant",
                "selection",
                "character_id",
                "option_id",
                "room_model_id",
                "run_mode_action",
                "menu_action",
                "shop_action",
                "source",
                "mode",
            ),
        )
        role_label = f"{kind}:{root_variant}" if root_variant else kind
        owner = _owner_label(action, ("candidate",), "neutral")
        roots = _candidate_local_roots(
            action,
            model_kind=kind,
            target_lookup=target_lookup,
        )
        source: Mapping[str, Any] | None = None
        source_zone = "candidate"
        for key in _SOURCE_KEYS:
            candidate = roots.get(key)
            if candidate is not None:
                source = candidate
                source_zone = key
                break
        entity = _string_field(source or action, _IDENTITY_KEYS)
        if not entity and source is not None and source_zone == "option":
            # Event choices do not expose a canonical game ID today.  Hash the
            # raw player-visible option text as opaque identity; do not parse it
            # into effect deltas or use the action root's position-bearing UI
            # label.  This preserves distinguishability without a heuristic.
            visible_parts = [
                str(source[key]).strip()
                for key in ("title", "description", "text", "label")
                if source.get(key) is not None and str(source[key]).strip()
            ]
            entity = " | ".join(visible_parts)
        token = _Token(
            # Root action DTOs differ substantially between live/headless and
            # often carry previews or transport positions.  Only the action
            # kind/grounded entity and explicitly admitted nested objects are
            # model inputs; root numeric extras are never hashed.
            features=self._numeric_features(
                {},
                depth=0,
                order=0,
                path=("candidate",),
                excluded=_CANDIDATE_EXCLUDED_KEYS,
            ),
            type_id=_hash_id("type", "candidate", self.config.type_vocab_size),
            role_id=_hash_id("role", role_label, self.config.role_vocab_size),
            owner_id=_hash_id("owner", owner, self.config.owner_vocab_size),
            entity_id=_hash_id("entity", entity, self.config.entity_vocab_size),
            entity_aux_id=_hash_id(
                "entity_aux",
                entity,
                self.config.entity_vocab_size,
            ),
            zone_id=_hash_id("zone", source_zone, self.config.zone_vocab_size),
            order_id=0,
        )
        target_value = roots.get("target")
        target = target_value if isinstance(target_value, Mapping) else {}
        target_owner = _owner_label(target, ("candidate", "target"), "neutral")
        target_entity = _string_field(target, _IDENTITY_KEYS)

        locals_: list[_Token] = []
        queue: deque[_WalkItem] = deque()
        for key in sorted(roots):
            normalized_key = _normalize_key(key)
            if (
                normalized_key not in _CANDIDATE_LOCAL_ROOTS
                or self._excluded(key, _CANDIDATE_EXCLUDED_KEYS)
            ):
                continue
            child = roots[key]
            if isinstance(child, Mapping | list | tuple) and self._node_admitted(
                child,
                path=("candidate", str(key)),
                excluded=_CANDIDATE_EXCLUDED_KEYS,
            ):
                queue.append(_WalkItem(child, ("candidate", str(key)), owner, 0, 1))
        while queue and len(locals_) < self.config.max_candidate_local_tokens:
            item = queue.popleft()
            local, children = self._tokenize_node(item, excluded=_CANDIDATE_EXCLUDED_KEYS)
            locals_.append(local)
            queue.extend(children)
        if queue:
            # Finish walking only to report the exact required capacity.  The
            # encoder still fails closed: no legal-action fact is silently
            # truncated and no partial snapshot can reach the learner.
            required_tokens = len(locals_)
            while queue:
                item = queue.popleft()
                _local, children = self._tokenize_node(
                    item,
                    excluded=_CANDIDATE_EXCLUDED_KEYS,
                )
                required_tokens += 1
                queue.extend(children)
            raise ValueError(
                "candidate-local observation exceeds grounded token capacity; "
                "refusing lossy training input: "
                f"capacity={self.config.max_candidate_local_tokens} "
                f"required_tokens={required_tokens} "
                f"action_kind={kind!r}"
            )
        enabled = bool(action.get("is_enabled", action.get("enabled", True)))
        return _Candidate(
            token=token,
            target_owner_id=_hash_id("owner", target_owner, self.config.owner_vocab_size),
            target_entity_id=_hash_id("entity", target_entity, self.config.entity_vocab_size),
            target_entity_aux_id=_hash_id(
                "entity_aux",
                target_entity,
                self.config.entity_vocab_size,
            ),
            locals=tuple(locals_),
            enabled=enabled,
        )

    @staticmethod
    def _excluded(key: str, excluded: frozenset[str]) -> bool:
        lowered = _normalize_key(key)
        return (
            lowered.startswith("_")
            or lowered in excluded
            or any(fragment in lowered for fragment in _ENGINEERED_KEY_FRAGMENTS)
        )

    def _node_admitted(
        self,
        value: Any,
        *,
        path: tuple[str, ...],
        excluded: frozenset[str],
    ) -> bool:
        """Return whether a subtree contains a reviewed model-facing fact."""

        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                if self._excluded(key, excluded):
                    continue
                if self._is_factual_categorical(key, child) or self._is_factual_numeric(
                    key,
                    child,
                    path=path,
                    container=value,
                ):
                    return True
                if (
                    _normalize_key(key) in _FACT_CONTAINER_KEYS
                    and isinstance(child, Mapping | list | tuple)
                    and self._node_admitted(
                        child,
                        path=(*path, key),
                        excluded=excluded,
                    )
                ):
                    return True
            return False
        if isinstance(value, list | tuple):
            if not value:
                return bool(path and _normalize_key(path[-1]) in _FACT_CONTAINER_KEYS)
            for child in value:
                if isinstance(child, Mapping | list | tuple):
                    if self._node_admitted(
                        child,
                        path=(*path, "item"),
                        excluded=excluded,
                    ):
                        return True
                elif child is None or isinstance(child, str | int | float | bool):
                    return True
            return False
        return value is None or isinstance(value, str | int | float | bool)

    @staticmethod
    def _is_factual_categorical(key: str, value: Any) -> bool:
        lowered = _normalize_key(key)
        return (
            lowered in _FACT_CATEGORICAL_KEYS
            and not isinstance(value, bool)
            and isinstance(value, str | int)
            and bool(str(value).strip())
        )

    @staticmethod
    def _is_factual_numeric(
        key: str,
        value: Any,
        *,
        path: tuple[str, ...],
        container: Mapping[str, Any],
    ) -> bool:
        lowered = _normalize_key(key)
        normalized_path = {_normalize_key(part) for part in path}
        if not isinstance(value, bool | int | float):
            return False
        if "dynamic_vars" in normalized_path:
            return lowered in _DYNAMIC_VALUE_SLOT_BY_KEY
        identity = str(
            container.get("card_id")
            or container.get("id")
            or container.get("model_id")
            or ""
        ).strip().upper()
        if identity.startswith("CARD.") or normalized_path & _CARD_PATH_PARTS:
            if isinstance(value, bool):
                return lowered in _CARD_FACT_BOOLEAN_KEYS
            return lowered in _CARD_FACT_NUMERIC_KEYS
        if isinstance(value, bool):
            return lowered in _FACT_BOOLEAN_KEYS
        return lowered in _FACT_NUMERIC_KEYS or lowered.endswith(
            _FACT_NUMERIC_SUFFIXES
        )

    def _tokenize_node(
        self,
        item: _WalkItem,
        *,
        excluded: frozenset[str],
    ) -> tuple[_Token, tuple[_WalkItem, ...]]:
        value = item.value
        zone = item.path[-1] if item.path else "unknown"
        if isinstance(value, Mapping):
            owner = _owner_label(value, item.path, item.inherited_owner)
            role = _string_field(value, _ROLE_KEYS) or zone
            entity = _string_field(value, _IDENTITY_KEYS)
            mapping_children: list[_WalkItem] = []
            for key in sorted(value):
                if self._excluded(str(key), excluded):
                    continue
                child = value[key]
                if (
                    _normalize_key(key) in _FACT_CONTAINER_KEYS
                    and isinstance(child, Mapping | list | tuple)
                    and self._node_admitted(
                        child,
                        path=(*item.path, str(key)),
                        excluded=excluded,
                    )
                ):
                    mapping_children.append(
                        _WalkItem(
                            child,
                            (*item.path, str(key)),
                            owner,
                            0,
                            item.depth + 1,
                        )
                    )
            mapping_features = self._numeric_features(
                value,
                depth=item.depth,
                order=item.order,
                path=item.path,
                excluded=excluded,
            )
            token = _Token(
                features=mapping_features,
                type_id=_hash_id("type", f"mapping:{zone}", self.config.type_vocab_size),
                role_id=_hash_id("role", role, self.config.role_vocab_size),
                owner_id=_hash_id("owner", owner, self.config.owner_vocab_size),
                entity_id=_hash_id("entity", entity, self.config.entity_vocab_size),
                entity_aux_id=_hash_id(
                    "entity_aux",
                    entity,
                    self.config.entity_vocab_size,
                ),
                zone_id=_hash_id("zone", "/".join(item.path[-2:]), self.config.zone_vocab_size),
                order_id=min(max(item.order + 1, 0), self.config.max_order_id - 1),
            )
            return token, tuple(mapping_children)
        if isinstance(value, list | tuple):
            sequence_children = tuple(
                _WalkItem(child, (*item.path, "item"), item.inherited_owner, index, item.depth + 1)
                for index, child in enumerate(value)
                if not isinstance(child, Mapping | list | tuple)
                or self._node_admitted(
                    child,
                    path=(*item.path, "item"),
                    excluded=excluded,
                )
            )
            sequence_features = [0.0] * self.config.feature_dim
            sequence_features[0] = 1.0
            sequence_features[1] = min(1.0, len(value) / 64.0)
            sequence_features[2] = min(1.0, item.depth / 16.0)
            token = _Token(
                features=tuple(sequence_features),
                type_id=_hash_id("type", f"sequence:{zone}", self.config.type_vocab_size),
                role_id=_hash_id("role", zone, self.config.role_vocab_size),
                owner_id=_hash_id("owner", item.inherited_owner, self.config.owner_vocab_size),
                entity_id=1,
                entity_aux_id=1,
                zone_id=_hash_id("zone", "/".join(item.path[-2:]), self.config.zone_vocab_size),
                order_id=min(max(item.order + 1, 0), self.config.max_order_id - 1),
            )
            return token, sequence_children

        scalar_features = [0.0] * self.config.feature_dim
        scalar_features[0] = 1.0
        number = _bounded_number(value)
        if number is not None:
            scalar_features[3] = number
        entity = str(value) if isinstance(value, str | int) else type(value).__name__
        token = _Token(
            features=tuple(scalar_features),
            type_id=_hash_id("type", f"scalar:{type(value).__name__}", self.config.type_vocab_size),
            role_id=_hash_id("role", zone, self.config.role_vocab_size),
            owner_id=_hash_id("owner", item.inherited_owner, self.config.owner_vocab_size),
            entity_id=_hash_id("entity", entity, self.config.entity_vocab_size),
            entity_aux_id=_hash_id(
                "entity_aux",
                entity,
                self.config.entity_vocab_size,
            ),
            zone_id=_hash_id("zone", "/".join(item.path[-2:]), self.config.zone_vocab_size),
            order_id=min(max(item.order + 1, 0), self.config.max_order_id - 1),
        )
        return token, ()

    def _numeric_features(
        self,
        value: Mapping[str, Any],
        *,
        depth: int,
        order: int,
        path: tuple[str, ...],
        excluded: frozenset[str],
    ) -> tuple[float, ...]:
        features = [0.0] * self.config.feature_dim
        features[0] = 1.0
        features[1] = min(1.0, depth / 16.0)
        features[2] = min(1.0, order / 64.0)
        numeric_count = 0
        categorical_count = 0
        true_count = 0
        for key in sorted(value):
            if self._excluded(str(key), excluded):
                continue
            raw = value[key]
            if self._is_factual_categorical(str(key), raw):
                categorical_count += 1
                digest = hashlib.sha256(
                    f"category\0{_normalize_key(key)}\0{str(raw).lower()}".encode()
                ).digest()
                slot = _CATEGORY_SLOT_START + int.from_bytes(
                    digest[:4], "big"
                ) % _CATEGORY_SLOT_COUNT
                sign = 1.0 if digest[4] & 1 else -1.0
                features[slot] = max(
                    -1.0,
                    min(1.0, features[slot] + 0.5 * sign),
                )
                continue
            if not self._is_factual_numeric(
                str(key),
                raw,
                path=path,
                container=value,
            ):
                continue
            number = _bounded_number(raw)
            if number is None:
                continue
            numeric_count += 1
            true_count += int(isinstance(raw, bool) and raw)
            normalized_key = _normalize_key(key)
            normalized_path = {_normalize_key(part) for part in path}
            if "dynamic_vars" in normalized_path:
                if normalized_key in _DYNAMIC_VALUE_SLOT_BY_KEY:
                    slot = _DYNAMIC_VALUE_SLOT_BY_KEY[normalized_key]
                else:  # pragma: no cover - guarded by _is_factual_numeric
                    digest = hashlib.sha256(
                        f"dynamic\0{normalized_key}".encode()
                    ).digest()
                    slot = _DYNAMIC_HASH_SLOT_START + int.from_bytes(
                        digest[:4], "big"
                    ) % _DYNAMIC_HASH_SLOT_COUNT
            else:
                # `_is_factual_numeric` admits only reviewed fixed keys outside
                # dynamic_vars.  Index directly so two mechanics never alias.
                slot = _NUMERIC_SLOT_BY_KEY[normalized_key]
            features[slot] = max(-1.0, min(1.0, features[slot] + number))
        features[3] = min(1.0, numeric_count / 32.0)
        features[4] = min(1.0, categorical_count / 32.0)
        if numeric_count:
            features[5] = true_count / numeric_count
        return tuple(features)


__all__ = [
    "GROUNDING_ENCODING_VERSION",
    "MODEL_ACTION_KIND_VOCABULARY",
    "ActionReference",
    "EncodedDecision",
    "EncodedDecisionSnapshot",
    "GroundedEncodingConfig",
    "GroundedObservationEncoder",
    "grounding_encoding_identity",
]
