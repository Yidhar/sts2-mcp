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
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

import numpy as np
import torch

from sts2_rl.models.grounded_candidate import (
    MIN_TOKEN_FEATURE_DIM,
    GroundedCandidateBatch,
)
from sts2_rl.semantics.grouping import (
    strict_action_grouping_contract,
    strict_action_groups,
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
_INSTANCE_KEYS: Final[tuple[str, ...]] = (
    # These identifiers never define what an entity *is*.  They bind a legal
    # action to the same concrete runtime object in the world observation.
    # The model receives them through the separate entity_aux channel.
    "instance_uuid",
    "card_instance_id",
    "combat_uuid",
    "instance_id",
    "card_ref",
    "ref",
    "uuid",
    "uid",
    "combat_id",
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
        "act_floor",
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
        "heal_amount",
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
        "option_count",
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
        "rows",
        "columns",
        "slot_index",
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
        "upgrade_previews",
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
        "quantity",
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
        "act_id",
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
        "item_kind",
        "layout_type",
        "mode",
        "next_move_id",
        "next_move_state_id",
        "operation_type",
        "option_type",
        "owner_model_id",
        "owner_side",
        "applier_model_id",
        "applier_side",
        "pile",
        "prompt_id",
        "power_type",
        "rarity",
        "resource",
        "room_model",
        "room_model_id",
        "room_type",
        "run_mode_action",
        "screen",
        "selection_membership",
        "slot_name",
        "slot_type",
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
        "can_reroll",
        "can_throw_at_ally",
        "can_transition_away",
        "can_use_in_combat",
        "enough_gold",
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
        "is_affordable",
        "is_alive",
        "is_deterministic",
        "is_finished",
        "is_enabled",
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
        "is_on_sale",
        "is_open",
        "is_stocked",
        "is_visible",
        "is_travel_enabled",
        "is_traveling",
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
        "used",
        "visible",
        "proceed_visible",
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
# Exact native card-upgrade projections are factual alternative card states,
# not planner/action-effect previews.  Keep this exemption deliberately exact:
# ``effect_preview``, ``damage_preview`` and any future preview-like key remain
# behind the engineered-input firewall.
_ENGINEERED_FRAGMENT_EXEMPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "upgrade_preview",
        "upgrade_previews",
    }
)
_FACT_CONTAINER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "affliction",
        "afflictions",
        "action_group",
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
        "edges",
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
        "next_boss",
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
        "upgrade_previews",
        "keywords",
        "tags",
        "traits",
        "decision",
        "dimensions",
        "current_coord",
        "transaction",
    }
)
_SOURCE_KEYS: Final[tuple[str, ...]] = (
    "card",
    "character",
    "selected_character",
    "potion",
    "relic",
    "reward",
    "item",
    "option",
    "map_node",
)
_CANDIDATE_LOCAL_ROOTS: Final[frozenset[str]] = frozenset(
    {
        *_SOURCE_KEYS,
        "action_group",
        "target",
        "coord",
        "selection",
        "transaction",
        "upgrade_preview",
    }
)

# Stable physical/decision regions.  Known zones are never sent through a
# collision-prone hash in production.  Unknown future zones still have a
# deterministic fallback so a game update remains representable until the
# vocabulary is reviewed.
_ZONE_IDS: Final[dict[str, int]] = {
    "deck": 2,
    "hand": 3,
    "draw": 4,
    "discard": 5,
    "exhaust": 6,
    "play": 7,
    "reward": 8,
    "shop": 9,
    "map": 10,
    "rest_site": 11,
    "event": 12,
    "player": 13,
    "enemy": 14,
    "relic": 15,
    "potion": 16,
    "selection": 17,
    "power": 18,
    "intent": 19,
    "modifier": 20,
    "transaction": 21,
    "run": 22,
    "combat": 23,
    "candidate": 24,
    "world": 25,
}
_UNKNOWN_ZONE_HASH_START: Final = max(_ZONE_IDS.values()) + 1

# Feature ABI v1.  Numeric state keys must never share a slot: doing so makes
# observations such as ``block=3,max_hp=80`` alias a value-swapped state.  The
# fixed table is sorted for deterministic versioning and occupies a region
# disjoint from categorical and arbitrary ``dynamic_vars`` hashing.
_SUMMARY_SLOT_COUNT: Final = 6
_FIXED_NUMERIC_KEYS: Final[tuple[str, ...]] = tuple(
    sorted(_FACT_NUMERIC_KEYS | _FACT_BOOLEAN_KEYS | _CARD_FACT_NUMERIC_KEYS | _CARD_FACT_BOOLEAN_KEYS)
)
_NUMERIC_SLOT_BY_KEY: Final[dict[str, int]] = {
    key: _SUMMARY_SLOT_COUNT + index for index, key in enumerate(_FIXED_NUMERIC_KEYS)
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
    key: _DYNAMIC_SLOT_START + index for index, key in enumerate(_DYNAMIC_VALUE_KEYS)
}
_DYNAMIC_HASH_SLOT_START: Final = _DYNAMIC_SLOT_START + len(_DYNAMIC_VALUE_KEYS)
_DYNAMIC_HASH_SLOT_COUNT: Final = _DYNAMIC_SLOT_COUNT - len(_DYNAMIC_VALUE_KEYS)
_V8_FEATURE_ABI_END: Final = _DYNAMIC_SLOT_START + _DYNAMIC_SLOT_COUNT
# Preserve every v8 slot exactly.  Strict action-group multiplicity occupies a
# previously unused trailing feature rather than entering the sorted numeric
# table and shifting all later learned meanings.
_ACTION_GROUP_MULTIPLICITY_SLOT: Final = _V8_FEATURE_ABI_END
_FEATURE_ABI_END: Final = _ACTION_GROUP_MULTIPLICITY_SLOT + 1
# V13 admits only the exact native ``upgrade_preview(s)`` factual containers
# through the preview firewall and moves unknown future zones into the region
# above every fixed zone ID.  Tensor shapes and feature slots remain stable,
# but token presence and zone-embedding semantics change.  Exact resume must
# therefore fail closed; explicitly reviewed model-parameter initialization is
# still shape compatible.
GROUNDING_ENCODING_VERSION: Final = "grounded-relational-runtime-encoding-v13"

if _FEATURE_ABI_END > MIN_TOKEN_FEATURE_DIM:  # pragma: no cover - import invariant
    raise RuntimeError(
        "grounded feature ABI exceeds MIN_TOKEN_FEATURE_DIM: " f"{_FEATURE_ABI_END} > {MIN_TOKEN_FEATURE_DIM}"
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
        "entity_collision_policy": (
            "stable-hash-embeddings-plus-exact-decision-local-bindings-v1"
        ),
        "summary_slot_count": _SUMMARY_SLOT_COUNT,
        "numeric_slots": _NUMERIC_SLOT_BY_KEY,
        "category_slot_start": _CATEGORY_SLOT_START,
        "category_slot_count": _CATEGORY_SLOT_COUNT,
        "dynamic_slot_start": _DYNAMIC_SLOT_START,
        "dynamic_slot_count": _DYNAMIC_SLOT_COUNT,
        "dynamic_value_slots": _DYNAMIC_VALUE_SLOT_BY_KEY,
        "dynamic_hash_slot_start": _DYNAMIC_HASH_SLOT_START,
        "dynamic_hash_slot_count": _DYNAMIC_HASH_SLOT_COUNT,
        "action_group_multiplicity_slot": _ACTION_GROUP_MULTIPLICITY_SLOT,
        "world_excluded_keys": sorted(_WORLD_EXCLUDED_KEYS),
        "candidate_excluded_keys": sorted(_CANDIDATE_EXCLUDED_KEYS),
        "engineered_key_fragments": list(_ENGINEERED_KEY_FRAGMENTS),
        "engineered_fragment_exempt_keys": sorted(
            _ENGINEERED_FRAGMENT_EXEMPT_KEYS
        ),
        "strict_action_grouping": strict_action_grouping_contract(),
        "identity_keys": list(_IDENTITY_KEYS),
        "instance_keys": list(_INSTANCE_KEYS),
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
        "zone_ids": _ZONE_IDS,
        "unknown_zone_hash_start": _UNKNOWN_ZONE_HASH_START,
        "snapshot_version": ENCODED_DECISION_SNAPSHOT_VERSION,
    }
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return {
        "version": GROUNDING_ENCODING_VERSION,
        "min_token_feature_dim": MIN_TOKEN_FEATURE_DIM,
        "feature_abi_end": _FEATURE_ABI_END,
        "fingerprint_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }


_CAMEL_CASE_BOUNDARY: Final = re.compile(r"(?<!^)(?=[A-Z])")


@lru_cache(maxsize=8_192)
def _normalize_key_text(text: str) -> str:
    normalized = text.strip().replace("-", "_")
    return _CAMEL_CASE_BOUNDARY.sub("_", normalized).lower()


def _normalize_key(value: Any) -> str:
    # Field names, entity IDs, and relation labels recur thousands of times in
    # one run.  The normalization is pure and the bounded cache carries no
    # stochastic or checkpoint continuation state.
    return _normalize_key_text(str(value))


@dataclass(frozen=True, slots=True)
class ActionReference:
    """Dispatch-only identity for one semantic action candidate.

    ``position`` and ``handle`` identify the deterministic representative in
    the *raw* legal-action list.  The policy index is the position of this
    reference in :attr:`EncodedDecision.actions`; the two indexes intentionally
    diverge after strict action grouping.
    """

    position: int
    handle: str | None
    enabled: bool
    multiplicity: int = 1
    equivalence_fingerprint: str | None = None
    member_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or not isinstance(self.position, int):
            raise TypeError("action representative position must be an integer")
        if self.position < 0:
            raise ValueError("action representative position must be non-negative")
        if isinstance(self.multiplicity, bool) or not isinstance(self.multiplicity, int):
            raise TypeError("action multiplicity must be an integer")
        if self.multiplicity <= 0:
            raise ValueError("action multiplicity must be positive")
        members = self.member_positions or (self.position,)
        if len(members) != self.multiplicity:
            raise ValueError("action member count must equal multiplicity")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in members):
            raise TypeError("action member positions must be non-negative integers")
        if len(set(members)) != len(members):
            raise ValueError("action member positions must be unique")
        if members[0] != self.position:
            raise ValueError("action representative must be the first member position")
        object.__setattr__(self, "member_positions", tuple(members))

    @property
    def representative_position(self) -> int:
        """Return the raw legal-action position used for dispatch."""

        return self.position


@dataclass(frozen=True, slots=True)
class SemanticActionGroup:
    """One strict model-facing action prototype and its dispatch reference."""

    prototype: Mapping[str, Any]
    reference: ActionReference

    @property
    def multiplicity(self) -> int:
        return self.reference.multiplicity


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    """One model batch plus the non-learned action dispatch table."""

    batch: GroundedCandidateBatch
    actions: tuple[ActionReference, ...]
    snapshot: EncodedDecisionSnapshot
    encoding_fingerprint: str
    semantic_groups: tuple[SemanticActionGroup, ...]
    definition_hash_collisions: int = 0
    relation_hash_collisions: int = 0

    def __post_init__(self) -> None:
        if len(self.actions) != self.snapshot.candidate_count:
            raise ValueError("dispatch reference count must equal snapshot candidate count")
        if len(self.semantic_groups) != len(self.actions):
            raise ValueError("semantic group count must equal dispatch reference count")
        if any(
            group.reference != reference
            for group, reference in zip(
                self.semantic_groups,
                self.actions,
                strict=True,
            )
        ):
            raise ValueError("semantic group references must match the dispatch table")
        for label, value in (
            ("definition_hash_collisions", self.definition_hash_collisions),
            ("relation_hash_collisions", self.relation_hash_collisions),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an integer")
            if value < 0:
                raise ValueError(f"{label} must be non-negative")

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
    entity_key: str
    entity_aux_key: str
    entity_aux_is_relation: bool
    zone_id: int
    order_id: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    token: _Token
    target_owner_id: int
    target_entity_id: int
    target_entity_aux_id: int
    target_entity_key: str
    target_entity_aux_key: str
    target_entity_aux_is_relation: bool
    locals: tuple[_Token, ...]
    enabled: bool


@dataclass(frozen=True, slots=True)
class _EntityBindingAllocation:
    """Collision-free equality IDs plus auditable stable-hash collisions."""

    definitions: dict[str, int]
    relations: dict[tuple[str, bool], int]
    definition_hash_collisions: int
    relation_hash_collisions: int


@dataclass(frozen=True, slots=True)
class _WalkItem:
    value: Any
    path: tuple[str, ...]
    inherited_owner: str
    inherited_relation: str
    order: int
    depth: int


def _hash_id(namespace: str, value: Any, size: int) -> int:
    """Return a process-independent ID while reserving 0/1 for pad/unknown."""

    text = str(value).strip().lower()
    if not text:
        return 1
    digest = hashlib.sha256(f"{namespace}\0{text}".encode()).digest()
    return 2 + int.from_bytes(digest[:8], "big") % (size - 2)


def _stable_zone_id(value: Any, size: int) -> int:
    """Encode fixed zones collision-free and hash future zones disjointly.

    IDs 0/1 remain padding/unknown.  Reviewed zones retain their fixed IDs.
    Unknown future zones use only the remaining vocabulary above the largest
    fixed ID; a deliberately undersized test/model vocabulary has no such
    region and therefore returns the honest unknown ID instead of colliding
    with an unrelated reviewed zone.
    """

    normalized = _normalize_key(value)
    aliases = {
        "draw_pile": "draw",
        "discard_pile": "discard",
        "exhaust_pile": "exhaust",
        "play_pile": "play",
        "deck_cards": "deck",
        "card_reward": "reward",
        "card_reward_selection": "reward",
        "rewards": "reward",
        "rest": "rest_site",
        "restsite": "rest_site",
        "enchantments": "modifier",
        "afflictions": "modifier",
        "dynamic_vars": "modifier",
        "powers": "power",
        "intents": "intent",
        "enemies": "enemy",
        "relics": "relic",
        "potions": "potion",
        "decision": "selection",
    }
    canonical = aliases.get(normalized, normalized)
    stable = _ZONE_IDS.get(canonical)
    if stable is not None:
        return stable if stable < size else 1
    if not canonical or canonical == "unknown" or size <= _UNKNOWN_ZONE_HASH_START:
        return 1
    digest = hashlib.sha256(f"zone\0{canonical}".encode()).digest()
    return _UNKNOWN_ZONE_HASH_START + int.from_bytes(digest[:8], "big") % (
        size - _UNKNOWN_ZONE_HASH_START
    )


def _instance_field(value: Mapping[str, Any]) -> str:
    return _string_field(value, _INSTANCE_KEYS)


def _coord_relation(value: Mapping[str, Any]) -> str:
    raw_coord = value.get("coord")
    coord = raw_coord if isinstance(raw_coord, Mapping) else value
    x = _first_present(coord, "x", "col", "column")
    y = _first_present(coord, "y", "row")
    if (
        isinstance(x, int | float)
        and not isinstance(x, bool)
        and isinstance(y, int | float)
        and not isinstance(y, bool)
    ):
        return f"map_coord:{int(x)}:{int(y)}"
    return ""


def _relation_identity(
    value: Mapping[str, Any],
    *,
    path: tuple[str, ...],
    inherited: str,
) -> str:
    """Return a runtime relation key, distinct from definition identity.

    Per-instance IDs bind cards and enemies across piles/actions.  Coordinates
    bind map choices to the corresponding graph node.  Children without their
    own identity inherit the enclosing entity relation, which explicitly ties
    dynamic vars, modifiers, powers and intents to their owner.
    """

    instance = _instance_field(value)
    if instance:
        return f"instance:{instance}"
    coord = _coord_relation(value)
    if coord:
        return coord
    normalized_path = {_normalize_key(part) for part in path}
    leaf = _normalize_key(path[-1]) if path else "world"
    if leaf in {
        "player",
        "run",
        "combat",
        "event",
        "map",
        "shop",
        "rest_site",
        "rewards",
        "card_selection",
        "deck_upgrade_selection",
    }:
        return f"group:{leaf}"
    identity = _string_field(value, _IDENTITY_KEYS)
    if inherited and normalized_path & {
        "powers",
        "intents",
        "dynamic_vars",
        "enchantments",
        "afflictions",
    }:
        # Definition identity still distinguishes the child token. Sharing the
        # concrete relation ID binds it to the exact owning card/enemy/player.
        return inherited
    if identity and normalized_path & {
        "card",
        "cards",
        "deck_cards",
        "hand",
        "discard_pile",
        "exhaust_pile",
        "play_pile",
        "enemies",
        "relics",
        "potions",
        "powers",
        "intents",
        "items",
    }:
        return f"entity:{identity}"
    return inherited


def _zone_label(value: Mapping[str, Any], path: tuple[str, ...]) -> str:
    physical = _first_present(value, "source_pile", "pile", "zone")
    if physical is not None and str(physical).strip():
        return str(physical)
    for part in reversed(path):
        normalized = _normalize_key(part)
        if normalized in _ZONE_IDS or normalized in {
            "draw_pile",
            "discard_pile",
            "exhaust_pile",
            "play_pile",
            "deck_cards",
            "card_reward_selection",
            "rewards",
            "enchantments",
            "afflictions",
            "dynamic_vars",
            "powers",
            "intents",
            "enemies",
            "relics",
            "potions",
            "decision",
        }:
            return normalized
    return path[-1] if path else "unknown"


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
    # The simulator's maintained DTOs already use the canonical snake_case
    # spellings below.  Building a normalized copy of an entire card/entity
    # mapping for every individual field lookup made encoding effectively
    # quadratic in the number of fields (hundreds of thousands of regex calls
    # for a late-run deck).  Preserve the compatibility fallback for legacy
    # camel/Pascal-case payloads, but keep the overwhelmingly common canonical
    # path free of regex normalization.
    exact: Any = None
    for key in keys:
        if key in value and value[key] is not None:
            exact = value[key]
            break

    requested = {
        key if key == key.strip() and key == key.lower() and "-" not in key else _normalize_key(key)
        for key in keys
    }
    relevant_compatibility_alias = False
    for raw_key in value:
        if (
            isinstance(raw_key, str)
            and raw_key == raw_key.strip()
            and raw_key == raw_key.lower()
            and "-" not in raw_key
        ):
            continue
        if _normalize_key(raw_key) in requested:
            relevant_compatibility_alias = True
            break
    if not relevant_compatibility_alias:
        return exact

    # A relevant compatibility alias can collide with an exact spelling.  Use
    # the original full-map projection in that uncommon case so insertion-order
    # last-wins behavior remains byte-for-byte compatible with old checkpoints.
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
                {"name": key, **dict(item)}
                if isinstance(item, Mapping)
                else {
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
    instance = _first_present(value, *_INSTANCE_KEYS)
    if instance is not None and str(instance).strip():
        result["instance_id"] = str(instance).strip()
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
    instance = _first_present(value, *_INSTANCE_KEYS)
    if instance is not None and str(instance).strip():
        result["instance_id"] = str(instance).strip()
    slot_index = _first_present(value, "slot_index", "slot")
    if isinstance(slot_index, int) and not isinstance(slot_index, bool):
        result["slot_index"] = slot_index
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
    instance = _first_present(value, *_INSTANCE_KEYS)
    if instance is not None and str(instance).strip():
        result["instance_id"] = str(instance).strip()
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
    result["powers"].sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
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


def _visible_card_quantity(value: Mapping[str, Any], *, label: str) -> int:
    raw = value.get("quantity", 1)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"{label}.quantity must be a positive integer")
    return int(raw)


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
        values = _as_mapping_list(cards, label=f"{label}.cards")
        return sum(
            _visible_card_quantity(card, label=f"{label}.cards[{index}]")
            for index, card in enumerate(values)
        )
    values = _as_mapping_list(value, label=label)
    return sum(
        _visible_card_quantity(card, label=f"{label}[{index}]")
        for index, card in enumerate(values)
    )


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
    preserve_quantity: bool = False,
) -> list[dict[str, Any]] | None:
    cards = _pile_cards(value, label=label)
    if cards is None:
        return None
    result: list[dict[str, Any]] = []
    for index, card in enumerate(cards):
        enriched = dict(card)
        if pile is not None and _first_present(enriched, "source_pile", "pile", "zone") is None:
            enriched["pile"] = pile
        if membership is not None:
            enriched["selection_membership"] = membership
            enriched["is_selected"] = membership == "selected"
        canonical = _canonical_card(enriched)
        if preserve_quantity:
            canonical["quantity"] = _visible_card_quantity(
                enriched,
                label=f"{label}[{index}]",
            )
        result.append(canonical)
    if sort_as_set:
        result.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return result


def _aggregate_orderless_card_multiset(
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse fact-identical cards into an exact counted multiset.

    Permanent decks and public combat piles are orderless model surfaces.  A
    full card subtree for every physical copy duplicated immutable keywords,
    DynamicVars and modifier facts across both the deck and the current combat
    pile.  Long runs therefore grew the transformer input with copy count
    rather than semantic variety and eventually exceeded the hard world-token
    limit.

    Instance identity is deliberately removed only from these orderless
    multisets.  Every distinct factual variant (upgrade, cost, modifier,
    lifecycle flag, etc.) remains a separate entry, and ``quantity`` preserves
    exact multiplicity.  Hand/selection entities and legal-action locals remain
    unaggregated so the dispatcher can still bind a concrete selectable card.
    """

    grouped: dict[str, tuple[dict[str, Any], dict[str, Any], int]] = {}
    for raw_card in cards:
        original = dict(raw_card)
        quantity = _visible_card_quantity(original, label="orderless card")
        card = dict(original)
        card.pop("instance_id", None)
        card.pop("instance_uuid", None)
        card.pop("quantity", None)
        identity = json.dumps(card, sort_keys=True, separators=(",", ":"))
        existing = grouped.get(identity)
        if existing is None:
            grouped[identity] = (card, original, quantity)
        else:
            grouped[identity] = (
                existing[0],
                existing[1],
                existing[2] + quantity,
            )

    result: list[dict[str, Any]] = []
    for identity in sorted(grouped):
        card, original, quantity = grouped[identity]
        # A singleton can retain its concrete relation identity.  Once several
        # fact-identical copies collapse, the counted multiset intentionally
        # uses the shared factual variant instead of choosing one arbitrary
        # instance as representative.
        representative = original if quantity == 1 else card
        result.append({**representative, "quantity": quantity})
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


def _canonical_map(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, aliases in {
        "is_open": ("is_open",),
        "is_travel_enabled": ("is_travel_enabled",),
        "is_traveling": ("is_traveling",),
    }.items():
        item = _first_present(value, *aliases)
        if item is not None:
            result[key] = item
    raw_current = _first_present(value, "current_coord", "coord")
    if raw_current is not None:
        coord = _canonical_coord(_as_mapping(raw_current, label="map.current_coord"))
        if coord:
            result["current_coord"] = coord
    raw_dimensions = _first_present(value, "dimensions")
    if isinstance(raw_dimensions, Mapping):
        dimensions: dict[str, Any] = {}
        for key in ("rows", "columns"):
            item = _first_present(raw_dimensions, key)
            if item is not None:
                dimensions[key] = item
        if dimensions:
            result["dimensions"] = dimensions
    raw_points = _first_present(value, "points", "nodes")
    if raw_points is not None:
        points = [_canonical_map_node(point) for point in _as_mapping_list(raw_points, label="map.points")]
        canonical_points = [point for point in points if point]
        edges: list[dict[str, Any]] = []
        for point in canonical_points:
            source = str(point.get("instance_id") or "")
            for child in point.get("children", []):
                if not isinstance(child, Mapping):
                    continue
                target = str(child.get("instance_id") or "")
                if source and target:
                    edges.append(
                        {
                            "id": source,
                            "instance_id": target,
                            "type": "map_edge",
                        }
                    )
        if edges:
            result["edges"] = edges
        # Edges now carry the exact source/target relation pair, so retaining
        # nested child-coordinate copies would only multiply attention cost.
        result["points"] = [
            {key: item for key, item in point.items() if key != "children"} for point in canonical_points
        ]
    raw_next = _first_present(value, "next_options")
    if raw_next is not None:
        options = [_canonical_map_node(point) for point in _as_mapping_list(raw_next, label="map.next_options")]
        result["next_options"] = [point for point in options if point]
    return result


def _canonical_shop(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "visible",
        "is_open",
        "gold",
        "merchant_button_visible",
        "back_button_visible",
        "proceed_visible",
    ):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
    raw_items = _first_present(value, "items")
    if raw_items is not None:
        result["items"] = [
            item for raw in _as_mapping_list(raw_items, label="shop.items") if (item := _canonical_item(raw))
        ]
    return result


def _canonical_rest_site(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("visible", "proceed_visible"):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
    raw_options = _first_present(value, "options")
    if raw_options is not None:
        options: list[dict[str, Any]] = []
        for raw in _as_mapping_list(raw_options, label="rest_site.options"):
            option = _canonical_option(raw)
            heal_amount = _first_present(raw, "heal_amount")
            if heal_amount is not None:
                option["heal_amount"] = heal_amount
            if option:
                options.append(option)
        result["options"] = options
    return result


def _canonical_reward(
    value: Mapping[str, Any],
    *,
    slot_index: int | None = None,
) -> dict[str, Any]:
    """Project one claimable reward without inventing hidden contents."""

    result: dict[str, Any] = {}
    reward_type = _first_present(value, "reward_type", "type", "kind")
    if reward_type is not None and str(reward_type).strip():
        normalized_type = str(reward_type).strip()
        result["id"] = f"reward:{normalized_type}"
        result["type"] = normalized_type
    resolved_slot = _first_present(value, "slot_index", "index")
    if resolved_slot is None:
        resolved_slot = slot_index
    if isinstance(resolved_slot, int) and not isinstance(resolved_slot, bool):
        result["slot_index"] = resolved_slot
        result["instance_id"] = f"reward_slot:{resolved_slot}"
    for key in ("amount", "rarity", "option_count", "card_count"):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
    for key in ("can_skip", "can_reroll"):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
    for key, projector in (
        ("card", _canonical_card),
        ("relic", _canonical_relic),
        ("potion", _canonical_potion),
    ):
        raw = _first_present(value, key)
        if raw is not None:
            result[key] = projector(_as_mapping(raw, label=f"reward.{key}"))
    return result


def _canonical_rewards(value: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only explicit reward entities; no reward-quality annotation."""

    result: dict[str, Any] = {}
    for field, projector in (
        ("cards", _canonical_card),
        ("relics", _canonical_relic),
        ("potions", _canonical_potion),
    ):
        raw_values = _first_present(value, field)
        if raw_values is None:
            continue
        result[field] = [
            projected for raw in _as_mapping_list(raw_values, label=f"rewards.{field}") if (projected := projector(raw))
        ]
    for key in ("can_skip", "proceed_visible"):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
    raw_entries = _first_present(value, "rewards", "items")
    if raw_entries is not None:
        items: list[dict[str, Any]] = []
        for entry in _as_mapping_list(raw_entries, label="rewards.items"):
            raw_reward = _first_present(entry, "reward")
            reward = _as_mapping(raw_reward, label="rewards.items.reward") if raw_reward is not None else entry
            raw_slot = _first_present(entry, "slot_index", "index")
            slot = raw_slot if isinstance(raw_slot, int) and not isinstance(raw_slot, bool) else None
            normalized = _canonical_reward(reward, slot_index=slot)
            if normalized:
                items.append(normalized)
        if items:
            result["items"] = items
    return result


def _canonical_model_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Project live/headless state to one deliberately shared model DTO.

    Inactive UI sections are not part of the model contract.  Public discard
    and exhaust composition, physical source piles, and exact multi-selection
    membership are retained because they change legal decisions.  Public draw
    composition is represented as an orderless multiset; hidden draw order is
    never retained.
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
        deck_count = _pile_count(deck_cards, label="player deck") or 0

    canonical_deck_sequence = _canonical_card_sequence(
        deck_cards,
        label="player.deck_cards",
        pile="Deck",
        sort_as_set=True,
        preserve_quantity=True,
    ) or []
    canonical_deck_cards = _aggregate_orderless_card_multiset(
        canonical_deck_sequence
    )
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
    player["powers"].sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
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
        player[field].sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
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
    next_boss_id = _first_present(raw_run, "next_boss_id")
    if next_boss_id is not None and str(next_boss_id).strip():
        # Keep the visible encounter as its own entity instead of folding it
        # into the run token's categorical hash.  The current room already
        # owns the run token's definition identity; a child preserves both
        # identities without adding a boss rule, score, or future outcome.
        run["next_boss"] = {"id": str(next_boss_id).strip()}

    canonical: dict[str, Any] = {
        "phase": phase,
        "decision_domain": domain,
        "run": run,
        "player": player,
    }
    if bool(observation.get("terminated", False)):
        canonical["terminated"] = True

    raw_enemies = _first_present(raw_combat, "enemies")
    combat_active = phase == "combat" or _first_present(raw_combat, "in_progress") is True or bool(raw_enemies)
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
            )
            or [],
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
            ("draw_pile", ("draw_pile",), "Draw"),
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
                preserve_quantity=True,
            )
            if cards is not None:
                combat[key] = _aggregate_orderless_card_multiset(cards)
        canonical["combat"] = combat

    raw_event = _as_mapping(observation.get("event"), label="observation.event")
    if raw_event:
        event = _canonical_event(raw_event)
        if event:
            canonical["event"] = event

    raw_map = _as_mapping(observation.get("map"), label="observation.map")
    if raw_map and domain != "combat":
        map_state = _canonical_map(raw_map)
        if map_state:
            canonical["map"] = map_state

    raw_shop = _as_mapping(observation.get("shop"), label="observation.shop")
    if raw_shop and (
        _first_present(raw_shop, "visible", "is_open") is True
        or bool(_first_present(raw_shop, "items"))
        or phase == "shop"
    ):
        shop = _canonical_shop(raw_shop)
        if shop:
            canonical["shop"] = shop

    raw_rest_site = _as_mapping(
        observation.get("rest_site"),
        label="observation.rest_site",
    )
    if raw_rest_site and (
        _first_present(raw_rest_site, "visible") is True
        or bool(_first_present(raw_rest_site, "options"))
        or phase == "rest_site"
    ):
        rest_site = _canonical_rest_site(raw_rest_site)
        if rest_site:
            canonical["rest_site"] = rest_site

    raw_rewards = _as_mapping(
        observation.get("rewards"),
        label="observation.rewards",
    )
    rewards = _canonical_rewards(raw_rewards) if raw_rewards else {}
    raw_card_reward = _as_mapping(
        observation.get("card_reward_selection"),
        label="observation.card_reward_selection",
    )
    if raw_card_reward:
        cards = _canonical_card_sequence(
            _first_present(raw_card_reward, "cards"),
            label="card_reward_selection.cards",
            pile="Reward",
        )
        if cards is not None:
            rewards["cards"] = cards
        can_skip = _first_present(raw_card_reward, "can_skip")
        if can_skip is not None:
            rewards["can_skip"] = can_skip
    if rewards:
        canonical["rewards"] = rewards

    raw_decision = _as_mapping(
        observation.get("decision"),
        label="observation.decision",
    )
    explicit_selection = _as_mapping(
        observation.get("card_selection"),
        label="observation.card_selection",
    )
    explicit_upgrade = _as_mapping(
        observation.get("deck_upgrade_selection"),
        label="observation.deck_upgrade_selection",
    )
    # Live places the card option membership on top-level card_selection while
    # its compact decision block carries counts.  Headless places both in the
    # explicit selection DTO.  Merge the two instead of allowing the summary
    # block to hide card identity.
    raw_selection: dict[str, Any] = dict(explicit_selection)
    if explicit_upgrade and (explicit_upgrade.get("visible") is True or phase == "deck_upgrade"):
        raw_selection.update(explicit_upgrade)
        raw_selection.setdefault("operation_type", "upgrade")
        raw_selection.setdefault("source_zone", "Deck")
        raw_selection.setdefault("destination_zone", "Deck")
    raw_selection.update(raw_decision)
    # Card/hand prompts can occur inside combat while the top-level phase stays
    # ``combat``.  Presence of the explicit selection DTO, rather than the
    # screen phase, determines whether selection state is model-visible.
    if raw_selection:
        selected_cards = _first_present(raw_selection, "selected_cards")
        selected_count = _first_present(raw_selection, "selected_count")
        if selected_count is None and selected_cards is not None:
            selected_count = _pile_count(
                selected_cards,
                label="selection.selected_cards",
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
            preserve_quantity=True,
        )
        selected = _canonical_card_sequence(
            selected_cards,
            label="selection.selected_cards",
            membership="selected",
            sort_as_set=True,
            preserve_quantity=True,
        )
        raw_options = _first_present(raw_selection, "options")
        if raw_options is not None:
            option_selectable: list[dict[str, Any]] = []
            option_selected: list[dict[str, Any]] = []
            upgrade_previews: list[dict[str, Any]] = []
            for option in _as_mapping_list(raw_options, label="selection.options"):
                raw_card = _first_present(option, "card")
                if raw_card is None:
                    continue
                card = _as_mapping(raw_card, label="selection.options.card")
                membership = "selected" if option.get("is_selected") is True else "selectable"
                canonical_cards = (
                    _canonical_card_sequence(
                        [card],
                        label="selection.options.card",
                        membership=membership,
                    )
                    or []
                )
                (option_selected if membership == "selected" else option_selectable).extend(canonical_cards)
                raw_preview = _first_present(option, "upgrade_preview")
                if isinstance(raw_preview, Mapping):
                    preview = _canonical_card(raw_preview)
                    source_instance = canonical_cards[0].get("instance_id") if canonical_cards else None
                    if source_instance:
                        preview["instance_id"] = source_instance
                    if preview:
                        upgrade_previews.append(preview)
            if selectable is None:
                selectable = option_selectable
            if selected is None:
                selected = option_selected
            if upgrade_previews:
                selection["upgrade_previews"] = upgrade_previews
        if selectable is not None:
            # Selection grids are orderless.  Keep selectable and selected
            # membership as separate multisets, but do not spend one world
            # token subtree per fact-identical physical copy.  This mirrors
            # the strict card-fact projection used by semantic action groups
            # while retaining exact multiplicity.
            selection["selectable_cards"] = _aggregate_orderless_card_multiset(
                selectable
            )
        if selected is not None:
            selection["selected_cards"] = _aggregate_orderless_card_multiset(
                selected
            )
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
        "is_enabled": ("is_enabled",),
        "is_proceed": ("is_proceed", "proceed"),
        "is_selected": ("is_selected",),
        "heal_amount": ("heal_amount",),
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
    coord_source = _as_mapping(raw_coord, label="map_node.coord") if raw_coord is not None else value
    coord = _canonical_coord(coord_source)
    if coord:
        result["coord"] = coord
        result["instance_id"] = f"map_coord:{int(coord['x'])}:{int(coord['y'])}"
    for key, aliases in {
        "point_type": ("point_type", "room_type", "type"),
        "room_model_id": ("room_model_id", "room_model", "model_id"),
    }.items():
        item = _first_present(value, *aliases)
        if item is not None:
            result[key] = item
    raw_children = _first_present(value, "children")
    if isinstance(raw_children, Sequence) and not isinstance(raw_children, str | bytes):
        children: list[dict[str, Any]] = []
        for index, raw_child in enumerate(raw_children):
            if isinstance(raw_child, Mapping):
                child = _canonical_coord(raw_child)
            elif isinstance(raw_child, Sequence) and not isinstance(raw_child, str | bytes) and len(raw_child) >= 2:
                child = {"x": raw_child[0], "y": raw_child[1]}
            else:
                raise TypeError(f"map_node.children[{index}] must be a coordinate")
            if child:
                child["instance_id"] = f"map_coord:{int(child['x'])}:{int(child['y'])}"
                children.append(child)
        result["children"] = children
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
        "type": ("type", "item_type", "item_kind", "kind"),
        "price": ("price", "cost"),
        "rarity": ("rarity",),
        "slot_type": ("slot_type",),
        "is_affordable": ("is_affordable",),
        "enough_gold": ("enough_gold",),
        "is_stocked": ("is_stocked",),
        "is_on_sale": ("is_on_sale",),
        "used": ("used",),
        "slot_index": ("slot_index", "index"),
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
    # Shop slots often have no game-model ID of their own.  Their coordinate
    # within the current inventory is a factual relation key shared by the
    # world item and its legal purchase candidate; it is not candidate order.
    if "id" not in result and isinstance(result.get("slot_index"), int):
        result["instance_id"] = f"shop_slot:{result['slot_index']}"
    for nested_key in ("card", "relic", "potion"):
        nested = result.get(nested_key)
        if not isinstance(nested, Mapping):
            continue
        if "id" not in result and nested.get("id"):
            result["id"] = nested["id"]
        if "instance_id" not in result and nested.get("instance_id"):
            result["instance_id"] = nested["instance_id"]
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


def _candidate_transaction(
    action: Mapping[str, Any],
    *,
    model_kind: str,
    roots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe the exact immediate mutation family of a legal action.

    This is protocol semantics, not a predicted outcome.  It intentionally
    stops at effects the environment guarantees immediately: a shop purchase
    spends gold and adds/opens the advertised item, a card reward adds a card,
    a selection moves or mutates the selected instance, and leaving preserves
    inventory.  Combat damage, future draws and event outcomes are never
    fabricated here.
    """

    variant = _string_field(
        action,
        (
            "model_action_variant",
            "shop_action",
            "selection_operation",
            "operation_type",
            "kind",
            "action",
        ),
    )
    normalized_variant = _normalize_key(variant)
    transaction: dict[str, Any] = {"type": "transaction"}

    if model_kind == "shop":
        if normalized_variant in {"back", "leave", "skip", "shop_skip"}:
            transaction.update(
                operation_type="leave_shop",
                resource="gold",
                amount=0,
                source_zone="Shop",
                destination_zone="Run",
            )
            return transaction
        item = roots.get("item", {})
        item_type = _normalize_key(item.get("type", "item"))
        operation = "purchase_card_removal" if item_type == "card_removal" else f"purchase_{item_type or 'item'}"
        price = item.get("price")
        transaction.update(
            operation_type=operation,
            resource="gold",
            source_zone="Shop",
            destination_zone={
                "card": "Deck",
                "relic": "Relic",
                "potion": "Potion",
                "card_removal": "Selection",
            }.get(item_type, "Run"),
        )
        if isinstance(price, int | float) and not isinstance(price, bool):
            transaction["price"] = price
            transaction["amount"] = -float(price)
        return transaction

    if model_kind == "rest_site":
        option = roots.get("option", {})
        option_type = _normalize_key(option.get("option_type", option.get("option_id", normalized_variant)))
        if "heal" in option_type or option_type in {"rest", "sleep"}:
            operation = "heal"
            resource = "hp"
        elif "upgrade" in option_type or "smith" in option_type:
            operation = "open_upgrade_selection"
            resource = "upgrade_level"
        else:
            operation = f"rest_option:{option_type or 'unknown'}"
            resource = "run_state"
        transaction.update(
            operation_type=operation,
            resource=resource,
            source_zone="RestSite",
            destination_zone="Run" if operation == "heal" else "Selection",
        )
        heal_amount = option.get("heal_amount")
        if isinstance(heal_amount, int | float) and not isinstance(heal_amount, bool):
            transaction["amount"] = heal_amount
        return transaction

    if model_kind == "card_reward":
        has_card = isinstance(roots.get("card"), Mapping)
        transaction.update(
            operation_type="add_card" if has_card else "skip_card_reward",
            resource="deck",
            amount=1 if has_card else 0,
            source_zone="Reward",
            destination_zone="Deck" if has_card else "Run",
        )
        return transaction

    if model_kind == "reward":
        reward = roots.get("reward", {})
        reward_type = _normalize_key(reward.get("type", "unknown"))
        operation, resource, destination = {
            "gold": ("claim_gold", "gold", "Run"),
            "card": ("open_card_reward", "card_options", "Reward"),
            "relic": ("claim_relic_reward", "relic", "Relic"),
            "potion": ("claim_potion_reward", "potion_slot", "Potion"),
        }.get(
            reward_type,
            ("claim_reward", "run_state", "Run"),
        )
        transaction.update(
            operation_type=operation,
            resource=resource,
            source_zone="Reward",
            destination_zone=destination,
        )
        amount = reward.get("amount")
        if isinstance(amount, int | float) and not isinstance(amount, bool):
            transaction["amount"] = amount
        elif reward_type in {"relic", "potion"}:
            transaction["amount"] = 1
        return transaction

    if model_kind == "card_selection":
        selection = roots.get("selection", {})
        operation = str(selection.get("operation_type") or normalized_variant or "select")
        transaction.update(
            operation_type=operation,
            resource="selection_membership",
            source_zone=selection.get("source_zone", "Selection"),
            destination_zone="Selection",
        )
        normalized_operation = _normalize_key(operation)
        if normalized_operation in {"select", "toggle_on"}:
            transaction["amount"] = 1
        elif normalized_operation in {"deselect", "toggle_off"}:
            transaction["amount"] = -1
        elif normalized_operation in {"confirm", "cancel"}:
            transaction["amount"] = 0
        return transaction

    if model_kind == "deck_upgrade":
        transaction.update(
            operation_type="upgrade_card",
            resource="upgrade_level",
            amount=1,
            source_zone="Deck",
            destination_zone="Deck",
        )
        return transaction

    if model_kind == "map":
        transaction.update(
            operation_type="move_to_map_node",
            resource="map_position",
            amount=1,
            source_zone="Map",
            destination_zone="Map",
        )
        return transaction

    operation_by_kind = {
        "play_card": ("play_card", "energy", "Hand", "Play", None),
        "use_potion": ("use_potion", "potion_slot", "Potion", "Combat", -1),
        "discard_potion": ("discard_potion", "potion_slot", "Potion", "Run", -1),
        "end_turn": ("end_turn", "turn", "Combat", "Combat", 1),
        "treasure": ("claim_treasure", "inventory", "Reward", "Run", None),
        "treasure_relic": ("add_relic", "relic", "Reward", "Relic", 1),
        "event_option": (
            "choose_event_option",
            "event_state",
            "Event",
            "Event",
            None,
        ),
        "proceed": ("proceed", "run_state", "Run", "Run", None),
    }
    resolved = operation_by_kind.get(model_kind)
    if resolved is None:
        return {}
    operation, resource, source_zone, destination_zone, amount = resolved
    transaction.update(
        operation_type=operation,
        resource=resource,
        source_zone=source_zone,
        destination_zone=destination_zone,
    )
    if model_kind == "play_card":
        card = roots.get("card", {})
        cost = card.get("cost")
        costs_x = any(isinstance(trait, Mapping) and trait.get("id") == "costs_x" for trait in card.get("traits", []))
        if isinstance(cost, int | float) and not isinstance(cost, bool) and not costs_x:
            transaction["amount"] = -float(cost)
    elif amount is not None:
        transaction["amount"] = amount
    return transaction


def _candidate_local_roots(
    action: Mapping[str, Any],
    *,
    model_kind: str,
    target_lookup: Mapping[str, dict[str, Any]],
    multiplicity: int = 1,
) -> dict[str, dict[str, Any]]:
    roots: dict[str, dict[str, Any]] = {}
    projectors: dict[
        str,
        Callable[[Mapping[str, Any]], dict[str, Any]],
    ] = {
        "card": _canonical_card,
        "character": lambda value: {"id": _first_present(value, "id", "character_id", "model_id")},
        "selected_character": lambda value: {"id": _first_present(value, "id", "character_id", "model_id")},
        "potion": _canonical_potion,
        "relic": _canonical_relic,
        "reward": _canonical_reward,
        "item": _canonical_item,
        "option": _canonical_option,
        "map_node": _canonical_map_node,
        "coord": _canonical_coord,
        "selection": _canonical_selection,
        "typed_selection": _canonical_selection,
        "upgrade_preview": _canonical_card,
    }
    for key, projector in projectors.items():
        raw = action.get(key)
        if isinstance(raw, Mapping):
            projected = projector(raw)
            if projected:
                roots[key] = projected
    source_card = roots.get("card")
    upgrade_preview = roots.get("upgrade_preview")
    if source_card is not None and upgrade_preview is not None:
        source_instance = source_card.get("instance_id")
        if source_instance:
            upgrade_preview["instance_id"] = source_instance

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
    transaction = _candidate_transaction(
        action,
        model_kind=model_kind,
        roots=roots,
    )
    if transaction:
        roots["transaction"] = transaction
    if multiplicity > 1:
        roots["action_group"] = {
            "type": "strict_equivalence",
            "multiplicity": multiplicity,
        }
    return roots


class GroundedObservationEncoder:
    """Convert bridge observations to the grounded candidate tensor contract."""

    def __init__(self, config: GroundedEncodingConfig | None = None) -> None:
        self.config = config or GroundedEncodingConfig()

    def semantic_action_groups(
        self,
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> tuple[SemanticActionGroup, ...]:
        """Return the lightweight strict action surface without tensorizing.

        This is the shared authority for collector diagnostics, transaction
        tracking, next-state deadlock checks, and :meth:`encode`.  It performs
        no world walk and does not enforce the model capacity; callers that
        only need the semantic action surface can therefore inspect an
        overflow state without repeating the grouping algorithm.
        """

        if isinstance(legal_actions, str | bytes) or not isinstance(
            legal_actions,
            Sequence,
        ):
            raise TypeError("legal_actions must be a sequence of mappings")
        validated_actions: list[Mapping[str, Any]] = []
        for position, action in enumerate(legal_actions):
            if not isinstance(action, Mapping):
                raise TypeError(f"legal action {position} must be a mapping")
            validated_actions.append(action)
        # ``semantics.grouping`` is the sole grouping authority.  Encoding,
        # deadlock detection, and action dispatch must never maintain parallel
        # notions of strict equality.
        builders = strict_action_groups(validated_actions)
        result: list[SemanticActionGroup] = []
        for group in builders:
            representative_position = group.member_positions[0]
            representative = validated_actions[representative_position]
            handle_value = representative.get(
                "action_handle",
                representative.get("action_id"),
            )
            handle = (
                str(handle_value)
                if handle_value is not None and str(handle_value)
                else None
            )
            result.append(
                SemanticActionGroup(
                    prototype=group.prototype,
                    reference=ActionReference(
                        position=representative_position,
                        handle=handle,
                        enabled=bool(
                            representative.get(
                                "is_enabled",
                                representative.get("enabled", True),
                            )
                        ),
                        multiplicity=len(group.member_positions),
                        equivalence_fingerprint=group.equivalence_fingerprint,
                        member_positions=tuple(group.member_positions),
                    ),
                )
            )
        return tuple(result)

    def encode(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
        *,
        device: torch.device | str | None = None,
    ) -> EncodedDecision:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        action_groups = self.semantic_action_groups(legal_actions)
        if len(action_groups) > self.config.max_candidates:
            raise ValueError(
                "legal action count exceeds the grounded model capacity; refusing to "
                "silently hide dispatchable candidates: "
                f"count={len(action_groups)} capacity={self.config.max_candidates} "
                f"raw_count={len(legal_actions)}"
            )
        model_observation = _canonical_model_observation(observation)
        world_tokens = self._world_tokens(model_observation)
        target_lookup = _target_fact_lookup(observation)
        candidates: list[_Candidate] = []
        references: list[ActionReference] = []
        # ``role_id`` is also the parameter-free policy branch ID. Hashing is
        # still appropriate for embeddings, but two distinct co-occurring
        # branch labels must never be silently merged into one probability
        # marginal. Fail closed at the untrusted action boundary if the finite
        # role vocabulary collides for this decision.
        branch_labels_by_id: dict[int, str] = {}
        for group in action_groups:
            branch_label = self._candidate_role_label(group.prototype)
            branch_id = _hash_id(
                "role",
                branch_label,
                self.config.role_vocab_size,
            )
            previous_branch_label = branch_labels_by_id.setdefault(
                branch_id,
                branch_label,
            )
            if previous_branch_label != branch_label:
                raise ValueError(
                    "co-occurring semantic action branches collide in the "
                    "configured role vocabulary; refusing to merge policy "
                    "marginals: "
                    f"role_id={branch_id} "
                    f"first={previous_branch_label!r} "
                    f"second={branch_label!r}"
                )
            candidate = self._candidate_token(
                group.prototype,
                target_lookup=target_lookup,
                multiplicity=group.multiplicity,
            )
            candidates.append(candidate)
            if candidate.enabled is not group.reference.enabled:  # pragma: no cover - strict invariant
                raise RuntimeError("semantic action prototype changed representative enabled state")
            references.append(group.reference)
        bindings = self._entity_binding_allocation(world_tokens, candidates)
        domain_id = self._domain_id(model_observation)
        fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        snapshot = self._snapshot(
            world_tokens,
            candidates,
            bindings=bindings,
            domain_id=domain_id,
            encoding_fingerprint=fingerprint,
        )
        batch = self.collate_snapshots((snapshot,), device=device)
        return EncodedDecision(
            batch=batch,
            actions=tuple(references),
            snapshot=snapshot,
            encoding_fingerprint=fingerprint,
            semantic_groups=action_groups,
            definition_hash_collisions=(
                bindings.definition_hash_collisions
            ),
            relation_hash_collisions=bindings.relation_hash_collisions,
        )

    @staticmethod
    def _definition_binding_key(value: str) -> str:
        return str(value).strip().lower()

    @classmethod
    def _relation_binding_key(
        cls,
        value: str,
        is_relation: bool,
    ) -> tuple[str, bool]:
        return cls._definition_binding_key(value), bool(is_relation)

    def _entity_binding_allocation(
        self,
        world_tokens: Sequence[_Token],
        candidates: Sequence[_Candidate],
    ) -> _EntityBindingAllocation:
        """Allocate exact, deterministic equality IDs for one decision.

        Learned entity embeddings intentionally retain their stable finite SHA
        buckets.  Equality pooling must not inherit those buckets' collisions,
        so this method gathers semantic keys across world, candidate, target,
        and local tables and assigns sorted unique keys dense decision-local
        IDs starting at 2 (0=padding, 1=unknown).  The auxiliary namespace
        includes the relation-vs-definition-fallback marker: identical text in
        those two roles is not the same exact relation.

        Collision counts report the number of excess distinct semantic keys
        occupying a stable embedding bucket.  They are diagnostic only; a
        collision no longer stops collection or corrupts equality pooling.
        """

        definition_keys: set[str] = set()
        relation_keys: set[tuple[str, bool]] = set()
        definition_hash_buckets: dict[int, set[str]] = {}
        relation_hash_buckets: dict[int, set[tuple[str, bool]]] = {}

        def register(
            *,
            entity_id: int,
            entity_aux_id: int,
            entity_key: str,
            entity_aux_key: str,
            entity_aux_is_relation: bool,
        ) -> None:
            normalized_entity = self._definition_binding_key(entity_key)
            normalized_relation = self._relation_binding_key(
                entity_aux_key,
                entity_aux_is_relation,
            )
            if normalized_entity:
                definition_keys.add(normalized_entity)
                definition_hash_buckets.setdefault(entity_id, set()).add(
                    normalized_entity
                )
            if normalized_relation[0]:
                relation_keys.add(normalized_relation)
                relation_hash_buckets.setdefault(entity_aux_id, set()).add(
                    normalized_relation
                )

        def register_token(token: _Token) -> None:
            register(
                entity_id=token.entity_id,
                entity_aux_id=token.entity_aux_id,
                entity_key=token.entity_key,
                entity_aux_key=token.entity_aux_key,
                entity_aux_is_relation=token.entity_aux_is_relation,
            )

        for token in world_tokens:
            register_token(token)
        for candidate in candidates:
            register_token(candidate.token)
            register(
                entity_id=candidate.target_entity_id,
                entity_aux_id=candidate.target_entity_aux_id,
                entity_key=candidate.target_entity_key,
                entity_aux_key=candidate.target_entity_aux_key,
                entity_aux_is_relation=(
                    candidate.target_entity_aux_is_relation
                ),
            )
            for token in candidate.locals:
                register_token(token)

        max_binding_id = int(np.iinfo(np.int32).max)
        if len(definition_keys) > max_binding_id - 1:
            raise ValueError("definition binding namespace exceeds int32 capacity")
        if len(relation_keys) > max_binding_id - 1:
            raise ValueError("relation binding namespace exceeds int32 capacity")

        def collision_count(buckets: Mapping[int, set[Any]]) -> int:
            return sum(max(0, len(keys) - 1) for keys in buckets.values())

        return _EntityBindingAllocation(
            definitions={
                key: index
                for index, key in enumerate(sorted(definition_keys), start=2)
            },
            relations={
                key: index
                for index, key in enumerate(sorted(relation_keys), start=2)
            },
            definition_hash_collisions=collision_count(
                definition_hash_buckets
            ),
            relation_hash_collisions=collision_count(relation_hash_buckets),
        )

    def stack(self, decisions: Sequence[EncodedDecision]) -> GroundedCandidateBatch:
        """Collate decision snapshots without repeating structural encoding."""

        if not decisions:
            raise ValueError("at least one decision is required")
        expected_fingerprint = decisions[0].encoding_fingerprint
        if any(item.encoding_fingerprint != expected_fingerprint for item in decisions[1:]):
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
        bindings: _EntityBindingAllocation,
        domain_id: int,
        encoding_fingerprint: str,
    ) -> EncodedDecisionSnapshot:
        def definition_binding(value: str) -> int:
            key = self._definition_binding_key(value)
            return bindings.definitions[key] if key else 1

        def relation_binding(value: str, is_relation: bool) -> int:
            key = self._relation_binding_key(value, is_relation)
            return bindings.relations[key] if key[0] else 1

        world = sparse_token_table(
            features=tuple(token.features for token in world_tokens),
            ids=tuple(
                (
                    token.type_id,
                    token.role_id,
                    token.owner_id,
                    token.entity_id,
                    token.entity_aux_id,
                    definition_binding(token.entity_key),
                    relation_binding(
                        token.entity_aux_key,
                        token.entity_aux_is_relation,
                    ),
                    token.zone_id,
                    token.order_id,
                )
                for token in world_tokens
            ),
            feature_dim=self.config.feature_dim,
            id_width=9,
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
                    definition_binding(token.entity_key),
                    relation_binding(
                        token.entity_aux_key,
                        token.entity_aux_is_relation,
                    ),
                    token.zone_id,
                    candidate.target_owner_id,
                    candidate.target_entity_id,
                    candidate.target_entity_aux_id,
                    definition_binding(candidate.target_entity_key),
                    relation_binding(
                        candidate.target_entity_aux_key,
                        candidate.target_entity_aux_is_relation,
                    ),
                )
                for token, candidate in zip(candidate_tokens, candidates, strict=True)
            ),
            feature_dim=self.config.feature_dim,
            id_width=13,
        )
        flattened_locals = tuple(local for candidate in candidates for local in candidate.locals)
        local_table = sparse_token_table(
            features=tuple(token.features for token in flattened_locals),
            ids=tuple(
                (
                    token.type_id,
                    token.role_id,
                    token.owner_id,
                    token.entity_id,
                    token.entity_aux_id,
                    definition_binding(token.entity_key),
                    relation_binding(
                        token.entity_aux_key,
                        token.entity_aux_is_relation,
                    ),
                    token.zone_id,
                    token.order_id,
                )
                for token in flattened_locals
            ),
            feature_dim=self.config.feature_dim,
            id_width=9,
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
        raw = (
            str(
                observation.get("decision_domain") or observation.get("domain") or observation.get("phase") or "unknown"
            )
            .strip()
            .lower()
        )
        if raw in _DOMAIN_IDS:
            return _DOMAIN_IDS[raw]
        if raw in {"map", "navigation"}:
            return _DOMAIN_IDS["route"]
        return _DOMAIN_IDS["unknown"]

    def _world_tokens(self, observation: Mapping[str, Any]) -> tuple[_Token, ...]:
        queue: deque[_WalkItem] = deque([_WalkItem(observation, ("world",), "neutral", "", 0, 0)])
        result: list[_Token] = []
        branch_tokens: Counter[str] = Counter()
        while queue and len(result) < self.config.max_world_tokens:
            item = queue.popleft()
            token, children = self._tokenize_node(item, excluded=_WORLD_EXCLUDED_KEYS)
            result.append(token)
            branch_tokens[item.path[1] if len(item.path) > 1 else "world"] += 1
            queue.extend(children)
        if queue:
            # Overflow is exceptional, so finish the structural walk only on
            # this path.  Exact demand and top-level branch counts make the
            # next failure actionable without adding per-decision logging or
            # silently truncating any game fact.
            required_tokens = len(result)
            while queue:
                item = queue.popleft()
                _token, children = self._tokenize_node(
                    item,
                    excluded=_WORLD_EXCLUDED_KEYS,
                )
                required_tokens += 1
                branch_tokens[item.path[1] if len(item.path) > 1 else "world"] += 1
                queue.extend(children)
            raise ValueError(
                "world observation exceeds grounded token capacity; refusing "
                "lossy training input: "
                f"capacity={self.config.max_world_tokens} "
                f"required_tokens={required_tokens} "
                "branch_tokens="
                f"{json.dumps(dict(sorted(branch_tokens.items())), separators=(',', ':'))}"
            )
        return tuple(result)

    def _candidate_token(
        self,
        action: Mapping[str, Any],
        *,
        target_lookup: Mapping[str, dict[str, Any]],
        multiplicity: int = 1,
    ) -> _Candidate:
        role_label = self._candidate_role_label(action)
        kind = str(action["model_action_kind"]).strip()
        owner = _owner_label(action, ("candidate",), "neutral")
        roots = _candidate_local_roots(
            action,
            model_kind=kind,
            target_lookup=target_lookup,
            multiplicity=multiplicity,
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
        source_relation = _relation_identity(
            source or action,
            path=("candidate", source_zone),
            inherited="",
        )
        source_zone_label = _zone_label(source, ("candidate", source_zone)) if source is not None else source_zone
        if source_zone == "item" and kind == "shop":
            source_zone_label = "Shop"
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
        source_aux = source_relation or entity
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
                source_aux,
                self.config.entity_vocab_size,
            ),
            entity_key=entity,
            entity_aux_key=source_aux,
            entity_aux_is_relation=bool(source_relation),
            zone_id=_stable_zone_id(
                source_zone_label,
                self.config.zone_vocab_size,
            ),
            order_id=0,
        )
        target_value = roots.get("target")
        target = target_value if isinstance(target_value, Mapping) else {}
        target_owner = _owner_label(target, ("candidate", "target"), "neutral")
        target_entity = _string_field(target, _IDENTITY_KEYS)
        target_relation = _relation_identity(
            target,
            path=("candidate", "target"),
            inherited="",
        )
        target_aux = target_relation or target_entity

        locals_: list[_Token] = []
        queue: deque[_WalkItem] = deque()
        for key in sorted(roots):
            normalized_key = _normalize_key(key)
            if normalized_key not in _CANDIDATE_LOCAL_ROOTS or self._excluded(key, _CANDIDATE_EXCLUDED_KEYS):
                continue
            child = roots[key]
            if isinstance(child, Mapping | list | tuple) and self._node_admitted(
                child,
                path=("candidate", str(key)),
                excluded=_CANDIDATE_EXCLUDED_KEYS,
            ):
                queue.append(
                    _WalkItem(
                        child,
                        ("candidate", str(key)),
                        owner,
                        source_relation,
                        0,
                        1,
                    )
                )
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
                target_aux,
                self.config.entity_vocab_size,
            ),
            target_entity_key=target_entity,
            target_entity_aux_key=target_aux,
            target_entity_aux_is_relation=bool(target_relation),
            locals=tuple(locals_),
            enabled=enabled,
        )

    @staticmethod
    def _candidate_role_label(action: Mapping[str, Any]) -> str:
        """Return the exact semantic branch label encoded into ``role_id``."""

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
        return f"{kind}:{root_variant}" if root_variant else kind

    @staticmethod
    def _excluded(key: str, excluded: frozenset[str]) -> bool:
        lowered = _normalize_key(key)
        return (
            lowered.startswith("_")
            or lowered in excluded
            or (
                lowered not in _ENGINEERED_FRAGMENT_EXEMPT_KEYS
                and any(
                    fragment in lowered
                    for fragment in _ENGINEERED_KEY_FRAGMENTS
                )
            )
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
        if "action_group" in normalized_path:
            return lowered == "multiplicity" and not isinstance(value, bool)
        if "dynamic_vars" in normalized_path:
            return lowered in _DYNAMIC_VALUE_SLOT_BY_KEY
        identity = (
            str(container.get("card_id") or container.get("id") or container.get("model_id") or "").strip().upper()
        )
        leaf = _normalize_key(path[-1]) if path else ""
        # Shop-item wrappers inherit their nested card's stable definition ID
        # so purchase candidates bind to the exact world slot.  The wrapper is
        # still an item: price/affordability/sale flags must use the generic
        # factual contract rather than the narrower card-instance contract.
        identity_is_card = identity.startswith("CARD.") and leaf != "item"
        if identity_is_card or normalized_path & _CARD_PATH_PARTS:
            if isinstance(value, bool):
                return lowered in _CARD_FACT_BOOLEAN_KEYS
            return lowered in _CARD_FACT_NUMERIC_KEYS
        if isinstance(value, bool):
            return lowered in _FACT_BOOLEAN_KEYS
        return lowered in _FACT_NUMERIC_KEYS or lowered.endswith(_FACT_NUMERIC_SUFFIXES)

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
            relation = _relation_identity(
                value,
                path=item.path,
                inherited=item.inherited_relation,
            )
            entity_aux = relation or entity
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
                            relation,
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
                    entity_aux,
                    self.config.entity_vocab_size,
                ),
                entity_key=entity,
                entity_aux_key=entity_aux,
                entity_aux_is_relation=bool(relation),
                zone_id=_stable_zone_id(
                    _zone_label(value, item.path),
                    self.config.zone_vocab_size,
                ),
                order_id=min(max(item.order + 1, 0), self.config.max_order_id - 1),
            )
            return token, tuple(mapping_children)
        if isinstance(value, list | tuple):
            sequence_children = tuple(
                _WalkItem(
                    child,
                    (*item.path, "item"),
                    item.inherited_owner,
                    item.inherited_relation,
                    index,
                    item.depth + 1,
                )
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
                entity_aux_id=_hash_id(
                    "entity_aux",
                    item.inherited_relation,
                    self.config.entity_vocab_size,
                ),
                entity_key="",
                entity_aux_key=item.inherited_relation,
                entity_aux_is_relation=bool(item.inherited_relation),
                zone_id=_stable_zone_id(zone, self.config.zone_vocab_size),
                order_id=min(max(item.order + 1, 0), self.config.max_order_id - 1),
            )
            return token, sequence_children

        scalar_features = [0.0] * self.config.feature_dim
        scalar_features[0] = 1.0
        number = _bounded_number(value)
        if number is not None:
            scalar_features[3] = number
        entity = str(value) if isinstance(value, str | int) else type(value).__name__
        entity_aux = item.inherited_relation or entity
        token = _Token(
            features=tuple(scalar_features),
            type_id=_hash_id("type", f"scalar:{type(value).__name__}", self.config.type_vocab_size),
            role_id=_hash_id("role", zone, self.config.role_vocab_size),
            owner_id=_hash_id("owner", item.inherited_owner, self.config.owner_vocab_size),
            entity_id=_hash_id("entity", entity, self.config.entity_vocab_size),
            entity_aux_id=_hash_id(
                "entity_aux",
                entity_aux,
                self.config.entity_vocab_size,
            ),
            entity_key=entity,
            entity_aux_key=entity_aux,
            entity_aux_is_relation=bool(item.inherited_relation),
            zone_id=_stable_zone_id(zone, self.config.zone_vocab_size),
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
                digest = hashlib.sha256(f"category\0{_normalize_key(key)}\0{str(raw).lower()}".encode()).digest()
                slot = _CATEGORY_SLOT_START + int.from_bytes(digest[:4], "big") % _CATEGORY_SLOT_COUNT
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
            if "action_group" in normalized_path and normalized_key == "multiplicity":
                slot = _ACTION_GROUP_MULTIPLICITY_SLOT
            elif "dynamic_vars" in normalized_path:
                if normalized_key in _DYNAMIC_VALUE_SLOT_BY_KEY:
                    slot = _DYNAMIC_VALUE_SLOT_BY_KEY[normalized_key]
                else:  # pragma: no cover - guarded by _is_factual_numeric
                    digest = hashlib.sha256(f"dynamic\0{normalized_key}".encode()).digest()
                    slot = _DYNAMIC_HASH_SLOT_START + int.from_bytes(digest[:4], "big") % _DYNAMIC_HASH_SLOT_COUNT
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
    "SemanticActionGroup",
    "grounding_encoding_identity",
]
