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

import torch

from sts2_rl.models.grounded_candidate import (
    MIN_TOKEN_FEATURE_DIM,
    CandidateTokenBatch,
    GroundedCandidateBatch,
    GroundedCandidateConfig,
    WorldTokenBatch,
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
    "option_id",
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
        "draw",
        "energy",
        "exhaust",
        "floor",
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
        "min_count",
        "min_select",
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
        "cost",
        "current_upgrade_level",
        "energy_cost",
        "max_upgrade_level",
        "price",
        "upgrade_level",
        "upgrades",
    }
)
_CARD_FACT_BOOLEAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "is_playable",
        "is_upgraded",
        "playable",
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
        "decision_domain",
        "facing",
        "intent_type",
        "next_move_id",
        "pile",
        "rarity",
        "room_model",
        "room_model_id",
        "room_type",
        "run_mode_action",
        "screen",
        "state_type",
        "target_type",
        "transport_kind",
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
        "can_skip",
        "cancelable",
        "confirm_ready",
        "exhaust",
        "exhausts",
        "ethereal",
        "in_combat",
        "in_progress",
        "is_alive",
        "is_hittable",
        "is_playable",
        "is_upgraded",
        "playable",
        "retain",
        "retained",
        "run_active",
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
        "enchantments",
        "enemies",
        "event",
        "exhaust_pile",
        "game_over",
        "hand",
        "hand_select",
        "intents",
        "intent",
        "item",
        "items",
        "map",
        "map_node",
        "modifiers",
        "next_options",
        "nodes",
        "option",
        "options",
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
_FEATURE_ABI_END: Final = _DYNAMIC_SLOT_START + _DYNAMIC_SLOT_COUNT
GROUNDING_ENCODING_VERSION: Final = "grounded-structural-encoding-v1"

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
class GroundedEncodingConfig:
    """Fixed-shape tensor and hashing contract for the new baseline."""

    feature_dim: int = 160
    max_world_tokens: int = 256
    max_candidates: int = 96
    max_candidate_local_tokens: int = 16
    type_vocab_size: int = 128
    role_vocab_size: int = 64
    owner_vocab_size: int = 128
    entity_vocab_size: int = 8192
    zone_vocab_size: int = 32
    max_order_id: int = 128
    domain_count: int = 8

    @classmethod
    def from_model_config(
        cls,
        model: GroundedCandidateConfig,
        *,
        max_world_tokens: int = 256,
        max_candidates: int = 96,
        max_candidate_local_tokens: int = 16,
    ) -> GroundedEncodingConfig:
        """Create an encoder whose tensor IDs are valid for ``model``."""

        return cls(
            feature_dim=model.token_feature_dim,
            max_world_tokens=max_world_tokens,
            max_candidates=max_candidates,
            max_candidate_local_tokens=max_candidate_local_tokens,
            type_vocab_size=model.type_vocab_size,
            role_vocab_size=model.role_vocab_size,
            owner_vocab_size=model.owner_vocab_size,
            entity_vocab_size=model.entity_vocab_size,
            zone_vocab_size=model.zone_vocab_size,
            max_order_id=model.order_vocab_size,
            domain_count=model.domain_count,
        )

    def __post_init__(self) -> None:
        integer_fields = (
            "feature_dim",
            "max_world_tokens",
            "max_candidates",
            "max_candidate_local_tokens",
            "type_vocab_size",
            "role_vocab_size",
            "owner_vocab_size",
            "entity_vocab_size",
            "zone_vocab_size",
            "max_order_id",
            "domain_count",
        )
        wrong_types = [
            name
            for name in integer_fields
            if isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), int)
        ]
        if wrong_types:
            raise TypeError(
                "grounded encoding dimensions must be exact integers: "
                + ", ".join(wrong_types)
            )
        if self.feature_dim < MIN_TOKEN_FEATURE_DIM:
            raise ValueError(
                "feature_dim must be at least "
                f"{MIN_TOKEN_FEATURE_DIM} for the grounded feature ABI"
            )
        for name in (
            "max_world_tokens",
            "max_candidates",
            "max_candidate_local_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "type_vocab_size",
            "role_vocab_size",
            "owner_vocab_size",
            "entity_vocab_size",
            "zone_vocab_size",
        ):
            if getattr(self, name) < 4:
                raise ValueError(f"{name} must be at least 4")
        if self.max_order_id < 2:
            raise ValueError("max_order_id must be at least 2")
        if self.domain_count <= max(_DOMAIN_IDS.values()):
            raise ValueError("domain_count is too small for the fixed domain vocabulary")


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


def _canonical_card(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project a card to facts observable on both maintained backends."""

    result: dict[str, Any] = {}
    aliases = {
        "id": ("id", "card_id", "model_id"),
        "type": ("type", "card_type"),
        "target_type": ("target_type",),
        "cost": ("cost", "energy_cost", "resolved_energy_cost"),
        "upgrade_level": ("upgrade_level", "current_upgrade_level"),
        "max_upgrade_level": ("max_upgrade_level",),
        "is_playable": ("is_playable", "playable"),
        "is_upgraded": ("is_upgraded", "upgraded"),
        "exhaust": ("exhaust",),
        "ethereal": ("ethereal",),
        "retain": ("retain", "retained"),
    }
    for canonical, source_keys in aliases.items():
        item = _first_present(value, *source_keys)
        if item is not None:
            result[canonical] = item
    return result


def _canonical_power(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    aliases = {
        "id": ("id", "power_id", "model_id"),
        "amount": ("amount", "stacks", "value"),
        "type": ("type", "stack_type"),
    }
    for canonical, source_keys in aliases.items():
        item = _first_present(value, *source_keys)
        if item is not None:
            result[canonical] = item
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
    for key in ("rarity", "charges", "target_type", "type"):
        item = _first_present(value, key)
        if item is not None:
            result[key] = item
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
    for key in ("side", "hp", "max_hp", "block", "is_alive", "is_hittable"):
        item = _first_present(value, key, f"current_{key}")
        if item is not None:
            result[key] = item
    powers = _as_mapping_list(
        _first_present(value, "powers", "status"),
        label="enemy powers",
    )
    result["powers"] = [_canonical_power(item) for item in powers]
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
    return result


def _pile_count(value: Any, *, label: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a number or card sequence")
    if isinstance(value, int | float):
        return value
    return len(_as_mapping_list(value, label=label))


def _canonical_model_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Project live/headless state to one deliberately shared model DTO.

    Inactive UI sections and simulator-only pile identities are not part of the
    model contract. Legal candidates carry active choices; world state contains
    only run, player, combat, and typed selection progress available in both
    maintained deployments.
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

    player: dict[str, Any] = {
        "deck": deck_count,
        "deck_cards": [_canonical_card(card) for card in deck_cards],
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
    for relic in _as_mapping_list(
        _first_present(raw_player, "relics"),
        label="player relics",
    ):
        normalized = _canonical_inventory_entity(
            relic,
            id_keys=("id", "relic_id", "model_id"),
        )
        if normalized.get("id"):
            player["relics"].append(normalized)
    for potion in _as_mapping_list(
        _first_present(raw_player, "potions"),
        label="player potions",
    ):
        normalized = _canonical_inventory_entity(
            potion,
            id_keys=("id", "potion_id", "model_id"),
        )
        if normalized.get("id"):
            player["potions"].append(normalized)
    for key, aliases in {
        "character_id": ("character_id", "character"),
        "hp": ("hp", "current_hp"),
        "max_hp": ("max_hp",),
        "block": ("block",),
        "gold": ("gold",),
    }.items():
        item = _first_present(raw_player, *aliases)
        if item is not None and not isinstance(item, Mapping):
            player[key] = item

    run: dict[str, Any] = {}
    for key, aliases in {
        "active": ("active", "run_active"),
        "game_over": ("game_over",),
        "floor": ("floor", "total_floor"),
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
            "hand": [
                _canonical_card(card)
                for card in _as_mapping_list(raw_hand, label="combat hand")
            ],
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
        canonical["combat"] = combat

    raw_decision = _as_mapping(
        observation.get("decision"),
        label="observation.decision",
    )
    raw_selection = raw_decision
    if not raw_selection and phase in {"card_selection", "deck_upgrade"}:
        raw_selection = _as_mapping(
            observation.get("card_selection"),
            label="observation.card_selection",
        )
    if raw_selection and phase in {"card_selection", "deck_upgrade"}:
        selected_cards = _first_present(raw_selection, "selected_cards")
        selected_count = _first_present(raw_selection, "selected_count")
        if selected_count is None and selected_cards is not None:
            selected_count = len(
                _as_mapping_list(
                    selected_cards,
                    label="selection.selected_cards",
                )
            )
        selection: dict[str, Any] = {}
        for key, item in {
            "selected_count": selected_count,
            "min_select": _first_present(raw_selection, "min_select"),
            "max_select": _first_present(raw_selection, "max_select"),
            "remaining_select": _first_present(raw_selection, "remaining_select"),
            "confirm_ready": _first_present(
                raw_selection,
                "confirm_ready",
                "can_confirm",
            ),
            "can_skip": _first_present(raw_selection, "can_skip"),
            "cancelable": _first_present(raw_selection, "cancelable"),
        }.items():
            if item is not None:
                selection[key] = item
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
        "option_id": ("option_id", "id", "model_id"),
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
    for key, aliases in {
        "selected_count": ("selected_count",),
        "min_select": ("min_select", "min_count"),
        "max_select": ("max_select", "max_count"),
        "remaining_select": ("remaining_select",),
        "confirm_ready": ("confirm_ready", "can_confirm"),
        "can_skip": ("can_skip",),
        "cancelable": ("cancelable",),
        "is_selected": ("is_selected",),
        "operation_type": ("operation_type",),
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
            lambda item: _canonical_inventory_entity(
                item,
                id_keys=("id", "potion_id", "model_id"),
            ),
        ),
        (
            "relic",
            lambda item: _canonical_inventory_entity(
                item,
                id_keys=("id", "relic_id", "model_id"),
            ),
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
        "potion": lambda value: _canonical_inventory_entity(
            value,
            id_keys=("id", "potion_id", "model_id"),
        ),
        "relic": lambda value: _canonical_inventory_entity(
            value,
            id_keys=("id", "relic_id", "model_id"),
        ),
        "item": _canonical_item,
        "option": _canonical_option,
        "map_node": _canonical_map_node,
        "coord": _canonical_coord,
        "selection": _canonical_selection,
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
        batch = self._materialize(world_tokens, candidates, domain_id, device=device)
        return EncodedDecision(
            batch=batch,
            actions=tuple(references),
            encoding_fingerprint=grounding_encoding_identity()[
                "fingerprint_sha256"
            ],
        )

    def stack(self, decisions: Sequence[EncodedDecision]) -> GroundedCandidateBatch:
        """Stack fixed-shape single-decision batches for learning."""

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
        return GroundedCandidateBatch(
            world=WorldTokenBatch(
                features=torch.cat([item.batch.world.features for item in decisions], dim=0),
                mask=torch.cat([item.batch.world.mask for item in decisions], dim=0),
                type_ids=torch.cat([item.batch.world.type_ids for item in decisions], dim=0),
                role_ids=torch.cat([item.batch.world.role_ids for item in decisions], dim=0),
                owner_ids=torch.cat([item.batch.world.owner_ids for item in decisions], dim=0),
                entity_ids=torch.cat([item.batch.world.entity_ids for item in decisions], dim=0),
                entity_aux_ids=torch.cat(
                    [item.batch.world.entity_aux_ids for item in decisions],
                    dim=0,
                ),
                zone_ids=torch.cat([item.batch.world.zone_ids for item in decisions], dim=0),
                order_ids=torch.cat([item.batch.world.order_ids for item in decisions], dim=0),
            ),
            candidates=CandidateTokenBatch(
                features=torch.cat(
                    [item.batch.candidates.features for item in decisions], dim=0
                ),
                type_ids=torch.cat(
                    [item.batch.candidates.type_ids for item in decisions], dim=0
                ),
                role_ids=torch.cat(
                    [item.batch.candidates.role_ids for item in decisions], dim=0
                ),
                owner_ids=torch.cat(
                    [item.batch.candidates.owner_ids for item in decisions], dim=0
                ),
                entity_ids=torch.cat(
                    [item.batch.candidates.entity_ids for item in decisions], dim=0
                ),
                entity_aux_ids=torch.cat(
                    [item.batch.candidates.entity_aux_ids for item in decisions],
                    dim=0,
                ),
                zone_ids=torch.cat(
                    [item.batch.candidates.zone_ids for item in decisions], dim=0
                ),
                target_owner_ids=torch.cat(
                    [item.batch.candidates.target_owner_ids for item in decisions], dim=0
                ),
                target_entity_ids=torch.cat(
                    [item.batch.candidates.target_entity_ids for item in decisions], dim=0
                ),
                target_entity_aux_ids=torch.cat(
                    [
                        item.batch.candidates.target_entity_aux_ids
                        for item in decisions
                    ],
                    dim=0,
                ),
                local_features=torch.cat(
                    [item.batch.candidates.local_features for item in decisions], dim=0
                ),
                local_mask=torch.cat(
                    [item.batch.candidates.local_mask for item in decisions], dim=0
                ),
                local_type_ids=torch.cat(
                    [item.batch.candidates.local_type_ids for item in decisions], dim=0
                ),
                local_role_ids=torch.cat(
                    [item.batch.candidates.local_role_ids for item in decisions], dim=0
                ),
                local_owner_ids=torch.cat(
                    [item.batch.candidates.local_owner_ids for item in decisions], dim=0
                ),
                local_entity_ids=torch.cat(
                    [item.batch.candidates.local_entity_ids for item in decisions], dim=0
                ),
                local_entity_aux_ids=torch.cat(
                    [
                        item.batch.candidates.local_entity_aux_ids
                        for item in decisions
                    ],
                    dim=0,
                ),
                local_zone_ids=torch.cat(
                    [item.batch.candidates.local_zone_ids for item in decisions], dim=0
                ),
                local_order_ids=torch.cat(
                    [item.batch.candidates.local_order_ids for item in decisions], dim=0
                ),
                action_mask=torch.cat(
                    [item.batch.candidates.action_mask for item in decisions], dim=0
                ),
            ),
            domain_ids=torch.cat([item.batch.domain_ids for item in decisions], dim=0),
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
            raise ValueError(
                "candidate-local observation exceeds grounded token capacity; "
                "refusing lossy training input: "
                f"capacity={self.config.max_candidate_local_tokens} "
                f"pending_nodes={len(queue)}"
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
            return isinstance(value, int | float) and not isinstance(value, bool)
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
                digest = hashlib.sha256(
                    f"dynamic\0{normalized_key}".encode()
                ).digest()
                slot = _DYNAMIC_SLOT_START + int.from_bytes(
                    digest[:4], "big"
                ) % _DYNAMIC_SLOT_COUNT
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

    def _materialize(
        self,
        world_tokens: Sequence[_Token],
        candidates: Sequence[_Candidate],
        domain_id: int,
        *,
        device: torch.device | str | None,
    ) -> GroundedCandidateBatch:
        cfg = self.config
        dev = torch.device(device) if device is not None else torch.device("cpu")
        world_features = torch.zeros((1, cfg.max_world_tokens, cfg.feature_dim), dtype=torch.float32, device=dev)
        world_mask = torch.zeros((1, cfg.max_world_tokens), dtype=torch.bool, device=dev)
        world_ids = [
            torch.zeros((1, cfg.max_world_tokens), dtype=torch.long, device=dev)
            for _ in range(7)
        ]
        for index, token in enumerate(world_tokens[: cfg.max_world_tokens]):
            world_features[0, index] = torch.tensor(token.features, dtype=torch.float32, device=dev)
            world_mask[0, index] = True
            for tensor, value in zip(
                world_ids,
                (
                    token.type_id,
                    token.role_id,
                    token.owner_id,
                    token.entity_id,
                    token.entity_aux_id,
                    token.zone_id,
                    token.order_id,
                ),
                strict=True,
            ):
                tensor[0, index] = value

        shape = (1, cfg.max_candidates)
        candidate_features = torch.zeros((*shape, cfg.feature_dim), dtype=torch.float32, device=dev)
        candidate_ids = [
            torch.zeros(shape, dtype=torch.long, device=dev) for _ in range(9)
        ]
        action_mask = torch.zeros(shape, dtype=torch.bool, device=dev)
        local_shape = (1, cfg.max_candidates, cfg.max_candidate_local_tokens)
        local_features = torch.zeros((*local_shape, cfg.feature_dim), dtype=torch.float32, device=dev)
        local_mask = torch.zeros(local_shape, dtype=torch.bool, device=dev)
        local_ids = [
            torch.zeros(local_shape, dtype=torch.long, device=dev) for _ in range(7)
        ]
        for action_index, candidate in enumerate(candidates[: cfg.max_candidates]):
            token = candidate.token
            candidate_features[0, action_index] = torch.tensor(
                token.features, dtype=torch.float32, device=dev
            )
            for tensor, value in zip(
                candidate_ids,
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
                ),
                strict=True,
            ):
                tensor[0, action_index] = value
            action_mask[0, action_index] = candidate.enabled
            for local_index, local in enumerate(
                candidate.locals[: cfg.max_candidate_local_tokens]
            ):
                local_features[0, action_index, local_index] = torch.tensor(
                    local.features, dtype=torch.float32, device=dev
                )
                local_mask[0, action_index, local_index] = True
                for tensor, value in zip(
                    local_ids,
                    (
                        local.type_id,
                        local.role_id,
                        local.owner_id,
                        local.entity_id,
                        local.entity_aux_id,
                        local.zone_id,
                        local.order_id,
                    ),
                    strict=True,
                ):
                    tensor[0, action_index, local_index] = value

        return GroundedCandidateBatch(
            world=WorldTokenBatch(
                features=world_features,
                mask=world_mask,
                type_ids=world_ids[0],
                role_ids=world_ids[1],
                owner_ids=world_ids[2],
                entity_ids=world_ids[3],
                entity_aux_ids=world_ids[4],
                zone_ids=world_ids[5],
                order_ids=world_ids[6],
            ),
            candidates=CandidateTokenBatch(
                features=candidate_features,
                type_ids=candidate_ids[0],
                role_ids=candidate_ids[1],
                owner_ids=candidate_ids[2],
                entity_ids=candidate_ids[3],
                entity_aux_ids=candidate_ids[4],
                zone_ids=candidate_ids[5],
                target_owner_ids=candidate_ids[6],
                target_entity_ids=candidate_ids[7],
                target_entity_aux_ids=candidate_ids[8],
                local_features=local_features,
                local_mask=local_mask,
                local_type_ids=local_ids[0],
                local_role_ids=local_ids[1],
                local_owner_ids=local_ids[2],
                local_entity_ids=local_ids[3],
                local_entity_aux_ids=local_ids[4],
                local_zone_ids=local_ids[5],
                local_order_ids=local_ids[6],
                action_mask=action_mask,
            ),
            domain_ids=torch.tensor([domain_id], dtype=torch.long, device=dev),
        )


__all__ = [
    "GROUNDING_ENCODING_VERSION",
    "MODEL_ACTION_KIND_VOCABULARY",
    "ActionReference",
    "EncodedDecision",
    "GroundedEncodingConfig",
    "GroundedObservationEncoder",
    "grounding_encoding_identity",
]
