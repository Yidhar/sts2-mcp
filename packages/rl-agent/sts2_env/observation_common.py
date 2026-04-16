"""Shared dense observation primitives for STS2 RL training.

This module is the compatibility-neutral home for the legacy dense feature
encoder and the feature-extraction helpers that are still reused by the newer
token-world observation stack.

Observation structure is intentionally split into:
  - shared global context
  - combat entities
  - build/deck entities
  - candidate actions

The bridge is the canonical source for semantic text. Python only embeds
bridge-provided canonical_text and compact decision text.
"""

from __future__ import annotations

import math
from typing import Any, Callable
import numpy as np

from content_registry import (
    build_enemy_intent_semantic_text,
    build_live_card_semantic_text,
    build_live_enemy_semantic_text,
    build_live_potion_semantic_text,
    build_live_relic_semantic_text,
    get_card_metadata,
    get_relic_metadata,
)

from .run_memory import OBJECTIVE_CONTEXT_DIM as _OBJECTIVE_CONTEXT_DIM, RUN_MEMORY_DIM as _RUN_MEMORY_DIM
from .semantic_action import (
    SEMANTIC_ACTION_DIM,
    encode_semantic_action_numeric,
    semantic_action_signature,
    semantic_action_text as build_semantic_action_text,
)
from .text_encoder import TEXT_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASES = [
    "combat", "map", "reward", "card_reward", "event", "event_crystal_sphere",
    "rest_site", "deck_upgrade", "card_selection", "shop", "treasure",
    "actions", "settling", "terminal",
    "startup_main_menu", "startup_run_mode", "startup_character_select",
]
PHASE_TO_IDX = {phase: index for index, phase in enumerate(PHASES)}
NUM_PHASES = len(PHASES)

DECISION_DOMAINS = ["combat", "build", "route"]
DOMAIN_TO_IDX = {domain: index for index, domain in enumerate(DECISION_DOMAINS)}
NUM_DOMAINS = len(DECISION_DOMAINS)

ROOM_TYPES = ["Monster", "Elite", "Boss", "Event", "Rest", "Merchant", "Treasure"]
ROOM_TYPE_TO_ORD = {room_type: index + 1 for index, room_type in enumerate(ROOM_TYPES)}
NUM_ROOM_TYPES = len(ROOM_TYPES) + 1

MAX_HAND = 12
MAX_DECK = 40
MAX_ENEMIES = 5
MAX_RELICS = 20
MAX_POTIONS = 5
# Single-source action cap for both env wrappers and the policy head.
# 80 is large enough for dense combat turns without exploding tensor size.
MAX_ACTIONS = 80
MAX_ROUTE_NODES = 24

SCALAR_DIM = 61
CARD_FEAT_DIM = 41
DECK_FEAT_DIM = 41
ENEMY_FEAT_DIM = 27
POWER_DIM = 24
RELIC_SIGNAL_DIM = 12
ACTION_FEAT_DIM = 49
RUN_MEMORY_DIM = _RUN_MEMORY_DIM
OBJECTIVE_DIM = _OBJECTIVE_CONTEXT_DIM
SEM_ACTION_FEAT_DIM = SEMANTIC_ACTION_DIM

_ACTION_KINDS = [
    "play_card", "use_potion", "discard_potion", "combat",
    "reward", "card_reward", "event_option", "map",
    "rest_site", "deck_upgrade", "card_selection", "combat_select_card", "combat_select", "shop",
    "treasure_relic", "treasure", "character_select",
    "run_mode_selection", "main_menu", "proceed",
]
_KIND_TO_ORD = {kind: index + 1 for index, kind in enumerate(_ACTION_KINDS)}
_NUM_KINDS = len(_ACTION_KINDS) + 1
ACTION_KIND_TO_ORD = dict(_KIND_TO_ORD)
NUM_ACTION_KINDS = _NUM_KINDS

_MAP_POINT_TYPES = ["Monster", "Elite", "Boss", "Event", "QuestionMark", "RestSite", "Shop", "Treasure"]
_PT_TO_ORD = {point_type: index + 1 for index, point_type in enumerate(_MAP_POINT_TYPES)}
_NUM_PT = len(_MAP_POINT_TYPES) + 1

_ROUTE_POINT_TYPES = ["Monster", "Elite", "Boss", "Event", "QuestionMark", "RestSite", "Shop", "Treasure"]
_ROUTE_PT_TO_IDX = {point_type: index for index, point_type in enumerate(_ROUTE_POINT_TYPES)}
_NUM_ROUTE_PT = len(_ROUTE_POINT_TYPES)

ROUTE_SUMMARY_DIM = 20
ROUTE_NODE_FEAT_DIM = _NUM_ROUTE_PT + 5

_PLAYER_POWER_FEATURES = [
    "strength", "dexterity", "weak", "vulnerable", "frail",
    "plating", "ritual", "metallicize", "barricade", "rage",
    "vigor", "intangible", "thorns", "regen", "artifact",
    "poison", "calamity", "buffer", "entangled", "lockon",
]
_ENEMY_POWER_FEATURES = [
    "vulnerable", "weak", "strength", "dexterity", "artifact",
    "poison", "calamity", "thorns", "intangible", "regen",
    "plating", "metallicize", "ritual",
]
_POWER_FEATURE_TO_IDX = {power: index for index, power in enumerate(_PLAYER_POWER_FEATURES)}
_ENEMY_POWER_TO_OFFSET = {power: index for index, power in enumerate(_ENEMY_POWER_FEATURES)}
_POWER_ALIASES = {
    "strength": ("strength", "力量"),
    "dexterity": ("dexterity", "敏捷"),
    "weak": ("weak", "虚弱"),
    "vulnerable": ("vulnerable", "易伤"),
    "frail": ("frail", "脆弱"),
    "plating": ("plating", "plated armor", "覆甲", "护甲"),
    "ritual": ("ritual", "仪式"),
    "metallicize": ("metallicize", "金属化"),
    "barricade": ("barricade", "壁垒"),
    "rage": ("rage", "愤怒"),
    "vigor": ("vigor", "活力"),
    "intangible": ("intangible", "无实体"),
    "thorns": ("thorns", "荆棘"),
    "regen": ("regen", "再生", "恢复", "回复"),
    "artifact": ("artifact", "人工制品"),
    "poison": ("poison", "毒", "毒素"),
    "calamity": ("calamity", "灾厄"),
    "buffer": ("buffer", "缓冲"),
    "entangled": ("entangled", "缠绕"),
    "lockon": ("lock on", "lockon", "锁定"),
}
_RELIC_SIGNAL_FEATURES = [
    "energy", "draw", "strength", "dexterity", "vigor", "thorns",
    "intangible", "artifact", "poison", "calamity", "defense", "regen",
]
_RELIC_SIGNAL_ALIASES = {
    "energy": ("energy", "能量"),
    "draw": ("draw", "抽"),
    "strength": _POWER_ALIASES["strength"],
    "dexterity": _POWER_ALIASES["dexterity"],
    "vigor": _POWER_ALIASES["vigor"],
    "thorns": _POWER_ALIASES["thorns"],
    "intangible": _POWER_ALIASES["intangible"],
    "artifact": _POWER_ALIASES["artifact"],
    "poison": _POWER_ALIASES["poison"],
    "calamity": _POWER_ALIASES["calamity"],
    "defense": ("block", "格挡", "防御", "护甲", "壁垒", "覆甲", "金属化"),
    "regen": _POWER_ALIASES["regen"],
}


def _float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _bool(val):
    return 1.0 if val else 0.0


def _metric(source, key, default=0.0):
    if not isinstance(source, dict):
        return default
    return _float(source.get(key), default)


_LOG1P_200 = math.log1p(200.0)
_LOG1P_1200 = math.log1p(1200.0)
_LOG1P_500 = math.log1p(500.0)
_LOG1P_200_F = math.log1p(200.0)
_LOG1P_100 = math.log1p(100.0)
_LOG1P_20 = math.log1p(20.0)

_CARD_KEYWORDS = {"exhaust": 0, "ethereal": 1, "retain": 2, "innate": 3}
_RARITY_MAP = {"Common": 0.33, "Uncommon": 0.67, "Rare": 1.0}


def _log_norm(value: float, anchor: float) -> float:
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / anchor, 1.0)


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _signed_log_norm(value: float, anchor: float) -> float:
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    return sign * _log_norm(abs(value), anchor)


def _nested_value(source: dict | None, *keys: str):
    current = source
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _preview_metric(source: dict | None, key: str, default: float = 0.0) -> float:
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


def _normalized_cost_for_efficiency(cost: float) -> float:
    if cost <= 0:
        return 1.0
    return cost


def _runtime_spend_cost(card: dict | None) -> float:
    if not isinstance(card, dict):
        return 0.0

    if card.get("x_cost"):
        x_cost_value = _preview_metric(card, "x_cost_value")
        if x_cost_value > 0:
            return x_cost_value

    return _float(card.get("cost"))


def _normalize_power_amount(amount: float) -> float:
    return _log_norm(abs(amount), _LOG1P_20)


def _get_card_static_metadata(card: dict | None) -> dict | None:
    if not isinstance(card, dict):
        return None
    card_id = str(card.get("id") or "").strip()
    if not card_id:
        return None
    metadata = get_card_metadata(card_id)
    return metadata if isinstance(metadata, dict) else None


def _canonicalize_power_title(title: str) -> str | None:
    lower = title.lower()
    for canonical, aliases in _POWER_ALIASES.items():
        for alias in aliases:
            if alias and alias in lower:
                return canonical
    return None


def _power_amount_map(powers: list | None) -> dict[str, float]:
    amounts: dict[str, float] = {}
    if not isinstance(powers, list):
        return amounts

    for power in powers:
        if not isinstance(power, dict):
            continue
        title = str(power.get("title") or "").strip()
        if not title:
            continue
        canonical = _canonicalize_power_title(title)
        if canonical is None:
            continue
        amount = _float(power.get("amount"), 1.0)
        if amount == 0.0:
            amount = 1.0
        amounts[canonical] = amounts.get(canonical, 0.0) + amount
    return amounts


def _get_relic_text_candidates(relic: dict | str | None) -> list[str]:
    candidates: list[str] = []
    if isinstance(relic, dict):
        candidates.extend(
            str(relic.get(key) or "").strip()
            for key in ("title", "summary", "description", "canonical_text")
            if str(relic.get(key) or "").strip()
        )
        relic_id = str(relic.get("id") or "").strip()
        metadata = get_relic_metadata(relic_id)
        if isinstance(metadata, dict):
            candidates.extend(
                str(metadata.get(key) or "").strip()
                for key in ("title", "summary", "description", "canonical_text")
                if str(metadata.get(key) or "").strip()
            )
        semantic_text = build_live_relic_semantic_text(relic)
        if semantic_text:
            candidates.append(semantic_text)
    elif isinstance(relic, str):
        text = relic.strip()
        if text:
            candidates.append(text)
    return candidates


def _encode_relic_signals(vector: np.ndarray, relics: list | None) -> None:
    if not isinstance(relics, list):
        return

    counts = np.zeros(RELIC_SIGNAL_DIM, dtype=np.float32)
    for relic in relics:
        joined = " | ".join(_get_relic_text_candidates(relic)).lower()
        if not joined:
            continue
        for idx, feature in enumerate(_RELIC_SIGNAL_FEATURES):
            aliases = _RELIC_SIGNAL_ALIASES.get(feature, ())
            if any(alias in joined for alias in aliases):
                counts[idx] += 1.0

    if counts.any():
        vector[:] = np.clip(counts / 3.0, 0.0, 1.0)


def _iter_card_keyword_candidates(card: dict) -> list[str]:
    candidates: list[str] = []
    keywords = card.get("keywords")
    if isinstance(keywords, list):
        candidates.extend(str(keyword or "").strip() for keyword in keywords)

    metadata = _get_card_static_metadata(card)
    if isinstance(metadata, dict):
        static_keywords = metadata.get("keywords")
        if isinstance(static_keywords, list):
            candidates.extend(str(keyword or "").strip() for keyword in static_keywords)
        semantic_tags = metadata.get("semantic_tags")
        if isinstance(semantic_tags, list):
            candidates.extend(str(keyword or "").strip() for keyword in semantic_tags)

    return candidates


def _get_card_keywords(card: dict) -> tuple[list[bool], float]:
    flags = [False, False, False, False]
    rarity_val = 0.0

    for keyword in _iter_card_keyword_candidates(card):
        kw_lower = keyword.lower()
        for key, index in _CARD_KEYWORDS.items():
            if key in kw_lower:
                flags[index] = True

    rarity = str(card.get("rarity") or "").strip()
    if not rarity:
        metadata = _get_card_static_metadata(card)
        if isinstance(metadata, dict):
            rarity = str(metadata.get("rarity") or "").strip()
    if rarity:
        rarity_val = _RARITY_MAP.get(rarity, 0.0)

    return flags, rarity_val


def _infer_upgrade_level(card: dict) -> int:
    level = int(_float(card.get("upgrade_level"), -1))
    if level >= 0:
        return level

    title = str(card.get("title") or "").strip()
    if not title:
        return 0

    plus_count = 0
    for ch in reversed(title):
        if ch == "+":
            plus_count += 1
        else:
            break
    return plus_count


def _get_card_level_metadata(card: dict) -> dict | None:
    metadata = _get_card_static_metadata(card)
    if not isinstance(metadata, dict):
        return None

    level_payloads = metadata.get("upgrade_level_texts")
    if not isinstance(level_payloads, dict):
        return metadata

    target_level = _infer_upgrade_level(card)
    if target_level <= 0:
        payload = level_payloads.get("0")
        return payload if isinstance(payload, dict) else metadata

    if str(target_level) in level_payloads and isinstance(level_payloads[str(target_level)], dict):
        return level_payloads[str(target_level)]

    fallback_levels: list[int] = []
    for key in level_payloads.keys():
        try:
            fallback_levels.append(int(key))
        except (TypeError, ValueError):
            continue
    if not fallback_levels:
        return metadata

    fallback = max((lvl for lvl in fallback_levels if lvl <= target_level), default=min(fallback_levels))
    payload = level_payloads.get(str(fallback))
    return payload if isinstance(payload, dict) else metadata


def _static_signal_metric(card: dict, key: str, default: float = 0.0) -> float:
    level_metadata = _get_card_level_metadata(card)
    if not isinstance(level_metadata, dict):
        return default

    signals = level_metadata.get("semantic_signals")
    if isinstance(signals, dict):
        direct = signals.get(key)
        if direct is not None:
            return _float(direct, default)

    direct = level_metadata.get(key)
    if direct is not None:
        return _float(direct, default)

    key_aliases = {
        "strength": ("strengthGain",),
        "dexterity": ("dexterityGain",),
        "energy": ("energyGain", "energy_cost"),
    }
    for alias in key_aliases.get(key, ()):
        alias_value = level_metadata.get(alias)
        if alias_value is not None:
            return _float(alias_value, default)
        if isinstance(signals, dict) and signals.get(alias) is not None:
            return _float(signals.get(alias), default)

    return default


def _get_card_extra_metrics(card: dict) -> tuple[float, float, float, float]:
    strength = _preview_metric(card, "strength") or _metric(card, "strengthGain")
    dexterity = _preview_metric(card, "dexterity") or _metric(card, "dexterityGain")
    energy = _preview_metric(card, "energy") or _metric(card, "energyGain") or _metric(card, "energy")
    hits = _preview_metric(card, "hits", default=1.0) or _metric(card, "hits", default=1.0)

    signals = card.get("semantic_signals")
    if not isinstance(signals, dict):
        metadata = _get_card_static_metadata(card)
        if isinstance(metadata, dict) and isinstance(metadata.get("semantic_signals"), dict):
            signals = metadata.get("semantic_signals")

    if isinstance(signals, dict):
        if not strength:
            strength = _float(signals.get("strengthGain"))
        if not dexterity:
            dexterity = _float(signals.get("dexterityGain"))
        if not energy:
            energy = _float(signals.get("energyGain"))
        if hits <= 1.0:
            hits = _float(signals.get("hits"), 1.0)

    return strength, dexterity, energy, hits


def _build_card_preview_bundle(card: dict) -> dict[str, float]:
    spend_cost = _runtime_spend_cost(card)
    hits = _get_card_extra_metrics(card)[3]

    base_damage = _static_signal_metric(card, "damage")
    base_block = _static_signal_metric(card, "block")
    preview_damage = _preview_metric(card, "damage")
    preview_block = _preview_metric(card, "block")
    preview_damage_per_hit = _preview_metric(card, "damage_per_hit")
    if preview_damage_per_hit <= 0 and preview_damage > 0:
        preview_damage_per_hit = preview_damage / max(hits, 1.0)

    preview_damage_per_energy = (
        preview_damage / _normalized_cost_for_efficiency(spend_cost)
        if preview_damage > 0 else 0.0
    )
    preview_block_per_energy = (
        preview_block / _normalized_cost_for_efficiency(spend_cost)
        if preview_block > 0 else 0.0
    )

    if base_damage <= 0 and preview_damage > 0:
        base_damage = preview_damage
    if base_block <= 0 and preview_block > 0:
        base_block = preview_block

    return {
        "base_damage": base_damage,
        "base_block": base_block,
        "preview_damage": preview_damage,
        "preview_block": preview_block,
        "preview_damage_per_hit": preview_damage_per_hit,
        "preview_damage_per_energy": preview_damage_per_energy,
        "preview_block_per_energy": preview_block_per_energy,
        "preview_damage_delta": preview_damage - base_damage,
        "preview_block_delta": preview_block - base_block,
    }


def _infer_enemy_intent_flags(intent: dict, total_damage: float) -> tuple[float, float, float, float]:
    text_parts: list[str] = []
    for key in ("type", "id", "intent_type", "intent_class", "state_id", "title", "text", "description", "label"):
        value = intent.get(key)
        if value:
            text_parts.append(str(value).lower())

    nested_intents = intent.get("intents")
    if isinstance(nested_intents, list):
        for nested in nested_intents:
            if not isinstance(nested, dict):
                continue
            for key in ("intent_type", "intent_class", "title", "label", "description"):
                value = nested.get(key)
                if value:
                    text_parts.append(str(value).lower())

    intent_text = " ".join(text_parts)
    is_attack = 1.0 if total_damage > 0 or any(
        keyword in intent_text for keyword in ("attack", "deathblow", "damage", "hit")
    ) else 0.0
    is_defend = 1.0 if any(keyword in intent_text for keyword in ("defend", "block", "shield", "guard", "armor")) else 0.0
    is_buff = 1.0 if any(keyword in intent_text for keyword in ("buff", "strength", "ritual", "enrage", "metallicize", "regen")) else 0.0
    is_debuff = 1.0 if any(keyword in intent_text for keyword in ("debuff", "weak", "vulnerable", "frail", "poison", "calamity")) else 0.0
    return is_attack, is_defend, is_buff, is_debuff


class DenseObservationEncoder:
    """Encode bridge observation + legal actions into the shared dense schema."""

    def __init__(self, use_text: bool = True, text_device: str = "cpu"):
        self.use_text = use_text
        self.text_device = str(text_device or "cpu").strip() or "cpu"
        self._encoder = None
        self._text_registry_active = False
        self._text_registry_order: list[str] = []
        self._text_registry_assignments: dict[str, list[Callable[[np.ndarray], None]]] = {}

    def _get_encoder(self):
        if self._encoder is None and self.use_text:
            from .text_encoder import get_text_encoder

            self._encoder = get_text_encoder(device=self.text_device).ensure_ready()
        return self._encoder

    def _begin_text_registry(self) -> None:
        self._text_registry_active = True
        self._text_registry_order = []
        self._text_registry_assignments = {}

    def _clear_text_registry(self) -> None:
        self._text_registry_active = False
        self._text_registry_order = []
        self._text_registry_assignments = {}

    def _register_text_assignment(
        self,
        text: str | None,
        assign: Callable[[np.ndarray], None],
        *,
        postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        raw_text = str(text or "").strip()

        def _zero_embedding() -> np.ndarray:
            base = np.zeros(TEXT_DIM, dtype=np.float32)
            if postprocess is not None:
                return np.asarray(postprocess(base), dtype=np.float32)
            return base

        if not self.use_text or not raw_text:
            assign(_zero_embedding())
            return

        if not self._text_registry_active:
            embedding = np.asarray(self._get_encoder().encode(raw_text), dtype=np.float32)
            if postprocess is not None:
                embedding = np.asarray(postprocess(embedding), dtype=np.float32)
            assign(embedding)
            return

        if raw_text not in self._text_registry_assignments:
            self._text_registry_order.append(raw_text)
            self._text_registry_assignments[raw_text] = []

        def _wrapped(embedding: np.ndarray) -> None:
            final_embedding = np.asarray(postprocess(embedding), dtype=np.float32) if postprocess is not None else np.asarray(embedding, dtype=np.float32)
            assign(final_embedding)

        self._text_registry_assignments[raw_text].append(_wrapped)

    def _resolve_text_registry(self) -> None:
        try:
            if not self._text_registry_active:
                return
            if not self._text_registry_order:
                return
            embeddings = np.asarray(self._get_encoder().encode_batch(self._text_registry_order), dtype=np.float32)
            for index, text in enumerate(self._text_registry_order):
                embedding = embeddings[index]
                for assign in self._text_registry_assignments.get(text, ()):
                    assign(embedding)
        finally:
            self._clear_text_registry()

    def _normalize_planner_context(
        self,
        obs: dict | None,
        legal_actions: list | None,
        planner_context: dict | None,
    ) -> dict[str, Any]:
        context = dict(planner_context) if isinstance(planner_context, dict) else {}
        run_memory_vector = np.asarray(context.get("run_memory_vector", np.zeros(RUN_MEMORY_DIM, dtype=np.float32)), dtype=np.float32)
        if run_memory_vector.shape != (RUN_MEMORY_DIM,):
            run_memory_vector = np.zeros(RUN_MEMORY_DIM, dtype=np.float32)
        objective_context_vector = np.asarray(context.get("objective_context_vector", np.zeros(OBJECTIVE_DIM, dtype=np.float32)), dtype=np.float32)
        if objective_context_vector.shape != (OBJECTIVE_DIM,):
            objective_context_vector = np.zeros(OBJECTIVE_DIM, dtype=np.float32)

        semantic_actions = context.get("semantic_actions")
        if not isinstance(semantic_actions, list):
            semantic_actions = [
                semantic_action_signature(action)
                for action in (legal_actions or [])
                if isinstance(action, dict)
            ]

        run_memory_text = str(context.get("run_memory_text") or "").strip()
        objective_text = str(context.get("objective_text") or "").strip()
        episode_mode = str(context.get("episode_mode") or "").strip()
        if not episode_mode:
            episode_mode = "combat_sandbox" if run_memory_vector.shape == (RUN_MEMORY_DIM,) and float(run_memory_vector[42]) > 0.5 else "full_run"
        potion_mechanics_available = context.get("potion_mechanics_available")
        if potion_mechanics_available is None:
            potion_mechanics_available = 0.0 if episode_mode == "combat_sandbox" else 1.0
        context["run_memory_vector"] = run_memory_vector
        context["objective_context_vector"] = objective_context_vector
        context["semantic_actions"] = semantic_actions
        context["run_memory_text"] = run_memory_text
        context["objective_text"] = objective_text
        context["episode_mode"] = episode_mode
        context["potion_mechanics_available"] = _clip01(_float(potion_mechanics_available, 1.0))
        return context

    @property
    def obs_space(self):
        from gymnasium import spaces

        inf = np.inf
        return spaces.Dict(
            {
                "scalars": spaces.Box(0, 1, (SCALAR_DIM,), dtype=np.float32),
                "decision_domain": spaces.Box(0, 1, (NUM_DOMAINS,), dtype=np.float32),
                "hand": spaces.Box(-inf, inf, (MAX_HAND, CARD_FEAT_DIM), dtype=np.float32),
                "hand_text": spaces.Box(-inf, inf, (MAX_HAND, TEXT_DIM), dtype=np.float32),
                "hand_mask": spaces.Box(0, 1, (MAX_HAND,), dtype=np.float32),
                "deck": spaces.Box(-inf, inf, (MAX_DECK, DECK_FEAT_DIM), dtype=np.float32),
                "deck_text": spaces.Box(-inf, inf, (MAX_DECK, TEXT_DIM), dtype=np.float32),
                "deck_mask": spaces.Box(0, 1, (MAX_DECK,), dtype=np.float32),
                "enemies": spaces.Box(-inf, inf, (MAX_ENEMIES, ENEMY_FEAT_DIM), dtype=np.float32),
                "enemy_text": spaces.Box(-inf, inf, (MAX_ENEMIES, TEXT_DIM), dtype=np.float32),
                "enemy_mask": spaces.Box(0, 1, (MAX_ENEMIES,), dtype=np.float32),
                "player_powers": spaces.Box(0, 1, (POWER_DIM,), dtype=np.float32),
                "relic_signals": spaces.Box(0, 1, (RELIC_SIGNAL_DIM,), dtype=np.float32),
                "run_memory": spaces.Box(0, 1, (RUN_MEMORY_DIM,), dtype=np.float32),
                "objective_context": spaces.Box(0, 1, (OBJECTIVE_DIM,), dtype=np.float32),
                "relics": spaces.Box(-inf, inf, (MAX_RELICS, TEXT_DIM), dtype=np.float32),
                "relic_mask": spaces.Box(0, 1, (MAX_RELICS,), dtype=np.float32),
                "potions": spaces.Box(-inf, inf, (MAX_POTIONS, TEXT_DIM), dtype=np.float32),
                "potion_mask": spaces.Box(0, 1, (MAX_POTIONS,), dtype=np.float32),
                "context_text": spaces.Box(-inf, inf, (TEXT_DIM,), dtype=np.float32),
                "actions": spaces.Box(-inf, inf, (MAX_ACTIONS, ACTION_FEAT_DIM), dtype=np.float32),
                "action_text": spaces.Box(-inf, inf, (MAX_ACTIONS, TEXT_DIM), dtype=np.float32),
                "semantic_actions": spaces.Box(-inf, inf, (MAX_ACTIONS, SEM_ACTION_FEAT_DIM), dtype=np.float32),
                "semantic_action_text": spaces.Box(-inf, inf, (MAX_ACTIONS, TEXT_DIM), dtype=np.float32),
                "route_summary": spaces.Box(-inf, inf, (MAX_ACTIONS, ROUTE_SUMMARY_DIM), dtype=np.float32),
                "route_nodes": spaces.Box(-inf, inf, (MAX_ACTIONS, MAX_ROUTE_NODES, ROUTE_NODE_FEAT_DIM), dtype=np.float32),
                "route_node_mask": spaces.Box(0, 1, (MAX_ACTIONS, MAX_ROUTE_NODES), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, (MAX_ACTIONS,), dtype=np.float32),
            }
        )

    def encode(
        self,
        obs: dict | None,
        legal_actions: list | None = None,
        planner_context: dict | None = None,
        *,
        _resolve_text: bool = True,
    ) -> dict[str, np.ndarray]:
        self._begin_text_registry()
        scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
        decision_domain = np.zeros(NUM_DOMAINS, dtype=np.float32)
        hand = np.zeros((MAX_HAND, CARD_FEAT_DIM), dtype=np.float32)
        hand_text = np.zeros((MAX_HAND, TEXT_DIM), dtype=np.float32)
        hand_mask = np.zeros(MAX_HAND, dtype=np.float32)
        deck = np.zeros((MAX_DECK, DECK_FEAT_DIM), dtype=np.float32)
        deck_text = np.zeros((MAX_DECK, TEXT_DIM), dtype=np.float32)
        deck_mask = np.zeros(MAX_DECK, dtype=np.float32)
        enemies = np.zeros((MAX_ENEMIES, ENEMY_FEAT_DIM), dtype=np.float32)
        enemy_text = np.zeros((MAX_ENEMIES, TEXT_DIM), dtype=np.float32)
        enemy_mask = np.zeros(MAX_ENEMIES, dtype=np.float32)
        player_powers = np.zeros(POWER_DIM, dtype=np.float32)
        relic_signals = np.zeros(RELIC_SIGNAL_DIM, dtype=np.float32)
        run_memory = np.zeros(RUN_MEMORY_DIM, dtype=np.float32)
        objective_context = np.zeros(OBJECTIVE_DIM, dtype=np.float32)
        relics = np.zeros((MAX_RELICS, TEXT_DIM), dtype=np.float32)
        relic_mask = np.zeros(MAX_RELICS, dtype=np.float32)
        potions = np.zeros((MAX_POTIONS, TEXT_DIM), dtype=np.float32)
        potion_mask = np.zeros(MAX_POTIONS, dtype=np.float32)
        context_text = np.zeros(TEXT_DIM, dtype=np.float32)
        actions = np.zeros((MAX_ACTIONS, ACTION_FEAT_DIM), dtype=np.float32)
        action_text = np.zeros((MAX_ACTIONS, TEXT_DIM), dtype=np.float32)
        semantic_actions = np.zeros((MAX_ACTIONS, SEM_ACTION_FEAT_DIM), dtype=np.float32)
        semantic_action_text = np.zeros((MAX_ACTIONS, TEXT_DIM), dtype=np.float32)
        route_summary = np.zeros((MAX_ACTIONS, ROUTE_SUMMARY_DIM), dtype=np.float32)
        route_nodes = np.zeros((MAX_ACTIONS, MAX_ROUTE_NODES, ROUTE_NODE_FEAT_DIM), dtype=np.float32)
        route_node_mask = np.zeros((MAX_ACTIONS, MAX_ROUTE_NODES), dtype=np.float32)
        action_mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
        planner_context = self._normalize_planner_context(obs, legal_actions, planner_context)

        try:
            if obs:
                self._enc_scalars(scalars, obs, legal_actions or [], planner_context)
                self._enc_decision_domain(decision_domain, obs)
                self._enc_hand(hand, hand_text, hand_mask, obs)
                self._enc_deck(deck, deck_text, deck_mask, obs)
                self._enc_enemies(enemies, enemy_text, enemy_mask, obs)
                self._enc_powers(player_powers, obs)
                self._enc_relic_signals(relic_signals, obs)
                self._enc_relics(relics, relic_mask, obs)
                self._enc_potions(potions, potion_mask, obs)
                run_memory[:] = planner_context["run_memory_vector"]
                objective_context[:] = planner_context["objective_context_vector"]
                self._enc_context(context_text, obs, planner_context)

            if legal_actions:
                self._enc_actions(
                    actions,
                    action_text,
                    semantic_actions,
                    semantic_action_text,
                    route_summary,
                    route_nodes,
                    route_node_mask,
                    action_mask,
                    legal_actions,
                    planner_context,
                )

            if _resolve_text:
                self._resolve_text_registry()

            return {
                "scalars": scalars,
                "decision_domain": decision_domain,
                "hand": hand,
                "hand_text": hand_text,
                "hand_mask": hand_mask,
                "deck": deck,
                "deck_text": deck_text,
                "deck_mask": deck_mask,
                "enemies": enemies,
                "enemy_text": enemy_text,
                "enemy_mask": enemy_mask,
                "player_powers": player_powers,
                "relic_signals": relic_signals,
                "run_memory": run_memory,
                "objective_context": objective_context,
                "relics": relics,
                "relic_mask": relic_mask,
                "potions": potions,
                "potion_mask": potion_mask,
                "context_text": context_text,
                "actions": actions,
                "action_text": action_text,
                "semantic_actions": semantic_actions,
                "semantic_action_text": semantic_action_text,
                "route_summary": route_summary,
                "route_nodes": route_nodes,
                "route_node_mask": route_node_mask,
                "action_mask": action_mask,
            }
        except Exception:
            self._clear_text_registry()
            raise

    def _enc_scalars(
        self,
        vector: np.ndarray,
        obs: dict,
        legal_actions: list,
        planner_context: dict[str, Any] | None = None,
    ) -> None:
        offset = 0

        phase = obs.get("phase", "")
        phase_idx = PHASE_TO_IDX.get(phase, -1)
        if 0 <= phase_idx < NUM_PHASES:
            vector[offset + phase_idx] = 1.0
        offset += NUM_PHASES

        run = obs.get("run") or {}
        vector[offset] = _bool(run.get("active"))
        vector[offset + 1] = _bool(run.get("game_over"))
        vector[offset + 2] = min(self._parse_act(run.get("act_id")) / 4.0, 1.0)
        vector[offset + 3] = min(_float(run.get("act_floor")) / 20.0, 1.0)
        vector[offset + 4] = min(_float(run.get("floor")) / 48.0, 1.0)
        vector[offset + 5] = ROOM_TYPE_TO_ORD.get(run.get("room_type", ""), 0) / NUM_ROOM_TYPES
        offset += 6

        player = obs.get("player") or {}
        combat = obs.get("combat") or {}
        hp = _float(player.get("hp"))
        max_hp = _float(player.get("max_hp"))
        vector[offset] = min(hp / max_hp, 1.0) if max_hp > 0 else 0.0
        vector[offset + 1] = min(hp / 100.0, 1.0)
        vector[offset + 2] = min(max_hp / 100.0, 1.0)
        vector[offset + 3] = _log_norm(_float(player.get("block")), _LOG1P_200)
        vector[offset + 4] = _log_norm(_float(player.get("gold")), _LOG1P_500)
        energy = _float(combat.get("energy"))
        max_energy = _float(combat.get("max_energy"))
        vector[offset + 5] = min(energy / max_energy, 1.0) if max_energy > 0 else 0.0
        vector[offset + 6] = min(energy / 10.0, 1.0)
        vector[offset + 7] = min(_float(combat.get("stars")) / 10.0, 1.0)
        offset += 8

        if combat:
            vector[offset] = 1.0
            vector[offset + 1] = min(_float(combat.get("round")) / 20.0, 1.0)
            vector[offset + 2] = _bool(combat.get("play_phase"))
            vector[offset + 3] = _bool(combat.get("can_act"))
            hand = combat.get("hand") or []
            vector[offset + 4] = min(len(hand) / 10.0, 1.0)
            vector[offset + 5] = min(_float(combat.get("draw")) / 40.0, 1.0)
            vector[offset + 6] = min(_float(combat.get("discard")) / 40.0, 1.0)
            vector[offset + 7] = min(_float(combat.get("exhaust")) / 20.0, 1.0)
        offset += 8

        decision = obs.get("decision") or {}
        if isinstance(decision, dict):
            vector[offset] = min(_float(decision.get("option_count")) / 10.0, 1.0)
            vector[offset + 1] = _bool(decision.get("can_skip"))
            vector[offset + 2] = min(_float(decision.get("selected_count")) / 5.0, 1.0)
            vector[offset + 3] = min(_float(decision.get("min_select")) / 5.0, 1.0)
            vector[offset + 4] = min(_float(decision.get("max_select")) / 5.0, 1.0)
            vector[offset + 5] = _bool(decision.get("is_open"))
            vector[offset + 6] = min(_float(decision.get("travelable_count")) / 10.0, 1.0)
            vector[offset + 7] = _bool(decision.get("can_proceed") or decision.get("proceed_only"))
            vector[offset + 8] = min(_float(decision.get("reward_count")) / 10.0, 1.0)
            vector[offset + 9] = min(_float(decision.get("item_count")) / 20.0, 1.0)
        offset += 10

        relics = player.get("relics") or []
        potions = player.get("potions") or []
        relic_count = len(relics) if isinstance(relics, list) else 0
        potion_mechanics_available = _clip01(_float((planner_context or {}).get("potion_mechanics_available"), 1.0))
        potion_count = sum(
            1
            for potion in (potions if isinstance(potions, list) else [])
            if isinstance(potion, (str, dict))
            and (potion if isinstance(potion, str) else potion.get("title", "")) != "[empty]"
        )
        deck_cards = player.get("deck_cards")
        deck_size = len(deck_cards) if isinstance(deck_cards, list) else _float(player.get("deck"))
        vector[offset] = min(deck_size / 50.0, 1.0)
        vector[offset + 1] = min(relic_count / 20.0, 1.0)
        vector[offset + 2] = min(potion_count / 5.0, 1.0)
        vector[offset + 3] = min((len(potions) - potion_count) / 5.0, 1.0) if isinstance(potions, list) else 0.0
        vector[offset + 4] = min(len(potions) / 5.0, 1.0) if isinstance(potions, list) else 0.0
        # Explicitly distinguish "no potions carried" from "this environment
        # does not model potions at all" without changing the schema width.
        vector[offset + 5] = potion_mechanics_available
        offset += 6

        total_actions = 0
        combat_continue_actions = 0
        play_card_actions = 0
        zero_cost_play_actions = 0
        positive_preview_actions = 0
        has_end_turn = False

        for action in legal_actions:
            if not isinstance(action, dict):
                continue
            total_actions += 1
            action_id = action.get("action_id") or ""
            if action_id == "end_turn":
                has_end_turn = True
                continue

            kind = action.get("kind") or ""
            if kind not in ("play_card", "use_potion"):
                continue

            combat_continue_actions += 1
            source = action.get("card") if kind == "play_card" else action.get("potion")
            if kind == "play_card":
                play_card_actions += 1
                if isinstance(source, dict) and _float(source.get("cost")) == 0:
                    zero_cost_play_actions += 1

            if isinstance(source, dict) and (
                _preview_metric(source, "damage") > 0
                or _preview_metric(source, "block") > 0
                or _preview_metric(source, "draw") > 0
                or _preview_metric(source, "weak") > 0
                or _preview_metric(source, "vulnerable") > 0
                or _preview_metric(source, "heal") > 0
                or _preview_metric(source, "strength") > 0
                or _preview_metric(source, "dexterity") > 0
                or _preview_metric(source, "summon") > 0
            ):
                positive_preview_actions += 1

        vector[offset] = min(total_actions / 50.0, 1.0)
        vector[offset + 1] = min(combat_continue_actions / 20.0, 1.0)
        vector[offset + 2] = min(play_card_actions / 20.0, 1.0)
        vector[offset + 3] = min(zero_cost_play_actions / 10.0, 1.0)
        vector[offset + 4] = min(positive_preview_actions / 10.0, 1.0)
        vector[offset + 5] = _bool(has_end_turn)
        offset += 6

        assert offset == SCALAR_DIM

    def _enc_decision_domain(self, vector: np.ndarray, obs: dict) -> None:
        domain = self._resolve_domain(obs)
        index = DOMAIN_TO_IDX.get(domain, DOMAIN_TO_IDX["build"])
        vector[index] = 1.0

    def _enc_hand(self, hand: np.ndarray, hand_text: np.ndarray, hand_mask: np.ndarray, obs: dict) -> None:
        combat = obs.get("combat") or {}
        cards = combat.get("hand") or []
        self._enc_card_collection(cards, hand, hand_text, hand_mask)

    def _enc_deck(self, deck: np.ndarray, deck_text: np.ndarray, deck_mask: np.ndarray, obs: dict) -> None:
        player = obs.get("player") or {}
        cards = player.get("deck_cards") or []
        self._enc_card_collection(cards, deck, deck_text, deck_mask)

    def _enc_card_collection(
        self,
        cards: list,
        numeric: np.ndarray,
        text: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        max_items = numeric.shape[0]
        texts: list[str] = []
        text_slots: list[int] = []
        for index, card in enumerate(cards[:max_items]):
            if not isinstance(card, dict):
                continue
            mask[index] = 1.0
            row = numeric[index]
            preview = _build_card_preview_bundle(card)
            strength, dexterity, energy, hits = _get_card_extra_metrics(card)
            kw_flags, rarity_val = _get_card_keywords(card)

            row[0] = min(_float(card.get("cost")) / 5.0, 1.0)
            card_type = (card.get("type") or "").capitalize()
            row[1] = 1.0 if card_type == "Attack" else 0.0
            row[2] = 1.0 if card_type == "Skill" else 0.0
            row[3] = 1.0 if card_type == "Power" else 0.0
            row[4] = _bool(card.get("x_cost"))
            row[5] = min(_float(card.get("star")) / 5.0, 1.0) if card.get("star") is not None else 0.0
            row[6] = _bool(card.get("star_x"))
            target = (card.get("target") or "").lower()
            row[7] = 1.0 if "single" in target or "anyenemy" in target else 0.0
            row[8] = 1.0 if "all" in target else 0.0
            row[9] = 1.0 if "self" in target else 0.0
            row[10] = 1.0 if card_type == "Status" else 0.0
            row[11] = 1.0 if card_type == "Curse" else 0.0

            row[12] = _log_norm(preview["base_damage"], _LOG1P_200)
            row[13] = _log_norm(preview["base_block"], _LOG1P_200)
            row[14] = _log_norm(preview["preview_damage"], _LOG1P_200)
            row[15] = _log_norm(preview["preview_block"], _LOG1P_200)
            row[16] = min(_preview_metric(card, "draw") / 5.0, 1.0)
            row[17] = min(_preview_metric(card, "weak") / 5.0, 1.0)
            row[18] = min(_preview_metric(card, "vulnerable") / 5.0, 1.0)
            row[19] = _log_norm(_preview_metric(card, "heal"), _LOG1P_200)
            row[20] = _log_norm(_preview_metric(card, "hp_loss"), _LOG1P_200)
            row[21] = min(_preview_metric(card, "summon") / 5.0, 1.0)

            row[22] = min(strength / 10.0, 1.0)
            row[23] = min(dexterity / 10.0, 1.0)
            row[24] = min(energy / 5.0, 1.0)
            row[25] = min(hits / 10.0, 1.0)
            row[26] = 1.0 if kw_flags[0] else 0.0
            row[27] = 1.0 if kw_flags[1] else 0.0
            row[28] = 1.0 if kw_flags[2] else 0.0
            row[29] = 1.0 if kw_flags[3] else 0.0
            row[30] = rarity_val
            row[31] = _log_norm(preview["preview_damage_per_hit"], _LOG1P_100)
            row[32] = _log_norm(preview["preview_damage_per_energy"], _LOG1P_100)
            row[33] = _log_norm(preview["preview_block_per_energy"], _LOG1P_100)
            row[34] = _signed_log_norm(preview["preview_damage_delta"], _LOG1P_100)
            row[35] = _signed_log_norm(preview["preview_block_delta"], _LOG1P_100)

            build_aux = card.get("build_aux")
            if isinstance(build_aux, dict):
                option_total = max(_float(build_aux.get("option_total")), 1.0)
                row[36] = min(_float(build_aux.get("remove_rank")) / option_total, 1.0)
                row[37] = min(_float(build_aux.get("keep_rank")) / option_total, 1.0)
                row[38] = _clip01((_float(build_aux.get("starter_gap_after")) + 5.0) / 10.0)

            row[39] = min(_infer_upgrade_level(card) / 3.0, 1.0)
            row[40] = min(_preview_metric(card, "x_cost_value") / 10.0, 1.0)
            text_value = self._build_live_card_text(card)
            if text_value and self.use_text:
                texts.append(text_value)
                text_slots.append(index)

        for slot, text_value in zip(text_slots, texts):
            self._register_text_assignment(
                text_value,
                lambda embedding, slot=slot, text=text: text.__setitem__(slot, embedding),
            )

    def _enc_enemies(self, enemies: np.ndarray, enemy_text: np.ndarray, enemy_mask: np.ndarray, obs: dict) -> None:
        combat = obs.get("combat") or {}
        entries = combat.get("enemies") or []
        texts: list[str] = []
        text_slots: list[int] = []

        for index, enemy in enumerate(entries[:MAX_ENEMIES]):
            if not isinstance(enemy, dict):
                continue
            enemy_mask[index] = 1.0
            row = enemies[index]
            power_amounts = _power_amount_map(enemy.get("powers"))

            hp = _float(enemy.get("hp", enemy.get("current_hp")))
            max_hp = _float(enemy.get("max_hp"))
            row[0] = min(hp / max_hp, 1.0) if max_hp > 0 else 0.0
            row[1] = _log_norm(hp, _LOG1P_1200)
            row[2] = _log_norm(max_hp, _LOG1P_1200)
            row[3] = _log_norm(_float(enemy.get("block")), _LOG1P_200_F)

            intent = enemy.get("intent") or {}
            total_damage = _float(intent.get("total_damage"))
            repeats = _float(intent.get("repeats"))
            damage_per_hit = _float(intent.get("damage_per_hit"))
            row[4] = _log_norm(total_damage, _LOG1P_200)
            row[5] = min(repeats / 5.0, 1.0)
            row[6] = min(len(enemy.get("powers") or []) / 8.0, 1.0)
            row[7] = _normalize_power_amount(power_amounts.get("vulnerable", 0.0))
            row[8] = _normalize_power_amount(power_amounts.get("weak", 0.0))
            row[9] = _normalize_power_amount(power_amounts.get("strength", 0.0))
            row[10] = _normalize_power_amount(power_amounts.get("dexterity", 0.0))
            row[11] = _normalize_power_amount(power_amounts.get("artifact", 0.0))
            row[12] = _normalize_power_amount(power_amounts.get("poison", 0.0))
            row[13] = _normalize_power_amount(power_amounts.get("calamity", 0.0))
            row[14] = _normalize_power_amount(power_amounts.get("thorns", 0.0))
            row[15] = _normalize_power_amount(power_amounts.get("intangible", 0.0))
            row[16] = _normalize_power_amount(power_amounts.get("regen", 0.0))
            row[17] = _normalize_power_amount(power_amounts.get("plating", 0.0))
            row[18] = _normalize_power_amount(power_amounts.get("metallicize", 0.0))
            row[19] = _normalize_power_amount(power_amounts.get("ritual", 0.0))

            row[20], row[21], row[22], row[23] = _infer_enemy_intent_flags(intent, total_damage)
            row[24] = _log_norm(damage_per_hit, _LOG1P_100)
            row[25] = min(_float(intent.get("candidate_count")) / 4.0, 1.0)
            row[26] = min(_float(intent.get("alt_total_damage_count")) / 3.0, 1.0)

            if self.use_text:
                text_value = build_live_enemy_semantic_text(enemy)
                if text_value:
                    texts.append(text_value)
                    text_slots.append(index)

        if texts:
            for slot, text_value in zip(text_slots, texts):
                self._register_text_assignment(
                    text_value,
                    lambda embedding, slot=slot, enemy_text=enemy_text: enemy_text.__setitem__(slot, embedding),
                )

    def _enc_powers(self, vector: np.ndarray, obs: dict) -> None:
        combat = obs.get("combat") or {}
        powers = combat.get("player_powers") or []
        buff_count = 0
        debuff_count = 0
        positive_total = 0.0
        negative_total = 0.0
        power_amounts = _power_amount_map(powers)

        for canonical, amount in power_amounts.items():
            index = _POWER_FEATURE_TO_IDX.get(canonical)
            if index is not None and index < len(_PLAYER_POWER_FEATURES):
                vector[index] = _normalize_power_amount(amount)
            if canonical in ("weak", "vulnerable", "frail", "poison", "calamity", "entangled", "lockon"):
                debuff_count += 1
                negative_total += abs(amount)
            else:
                buff_count += 1
                positive_total += abs(amount)

        extras_offset = len(_PLAYER_POWER_FEATURES)
        if POWER_DIM > extras_offset:
            vector[extras_offset] = min(buff_count / 10.0, 1.0)
        if POWER_DIM > extras_offset + 1:
            vector[extras_offset + 1] = min(debuff_count / 10.0, 1.0)
        if POWER_DIM > extras_offset + 2:
            vector[extras_offset + 2] = _normalize_power_amount(positive_total)
        if POWER_DIM > extras_offset + 3:
            vector[extras_offset + 3] = _normalize_power_amount(negative_total)

    def _enc_relic_signals(self, vector: np.ndarray, obs: dict) -> None:
        player = obs.get("player") or {}
        _encode_relic_signals(vector, player.get("relics"))

    def _enc_relics(self, relics: np.ndarray, relic_mask: np.ndarray, obs: dict) -> None:
        player = obs.get("player") or {}
        entries = player.get("relics") or []
        if not isinstance(entries, list) or not self.use_text:
            return

        texts: list[str] = []
        slots: list[int] = []
        for index, relic in enumerate(entries[:MAX_RELICS]):
            if isinstance(relic, dict):
                canonical_text = build_live_relic_semantic_text(relic)
                if not canonical_text:
                    canonical_text = relic.get("canonical_text", "") or relic.get("title", "")
            elif isinstance(relic, str):
                canonical_text = relic
            else:
                continue
            if canonical_text and canonical_text != "[empty]":
                relic_mask[index] = 1.0
                texts.append(canonical_text)
                slots.append(index)

        if texts:
            for slot, text_value in zip(slots, texts):
                self._register_text_assignment(
                    text_value,
                    lambda embedding, slot=slot, relics=relics: relics.__setitem__(slot, embedding),
                )

    def _enc_potions(self, potions: np.ndarray, potion_mask: np.ndarray, obs: dict) -> None:
        player = obs.get("player") or {}
        entries = player.get("potions") or []
        if not isinstance(entries, list) or not self.use_text:
            return

        texts: list[str] = []
        slots: list[int] = []
        for index, potion in enumerate(entries[:MAX_POTIONS]):
            if isinstance(potion, dict):
                canonical_text = build_live_potion_semantic_text(potion)
                if not canonical_text:
                    canonical_text = potion.get("canonical_text", "")
                title = potion.get("title", "")
            elif isinstance(potion, str):
                canonical_text = ""
                title = potion
            else:
                continue
            if title == "[empty]" or not (canonical_text or title):
                continue
            potion_mask[index] = 1.0
            texts.append(canonical_text or title)
            slots.append(index)

        if texts:
            for slot, text_value in zip(slots, texts):
                self._register_text_assignment(
                    text_value,
                    lambda embedding, slot=slot, potions=potions: potions.__setitem__(slot, embedding),
                )

    def _enc_context(self, vector: np.ndarray, obs: dict, planner_context: dict[str, Any] | None = None) -> None:
        if not self.use_text:
            return
        decision = obs.get("decision") or {}
        text = ""
        if isinstance(decision, dict):
            text = decision.get("decision_text", "")
        if not text:
            domain = self._resolve_domain(obs)
            phase = obs.get("phase", "")
            combat = obs.get("combat") or {}
            enemy_intents: list[str] = []
            for enemy in (combat.get("enemies") or [])[:3]:
                if not isinstance(enemy, dict):
                    continue
                intent_text = build_enemy_intent_semantic_text(enemy.get("intent"))
                if intent_text:
                    enemy_intents.append(intent_text)
            if enemy_intents:
                text = f"阶段：{phase}｜决策域：{domain}｜敌人意图：{' || '.join(enemy_intents)}"
            else:
                text = f"阶段：{phase}｜决策域：{domain}"
        if isinstance(planner_context, dict):
            run_memory_text = str(planner_context.get("run_memory_text") or "").strip()
            objective_text = str(planner_context.get("objective_text") or "").strip()
            extra_parts = [part for part in (run_memory_text, objective_text) if part]
            if extra_parts:
                text = " ｜ ".join(part for part in (text, *extra_parts) if part)
        self._register_text_assignment(
            text,
            lambda embedding, vector=vector: vector.__setitem__(slice(None), embedding),
        )

    def _enc_actions(
        self,
        actions: np.ndarray,
        action_text: np.ndarray,
        semantic_actions: np.ndarray,
        semantic_action_text: np.ndarray,
        route_summary: np.ndarray,
        route_nodes: np.ndarray,
        route_node_mask: np.ndarray,
        action_mask: np.ndarray,
        legal_actions: list,
        planner_context: dict[str, Any] | None = None,
    ) -> None:
        texts: list[str] = []
        text_slots: list[int] = []
        semantic_texts: list[str] = []
        semantic_text_slots: list[int] = []
        semantic_signatures = (
            planner_context.get("semantic_actions")
            if isinstance(planner_context, dict) and isinstance(planner_context.get("semantic_actions"), list)
            else []
        )
        count = min(len(legal_actions), MAX_ACTIONS)
        for index in range(count):
            action = legal_actions[index]
            if not isinstance(action, dict):
                continue
            action_mask[index] = 1.0
            self._enc_action_numeric(actions[index], action)
            signature = semantic_signatures[index] if index < len(semantic_signatures) else semantic_action_signature(action)
            semantic_actions[index] = encode_semantic_action_numeric(signature)
            self._enc_route_action(route_summary[index], route_nodes[index], route_node_mask[index], action)
            text_value = self._build_action_text(action)
            if text_value and self.use_text:
                texts.append(text_value)
                text_slots.append(index)
            semantic_text_value = build_semantic_action_text(signature)
            if semantic_text_value and self.use_text:
                semantic_texts.append(semantic_text_value)
                semantic_text_slots.append(index)

        for slot, text_value in zip(text_slots, texts):
            self._register_text_assignment(
                text_value,
                lambda embedding, slot=slot, action_text=action_text: action_text.__setitem__(slot, embedding),
            )
        for slot, text_value in zip(semantic_text_slots, semantic_texts):
            self._register_text_assignment(
                text_value,
                lambda embedding, slot=slot, semantic_action_text=semantic_action_text: semantic_action_text.__setitem__(slot, embedding),
            )

    def _build_live_card_text(self, card: dict | None) -> str:
        if not isinstance(card, dict):
            return ""
        semantic_text = build_live_card_semantic_text(card)
        if semantic_text:
            return semantic_text
        return str(card.get("canonical_text") or card.get("title") or "").strip()

    def _build_action_text(self, action: dict | None) -> str:
        if not isinstance(action, dict):
            return ""

        kind = str(action.get("kind") or "").strip()
        canonical_text = str(action.get("canonical_text") or "").strip()
        target = action.get("target") if isinstance(action.get("target"), dict) else {}
        target_name = str((target or {}).get("name") or "").strip()

        card = action.get("card")
        if isinstance(card, dict):
            card_text = self._build_live_card_text(card)
            if kind == "play_card":
                parts = ["play", card_text]
                if target_name:
                    parts.append(f"tgt {target_name}")
                return " | ".join(part for part in parts if part)
            if kind == "card_reward":
                selection = str(action.get("selection") or "").strip().lower()
                if "skip" in selection:
                    return canonical_text or "skip card reward"
                return " | ".join(part for part in ("pick", card_text) if part)
            if kind == "deck_upgrade":
                selection = str(action.get("selection") or "").strip().lower()
                if any(token in selection for token in ("confirm", "cancel", "close")):
                    return canonical_text
                preview = action.get("upgrade_preview")
                preview_text = self._build_live_card_text(preview) if isinstance(preview, dict) else ""
                parts = ["upgrade", card_text]
                if preview_text:
                    parts.append(f"to {preview_text}")
                return " | ".join(part for part in parts if part)
            if kind in {"card_selection", "combat_select_card", "combat_select"}:
                selection = str(action.get("selection") or "").strip().lower()
                if any(token in selection for token in ("confirm", "cancel", "close", "skip")):
                    return canonical_text
                semantics = str(action.get("selection_semantics") or "").strip()
                prefix = f"select {semantics}".strip() if semantics else "select"
                return " | ".join(part for part in (prefix, card_text) if part)

        potion = action.get("potion")
        if isinstance(potion, dict):
            potion_text = build_live_potion_semantic_text(potion)
            if kind == "use_potion":
                parts = ["use potion", potion_text]
                if target_name:
                    parts.append(f"tgt {target_name}")
                return " | ".join(part for part in parts if part)
            if kind == "discard_potion":
                return " | ".join(part for part in ("discard potion", potion_text) if part)

        reward = action.get("reward")
        if isinstance(reward, dict):
            reward_relic = reward.get("relic")
            if isinstance(reward_relic, dict):
                relic_text = build_live_relic_semantic_text(reward_relic)
                if relic_text:
                    prefix = "take relic" if kind in ("reward", "treasure_relic") else (kind or "relic")
                    return " | ".join(part for part in (prefix, relic_text) if part)

            reward_potion = reward.get("potion")
            if isinstance(reward_potion, dict):
                potion_text = build_live_potion_semantic_text(reward_potion)
                if potion_text:
                    prefix = "take potion" if kind == "reward" else (kind or "potion")
                    return " | ".join(part for part in (prefix, potion_text) if part)

        if kind == "shop":
            item = action.get("item")
            if isinstance(item, dict):
                shop_action = str(action.get("shop_action") or "").strip().lower()
                item_cost = item.get("cost")
                cost_text = f"cost {_float(item_cost):.0f}" if item_cost is not None else ""
                item_card = item.get("card")
                if isinstance(item_card, dict):
                    item_text = self._build_live_card_text(item_card)
                    if item_text:
                        prefix = "leave shop" if any(token in shop_action for token in ("leave", "back")) else "buy"
                        return " | ".join(part for part in (prefix, item_text, cost_text) if part)
                item_relic = item.get("relic")
                if isinstance(item_relic, dict):
                    item_text = build_live_relic_semantic_text(item_relic)
                    if item_text:
                        prefix = "leave shop" if any(token in shop_action for token in ("leave", "back")) else "buy relic"
                        return " | ".join(part for part in (prefix, item_text, cost_text) if part)
                item_potion = item.get("potion")
                if isinstance(item_potion, dict):
                    item_text = build_live_potion_semantic_text(item_potion)
                    if item_text:
                        prefix = "leave shop" if any(token in shop_action for token in ("leave", "back")) else "buy potion"
                        return " | ".join(part for part in (prefix, item_text, cost_text) if part)

        relic = action.get("relic")
        if isinstance(relic, dict):
            relic_text = build_live_relic_semantic_text(relic)
            if kind == "treasure_relic":
                return " | ".join(part for part in ("take relic", relic_text) if part)
            if relic_text:
                return " | ".join(part for part in (kind or "relic", relic_text) if part)

        return canonical_text

    def _enc_action_numeric(self, row: np.ndarray, action: dict) -> None:
        kind = action.get("kind", "")
        row[0] = _KIND_TO_ORD.get(kind, 0) / _NUM_KINDS

        card = action.get("card")
        if isinstance(card, dict):
            row[1] = 1.0
            row[2] = min(_float(card.get("cost")) / 5.0, 1.0)
            row[3] = min(_float(card.get("star")) / 5.0, 1.0) if card.get("star") is not None else 0.0
            card_type = (card.get("type") or "").capitalize()
            row[4] = 1.0 if card_type == "Attack" else 0.0
            row[5] = 1.0 if card_type == "Skill" else 0.0
            row[6] = 1.0 if card_type == "Power" else 0.0
            row[20] = 1.0 if card_type == "Status" else 0.0
            row[21] = 1.0 if card_type == "Curse" else 0.0

        target = action.get("target")
        row[7] = 1.0 if isinstance(target, dict) and target.get("name") else 0.0
        row[8] = 1.0 if isinstance(target, dict) and target.get("side") == "Player" else 0.0
        row[9] = 1.0 if action.get("action_id") == "end_turn" else 0.0
        row[10] = 1.0 if kind == "proceed" and not action.get("skip") else 0.0
        row[11] = 1.0 if action.get("skip") or "skip" in (action.get("action_id") or "") else 0.0

        item = action.get("item")
        if isinstance(item, dict):
            row[12] = _log_norm(_float(item.get("cost")), _LOG1P_500)

        reward = action.get("reward")
        if isinstance(reward, dict):
            reward_type = reward.get("type", reward.get("reward_type", ""))
            row[13] = 1.0 if reward_type == "gold" else 0.0
            row[14] = 1.0 if reward_type == "card" else 0.0

        point_type = self._normalize_route_point_type(action.get("point_type_norm") or action.get("point_type", ""))
        row[15] = _PT_TO_ORD.get(point_type, 0) / _NUM_PT

        coord = action.get("coord")
        if isinstance(coord, dict):
            row[16] = min(_float(coord.get("row")) / 15.0, 1.0)
            row[17] = min(_float(coord.get("col")) / 7.0, 1.0)

        row[18] = 1.0 if isinstance(action.get("upgrade_preview"), dict) else 0.0

        option_index = action.get("index")
        if option_index is None:
            option_index = action.get("hand_index")
        if option_index is None:
            option_index = action.get("slot_index")
        row[19] = min(_float(option_index) / 20.0, 1.0) if option_index is not None else 0.0

        source = None
        if isinstance(card, dict):
            source = card
        else:
            potion = action.get("potion")
            if isinstance(potion, dict):
                source = potion

        if isinstance(source, dict):
            preview = _build_card_preview_bundle(source)
            strength, dexterity, energy, hits = _get_card_extra_metrics(source)
            kw_flags, _ = _get_card_keywords(source)
            row[22] = _log_norm(preview["preview_damage"], _LOG1P_200)
            row[23] = _log_norm(preview["preview_block"], _LOG1P_200)
            row[24] = min(_preview_metric(source, "draw") / 5.0, 1.0)
            row[25] = min(_preview_metric(source, "weak") / 5.0, 1.0)
            row[26] = min(_preview_metric(source, "vulnerable") / 5.0, 1.0)
            row[27] = _log_norm(_preview_metric(source, "heal"), _LOG1P_200)
            row[28] = _log_norm(_preview_metric(source, "hp_loss"), _LOG1P_200)
            row[29] = min(_preview_metric(source, "summon") / 5.0, 1.0)
            row[30] = min(strength / 10.0, 1.0)
            row[31] = min(dexterity / 10.0, 1.0)
            row[32] = min(energy / 5.0, 1.0)
            row[33] = min(hits / 10.0, 1.0)
            row[34] = 1.0 if kw_flags[0] else 0.0
            row[35] = 1.0 if kw_flags[1] else 0.0
            row[36] = 1.0 if kw_flags[2] else 0.0
            row[37] = 1.0 if kw_flags[3] else 0.0
            row[38] = 1.0 if (
                preview["preview_damage"] > 0
                or preview["preview_block"] > 0
                or _preview_metric(source, "draw") > 0
                or _preview_metric(source, "weak") > 0
                or _preview_metric(source, "vulnerable") > 0
                or _preview_metric(source, "heal") > 0
                or strength > 0
                or dexterity > 0
                or _preview_metric(source, "summon") > 0
            ) else 0.0
            row[39] = 1.0 if isinstance(card, dict) and _float(card.get("cost")) == 0 else 0.0
            row[40] = _log_norm(preview["base_damage"], _LOG1P_200)
            row[41] = _log_norm(preview["base_block"], _LOG1P_200)
            row[42] = _log_norm(preview["preview_damage_per_hit"], _LOG1P_100)
            row[43] = _log_norm(preview["preview_damage_per_energy"], _LOG1P_100)
            row[44] = _log_norm(preview["preview_block_per_energy"], _LOG1P_100)
            row[45] = _signed_log_norm(preview["preview_damage_delta"], _LOG1P_100)
            row[46] = _signed_log_norm(preview["preview_block_delta"], _LOG1P_100)
            row[47] = min(_infer_upgrade_level(source) / 3.0, 1.0) if isinstance(card, dict) else 0.0
            row[48] = min(_preview_metric(source, "x_cost_value") / 10.0, 1.0)

    def _enc_route_action(
        self,
        summary_row: np.ndarray,
        node_rows: np.ndarray,
        node_mask: np.ndarray,
        action: dict,
    ) -> None:
        route_summary = action.get("route_summary")
        if not isinstance(route_summary, dict):
            return

        summary_row[0] = min(_float(route_summary.get("reachable_node_count")) / 30.0, 1.0)
        summary_row[1] = min(_float(route_summary.get("max_depth")) / 15.0, 1.0)
        summary_row[2] = min(_float(route_summary.get("direct_child_count")) / 4.0, 1.0)
        summary_row[3] = min(_float(route_summary.get("forced_path_steps_before_branch")) / 10.0, 1.0)
        summary_row[4] = min(_float(route_summary.get("count_monster")) / 10.0, 1.0)
        summary_row[5] = min(_float(route_summary.get("count_elite")) / 5.0, 1.0)
        summary_row[6] = min(_float(route_summary.get("count_boss")) / 2.0, 1.0)
        summary_row[7] = min(_float(route_summary.get("count_event")) / 10.0, 1.0)
        summary_row[8] = min(_float(route_summary.get("count_question_mark")) / 10.0, 1.0)
        summary_row[9] = min(_float(route_summary.get("count_rest_site")) / 5.0, 1.0)
        summary_row[10] = min(_float(route_summary.get("count_shop")) / 5.0, 1.0)
        summary_row[11] = min(_float(route_summary.get("count_treasure")) / 5.0, 1.0)
        summary_row[12] = self._norm_step(route_summary.get("next_elite_steps"))
        summary_row[13] = self._norm_step(route_summary.get("next_rest_steps"))
        summary_row[14] = self._norm_step(route_summary.get("next_shop_steps"))
        summary_row[15] = self._norm_step(route_summary.get("next_event_steps"))
        summary_row[16] = self._norm_step(route_summary.get("next_question_mark_steps"))
        summary_row[17] = self._norm_step(route_summary.get("next_treasure_steps"))
        summary_row[18] = _bool(route_summary.get("can_reach_rest_site_before_elite"))
        summary_row[19] = _bool(route_summary.get("can_reach_elite_then_rest_site"))

        nodes = action.get("route_nodes") or []
        for index, node in enumerate(nodes[:MAX_ROUTE_NODES]):
            if not isinstance(node, dict):
                continue
            node_mask[index] = 1.0
            self._enc_route_node(node_rows[index], node)

    def _enc_route_node(self, row: np.ndarray, node: dict) -> None:
        point_type = self._normalize_route_point_type(node.get("point_type"))
        type_index = _ROUTE_PT_TO_IDX.get(point_type)
        if type_index is not None:
            row[type_index] = 1.0

        base = _NUM_ROUTE_PT
        row[base] = min(_float(node.get("depth")) / 15.0, 1.0)
        coord = node.get("coord")
        if isinstance(coord, dict):
            row[base + 1] = min(_float(coord.get("row")) / 15.0, 1.0)
            row[base + 2] = min(_float(coord.get("col")) / 7.0, 1.0)
        row[base + 3] = min(_float(node.get("child_count")) / 4.0, 1.0)
        row[base + 4] = _bool(node.get("is_leaf"))

    def _resolve_domain(self, obs: dict) -> str:
        decision_domain = obs.get("decision_domain")
        if isinstance(decision_domain, str) and decision_domain in DOMAIN_TO_IDX:
            return decision_domain

        phase = obs.get("phase", "")
        if phase == "combat":
            return "combat"
        if phase == "map":
            return "route"
        if phase == "card_selection":
            combat = obs.get("combat")
            return "combat" if combat else "build"
        if phase == "settling":
            combat = obs.get("combat")
            return "combat" if combat else "build"
        return "build"

    @staticmethod
    def _norm_step(value) -> float:
        if value is None:
            return 0.0
        return min(_float(value) / 10.0, 1.0)

    @staticmethod
    def _normalize_route_point_type(point_type) -> str:
        if point_type in ("Merchant", "Shop"):
            return "Shop"
        if point_type in ("Rest", "RestSite"):
            return "RestSite"
        if point_type in ("Unknown", "QuestionMark"):
            return "QuestionMark"
        if point_type in _ROUTE_PT_TO_IDX:
            return point_type
        return "Monster"

    @staticmethod
    def _parse_act(act_id):
        if not act_id or not isinstance(act_id, str):
            return 0.0
        for char in reversed(act_id):
            if char.isdigit():
                try:
                    return float(char)
                except Exception:
                    pass
        return 0.0


# Backward-compatible alias for legacy callers that still import the old name.
DictObservationEncoder = DenseObservationEncoder
