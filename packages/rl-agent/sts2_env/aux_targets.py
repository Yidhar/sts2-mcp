"""Auxiliary supervision targets for omni-attention policy training.

This module defines lightweight, action-grounded target construction for the
search-free policy.  The targets are intentionally split into:

- objective: one-step planner-aligned reward decomposition
- transition: action-conditioned combat next-state summary
- traits: action/context affordance flags that force candidate attention to
  surface potion/relic/energy/cycle/build/selection/route lines explicitly
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import observation_common as obs_common
from .objective_heads import (
    OBJECTIVE_HEAD_NAMES,
    NUM_OBJECTIVE_HEADS,
    compute_transition_objective_rewards,
)
from .run_memory import _build_profile
from .semantic_action import semantic_action_signature

TRANSITION_HEAD_NAMES = (
    "next_player_hp_ratio",
    "next_player_block_ratio",
    "next_energy_ratio",
    "next_enemy_hp_ratio",
    "next_intent_damage_ratio",
    "next_draw_pile_ratio",
    "next_discard_pile_ratio",
    "next_exhaust_pile_ratio",
)
NUM_TRANSITION_HEADS = len(TRANSITION_HEAD_NAMES)

TRAIT_HEAD_NAMES = (
    "energy_line",
    "potion_line",
    "relic_line",
    "enemy_risk_line",
    "draw_cycle_line",
    "discard_exhaust_line",
    "build_line",
    "route_line",
)
NUM_TRAIT_HEADS = len(TRAIT_HEAD_NAMES)

BUILD_HEAD_NAMES = (
    "frontload_fit",
    "defense_fit",
    "draw_fit",
    "energy_fit",
    "scaling_fit",
    "cycle_fit",
    "economy_fit",
    "novelty_fit",
)
NUM_BUILD_HEADS = len(BUILD_HEAD_NAMES)

ROUTE_HEAD_NAMES = (
    "safe_value",
    "elite_value",
    "rest_value",
    "shop_value",
    "event_value",
    "treasure_value",
    "branch_value",
    "overall_value",
)
NUM_ROUTE_HEADS = len(ROUTE_HEAD_NAMES)

SELECTION_HEAD_NAMES = (
    "source_hand",
    "source_draw",
    "source_discard",
    "source_exhaust",
    "source_deck",
    "source_play",
    "upgrade_smith_line",
    "transform_mutate_line",
    "remove_purge_line",
    "discard_exhaust_line",
    "reward_discover_line",
    "combat_runtime_line",
)
NUM_SELECTION_HEADS = len(SELECTION_HEAD_NAMES)

_COMBAT_ACTION_FAMILIES = {"play_card", "use_potion", "discard_potion", "end_turn"}
_BUILD_ACTION_FAMILIES = {"reward", "card_reward", "shop", "rest", "smith", "deck_upgrade", "event_option", "treasure_relic"}
_ROUTE_ACTION_FAMILIES = {"map"}


def _neutral_objective_vector() -> np.ndarray:
    """Return a stable, non-degenerate fallback objective context.

    Aux target generation is sometimes invoked from tests, offline transforms, or
    sparse info paths where planner_context has not been materialized yet.
    Returning an all-zero vector in those cases collapses route/build biases to
    hard zero and teaches the candidate heads the wrong thing.  A neutral
    baseline keeps the heads trainable without injecting an overly opinionated
    planner prior.
    """

    vector = np.zeros(16, dtype=np.float32)
    vector[0] = 1.0  # survival priority
    vector[1] = 0.7  # hp loss priority
    vector[2] = 0.5  # build priority
    vector[3] = 0.5  # resource priority
    vector[4] = 0.5  # preserve hp bias
    vector[5] = 0.5  # save potion mode
    vector[6] = 0.5  # force rest mode
    vector[7] = 0.5  # greed upgrade mode
    vector[8] = 0.5  # elite pressure
    vector[9] = 0.5  # boss pressure
    vector[10] = 0.5  # safe route bias
    vector[11] = 0.5  # shop value bias
    vector[12] = 0.5  # rest value bias
    vector[13] = 0.5  # smith value bias
    vector[14] = 0.5  # zero damage desire
    vector[15] = 0.0  # long horizon mode defaults off when unknown
    return vector


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _in_combat(obs: dict[str, Any] | None) -> bool:
    combat = (obs or {}).get("combat") if isinstance(obs, dict) else None
    return isinstance(combat, dict) and bool(combat)


def _combat_state(obs: dict[str, Any] | None) -> dict[str, Any]:
    combat = (obs or {}).get("combat") if isinstance(obs, dict) else None
    return combat if isinstance(combat, dict) else {}


def _player_state(obs: dict[str, Any] | None) -> dict[str, Any]:
    player = (obs or {}).get("player") if isinstance(obs, dict) else None
    return player if isinstance(player, dict) else {}


def _player_hp(obs: dict[str, Any] | None) -> float:
    player = _player_state(obs)
    return _float(player.get("hp", player.get("current_hp")))


def _player_max_hp(obs: dict[str, Any] | None) -> float:
    player = _player_state(obs)
    return max(_float(player.get("max_hp"), 1.0), 1.0)


def _player_block(obs: dict[str, Any] | None) -> float:
    player = _player_state(obs)
    combat = _combat_state(obs)
    return max(
        _float(combat.get("block")),
        _float(player.get("block")),
        _float((player.get("creature") or {}).get("block")) if isinstance(player.get("creature"), dict) else 0.0,
    )


def _combat_energy(obs: dict[str, Any] | None) -> float:
    combat = _combat_state(obs)
    return _float(combat.get("energy"))


def _combat_max_energy(obs: dict[str, Any] | None) -> float:
    combat = _combat_state(obs)
    player = _player_state(obs)
    return max(_float(combat.get("max_energy")), _float(player.get("max_energy")), 1.0)


def _combat_enemies(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
    enemies = _combat_state(obs).get("enemies")
    if not isinstance(enemies, list):
        return []
    return [enemy for enemy in enemies if isinstance(enemy, dict)]


def _combat_enemy_total_hp(obs: dict[str, Any] | None) -> float:
    total = 0.0
    for enemy in _combat_enemies(obs):
        total += _float(enemy.get("hp", enemy.get("current_hp")))
    return total


def _combat_total_intent_damage(obs: dict[str, Any] | None) -> float:
    total = 0.0
    for enemy in _combat_enemies(obs):
        intent = enemy.get("intent")
        if not isinstance(intent, dict):
            continue
        total_damage = _float(intent.get("total_damage"))
        if total_damage <= 0.0:
            total_damage = _float(intent.get("damage_per_hit")) * max(_float(intent.get("repeats"), 1.0), 1.0)
        total += max(total_damage, 0.0)
    return total


def _runtime_cards(obs: dict[str, Any] | None, pile_name: str) -> list[dict[str, Any]]:
    combat = _combat_state(obs)
    value = combat.get(pile_name)
    if isinstance(value, list):
        return [card for card in value if isinstance(card, dict)]
    if isinstance(value, dict):
        cards = value.get("cards")
        if isinstance(cards, list):
            return [card for card in cards if isinstance(card, dict)]
    return []


def _nonempty_potions(obs: dict[str, Any] | None) -> list[Any]:
    player = _player_state(obs)
    potions = player.get("potions")
    if not isinstance(potions, list):
        return []
    out: list[Any] = []
    for potion in potions:
        if isinstance(potion, str) and potion.strip() and potion.strip() != "[empty]":
            out.append(potion)
        elif isinstance(potion, dict):
            title = str(potion.get("title") or potion.get("id") or "").strip()
            if title and title != "[empty]":
                out.append(potion)
    return out


def _player_relics(obs: dict[str, Any] | None) -> list[Any]:
    player = _player_state(obs)
    relics = player.get("relics")
    if not isinstance(relics, list):
        return []
    return [relic for relic in relics if isinstance(relic, (dict, str))]


def _player_gold(obs: dict[str, Any] | None) -> float:
    player = _player_state(obs)
    return _float(player.get("gold"))


def _player_deck_cards(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
    player = _player_state(obs)
    deck_cards = player.get("deck_cards")
    if not isinstance(deck_cards, list):
        return []
    return [card for card in deck_cards if isinstance(card, dict)]


def _action_source(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    for candidate in (
        action.get("card"),
        action.get("potion"),
        (action.get("item") or {}).get("card") if isinstance(action.get("item"), dict) else None,
        (action.get("item") or {}).get("potion") if isinstance(action.get("item"), dict) else None,
        (action.get("reward") or {}).get("card") if isinstance(action.get("reward"), dict) else None,
        action.get("upgrade_preview"),
    ):
        if isinstance(candidate, dict):
            return candidate
    return None


def _source_text(source: dict[str, Any] | None) -> str:
    if not isinstance(source, dict):
        return ""
    fragments: list[str] = []
    for key in ("title", "name", "description", "text", "canonical_text"):
        value = str(source.get(key) or "").strip()
        if value:
            fragments.append(value)
    keywords = source.get("keywords")
    if isinstance(keywords, list):
        fragments.extend(str(keyword or "").strip() for keyword in keywords if str(keyword or "").strip())
    return " | ".join(fragments).lower()


def _source_profile(source: dict[str, Any] | None) -> dict[str, float]:
    profile = {
        "cost": 0.0,
        "x_cost": 0.0,
        "attack": 0.0,
        "skill": 0.0,
        "power": 0.0,
        "zero_cost": 0.0,
        "damage": 0.0,
        "block": 0.0,
        "draw": 0.0,
        "energy": 0.0,
        "heal": 0.0,
        "weak": 0.0,
        "vulnerable": 0.0,
        "hits": 0.0,
        "exhaust": 0.0,
        "ethereal": 0.0,
        "retain": 0.0,
    }
    if not isinstance(source, dict):
        return profile

    preview = obs_common._build_card_preview_bundle(source)
    _, _, energy, hits = obs_common._get_card_extra_metrics(source)
    kw_flags, _ = obs_common._get_card_keywords(source)
    card_type = str(source.get("type") or "").capitalize()
    cost = _float(source.get("cost"))
    profile.update(
        {
            "cost": max(cost, 0.0),
            "x_cost": float(bool(source.get("x_cost"))),
            "attack": 1.0 if card_type == "Attack" else 0.0,
            "skill": 1.0 if card_type == "Skill" else 0.0,
            "power": 1.0 if card_type == "Power" else 0.0,
            "zero_cost": 1.0 if cost == 0.0 else 0.0,
            "damage": float(preview["preview_damage"]),
            "block": float(preview["preview_block"]),
            "draw": obs_common._preview_metric(source, "draw"),
            "energy": float(energy),
            "heal": obs_common._preview_metric(source, "heal"),
            "weak": obs_common._preview_metric(source, "weak"),
            "vulnerable": obs_common._preview_metric(source, "vulnerable"),
            "hits": float(hits),
            "exhaust": 1.0 if kw_flags[0] else 0.0,
            "ethereal": 1.0 if kw_flags[1] else 0.0,
            "retain": 1.0 if kw_flags[2] else 0.0,
        }
    )
    return profile


def _source_supports_draw_cycle(source: dict[str, Any] | None, signature: dict[str, Any]) -> float:
    text = _source_text(source)
    profile = _source_profile(source)
    drawish = max(
        float(profile["draw"] > 0.0),
        float("draw" in text),
        float("shuffle" in text),
        float("top of your draw pile" in text or "draw pile" in text),
        float("discard pile" in text and ("draw" in text or "shuffle" in text)),
        float("role=draw" in str(signature).lower()),
    )
    return float(np.clip(drawish, 0.0, 1.0))


def _source_supports_discard_or_exhaust(source: dict[str, Any] | None) -> float:
    text = _source_text(source)
    profile = _source_profile(source)
    discardish = 0.0
    if "discard" in text:
        discardish += 0.70
    if "exhaust" in text:
        discardish += 0.80
    if "ethereal" in text:
        discardish += 0.35
    discardish += 0.50 * profile["exhaust"]
    discardish += 0.25 * profile["ethereal"]
    return float(np.clip(discardish, 0.0, 1.0))


def _choose_target_enemy(obs: dict[str, Any] | None, action: dict[str, Any] | None) -> dict[str, Any] | None:
    enemies = _combat_enemies(obs)
    if not enemies or not isinstance(action, dict):
        return enemies[0] if enemies else None
    target = action.get("target")
    if isinstance(target, dict):
        combat_id = target.get("combat_id")
        target_name = str(target.get("name") or "").strip().lower()
        for enemy in enemies:
            if combat_id is not None and enemy.get("combat_id") == combat_id:
                return enemy
            if target_name and str(enemy.get("name") or "").strip().lower() == target_name:
                return enemy
    return enemies[0]


def _enemy_risk_score(enemy: dict[str, Any] | None) -> float:
    if not isinstance(enemy, dict):
        return 0.0
    texts: list[str] = []
    for collection_name in ("powers", "static_traits", "reactive_triggers", "phase_rules"):
        collection = enemy.get(collection_name)
        if isinstance(collection, list):
            for item in collection:
                if not isinstance(item, dict):
                    continue
                texts.extend(
                    str(item.get(key) or "").strip().lower()
                    for key in ("title", "description", "trait", "effect_type", "condition", "state")
                    if str(item.get(key) or "").strip()
                )
    intent = enemy.get("intent")
    if isinstance(intent, dict):
        texts.extend(
            str(intent.get(key) or "").strip().lower()
            for key in ("description", "label", "intent_type")
            if str(intent.get(key) or "").strip()
        )
    joined = " | ".join(texts)
    if not joined:
        return 0.0
    risk = 0.0
    if any(keyword in joined for keyword in ("thorn", "retaliat", "contact", "spike", "punish")):
        risk += 0.75
    if any(keyword in joined for keyword in ("split", "threshold", "phase", "stun")):
        risk += 0.65
    if any(keyword in joined for keyword in ("intangible", "artifact", "buffer")):
        risk += 0.35
    return float(np.clip(risk, 0.0, 1.0))


def _relic_line_score(obs: dict[str, Any] | None, action: dict[str, Any] | None, signature: dict[str, Any]) -> float:
    relics = _player_relics(obs)
    if not relics:
        return 0.0
    signal_vector = np.zeros(obs_common.RELIC_SIGNAL_DIM, dtype=np.float32)
    obs_common._encode_relic_signals(signal_vector, relics)
    source_profile = _source_profile(_action_source(action))
    score = float(np.clip(signal_vector.sum() / 4.0, 0.0, 1.0))
    score += 0.20 * signal_vector[0] * float(signature.get("is_x_cost") or source_profile["cost"] >= 2.0)
    score += 0.15 * signal_vector[1] * float(source_profile["draw"] > 0.0 or source_profile["zero_cost"] > 0.5)
    score += 0.20 * max(signal_vector[2], signal_vector[4]) * float(signature.get("is_attack") or source_profile["damage"] > 0.0)
    score += 0.20 * max(signal_vector[3], signal_vector[10], signal_vector[11]) * float(signature.get("is_skill") or source_profile["block"] > 0.0)
    score += 0.10 * float(signature.get("is_power"))
    return float(np.clip(score, 0.0, 1.0))


def _same_card_count(obs: dict[str, Any] | None, source: dict[str, Any] | None) -> float:
    if not isinstance(source, dict):
        return 0.0
    player = _player_state(obs)
    deck_cards = player.get("deck_cards")
    if not isinstance(deck_cards, list):
        return 0.0
    source_id = str(source.get("id") or "").strip()
    source_title = str(source.get("title") or "").strip().lower()
    count = 0.0
    for card in deck_cards:
        if not isinstance(card, dict):
            continue
        card_id = str(card.get("id") or "").strip()
        card_title = str(card.get("title") or "").strip().lower()
        if source_id and card_id and source_id == card_id:
            count += 1.0
        elif source_title and card_title and source_title == card_title:
            count += 1.0
    return count


def _item_price(action: dict[str, Any] | None) -> float:
    if not isinstance(action, dict):
        return 0.0
    if action.get("price") is not None:
        return _float(action.get("price"))
    if action.get("cost") is not None:
        return _float(action.get("cost"))
    item = action.get("item") if isinstance(action.get("item"), dict) else None
    if isinstance(item, dict):
        return _float(item.get("cost"))
    return 0.0


def _planner_objective_vector(planner_context: dict[str, Any] | None) -> np.ndarray:
    if not isinstance(planner_context, dict):
        return _neutral_objective_vector()
    value = planner_context.get("objective_context_vector")
    if value is None:
        return _neutral_objective_vector()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 16:
        padded = _neutral_objective_vector()
        padded[: array.size] = array
        return padded
    if float(np.abs(array[:16]).sum()) <= 1e-6:
        return _neutral_objective_vector()
    return array[:16]


def _card_identity(card: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(card, dict):
        return "", ""
    card_id = str(card.get("id") or "").strip().lower()
    title = str(card.get("title") or card.get("name") or "").strip().lower()
    return card_id, title


def _cards_match(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    left_id, left_title = _card_identity(left)
    right_id, right_title = _card_identity(right)
    if left_id and right_id:
        return left_id == right_id
    if left_title and right_title:
        return left_title == right_title
    return False


def _selection_text(action: dict[str, Any] | None, signature: dict[str, Any] | None = None) -> str:
    if not isinstance(action, dict):
        return ""
    signature = signature if isinstance(signature, dict) else {}
    fragments: list[str] = []
    for value in (
        signature.get("selection_semantics"),
        action.get("selection_semantics"),
        action.get("selection_prompt"),
        action.get("selection_action"),
        action.get("selection"),
        action.get("surface"),
        action.get("screen_type"),
        action.get("action_id"),
    ):
        text = str(value or "").strip()
        if text:
            fragments.append(text)
    return " | ".join(fragments)


def _selection_source_zone(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    signature: dict[str, Any] | None = None,
) -> str:
    signature = signature if isinstance(signature, dict) else {}
    action_card = action.get("card") if isinstance(action, dict) and isinstance(action.get("card"), dict) else None
    if isinstance(action_card, dict):
        for zone_name, pile_name in (
            ("hand", "hand"),
            ("draw", "draw_pile"),
            ("discard", "discard_pile"),
            ("exhaust", "exhaust_pile"),
            ("play", "play_pile"),
        ):
            if any(_cards_match(card, action_card) for card in _runtime_cards(prev_obs, pile_name)):
                return zone_name
        if any(_cards_match(card, action_card) for card in _player_deck_cards(prev_obs)):
            return "deck"

    selection_text = _selection_text(action, signature).lower()
    keyword_map = (
        ("play", ("play pile", "played")),
        ("discard", ("discard pile", "discard")),
        ("exhaust", ("exhaust pile", "exhaust")),
        ("draw", ("draw pile", "draw")),
        ("hand", ("hand",)),
        ("deck", ("deck", "master deck")),
    )
    for zone_name, keywords in keyword_map:
        if any(keyword in selection_text for keyword in keywords):
            return zone_name
    return ""


def compute_build_targets(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    *,
    planner_context: dict[str, Any] | None = None,
) -> np.ndarray:
    target = np.zeros(NUM_BUILD_HEADS, dtype=np.float32)
    signature = semantic_action_signature(action)
    family = str(signature.get("family") or "")
    source = _action_source(action)
    source_profile = _source_profile(source)
    build = _build_profile(prev_obs)
    objective = _planner_objective_vector(planner_context)
    greed_upgrade_mode = float(objective[7]) if objective.size > 7 else 0.0
    resource_priority = float(objective[3]) if objective.size > 3 else 0.0

    frontload_need = float(np.clip(0.42 - build["frontload"], 0.0, 1.0))
    defense_need = float(np.clip(0.40 - build["block"], 0.0, 1.0))
    draw_need = float(np.clip(0.24 - build["draw"], 0.0, 1.0))
    scaling_need = float(np.clip(0.18 - build["scaling"], 0.0, 1.0))
    energy_need = float(np.clip(build["high_cost_density"] * 0.7 + build["x_cost_density"] * 0.8 - build["zero_cost_density"] * 0.3, 0.0, 1.0))
    cycle_need = float(np.clip((1.0 - build["consistency"]) * 0.6 + build["build_gap_risk"] * 0.2, 0.0, 1.0))

    target[0] = float(np.clip(frontload_need * float(source_profile["damage"] > 0.0), 0.0, 1.0))
    target[1] = float(np.clip(defense_need * float(source_profile["block"] > 0.0), 0.0, 1.0))
    target[2] = float(np.clip(draw_need * max(float(source_profile["draw"] > 0.0), _source_supports_draw_cycle(source, signature)), 0.0, 1.0))
    target[3] = float(np.clip(energy_need * max(float(source_profile["energy"] > 0.0), float(source_profile["zero_cost"] > 0.0), float(signature.get("is_x_cost"))), 0.0, 1.0))
    target[4] = float(np.clip(scaling_need * max(float(source_profile["power"] > 0.0), float(source_profile["attack"] > 0.0 and source_profile["damage"] >= 12.0), greed_upgrade_mode), 0.0, 1.0))
    target[5] = float(np.clip(cycle_need * max(_source_supports_discard_or_exhaust(source), _source_supports_draw_cycle(source, signature)), 0.0, 1.0))

    gold = _player_gold(prev_obs)
    price = _item_price(action)
    if family == "shop":
        target[6] = float(np.clip((1.0 if price <= 0.0 else gold / max(price, 1.0)) * (0.5 + 0.5 * resource_priority), 0.0, 1.0))
    else:
        target[6] = 1.0 if family in {"card_reward", "reward", "deck_upgrade", "treasure_relic"} else 0.0

    same_count = _same_card_count(prev_obs, source)
    target[7] = float(np.clip(1.0 - min(same_count / 4.0, 1.0), 0.0, 1.0))
    if family == "deck_upgrade":
        target[4] = max(target[4], 0.55 + 0.35 * greed_upgrade_mode)
        target[6] = max(target[6], 0.95)
    return target


def compute_selection_targets(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    *,
    planner_context: dict[str, Any] | None = None,
) -> np.ndarray:
    target = np.zeros(NUM_SELECTION_HEADS, dtype=np.float32)
    if not isinstance(action, dict):
        return target

    signature = semantic_action_signature(action)
    if str(signature.get("family") or "") != "card_selection":
        return target

    objective = _planner_objective_vector(planner_context)
    build_priority = float(objective[2]) if objective.size > 2 else 0.5
    resource_priority = float(objective[3]) if objective.size > 3 else 0.5
    greed_upgrade_mode = float(objective[7]) if objective.size > 7 else 0.5

    selection_text = _selection_text(action, signature).lower()
    source_zone = _selection_source_zone(prev_obs, action, signature)
    zone_index = {
        "hand": 0,
        "draw": 1,
        "discard": 2,
        "exhaust": 3,
        "deck": 4,
        "play": 5,
    }.get(source_zone)
    if zone_index is not None:
        target[zone_index] = 1.0

    has_upgrade_preview = isinstance(action.get("upgrade_preview"), dict)
    upgrade_flag = has_upgrade_preview or any(token in selection_text for token in ("upgrade", "smith"))
    transform_flag = any(token in selection_text for token in ("transform", "mutate", "change"))
    remove_flag = any(token in selection_text for token in ("remove", "purge"))
    discard_exhaust_flag = any(token in selection_text for token in ("discard", "exhaust", "consume"))
    reward_discover_flag = any(token in selection_text for token in ("reward", "discover", "draft", "obtain", "gain"))

    target[6] = float(np.clip(float(upgrade_flag) * (0.65 + 0.35 * greed_upgrade_mode), 0.0, 1.0))
    target[7] = float(np.clip(float(transform_flag) * (0.70 + 0.20 * build_priority + 0.10 * resource_priority), 0.0, 1.0))
    target[8] = float(np.clip(float(remove_flag) * (0.70 + 0.30 * resource_priority), 0.0, 1.0))
    target[9] = float(np.clip(float(discard_exhaust_flag) * (0.75 + 0.25 * float(source_zone in {"discard", "exhaust", "play"})), 0.0, 1.0))
    target[10] = float(np.clip(float(reward_discover_flag) * (0.60 + 0.25 * build_priority + 0.15 * resource_priority), 0.0, 1.0))

    combat_runtime_line = 0.0
    if _in_combat(prev_obs):
        source = _action_source(action)
        draw_count = len(_runtime_cards(prev_obs, "draw_pile"))
        discard_count = len(_runtime_cards(prev_obs, "discard_pile"))
        exhaust_count = len(_runtime_cards(prev_obs, "exhaust_pile"))
        play_count = len(_runtime_cards(prev_obs, "play_pile"))
        combat_runtime_line += 0.25
        combat_runtime_line += 0.20 * float(source_zone in {"draw", "discard", "exhaust", "play"})
        combat_runtime_line += 0.20 * max(_source_supports_draw_cycle(source, signature), _source_supports_discard_or_exhaust(source))
        combat_runtime_line += 0.10 * float(draw_count > 0)
        combat_runtime_line += 0.10 * float(discard_count > 0 or exhaust_count > 0 or play_count > 0)
        combat_runtime_line += 0.05 * float(bool(_nonempty_potions(prev_obs)))
        combat_runtime_line += 0.05 * float(bool(_player_relics(prev_obs)))
        combat_runtime_line += 0.10 * float(discard_exhaust_flag)
        combat_runtime_line += 0.05 * float(target[1] > 0.0 or target[2] > 0.0 or target[3] > 0.0)
    target[11] = float(np.clip(combat_runtime_line, 0.0, 1.0))
    return target


def _norm_step(value: Any, default_far: float = 1.0) -> float:
    raw = _float(value, -1.0)
    if raw < 0.0:
        return default_far
    return float(np.clip(raw / 10.0, 0.0, 1.0))


def compute_route_targets(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    *,
    planner_context: dict[str, Any] | None = None,
) -> np.ndarray:
    target = np.zeros(NUM_ROUTE_HEADS, dtype=np.float32)
    if not isinstance(action, dict):
        return target
    summary = action.get("route_summary") if isinstance(action.get("route_summary"), dict) else None
    if not isinstance(summary, dict):
        return target

    objective = _planner_objective_vector(planner_context)
    safe_route_bias = float(objective[10]) if objective.size > 10 else 0.5
    shop_value_bias = float(objective[11]) if objective.size > 11 else 0.5
    rest_value_bias = float(objective[12]) if objective.size > 12 else 0.5
    elite_pressure = float(objective[8]) if objective.size > 8 else 0.5

    hp_ratio = min(_player_hp(prev_obs) / _player_max_hp(prev_obs), 1.0)
    gold_norm = min(np.log1p(max(_player_gold(prev_obs), 0.0)) / np.log1p(500.0), 1.0)
    elite_count = min(_float(summary.get("count_elite")) / 5.0, 1.0)
    rest_count = min(_float(summary.get("count_rest_site")) / 5.0, 1.0)
    shop_count = min(_float(summary.get("count_shop")) / 5.0, 1.0)
    event_count = min(_float(summary.get("count_event")) / 10.0, 1.0)
    question_count = min(_float(summary.get("count_question_mark")) / 10.0, 1.0)
    treasure_count = min(_float(summary.get("count_treasure")) / 5.0, 1.0)
    branch_count = min(_float(summary.get("direct_child_count")) / 4.0, 1.0)
    reach_count = min(_float(summary.get("reachable_node_count")) / 30.0, 1.0)

    next_elite = _norm_step(summary.get("next_elite_steps"))
    next_rest = _norm_step(summary.get("next_rest_steps"))
    next_shop = _norm_step(summary.get("next_shop_steps"))
    next_event = _norm_step(summary.get("next_event_steps"))
    next_treasure = _norm_step(summary.get("next_treasure_steps"))

    can_rest_before_elite = float(bool(summary.get("can_reach_rest_site_before_elite")))
    can_elite_then_rest = float(bool(summary.get("can_reach_elite_then_rest_site")))

    target[0] = float(np.clip(safe_route_bias * (0.45 * (1.0 - elite_count) + 0.25 * rest_count + 0.15 * (1.0 - next_elite) + 0.15 * hp_ratio), 0.0, 1.0))
    target[1] = float(np.clip((1.0 - safe_route_bias) * (0.45 * elite_count + 0.25 * (1.0 - next_elite) + 0.15 * can_rest_before_elite + 0.15 * can_elite_then_rest) + 0.15 * elite_pressure, 0.0, 1.0))
    target[2] = float(np.clip(rest_value_bias * (0.50 * rest_count + 0.30 * (1.0 - next_rest) + 0.20 * (1.0 - hp_ratio)), 0.0, 1.0))
    target[3] = float(np.clip(shop_value_bias * (0.45 * shop_count + 0.30 * (1.0 - next_shop) + 0.25 * gold_norm), 0.0, 1.0))
    target[4] = float(np.clip(0.55 * event_count + 0.45 * question_count + 0.15 * (1.0 - next_event), 0.0, 1.0))
    target[5] = float(np.clip(0.60 * treasure_count + 0.40 * (1.0 - next_treasure), 0.0, 1.0))
    target[6] = float(np.clip(0.60 * branch_count + 0.40 * reach_count, 0.0, 1.0))
    target[7] = float(
        np.clip(
            0.28 * target[0] + 0.14 * target[1] + 0.14 * target[2] + 0.14 * target[3] + 0.10 * target[4] + 0.08 * target[5] + 0.12 * target[6],
            0.0,
            1.0,
        )
    )
    return target


def compute_transition_targets(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    next_obs: dict[str, Any] | None,
) -> np.ndarray:
    """Combat-specific next-state summary for action-conditioned auxiliary loss."""
    del action  # transition summary is action-conditioned via the chosen rollout action.
    target = np.zeros(NUM_TRANSITION_HEADS, dtype=np.float32)
    target[0] = min(_player_hp(next_obs) / _player_max_hp(next_obs), 1.0)
    target[1] = min(_player_block(next_obs) / 60.0, 1.0)
    target[2] = min(_combat_energy(next_obs) / _combat_max_energy(next_obs), 1.0) if _in_combat(next_obs) else 0.0

    prev_enemy_hp = max(_combat_enemy_total_hp(prev_obs), _combat_enemy_total_hp(next_obs), 1.0)
    target[3] = min(_combat_enemy_total_hp(next_obs) / prev_enemy_hp, 1.0)
    target[4] = min(_combat_total_intent_damage(next_obs) / 200.0, 1.0)
    target[5] = min(len(_runtime_cards(next_obs, "draw_pile")) / 30.0, 1.0)
    target[6] = min(len(_runtime_cards(next_obs, "discard_pile")) / 30.0, 1.0)
    target[7] = min(len(_runtime_cards(next_obs, "exhaust_pile")) / 30.0, 1.0)
    return target


def compute_trait_targets(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    *,
    legal_actions_before: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Action/context affordance flags aligned with the candidate cross-attention path."""
    signature = semantic_action_signature(action)
    family = str(signature.get("family") or "")
    domain = str(signature.get("domain") or "")
    source = _action_source(action)
    source_profile = _source_profile(source)

    combat = _combat_state(prev_obs)
    current_energy = _float(combat.get("energy"))
    draw_count = len(_runtime_cards(prev_obs, "draw_pile"))
    discard_count = len(_runtime_cards(prev_obs, "discard_pile"))
    exhaust_count = len(_runtime_cards(prev_obs, "exhaust_pile"))
    potions = _nonempty_potions(prev_obs)

    target = np.zeros(NUM_TRAIT_HEADS, dtype=np.float32)

    energy_line = 0.0
    energy_line += float(family in _COMBAT_ACTION_FAMILIES) * 0.10
    energy_line += float(bool(signature.get("is_x_cost"))) * 0.80
    energy_line += float(source_profile["cost"] > 0.0) * 0.45
    energy_line += float(source_profile["energy"] > 0.0) * 0.30
    energy_line += float(current_energy > 0.0 and current_energy <= max(source_profile["cost"], 1.0)) * 0.20
    target[0] = float(np.clip(energy_line, 0.0, 1.0))

    potion_line = 0.0
    potion_line += float(family in {"use_potion", "discard_potion"}) * 1.00
    potion_line += float(bool(potions) and family in _COMBAT_ACTION_FAMILIES) * 0.30
    potion_line += float(bool(potions) and (source_profile["damage"] > 0.0 or source_profile["block"] > 0.0 or source_profile["draw"] > 0.0 or bool(signature.get("is_x_cost")))) * 0.35
    if isinstance(legal_actions_before, list):
        potion_line += 0.20 * float(any(str((candidate or {}).get("kind") or "") == "use_potion" for candidate in legal_actions_before if isinstance(candidate, dict)))
    target[1] = float(np.clip(potion_line, 0.0, 1.0))

    target[2] = float(np.clip(_relic_line_score(prev_obs, action, signature), 0.0, 1.0))

    enemy_risk = 0.0
    if domain == "combat":
        if signature.get("target_scope") in {"single_enemy", "all_enemies"} or signature.get("is_attack"):
            if signature.get("target_scope") == "single_enemy":
                enemy_risk = _enemy_risk_score(_choose_target_enemy(prev_obs, action))
            else:
                enemy_risk = max((_enemy_risk_score(enemy) for enemy in _combat_enemies(prev_obs)), default=0.0)
        else:
            enemy_risk = max((_enemy_risk_score(enemy) for enemy in _combat_enemies(prev_obs)), default=0.0) * 0.50
    target[3] = float(np.clip(enemy_risk, 0.0, 1.0))

    draw_cycle = 0.0
    draw_cycle += 0.70 * _source_supports_draw_cycle(source, signature)
    draw_cycle += 0.30 * float(domain == "combat" and draw_count <= 3 and discard_count >= 4)
    draw_cycle += 0.15 * float(domain == "build" and _source_supports_draw_cycle(source, signature) > 0.0)
    target[4] = float(np.clip(draw_cycle, 0.0, 1.0))

    discard_exhaust = 0.0
    discard_exhaust += 0.75 * _source_supports_discard_or_exhaust(source)
    discard_exhaust += 0.40 * float(family == "discard_potion")
    discard_exhaust += 0.20 * float(domain == "combat" and (discard_count > 0 or exhaust_count > 0))
    target[5] = float(np.clip(discard_exhaust, 0.0, 1.0))

    target[6] = float(family in _BUILD_ACTION_FAMILIES or (domain == "build" and family not in {"startup", "proceed"}))
    target[7] = float(family in _ROUTE_ACTION_FAMILIES or domain == "route")
    return target


def build_aux_targets(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    next_obs: dict[str, Any] | None,
    *,
    prev_planner_context: dict[str, Any] | None = None,
    next_planner_context: dict[str, Any] | None = None,
    terminated: bool = False,
    truncated: bool = False,
    legal_actions_before: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the aux-target payload attached to environment ``info`` dicts."""
    signature = semantic_action_signature(action)
    family = str(signature.get("family") or "")
    domain = str(signature.get("domain") or "")
    objective = compute_transition_objective_rewards(
        prev_obs,
        action,
        next_obs,
        prev_planner_context=prev_planner_context,
        next_planner_context=next_planner_context,
        terminated=terminated,
        truncated=truncated,
    ).astype(np.float32, copy=False)
    transition_mask = 1.0 if (_in_combat(prev_obs) or _in_combat(next_obs) or family in _COMBAT_ACTION_FAMILIES) else 0.0
    transition = compute_transition_targets(prev_obs, action, next_obs)
    traits = compute_trait_targets(prev_obs, action, legal_actions_before=legal_actions_before)
    build_mask = 1.0 if (domain == "build" and family not in {"startup", "proceed", "rest"}) else 0.0
    selection_mask = 1.0 if domain == "selection" and family == "card_selection" else 0.0
    route_mask = 1.0 if family in _ROUTE_ACTION_FAMILIES else 0.0
    build_targets = compute_build_targets(prev_obs, action, planner_context=prev_planner_context)
    selection_targets = compute_selection_targets(prev_obs, action, planner_context=prev_planner_context)
    route_targets = compute_route_targets(prev_obs, action, planner_context=prev_planner_context)
    return {
        "objective": objective,
        "objective_mask": 1.0,
        "transition": transition,
        "transition_mask": transition_mask,
        "traits": traits,
        "traits_mask": 1.0,
        "build": build_targets,
        "build_mask": build_mask,
        "selection": selection_targets,
        "selection_mask": selection_mask,
        "route": route_targets,
        "route_mask": route_mask,
        "objective_names": OBJECTIVE_HEAD_NAMES,
        "transition_names": TRANSITION_HEAD_NAMES,
        "trait_names": TRAIT_HEAD_NAMES,
        "build_names": BUILD_HEAD_NAMES,
        "selection_names": SELECTION_HEAD_NAMES,
        "route_names": ROUTE_HEAD_NAMES,
        "version": 2,
    }


__all__ = [
    "OBJECTIVE_HEAD_NAMES",
    "NUM_OBJECTIVE_HEADS",
    "TRANSITION_HEAD_NAMES",
    "NUM_TRANSITION_HEADS",
    "TRAIT_HEAD_NAMES",
    "NUM_TRAIT_HEADS",
    "BUILD_HEAD_NAMES",
    "NUM_BUILD_HEADS",
    "SELECTION_HEAD_NAMES",
    "NUM_SELECTION_HEADS",
    "ROUTE_HEAD_NAMES",
    "NUM_ROUTE_HEADS",
    "build_aux_targets",
    "compute_build_targets",
    "compute_selection_targets",
    "compute_route_targets",
    "compute_transition_targets",
    "compute_trait_targets",
]
