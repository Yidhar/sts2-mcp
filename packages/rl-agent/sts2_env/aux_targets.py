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
from .card_identity import card_identity
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

ENEMY_STATE_FIELD_NAMES = (
    "next_hp_delta_ratio",
    "attributable_player_hp_loss_ratio",
    "alive_next",
)
NUM_ENEMY_STATE_FIELDS = len(ENEMY_STATE_FIELD_NAMES)
ENEMY_STATE_SLOT_COUNT = obs_common.MAX_ENEMIES

# Phase 8 Tier 2: per-step causality target. 8-d delta describing what
# the chosen action ACTUALLY did (damage/block/hp_loss/draw/energy/
# strength/dex/vuln). Self-supervised from the (prev_obs, next_obs)
# pair — no labeling needed. The aux head learns to predict this
# delta from the candidate token, conditioned on history/powers/enemy
# context.
from .action_history import (
    CAUSALITY_HEAD_NAMES,
    NUM_CAUSALITY_HEADS,
    _build_causality_delta,
)


# Phase 3 TASK-D3: future-world card-lifecycle targets.  Self-supervised from
# (prev_obs, action, next_obs) tuples — no labels needed.  The aux head learns
# to predict next-state pile/hand/energy/block/incoming-damage and the
# action-conditioned card-destination probabilities so the search-free planner
# can perform approximate lookahead even when MCTS is disabled.
FUTURE_LIFECYCLE_HEAD_NAMES = (
    "next_hand_count_ratio",
    "next_draw_count_ratio",
    "next_discard_count_ratio",
    "next_exhaust_count_ratio",
    "next_energy_ratio",
    "next_block_ratio",
    "next_incoming_damage_ratio",
    "card_moved_to_exhaust_prob",
    "card_moved_to_discard_prob",
    "card_retained_prob",
    "hand_upgraded_count",
    "hand_transformed_count",
    "hand_copied_count",
    "cost_reduced_count",
    "created_card_count",
    "drawn_card_count",
    "next_kaiser_facing",
    "next_back_attack_risk",
    "next_ceremonial_lock_state",
)
NUM_FUTURE_LIFECYCLE_HEADS = len(FUTURE_LIFECYCLE_HEAD_NAMES)


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


def _self_inflicted_hp_loss_cumulative(obs: dict[str, Any] | None) -> float:
    """Bridge-side cumulative counter; see combat_memory._self_inflicted_hp_loss_cumulative."""
    return _float(_combat_state(obs).get("self_inflicted_hp_loss_cumulative"))


def _player_hp(obs: dict[str, Any] | None) -> float:
    player = _player_state(obs)
    return obs_common._player_hp_value(player)


def _player_max_hp(obs: dict[str, Any] | None) -> float:
    player = _player_state(obs)
    return obs_common._player_max_hp_value(player)


def _player_hp_ratio(obs: dict[str, Any] | None) -> float:
    player = _player_state(obs)
    return obs_common._player_hp_triplet(player)[2]


def _player_max_hp_denominator(*obs_values: dict[str, Any] | None) -> float:
    """Safe normalizer for damage attribution; never use suspicious max_hp=1."""
    best_max = 0.0
    best_hp = 0.0
    for obs in obs_values:
        player = _player_state(obs)
        hp, max_hp, _ratio = obs_common._player_hp_triplet(player)
        best_hp = max(best_hp, hp)
        if max_hp > 1.0:
            best_max = max(best_max, max_hp)
    return max(best_max, best_hp if best_hp > 1.0 else 0.0, 1.0)


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


def _action_numeric_value(
    action: dict[str, Any] | None,
    source: dict[str, Any] | None,
    signature: dict[str, Any] | None,
    keys: tuple[str, ...],
) -> float:
    """Read an action/potion/card numeric preview from all bridge variants.

    The live bridge and the simulator have evolved through several payload
    shapes.  Potion timing supervision must not depend on one exact key path,
    otherwise the trait head silently regresses into "has potion => use it".
    """

    alias_map = {
        "damage": ("damage", "total_damage", "preview_damage"),
        "block": ("block", "total_block", "preview_block"),
        "heal": ("heal", "hp_gain", "healing"),
        "draw": ("draw", "cards_drawn", "card_draw"),
        "energy": ("energy", "energy_gain", "gain_energy", "energy_delta"),
        "weak": ("weak", "apply_weak"),
        "vulnerable": ("vulnerable", "apply_vulnerable"),
        "poison": ("poison", "apply_poison"),
        "hits": ("hits", "hit_count"),
    }
    expanded_keys: list[str] = []
    for key in keys:
        expanded_keys.extend(alias_map.get(key, (key,)))

    values: list[float] = []
    signature = signature if isinstance(signature, dict) else {}
    for key in expanded_keys:
        if key in signature:
            value = _float(signature.get(key), float("nan"))
            if not np.isnan(value):
                values.append(value)

    candidates: list[Any] = []
    if isinstance(action, dict):
        candidates.extend(
            [
                action,
                action.get("semantic") if isinstance(action.get("semantic"), dict) else None,
                action.get("preview") if isinstance(action.get("preview"), dict) else None,
                action.get("effect_preview") if isinstance(action.get("effect_preview"), dict) else None,
                action.get("card") if isinstance(action.get("card"), dict) else None,
                action.get("potion") if isinstance(action.get("potion"), dict) else None,
            ]
        )
    if isinstance(source, dict):
        candidates.extend(
            [
                source,
                source.get("preview") if isinstance(source.get("preview"), dict) else None,
                source.get("effect_preview") if isinstance(source.get("effect_preview"), dict) else None,
            ]
        )

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in expanded_keys:
            value = candidate.get(key)
            if value is None:
                continue
            numeric = _float(value, float("nan"))
            if not np.isnan(numeric):
                values.append(numeric)
    return max(values) if values else 0.0


def _alive_enemy_hp_values(obs: dict[str, Any] | None) -> list[float]:
    values: list[float] = []
    for enemy in _combat_enemies(obs):
        hp = _float(enemy.get("hp", enemy.get("current_hp")))
        if hp > 0.0:
            values.append(hp)
    return values


def _target_enemy_hp(obs: dict[str, Any] | None, action: dict[str, Any] | None) -> float:
    enemy = _choose_target_enemy(obs, action)
    if isinstance(enemy, dict):
        hp = _float(enemy.get("hp", enemy.get("current_hp")))
        if hp > 0.0:
            return hp
    values = _alive_enemy_hp_values(obs)
    return min(values) if values else 0.0


def _action_energy_cost(action: dict[str, Any] | None, signature: dict[str, Any] | None, source_profile: dict[str, float] | None = None) -> float:
    signature = signature if isinstance(signature, dict) else {}
    profile = source_profile if isinstance(source_profile, dict) else {}
    if bool(signature.get("is_x_cost")) or float(profile.get("x_cost", 0.0)) > 0.5:
        return 999.0
    if "cost" in signature:
        cost = _float(signature.get("cost"), -1.0)
        if cost >= 0.0:
            return cost
    source = _action_source(action)
    if isinstance(source, dict):
        cost = _float(source.get("cost"), -1.0)
        if cost >= 0.0:
            return cost
    return max(float(profile.get("cost", 0.0)), 0.0)


def _action_positive_preview(action: dict[str, Any] | None, signature: dict[str, Any] | None = None) -> bool:
    if not isinstance(action, dict):
        return False
    signature = signature if isinstance(signature, dict) else semantic_action_signature(action)
    family = str(signature.get("family") or "")
    if family != "play_card":
        return False
    source = _action_source(action)
    profile = _source_profile(source)
    roles = {str(role or "").lower() for role in (signature.get("roles") or []) if str(role or "").strip()}
    damage = max(float(profile.get("damage", 0.0)), _action_numeric_value(action, source, signature, ("damage",)))
    block = max(float(profile.get("block", 0.0)), _action_numeric_value(action, source, signature, ("block",)))
    draw = max(float(profile.get("draw", 0.0)), _action_numeric_value(action, source, signature, ("draw",)))
    energy = max(float(profile.get("energy", 0.0)), _action_numeric_value(action, source, signature, ("energy",)))
    return bool(
        damage > 0.0
        or block > 0.0
        or draw > 0.0
        or energy > 0.0
        or roles.intersection({"attack", "block", "draw", "debuff", "buff", "heal", "setup", "scaling", "resource"})
    )


def _has_resource_followup_for_potion(
    legal_actions_before: list[dict[str, Any]] | None,
    action: dict[str, Any] | None,
    energy_after: float,
) -> bool:
    """Whether a resource/draw/energy potion can be converted this turn.

    This is deliberately conservative: if the only exposed follow-up is another
    potion/end-turn, a resource potion should be considered deferable instead of
    automatically positive.
    """

    if not isinstance(legal_actions_before, list):
        return False
    action_id = str((action or {}).get("action_id") or "")
    for candidate in legal_actions_before:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("action_id") or "")
        if action_id and candidate_id and candidate_id == action_id:
            continue
        candidate_sig = semantic_action_signature(candidate)
        family = str(candidate_sig.get("family") or "")
        if family != "play_card":
            continue
        candidate_source = _action_source(candidate)
        candidate_profile = _source_profile(candidate_source)
        if bool(candidate_sig.get("is_x_cost")):
            if energy_after <= 0.05:
                continue
        elif _action_energy_cost(candidate, candidate_sig, candidate_profile) > max(energy_after, 0.0) + 1e-3:
            continue
        if _action_positive_preview(candidate, candidate_sig):
            return True
    return False


def _potion_timing_line_score(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    signature: dict[str, Any],
    source_profile: dict[str, float],
    legal_actions_before: list[dict[str, Any]] | None,
) -> float:
    """Trait supervision for *when* to use a potion, not merely that one exists.

    Head index 1 remains named ``potion_line`` for checkpoint compatibility, but
    its target is now "potion timing quality":

    - high when a selected potion is lethal, prevents lethal/major HP loss, or
      answers a visible dangerous mechanic;
    - moderate when it creates resources that can immediately be converted;
    - low when using it is overkill, block waste, or resource-without-follow-up;
    - tiny context signal for non-potion combat actions while potions are held.
    """

    family = str(signature.get("family") or "")
    if not _in_combat(prev_obs):
        return 0.08 if family in {"use_potion", "discard_potion"} else 0.0

    potions = _nonempty_potions(prev_obs)
    if family not in {"use_potion", "discard_potion"}:
        if potions and family in _COMBAT_ACTION_FAMILIES:
            # Keep only a faint context affordance.  The old 0.30-0.65 target was
            # enough to make the candidate path equate "has potion" with
            # "potion should be used now".
            return 0.04 + 0.04 * float(_action_positive_preview(action, signature))
        return 0.0

    if family == "discard_potion":
        return 0.06 + 0.04 * float(bool(potions))

    source = _action_source(action)
    roles = {str(role or "").lower() for role in (signature.get("roles") or []) if str(role or "").strip()}
    text = _source_text(source)
    incoming = _combat_total_intent_damage(prev_obs)
    current_block = _player_block(prev_obs)
    hp = _player_hp(prev_obs)
    hp_ratio = _player_hp_ratio(prev_obs)
    max_hp = _player_max_hp_denominator(prev_obs)
    threat_gap = max(incoming - current_block, 0.0)
    target_hp = _target_enemy_hp(prev_obs, action)

    damage = max(float(source_profile.get("damage", 0.0)), _action_numeric_value(action, source, signature, ("damage",)))
    block = max(float(source_profile.get("block", 0.0)), _action_numeric_value(action, source, signature, ("block",)))
    draw = max(float(source_profile.get("draw", 0.0)), _action_numeric_value(action, source, signature, ("draw",)))
    energy_gain = max(float(source_profile.get("energy", 0.0)), _action_numeric_value(action, source, signature, ("energy",)))
    heal = max(float(source_profile.get("heal", 0.0)), _action_numeric_value(action, source, signature, ("heal",)))
    weak = max(float(source_profile.get("weak", 0.0)), _action_numeric_value(action, source, signature, ("weak",)))
    vulnerable = max(float(source_profile.get("vulnerable", 0.0)), _action_numeric_value(action, source, signature, ("vulnerable",)))
    poison = _action_numeric_value(action, source, signature, ("poison",))
    debuff = bool(roles.intersection({"debuff", "weak", "vulnerable"}) or weak > 0.0 or vulnerable > 0.0 or poison > 0.0)
    resource_potion = bool(draw > 0.0 or energy_gain > 0.0 or "energy" in text or "draw" in text or roles.intersection({"draw", "resource"}))
    current_energy = _combat_energy(prev_obs)
    followup_available = _has_resource_followup_for_potion(legal_actions_before, action, current_energy + max(energy_gain, 0.0))

    lethal = bool(damage > 0.0 and target_hp > 0.0 and damage >= target_hp)
    high_damage = bool(
        damage > 0.0
        and (
            (target_hp > 0.0 and damage >= min(target_hp, max(12.0, 0.35 * target_hp)))
            or (target_hp <= 0.0 and damage >= 18.0)
        )
    )
    prevent_lethal = bool(threat_gap >= max(hp, 1.0) and (block + heal >= min(max(threat_gap, 1.0), 24.0) or debuff or lethal))
    prevent_major_loss = bool(
        threat_gap >= max(8.0, 0.24 * max(hp, 1.0))
        and (block + heal >= min(threat_gap, 18.0) * 0.45 or debuff or high_damage)
    )
    any_enemy_risk = max((_enemy_risk_score(enemy) for enemy in _combat_enemies(prev_obs)), default=0.0)
    mechanism_answer = bool(any_enemy_risk >= 0.55 and (lethal or high_damage or debuff or block > 0.0 or heal > 0.0))
    overkill = bool(damage > 0.0 and target_hp > 0.0 and damage > max(target_hp + 8.0, target_hp * 1.75) and not roles.intersection({"aoe"}))
    block_waste = bool(block > 0.0 and threat_gap <= 1.0 and hp_ratio >= 0.45)
    no_followup = bool(resource_potion and not followup_available)
    save_recommended = bool(
        not lethal
        and not prevent_lethal
        and not mechanism_answer
        and threat_gap <= max(2.0, 0.08 * max_hp)
        and hp_ratio >= 0.62
    )

    score = 0.14
    score += 0.72 * float(lethal)
    score += 0.72 * float(prevent_lethal)
    score += 0.36 * float(prevent_major_loss)
    score += 0.28 * float(high_damage and not overkill)
    score += 0.32 * float(block > 0.0 and threat_gap > 1.0 and not block_waste)
    score += 0.18 * float(heal > 0.0 and hp_ratio < 0.75)
    score += 0.25 * float(debuff and incoming > 0.0)
    score += 0.35 * float(resource_potion and followup_available)
    score += 0.24 * float(mechanism_answer)

    score -= 0.34 * float(overkill)
    score -= 0.36 * float(block_waste)
    score -= 0.48 * float(no_followup)
    score -= 0.28 * float(save_recommended)
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

    hp_ratio = _player_hp_ratio(prev_obs)
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


def _card_identity_key(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    for key in ("uid", "instance_id", "combat_uuid", "id"):
        value = card.get(key)
        if value not in (None, ""):
            return str(value)
    title = str(card.get("title") or card.get("name") or "").strip()
    return title


def _is_upgraded(card: dict[str, Any] | None) -> bool:
    if not isinstance(card, dict):
        return False
    for key in ("is_upgraded", "upgraded"):
        if card.get(key):
            return True
    for key in ("upgrade_level", "current_upgrade_level", "level"):
        try:
            if float(card.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
    title = str(card.get("title") or card.get("name") or "").strip()
    return title.endswith("+")


def _hand_upgrade_count(obs: dict[str, Any] | None) -> int:
    return sum(1 for card in _runtime_cards(obs, "hand") if _is_upgraded(card))


def _all_pile_cards(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pile_name in ("hand", "draw_pile", "discard_pile"):
        out.extend(_runtime_cards(obs, pile_name))
    return out


def _played_card_payload(action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    card = action.get("card")
    return card if isinstance(card, dict) else {}


def _first_enemy_field(obs: dict[str, Any] | None, *fields: str) -> float:
    enemies = _combat_enemies(obs)
    if not enemies:
        return 0.0
    enemy = enemies[0]
    cursor: Any = enemy
    for field in fields:
        if not isinstance(cursor, dict):
            return 0.0
        cursor = cursor.get(field)
    try:
        return float(cursor or 0)
    except (TypeError, ValueError):
        return 0.0


def _enemy_facing_metric(obs: dict[str, Any] | None) -> float:
    """Normalized "looking at player" signal in [0, 1].

    The bridge surfaces ``facing`` either as a string (``"front"`` /
    ``"back"`` / ``"side"``) or a numeric ``facing_player`` boolean.  We
    collapse to a probability the first enemy is currently facing the player.
    """
    enemies = _combat_enemies(obs)
    if not enemies:
        return 0.0
    enemy = enemies[0]
    facing = enemy.get("facing")
    if isinstance(facing, str):
        token = facing.strip().lower()
        if token in {"player", "front"}:
            return 1.0
        if token in {"back", "away"}:
            return 0.0
        if token == "side":
            return 0.5
    if isinstance(facing, dict):
        towards_player = facing.get("towards_player")
        if isinstance(towards_player, bool):
            return 1.0 if towards_player else 0.0
        if isinstance(towards_player, (int, float)):
            return 1.0 if float(towards_player) > 0 else 0.0
    fp = enemy.get("facing_player")
    if isinstance(fp, bool):
        return 1.0 if fp else 0.0
    if isinstance(fp, (int, float)):
        return 1.0 if float(fp) > 0 else 0.0
    return 0.5


def _ceremonial_lock_metric(obs: dict[str, Any] | None) -> float:
    """1.0 if any enemy power name signals a Ceremonial one-card lock."""
    for enemy in _combat_enemies(obs):
        powers = enemy.get("powers") if isinstance(enemy, dict) else None
        if not isinstance(powers, list):
            continue
        for power in powers:
            if isinstance(power, dict):
                pid = str(power.get("id") or power.get("power_id") or power.get("name") or "").lower()
            else:
                pid = str(power or "").lower()
            if "ceremonial" in pid or "one_card_lock" in pid:
                return 1.0
    return 0.0


def compute_future_lifecycle_targets(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    next_obs: dict[str, Any] | None,
) -> np.ndarray:
    """19-d future-world target vector (TASK-D3).

    Self-supervised from observed transitions.  Pile / energy / block / damage
    fields are normalized to [0, 1] in the same fashion as
    :func:`compute_transition_targets`.  Card destination probabilities are
    Bernoulli targets in {0, 1} derived from the played card's identity:
    if the played card now appears in exhaust / discard / hand it scores 1.0
    on the matching head.
    """
    target = np.zeros(NUM_FUTURE_LIFECYCLE_HEADS, dtype=np.float32)

    # Counts (ratio-normalized to keep loss in a stable scale).
    target[0] = min(len(_runtime_cards(next_obs, "hand")) / 12.0, 1.0)
    target[1] = min(len(_runtime_cards(next_obs, "draw_pile")) / 30.0, 1.0)
    target[2] = min(len(_runtime_cards(next_obs, "discard_pile")) / 30.0, 1.0)
    target[3] = min(len(_runtime_cards(next_obs, "exhaust_pile")) / 30.0, 1.0)
    if _in_combat(next_obs):
        target[4] = min(_combat_energy(next_obs) / max(_combat_max_energy(next_obs), 1.0), 1.0)
    target[5] = min(_player_block(next_obs) / 60.0, 1.0)
    target[6] = min(_combat_total_intent_damage(next_obs) / 200.0, 1.0)

    # Card destination probabilities — only meaningful for play_card actions.
    # P0-6: only emit hard Bernoulli targets when the played card carries a
    # stable runtime instance UUID.  Without it, ``id``/``title`` collisions
    # (two copies of Strike, transform-renamed cards, etc.) produce false-
    # confidence targets that train the future-world head on noise.
    played = _played_card_payload(action)
    played_id = card_identity(played)
    if played_id.get("confidence") == "runtime_internal":
        played_key = played_id.get("key", "")
        in_next_hand = any(card_identity(c).get("key") == played_key for c in _runtime_cards(next_obs, "hand"))
        in_next_discard = any(card_identity(c).get("key") == played_key for c in _runtime_cards(next_obs, "discard_pile"))
        in_next_exhaust = any(card_identity(c).get("key") == played_key for c in _runtime_cards(next_obs, "exhaust_pile"))
        target[7] = 1.0 if in_next_exhaust else 0.0
        target[8] = 1.0 if in_next_discard else 0.0
        target[9] = 1.0 if in_next_hand else 0.0
    # else: targets remain zero — the head will not be supervised on this
    # transition's card destination, which is the correct behaviour when
    # identity is ambiguous.

    # Hand mutation counts — derived from before/after diffs across runtime piles.
    prev_upgrades = _hand_upgrade_count(prev_obs)
    next_upgrades = _hand_upgrade_count(next_obs)
    target[10] = float(max(0, next_upgrades - prev_upgrades))

    prev_titles = [str(c.get("title") or c.get("name") or "").strip() for c in _all_pile_cards(prev_obs)]
    next_titles = [str(c.get("title") or c.get("name") or "").strip() for c in _all_pile_cards(next_obs)]
    prev_unique = len(set(t for t in prev_titles if t))
    next_unique = len(set(t for t in next_titles if t))
    target[11] = float(max(0, abs(next_unique - prev_unique) - 1))  # transformations grow unique-titles minus the played card delta

    target[12] = float(max(0, len(next_titles) - len(prev_titles) - max(0, len(_runtime_cards(next_obs, "hand")) - len(_runtime_cards(prev_obs, "hand")))))
    target[13] = float(max(0, sum(
        1 for c in _runtime_cards(next_obs, "hand")
        if isinstance(c.get("current_cost"), (int, float))
        and isinstance(c.get("cost"), (int, float))
        and float(c["current_cost"]) < float(c["cost"])
    )))
    target[14] = float(max(0, len(next_titles) - len(prev_titles)))
    drawn_proxy = max(0, len(_runtime_cards(next_obs, "hand")) - (len(_runtime_cards(prev_obs, "hand")) - 1))
    target[15] = float(drawn_proxy)

    # Boss-mechanic next-state metrics.
    target[16] = _enemy_facing_metric(next_obs)
    # Back-attack risk: 1.0 if enemy now facing back AND has an attack intent next.
    facing_metric = _enemy_facing_metric(next_obs)
    intent_damage = _combat_total_intent_damage(next_obs)
    target[17] = 1.0 if (facing_metric < 0.5 and intent_damage > 0) else 0.0
    target[18] = _ceremonial_lock_metric(next_obs)

    return target


def compute_transition_targets(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    next_obs: dict[str, Any] | None,
) -> np.ndarray:
    """Combat-specific next-state summary for action-conditioned auxiliary loss."""
    del action  # transition summary is action-conditioned via the chosen rollout action.
    target = np.zeros(NUM_TRANSITION_HEADS, dtype=np.float32)
    target[0] = _player_hp_ratio(next_obs)
    target[1] = min(_player_block(next_obs) / 60.0, 1.0)
    target[2] = min(_combat_energy(next_obs) / _combat_max_energy(next_obs), 1.0) if _in_combat(next_obs) else 0.0

    prev_enemy_hp = max(_combat_enemy_total_hp(prev_obs), _combat_enemy_total_hp(next_obs), 1.0)
    target[3] = min(_combat_enemy_total_hp(next_obs) / prev_enemy_hp, 1.0)
    target[4] = min(_combat_total_intent_damage(next_obs) / 200.0, 1.0)
    target[5] = min(len(_runtime_cards(next_obs, "draw_pile")) / 30.0, 1.0)
    target[6] = min(len(_runtime_cards(next_obs, "discard_pile")) / 30.0, 1.0)
    target[7] = min(len(_runtime_cards(next_obs, "exhaust_pile")) / 30.0, 1.0)
    return target


def _enemy_identity_key(enemy: dict[str, Any] | None, fallback_index: int) -> str:
    if isinstance(enemy, dict):
        for key in ("combat_id", "id", "model_id", "name"):
            value = enemy.get(key)
            if value not in (None, ""):
                return str(value)
    return f"idx_{fallback_index}"


def compute_enemy_state_targets(
    prev_obs: dict[str, Any] | None,
    next_obs: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-enemy next-step prediction targets.

    Self-supervised — uses only prev/next observation diffs so the signal
    covers on-death bursts, enrage scaling, summons, and any unseen or
    modded mechanic without requiring hand labels.
    """
    targets = np.zeros(
        (ENEMY_STATE_SLOT_COUNT, NUM_ENEMY_STATE_FIELDS), dtype=np.float32
    )
    mask = np.zeros(ENEMY_STATE_SLOT_COUNT, dtype=np.float32)

    prev_enemies = _combat_enemies(prev_obs)
    next_enemies = _combat_enemies(next_obs)
    if not prev_enemies:
        return targets, mask

    next_by_key: dict[str, dict[str, Any]] = {}
    for index, enemy in enumerate(next_enemies):
        key = _enemy_identity_key(enemy, index)
        next_by_key[key] = enemy

    prev_player_hp = _player_hp(prev_obs)
    next_player_hp = _player_hp(next_obs)
    player_max_hp = _player_max_hp_denominator(prev_obs, next_obs)
    raw_player_hp_loss = max(prev_player_hp - next_player_hp, 0.0)

    # Strip self-inflicted HP loss (Offering / Bloodletting / etc) using the
    # bridge's cumulative counter so enemy attribution targets cleanly.
    prev_self_cum = _self_inflicted_hp_loss_cumulative(prev_obs)
    next_self_cum = _self_inflicted_hp_loss_cumulative(next_obs)
    self_inflicted_delta = max(next_self_cum - prev_self_cum, 0.0)
    actual_player_hp_loss = max(raw_player_hp_loss - self_inflicted_delta, 0.0)

    total_predicted = 0.0
    predicted_damages: list[float] = []
    for enemy in prev_enemies[:ENEMY_STATE_SLOT_COUNT]:
        intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
        if not isinstance(intent, dict):
            intent = {}
        predicted = _float(intent.get("total_damage"))
        predicted_damages.append(predicted)
        total_predicted += predicted

    for slot_index, enemy in enumerate(prev_enemies[:ENEMY_STATE_SLOT_COUNT]):
        prev_hp = _float(enemy.get("hp", enemy.get("current_hp")))
        prev_max_hp = max(_float(enemy.get("max_hp"), prev_hp), 1.0)
        key = _enemy_identity_key(enemy, slot_index)
        next_enemy = next_by_key.get(key)
        if next_enemy is not None:
            next_hp = _float(next_enemy.get("hp", next_enemy.get("current_hp")))
            alive_next = 1.0 if next_hp > 0.0 else 0.0
        else:
            next_hp = 0.0
            alive_next = 0.0
        delta = next_hp - prev_hp
        delta_ratio = delta / max(prev_max_hp, 1.0)
        if delta_ratio > 1.0:
            delta_ratio = 1.0
        elif delta_ratio < -1.0:
            delta_ratio = -1.0

        predicted = predicted_damages[slot_index]
        if total_predicted > 0.0 and actual_player_hp_loss > 0.0:
            attributable = predicted * (actual_player_hp_loss / total_predicted)
        else:
            attributable = 0.0
        attribution_ratio = min(attributable / player_max_hp, 1.0)

        targets[slot_index, 0] = delta_ratio
        targets[slot_index, 1] = attribution_ratio
        targets[slot_index, 2] = alive_next
        mask[slot_index] = 1.0 if prev_hp > 0.0 else 0.0

    return targets, mask


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

    target = np.zeros(NUM_TRAIT_HEADS, dtype=np.float32)

    energy_line = 0.0
    energy_line += float(family in _COMBAT_ACTION_FAMILIES) * 0.10
    energy_line += float(bool(signature.get("is_x_cost"))) * 0.80
    energy_line += float(source_profile["cost"] > 0.0) * 0.45
    energy_line += float(source_profile["energy"] > 0.0) * 0.30
    energy_line += float(current_energy > 0.0 and current_energy <= max(source_profile["cost"], 1.0)) * 0.20
    target[0] = float(np.clip(energy_line, 0.0, 1.0))

    target[1] = float(
        np.clip(
            _potion_timing_line_score(prev_obs, action, signature, source_profile, legal_actions_before),
            0.0,
            1.0,
        )
    )

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
    enemy_state_targets, enemy_state_mask = compute_enemy_state_targets(prev_obs, next_obs)
    # Phase 8 Tier 2 causality target. Single 8-d delta per step (the
    # actually-observed post-pre diff). The AuxMaskablePPO buffer
    # expands this into a (n_actions, 8) tensor with the vector placed
    # at the chosen-candidate row (action index) and a scalar mask=1;
    # all other rows zero and masked out.
    causality_delta = np.asarray(_build_causality_delta(prev_obs, next_obs), dtype=np.float32)
    future_lifecycle = compute_future_lifecycle_targets(prev_obs, action, next_obs)
    future_lifecycle_mask = 1.0 if (_in_combat(prev_obs) or _in_combat(next_obs) or family in _COMBAT_ACTION_FAMILIES) else 0.0
    # Mask the causality target in the same cases where no combat
    # advancement can happen (truncation with rebind, recovery step).
    # We still credit non-combat actions because their effect on the
    # state (e.g. rest-site heal → player_hp_post > pre → damage=0,
    # but self_hp_loss=0 and hp gain signals block the 3rd field) is
    # informative — the head can learn that map choice usually has
    # damage=block=hp_loss=0 and small draw/energy changes.
    causality_mask = 0.0 if (terminated and truncated) else 1.0
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
        "enemy_state": enemy_state_targets,
        "enemy_state_mask": enemy_state_mask,
        "causality": causality_delta,
        "causality_mask": float(causality_mask),
        "future_lifecycle": future_lifecycle,
        "future_lifecycle_mask": float(future_lifecycle_mask),
        "objective_names": OBJECTIVE_HEAD_NAMES,
        "transition_names": TRANSITION_HEAD_NAMES,
        "trait_names": TRAIT_HEAD_NAMES,
        "build_names": BUILD_HEAD_NAMES,
        "selection_names": SELECTION_HEAD_NAMES,
        "route_names": ROUTE_HEAD_NAMES,
        "enemy_state_field_names": ENEMY_STATE_FIELD_NAMES,
        "causality_names": CAUSALITY_HEAD_NAMES,
        "future_lifecycle_names": FUTURE_LIFECYCLE_HEAD_NAMES,
        "version": 5,
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
    "CAUSALITY_HEAD_NAMES",
    "NUM_CAUSALITY_HEADS",
    "FUTURE_LIFECYCLE_HEAD_NAMES",
    "NUM_FUTURE_LIFECYCLE_HEADS",
    "compute_future_lifecycle_targets",
    "build_aux_targets",
    "compute_build_targets",
    "compute_selection_targets",
    "compute_route_targets",
    "compute_transition_targets",
    "compute_trait_targets",
    "compute_enemy_state_targets",
    "ENEMY_STATE_FIELD_NAMES",
    "NUM_ENEMY_STATE_FIELDS",
    "ENEMY_STATE_SLOT_COUNT",
]
