"""Shared potion timing evaluator — Phase 4b of potion-timing-modeling-plan.md.

This module hosts a pure-function version of the potion timing logic that
both `combat_env.py` (for per-step reward shaping) and `muzero/train.py`
(for planner bias + metrics) can consume.  The trainer keeps its richer
`_potion_timing_profile()` for backwards compatibility — that path uses
self-bound helpers (action_metric, semantic_family, target_enemy_hp) that
this module deliberately replicates as plain helpers so the env can call
them without depending on the trainer instance.

Inputs:
    action            : the legal action dict (kind="use_potion" expected)
    raw_obs           : current bridge raw observation (with player/combat)
    legal_actions     : list of legal action dicts for follow-up checks
    mask              : numpy mask aligned with legal_actions (1 = legal)
    energy            : current player energy (float)

Output: dict with use_quality / waste_risk / save_value / lethal /
prevent_lethal / mechanism_answer / facing_change / overkill / block_waste
/ no_followup / hand_context_good / hand_context_bad / long_term_value /
passive_or_triggered / requires_followup / followup_available / aoe /
damage / block / heal / draw / energy_gain / weak / vulnerable / poison /
debuff / incoming / current_block / hp / threat_gap / target_hp /
positive / urgent / deferable / low_urgency / save_recommended
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .boss_mechanics import build_boss_mechanics_context
from .potion_profiles import DEFAULT_EFFECT_PROFILE, get_potion_profile


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _action_family(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    kind = str(action.get("kind") or "").strip()
    action_id = str(action.get("action_id") or "").strip()
    if kind == "play_card" or action_id.startswith("play_card:"):
        return "play_card"
    if kind in {"use_potion", "potion"} or action_id.startswith("use_potion:"):
        return "use_potion"
    if action_id == "end_turn":
        return "end_turn"
    return kind or ""


def _action_roles(action: dict[str, Any] | None) -> set[str]:
    if not isinstance(action, dict):
        return set()
    semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
    roles = semantic.get("roles") if isinstance(semantic, dict) else None
    if isinstance(roles, list):
        return {str(r).strip().lower() for r in roles if r}
    return set()


def _resolve_potion_effect(action: dict[str, Any]) -> dict[str, Any]:
    """Merge bridge live effect_profile with the Python registry (bridge wins)."""
    potion = action.get("potion") if isinstance(action.get("potion"), dict) else None
    pid = ""
    if potion:
        pid = str(potion.get("id") or "").strip()
    registry = get_potion_profile(pid) if pid else {}
    effect = dict(DEFAULT_EFFECT_PROFILE)
    effect.update(registry.get("effect_profile") or {})
    if potion and isinstance(potion.get("effect_profile"), dict):
        effect.update({k: v for k, v in potion["effect_profile"].items() if v is not None})

    def _pick(field: str) -> Any:
        if potion is not None and potion.get(field):
            return potion.get(field)
        return registry.get(field) or []

    return {
        "effect_profile": effect,
        "effect_family": list(_pick("effect_family") or []),
        "semantic_tags": list(_pick("semantic_tags") or []),
        "timing_tags": list(_pick("timing_tags") or []),
        "training_tags": list(_pick("training_tags") or []),
        "target_scope": str((potion.get("target_scope") if potion else None) or registry.get("target_scope") or ""),
        "potion_id": pid,
    }


def _incoming_damage_pressure(raw_obs: dict[str, Any] | None) -> tuple[float, float, float]:
    if not isinstance(raw_obs, dict):
        return (0.0, 0.0, 0.0)
    player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
    combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
    enemies = combat.get("enemies") if isinstance(combat, dict) else []
    incoming = 0.0
    if isinstance(enemies, list):
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
            incoming += _safe_float(intent.get("total_damage"))
    return (
        incoming,
        _safe_float(player.get("block")),
        _safe_float(player.get("hp", player.get("current_hp"))),
    )


def _target_enemy_hp(action: dict[str, Any], raw_obs: dict[str, Any] | None) -> float:
    """Best-effort: pull HP of the enemy this potion targets, else 0."""
    if not isinstance(raw_obs, dict):
        return 0.0
    target = action.get("target")
    target_id = ""
    if isinstance(target, dict):
        target_id = str(target.get("combat_id") or target.get("id") or "")
    elif isinstance(target, str):
        target_id = target
    combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
    enemies = combat.get("enemies") if isinstance(combat, dict) else []
    if isinstance(enemies, list):
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            cid = str(enemy.get("combat_id") or enemy.get("id") or "")
            if target_id and cid == target_id:
                return _safe_float(enemy.get("hp", enemy.get("current_hp")))
        # Fallback: lowest-hp enemy as a damage-potion target proxy.
        hps = [_safe_float(e.get("hp", e.get("current_hp"))) for e in enemies if isinstance(e, dict)]
        hps = [h for h in hps if h > 0.0]
        if hps:
            return min(hps)
    return 0.0


def _has_resource_followup(
    action_index: int,
    legal_actions: Iterable[Any] | None,
    mask: np.ndarray | None,
    energy_after: float,
) -> bool:
    if legal_actions is None:
        return False
    actions = list(legal_actions)
    for idx, other in enumerate(actions):
        if idx == action_index:
            continue
        if mask is not None and idx < mask.shape[0] and mask[idx] <= 0:
            continue
        if not isinstance(other, dict):
            continue
        if _action_family(other) != "play_card":
            continue
        cost_raw = other.get("card_cost")
        if cost_raw is None:
            card = other.get("card") if isinstance(other.get("card"), dict) else {}
            cost_raw = card.get("cost") if isinstance(card, dict) else 0
        try:
            cost = max(float(cost_raw or 0.0), 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        if cost > energy_after + 1e-6:
            continue
        roles = _action_roles(other)
        if roles.intersection({"attack", "block", "draw", "debuff", "weak", "vulnerable", "scaling", "power", "resource"}):
            return True
    return False


def compute_potion_timing(
    action: dict[str, Any],
    raw_obs: dict[str, Any] | None,
    legal_actions: Iterable[Any] | None,
    mask: np.ndarray | None,
    energy: float,
    encounter_tier: str = "normal",
) -> dict[str, Any]:
    """Pure-function potion timing evaluator.  See module docstring."""
    default = {
        "is_potion": False, "available": False,
        "use_quality": 0.0, "waste_risk": 0.0, "save_value": 0.0,
        "lethal": False, "prevent_lethal": False, "prevent_major_loss": False,
        "mechanism_answer": False, "facing_change": False,
        "overkill": False, "block_waste": False, "no_followup": False,
        "save_recommended": False, "low_urgency": False,
        "positive": False, "urgent": False, "deferable": False,
        "requires_followup": False, "followup_available": False,
        "hand_context_good": False, "hand_context_bad": False,
        "long_term_value": False, "passive_or_triggered": False,
        "aoe": False, "debuff": False,
        "damage": 0.0, "block": 0.0, "draw": 0.0, "energy_gain": 0.0,
        "heal": 0.0, "weak": 0.0, "vulnerable": 0.0, "poison": 0.0,
        "incoming": 0.0, "current_block": 0.0, "hp": 0.0,
        "threat_gap": 0.0, "target_hp": 0.0,
    }
    if not isinstance(action, dict) or _action_family(action) != "use_potion":
        return default

    merged = _resolve_potion_effect(action)
    eff = merged["effect_profile"]
    timing_tags = merged["timing_tags"]
    effect_family = merged["effect_family"]
    target_scope = merged["target_scope"]
    training_tags = merged["training_tags"]

    damage = _safe_float(eff.get("damage"))
    block = _safe_float(eff.get("block"))
    heal = _safe_float(eff.get("heal"))
    draw = _safe_float(eff.get("draw"))
    energy_gain = _safe_float(eff.get("energy_gain"))
    weak_v = _safe_float(eff.get("weak"))
    vuln_v = _safe_float(eff.get("vulnerable"))
    poison_v = _safe_float(eff.get("poison"))
    gen_card_v = _safe_float(eff.get("generate_card_count"))
    discover_v = _safe_float(eff.get("discover_count"))
    retrieve_v = _safe_float(eff.get("retrieve_from_discard"))
    upgrade_v = _safe_float(eff.get("upgrade_hand"))
    dup_v = _safe_float(eff.get("duplicate_next"))
    replace_v = _safe_float(eff.get("replace_or_transform_hand"))
    debuff = bool(weak_v > 0 or vuln_v > 0 or poison_v > 0 or "debuff" in effect_family)
    resource_like = bool(
        energy_gain > 0 or draw > 0
        or gen_card_v > 0 or discover_v > 0 or retrieve_v > 0
    )
    hand_transform_like = bool(upgrade_v > 0 or dup_v > 0 or replace_v > 0)
    long_term_like = bool(eff.get("long_term_value") or "long_term_value" in timing_tags)
    passive_or_triggered = bool(eff.get("passive_or_triggered"))

    incoming, current_block, hp = _incoming_damage_pressure(raw_obs)
    threat_gap = max(0.0, incoming - current_block)
    target_hp = _target_enemy_hp(action, raw_obs)
    aoe = bool(
        eff.get("aoe")
        or str(action.get("target_scope") or target_scope).lower() in {"all_enemies", "aoe", "allenemies", "allcreatures"}
    )
    lethal = bool(damage > 0 and target_hp > 0 and damage >= target_hp)
    overkill = bool(
        damage > 0 and target_hp > 0 and not aoe
        and damage > target_hp + max(6.0, 0.50 * target_hp)
    )

    defensive = bool(block > 0 or heal > 0 or debuff)
    prevent_lethal = bool(hp > 0 and threat_gap >= max(hp, 1.0) and defensive)
    prevent_major_loss = bool(threat_gap >= max(8.0, 0.25 * max(hp, 1.0)) and defensive)
    block_waste = bool(block > 0 and threat_gap <= 0.05)

    energy_after = max(0.0, float(energy) + energy_gain)
    # Find this action's index in legal_actions for follow-up scan.
    action_index = -1
    if legal_actions is not None:
        actions_list = list(legal_actions)
        for idx, candidate in enumerate(actions_list):
            if candidate is action or candidate == action:
                action_index = idx
                break
    followup_available = (
        _has_resource_followup(action_index, legal_actions, mask, energy_after)
        if resource_like else False
    )
    no_followup = bool(resource_like and not followup_available)

    # Kaiser facing detection — env-side approximation.
    facing_change = False
    kaiser_risk = 0.0
    try:
        boss_ctx = build_boss_mechanics_context(raw_obs) if isinstance(raw_obs, dict) else {}
        if isinstance(boss_ctx, dict):
            player_state = boss_ctx.get("player_state") if isinstance(boss_ctx.get("player_state"), dict) else {}
            kaiser_risk = max(
                _safe_float(player_state.get("primary_back_attack_risk")),
                _safe_float(player_state.get("primary_back_attack_active")),
            )
    except Exception:
        kaiser_risk = 0.0

    mechanism_answer = False
    if facing_change:
        mechanism_answer = True
    elif kaiser_risk > 0.05:
        mechanism_answer = bool(
            lethal
            or (damage >= 12.0 and target_hp <= 0.0)
            or (target_hp > 0.0 and damage >= min(target_hp, max(12.0, 0.35 * target_hp)))
            or block > 0.0
            or debuff
        )

    hp_ratio = 1.0
    if isinstance(raw_obs, dict):
        player_max_hp = _safe_float((raw_obs.get("player") or {}).get("max_hp"))
        if hp > 0 and player_max_hp > 0:
            hp_ratio = hp / player_max_hp
    high_damage = bool(damage >= 18.0 or (target_hp > 0 and damage >= max(12.0, 0.35 * target_hp)))

    use_quality = 0.08
    if lethal:
        use_quality += 0.85
    elif damage > 0:
        use_quality += min(0.34, damage / 55.0)
        if high_damage:
            use_quality += 0.16
    if prevent_lethal:
        use_quality += 0.95
    elif prevent_major_loss:
        use_quality += 0.52
    elif block > 0 and threat_gap > 0:
        use_quality += 0.35 * min(block / max(threat_gap, 1.0), 1.0)
    if heal > 0:
        use_quality += 0.25 if hp_ratio <= 0.55 else 0.10
    if debuff and incoming > 0:
        use_quality += 0.30
    if mechanism_answer:
        use_quality += 0.62
    if resource_like:
        use_quality += 0.36 if followup_available else -0.42
    tier_low = encounter_tier in ("elite", "boss")
    if tier_low and (lethal or prevent_major_loss or mechanism_answer or high_damage):
        use_quality += 0.12

    waste_risk = 0.0
    if no_followup:
        waste_risk += 0.45
    if block_waste:
        waste_risk += 0.35
    if overkill and not mechanism_answer:
        waste_risk += 0.25
    low_threat = threat_gap <= 2.0 and not prevent_major_loss and not prevent_lethal
    save_recommended = bool(
        low_threat and hp_ratio >= 0.55
        and not lethal and not mechanism_answer
        and not (tier_low and high_damage)
    )
    if save_recommended:
        waste_risk += 0.42 if encounter_tier in ("weak", "normal") else 0.22

    # Hand-transform / long-term adjustments.
    hand_size = 0
    if isinstance(raw_obs, dict):
        player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
        hand = player.get("hand") if isinstance(player, dict) else None
        hand_size = len(hand) if isinstance(hand, list) else 0
    hand_context_good = bool(hand_transform_like and hand_size >= 3)
    hand_context_bad = bool(hand_transform_like and hand_size <= 1)
    if hand_transform_like and hand_context_good:
        use_quality += 0.30
    if hand_transform_like and hand_context_bad:
        waste_risk += 0.30
        save_recommended = True
    urgent_threat = prevent_lethal or prevent_major_loss
    if long_term_like and not urgent_threat:
        use_quality = max(0.0, use_quality - 0.20)
        save_recommended = True
    if passive_or_triggered:
        use_quality = max(0.0, use_quality - 0.10)

    use_quality = float(np.clip(use_quality - waste_risk, 0.0, 1.0))
    waste_risk = float(np.clip(waste_risk, 0.0, 1.0))
    save_value = float(np.clip(
        (1.0 - use_quality) * 0.6
        + (0.25 if any(t in timing_tags for t in ("prevent_lethal_tool", "lethal_tool", "mechanism_answer_candidate")) else 0.0)
        + (0.20 if long_term_like else 0.0)
        - (0.30 if (lethal or prevent_lethal or mechanism_answer) else 0.0),
        0.0, 1.0,
    ))

    urgent = bool(
        lethal or prevent_lethal or mechanism_answer
        or (prevent_major_loss and use_quality >= 0.45)
        or use_quality >= 0.62
    )
    low_urgency = bool((use_quality < 0.35) or (waste_risk > use_quality and not urgent))
    positive = bool(urgent or use_quality >= 0.32)
    deferable = bool(not urgent and (low_urgency or save_recommended or no_followup or block_waste or overkill))
    requires_followup = bool(resource_like or hand_transform_like)

    return {
        "is_potion": True, "available": True,
        "potion_id": merged.get("potion_id", ""),
        "effect_family": effect_family, "timing_tags": timing_tags,
        "training_tags": training_tags,
        "use_quality": use_quality, "waste_risk": waste_risk, "save_value": save_value,
        "lethal": lethal, "prevent_lethal": prevent_lethal,
        "prevent_major_loss": prevent_major_loss,
        "mechanism_answer": mechanism_answer, "facing_change": facing_change,
        "overkill": overkill, "block_waste": block_waste, "no_followup": no_followup,
        "save_recommended": save_recommended, "low_urgency": low_urgency,
        "positive": positive, "urgent": urgent, "deferable": deferable,
        "requires_followup": requires_followup, "followup_available": followup_available,
        "hand_context_good": hand_context_good, "hand_context_bad": hand_context_bad,
        "long_term_value": long_term_like, "passive_or_triggered": passive_or_triggered,
        "aoe": aoe, "debuff": debuff,
        "damage": damage, "block": block, "draw": draw, "energy_gain": energy_gain,
        "heal": heal, "weak": weak_v, "vulnerable": vuln_v, "poison": poison_v,
        "incoming": incoming, "current_block": current_block, "hp": hp,
        "threat_gap": threat_gap, "target_hp": target_hp,
    }
