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
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return float(result)


_PLAYER_HP_KEYS = ("hp", "current_hp", "currentHealth", "current_health")
_PLAYER_MAX_HP_KEYS = ("max_hp", "maxHealth", "max_health", "maximum_hp", "max_hp_raw")


def _player_number(player: dict[str, Any], keys: tuple[str, ...]) -> float:
    if not isinstance(player, dict):
        return 0.0
    for key in keys:
        if key in player and player.get(key) is not None:
            return _safe_float(player.get(key))
    return 0.0


def _player_hp_ratio_from_values(hp: float, max_hp: float) -> float:
    """Safe potion-timing HP ratio.

    Do not allow bridge snapshots with max_hp<=1 to become a clipped
    full-health signal.  Those frames are either critical (hp<=1) or unknown
    (hp>1/max_hp=1), but never healthy.
    """

    hp_f = _safe_float(hp)
    max_hp_f = _safe_float(max_hp)
    if hp_f <= 0.0:
        return 0.0
    if max_hp_f <= 1.0:
        return 0.0
    return float(np.clip(hp_f / max_hp_f, 0.0, 1.0))


def _player_hp_triplet_from_raw(raw_obs: dict[str, Any] | None) -> tuple[float, float, float, bool]:
    if not isinstance(raw_obs, dict):
        return 0.0, 0.0, 0.0, False
    player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
    if not player and isinstance(raw_obs.get("combat"), dict):
        combat_player = raw_obs["combat"].get("player")
        if isinstance(combat_player, dict):
            player = combat_player
    hp = _player_number(player, _PLAYER_HP_KEYS)
    max_hp = _player_number(player, _PLAYER_MAX_HP_KEYS)
    normal_valid = bool(hp > 0.0 and max_hp > 1.0)
    critical_valid = bool(0.0 < hp <= 1.0 and 0.0 < max_hp <= 1.0)
    return hp, max_hp, _player_hp_ratio_from_values(hp, max_hp), bool(normal_valid or critical_valid)


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
        _player_number(player, _PLAYER_HP_KEYS),
    )


def _discard_pile_count_from_raw(raw_obs: dict[str, Any] | None) -> int:
    """Best-effort discard-pile size from bridge/raw combat observations.

    Retrieve-from-discard potions are only real immediate resources when the
    discard pile has a target.  Bridge payloads have used several spellings, so
    keep this parser permissive and monotonic with the trainer-side helper.
    """
    if not isinstance(raw_obs, dict):
        return 0

    best = 0
    sources: list[Any] = [raw_obs]
    combat = raw_obs.get("combat")
    player = raw_obs.get("player")
    if isinstance(combat, dict):
        sources.append(combat)
        nested_player = combat.get("player")
        if isinstance(nested_player, dict):
            sources.append(nested_player)
    if isinstance(player, dict):
        sources.append(player)

    list_keys = (
        "discard_pile",
        "discard_cards",
        "discard",
        "discardPile",
        "discardCards",
        "discardPileCards",
    )
    count_keys = (
        "discard_count",
        "discard_pile_count",
        "discardPileCount",
        "discard_size",
        "discardSize",
    )
    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in list_keys:
            val = src.get(key)
            if isinstance(val, list):
                best = max(best, len(val))
        for key in count_keys:
            try:
                val = src.get(key)
                if val is not None:
                    best = max(best, int(float(val)))
            except (TypeError, ValueError):
                continue
    return int(max(best, 0))


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
        "lethal_attacker_killable": False, "aoe_lethal_clear": False,
        "critical_hp_usable_survival_potion": False,
        "mechanism_answer": False, "facing_change": False,
        "overkill": False, "block_waste": False, "no_followup": False,
        "save_recommended": False, "low_urgency": False,
        "positive": False, "urgent": False, "deferable": False,
        "requires_followup": False, "followup_available": False,
        "hand_context_good": False, "hand_context_bad": False,
        "long_term_value": False, "passive_or_triggered": False,
        "aoe": False, "debuff": False,
        "damage": 0.0, "block": 0.0, "prevent_damage": 0.0, "draw": 0.0, "energy_gain": 0.0,
        "heal": 0.0, "weak": 0.0, "vulnerable": 0.0, "poison": 0.0,
        "incoming": 0.0, "current_block": 0.0, "hp": 0.0,
        "threat_gap": 0.0, "target_hp": 0.0,
        "buffer_like": False,
        "amplify_block_like": False, "amplify_block_added": 0.0,
        "amplify_block_noop": False,
        "resource_like": False, "random_potion_resource_like": False,
        "new_option_resource_like": False,
        "retrieve_from_discard": 0.0, "retrieve_from_discard_like": False,
        "discard_count": 0, "retrieve_has_target": False,
        "free_play_like": False, "resource_survival_tool": False,
        "critical_hp_survival_tool": False,
        "near_death_after_incoming": False, "near_death_margin": 0.0,
    }
    if not isinstance(action, dict) or _action_family(action) != "use_potion":
        return default

    merged = _resolve_potion_effect(action)
    eff = merged["effect_profile"]
    timing_tags = merged["timing_tags"]
    effect_family = merged["effect_family"]
    semantic_tags = merged["semantic_tags"]
    target_scope = merged["target_scope"]
    training_tags = merged["training_tags"]
    tag_tokens = {
        str(x).strip().lower()
        for x in list(effect_family) + list(semantic_tags) + list(timing_tags) + list(training_tags)
        if str(x).strip()
    }
    potion_id_upper = str(merged.get("potion_id") or "").strip().upper()
    potion_payload = action.get("potion") if isinstance(action.get("potion"), dict) else {}
    identity_text = " ".join(
        str(v)
        for container in (action, potion_payload)
        if isinstance(container, dict)
        for k, v in container.items()
        if k
        in {
            "id",
            "potion_id",
            "model_id",
            "normalized_id",
            "title",
            "name",
            "label",
            "title_en",
            "title_zhs",
            "localized_title",
            "description",
            "summary",
            "action_id",
        }
        and v not in (None, "")
    ).lower()
    amplify_block_like = bool(
        "FORTIFIER" in potion_id_upper
        or "amplify_block" in tag_tokens
        or "triple_block" in tag_tokens
        or "requires_block_in_play" in tag_tokens
        or "固化" in identity_text
        or "三倍" in identity_text
        or ("triple" in identity_text and "block" in identity_text)
    )

    damage = _safe_float(eff.get("damage"))
    block = _safe_float(eff.get("block"))
    prevent_damage = _safe_float(eff.get("prevent_damage"))
    heal = _safe_float(eff.get("heal"))
    draw = _safe_float(eff.get("draw"))
    energy_gain = _safe_float(eff.get("energy_gain"))
    weak_v = _safe_float(eff.get("weak"))
    vuln_v = _safe_float(eff.get("vulnerable"))
    poison_v = _safe_float(eff.get("poison"))
    gen_card_v = _safe_float(eff.get("generate_card_count"))
    discover_v = _safe_float(eff.get("discover_count"))
    retrieve_v = _safe_float(eff.get("retrieve_from_discard"))
    retrieve_from_discard_like = bool(
        retrieve_v > 0.0
        or "LIQUID_MEMORIES" in potion_id_upper
        or ("discard_pile" in tag_tokens and "tutor" in tag_tokens)
    )
    random_potion_resource_like = bool(
        "ENTROPIC_BREW" in potion_id_upper
        or "fill_potion_slots" in tag_tokens
        or "refill_potion_slots" in tag_tokens
        or ("potion" in tag_tokens and ("random" in tag_tokens or "random_outcome" in tag_tokens))
        or bool(eff.get("fill_potion_slots"))
        or bool(eff.get("random_outcome"))
        or "混沌药水" in identity_text
        or "entropic" in identity_text
    )
    free_play_like = bool(
        retrieve_from_discard_like
        or _safe_float(eff.get("set_cost_zero")) > 0.0
        or "set_cost_zero" in tag_tokens
        or "free_play" in tag_tokens
        or "free" in tag_tokens
    )
    upgrade_v = _safe_float(eff.get("upgrade_hand"))
    dup_v = _safe_float(eff.get("duplicate_next"))
    replace_v = _safe_float(eff.get("replace_or_transform_hand"))
    debuff = bool(weak_v > 0 or vuln_v > 0 or poison_v > 0 or "debuff" in effect_family)
    buffer_like = bool(
        prevent_damage > 0.0
        or "buffer" in tag_tokens
        or "prevent_damage" in tag_tokens
        or "LUCKY_TONIC" in potion_id_upper
        or "lucky tonic" in identity_text
        or "lucky_tonic" in identity_text
        or "幸运补剂" in identity_text
        or "幸运药剂" in identity_text
    )
    resource_like = bool(
        energy_gain > 0 or draw > 0
        or gen_card_v > 0 or discover_v > 0 or retrieve_v > 0
        or retrieve_from_discard_like
        or random_potion_resource_like
        or "resource" in tag_tokens
        or "energy" in tag_tokens
        or "draw" in tag_tokens
    )
    hand_transform_like = bool(upgrade_v > 0 or dup_v > 0 or replace_v > 0)
    requires_followup_profile = bool(
        eff.get("requires_followup")
        or "requires_followup" in tag_tokens
    )
    if amplify_block_like:
        # Fortifier/固化 requires block *before* use, not a post-use card
        # follow-up.  Do not punish real block-amplification turns as
        # no-followup resource waste.
        requires_followup_profile = False
    followup_dependent = bool(resource_like or hand_transform_like or requires_followup_profile)
    long_term_like = bool(eff.get("long_term_value") or "long_term_value" in timing_tags)
    passive_or_triggered = bool(eff.get("passive_or_triggered"))

    incoming, current_block, hp = _incoming_damage_pressure(raw_obs)
    threat_gap = max(0.0, incoming - current_block)
    max_hp = 0.0
    hp_ratio = 0.0
    hp_valid = False
    if isinstance(raw_obs, dict):
        hp_from_player, max_hp, hp_ratio, hp_valid = _player_hp_triplet_from_raw(raw_obs)
        if hp <= 0.0 and hp_from_player > 0.0:
            hp = hp_from_player
    discard_count = _discard_pile_count_from_raw(raw_obs)
    retrieve_has_target = bool(discard_count > 0)
    new_option_resource_like = bool(
        draw > 0.0
        or gen_card_v > 0.0
        or discover_v > 0.0
        or random_potion_resource_like
        or (retrieve_from_discard_like and retrieve_has_target)
    )
    near_death_margin = max(3.0, 0.05 * max(max_hp, 1.0))
    near_death_after_incoming = bool(
        hp_valid
        and threat_gap > 0.05
        and hp > 0.0
        and (hp - threat_gap) <= near_death_margin
    )
    amplify_block_added = 0.0
    amplify_block_noop = False
    if amplify_block_like:
        # Fortifier/固化 triples current block.  The static profile encodes
        # block=0 because the effect is state-dependent, so derive the
        # immediate added block here.  At 0 current block it is a pure no-op.
        amplify_block_added = max(0.0, 2.0 * float(current_block))
        block = max(float(block), float(amplify_block_added))
        amplify_block_noop = bool(current_block <= 0.05)
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
    lethal_threat_window = bool(hp > 0 and threat_gap >= max(hp, 1.0))
    alive_enemy_hps: list[float] = []
    combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
    enemies = combat.get("enemies") if isinstance(combat, dict) else []
    if isinstance(enemies, list):
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            hp_val = _safe_float(enemy.get("hp", enemy.get("current_hp", enemy.get("health"))))
            if hp_val > 0.0:
                alive_enemy_hps.append(float(hp_val))
    lethal_attacker_killable = bool(
        lethal_threat_window
        and damage > 0
        and target_hp > 0
        and damage >= target_hp
        and not aoe
    )
    aoe_lethal_clear = bool(
        lethal_threat_window
        and aoe
        and damage > 0
        and (
            (alive_enemy_hps and damage >= max(alive_enemy_hps))
            or (not alive_enemy_hps and target_hp > 0 and damage >= target_hp)
        )
    )

    defensive = bool(block > 0 or heal > 0 or debuff or buffer_like)
    prevents_lethal_now = bool(defensive or lethal_attacker_killable or aoe_lethal_clear)
    prevent_lethal = bool(lethal_threat_window and prevents_lethal_now)
    prevent_major_loss = bool(threat_gap >= max(8.0, 0.25 * max(hp, 1.0)) and defensive)
    block_waste = bool((block > 0 and threat_gap <= 0.05) or amplify_block_noop)
    critical_hp_survival_tool = bool(
        hp_valid
        and encounter_tier in {"elite", "boss"}
        and hp_ratio <= 0.35
        and (heal > 0.0 or block > 0.0 or buffer_like or (retrieve_from_discard_like and retrieve_has_target))
    )
    if (heal > 0.0 or buffer_like) and critical_hp_survival_tool:
        prevent_major_loss = True

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
        if followup_dependent else False
    )
    resource_survival_tool = bool(
        (
            retrieve_from_discard_like
            and hp_valid
            and encounter_tier in {"elite", "boss"}
            and (
                hp_ratio <= 0.15
                or threat_gap >= max(1.0, hp - 1.0)
                or (hp_ratio <= 0.35 and threat_gap >= max(6.0, 0.25 * max(hp, 1.0)))
                or (encounter_tier == "boss" and hp_ratio <= 0.50)
            )
        )
        or (
            near_death_after_incoming
            and (
                new_option_resource_like
                or (energy_gain > 0.0 and followup_available)
                or defensive
            )
            and not (retrieve_from_discard_like and not retrieve_has_target)
        )
    )
    if retrieve_from_discard_like and encounter_tier in {"elite", "boss"} and retrieve_has_target:
        # Retrieve/free-play potions expose their true follow-up only after the
        # potion resolves into a discard-pile selection.
        followup_available = True
    if resource_survival_tool and retrieve_from_discard_like and retrieve_has_target:
        followup_available = True
        prevent_major_loss = True
    elif resource_survival_tool and retrieve_from_discard_like and not retrieve_has_target:
        resource_survival_tool = False
    elif resource_survival_tool and not retrieve_from_discard_like:
        # Near-death stochastic/resource potions such as Entropic Brew or card-
        # discovery potions create their real answer only after use; do not mark
        # them as no-followup/save-only in a death-margin frame.
        followup_available = True
        prevent_major_loss = True
    no_followup = bool(followup_dependent and not followup_available)
    critical_hp_usable_survival_potion = bool(
        hp_valid
        and threat_gap > 0.0
        and (
            (
                hp_ratio <= 0.10
                and (
                    defensive
                    or damage > 0.0
                    or critical_hp_survival_tool
                    or resource_survival_tool
                    or ((energy_gain > 0.0 or draw > 0.0) and not no_followup)
                )
            )
            or (
                near_death_after_incoming
                and (
                    defensive
                    or critical_hp_survival_tool
                    or resource_survival_tool
                    or (new_option_resource_like and not no_followup)
                )
            )
        )
    )

    # Kaiser facing detection — P0-3 mandates the shared resolver instead
    # of the previous permanent ``facing_change = False`` placeholder.
    # Position (left/right) of any back-attack enemy comes from
    # BACK_ATTACK_{LEFT,RIGHT}_POWER on enemy powers, never faction side.
    facing_change = False
    kaiser_risk = 0.0
    try:
        boss_ctx = build_boss_mechanics_context(raw_obs) if isinstance(raw_obs, dict) else {}
        if isinstance(boss_ctx, dict):
            encounter_key = str(
                boss_ctx.get("encounter_key")
                or (raw_obs or {}).get("encounter")
                or (raw_obs or {}).get("encounter_id")
                or ""
            ).lower()
            if "kaiser" in encounter_key:
                player_state = boss_ctx.get("player_state") if isinstance(boss_ctx.get("player_state"), dict) else {}
                # Do not treat the normalized default incoming multiplier
                # (1.0 -> 0.5) as Kaiser risk in non-Kaiser hallway fights.
                # Gate by encounter first, mirroring MuZeroTrainer's
                # _kaiser_back_attack_risk_from_context.
                kaiser_risk = max(
                    _safe_float(player_state.get("primary_back_attack_risk")),
                    _safe_float(player_state.get("primary_back_attack_active")),
                    _safe_float(player_state.get("back_attack_risk")),
                    _safe_float(player_state.get("back_attack_active")),
                )
    except Exception:
        kaiser_risk = 0.0
    try:
        from .boss_kaiser import classify_kaiser_action_mechanism  # noqa: WPS433
        kaiser_mech = classify_kaiser_action_mechanism(
            (raw_obs or {}).get("combat") if isinstance(raw_obs, dict) else None,
            action,
            player_obs=(raw_obs or {}).get("player") if isinstance(raw_obs, dict) else None,
        )
        if kaiser_mech.get("kaiser_changes_facing"):
            facing_change = True
    except Exception:
        pass

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
    if buffer_like and threat_gap > 0.05:
        # Lucky Tonic / Buffer does not show up as block.  In lethal or near-
        # lethal pressure windows it is still an immediate survival tool, and
        # live bridge payloads may only expose the localized title ("幸运药剂").
        use_quality += 0.42 + min(0.28, float(threat_gap) / max(float(hp), 1.0))
        if encounter_tier in ("elite", "boss") and (prevent_lethal or prevent_major_loss or hp_ratio <= 0.35):
            use_quality += 0.25
    if debuff and incoming > 0:
        use_quality += 0.30
    if mechanism_answer:
        use_quality += 0.62
    if followup_dependent:
        use_quality += 0.36 if followup_available else -0.42
    if critical_hp_usable_survival_potion:
        use_quality += 0.45
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
        or critical_hp_usable_survival_potion
        or (prevent_major_loss and use_quality >= 0.45)
        or use_quality >= 0.62
    )
    low_urgency = bool((use_quality < 0.35) or (waste_risk > use_quality and not urgent))
    positive = bool(urgent or use_quality >= 0.32)
    deferable = bool(not urgent and (low_urgency or save_recommended or no_followup or block_waste or overkill))
    requires_followup = bool(followup_dependent)

    return {
        "is_potion": True, "available": True,
        "potion_id": merged.get("potion_id", ""),
        "effect_family": effect_family, "timing_tags": timing_tags,
        "semantic_tags": semantic_tags,
        "training_tags": training_tags,
        "use_quality": use_quality, "waste_risk": waste_risk, "save_value": save_value,
        "lethal": lethal, "prevent_lethal": prevent_lethal,
        "prevent_major_loss": prevent_major_loss,
        "lethal_attacker_killable": lethal_attacker_killable,
        "aoe_lethal_clear": aoe_lethal_clear,
        "critical_hp_usable_survival_potion": critical_hp_usable_survival_potion,
        "mechanism_answer": mechanism_answer, "facing_change": facing_change,
        "overkill": overkill, "block_waste": block_waste, "no_followup": no_followup,
        "save_recommended": save_recommended, "low_urgency": low_urgency,
        "positive": positive, "urgent": urgent, "deferable": deferable,
        "requires_followup": requires_followup, "followup_available": followup_available,
        "hand_context_good": hand_context_good, "hand_context_bad": hand_context_bad,
        "long_term_value": long_term_like, "passive_or_triggered": passive_or_triggered,
        "aoe": aoe, "debuff": debuff,
        "damage": damage, "block": block, "prevent_damage": prevent_damage,
        "buffer_like": buffer_like,
        "draw": draw, "energy_gain": energy_gain,
        "amplify_block_like": amplify_block_like,
        "amplify_block_added": amplify_block_added,
        "amplify_block_noop": amplify_block_noop,
        "resource_like": resource_like,
        "random_potion_resource_like": random_potion_resource_like,
        "new_option_resource_like": new_option_resource_like,
        "retrieve_from_discard": retrieve_v,
        "retrieve_from_discard_like": retrieve_from_discard_like,
        "discard_count": discard_count,
        "retrieve_has_target": retrieve_has_target,
        "free_play_like": free_play_like,
        "resource_survival_tool": resource_survival_tool,
        "critical_hp_survival_tool": critical_hp_survival_tool,
        "near_death_after_incoming": near_death_after_incoming,
        "near_death_margin": near_death_margin,
        "heal": heal, "weak": weak_v, "vulnerable": vuln_v, "poison": poison_v,
        "incoming": incoming, "current_block": current_block, "hp": hp,
        "threat_gap": threat_gap, "target_hp": target_hp,
    }
