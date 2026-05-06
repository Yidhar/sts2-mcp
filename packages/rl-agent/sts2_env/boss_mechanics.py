"""Boss-special combat mechanic inference for STS2 observations.

This module deliberately keeps the dense observation shapes unchanged.
Instead, it extracts explicit boss/runtime mechanic state into lightweight
dicts that the token observation encoder can inject into existing token
numerics and trait tokens.

Design goals:
- work on both live bridge payloads and sim-translated payloads
- prefer concrete runtime fields when present
- fall back to metadata + robust text matching when runtime exposure is thin
- stay pure-python and cheap enough to call once per observation encode
"""

from __future__ import annotations

import re
from typing import Any

from content_registry import build_live_enemy_semantic_text, get_enemy_metadata

from . import observation_common as obs_common


_WORD_RE = re.compile(r"[a-z0-9_]+")


def enemy_mechanics_key(enemy: dict[str, Any] | None, fallback: Any = "") -> str:
    if isinstance(enemy, dict):
        for key in ("combat_id", "id", "model_id", "name"):
            value = enemy.get(key)
            if value not in (None, ""):
                return str(value)
    return str(fallback)


def combat_encounter_key(obs: dict[str, Any] | None) -> str:
    if not isinstance(obs, dict):
        return ""
    run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    for value in (
        obs.get("encounter_id"),
        obs.get("room_model_id"),
        combat.get("encounter_id"),
        run.get("room_model"),
        run.get("room_model_id"),
        run.get("room_model_id"),
    ):
        text = str(value or "").strip()
        if text:
            return text.upper()
    return ""


def build_boss_mechanics_context(obs: dict[str, Any] | None) -> dict[str, Any]:
    obs = obs if isinstance(obs, dict) else {}
    enemies = _combat_enemies(obs)
    player_state = infer_player_boss_state(obs)
    enemy_states_by_key: dict[str, dict[str, float]] = {}
    enemy_states_by_index: list[dict[str, float]] = []
    enemy_traits_by_key: dict[str, dict[str, list[dict[str, Any]]]] = {}
    enemy_traits_by_index: list[dict[str, list[dict[str, Any]]]] = []

    for enemy_index, enemy in enumerate(enemies):
        key = enemy_mechanics_key(enemy, enemy_index)
        enemy_state = infer_enemy_boss_state(
            obs,
            enemy,
            enemy_index,
            player_state=player_state,
        )
        reactive_traits, phase_rules = infer_enemy_boss_traits(
            obs,
            enemy,
            enemy_index,
            player_state=player_state,
            enemy_state=enemy_state,
        )
        enemy_states_by_key[key] = enemy_state
        enemy_states_by_index.append(enemy_state)
        packed = {
            "reactive_traits": reactive_traits,
            "phase_rules": phase_rules,
        }
        enemy_traits_by_key[key] = packed
        enemy_traits_by_index.append(packed)

    if enemy_states_by_index:
        player_state["back_attack_risk"] = max(
            float(state.get("incoming_damage_multiplier_norm", 0.0))
            for state in enemy_states_by_index
        )
        # Primary-threat back attack detection (Kaiser Crab fix).  Kaiser has
        # two parts each with a BackAttackPower in opposite directions; per
        # turn, ONE part deals high damage and the other deals low.  The
        # correct play is to face away from the high-damage attacker.  The
        # broad `back_attack_risk = max(...)` above always reads 1.5 because
        # the *other* (low-damage) part also flags 1.5 → facing change can
        # never reduce the metric and the §12 KAISER_FACING_CHANGE_BONUS
        # never triggers.  We pick the enemy with the highest intent damage
        # and read ONLY that enemy's incoming_damage_multiplier_norm so the
        # signal flips when the player correctly faces the primary threat.
        primary_threat_score = -1.0
        primary_back_active = 0.0
        primary_back_risk = 0.0
        for index, state in enumerate(enemy_states_by_index):
            enemy = enemies[index] if index < len(enemies) else None
            intent = enemy.get("intent") if isinstance(enemy, dict) else None
            damage = obs_common._float(intent.get("total_damage")) if isinstance(intent, dict) else 0.0
            if damage > primary_threat_score:
                primary_threat_score = damage
                primary_back_active = float(state.get("back_attack_active", 0.0))
                primary_back_risk = float(state.get("incoming_damage_multiplier_norm", 0.0))
        player_state["primary_back_attack_active"] = primary_back_active
        player_state["primary_back_attack_risk"] = primary_back_risk
        player_state["primary_threat_intent_damage"] = max(0.0, primary_threat_score)

        player_state["linked_support_alive"] = max(
            float(state.get("linked_support_alive", 0.0))
            for state in enemy_states_by_index
        )
        player_state["countdown_active"] = max(
            player_state.get("countdown_active", 0.0),
            max(float(state.get("countdown_active", 0.0)) for state in enemy_states_by_index),
        )
        player_state["escape_card_tax"] = max(
            player_state.get("escape_card_tax", 0.0),
            max(float(state.get("escape_card_tax", 0.0)) for state in enemy_states_by_index),
        )

    return {
        "encounter_key": combat_encounter_key(obs),
        "player_state": player_state,
        "enemy_states_by_key": enemy_states_by_key,
        "enemy_states_by_index": enemy_states_by_index,
        "enemy_traits_by_key": enemy_traits_by_key,
        "enemy_traits_by_index": enemy_traits_by_index,
    }


def infer_player_boss_state(obs: dict[str, Any] | None) -> dict[str, float]:
    obs = obs if isinstance(obs, dict) else {}
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    player_powers = _player_power_entries(obs)
    facing = str(combat.get("facing") or "").strip().lower()

    hand_cards = _runtime_cards(obs, "hand", "hand")
    draw_cards = _runtime_cards(obs, "draw_pile", "draw_preview_cards")
    discard_cards = _runtime_cards(obs, "discard_pile", "discard_cards")
    exhaust_cards = _runtime_cards(obs, "exhaust_pile", "exhaust_cards")

    # Source-confirmed STS2 ids/classes for The Insatiable's lethal countdown:
    #   PowerModel id  = POWER.SANDPIT_POWER
    #   class name     = SandpitPower
    #
    # Important ownership detail from the game source:
    #   TheInsatiable applies SandpitPower to *the enemy creature* and stores
    #   the affected player/ally in SandpitPower.Target.  FranticEscape also
    #   searches enemies for SandpitPower before extending the counter.  Older
    #   synthetic tests and some translated payloads exposed it on the player,
    #   so read both sides and take the visible countdown maximum.
    #
    # Bridge exposes both id/model_id and class_name/kind, but keep the broad
    # "sandpit" fallback for older payloads that only had localized title text.
    sandpit_power_needles = (
        "power.sandpit_power",
        "sandpitpower",
        "sandpit_power",
        "sandpit",
    )
    sandpit_amount = max(
        _power_amount(player_powers, sandpit_power_needles),
        _power_amount(_enemy_power_entries(obs), sandpit_power_needles),
    )
    ringing_amount = _power_amount(player_powers, ("ringing",))
    chains_amount = _power_amount(player_powers, ("chains of binding", "chain of binding", "chains_of_binding", "chain_of_binding", "binding"))
    hunger_amount = _power_amount(player_powers, ("hunger",))
    scrutiny_amount = _power_amount(player_powers, ("scrutiny",))
    grasp_amount = _power_amount(player_powers, ("grasp",))

    state = _zero_player_state()
    state["facing_left"] = 1.0 if facing == "left" else 0.0
    state["facing_right"] = 1.0 if facing == "right" else 0.0
    state["sandpit_active"] = float(sandpit_amount > 0.0)
    state["sandpit_turns"] = float(max(sandpit_amount, 0.0))
    state["sandpit_turns_norm"] = _norm(sandpit_amount, 10.0)
    state["ringing_active"] = float(ringing_amount > 0.0 or _has_power(player_powers, ("ringing",)))
    state["ringing_amount_norm"] = _norm(ringing_amount, 10.0)
    state["chains_active"] = float(
        chains_amount > 0.0 or _has_power(player_powers, ("chains of binding", "chain of binding", "chains_of_binding", "chain_of_binding", "binding"))
    )
    state["chains_amount_norm"] = _norm(chains_amount, 10.0)
    state["bound_active"] = float(any(_contains_word(_power_text(power), "bound") for power in player_powers))
    state["hunger_active"] = float(hunger_amount > 0.0 or _has_power(player_powers, ("hunger",)))
    state["hunger_amount_norm"] = _norm(hunger_amount, 10.0)
    state["scrutiny_active"] = float(scrutiny_amount > 0.0 or _has_power(player_powers, ("scrutiny",)))
    state["scrutiny_amount_norm"] = _norm(scrutiny_amount, 10.0)
    state["grasp_active"] = float(grasp_amount > 0.0 or _has_power(player_powers, ("grasp",)))
    state["grasp_amount_norm"] = _norm(grasp_amount, 10.0)

    # Frantic Escape (沙虫/Insatiable 倒计时机制) — needles must cover the
    # actual card-data shape:
    #   id            "CARD.FRANTIC_ESCAPE" → underscore form
    #   class_name    "FranticEscape"       (new bridge class metadata)
    #   title         "狂乱逃离"           (Chinese)
    #   description   contains "远离" / "沙坑"
    # Original needles tuple ("frantic escape",) only matched a bare-space
    # English string that NONE of those fields contain, so every counter was
    # pinned at 0 and the escape mechanic was invisible to the model.
    _FRANTIC_NEEDLES = (
        "frantic_escape",
        "frantic escape",
        "franticescape",
        "狂乱逃离",
        "card.frantic_escape",
    )
    frantic_escape_hand = _count_named_cards(hand_cards, _FRANTIC_NEEDLES)
    frantic_escape_draw = _count_named_cards(draw_cards, _FRANTIC_NEEDLES)
    frantic_escape_discard = _count_named_cards(discard_cards, _FRANTIC_NEEDLES)
    frantic_escape_exhaust = _count_named_cards(exhaust_cards, _FRANTIC_NEEDLES)
    frantic_escape_total = (
        frantic_escape_hand
        + frantic_escape_draw
        + frantic_escape_discard
        + frantic_escape_exhaust
    )
    state["frantic_escape_hand_norm"] = _norm(frantic_escape_hand, 3.0)
    state["frantic_escape_draw_norm"] = _norm(frantic_escape_draw, 5.0)
    state["frantic_escape_discard_norm"] = _norm(frantic_escape_discard, 5.0)
    state["frantic_escape_total_norm"] = _norm(frantic_escape_total, 8.0)
    # Raw counters are intentionally kept out of the dense observation slots.
    # Planner-side boss priors / diagnostics can consume the concrete countdown
    # and pile counts without changing the trained network input shape.
    state["frantic_escape_hand_count"] = float(frantic_escape_hand)
    state["frantic_escape_draw_count"] = float(frantic_escape_draw)
    state["frantic_escape_discard_count"] = float(frantic_escape_discard)
    state["frantic_escape_exhaust_count"] = float(frantic_escape_exhaust)
    state["frantic_escape_total_count"] = float(frantic_escape_total)
    state["escape_card_available"] = float(frantic_escape_hand > 0)

    state["play_budget_lock"] = max(
        state["ringing_active"],
        state["chains_active"],
        state["bound_active"],
    )
    state["doormaker_lock_pressure"] = max(
        state["hunger_active"],
        state["scrutiny_active"],
        state["grasp_active"],
    )
    state["countdown_active"] = max(
        state["sandpit_active"],
        float(frantic_escape_total > 0),
    )
    state["escape_card_tax"] = float(frantic_escape_total > 0 or state["sandpit_active"] > 0.0)
    return state


def infer_enemy_boss_state(
    obs: dict[str, Any] | None,
    enemy: dict[str, Any] | None,
    enemy_index: int,
    *,
    player_state: dict[str, float] | None = None,
) -> dict[str, float]:
    obs = obs if isinstance(obs, dict) else {}
    enemy = enemy if isinstance(enemy, dict) else {}
    player_state = player_state if isinstance(player_state, dict) else infer_player_boss_state(obs)
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    encounter_key = combat_encounter_key(obs)
    texts = _enemy_search_texts(enemy, encounter_key=encounter_key)
    joined = " | ".join(texts)
    round_number = max(0.0, obs_common._float(combat.get("round")))
    hp = max(0.0, obs_common._float(enemy.get("hp", enemy.get("current_hp"))))
    max_hp = max(1.0, obs_common._float(enemy.get("max_hp"), max(hp, 1.0)))
    hp_ratio = min(hp / max_hp, 1.0) if max_hp > 0 else 0.0

    state = _zero_enemy_state()
    state["hp_ratio"] = hp_ratio

    incoming_mult_raw = max(1.0, obs_common._float(enemy.get("incoming_damage_multiplier"), 1.0))
    state["incoming_damage_multiplier_raw"] = incoming_mult_raw
    state["incoming_damage_multiplier_norm"] = _norm(incoming_mult_raw, 2.0)
    # back_attack_active must be a DYNAMIC signal that flips when the player
    # re-faces.  The bridge's ComputeIncomingDamageMultiplier returns 1.5
    # only when the player's CURRENT facing matches the enemy's
    # BackAttackLeft/Right power (i.e. the player is currently in the
    # back-attack-vulnerable orientation); it returns 1.0 once the player
    # turns to address the boss correctly.  Earlier this field also OR'd in
    # a text-match against "back attack ..." — but Kaiser Crab's enemy text
    # *always* contains that string as the static power description, so the
    # text-OR pinned the value at 1.0 for every step regardless of facing.
    # That broke the §12 KAISER_FACING_CHANGE_BONUS gating
    # (`before_active > 0.5 and after_active <= 0.5` could never fire).
    # Keep the text match around as a separate "mechanic present" flag for
    # encoder coverage, but drop it from the dynamic active signal.
    has_back_attack_power = _matches_any(
        joined,
        ("back attack left", "back attack right", "backattackleft", "backattackright"),
    )
    state["back_attack_active"] = float(incoming_mult_raw > 1.0)
    state["back_attack_mechanic_present"] = float(has_back_attack_power)

    threshold_value = _enemy_threshold_value(enemy)
    state["threshold_value_norm"] = (
        min(threshold_value / max(max_hp, 1.0), 1.0) if threshold_value > 0.0 else 0.0
    )
    state["threshold_active"] = float(threshold_value > 0.0 and hp <= threshold_value)
    if threshold_value > 0.0 and hp > threshold_value:
        pending_window = max(25.0, max_hp * 0.15)
        state["transform_pending"] = float((hp - threshold_value) <= pending_window)

    is_vantom = _enemy_is(enemy, joined, ("vantom", "monster.vantom"))
    if is_vantom:
        state["boss_special_active"] = 1.0
        damage_cap_active = (
            _matches_any(joined, ("slippery", "damage_cap", "damage cap"))
            or _has_enemy_power(enemy, ("slippery", "intangible"))
            or round_number <= 2.0
        )
        if damage_cap_active:
            state["damage_cap_active"] = 1.0
            state["damage_cap_value"] = max(1.0, _power_amount(enemy.get("powers"), ("slippery", "intangible")))
            if state["damage_cap_value"] <= 0.0:
                state["damage_cap_value"] = 1.0
            state["damage_cap_value_norm"] = _norm(state["damage_cap_value"], 10.0)
            state["special_phase_active"] = 1.0

    is_ceremonial_beast = _enemy_is(enemy, joined, ("ceremonial beast", "monster.ceremonial_beast"))
    if is_ceremonial_beast:
        state["boss_special_active"] = 1.0
        if state["threshold_active"] > 0.0:
            state["one_card_lock"] = max(state["one_card_lock"], 1.0)
            state["special_phase_active"] = 1.0
        if player_state.get("ringing_active", 0.0) > 0.0:
            state["one_card_lock"] = max(state["one_card_lock"], 1.0)
        intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
        if state["threshold_active"] > 0.0 and obs_common._float(intent.get("total_damage")) <= 0.0:
            state["stun_window"] = 1.0

    is_waterfall_giant = _enemy_is(enemy, joined, ("waterfall giant", "monster.waterfall_giant"))
    if is_waterfall_giant:
        state["boss_special_active"] = 1.0
        state["deathburst"] = 1.0
        raw_deathburst = _power_amount(enemy.get("powers"), ("steam eruption", "eruption", "explosion", "deathburst"))
        if raw_deathburst <= 0.0:
            raw_deathburst = max(10.0, 6.0 + 2.0 * round_number)
        state["deathburst_damage"] = raw_deathburst
        state["deathburst_damage_norm"] = _norm(raw_deathburst, 100.0)
        state["special_phase_active"] = 1.0

    is_test_subject = _enemy_is(enemy, joined, ("test subject", "monster.test_subject"))
    if is_test_subject:
        state["boss_special_active"] = 1.0
        if _has_enemy_power(enemy, ("intangible",)):
            state["intangible_phase"] = 1.0
            state["special_phase_active"] = 1.0
        elif hp_ratio >= 0.66:
            state["skill_punish"] = 1.0
        elif hp_ratio >= 0.33:
            state["wound_phase"] = 1.0
            state["special_phase_active"] = 1.0
        else:
            state["special_phase_active"] = 1.0
        if _matches_any(joined, ("revive", "reborn", "resurrect", "adaptable", "second life", "second_life")):
            state["revive_once"] = 1.0

    is_queen = _enemy_is(enemy, joined, ("queen", "monster.queen"))
    if is_queen:
        state["boss_special_active"] = 1.0
        state["binding_control"] = max(
            player_state.get("chains_active", 0.0),
            player_state.get("bound_active", 0.0),
            1.0 if _matches_any(joined, ("binding_control", "chains of binding", "binding")) else 0.0,
        )
        state["linked_support_alive"] = float(
            any(
                other_index != enemy_index
                and _enemy_alive(other)
                and _enemy_is(other, " | ".join(_enemy_search_texts(other, encounter_key=encounter_key)), ("torch head amalgam", "monster.torch_head_amalgam"))
                for other_index, other in enumerate(_combat_enemies(obs))
            )
        )

    is_doormaker = _enemy_is(enemy, joined, ("doormaker", "monster.doormaker"))
    if is_doormaker:
        state["boss_special_active"] = 1.0
        state["linked_support_alive"] = max(
            state["linked_support_alive"],
            float(
                any(
                    other_index != enemy_index
                    and _enemy_alive(other)
                    and _enemy_is(other, " | ".join(_enemy_search_texts(other, encounter_key=encounter_key)), ("door", "monster.door"))
                    for other_index, other in enumerate(_combat_enemies(obs))
                )
            ),
        )
        state["binding_control"] = max(
            state["binding_control"],
            player_state.get("doormaker_lock_pressure", 0.0),
        )
        if state["linked_support_alive"] > 0.0:
            state["transform_pending"] = max(state["transform_pending"], 1.0)

    is_knowledge_demon = _enemy_is(enemy, joined, ("knowledge demon", "monster.knowledge_demon"))
    if is_knowledge_demon:
        state["boss_special_active"] = 1.0
        state["choice_debuffs"] = 1.0

    is_insatiable = _enemy_is(enemy, joined, ("the insatiable", "monster.the_insatiable"))
    if is_insatiable:
        state["boss_special_active"] = 1.0
        state["countdown_active"] = max(
            1.0,
            player_state.get("countdown_active", 0.0),
        )
        state["escape_card_tax"] = 1.0
        state["countdown_turns_norm"] = max(
            state["countdown_turns_norm"],
            player_state.get("sandpit_turns_norm", 0.0),
        )
        state["special_phase_active"] = 1.0

    if _matches_any(joined, ("binding_control", "chains of binding", "binding")):
        state["binding_control"] = max(state["binding_control"], player_state.get("chains_active", 0.0), 1.0)
    if _matches_any(joined, ("choice_debuffs", "least-bad constraint", "debuff menu")):
        state["choice_debuffs"] = max(state["choice_debuffs"], 1.0)
    if _matches_any(joined, ("death_explosion", "steam eruption", "deathburst", "on_death")):
        state["deathburst"] = max(state["deathburst"], 1.0)
        if state["deathburst_damage"] <= 0.0 and round_number > 0.0:
            state["deathburst_damage"] = max(10.0, 6.0 + 2.0 * round_number)
            state["deathburst_damage_norm"] = _norm(state["deathburst_damage"], 100.0)
    if _matches_any(joined, ("one_card_lock", "one-card lockdown", "restricts the number of cards you can play")):
        state["one_card_lock"] = max(state["one_card_lock"], 1.0)
    if _matches_any(joined, ("damage_cap", "slippery", "hit_tax_opening")):
        state["damage_cap_active"] = max(state["damage_cap_active"], 1.0)
        if state["damage_cap_value"] <= 0.0:
            state["damage_cap_value"] = 1.0
            state["damage_cap_value_norm"] = _norm(1.0, 10.0)
    if _matches_any(joined, ("gain_intangible", "intangible_phase", "intangible")) and _has_enemy_power(enemy, ("intangible",)):
        state["intangible_phase"] = max(state["intangible_phase"], 1.0)
        state["special_phase_active"] = max(state["special_phase_active"], 1.0)
    if _matches_any(joined, ("revive", "reborn", "resurrect", "adaptable", "second life", "second_life")):
        state["revive_once"] = max(state["revive_once"], 1.0)
    if _matches_any(joined, ("escape_card_tax", "frantic escape", "frantic_escape", "狂乱逃离", "沙坑", "doom_clock", "countdown_tick")):
        state["countdown_active"] = max(state["countdown_active"], 1.0)
        state["escape_card_tax"] = max(state["escape_card_tax"], 1.0)

    return state


def infer_enemy_boss_traits(
    obs: dict[str, Any] | None,
    enemy: dict[str, Any] | None,
    enemy_index: int,
    *,
    player_state: dict[str, float] | None = None,
    enemy_state: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    obs = obs if isinstance(obs, dict) else {}
    enemy = enemy if isinstance(enemy, dict) else {}
    player_state = player_state if isinstance(player_state, dict) else infer_player_boss_state(obs)
    enemy_state = (
        enemy_state
        if isinstance(enemy_state, dict)
        else infer_enemy_boss_state(obs, enemy, enemy_index, player_state=player_state)
    )

    reactive_traits: list[dict[str, Any]] = []
    phase_rules: list[dict[str, Any]] = []

    def add_reactive(**kwargs: Any) -> None:
        reactive_traits.append({"category": "reactive", **kwargs})

    def add_phase(**kwargs: Any) -> None:
        phase_rules.append({"category": "phase", **kwargs})

    if enemy_state.get("back_attack_active", 0.0) > 0.0:
        add_reactive(
            trait="back_attack",
            trigger_type="on_attack",
            condition="matched_facing",
            effect_type="damage_multiplier",
            effect_amount=enemy_state.get("incoming_damage_multiplier_raw", 1.0),
            description="Matched back-attack side increases incoming damage on the player.",
            severity="high",
        )
    if enemy_state.get("damage_cap_active", 0.0) > 0.0:
        add_phase(
            trait="damage_cap",
            trigger_type="on_turn_start",
            condition="opening_cycle",
            state="opening",
            effect_amount=enemy_state.get("damage_cap_value", 1.0),
            description="Per-hit damage is capped; multi-hit lines are preferred while this phase lasts.",
            severity="high",
        )
    if enemy_state.get("deathburst", 0.0) > 0.0:
        add_reactive(
            trait="death_explosion",
            trigger_type="on_death",
            condition="killed",
            effect_type="retaliate",
            effect_amount=enemy_state.get("deathburst_damage", 0.0),
            description="Killing this enemy can still deal a delayed death explosion / deathburst.",
            severity="high",
        )
    if enemy_state.get("revive_once", 0.0) > 0.0:
        add_phase(
            trait="revive_once",
            trigger_type="on_death",
            condition="revive_pending",
            state="revive",
            description="A lethal push may trigger a revive / reborn phase instead of ending the fight.",
            severity="high",
        )
    if enemy_state.get("one_card_lock", 0.0) > 0.0:
        add_phase(
            trait="one_card_lock",
            trigger_type="on_phase_start",
            condition="play_budget_locked",
            state="phase_two",
            description="This phase restricts the number of cards you can play each turn.",
            severity="high",
        )
    if enemy_state.get("skill_punish", 0.0) > 0.0:
        add_phase(
            trait="skill_punish_phase",
            trigger_type="on_play_skill",
            condition="phase_one",
            description="Early phase punishes unnecessary skill-heavy turns.",
            severity="medium",
        )
    if enemy_state.get("wound_phase", 0.0) > 0.0:
        add_phase(
            trait="wound_phase",
            trigger_type="on_player_hit",
            condition="chip_damage_taken",
            description="This phase punishes letting chip damage through block.",
            severity="high",
        )
    if enemy_state.get("intangible_phase", 0.0) > 0.0:
        add_phase(
            trait="gain_intangible",
            trigger_type="on_turn_start",
            condition="intangible_window",
            state="phase_three",
            description="Current phase includes intangible turns, which devalue burst timing.",
            severity="high",
        )
    if enemy_state.get("binding_control", 0.0) > 0.0:
        add_reactive(
            trait="binding_control",
            trigger_type="on_turn_start",
            condition="binding_affliction",
            description="This encounter imposes binding-style play restrictions on the player.",
            severity="high",
        )
    if enemy_state.get("choice_debuffs", 0.0) > 0.0:
        add_reactive(
            trait="choice_debuffs",
            trigger_type="on_turn_start",
            condition="menu_constraint",
            description="The fight repeatedly imposes build-specific choice debuffs.",
            severity="high",
        )
    if enemy_state.get("linked_support_alive", 0.0) > 0.0:
        add_reactive(
            trait="linked_support_alive",
            trigger_type="while_alive",
            condition="support_body_present",
            description="A linked support body is still alive and materially changes the fight axis.",
            severity="high",
        )
    if enemy_state.get("countdown_active", 0.0) > 0.0:
        add_phase(
            trait="countdown_tick",
            trigger_type="on_turn_start",
            condition="countdown_active",
            effect_amount=enemy_state.get("countdown_turns_norm", 0.0),
            description="An external countdown / doom clock is active in this encounter.",
            severity="high",
        )
    if enemy_state.get("escape_card_tax", 0.0) > 0.0:
        add_phase(
            trait="escape_card_tax",
            trigger_type="while_alive",
            condition="delay_cards_required",
            description="Dedicated escape / delay cards compete with your normal damage turn.",
            severity="high",
        )
    if enemy_state.get("transform_pending", 0.0) > 0.0:
        add_phase(
            trait="phase_shift",
            trigger_type="on_hp_threshold",
            condition="threshold_near",
            description="The enemy is close to a phase / threshold transition window.",
            severity="medium",
        )
    if enemy_state.get("stun_window", 0.0) > 0.0:
        add_phase(
            trait="threshold_stun",
            trigger_type="on_phase_start",
            condition="stunned",
            state="stunned",
            description="The current phase looks like a post-threshold stun / exposed window.",
            severity="high",
        )

    return reactive_traits, phase_rules


def _zero_player_state() -> dict[str, float]:
    return {
        "facing_left": 0.0,
        "facing_right": 0.0,
        "sandpit_active": 0.0,
        "sandpit_turns": 0.0,
        "sandpit_turns_norm": 0.0,
        "ringing_active": 0.0,
        "ringing_amount_norm": 0.0,
        "chains_active": 0.0,
        "chains_amount_norm": 0.0,
        "bound_active": 0.0,
        "hunger_active": 0.0,
        "hunger_amount_norm": 0.0,
        "scrutiny_active": 0.0,
        "scrutiny_amount_norm": 0.0,
        "grasp_active": 0.0,
        "grasp_amount_norm": 0.0,
        "frantic_escape_hand_norm": 0.0,
        "frantic_escape_draw_norm": 0.0,
        "frantic_escape_discard_norm": 0.0,
        "frantic_escape_total_norm": 0.0,
        "frantic_escape_hand_count": 0.0,
        "frantic_escape_draw_count": 0.0,
        "frantic_escape_discard_count": 0.0,
        "frantic_escape_exhaust_count": 0.0,
        "frantic_escape_total_count": 0.0,
        "escape_card_available": 0.0,
        "play_budget_lock": 0.0,
        "doormaker_lock_pressure": 0.0,
        "back_attack_risk": 0.0,
        "primary_back_attack_active": 0.0,
        "primary_back_attack_risk": 0.0,
        "primary_threat_intent_damage": 0.0,
        "linked_support_alive": 0.0,
        "countdown_active": 0.0,
        "escape_card_tax": 0.0,
    }


def _zero_enemy_state() -> dict[str, float]:
    return {
        "boss_special_active": 0.0,
        "hp_ratio": 0.0,
        "incoming_damage_multiplier_raw": 1.0,
        "incoming_damage_multiplier_norm": 0.0,
        "back_attack_active": 0.0,
        "back_attack_mechanic_present": 0.0,
        "threshold_value_norm": 0.0,
        "threshold_active": 0.0,
        "transform_pending": 0.0,
        "stun_window": 0.0,
        "damage_cap_active": 0.0,
        "damage_cap_value": 0.0,
        "damage_cap_value_norm": 0.0,
        "deathburst": 0.0,
        "deathburst_damage": 0.0,
        "deathburst_damage_norm": 0.0,
        "revive_once": 0.0,
        "linked_support_alive": 0.0,
        "special_phase_active": 0.0,
        "one_card_lock": 0.0,
        "skill_punish": 0.0,
        "wound_phase": 0.0,
        "intangible_phase": 0.0,
        "binding_control": 0.0,
        "choice_debuffs": 0.0,
        "countdown_active": 0.0,
        "countdown_turns_norm": 0.0,
        "escape_card_tax": 0.0,
    }


def _combat_enemies(obs: dict[str, Any]) -> list[dict[str, Any]]:
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    enemies = combat.get("enemies")
    return enemies if isinstance(enemies, list) else []


def _player_power_entries(obs: dict[str, Any]) -> list[dict[str, Any]]:
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
    collections: list[Any] = [
        combat.get("player_powers"),
        player.get("powers"),
        player.get("status"),
    ]
    player_creature = player.get("creature") if isinstance(player.get("creature"), dict) else {}
    collections.append(player_creature.get("powers"))
    for creature in combat.get("player_creatures") if isinstance(combat.get("player_creatures"), list) else []:
        if isinstance(creature, dict):
            collections.append(creature.get("powers"))
    for player_payload in obs.get("players") if isinstance(obs.get("players"), list) else []:
        if not isinstance(player_payload, dict):
            continue
        collections.append(player_payload.get("powers"))
        creature = player_payload.get("creature") if isinstance(player_payload.get("creature"), dict) else {}
        collections.append(creature.get("powers"))
    for player_payload in combat.get("players") if isinstance(combat.get("players"), list) else []:
        if not isinstance(player_payload, dict):
            continue
        collections.append(player_payload.get("powers"))
        creature = player_payload.get("creature") if isinstance(player_payload.get("creature"), dict) else {}
        collections.append(creature.get("powers"))

    powers: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for collection in collections:
        if isinstance(collection, list):
            for power in collection:
                if not isinstance(power, dict):
                    continue
                ident = str(
                    power.get("id")
                    or power.get("model_id")
                    or power.get("power_id")
                    or power.get("class_name")
                    or power.get("kind")
                    or power.get("title")
                    or power.get("name")
                    or ""
                )
                amount = str(power.get("amount") if power.get("amount") is not None else "")
                display_amount = str(
                    power.get("display_amount") if power.get("display_amount") is not None else ""
                )
                key = (ident, amount, display_amount)
                if key in seen:
                    continue
                seen.add(key)
                powers.append(power)
    return powers


def _enemy_power_entries(obs: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all visible enemy-owned powers from live/sim payload shapes.

    Most boss counters are enemy-owned in STS2.  In particular The
    Insatiable's SandpitPower is applied to the boss creature while targeting
    the player, so player-only power scans miss the lethal countdown entirely.
    """

    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    collections: list[Any] = []

    def _append_enemy_collections(enemies: Any) -> None:
        if not isinstance(enemies, list):
            return
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            collections.append(enemy.get("powers"))
            creature = enemy.get("creature") if isinstance(enemy.get("creature"), dict) else {}
            collections.append(creature.get("powers"))

    _append_enemy_collections(combat.get("enemies"))
    _append_enemy_collections(combat.get("monsters"))
    _append_enemy_collections(combat.get("enemy_creatures"))
    _append_enemy_collections(obs.get("enemies"))
    _append_enemy_collections(obs.get("monsters"))

    powers: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for collection in collections:
        if isinstance(collection, list):
            for power in collection:
                if not isinstance(power, dict):
                    continue
                ident = str(
                    power.get("id")
                    or power.get("model_id")
                    or power.get("power_id")
                    or power.get("class_name")
                    or power.get("kind")
                    or power.get("title")
                    or power.get("name")
                    or ""
                )
                amount = str(power.get("amount") if power.get("amount") is not None else "")
                display_amount = str(
                    power.get("display_amount") if power.get("display_amount") is not None else ""
                )
                key = (ident, amount, display_amount)
                if key in seen:
                    continue
                seen.add(key)
                powers.append(power)
    return powers


def _runtime_cards(obs: dict[str, Any], primary_key: str, fallback_key: str) -> list[Any]:
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    player = obs.get("player") if isinstance(obs.get("player"), dict) else {}

    def _cards_from_value(value: Any) -> tuple[bool, list[Any]]:
        if isinstance(value, list):
            return True, value
        if isinstance(value, dict) and isinstance(value.get("cards"), list):
            return True, value["cards"]
        return False, []

    candidates: list[Any] = [
        combat.get(primary_key),
        combat.get(fallback_key),
        player.get(primary_key),
        player.get(fallback_key),
    ]
    player_combat = player.get("combat") if isinstance(player.get("combat"), dict) else {}
    candidates.extend((player_combat.get(primary_key), player_combat.get(fallback_key)))
    for player_payload in obs.get("players") if isinstance(obs.get("players"), list) else []:
        if not isinstance(player_payload, dict):
            continue
        combat_payload = player_payload.get("combat") if isinstance(player_payload.get("combat"), dict) else {}
        candidates.extend((combat_payload.get(primary_key), combat_payload.get(fallback_key)))
    for player_payload in combat.get("players") if isinstance(combat.get("players"), list) else []:
        if not isinstance(player_payload, dict):
            continue
        combat_payload = player_payload.get("combat") if isinstance(player_payload.get("combat"), dict) else {}
        candidates.extend((combat_payload.get(primary_key), combat_payload.get(fallback_key)))

    saw_valid_empty = False
    for candidate in candidates:
        valid, cards = _cards_from_value(candidate)
        if not valid:
            continue
        if cards:
            return cards
        saw_valid_empty = True
    return [] if saw_valid_empty else []


def _count_named_cards(cards: list[Any], needles: tuple[str, ...]) -> int:
    total = 0
    for card in cards:
        if not isinstance(card, dict):
            continue
        text = " | ".join(
            str(card.get(key) or "").strip().lower()
            for key in (
                "title",
                "name",
                "id",
                "card_id",
                "model_id",
                "normalized_id",
                "class_name",
                "kind",
                "canonical_text",
                "description",
                "text",
            )
            if str(card.get(key) or "").strip()
        )
        if _matches_any(text, needles):
            total += 1
    return total


def _enemy_is(enemy: dict[str, Any], joined_text: str, needles: tuple[str, ...]) -> bool:
    enemy_key = enemy_mechanics_key(enemy, "")
    enemy_key_l = enemy_key.lower()
    return _matches_any(enemy_key_l, needles) or _matches_any(joined_text, needles)


def _enemy_alive(enemy: dict[str, Any] | None) -> bool:
    if not isinstance(enemy, dict):
        return False
    if enemy.get("is_alive") is False:
        return False
    return obs_common._float(enemy.get("hp", enemy.get("current_hp"))) > 0.0


def _enemy_threshold_value(enemy: dict[str, Any]) -> float:
    threshold = 0.0
    metadata = get_enemy_metadata(str(enemy.get("model_id") or "").strip())
    for collection in (
        metadata.get("phase_rules") if isinstance(metadata, dict) else [],
        metadata.get("trait_tokens") if isinstance(metadata, dict) else [],
        enemy.get("phase_rules"),
        enemy.get("reactive_triggers"),
        enemy.get("static_traits"),
    ):
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            threshold = max(threshold, abs(obs_common._float(item.get("threshold"))))
    return threshold


def _enemy_search_texts(enemy: dict[str, Any], *, encounter_key: str) -> list[str]:
    texts: list[str] = []
    for key in ("name", "model_id", "id"):
        value = str(enemy.get(key) or "").strip()
        if value:
            texts.append(value.lower())
    semantic = build_live_enemy_semantic_text(enemy)
    if semantic:
        texts.append(semantic.lower())
    metadata = get_enemy_metadata(str(enemy.get("model_id") or "").strip())
    if isinstance(metadata, dict):
        for key in ("title", "summary"):
            value = str(metadata.get(key) or "").strip()
            if value:
                texts.append(value.lower())
        for key in ("semantic_tags", "combat_tags"):
            values = metadata.get(key)
            if isinstance(values, list):
                texts.extend(str(value or "").strip().lower() for value in values if str(value or "").strip())
    if encounter_key:
        texts.append(encounter_key.lower())
    intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
    for key in ("title", "description", "intent_type"):
        value = str(intent.get(key) or "").strip()
        if value:
            texts.append(value.lower())
    for collection_name in ("powers", "static_traits", "reactive_triggers", "phase_rules"):
        collection = enemy.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            texts.append(_power_text(item))
    return [text for text in texts if text]


def _has_enemy_power(enemy: dict[str, Any] | None, needles: tuple[str, ...]) -> bool:
    if not isinstance(enemy, dict):
        return False
    return _has_power(enemy.get("powers"), needles)


def _has_power(powers: Any, needles: tuple[str, ...]) -> bool:
    if not isinstance(powers, list):
        return False
    return any(_matches_any(_power_text(power), needles) for power in powers if isinstance(power, dict))


def _power_amount(powers: Any, needles: tuple[str, ...]) -> float:
    if not isinstance(powers, list):
        return 0.0
    amount = 0.0
    for power in powers:
        if not isinstance(power, dict):
            continue
        if not _matches_any(_power_text(power), needles):
            continue
        amount = max(
            amount,
            abs(obs_common._float(power.get("amount"))),
            abs(obs_common._float(power.get("display_amount"))),
        )
    return amount


def _power_text(power: dict[str, Any] | None) -> str:
    if not isinstance(power, dict):
        return ""
    parts: list[str] = []
    for key in (
        "id",
        "model_id",
        "power_id",
        "class_name",
        "kind",
        "title",
        "name",
        "description",
        "trait",
        "effect_type",
        "condition",
        "state",
    ):
        value = str(power.get(key) or "").strip().lower()
        if value:
            parts.append(value)
    return " | ".join(parts)


def _matches_any(text: str, needles: tuple[str, ...]) -> bool:
    if not text:
        return False
    return any(needle in text for needle in needles if needle)


def _contains_word(text: str, word: str) -> bool:
    if not text or not word:
        return False
    tokens = _WORD_RE.findall(text.lower())
    return word.lower() in tokens


def _norm(value: float, denom: float) -> float:
    if denom <= 0.0:
        return 0.0
    return float(max(0.0, min(value / denom, 1.0)))


# ---------------------------------------------------------------------------
# Phase 4 consolidated boss-mechanics block (TASK-E1/E2/E3 wiring).
#
# Single entry-point used by ``CombatSandboxEnv.step`` to surface the
# Kaiser / Ceremonial / Insatiable state and per-action mechanism flags
# on every step's ``info`` dict.  The trainer reads this to populate the
# ``boss_combat/<encounter>/*`` metric mirrors.
# ---------------------------------------------------------------------------

from .boss_ceremonial import (  # noqa: E402  (intentional late import to avoid cycles)
    build_ceremonial_state,
    classify_ceremonial_action_mechanism,
)
from .boss_insatiable import (  # noqa: E402
    build_insatiable_state,
    classify_insatiable_action_offenders,
)
from .boss_kaiser import (  # noqa: E402
    build_kaiser_state,
    classify_kaiser_action_mechanism,
)


def _boss_combat_state(obs: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    combat = obs.get("combat")
    return combat if isinstance(combat, dict) else {}


def _boss_player_state(obs: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    player = obs.get("player")
    return player if isinstance(player, dict) else {}


def build_boss_mechanics_block(obs: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``boss_mechanics`` block for the trainer to consume.

    Stable shape: every key is always present with ``active=False`` defaults
    so downstream code can index without conditional guards.
    """
    combat = _boss_combat_state(obs)
    player = _boss_player_state(obs)
    return {
        "kaiser": build_kaiser_state(combat, player_obs=player),
        "ceremonial": build_ceremonial_state(combat),
        "insatiable": build_insatiable_state(combat),
    }


def classify_action_boss_mechanism(
    obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    *,
    action_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-action mechanism dict combining the three Phase 4 helpers."""
    combat = _boss_combat_state(obs)
    player = _boss_player_state(obs)
    return {
        "kaiser": classify_kaiser_action_mechanism(combat, action, player_obs=player),
        "ceremonial": classify_ceremonial_action_mechanism(combat, action, player_obs=player),
        "insatiable": classify_insatiable_action_offenders(
            combat, action, player_obs=player, action_diagnostics=action_diagnostics
        ),
    }
