# ruff: noqa: RUF001
"""Pure player, card, enemy, combat, map, and run translators."""

from __future__ import annotations

from typing import Any

from ._sim_translate_shared import _incoming_damage_multiplier


def _translate_player_powers(status_list: list[Any] | None) -> list[dict[str, Any]]:
    if not isinstance(status_list, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in status_list:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("id") or "")
        amount = int(entry.get("amount") or 0)
        out.append({
            "id": pid,
            "title": pid,
            "amount": amount,
            "display_amount": int(entry.get("display_amount", amount) or amount),
            "stack_type": str(entry.get("stack_type") or "Single"),
        })
    return out


def _translate_card(sim_card: Any, *, pile: str = "Deck") -> dict[str, Any]:
    """Turn a sim FullRunApiCardOption into our bridge card dict.

    Emits the bridge's *compact* card shape (BridgeGameApi.EnvCompact.cs
    :21-117). Downstream readers in observation_common/run_memory/
    semantic_action/combat_tactical_local/combat_env/env_v2 key off
    ``cost`` / ``upgrade_level`` / ``target`` / ``x_cost`` / ``canonical_text``
    — NOT the older ``canonical_energy_cost`` / ``current_upgrade_level`` /
    ``target_type`` / ``costs_x``. Legacy keys are kept alongside for any
    path that still reads them, but the new compact keys are authoritative.
    """
    if not isinstance(sim_card, dict):
        return {"missing": True}
    raw_id = str(sim_card.get("id") or "UNKNOWN")
    # Sim uses bare ids ("STRIKE_IRONCLAD"); bridge uses "CARD.STRIKE_IRONCLAD".
    card_id = raw_id if raw_id.startswith("CARD.") else f"CARD.{raw_id}"
    cost = sim_card.get("cost")
    cost_int = int(cost) if isinstance(cost, int) else 0
    # X-cost detection: sim runtime card DTO doesn't expose `is_x_cost` on the
    # hand payload (FullRunApiCardOption only has `cost: int?`), so infer from
    # the static card registry where energy_cost_text == "X". obs encoder
    # (observation_common.py:988) reads `x_cost`; legacy `costs_x` kept.
    is_x_cost = _card_is_x_cost_from_registry(card_id) or (cost is None and cost_int == 0)
    is_upgraded = bool(sim_card.get("is_upgraded"))
    upgrade_level = 1 if is_upgraded else 0
    target = str(sim_card.get("target_type") or "None")
    card_type = str(sim_card.get("type") or "Attack")
    title = str(sim_card.get("name") or raw_id)
    description = str(sim_card.get("description") or "")
    valid_targets = sim_card.get("valid_target_ids") or []
    effect_preview = _card_effect_preview_from_registry(card_id)
    # Mirror bridge's canonical_text construction (EnvCompact.cs:99-113) so
    # the text encoder gets the same Chinese semantic string it would see
    # in real-game training.
    ct_parts = ["卡牌", _normalize_semantic_text(title), card_type]
    if cost_int:
        ct_parts.append(f"能量{cost_int}")
    ct_parts.append(f"目标{_translate_target_type(target)}")
    if description:
        ct_parts.append(f"效果：{_normalize_semantic_text(description)}")
    canonical_text = "｜".join(ct_parts)
    payload: dict[str, Any] = {
        "id": card_id,
        # New compact keys (authoritative — matches bridge):
        "upgrade_level": upgrade_level,
        "cost": cost_int,
        "target": target,
        "type": card_type,
        "canonical_text": canonical_text,
        # Legacy keys (retained for any consumer still using them):
        "current_upgrade_level": upgrade_level,
        "max_upgrade_level": 1,
        "title": title,
        "description": description,
        "rarity": str(sim_card.get("rarity") or "Basic"),
        "target_type": target,
        "pile": pile,
        "is_playable": bool(sim_card.get("can_play", True)),
        "canonical_energy_cost": cost_int,
        "resolved_energy_cost": cost_int,
        "costs_x": is_x_cost,
        "x_cost": is_x_cost,
        # Regent-only star resource. Sim's FullRunApiCardOption has no
        # per-card star field (only the combat-global `stars` counter).
        # Emit None explicitly so the probe sees the key as "translated"
        # rather than "discarded"; obs encoder's 0.0 fallback for missing
        # keys is the correct signal for non-Regent characters.
        "star": None,
        "star_x": False,
        "keywords": list(sim_card.get("keywords") or []),
        "valid_target_ids": [int(t) for t in valid_targets if isinstance(t, int | float)],
        # Populate effect_preview from content_registry's semantic_signals
        # (static base values per card). Real bridge computes this live
        # including Strength/Weak/Vulnerable modifiers; we get the base
        # static values here. Upgrade modifiers skipped (Phase 5+).
        "effect_preview": effect_preview,
    }
    # Keep runtime per-card modifiers if the simulator exposes them.  Live
    # bridge emits the same compact fields; observation_v3 turns these into
    # CARD_KEYWORD_SLOT tokens (bound/card_lock/cost_lock/temporary/etc.).
    for _modifier_field in ("afflictions", "enchantments", "modifiers", "card_modifiers"):
        _mods = sim_card.get(_modifier_field)
        if isinstance(_mods, list) and _mods:
            payload[_modifier_field] = _mods[:16]
    # Bridge compact card emits ``effect`` (short summary) when present,
    # falling back to ``description``. effect_preview.summary isn't
    # populated today but description is non-empty — keep both.
    if description:
        payload["effect"] = description
    return payload


def _normalize_semantic_text(s: str) -> str:
    # Minimal sanitation — strip newlines, clamp length. The real bridge's
    # NormalizeSemanticText does similar; exact parity isn't critical
    # because the text encoder is robust to whitespace/newline differences.
    if not s:
        return ""
    return " ".join(s.replace("\n", " ").split())[:200]


def _translate_target_type(target: str) -> str:
    t = str(target or "").strip()
    mapping = {
        "AnyEnemy": "敌人",
        "AllEnemies": "所有敌人",
        "Self": "自身",
        "Ally": "友军",
        "None": "无",
    }
    return mapping.get(t, t or "无")


# Maps content_registry's semantic_signals keys to the keys that
# observation_common._preview_metric (and 30+ call sites across the codebase)
# read. We also copy through any already-canonical keys unchanged. Applied
# each time _translate_card runs; cheap dict lookup (content_registry cache
# is already warm).
_SEMANTIC_SIGNAL_TO_PREVIEW: dict[str, str] = {
    "damage": "damage",
    "block": "block",
    "draw": "draw",
    "heal": "heal",
    "weak": "weak",
    "vulnerable": "vulnerable",
    "frail": "frail",
    "strength": "strength",
    "strengthGain": "strength",
    "dexterity": "dexterity",
    "dexterityGain": "dexterity",
    "energy": "energy",
    "energyGain": "energy",
    "hpLoss": "hp_loss",
    "hp_loss": "hp_loss",
    "poison": "poison",
    "hits": "hits",
    "damage_per_hit": "damage_per_hit",
    "summon": "summon",
}


def _card_is_x_cost_from_registry(card_id: str) -> bool:
    try:
        from content_registry import get_card_metadata
        md = get_card_metadata(card_id)
    except Exception:
        return False
    if not isinstance(md, dict):
        return False
    text = str(md.get("energy_cost_text") or "").strip().upper()
    return text == "X"


def _card_effect_preview_from_registry(card_id: str) -> dict[str, Any]:
    try:
        from content_registry import get_card_metadata
        md = get_card_metadata(card_id)
    except Exception:
        md = None
    if not isinstance(md, dict):
        return {}
    signals = md.get("semantic_signals")
    if not isinstance(signals, dict):
        return {}
    preview: dict[str, Any] = {}
    for src_key, dst_key in _SEMANTIC_SIGNAL_TO_PREVIEW.items():
        if src_key in signals:
            try:
                preview[dst_key] = float(signals[src_key])
            except (TypeError, ValueError):
                continue
    # Downstream readers look for total_damage and total_block aliases
    # (_preview_metric's key_aliases). Mirror if single-hit.
    if "damage" in preview and "total_damage" not in preview:
        preview["total_damage"] = preview["damage"]
    if "block" in preview and "total_block" not in preview:
        preview["total_block"] = preview["block"]
    return preview


def _translate_player_block(
    sim_player: dict[str, Any],
    *,
    in_combat: bool,
    player_facing: str | None,
) -> dict[str, Any]:
    hp = int(sim_player.get("current_hp", sim_player.get("hp") or 0) or 0)
    max_hp = int(sim_player.get("max_hp") or 0)
    character = str(sim_player.get("character") or "IRONCLAD")
    deck_cards = [_translate_card(c, pile="Deck") for c in (sim_player.get("deck") or [])]
    return {
        "index": 0,
        "net_id": 1,
        "character": {
            "id": f"CHARACTER.{character}",
            "title": character,
            "description": "",
            "kind": character.title(),
        },
        "gold": int(sim_player.get("gold") or 0),
        "max_energy": int(sim_player.get("max_energy") or 3),
        "facing": player_facing,
        "creature": {
            "name": character,
            "model_id": f"CHARACTER.{character}",
            "combat_id": 0,
            "side": "Player",
            "current_hp": hp,
            "max_hp": max_hp,
            "block": int(sim_player.get("block") or 0),
            "is_alive": hp > 0,
            "is_hittable": hp > 0,
            "powers": _translate_player_powers(sim_player.get("status")),
            "intent": None,
            "static_traits": [],
            "reactive_triggers": [],
            "phase_rules": [],
            "combat_tags": [],
            "danger_profile": None,
            "target_priority_hints": [],
        },
        "combat": {"in_combat": in_combat},
        "deck": {
            "pile_type": "Deck",
            "is_combat_pile": False,
            "count": len(deck_cards),
            "cards": deck_cards,
        },
        "relics": [_translate_relic(r) for r in (sim_player.get("relics") or [])],
        "potions": [_translate_potion(p) for p in (sim_player.get("potions") or [])],
    }


def _translate_flat_player_block(
    sim_player: dict[str, Any],
    *,
    player_facing: str | None,
) -> dict[str, Any]:
    """Flat ``obs["player"]`` dict matching the real bridge's shape.

    Downstream consumers (combat_env reward shaping, observation_v3 encoder,
    aux_targets supervision) read fields directly off ``obs["player"]`` —
    they do not walk the nested ``obs["players"][0].creature.*`` tree.
    """
    hp = int(sim_player.get("current_hp", sim_player.get("hp") or 0) or 0)
    max_hp = int(sim_player.get("max_hp") or 0)
    character = str(sim_player.get("character") or "IRONCLAD")
    deck_cards = [_translate_card(c, pile="Deck") for c in (sim_player.get("deck") or [])]
    powers = _translate_player_powers(sim_player.get("status"))
    block = int(sim_player.get("block") or 0)
    return {
        "hp": hp,
        "current_hp": hp,
        "max_hp": max_hp,
        "block": block,
        "gold": int(sim_player.get("gold") or 0),
        "max_energy": int(sim_player.get("max_energy") or 3),
        "character": character,
        "facing": player_facing,
        "status": powers,
        "powers": powers,
        "deck_cards": deck_cards,
        "deck": len(deck_cards),
        "relics": [_translate_relic(r) for r in (sim_player.get("relics") or [])],
        "potions": [_translate_potion(p) for p in (sim_player.get("potions") or [])],
        # Legacy back-compat: aux_targets line 181 reads player["creature"]["block"]
        # as a fallback. Keep the nested form populated too.
        "creature": {
            "current_hp": hp,
            "max_hp": max_hp,
            "block": block,
        },
    }


def _translate_relic(sim_relic: Any) -> dict[str, Any]:
    if not isinstance(sim_relic, dict):
        return {}
    rid = str(sim_relic.get("id") or "")
    return {
        "id": rid if rid.startswith("RELIC.") else f"RELIC.{rid}",
        "title": str(sim_relic.get("name") or rid),
        "description": str(sim_relic.get("description") or ""),
        "rarity": str(sim_relic.get("rarity") or "Common"),
        "counter": int(sim_relic.get("counter") or 0),
    }


def _translate_potion(sim_potion: Any) -> dict[str, Any]:
    if not isinstance(sim_potion, dict):
        return {}
    pid = str(sim_potion.get("id") or "")
    return {
        "slot": int(sim_potion.get("slot") or 0),
        "id": pid if pid.startswith("POTION.") else f"POTION.{pid}",
        "title": str(sim_potion.get("name") or pid),
        "description": str(sim_potion.get("description") or ""),
        "target_type": str(sim_potion.get("target_type") or "None"),
        "can_use_in_combat": bool(sim_potion.get("can_use_in_combat", True)),
        "keywords": list(sim_potion.get("keywords") or []),
    }


def _translate_intent(intents: list[Any] | None) -> dict[str, Any]:
    if not isinstance(intents, list) or not intents:
        return {
            "intent_type": "",
            "title": "",
            "description": "",
            "total_damage": 0.0,
            "repeats": 0,
        }
    # Take the first listed intent — matches our bridge behavior.
    first = intents[0] if isinstance(intents[0], dict) else {}
    total_damage = first.get("total_damage")
    if total_damage is None:
        per_hit = first.get("damage") or 0
        repeats = first.get("repeats") or 1
        total_damage = float(per_hit) * float(repeats)
    return {
        "intent_type": str(first.get("type") or ""),
        "title": str(first.get("title") or first.get("label") or ""),
        "description": str(first.get("description") or ""),
        "total_damage": float(total_damage or 0.0),
        "damage_per_hit": float(first.get("damage") or 0.0),
        "repeats": int(first.get("repeats") or 1),
    }


def _build_enemy_trait_payloads(
    name: str, model_id: str, next_move_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Mirror of live bridge's BuildEnemyStaticTraitPayloads +
    BuildEnemyReactiveTriggerPayloads + BuildEnemyPhaseRulePayloads.
    Keyword-driven detection of combat-critical enemy traits.

    Returns (static_traits, reactive_triggers, phase_rules). Each entry has
    the 10-field trait payload shape the obs encoder reads (trait,
    description, trigger_type, condition, effect_type, effect_amount,
    severity, etc.).

    Obs encoder (_infer_enemy_traits in observation_v3.py) reads these
    three arrays directly from each enemy dict. Live bridge emits them;
    sim never did, so policy missed ~6-12 trait classes like
    summon_engine, gatekeeper, time_scaling, countdown_tick, reveal_boss.
    """
    search = " ".join([name, model_id, next_move_id]).lower()
    static_traits: list[dict[str, Any]] = []
    reactive_triggers: list[dict[str, Any]] = []
    phase_rules: list[dict[str, Any]] = []

    def _mk(category: str, trait: str, description: str,
            trigger_type: str | None = None, condition: str | None = None,
            effect_type: str | None = None, effect_amount: int | None = None,
            severity: str = "medium") -> dict[str, Any]:
        return {
            "category": category, "trait": trait, "description": description,
            "trigger_type": trigger_type, "condition": condition,
            "effect_type": effect_type, "effect_amount": effect_amount,
            "severity": severity,
        }

    # Static traits
    if any(k in search for k in ("nexus", "progenitor", "queen", "egg")):
        static_traits.append(_mk("static", "summon_engine",
            "Acts as a board-pressure engine or summon core.", severity="high"))
    if "door" in search:
        static_traits.append(_mk("static", "gatekeeper",
            "Encounter progression is gated until this unit's cycle is solved.", severity="high"))
    if any(k in search for k in ("matriarch", "nexus", "byrdonis", "wurm")):
        static_traits.append(_mk("static", "time_scaling",
            "Threat grows materially if the fight drags.", severity="high"))
    if any(k in search for k in ("retali", "thorn", "spiny")):
        static_traits.append(_mk("static", "contact_retaliate",
            "Punishes contact hits or spammy multi-hit plans.", severity="high"))

    # Reactive triggers
    if any(k in search for k in ("retali", "thorn", "spike", "spiny")):
        reactive_triggers.append(_mk("reactive", "retaliate",
            "On hit or contact, this enemy punishes damage with retaliation.",
            trigger_type="on_hit", condition="contact", effect_type="retaliate",
            severity="high"))
    if any(k in search for k in ("egg", "progenitor", "summon")):
        reactive_triggers.append(_mk("reactive", "summon",
            "If left alive or when killed, this enemy can continue board pressure via summons.",
            trigger_type="on_turn_end", condition="alive", effect_type="summon",
            severity="high"))

    # Phase rules
    if any(k in search for k in ("split", "prism")):
        phase_rules.append(_mk("phase", "split",
            "Crossing a threshold can split or multiply the board state.",
            trigger_type="on_hp_threshold", condition="threshold_crossed",
            effect_type="split", severity="medium"))
    if any(k in search for k in ("phase", "threshold", "subject", "doormaker")):
        phase_rules.append(_mk("phase", "phase_shift",
            "The enemy has threshold- or cycle-based phase changes.",
            trigger_type="on_hp_threshold", condition="threshold_crossed",
            effect_type="phase_shift", severity="high"))
    if "intang" in search:
        phase_rules.append(_mk("phase", "gain_intangible",
            "Intangible windows change when burst should be committed.",
            trigger_type="on_turn_start", condition="intangible_window",
            effect_type="gain_intangible", severity="high"))
    if "insatiable" in search:
        phase_rules.append(_mk("phase", "countdown_tick",
            "An external countdown or timer pressures the fight every turn.",
            trigger_type="on_turn_start", condition="countdown_active",
            effect_type="countdown_tick", severity="high"))
    if model_id.upper() == "MONSTER.DOOR":
        phase_rules.append(_mk("phase", "reveal_boss",
            "Destroying the door exposes the main boss window.",
            trigger_type="on_death", condition="door_destroyed",
            effect_type="reveal_boss", severity="high"))

    return static_traits, reactive_triggers, phase_rules


def _translate_combat_block(
    battle: dict[str, Any],
    sim_player: dict[str, Any],
    *,
    player_facing: str | None,
    self_inflicted: int,
) -> dict[str, Any]:
    enemies: list[dict[str, Any]] = []
    for enemy in battle.get("enemies") or []:
        if not isinstance(enemy, dict):
            continue
        powers = _translate_player_powers(enemy.get("status"))
        enemy_name = str(enemy.get("name") or enemy.get("entity_id") or "")
        enemy_model_id = str(enemy.get("entity_id") or "")
        enemy_next_move = str(enemy.get("next_move_id") or "")
        static_traits, reactive_triggers, phase_rules = _build_enemy_trait_payloads(
            enemy_name, enemy_model_id, enemy_next_move,
        )
        enemies.append({
            "id": int(enemy.get("combat_id") or 0),
            "name": enemy_name,
            "model_id": enemy_model_id,
            "hp": int(enemy.get("hp") or 0),
            "max_hp": int(enemy.get("max_hp") or 0),
            "block": int(enemy.get("block") or 0),
            "is_alive": bool(enemy.get("is_alive", True)),
            "is_hittable": bool(enemy.get("is_hittable", True)),
            "incoming_damage_multiplier": _incoming_damage_multiplier(
                enemy.get("status"), player_facing,
            ),
            "intent": _translate_intent(enemy.get("intents")),
            "powers": powers,
            "next_move_id": enemy_next_move,
            "intends_to_attack": bool(enemy.get("intends_to_attack", False)),
            # Effect algebra payloads (live parity). Obs encoder's
            # _infer_enemy_traits reads these to populate POWER_SLOT
            # structured effect vectors (Phase 6.1). Without them, sim
            # training saw zero structured effects for 6+ trait classes.
            "static_traits": static_traits,
            "reactive_triggers": reactive_triggers,
            "phase_rules": phase_rules,
        })
    hand_cards = [_translate_card(c, pile="Hand") for c in (sim_player.get("hand") or [])]
    # Sim exposes full pile lists under player.{draw,discard,exhaust}_pile.
    # Encoder's DRAW_PREVIEW_CARD / DISCARD_CARD / EXHAUST_CARD tokens look
    # for combat.{draw_pile,discard_pile,exhaust_pile} as lists (or fallback
    # keys draw_preview_cards / discard_cards / exhaust_cards). Before this,
    # the translator only emitted counts, so every cross-pile policy signal
    # (what's left to draw, what got discarded, scaling-card presence) was
    # zero-filled throughout training.
    draw_cards = [_translate_card(c, pile="Draw") for c in (sim_player.get("draw_pile") or [])]
    discard_cards = [_translate_card(c, pile="Discard") for c in (sim_player.get("discard_pile") or [])]
    exhaust_cards = [_translate_card(c, pile="Exhaust") for c in (sim_player.get("exhaust_pile") or [])]
    # Sim doesn't expose play_pile separately today; leave empty so the
    # encoder PLAY_PILE_CARD tokens mask-out cleanly. Powers still come via
    # player_powers / enemy.powers.
    play_pile_cards: list[dict[str, Any]] = []
    return {
        "round": int(battle.get("round") or 0),
        "side": str(battle.get("turn") or "Player"),
        "play_phase": bool(battle.get("is_play_phase", True)),
        "can_act": bool(battle.get("is_play_phase", True)),
        "facing": player_facing,
        "self_inflicted_hp_loss_cumulative": int(self_inflicted),
        "energy": int(sim_player.get("energy") or 0),
        "max_energy": int(sim_player.get("max_energy") or 3),
        "stars": int(sim_player.get("stars") or 0),
        "block": int(sim_player.get("block") or 0),
        "hand": hand_cards,
        "draw_pile": draw_cards,
        "discard_pile": discard_cards,
        "exhaust_pile": exhaust_cards,
        "play_pile": play_pile_cards,
        # Keep legacy count fields too — some training-time info/debug paths
        # read combat.draw / combat.discard as scalars.
        "draw": len(draw_cards) if draw_cards else int(sim_player.get("draw_pile_count") or 0),
        "discard": len(discard_cards) if discard_cards else int(sim_player.get("discard_pile_count") or 0),
        "exhaust": len(exhaust_cards) if exhaust_cards else int(sim_player.get("exhaust_pile_count") or 0),
        "allies": [],
        "enemies": enemies,
        "player_powers": _translate_player_powers(sim_player.get("status")),
    }


def _translate_map_block(sim_run: dict[str, Any], map_state: dict[str, Any]) -> dict[str, Any]:
    floor = int(sim_run.get("floor") or 0)
    current_coord = {"row": floor, "col": 0}
    points: list[dict[str, Any]] = []
    for opt in map_state.get("next_options") or []:
        if not isinstance(opt, dict):
            continue
        points.append({
            "coord": {"row": int(opt.get("row") or 0), "col": int(opt.get("col") or 0)},
            "point_type": str(opt.get("point_type") or "Monster").title(),
            "is_available": True,
        })
    return {
        "current_coord": current_coord,
        "dimensions": {"rows": 15, "cols": 7},
        "is_blocked_by_combat": False,
        "is_interactive_surface": bool(points),
        "is_open": bool(points),
        "is_open_raw": bool(points),
        "is_travel_enabled": bool(points),
        "is_travel_enabled_raw": bool(points),
        "is_traveling": False,
        "points": points,
    }


def _translate_run_block(sim_run: dict[str, Any], state_type: str, game_over: dict[str, Any]) -> dict[str, Any]:
    floor = int(sim_run.get("floor") or 0)
    act = int(sim_run.get("act") or 1)
    room_type = str(sim_run.get("room_type") or "").title() or "Monster"
    is_game_over = state_type in {"game_over", "victory"}
    # Map act 1/2/3 -> STS2 act model ids. Obs encoder's _parse_act reads the
    # trailing digit, so "ACT.UNDERDOCKS" would give 0. We need the real
    # ACT.<name> id matching the specific act played. Sim doesn't directly
    # expose act model id (just int index), but we can construct the
    # canonical form from act index + the character.
    # Act names per src/Core/Models/Acts/: UNDERDOCKS (1), HIVE (2), GLORY (3).
    act_id_canonical = {
        1: "ACT.UNDERDOCKS",
        2: "ACT.HIVE",
        3: "ACT.GLORY",
    }.get(act, "ACT.UNDERDOCKS")
    return {
        "has_run": True,
        "is_game_over": is_game_over,
        # Gate field: obs encoder reads run.active to know if we're mid-run.
        # True whenever we're past character-select and before game_over.
        # Sim doesn't expose this directly — infer from state_type.
        "active": state_type not in {"", "menu", "game_over"},
        # Obs encoder reads act_id via _parse_act which pulls the trailing
        # digit. For ACT.UNDERDOCKS/HIVE/GLORY that isn't a digit, so the
        # feature was always 0 on sim. Emit a synthesized id that ends in
        # the act number so _parse_act can extract it.
        "act_id": f"ACT.{act}",
        "current_location": f"act {act} coord (0, {floor})",
        "current_act_index": act - 1,  # bridge was 0-indexed
        "ascension_level": int(sim_run.get("ascension_level") or 0),
        # Real bridge emits all three; six downstream readers (obs encoder,
        # run_memory, combat_memory, objective_heads, env_v2 transition_state,
        # combat_env transition_state) key off bare ``floor`` and were reading
        # None from sim, zero-filling the /48-normalized floor feature and the
        # objective head's floor input.
        "floor": floor,
        "act_floor": floor,
        "total_floor": floor,
        "act": {
            "id": act_id_canonical,
            "title": "Underdocks" if act == 1 else ("Hive" if act == 2 else "Glory"),
            "description": "",
            "kind": "Underdocks" if act == 1 else ("Hive" if act == 2 else "Glory"),
        },
        "acts": [],
        "modifiers": [],
        "current_map_coord": {"row": floor, "col": 0},
        "current_map_point": {
            "coord": {"row": floor, "col": 0},
            "point_type": room_type,
        },
        "current_room": {
            "room_type": room_type,
            "model_id": str(sim_run.get("room_model_id") or ""),
            "is_pre_finished": False,
            "is_victory_room": False,
        },
        "player_count": 1,
    }
