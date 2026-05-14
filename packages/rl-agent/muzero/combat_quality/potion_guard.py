"""Potion-specific hard-guard helpers for combat action quality.

These functions were originally local helpers inside
``MuZeroTrainer._apply_combat_action_hard_guards``.  They are intentionally kept
pure and narrow so boss/potion emergency exceptions can be tested without
constructing the full trainer.
"""

from __future__ import annotations

import re
from typing import Any


def potion_slot_from_action_for_guard(action: Any) -> int | None:
    """Best-effort potion slot extraction from compact legal actions.

    The bridge is not fully consistent about potion action payloads.  In the
    good case we get a nested ``potion`` dict; in the bad case the only stable
    identity is the slot embedded in strings such as ``use_potion:0:self``.
    Current live bridge ids are ``use_potion:{playerIndex}:{slotIndex}:...``
    and ``discard_potion:{playerIndex}:{slotIndex}``, so structured parsing
    must prefer the *second* numeric segment when two leading numeric segments
    are present.  This matters for overflow discard guards: parsing
    ``discard_potion:0:1`` as slot 0 makes every candidate look like the first
    potion and can throw away a high-value Lucky Tonic.

    Keep this helper intentionally permissive but bounded to small potion slot
    indices so it cannot accidentally parse unrelated ids as inventory slots.
    """

    if not isinstance(action, dict):
        return None

    def _bounded_int(value: Any) -> int | None:
        try:
            if value is None or str(value).strip() == "":
                return None
            slot = int(value)
            if 0 <= slot < 10:
                return slot
        except (TypeError, ValueError):
            return None
        return None

    sources: list[dict[str, Any]] = [action]
    for key in ("target", "potion", "payload"):
        value = action.get(key)
        if isinstance(value, dict):
            sources.append(value)

    for source in sources:
        for key in ("potion_slot", "slot", "slot_index", "potion_index", "potion_idx", "index"):
            slot = _bounded_int(source.get(key))
            if slot is not None:
                return slot

    for key in ("action_id", "id", "selection_group_key", "name", "label", "display_name", "displayName"):
        text = str(action.get(key) or "")
        if not text:
            continue
        # Examples observed/expected:
        #   use_potion:0:self
        #   use_potion:0:0:self
        #   use_potion_0
        #   potion[0]
        #   discard_potion:0
        #   discard_potion:0:1
        structured = re.match(r"^\s*(use_potion|discard_potion):(.+)$", text, flags=re.IGNORECASE)
        if structured:
            parts = structured.group(2).split(":")
            leading_numbers: list[int] = []
            for part in parts:
                if not re.fullmatch(r"\d+", part.strip()):
                    break
                try:
                    leading_numbers.append(int(part.strip()))
                except ValueError:
                    break
            # Live bridge: {playerIndex}:{slotIndex}[:target...].  Legacy:
            # {slotIndex}[:target...].  Therefore two leading numeric fields
            # means the second one is the inventory slot; one means the first.
            if len(leading_numbers) >= 2:
                slot = _bounded_int(leading_numbers[1])
                if slot is not None:
                    return slot
            elif len(leading_numbers) == 1:
                slot = _bounded_int(leading_numbers[0])
                if slot is not None:
                    return slot

        match = re.search(
            r"(?:use[_: \-]?potion|discard[_: \-]?potion|potion)[_: \-\[]+(\d+)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            slot = _bounded_int(match.group(1))
            if slot is not None:
                return slot
    return None


def raw_potion_payload_for_guard(raw_obs: Any | None, action: Any) -> dict[str, Any] | None:
    """Resolve a slot-only potion action against raw observation inventory.

    This is the critical fallback for boss/elite death windows: legal actions
    can be reduced to ``{"action_id": "use_potion:0:self"}`` while the actual
    identity (Lucky Tonic / 幸运药剂) only exists in ``raw_obs.player.potions``.
    """

    slot = potion_slot_from_action_for_guard(action)
    if slot is None or not isinstance(raw_obs, dict):
        return None

    containers: list[tuple[str, Any]] = [("$", raw_obs)]
    for key in ("state", "raw_obs", "transition_state", "observation", "obs"):
        payload = raw_obs.get(key)
        if isinstance(payload, dict):
            containers.append((f"$.{key}", payload))

    for _prefix, obs in containers:
        if not isinstance(obs, dict):
            continue
        potion_lists: list[Any] = [obs.get("potions")]
        player = obs.get("player")
        if isinstance(player, dict):
            potion_lists.append(player.get("potions"))
        run = obs.get("run")
        if isinstance(run, dict):
            potion_lists.append(run.get("potions"))
        combat = obs.get("combat")
        if isinstance(combat, dict):
            potion_lists.append(combat.get("potions"))
            combat_player = combat.get("player")
            if isinstance(combat_player, dict):
                potion_lists.append(combat_player.get("potions"))

        for potions in potion_lists:
            if not isinstance(potions, list) or not (0 <= slot < len(potions)):
                continue
            payload = potions[slot]
            if isinstance(payload, dict):
                return payload
            text = str(payload or "").strip()
            if text and text.lower() not in {"[empty]", "empty", "none", "null"}:
                return {"title": text}
    return None


def potion_identity_text_for_guard(action: Any, profile: dict[str, Any], raw_obs: Any | None = None) -> str:
    """Best-effort potion identity text for narrow hard-guard classification.

    Bridge legal actions are not fully stable across compact/raw surfaces:
    some carry ``potion.id/title`` while others only expose a localized
    label or an action_id.  The timing profile may also have resolved the
    potion id from ``raw_obs.player.potions``.  Keep this helper local so
    the Act1 emergency guard does not broaden global potion semantics.
    """
    parts: list[str] = []
    if isinstance(profile, dict):
        for key in ("potion_id", "rarity"):
            value = str(profile.get(key) or "").strip()
            if value:
                parts.append(value)
        for key in ("effect_family", "semantic_tags", "timing_tags", "training_tags"):
            values = profile.get(key)
            if isinstance(values, (list, tuple, set)):
                parts.extend(str(v or "").strip() for v in values if str(v or "").strip())
    if isinstance(action, dict):
        containers: list[Any] = [action]
        for key in ("potion", "potion_info", "item"):
            payload = action.get(key)
            if isinstance(payload, dict):
                containers.append(payload)
        for container in containers:
            if not isinstance(container, dict):
                continue
            for key in (
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
            ):
                value = str(container.get(key) or "").strip()
                if value:
                    parts.append(value)
    raw_payload = raw_potion_payload_for_guard(raw_obs, action)
    if isinstance(raw_payload, dict):
        containers = [raw_payload]
        for key in ("potion", "potion_info", "item"):
            payload = raw_payload.get(key)
            if isinstance(payload, dict):
                containers.append(payload)
        for container in containers:
            for key in (
                "id",
                "potion_id",
                "potionId",
                "model_id",
                "modelId",
                "normalized_id",
                "normalizedId",
                "title",
                "name",
                "label",
                "display_name",
                "displayName",
                "title_en",
                "title_zhs",
                "localized_title",
                "localizedTitle",
                "description",
                "summary",
            ):
                value = str(container.get(key) or "").strip()
                if value:
                    parts.append(value)
    return " ".join(parts).lower()


def is_lucky_survival_potion_for_guard(
    action: Any,
    profile: dict[str, Any] | None,
    raw_obs: Any | None = None,
) -> bool:
    """Return true for Lucky Tonic / 幸运药剂 across compact bridge variants.

    The bridge has exposed this potion in several incompatible shapes during
    full-run training:

    * timing profile has ``potion_id=POTION.LUCKY_TONIC``;
    * legal action only carries localized title ``幸运补剂`` / ``幸运药剂``;
    * legal action is slot-only (``use_potion:0:self``) and identity exists only
      in ``raw_obs.player.potions[0]``.

    Keep the identity predicate centralized so boss-race, boss-survival, and
    future diagnostics cannot silently diverge again.
    """

    profile_dict = profile if isinstance(profile, dict) else {}
    potion_id = str(profile_dict.get("potion_id") or "").strip().upper()
    identity = potion_identity_text_for_guard(action, profile_dict, raw_obs)
    return bool(
        "LUCKY_TONIC" in potion_id
        or ("LUCKY" in potion_id and "TONIC" in potion_id)
        or "lucky_tonic" in identity
        or "lucky tonic" in identity
        or ("lucky" in identity and "tonic" in identity)
        or "幸运补剂" in identity
        or "幸运药剂" in identity
        or "幸運補劑" in identity
        or "幸運藥劑" in identity
    )


def boss_race_potion_traits(action: Any, profile: dict[str, Any], raw_obs: Any | None = None) -> dict[str, Any]:
    """Classify boss-only race/setup potions for the Act1 recovery guard.

    This deliberately recognises only high-confidence fight-progress or
    survival potions: Strength/Flex-style scaling, Glowing Water/burst draw,
    direct damage, Liquid-Memories-like retrieve when the discard pile actually
    has a target, and buffer/prevent-damage tools such as Lucky Tonic in a boss
    race/survival window.  It excludes known no-op contexts such as Fortifier
    at zero current block and Liquid Memories with empty discard.
    """
    if not isinstance(profile, dict):
        return {
            "candidate": False,
            "setup_like": False,
            "strength_like": False,
            "dex_like": False,
            "burst_draw_like": False,
            "damage_like": False,
            "retrieve_tool": False,
            "buffer_like": False,
            "survival_like": False,
            "prevent_damage_like": False,
            "invalid_context": True,
            "identity": "",
        }
    tags = {
        str(x).strip().lower()
        for key in ("effect_family", "semantic_tags", "timing_tags", "training_tags")
        for x in (profile.get(key) or [])
        if str(x).strip()
    }
    potion_id = str(profile.get("potion_id") or "").upper()
    identity = potion_identity_text_for_guard(action, profile, raw_obs)
    draw = float(profile.get("draw", 0.0) or 0.0)
    damage = float(profile.get("damage", 0.0) or 0.0)
    prevent_damage = float(profile.get("prevent_damage", 0.0) or 0.0)
    retrieve_like = bool(
        bool(profile.get("retrieve_from_discard_like", False))
        or float(profile.get("retrieve_from_discard", 0.0) or 0.0) > 0.0
        or "LIQUID_MEMORIES" in potion_id
        or ("discard_pile" in tags and "tutor" in tags)
    )
    retrieve_has_target = bool(profile.get("retrieve_has_target", False))
    retrieve_tool = bool(retrieve_like and retrieve_has_target)
    strength_like = bool(
        "strength" in tags
        or "scaling" in tags
        or "scaling_tool" in tags
        or "STRENGTH" in potion_id
        or "FLEX" in potion_id
        or "力量" in identity
        or "肌肉" in identity
        or "strength" in identity
        or "flex" in identity
    )
    dex_like = bool(
        "dexterity" in tags
        or "dex" in tags
        or "SPEED" in potion_id
        or "DEXTERITY" in potion_id
        or "速度" in identity
        or "敏捷" in identity
        or "dexterity" in identity
        or "dex" in identity
        or "speed" in identity
    )
    burst_draw_like = bool(
        "GLOWWATER" in potion_id
        or "GLOWING" in potion_id
        or "发光" in identity
        or (
            draw >= 8.0
            and (
                "draw" in tags
                or "burst" in tags
                or "burst_tool" in tags
            )
        )
    )
    damage_like = bool(damage > 0.0)
    buffer_like = bool(
        prevent_damage > 0.0
        or "buffer" in tags
        or "prevent_damage" in tags
        or is_lucky_survival_potion_for_guard(action, profile, raw_obs)
    )
    survival_like = bool(
        buffer_like
        or bool(profile.get("prevent_lethal", False))
        or bool(profile.get("prevent_major_loss", False))
        or bool(profile.get("resource_survival_tool", False))
        or "survival" in tags
        or "survival_tool" in tags
        or "boss_survival_tool" in tags
        or "prevent_lethal_tool" in tags
        or "prevent_major_loss_tool" in tags
    )
    amplify_noop = bool(profile.get("amplify_block_noop", False))
    empty_retrieve = bool(retrieve_like and not retrieve_has_target)
    invalid_context = bool(amplify_noop or empty_retrieve)
    setup_like = bool(strength_like or dex_like or burst_draw_like or retrieve_tool)
    candidate = bool((setup_like or damage_like or survival_like) and not invalid_context)
    return {
        "candidate": candidate,
        "setup_like": setup_like,
        "strength_like": strength_like,
        "dex_like": dex_like,
        "burst_draw_like": burst_draw_like,
        "damage_like": damage_like,
        "retrieve_tool": retrieve_tool,
        "buffer_like": buffer_like,
        "survival_like": survival_like,
        "prevent_damage_like": bool(prevent_damage > 0.0),
        "invalid_context": invalid_context,
        "identity": identity,
    }

def lagavulin_setup_window_for_potion(
    raw_obs_for_check: Any | None,
    *,
    encounter_tier: str,
    encounter_hint: str | None,
    threat_gap: float,
    current_energy: float,
    no_non_potion_alt: bool,
) -> bool:
    """Detect the narrow Lagavulin opening/stun race window.

    Act1 recovery diagnostics showed a specific bad target rewrite:
    at 0 energy the legal surface was only ``End Turn`` plus Liquid
    Memories, while Lagavulin Matriarch was still asleep/vulnerable and
    dealing 0 incoming.  The generic potion-waste guard saw an
    empty-discard Liquid Memories and rewrote it to End Turn, wasting
    the boss damage window.  This helper intentionally does *not* make
    Liquid Memories globally good: it requires boss + Lagavulin marker +
    0-energy + no playable non-potion action + no incoming damage + an
    explicit opening marker (ASLEEP/SLEEP/STUN or early high-HP
    Vulnerable).  Idle empty Liquid Memories without those markers
    remains blocked by the existing tests/guards.
    """
    if encounter_tier != "boss":
        return False
    if current_energy > 0.05 or threat_gap > 0.05 or not no_non_potion_alt:
        return False
    obs = raw_obs_for_check if isinstance(raw_obs_for_check, dict) else {}
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
    haystack_parts: list[str] = [
        str(encounter_hint or ""),
        str(obs.get("encounter") or ""),
        str(obs.get("encounter_id") or ""),
        str(obs.get("room_model") or ""),
        str(obs.get("roomModel") or ""),
    ]
    lagavulin_seen = False
    setup_marker = False
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        enemy_parts = [
            str(enemy.get(key) or "")
            for key in (
                "id",
                "model_id",
                "monster_id",
                "name",
                "title",
                "encounter_id",
            )
        ]
        intent = enemy.get("intent")
        if isinstance(intent, dict):
            enemy_parts.extend(
                str(intent.get(key) or "")
                for key in (
                    "id",
                    "intent",
                    "intent_type",
                    "type",
                    "name",
                    "title",
                    "description",
                )
            )
            try:
                if float(intent.get("total_damage") or 0.0) > 0.05:
                    return False
            except (TypeError, ValueError):
                pass
        else:
            enemy_parts.append(str(intent or ""))
        power_ids: list[str] = []
        for power in enemy.get("powers") or []:
            if isinstance(power, dict):
                power_ids.append(str(power.get("id") or power.get("model_id") or power.get("name") or ""))
            else:
                power_ids.append(str(power or ""))
        enemy_parts.extend(power_ids)
        enemy_text = " ".join(enemy_parts).upper()
        haystack_parts.append(enemy_text)
        if "LAGAVULIN" not in enemy_text and "MATRIARCH" not in enemy_text:
            continue
        lagavulin_seen = True
        if "ASLEEP_POWER" in enemy_text or "SLEEP" in enemy_text or "STUN" in enemy_text:
            setup_marker = True
        if "VULNERABLE_POWER" in enemy_text:
            try:
                enemy_hp = float(enemy.get("current_hp") or enemy.get("hp") or 0.0)
            except (TypeError, ValueError):
                enemy_hp = 0.0
            # Lagavulin starts around 222 HP in the current Act1 boss
            # build.  A still-high vulnerable Lagavulin at 0 incoming is
            # the same opening/race window even if ASLEEP_POWER was
            # removed from the live payload after the first action.
            if enemy_hp >= 120.0:
                setup_marker = True
    haystack = " ".join(haystack_parts).upper()
    if not lagavulin_seen and ("LAGAVULIN" in haystack or "MATRIARCH" in haystack):
        lagavulin_seen = True
    return bool(lagavulin_seen and setup_marker)

def lagavulin_setup_liquid_escape(
    action: Any,
    profile: dict[str, Any],
    raw_obs_for_check: Any | None,
    *,
    encounter_tier: str,
    encounter_hint: str | None,
    threat_gap: float,
    current_energy: float,
    no_non_potion_alt: bool,
) -> bool:
    if not isinstance(profile, dict):
        return False
    if not lagavulin_setup_window_for_potion(
        raw_obs_for_check,
        encounter_tier=encounter_tier,
        encounter_hint=encounter_hint,
        threat_gap=threat_gap,
        current_energy=current_energy,
        no_non_potion_alt=no_non_potion_alt,
    ):
        return False
    tags = {
        str(x).strip().lower()
        for key in ("effect_family", "semantic_tags", "timing_tags", "training_tags")
        for x in (profile.get(key) or [])
        if str(x).strip()
    }
    potion_id = str(profile.get("potion_id") or "").upper()
    identity = potion_identity_text_for_guard(action, profile, raw_obs_for_check)
    return bool(
        bool(profile.get("retrieve_from_discard_like", False))
        or float(profile.get("retrieve_from_discard", 0.0) or 0.0) > 0.0
        or "LIQUID_MEMORIES" in potion_id
        or "liquid_memories" in identity
        or "液态记忆" in identity
        or ("discard_pile" in tags and "tutor" in tags)
    )

def boss_zero_energy_liquid_escape(
    action: Any,
    profile: dict[str, Any],
    raw_obs: Any | None = None,
    *,
    encounter_tier: str,
    hp: float,
    max_hp: float,
    hp_ratio: float,
    threat_gap: float,
    current_energy: float,
    no_non_potion_alt: bool,
) -> bool:
    """Narrow Act1-boss escape hatch for Liquid Memories.

    The bridge can transiently report an empty discard pile even after
    the hand was just spent, so Liquid Memories is sometimes classified
    as ``empty_retrieve`` exactly when it is the only 0-energy boss
    escape.  Keep the exception boss-only, 0-energy-only, and require
    visible incoming pressure (or truly critical HP) so idle empty
    Liquid Memories remains blocked by the generic bad-use guard.
    """
    if not isinstance(profile, dict) or encounter_tier != "boss":
        return False
    if current_energy > 0.05 or not no_non_potion_alt:
        return False
    tags = {
        str(x).strip().lower()
        for key in ("effect_family", "semantic_tags", "timing_tags", "training_tags")
        for x in (profile.get(key) or [])
        if str(x).strip()
    }
    potion_id = str(profile.get("potion_id") or "").upper()
    identity = potion_identity_text_for_guard(action, profile, raw_obs)
    retrieve_like = bool(
        bool(profile.get("retrieve_from_discard_like", False))
        or float(profile.get("retrieve_from_discard", 0.0) or 0.0) > 0.0
        or "LIQUID_MEMORIES" in potion_id
        or "liquid_memories" in identity
        or "液态记忆" in identity
        or ("discard_pile" in tags and "tutor" in tags)
    )
    if not retrieve_like:
        return False
    effective_hp = float(hp)
    profile_hp = float(profile.get("hp", 0.0) or 0.0)
    if effective_hp <= 0.0 and profile_hp > 0.0:
        effective_hp = profile_hp
    effective_max_hp = float(max_hp)
    profile_max_hp = float(profile.get("max_hp", 0.0) or 0.0)
    if effective_max_hp <= 0.0 and profile_max_hp > 0.0:
        effective_max_hp = profile_max_hp
    effective_hp_ratio = float(hp_ratio)
    if effective_hp_ratio <= 0.0 and effective_hp > 0.0 and effective_max_hp > 0.0:
        effective_hp_ratio = effective_hp / max(effective_max_hp, 1.0)
    pressure = float(threat_gap)
    if pressure <= 0.05:
        # Preserve the existing idle-waste behavior at 25/91 HP; only
        # fail open on idle turns when the player is already in the
        # single-cycle critical band.
        return bool(effective_hp_ratio <= 0.15)
    return bool(
        pressure >= max(1.0, effective_hp - 1.0)
        or (effective_hp_ratio <= 0.60 and pressure >= 6.0)
        or (effective_hp_ratio <= 0.45 and pressure >= max(4.0, 0.20 * max(effective_hp, 1.0)))
    )

def boss_zero_energy_block_potion_escape(
    action: Any,
    profile: dict[str, Any],
    raw_obs: Any | None = None,
    *,
    encounter_tier: str,
    threat_gap: float,
    current_energy: float,
    no_non_potion_alt: bool,
) -> bool:
    """Boss-only 0-energy block potion escape.

    This primarily covers Fortifier with existing block: at 0 block it
    is a pure no-op and must stay rejected, but with current block and
    incoming damage it is concrete survivability and should beat End
    Turn when no card is playable.
    """
    if not isinstance(profile, dict) or encounter_tier != "boss":
        return False
    if current_energy > 0.05 or not no_non_potion_alt or threat_gap <= 0.05:
        return False
    if bool(profile.get("amplify_block_noop", False)):
        return False
    block_value = float(profile.get("block", 0.0) or 0.0)
    if block_value <= 0.05:
        return False
    tags = {
        str(x).strip().lower()
        for key in ("effect_family", "semantic_tags", "timing_tags", "training_tags")
        for x in (profile.get(key) or [])
        if str(x).strip()
    }
    potion_id = str(profile.get("potion_id") or "").upper()
    identity = potion_identity_text_for_guard(action, profile, raw_obs)
    return bool(
        "block" in tags
        or "defense" in tags
        or "defensive" in tags
        or "amplify_block" in tags
        or "FORTIFIER" in potion_id
        or "BLOCK" in potion_id
        or "固化" in identity
        or "block" in identity
    )


__all__ = [
    "boss_race_potion_traits",
    "boss_zero_energy_block_potion_escape",
    "boss_zero_energy_liquid_escape",
    "is_lucky_survival_potion_for_guard",
    "lagavulin_setup_liquid_escape",
    "lagavulin_setup_window_for_potion",
    "potion_identity_text_for_guard",
    "potion_slot_from_action_for_guard",
    "raw_potion_payload_for_guard",
]
