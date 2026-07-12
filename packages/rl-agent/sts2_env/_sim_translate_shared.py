# ruff: noqa: RUF002
"""Shared state/action mapping and headless-sim derived-state helpers."""

from __future__ import annotations

from typing import Any

# Sim action kind ↔ bridge action kind. We use the bridge kinds for our
# env / obs-encoder, so this is the one-way translation used at obs build
# time. The reverse (bridge → sim) is implicit because step() dispatches
# the raw sim action dict stored under ``_sim_raw``.
_SIM_KIND_TO_BRIDGE_KIND: dict[str, str] = {
    "play_card": "play_card",
    "use_potion": "use_potion",
    "discard_potion": "use_potion",
    "end_turn": "end_turn",
    "choose_event_option": "event_option",
    "choose_map_node": "map",
    "choose_rest_option": "rest_site",
    "choose_card_reward": "card_reward",
    "skip_card_reward": "reward",
    "shop_purchase": "shop",
    "shop_skip": "proceed",
    "claim_treasure": "treasure_relic",
    "choose_character": "startup",
    "embark": "startup",
    "proceed": "proceed",
    "select_card": "card_selection",
    "select_hand_card": "card_selection",
    "confirm_selection": "card_selection",
    "cancel_selection": "card_selection",
    "choose_card_select_option": "card_selection",
    # Combat-scope card selection (Scry / Discovery / keyword-scaled card pick
    # mid-combat). Sim emits these from BuildCombatLegalActions when
    # HandSelection or CardSelection is active inside a battle. Previously
    # unmapped → they fell through to "unknown" kind and the policy/encoder
    # couldn't recognize them.
    "combat_select_card": "card_selection",
    "combat_confirm_selection": "card_selection",
    "skip_rewards": "proceed",
    "claim_reward": "reward",
    "claim_relic": "treasure_relic",
    "skip_relic_select": "proceed",
    "confirm_menu_action": "startup",
}


def sim_kind_to_bridge_kind(sim_kind: str) -> str:
    return _SIM_KIND_TO_BRIDGE_KIND.get(sim_kind, sim_kind or "unknown")


# ---------------------------------------------------------------------------
# Screen / phase mapping
# ---------------------------------------------------------------------------

_STATE_TYPE_TO_SCREEN: dict[str, str] = {
    "combat": "COMBAT",
    "battle": "COMBAT",
    "event": "EVENT",
    "map": "MAP",
    "rest_site": "REST_SITE",
    "shop": "SHOP",
    "card_reward": "REWARDS",
    "rewards": "REWARDS",
    "treasure": "REWARDS",
    "card_select": "CARD_SELECTION",
    "hand_select": "CARD_SELECTION",
    "relic_select": "REWARDS",
    "game_over": "GAME_OVER",
    "victory": "GAME_OVER",
    "menu": "MAIN_MENU",
    "character_select": "STARTUP_CHARACTER_SELECT",
    "startup": "STARTUP_CHARACTER_SELECT",
}


def _screen_from_state_type(state_type: str, *, in_combat: bool = False) -> str:
    if in_combat:
        return "COMBAT"
    return _STATE_TYPE_TO_SCREEN.get(state_type, "UNKNOWN")


def _phase_from_state(state_type: str, *, in_combat: bool = False) -> str:
    # Sim's state_type labels the ROOM (monster/elite/boss/event/...), not
    # the current interaction phase. During an active combat encounter sim
    # keeps state_type="monster" while the battle dict carries player/
    # enemies/hand. The original check `state_type in {"combat","battle"}`
    # never fires in real sim flow → every combat step was mis-tagged as
    # phase="actions", obs["combat"]={"in_progress": False}, and every
    # enemy-based aux target (enemy_state, transition.next_enemy_hp_ratio,
    # objective.damage_dealt) silently zeroed out. Trust the battle-block
    # shape instead of the state_type string.
    if in_combat:
        return "combat"
    if state_type == "event":
        return "event"
    if state_type == "map":
        return "map"
    if state_type in {"game_over", "victory"}:
        return "settling"
    if state_type in {"card_select", "hand_select"}:
        return "card_selection"
    return "actions"


# ---------------------------------------------------------------------------
# Mod-specific derived fields
# ---------------------------------------------------------------------------

# Power IDs we recognize for the back-attack mechanic (Rocket/Crusher boss).
_BACK_ATTACK_LEFT_IDS = {"BACK_ATTACK_LEFT_POWER", "BackAttackLeftPower"}
_BACK_ATTACK_RIGHT_IDS = {"BACK_ATTACK_RIGHT_POWER", "BackAttackRightPower"}
_SURROUNDED_POWER_IDS = {"SURROUNDED_POWER", "SurroundedPower"}

# Cards that deal guaranteed self-damage when played. Used to attribute HP
# loss to the player rather than to enemy damage so the per-enemy HP-loss
# aux head doesn't get a false-positive signal from self-harm plays.
_SELF_DAMAGE_CARDS: dict[str, int] = {
    # id → approximate self-damage (unupgraded). If the card scales with
    # cost / other factors we err low — better to under-attribute than
    # silently double-count.
    "BLOODLETTING": 3,
    "OFFERING": 6,
    "HEMOKINESIS": 2,
    "REAPER": 0,  # heals, not self-damage
    "FEED": 0,
}


def _power_ids_of(powers: list[Any] | None) -> set[str]:
    if not isinstance(powers, list):
        return set()
    out: set[str] = set()
    for p in powers:
        if isinstance(p, dict):
            pid = p.get("id")
            if isinstance(pid, str) and pid:
                out.add(pid.upper())
    return out


def _player_facing_from_status(status: list[Any] | None) -> str | None:
    """Read SurroundedPower.Facing from player status list. Returns
    'right' / 'left' or None if the player isn't surrounded.

    The stock sim doesn't expose the Facing enum on its serialized power
    entry — only the amount. Without the field we fall back to None
    (unknown); our bridge's Rocket-aware shaping would zero-fill this
    too. Phase 3 (C#-side) could surface the enum.
    """
    if not isinstance(status, list):
        return None
    for entry in status:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("id") or "").upper()
        if pid not in _SURROUNDED_POWER_IDS:
            continue
        # facing may be exposed as a string field or an int (0/1).
        facing = entry.get("facing")
        if isinstance(facing, str):
            f = facing.strip().lower()
            if f in {"right", "left"}:
                return f
        if isinstance(facing, int):
            return "right" if facing == 0 else "left"
        # Unknown encoding — flag presence by returning None so downstream
        # incoming_damage_multiplier stays at 1.0 rather than guessing.
        return None
    return None


def _incoming_damage_multiplier(enemy_powers: list[Any] | None, player_facing: str | None) -> float:
    """Mirror our bridge's ``ComputeIncomingDamageMultiplier``.

    1.5× when the player faces RIGHT and enemy has BackAttackLeftPower, or
    the mirror case. 1.0× otherwise.
    """
    if player_facing is None:
        return 1.0
    ids = _power_ids_of(enemy_powers)
    has_left = bool(ids & _BACK_ATTACK_LEFT_IDS)
    has_right = bool(ids & _BACK_ATTACK_RIGHT_IDS)
    if player_facing == "right" and has_left:
        return 1.5
    if player_facing == "left" and has_right:
        return 1.5
    return 1.0


class SelfInflictedHpTracker:
    """Tracks cumulative HP loss attributable to the player's own actions
    within a combat. Reset when combat starts; advanced when a known
    self-damage card is played and the player's HP subsequently drops.

    Not reconstructible from a single snapshot — this object lives on the
    bridge client and is fed both the last action dispatched and the next
    observation's HP.
    """

    def __init__(self) -> None:
        self.cumulative: int = 0
        self._last_hp: int | None = None
        self._last_self_damage_card: str | None = None

    def reset(self, initial_hp: int | None) -> None:
        self.cumulative = 0
        self._last_hp = int(initial_hp) if isinstance(initial_hp, int) else None
        self._last_self_damage_card = None

    def note_action(self, action_raw: dict[str, Any] | None) -> None:
        """Called when the env dispatches an action. If it's playing a
        known self-damage card, remember the card id so the next HP drop
        can be attributed.
        """
        if not isinstance(action_raw, dict):
            self._last_self_damage_card = None
            return
        if str(action_raw.get("action") or "") != "play_card":
            self._last_self_damage_card = None
            return
        card_id = str(action_raw.get("card_id") or "").upper()
        # Strip CARD. prefix variants
        if card_id.startswith("CARD."):
            card_id = card_id.split(".", 1)[1]
        if card_id in _SELF_DAMAGE_CARDS:
            self._last_self_damage_card = card_id
        else:
            self._last_self_damage_card = None

    def observe_hp(self, current_hp: int | None) -> None:
        """Called after each step with the player's new HP. Attributes a
        drop to self-damage iff the previous action was a self-harm card.
        """
        if not isinstance(current_hp, int):
            return
        if self._last_hp is not None and self._last_self_damage_card is not None:
            delta = max(self._last_hp - current_hp, 0)
            if delta > 0:
                # Cap by the card's approximate self-damage — keeps enemy
                # damage dealt on the same tick from leaking in.
                cap = _SELF_DAMAGE_CARDS.get(self._last_self_damage_card, 0)
                if cap > 0:
                    delta = min(delta, cap)
                self.cumulative += int(delta)
        self._last_hp = int(current_hp)
        self._last_self_damage_card = None
