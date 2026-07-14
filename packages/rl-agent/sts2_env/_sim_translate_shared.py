"""Protocol enum translation shared by the headless transport adapter.

This module intentionally contains no gameplay inference.  It only maps enum
names used by the simulator to the corresponding names used at the typed
environment boundary.
"""

from __future__ import annotations

_SIM_KIND_TO_TRANSPORT_KIND: dict[str, str] = {
    "start_run": "startup",
    "choose_event_option": "event_option",
    "choose_map_node": "map",
    "choose_rest_option": "rest_site",
    "choose_card_reward": "card_reward",
    "select_card_reward": "card_reward",
    "skip_card_reward": "reward",
    "shop_purchase": "shop",
    "shop_skip": "proceed",
    "claim_treasure": "treasure_relic",
    "claim_treasure_relic": "treasure_relic",
    "choose_character": "startup",
    "embark": "startup",
    "select_card": "card_selection",
    "deselect_card": "card_selection",
    "select_hand_card": "card_selection",
    "deselect_hand_card": "card_selection",
    "confirm_selection": "card_selection",
    "cancel_selection": "card_selection",
    "choose_card_select_option": "card_selection",
    "select_card_option": "card_selection",
    "deselect_card_option": "card_selection",
    "combat_select_card": "card_selection",
    "combat_deselect_card": "card_selection",
    "combat_confirm_selection": "card_selection",
    "combat_cancel_selection": "card_selection",
    "skip_rewards": "proceed",
    "claim_reward": "reward",
    "claim_relic": "treasure_relic",
    "skip_relic_select": "proceed",
    "select_relic": "treasure_relic",
    "skip_relic_selection": "proceed",
    "confirm_menu_action": "startup",
}

_SIM_KIND_TO_MODEL_KIND: dict[str, str] = {
    "start_run": "main_menu",
    "play_card": "play_card",
    "end_turn": "end_turn",
    "use_potion": "use_potion",
    "discard_potion": "discard_potion",
    "proceed": "proceed",
    "choose_event_option": "event_option",
    "choose_map_node": "map",
    "choose_rest_option": "rest_site",
    "choose_card_reward": "card_reward",
    "select_card_reward": "card_reward",
    "skip_card_reward": "card_reward",
    "shop_purchase": "shop",
    "shop_skip": "shop",
    "claim_treasure": "treasure",
    "claim_treasure_relic": "treasure_relic",
    "claim_relic": "treasure_relic",
    "choose_character": "character_select",
    "embark": "character_select",
    "select_card": "card_selection",
    "deselect_card": "card_selection",
    "select_hand_card": "card_selection",
    "deselect_hand_card": "card_selection",
    "confirm_selection": "card_selection",
    "cancel_selection": "card_selection",
    "choose_card_select_option": "card_selection",
    "select_card_option": "card_selection",
    "deselect_card_option": "card_selection",
    "combat_select_card": "card_selection",
    "combat_deselect_card": "card_selection",
    "combat_confirm_selection": "card_selection",
    "combat_cancel_selection": "card_selection",
    "skip_rewards": "proceed",
    "claim_reward": "reward",
    "skip_relic_select": "treasure_relic",
    "select_relic": "treasure_relic",
    "skip_relic_selection": "treasure_relic",
    "confirm_menu_action": "main_menu",
    "choose_run_mode": "run_mode_selection",
    "game_over_continue": "game_over",
    "return_to_main_menu": "game_over",
}

_SIM_SELECTION_OPERATION: dict[str, str] = {
    "select_card": "select",
    "select_hand_card": "select",
    "select_card_option": "select",
    "combat_select_card": "select",
    "deselect_card": "deselect",
    "deselect_hand_card": "deselect",
    "deselect_card_option": "deselect",
    "combat_deselect_card": "deselect",
    "confirm_selection": "confirm",
    "combat_confirm_selection": "confirm",
    "cancel_selection": "cancel_prompt",
    "combat_cancel_selection": "cancel_prompt",
}


def sim_kind_to_bridge_kind(sim_kind: str) -> str:
    """Translate a simulator action enum without changing action legality."""

    normalized = str(sim_kind or "").strip()
    return _SIM_KIND_TO_TRANSPORT_KIND.get(normalized, normalized or "unknown")


def sim_kind_to_model_kind(sim_kind: str) -> str:
    """Map simulator enums to the closed live model-action vocabulary."""

    normalized = str(sim_kind or "").strip()
    try:
        return _SIM_KIND_TO_MODEL_KIND[normalized]
    except KeyError as exc:
        raise ValueError(
            f"simulator action enum has no canonical model mapping: {normalized!r}"
        ) from exc


def sim_kind_to_selection_operation(sim_kind: str) -> str | None:
    """Return the exact mutation performed by a selection action enum."""

    return _SIM_SELECTION_OPERATION.get(str(sim_kind or "").strip())


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
    return _STATE_TYPE_TO_SCREEN.get(str(state_type).lower(), "UNKNOWN")


def _phase_from_state(state_type: str, *, in_combat: bool = False) -> str:
    """Map the simulator state enum to the transport phase enum.

    ``battle`` presence is supplied explicitly because the simulator may keep
    a room type in ``state_type`` while an encounter is active.
    """

    normalized = str(state_type).lower()
    if in_combat:
        return "combat"
    if normalized == "event":
        return "event"
    if normalized == "map":
        return "map"
    if normalized in {"game_over", "victory"}:
        return "settling"
    if normalized in {"card_select", "hand_select"}:
        return "card_selection"
    return "actions"


def _decision_domain_from_phase(phase: str, *, in_combat: bool = False) -> str:
    if phase == "combat":
        return "combat"
    if phase == "map":
        return "route"
    if phase in {"card_selection", "settling"} and in_combat:
        return "combat"
    return "build"


__all__ = [
    "sim_kind_to_bridge_kind",
    "sim_kind_to_model_kind",
    "sim_kind_to_selection_operation",
]
