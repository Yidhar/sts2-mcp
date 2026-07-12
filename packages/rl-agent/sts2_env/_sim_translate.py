"""Translation facade: HeadlessSim state dict to bridge-shaped dict.

The domain translators are pure and live in focused private modules.  This file
keeps the historical import surface used by ``HeadlessSimBridgeClient``.
"""

from __future__ import annotations

from typing import Any

from ._sim_translate_actions import (
    _translate_legal_actions,
)
from ._sim_translate_decisions import (
    _build_decision_block,
    _translate_card_reward_sel_block,
    _translate_card_sel_block,
    _translate_event_options,
    _translate_rest_site_block,
    _translate_rewards_block,
    _translate_shop_block,
)
from ._sim_translate_entities import (
    _translate_combat_block,
    _translate_flat_player_block,
    _translate_map_block,
    _translate_player_block,
    _translate_run_block,
)
from ._sim_translate_shared import (
    SelfInflictedHpTracker,
    _phase_from_state,
    _player_facing_from_status,
    _screen_from_state_type,
    sim_kind_to_bridge_kind,
)


def translate_to_bridge_shape(
    sim_state: dict[str, Any],
    *,
    episode_id: str,
    self_inflicted_tracker: SelfInflictedHpTracker | None = None,
) -> dict[str, Any]:
    """Main entry: sim state dict → bridge-shaped dict the obs encoder reads."""
    state_type = str(sim_state.get("state_type") or "").lower()

    # Every sub-state section (when present) carries the current player
    # snapshot — we prefer battle.player when in combat, otherwise the
    # first available.
    sim_run = sim_state.get("run") or {}
    battle = sim_state.get("battle") or {}
    event = sim_state.get("event") or {}
    map_state = sim_state.get("map") or {}
    rest_site = sim_state.get("rest_site") or {}
    shop = sim_state.get("shop") or {}
    treasure = sim_state.get("treasure") or {}
    rewards_state = sim_state.get("rewards") or {}
    card_reward = sim_state.get("card_reward") or {}
    card_select = sim_state.get("card_select") or {}
    hand_select = sim_state.get("hand_select") or {}
    relic_select = sim_state.get("relic_select") or {}
    game_over = sim_state.get("game_over") or {}

    sim_player = (
        battle.get("player")
        or event.get("player")
        or map_state.get("player")
        or rest_site.get("player")
        or shop.get("player")
        or treasure.get("player")
        or rewards_state.get("player")
        or card_reward.get("player")
        or card_select.get("player")
        or hand_select.get("player")
        or relic_select.get("player")
        or sim_state.get("player")
        or {}
    )

    # Sim labels the *room* (monster/elite/boss) in state_type and carries
    # the live fight in the ``battle`` block. The presence of battle.player
    # (or battle.enemies) is the authoritative "we are currently mid-combat"
    # signal — much more reliable than state_type string matching.
    in_combat = (
        state_type in {"combat", "battle"}
        or bool(battle.get("player"))
        or bool(battle.get("enemies"))
    )
    screen = _screen_from_state_type(state_type, in_combat=in_combat)
    player_facing = _player_facing_from_status(sim_player.get("status"))

    sim_legal_actions = sim_state.get("legal_actions") or []

    # --- Combat block ---
    bridge_combat: dict[str, Any] = {"in_progress": in_combat}
    if in_combat and battle:
        bridge_combat.update(_translate_combat_block(
            battle, sim_player,
            player_facing=player_facing,
            self_inflicted=self_inflicted_tracker.cumulative if self_inflicted_tracker else 0,
        ))

    # --- Player / deck / relics / potions ---
    # Two shapes are needed: the legacy nested `players[0]` block (list with
    # `creature.current_hp` etc.) AND a flat `player` dict that all downstream
    # consumers (reward shaping in combat_env._player_hp_delta_reward, obs
    # encoder in observation_v3, aux_targets in aux_targets.py) actually read.
    # The real bridge emits both; the sim translator was only emitting the
    # nested list form, so every player-side reward and player-side obs
    # feature was zero-filled during sim training.
    bridge_players = [_translate_player_block(
        sim_player,
        in_combat=in_combat,
        player_facing=player_facing,
    )]
    bridge_player = _translate_flat_player_block(
        sim_player,
        player_facing=player_facing,
    )

    # --- Map ---
    bridge_map = _translate_map_block(sim_run, map_state)

    # --- Rewards / shops / rest / treasure / card-reward / card-select ---
    bridge_rewards = _translate_rewards_block(rewards_state, card_reward, treasure, relic_select)
    bridge_rest_site = _translate_rest_site_block(rest_site)
    bridge_shop = _translate_shop_block(shop)
    bridge_card_reward_sel = _translate_card_reward_sel_block(card_reward)
    bridge_card_sel = _translate_card_sel_block(card_select, hand_select, battle.get("card_selection") if battle else None)

    # --- Event options (with effect_deltas TODO) ---
    event_options = _translate_event_options(event)

    # --- Run block ---
    bridge_run = _translate_run_block(sim_run, state_type, game_over)

    # --- Legal actions (rich bridge-shaped entries) ---
    bridge_actions = _translate_legal_actions(
        sim_legal_actions,
        sim_player=sim_player,
        battle=battle,
        map_state=map_state,
        event=event,
        rest_site=rest_site,
        shop=shop,
        card_reward=card_reward,
        card_select=card_select,
        treasure=treasure,
    )

    phase = _phase_from_state(state_type, in_combat=in_combat)
    # Mirror live ResolveEnvDecisionDomain (BridgeGameApi.EnvPayloads.cs:442):
    # combat→combat, map→route, card_selection/settling→combat if in-combat
    # else build, everything else→build. Obs encoder (_resolve_domain at
    # observation_common.py:1547) reads this top-level.
    if phase == "combat":
        decision_domain = "combat"
    elif phase == "map":
        decision_domain = "route"
    elif phase in {"card_selection", "settling"}:
        decision_domain = "combat" if in_combat else "build"
    else:
        decision_domain = "build"

    bridge_state: dict[str, Any] = {
        "ok": True,
        "backend": "headless_sim",
        "captured_at_utc": "",
        "state_version": int(sim_state.get("state_version") or 0),
        "state_hash": str(sim_state.get("state_hash") or ""),
        "semantic_state_hash": str(sim_state.get("semantic_state_hash") or ""),
        "schema_version": "sim-v1",
        "bridge_version": "headless_sim",
        "decision_domain": decision_domain,
        "screen": screen,
        "players": bridge_players,
        "player": bridge_player,
        "combat": bridge_combat,
        "map": bridge_map,
        "rewards": bridge_rewards,
        "run": bridge_run,
        "event_options": event_options,
        "card_selection": bridge_card_sel,
        "card_reward_selection": bridge_card_reward_sel,
        "character_selection": {"visible": state_type == "character_select", "options": []},
        "run_mode_selection": {"visible": False},
        "deck_upgrade_selection": {"visible": False, "choices": []},
        "main_menu": {"visible": screen == "MAIN_MENU"},
        # Per-phase decision dict consumed by observation_common._append
        # (10 scalar features: option_count, can_skip, selected_count,
        # min_select, max_select, is_open, travelable_count, can_proceed,
        # reward_count, item_count). Real bridge emits via
        # BuildEnvDecisionPayload (BridgeGameApi.EnvPayloads.cs:691-762);
        # translator was missing it entirely → every decision/phase
        # feature read as zero during sim training.
        "decision": _build_decision_block(
            state_type=state_type,
            in_combat=in_combat,
            event=event,
            map_state=map_state,
            rest_site=rest_site,
            shop=shop,
            treasure=treasure,
            rewards=rewards_state,
            card_reward=card_reward,
            card_select=card_select,
            hand_select=hand_select,
        ),
        "rest_site": bridge_rest_site,
        "shop": bridge_shop,
        "crystal_sphere": {"visible": False},
        "automation": {"enabled": False},
        "available_actions": bridge_actions,
        "_sim_raw": sim_state,
        "episode_id": episode_id,
        "phase": phase,
    }
    return bridge_state

# Preserve the historical public type/function homes for diagnostics and any
# out-of-tree pickle/introspection consumers.
SelfInflictedHpTracker.__module__ = __name__
sim_kind_to_bridge_kind.__module__ = __name__

__all__ = [
    "SelfInflictedHpTracker",
    "sim_kind_to_bridge_kind",
    "translate_to_bridge_shape",
]
