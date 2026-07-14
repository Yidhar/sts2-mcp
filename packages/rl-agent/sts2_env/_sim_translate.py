"""Translate HeadlessSim JSON DTOs to the typed environment observation.

Only structural protocol adaptation belongs here.  The translator deliberately
does not create rewards, auxiliary targets, route summaries, inferred event
effects, boss mechanics, or action-quality signals.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ._sim_translate_actions import _translate_legal_actions
from ._sim_translate_decisions import (
    _translate_card_reward_sel_block,
    _translate_card_sel_block,
    _translate_event_options,
    _translate_rest_site_block,
    _translate_rewards_block,
    _translate_shop_block,
)
from ._sim_translate_entities import (
    _translate_combat_block,
    _translate_map_block,
    _translate_player,
    _translate_run_block,
)
from ._sim_translate_shared import (
    _decision_domain_from_phase,
    _phase_from_state,
    _screen_from_state_type,
    sim_kind_to_bridge_kind,
)


def _section(sim_state: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = sim_state.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"simulator {name} section must be a mapping")
    return value


def _player_snapshot(
    sim_state: Mapping[str, Any],
    sections: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    for section in sections:
        player = section.get("player")
        if isinstance(player, Mapping):
            return player
        if player is not None:
            raise TypeError("simulator player snapshot must be a mapping")
    direct = sim_state.get("player")
    if isinstance(direct, Mapping):
        return direct
    if direct is not None:
        raise TypeError("simulator player snapshot must be a mapping")
    return {}


def translate_to_bridge_shape(
    sim_state: dict[str, Any],
    *,
    episode_id: str,
) -> dict[str, Any]:
    """Translate one simulator state without modifying its legal choices."""

    if not isinstance(sim_state, Mapping):
        raise TypeError("simulator state must be a mapping")
    state_type = str(sim_state.get("state_type") or "").lower()

    sim_run = _section(sim_state, "run")
    battle = _section(sim_state, "battle")
    event = _section(sim_state, "event")
    map_state = _section(sim_state, "map")
    rest_site = _section(sim_state, "rest_site")
    shop = _section(sim_state, "shop")
    treasure = _section(sim_state, "treasure")
    rewards_state = _section(sim_state, "rewards")
    card_reward = _section(sim_state, "card_reward")
    card_select = _section(sim_state, "card_select")
    hand_select = _section(sim_state, "hand_select")
    relic_select = _section(sim_state, "relic_select")
    game_over = _section(sim_state, "game_over")

    sim_player = _player_snapshot(
        sim_state,
        (
            battle,
            event,
            map_state,
            rest_site,
            shop,
            treasure,
            rewards_state,
            card_reward,
            card_select,
            hand_select,
            relic_select,
            game_over,
        ),
    )
    in_combat = bool(
        state_type in {"combat", "battle"} or battle.get("player") is not None or battle.get("enemies") is not None
    )
    phase = _phase_from_state(state_type, in_combat=in_combat)

    raw_actions = sim_state.get("legal_actions")
    if raw_actions is None:
        raw_actions = []
    if not isinstance(raw_actions, list | tuple):
        raise TypeError("simulator legal_actions must be a sequence")
    combat_card_selection = battle.get("card_selection") if isinstance(battle.get("card_selection"), Mapping) else None
    action_card_selection = card_select or hand_select or combat_card_selection or {}
    actions = _translate_legal_actions(
        raw_actions,
        sim_player=sim_player,
        battle=battle,
        map_state=map_state,
        event=event,
        rest_site=rest_site,
        shop=shop,
        rewards=rewards_state,
        card_reward=card_reward,
        card_select=action_card_selection,
        treasure=treasure,
        relic_select=relic_select,
    )

    event_payload = {key: deepcopy(value) for key, value in event.items() if key not in {"player", "options"}}
    event_payload["options"] = _translate_event_options(event)

    observation: dict[str, Any] = {
        "ok": True,
        "backend": "headless_sim",
        "schema_version": "sim-v1",
        "bridge_version": "headless_sim",
        "episode_id": str(episode_id),
        "state_type": state_type,
        "state_version": int(sim_state.get("state_version", 0) or 0),
        "state_hash": str(sim_state.get("state_hash") or ""),
        "semantic_state_hash": str(sim_state.get("semantic_state_hash") or ""),
        "phase": phase,
        "decision_domain": _decision_domain_from_phase(phase, in_combat=in_combat),
        "screen": _screen_from_state_type(state_type, in_combat=in_combat),
        "terminated": bool(sim_state.get("terminal", False)),
        "truncated": bool(sim_state.get("truncated", False)),
        "player": _translate_player(sim_player),
        "combat": _translate_combat_block(battle, in_progress=in_combat),
        "run": _translate_run_block(sim_run),
        "map": _translate_map_block(map_state),
        "event": event_payload,
        "rewards": _translate_rewards_block(rewards_state, card_reward, treasure, relic_select),
        "rest_site": _translate_rest_site_block(rest_site),
        "shop": _translate_shop_block(shop),
        "card_selection": _translate_card_sel_block(
            card_select,
            hand_select,
            combat_card_selection,
        ),
        "card_reward_selection": _translate_card_reward_sel_block(card_reward),
        "available_actions": actions,
        # Exact simulator curriculum counters are reward/evaluation facts, not
        # policy features. The leading underscore keeps them outside the
        # grounded encoder while preserving an auditable transition source.
        "_training": {
            "revival_budget": sim_state.get("training_revival_budget"),
            "revivals_used": int(sim_state.get("training_revivals_used", 0) or 0),
            "player_hp_lost": float(sim_state.get("training_player_hp_lost", 0) or 0),
        },
        # Dispatch/debug provenance stays outside model features: the grounded
        # encoder rejects all underscore-prefixed fields.
        "_sim_raw": deepcopy(dict(sim_state)),
    }
    return observation


sim_kind_to_bridge_kind.__module__ = __name__

__all__ = ["sim_kind_to_bridge_kind", "translate_to_bridge_shape"]
