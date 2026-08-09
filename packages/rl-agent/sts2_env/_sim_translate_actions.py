"""Lossless legal-action translation for the headless simulator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from ._sim_translate_decisions import (
    _translate_reward_item,
    _translate_reward_payload,
    _translate_shop_item,
)
from ._sim_translate_entities import (
    _translate_card,
    _translate_enemy,
    _translate_potion,
    _translate_relic,
)
from ._sim_translate_route import _translate_route_point
from ._sim_translate_shared import (
    sim_kind_to_bridge_kind,
    sim_kind_to_model_kind,
    sim_kind_to_selection_operation,
)


def _find_index(records: Any, index: Any, *, index_field: str = "index") -> Mapping[str, Any] | None:
    if not isinstance(index, int) or not isinstance(records, list | tuple):
        return None
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError("simulator indexed record must be a mapping")
        record_index = record.get(index_field)
        if not isinstance(record_index, int):
            record_index = position
        if record_index == index:
            return record
    return None


def _find_enemy(enemies: Any, combat_id: Any) -> Mapping[str, Any] | None:
    if combat_id is None or not isinstance(enemies, list | tuple):
        return None
    for enemy in enemies:
        if not isinstance(enemy, Mapping):
            raise TypeError("simulator enemy must be a mapping")
        candidate = enemy.get("combat_id", enemy.get("id"))
        if candidate is not None and str(candidate) == str(combat_id):
            return enemy
    return None


def _action_handle(position: int, action: Mapping[str, Any]) -> str:
    existing = action.get("action_handle", action.get("action_id"))
    if existing is not None and str(existing).strip():
        return str(existing)
    return f"sim:{position}:{action.get('action') or 'unknown'!s}"


def _translate_legal_actions(
    sim_legal_actions: Sequence[Any],
    *,
    sim_player: Mapping[str, Any],
    battle: Mapping[str, Any],
    map_state: Mapping[str, Any],
    event: Mapping[str, Any],
    rest_site: Mapping[str, Any],
    shop: Mapping[str, Any],
    rewards: Mapping[str, Any],
    card_reward: Mapping[str, Any],
    card_select: Mapping[str, Any],
    treasure: Mapping[str, Any],
    relic_select: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Translate every simulator action exactly once and in source order.

    Joins to cards, targets, and decision options only copy protocol records
    that are already present in the same simulator snapshot.  There is no
    filtering, sorting, score, safety mask, or confirm-action hoist.
    """

    if isinstance(sim_legal_actions, str | bytes) or not isinstance(sim_legal_actions, Sequence):
        raise TypeError("simulator legal_actions must be a sequence")

    output: list[dict[str, Any]] = []
    for position, raw_action in enumerate(sim_legal_actions):
        if not isinstance(raw_action, Mapping):
            raise TypeError(f"simulator legal action {position} must be a mapping")
        action = deepcopy(dict(raw_action))
        sim_kind = str(raw_action.get("action") or "unknown")
        handle = _action_handle(position, raw_action)
        selection_operation = sim_kind_to_selection_operation(sim_kind)
        native_selection_operation = raw_action.get("selection_operation")
        if native_selection_operation is not None:
            native_selection_operation = str(native_selection_operation).strip()
            if selection_operation != native_selection_operation:
                raise ValueError(
                    "simulator selection action has inconsistent operation metadata: "
                    f"action={sim_kind!r} expected={selection_operation!r} "
                    f"actual={native_selection_operation!r}"
                )
        native_is_selected = raw_action.get("is_selected")
        if native_is_selected is not None and selection_operation in {"select", "deselect"}:
            if not isinstance(native_is_selected, bool):
                raise ValueError(
                    "simulator selection action has non-boolean membership metadata: "
                    f"action={sim_kind!r} is_selected={native_is_selected!r}"
                )
            expected_is_selected = selection_operation == "deselect"
            if native_is_selected is not expected_is_selected:
                raise ValueError(
                    "simulator selection action has inconsistent membership metadata: "
                    f"action={sim_kind!r} expected_is_selected={expected_is_selected!r} "
                    f"actual={native_is_selected!r}"
                )
        action.update(
            {
                "idx": position,
                "action_index": position,
                "action_id": handle,
                "action_handle": handle,
                # Preserve the exact simulator enum for the learner.  The
                # bridge compatibility enum is a separate transport field.
                "kind": sim_kind,
                "transport_kind": sim_kind_to_bridge_kind(sim_kind),
                "model_action_kind": sim_kind_to_model_kind(sim_kind),
                "is_enabled": bool(raw_action.get("is_enabled", True)),
                "_sim_raw": deepcopy(dict(raw_action)),
            }
        )
        if sim_kind == "shop_purchase":
            action["model_action_variant"] = "buy"
            action["shop_action"] = "buy"
        elif sim_kind == "shop_skip":
            action["model_action_variant"] = "leave"
            action["shop_action"] = "leave"
        if selection_operation is not None:
            # The policy vocabulary intentionally shares one card-selection
            # family, while this variant preserves the exact state mutation.
            # Select, deselect, confirm, and cancel-prompt must never alias.
            action["model_action_variant"] = selection_operation
            action["selection_operation"] = selection_operation
            selection: dict[str, Any] = {
                "operation_type": selection_operation,
            }
            for canonical, aliases in {
                "selected_count": ("selected_count",),
                "min_select": ("min_select", "min_count"),
                "max_select": ("max_select", "max_count"),
                "remaining_select": ("remaining_select", "remaining_picks"),
                "confirm_ready": ("confirm_ready", "can_confirm"),
                "cancelable": ("cancelable", "can_cancel"),
                "requires_manual_confirmation": ("requires_manual_confirmation",),
                "mode": ("mode", "screen_type"),
                "prompt_id": ("prompt_id",),
                "source_zone": ("source_zone", "source_pile"),
                "destination_zone": ("destination_zone", "destination_pile"),
            }.items():
                value = next(
                    (card_select.get(alias) for alias in aliases if card_select.get(alias) is not None),
                    None,
                )
                if value is not None:
                    selection[canonical] = deepcopy(value)
            if raw_action.get("is_selected") is not None:
                selection["is_selected"] = bool(raw_action["is_selected"])
            action["selection"] = selection

        if sim_kind == "play_card":
            card = _find_index(sim_player.get("hand"), raw_action.get("card_index"))
            if card is not None:
                action["card"] = _translate_card(card, pile="Hand")
            enemy = _find_enemy(battle.get("enemies"), raw_action.get("target_id"))
            if enemy is not None:
                action["target"] = _translate_enemy(enemy)
        elif sim_kind in {"use_potion", "discard_potion"}:
            potion = _find_index(sim_player.get("potions"), raw_action.get("slot"), index_field="slot")
            if potion is not None:
                slot = raw_action.get("slot")
                action["potion"] = _translate_potion(
                    potion,
                    slot_index=slot if isinstance(slot, int) else None,
                )
            enemy = _find_enemy(battle.get("enemies"), raw_action.get("target_id"))
            if enemy is not None:
                action["target"] = _translate_enemy(enemy)
        elif sim_kind == "choose_map_node":
            node = _find_index(map_state.get("next_options"), raw_action.get("index"))
            if node is not None:
                action["map_node"] = _translate_route_point(node)
        elif sim_kind == "choose_event_option":
            option = _find_index(event.get("options"), raw_action.get("index"))
            if option is not None:
                action["option"] = deepcopy(dict(option))
        elif sim_kind == "choose_rest_option":
            option = _find_index(rest_site.get("options"), raw_action.get("index"))
            if option is not None:
                action["option"] = deepcopy(dict(option))
        elif sim_kind == "shop_purchase":
            item = _find_index(shop.get("items"), raw_action.get("index"))
            if item is not None:
                item_index = raw_action.get("index")
                action["item"] = _translate_shop_item(
                    item,
                    index=item_index if isinstance(item_index, int) else 0,
                )
        elif sim_kind == "claim_reward":
            reward_index = raw_action.get("index")
            item = _find_index(rewards.get("items"), reward_index)
            if item is not None:
                index = reward_index if isinstance(reward_index, int) else 0
                translated_item = _translate_reward_item(item, index=index)
                nested_reward = translated_item.get("reward")
                action["reward"] = (
                    nested_reward
                    if isinstance(nested_reward, Mapping)
                    else _translate_reward_payload(translated_item, index=index)
                )
        elif sim_kind in {"choose_card_reward", "select_card_reward"}:
            card = _find_index(card_reward.get("cards"), raw_action.get("index"))
            if card is not None:
                action["card"] = _translate_card(card, pile="Reward")
        elif sim_kind in {
            "select_card",
            "deselect_card",
            "select_hand_card",
            "deselect_hand_card",
            "combat_select_card",
            "combat_deselect_card",
            "select_card_option",
            "deselect_card_option",
        }:
            index = raw_action.get("index", raw_action.get("card_index"))
            card = _find_index(card_select.get("cards"), index)
            if card is None:
                card = _find_index(card_select.get("selectable_cards"), index)
            if card is None:
                card = _find_index(card_select.get("selected_cards"), index)
            if card is None:
                card = _find_index(sim_player.get("hand"), index)
            if card is not None:
                membership = "selected" if selection_operation == "deselect" else "selectable"
                default_pile = "Hand" if "hand_card" in sim_kind else None
                translated_card = _translate_card(
                    card,
                    pile=default_pile,
                    selection_membership=membership,
                )
                action["card"] = translated_card
                # Candidate-local roots are intentionally flat.  Lift the
                # exact native alternative card state while retaining it on
                # the translated card for world-selection projection.
                upgrade_preview = translated_card.get("upgrade_preview")
                if isinstance(upgrade_preview, Mapping):
                    action["upgrade_preview"] = deepcopy(
                        dict(upgrade_preview)
                    )
        elif sim_kind in {"claim_treasure", "claim_treasure_relic", "claim_relic"}:
            relic = _find_index(treasure.get("relics"), raw_action.get("index"))
            if relic is not None:
                action["relic"] = _translate_relic(relic)
        elif sim_kind == "select_relic":
            relic = _find_index(relic_select.get("relics"), raw_action.get("index"))
            if relic is not None:
                action["relic"] = _translate_relic(relic)

        output.append(action)

    return output


__all__: list[str] = []
