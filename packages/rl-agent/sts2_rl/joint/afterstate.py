"""Fail-closed deterministic macro afterstate synthesis.

On verified deterministic macro-economic surfaces (card rewards, deterministic
loot claims, campfires with authoritative heal/upgrade previews, and shop
purchases without immediate upon-pickup mutations), every legal candidate's
exact persistent world post-state is known before acting.

This module projects the current observation into canonical world form once,
applies only exact factual transitions, and encodes each candidate's world
post-state through :meth:`GroundedObservationEncoder.encode_canonical_world_snapshot`.
If any candidate on the current surface is stochastic, missing authoritative
preview facts, or structurally ambiguous, the entire surface fails closed and
returns ``None`` so the learner falls back to trajectory V-trace credit.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from sts2_rl.encoding.grounded import (
    GroundedObservationEncoder,
    _aggregate_orderless_card_multiset,
    _canonical_card,
    _canonical_model_observation,
    _canonical_potion,
    _canonical_relic,
)
from sts2_rl.encoding.snapshot import EncodedWorldSnapshot
from sts2_rl.semantics.forward import ForwardDecision, SemanticCandidate

AFTERSTATE_SYNTHESIZER_VERSION: Final = "sts2-deterministic-afterstate-v2"

_CARD_IDENTITY_EXCLUDED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "instance_id",
        "instance_uuid",
        "quantity",
        "selection_membership",
        "is_selected",
    }
)


def _exact_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _finite_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        return None
    rounded = round(numeric)
    if abs(numeric - float(rounded)) > 1e-6:
        return None
    return rounded


def _deck_entry_identity(card: Mapping[str, Any]) -> str:
    projected = {
        key: value
        for key, value in card.items()
        if key not in _CARD_IDENTITY_EXCLUDED_KEYS
    }
    return json.dumps(projected, sort_keys=True, separators=(",", ":"))


def _match_deck_entry_index(
    deck_cards: Sequence[Mapping[str, Any]],
    target_card: Mapping[str, Any],
) -> int | None:
    canonical_target = _canonical_card({**dict(target_card), "pile": "Deck"})
    if not canonical_target.get("id"):
        return None
    wanted_identity = _deck_entry_identity(canonical_target)
    exact_matches = [
        index
        for index, entry in enumerate(deck_cards)
        if _deck_entry_identity(entry) == wanted_identity
        and (_exact_int(entry.get("quantity", 1)) or 0) >= 1
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        return None

    target_comparable = {
        key: value
        for key, value in canonical_target.items()
        if key not in _CARD_IDENTITY_EXCLUDED_KEYS and value is not None
    }
    shared_matches: list[int] = []
    for index, entry in enumerate(deck_cards):
        if (_exact_int(entry.get("quantity", 1)) or 0) < 1:
            continue
        entry_comparable = {
            key: value
            for key, value in entry.items()
            if key not in _CARD_IDENTITY_EXCLUDED_KEYS and value is not None
        }
        shared_keys = target_comparable.keys() & entry_comparable.keys()
        if "id" not in shared_keys:
            continue
        if all(target_comparable[key] == entry_comparable[key] for key in shared_keys):
            shared_matches.append(index)
    return shared_matches[0] if len(shared_matches) == 1 else None


def _decrement_deck_entry(
    deck_cards: Sequence[Mapping[str, Any]],
    match_index: int,
) -> list[dict[str, Any]] | None:
    updated: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(deck_cards):
        entry = copy.deepcopy(dict(raw_entry))
        quantity = _exact_int(entry.get("quantity", 1))
        if quantity is None or quantity < 1:
            return None
        if index == match_index:
            if quantity > 1:
                entry["quantity"] = quantity - 1
                updated.append(entry)
        else:
            updated.append(entry)
    return _aggregate_orderless_card_multiset(updated)


def _append_deck_card(
    deck_cards: Sequence[Mapping[str, Any]],
    raw_card: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    canonical = _canonical_card({**dict(raw_card), "pile": "Deck"})
    if not canonical.get("id"):
        return None
    canonical.pop("selection_membership", None)
    canonical.pop("is_selected", None)
    canonical["quantity"] = 1
    combined = [copy.deepcopy(dict(entry)) for entry in deck_cards]
    combined.append(canonical)
    return _aggregate_orderless_card_multiset(combined)


def _sort_inventory_list(items: list[dict[str, Any]]) -> None:
    items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))


def _resolve_rest_heal_amount(
    observation: Mapping[str, Any],
    candidate: SemanticCandidate,
) -> int | None:
    action = candidate.semantic_action
    option = action.get("option")
    if isinstance(option, Mapping):
        heal = _finite_non_negative_int(option.get("heal_amount"))
        if heal is not None:
            return heal
    direct = _finite_non_negative_int(action.get("heal_amount"))
    if direct is not None:
        return direct
    rest_site = observation.get("rest_site")
    if isinstance(rest_site, Mapping):
        options = rest_site.get("options")
        option_idx = (
            option.get("index")
            if isinstance(option, Mapping)
            else action.get("index")
        )
        if isinstance(options, Sequence) and not isinstance(options, str | bytes):
            for pos, raw_opt in enumerate(options):
                if not isinstance(raw_opt, Mapping):
                    continue
                raw_idx = raw_opt.get("index", pos)
                if option_idx is not None and raw_idx == option_idx:
                    return _finite_non_negative_int(raw_opt.get("heal_amount"))
    return None


def _resolve_smith_upgrade_preview(
    observation: Mapping[str, Any],
    candidate: SemanticCandidate,
) -> Mapping[str, Any] | None:
    action = candidate.semantic_action
    direct_preview = action.get("upgrade_preview")
    if isinstance(direct_preview, Mapping) and direct_preview:
        return direct_preview
    action_card = action.get("card")
    if isinstance(action_card, Mapping):
        nested_preview = action_card.get("upgrade_preview")
        if isinstance(nested_preview, Mapping) and nested_preview:
            return nested_preview
    if isinstance(candidate.target, Mapping):
        target_preview = candidate.target.get("upgrade_preview")
        if isinstance(target_preview, Mapping) and target_preview:
            return target_preview
    raw_player = observation.get("player")
    raw_deck = raw_player.get("deck") if isinstance(raw_player, Mapping) else None
    target_source = action_card if isinstance(action_card, Mapping) else candidate.target
    if (
        isinstance(target_source, Mapping)
        and isinstance(raw_deck, Sequence)
        and not isinstance(raw_deck, str | bytes)
    ):
        deck_mappings = [item for item in raw_deck if isinstance(item, Mapping)]
        canonical_deck = [
            {**_canonical_card({**dict(item), "pile": "Deck"}), "quantity": 1}
            for item in deck_mappings
        ]
        matched_index = _match_deck_entry_index(canonical_deck, target_source)
        if matched_index is not None:
            preview = deck_mappings[matched_index].get("upgrade_preview")
            if isinstance(preview, Mapping) and preview:
                return preview
    return None


def _synthesize_rest_candidate(
    base_canonical: Mapping[str, Any],
    observation: Mapping[str, Any],
    candidate: SemanticCandidate,
) -> dict[str, Any] | None:
    post = copy.deepcopy(dict(base_canonical))
    player = post.get("player")
    if not isinstance(player, Mapping):
        return None
    player_dict = dict(player)
    post["player"] = player_dict
    post.pop("rest_site", None)
    post.pop("decision", None)
    post["phase"] = "map"
    post["decision_domain"] = "map"

    if candidate.branch == "rest":
        current_hp = _exact_int(player_dict.get("hp"))
        max_hp = _exact_int(player_dict.get("max_hp"))
        heal_amount = _resolve_rest_heal_amount(observation, candidate)
        if (
            current_hp is None
            or max_hp is None
            or max_hp <= 0
            or heal_amount is None
        ):
            return None
        player_dict["hp"] = min(max_hp, current_hp + heal_amount)
        return post

    if candidate.branch == "smith":
        action_card = candidate.semantic_action.get("card")
        target_card = (
            action_card
            if isinstance(action_card, Mapping)
            else candidate.target
        )
        if not isinstance(target_card, Mapping):
            return None
        raw_preview = _resolve_smith_upgrade_preview(observation, candidate)
        if raw_preview is None:
            return None
        deck_cards = player_dict.get("deck_cards")
        if not isinstance(deck_cards, list):
            return None
        match_index = _match_deck_entry_index(deck_cards, target_card)
        if match_index is None:
            return None
        decremented = _decrement_deck_entry(deck_cards, match_index)
        if decremented is None:
            return None
        upgraded_deck = _append_deck_card(decremented, raw_preview)
        if upgraded_deck is None:
            return None
        player_dict["deck_cards"] = upgraded_deck
        return post

    return None


def _remove_shop_slot(
    post: dict[str, Any],
    *,
    slot_index: int | None,
    new_gold: int,
    category: str | None = None,
) -> bool:
    shop = post.get("shop")
    if not isinstance(shop, dict):
        return True
    items = shop.get("items")
    if not isinstance(items, list) or not items:
        return True
    matched_positions: list[int] = []
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            return False
        item_slot = _exact_int(item.get("slot_index", item.get("index")))
        if slot_index is not None and item_slot == slot_index:
            matched_positions.append(pos)
    if not matched_positions and category == "card_removal":
        for pos, item in enumerate(items):
            if str(item.get("type") or item.get("category") or "").lower() == "card_removal":
                matched_positions.append(pos)
    if len(matched_positions) != 1:
        return False
    remaining: list[dict[str, Any]] = []
    for pos, item in enumerate(items):
        if pos == matched_positions[0]:
            continue
        updated_item = dict(item)
        price = _finite_non_negative_int(
            updated_item.get("price", updated_item.get("cost"))
        )
        if price is not None and "can_afford" in updated_item:
            updated_item["can_afford"] = bool(new_gold >= price)
        remaining.append(updated_item)
    shop["items"] = remaining
    if "gold" in shop:
        shop["gold"] = new_gold
    return True


def _resolve_shop_item(
    observation: Mapping[str, Any],
    candidate: SemanticCandidate,
) -> Mapping[str, Any] | None:
    raw_item = candidate.semantic_action.get("item")
    slot_index = None
    if isinstance(raw_item, Mapping):
        slot_index = _exact_int(raw_item.get("slot_index", raw_item.get("index")))
    if slot_index is None and isinstance(candidate.target, Mapping):
        slot_index = _exact_int(candidate.target.get("index"))
    raw_shop = observation.get("shop")
    if slot_index is not None and isinstance(raw_shop, Mapping):
        items = raw_shop.get("items")
        if isinstance(items, Sequence) and not isinstance(items, str | bytes):
            for pos, entry in enumerate(items):
                if not isinstance(entry, Mapping):
                    continue
                entry_slot = _exact_int(entry.get("slot_index", entry.get("index")))
                if (entry_slot if entry_slot is not None else pos) == slot_index:
                    if isinstance(raw_item, Mapping):
                        return {**dict(entry), **dict(raw_item)}
                    return entry
    return raw_item if isinstance(raw_item, Mapping) else None


def _synthesize_shop_candidate(
    base_canonical: Mapping[str, Any],
    observation: Mapping[str, Any],
    candidate: SemanticCandidate,
) -> dict[str, Any] | None:
    post = copy.deepcopy(dict(base_canonical))
    player = post.get("player")
    if not isinstance(player, Mapping):
        return None
    player_dict = dict(player)
    post["player"] = player_dict
    post.pop("decision", None)

    if candidate.branch == "leave":
        post.pop("shop", None)
        post["phase"] = "map"
        post["decision_domain"] = "map"
        return post

    current_gold = _exact_int(player_dict.get("gold"))
    if current_gold is None or current_gold < 0:
        return None
    item = _resolve_shop_item(observation, candidate)
    if item is None:
        return None
    cost = _finite_non_negative_int(item.get("cost", item.get("price")))
    if cost is None or cost > current_gold:
        return None
    new_gold = current_gold - cost
    slot_index = _exact_int(item.get("slot_index", item.get("index")))
    post["phase"] = "shop"
    post["decision_domain"] = "shop"

    if candidate.branch == "buy_card":
        raw_card = item.get("card")
        deck_cards = player_dict.get("deck_cards")
        deck_count = _exact_int(player_dict.get("deck"))
        if (
            not isinstance(raw_card, Mapping)
            or not isinstance(deck_cards, list)
            or deck_count is None
        ):
            return None
        updated_deck = _append_deck_card(deck_cards, raw_card)
        if updated_deck is None:
            return None
        player_dict["gold"] = new_gold
        player_dict["deck_cards"] = updated_deck
        player_dict["deck"] = deck_count + 1
        if not _remove_shop_slot(post, slot_index=slot_index, new_gold=new_gold):
            return None
        return post

    if candidate.branch == "buy_relic":
        raw_relic = item.get("relic")
        relics = player_dict.get("relics")
        if not isinstance(raw_relic, Mapping) or not isinstance(relics, list):
            return None
        if raw_relic.get("has_upon_pickup_effect") is not False:
            return None
        canonical_relic = _canonical_relic(raw_relic)
        if not canonical_relic.get("id"):
            return None
        updated_relics = [copy.deepcopy(dict(entry)) for entry in relics]
        updated_relics.append(canonical_relic)
        _sort_inventory_list(updated_relics)
        player_dict["gold"] = new_gold
        player_dict["relics"] = updated_relics
        if not _remove_shop_slot(post, slot_index=slot_index, new_gold=new_gold):
            return None
        return post

    if candidate.branch == "buy_potion":
        raw_potion = item.get("potion")
        potions = player_dict.get("potions")
        open_slots = _exact_int(player_dict.get("open_potion_slots"))
        if (
            not isinstance(raw_potion, Mapping)
            or not isinstance(potions, list)
            or open_slots is None
            or open_slots < 1
        ):
            return None
        canonical_potion = _canonical_potion(raw_potion)
        if not canonical_potion.get("id"):
            return None
        updated_potions = [copy.deepcopy(dict(entry)) for entry in potions]
        updated_potions.append(canonical_potion)
        _sort_inventory_list(updated_potions)
        player_dict["gold"] = new_gold
        player_dict["potions"] = updated_potions
        player_dict["open_potion_slots"] = open_slots - 1
        if not _remove_shop_slot(post, slot_index=slot_index, new_gold=new_gold):
            return None
        return post

    if candidate.branch == "remove":
        action_card = candidate.semantic_action.get("card")
        target_card = (
            action_card
            if isinstance(action_card, Mapping)
            else candidate.target
        )
        deck_cards = player_dict.get("deck_cards")
        deck_count = _exact_int(player_dict.get("deck"))
        if (
            not isinstance(target_card, Mapping)
            or not isinstance(deck_cards, list)
            or deck_count is None
            or deck_count < 1
        ):
            return None
        match_index = _match_deck_entry_index(deck_cards, target_card)
        if match_index is None:
            return None
        updated_deck = _decrement_deck_entry(deck_cards, match_index)
        if updated_deck is None:
            return None
        player_dict["gold"] = new_gold
        player_dict["deck_cards"] = updated_deck
        player_dict["deck"] = deck_count - 1
        if not _remove_shop_slot(
            post,
            slot_index=slot_index,
            new_gold=new_gold,
            category="card_removal",
        ):
            return None
        return post

    return None


def _remove_reward_slot(
    post: dict[str, Any],
    *,
    slot_index: int | None,
) -> bool:
    rewards = post.get("rewards")
    if not isinstance(rewards, dict):
        return True
    items = rewards.get("items")
    if not isinstance(items, list) or not items:
        return True
    if slot_index is None:
        return False
    remaining = [
        dict(entry)
        for entry in items
        if isinstance(entry, Mapping)
        and _exact_int(entry.get("slot_index", entry.get("index"))) != slot_index
    ]
    if len(remaining) != len(items) - 1:
        return False
    if remaining:
        rewards["items"] = remaining
    else:
        rewards.pop("items", None)
    if not rewards:
        post.pop("rewards", None)
        post["phase"] = "map"
        post["decision_domain"] = "map"
    else:
        post["phase"] = "rewards"
        post["decision_domain"] = "rewards"
    return True


def _synthesize_reward_candidate(
    base_canonical: Mapping[str, Any],
    candidate: SemanticCandidate,
    *,
    is_card_reward_surface: bool,
) -> dict[str, Any] | None:
    post = copy.deepcopy(dict(base_canonical))
    player = post.get("player")
    if not isinstance(player, Mapping):
        return None
    player_dict = dict(player)
    post["player"] = player_dict
    post.pop("decision", None)

    if is_card_reward_surface:
        rewards = post.get("rewards")
        if isinstance(rewards, dict):
            rewards.pop("cards", None)
            rewards.pop("can_skip", None)
            if not rewards.get("items"):
                post.pop("rewards", None)
                post["phase"] = "map"
                post["decision_domain"] = "map"
            else:
                post["phase"] = "rewards"
                post["decision_domain"] = "rewards"
        else:
            post["phase"] = "map"
            post["decision_domain"] = "map"

        if candidate.branch == "skip":
            return post
        if candidate.branch == "take":
            raw_card = candidate.semantic_action.get("card") or candidate.target
            deck_cards = player_dict.get("deck_cards")
            deck_count = _exact_int(player_dict.get("deck"))
            if (
                not isinstance(raw_card, Mapping)
                or not isinstance(deck_cards, list)
                or deck_count is None
            ):
                return None
            updated_deck = _append_deck_card(deck_cards, raw_card)
            if updated_deck is None:
                return None
            player_dict["deck_cards"] = updated_deck
            player_dict["deck"] = deck_count + 1
            return post
        return None

    if candidate.branch == "leave":
        post.pop("rewards", None)
        post["phase"] = "map"
        post["decision_domain"] = "map"
        return post

    raw_reward = candidate.semantic_action.get("reward") or candidate.target
    if not isinstance(raw_reward, Mapping):
        return None
    slot_index = _exact_int(raw_reward.get("slot_index", raw_reward.get("index")))

    if candidate.branch == "claim_gold":
        current_gold = _exact_int(player_dict.get("gold"))
        amount = _finite_non_negative_int(raw_reward.get("amount"))
        if current_gold is None or amount is None:
            return None
        player_dict["gold"] = current_gold + amount
        if not _remove_reward_slot(post, slot_index=slot_index):
            return None
        return post

    if candidate.branch == "claim_potion":
        raw_potion = raw_reward.get("potion")
        potions = player_dict.get("potions")
        open_slots = _exact_int(player_dict.get("open_potion_slots"))
        if (
            not isinstance(raw_potion, Mapping)
            or not isinstance(potions, list)
            or open_slots is None
            or open_slots < 1
        ):
            return None
        canonical_potion = _canonical_potion(raw_potion)
        if not canonical_potion.get("id"):
            return None
        updated_potions = [copy.deepcopy(dict(entry)) for entry in potions]
        updated_potions.append(canonical_potion)
        _sort_inventory_list(updated_potions)
        player_dict["potions"] = updated_potions
        player_dict["open_potion_slots"] = open_slots - 1
        if not _remove_reward_slot(post, slot_index=slot_index):
            return None
        return post

    if candidate.branch == "claim_relic":
        raw_relic = raw_reward.get("relic")
        relics = player_dict.get("relics")
        if not isinstance(raw_relic, Mapping) or not isinstance(relics, list):
            return None
        if raw_relic.get("has_upon_pickup_effect") is not False:
            return None
        canonical_relic = _canonical_relic(raw_relic)
        if not canonical_relic.get("id"):
            return None
        updated_relics = [copy.deepcopy(dict(entry)) for entry in relics]
        updated_relics.append(canonical_relic)
        _sort_inventory_list(updated_relics)
        player_dict["relics"] = updated_relics
        if not _remove_reward_slot(post, slot_index=slot_index):
            return None
        return post

    return None


def _resolve_selection_upgrade_preview(
    observation: Mapping[str, Any],
    candidate: SemanticCandidate,
) -> Mapping[str, Any] | None:
    direct = _resolve_smith_upgrade_preview(observation, candidate)
    if direct is not None:
        return direct
    action_card = candidate.semantic_action.get("card")
    target_source = action_card if isinstance(action_card, Mapping) else candidate.target
    if not isinstance(target_source, Mapping):
        return None
    raw_sel: dict[str, Any] = {}
    for top_key in ("card_selection", "selection", "decision"):
        top_val = observation.get(top_key)
        if isinstance(top_val, Mapping):
            nested_sel = top_val.get("selection")
            if isinstance(nested_sel, Mapping):
                raw_sel.update(dict(nested_sel))
            else:
                raw_sel.update(dict(top_val))
    candidate_pool: list[Mapping[str, Any]] = []
    for field in ("cards", "selectable_cards"):
        seq = raw_sel.get(field)
        if isinstance(seq, Sequence) and not isinstance(seq, str | bytes):
            candidate_pool.extend(item for item in seq if isinstance(item, Mapping))
    opts = raw_sel.get("options")
    if isinstance(opts, Sequence) and not isinstance(opts, str | bytes):
        for opt in opts:
            if isinstance(opt, Mapping) and isinstance(opt.get("card"), Mapping):
                merged = dict(opt["card"])
                if isinstance(opt.get("upgrade_preview"), Mapping) and "upgrade_preview" not in merged:
                    merged["upgrade_preview"] = opt["upgrade_preview"]
                candidate_pool.append(merged)
    if candidate_pool:
        canonical_pool = [
            {**_canonical_card({**dict(item), "pile": "Deck"}), "quantity": 1}
            for item in candidate_pool
        ]
        matched_index = _match_deck_entry_index(canonical_pool, target_source)
        if matched_index is not None:
            preview = candidate_pool[matched_index].get("upgrade_preview")
            if isinstance(preview, Mapping) and preview:
                return preview
    return None


def _synthesize_selection_candidate(
    base_canonical: Mapping[str, Any],
    observation: Mapping[str, Any],
    candidate: SemanticCandidate,
) -> dict[str, Any] | None:
    if candidate.branch != "selection_add":
        return None
    decision_block = base_canonical.get("decision")
    sel = decision_block.get("selection") if isinstance(decision_block, Mapping) else None
    if not isinstance(sel, Mapping):
        return None
    raw_sel: dict[str, Any] = {}
    for top_key in ("card_selection", "selection", "decision"):
        top_val = observation.get(top_key)
        if isinstance(top_val, Mapping):
            nested_sel = top_val.get("selection")
            if isinstance(nested_sel, Mapping):
                raw_sel.update(dict(nested_sel))
            else:
                raw_sel.update(dict(top_val))

    if (
        sel.get("confirmation_mode") == "manual"
        or raw_sel.get("requires_manual_confirmation") is True
        or raw_sel.get("requires_confirm") is True
        or raw_sel.get("confirm_enabled") is True
        or sel.get("confirm_ready") is True
    ):
        return None
    is_auto_confirm = (
        sel.get("confirmation_mode") == "automatic"
        or raw_sel.get("requires_manual_confirmation") is False
        or raw_sel.get("requires_confirm") is False
    )
    if not is_auto_confirm:
        return None
    if _exact_int(sel.get("selected_count")) not in (None, 0) or sel.get("selected_cards"):
        return None
    if (
        sel.get("can_skip") is True
        or sel.get("cancelable") is True
        or raw_sel.get("can_skip") is True
        or raw_sel.get("cancelable") is True
        or raw_sel.get("can_cancel") is True
    ):
        return None
    rem = _exact_int(sel.get("remaining_select") if sel.get("remaining_select") is not None else raw_sel.get("remaining_picks"))
    min_s = _exact_int(sel.get("min_select") if sel.get("min_select") is not None else raw_sel.get("min_select"))
    max_s = _exact_int(sel.get("max_select") if sel.get("max_select") is not None else raw_sel.get("max_select"))
    if max_s not in (None, 1) or min_s not in (None, 1):
        return None
    if rem is not None:
        if rem != 1:
            return None
    elif not (min_s == 1 and max_s == 1):
        return None

    source_zone = str(
        sel.get("source_zone")
        or raw_sel.get("source_zone")
        or raw_sel.get("source_pile")
        or "Deck"
    ).strip().lower()
    if source_zone != "deck":
        return None

    op = str(sel.get("operation_type") or raw_sel.get("operation_type") or "").strip().lower()
    mode = str(
        sel.get("mode")
        or raw_sel.get("mode")
        or raw_sel.get("screen_type")
        or raw_sel.get("screen_subtype")
        or ""
    ).strip().lower()
    prompt = str(sel.get("prompt_id") or raw_sel.get("prompt_id") or "").strip().lower()
    is_upgrade = (
        op == "upgrade"
        or mode in {"deckupgrade", "upgrade"}
        or prompt == "card_selection.to_upgrade"
    )
    is_remove = (
        op == "remove"
        or mode in {"deckremove", "remove", "purge"}
        or prompt == "card_selection.to_remove"
    )
    if is_upgrade == is_remove:
        return None

    post = copy.deepcopy(dict(base_canonical))
    player = post.get("player")
    if not isinstance(player, Mapping):
        return None
    player_dict = dict(player)
    post["player"] = player_dict
    post.pop("decision", None)
    post.pop("event", None)
    post["phase"] = "map"
    post["decision_domain"] = "map"

    action_card = candidate.semantic_action.get("card")
    target_card = action_card if isinstance(action_card, Mapping) else candidate.target
    deck_cards = player_dict.get("deck_cards")
    deck_count = _exact_int(player_dict.get("deck"))
    if not isinstance(target_card, Mapping) or not isinstance(deck_cards, list) or deck_count is None:
        return None

    match_index = _match_deck_entry_index(deck_cards, target_card)
    if match_index is None:
        return None
    decremented = _decrement_deck_entry(deck_cards, match_index)
    if decremented is None:
        return None

    if is_upgrade:
        raw_preview = _resolve_selection_upgrade_preview(observation, candidate)
        if raw_preview is None:
            return None
        upgraded_deck = _append_deck_card(decremented, raw_preview)
        if upgraded_deck is None:
            return None
        player_dict["deck_cards"] = upgraded_deck
        return post

    if deck_count < 1:
        return None
    player_dict["deck_cards"] = decremented
    player_dict["deck"] = deck_count - 1
    return post


def synthesize_deterministic_afterstates(
    *,
    encoder: GroundedObservationEncoder,
    observation: Mapping[str, Any],
    decision: ForwardDecision,
    candidates: Sequence[SemanticCandidate] | None = None,
) -> tuple[EncodedWorldSnapshot, ...] | None:
    """Synthesize encoded post-state world snapshots for every legal candidate.

    Returns ``None`` if the surface or any candidate is stochastic, unsupported,
    or missing authoritative facts (fail-closed).
    """

    active_candidates = tuple(decision.candidates if candidates is None else candidates)
    if len(active_candidates) < 2:
        return None
    if decision.surface not in {"rest", "shop", "reward", "selection"}:
        return None

    try:
        base_canonical = _canonical_model_observation(observation)
    except Exception:
        return None

    if "combat" in base_canonical:
        return None

    branches = {candidate.branch for candidate in active_candidates}
    is_card_reward_surface = False
    if decision.surface == "reward":
        if branches <= {"take", "skip"}:
            is_card_reward_surface = True
        elif branches <= {"claim_gold", "claim_potion", "claim_relic", "leave"}:
            is_card_reward_surface = False
        else:
            # Includes claim_card, claim_special_card, or mixed card/claim surfaces
            # whose post-state draws unseen random cards.
            return None
    elif decision.surface == "selection" and branches != {"selection_add"}:
        return None

    encoded_post_worlds: list[EncodedWorldSnapshot] = []
    for candidate in active_candidates:
        if decision.surface == "rest":
            post_canonical = _synthesize_rest_candidate(
                base_canonical,
                observation,
                candidate,
            )
        elif decision.surface == "shop":
            post_canonical = _synthesize_shop_candidate(
                base_canonical,
                observation,
                candidate,
            )
        elif decision.surface == "reward":
            post_canonical = _synthesize_reward_candidate(
                base_canonical,
                candidate,
                is_card_reward_surface=is_card_reward_surface,
            )
        elif decision.surface == "selection":
            post_canonical = _synthesize_selection_candidate(
                base_canonical,
                observation,
                candidate,
            )
        else:
            return None

        if post_canonical is None:
            return None
        try:
            encoded = encoder.encode_canonical_world_snapshot(post_canonical)
        except Exception:
            return None
        encoded_post_worlds.append(encoded)

    return tuple(encoded_post_worlds)


def synthesize_partial_deterministic_afterstates(
    *,
    encoder: GroundedObservationEncoder,
    observation: Mapping[str, Any],
    decision: ForwardDecision,
    candidates: Sequence[SemanticCandidate] | None = None,
) -> tuple[tuple[int, ...], tuple[EncodedWorldSnapshot, ...]] | None:
    """Synthesize verified deterministic afterstate subset when >= 2 candidates qualify.

    For full deterministic surfaces (card rewards, deterministic loot, campfires,
    shops without pickup-effect relics, single-pick deck upgrade/remove selections),
    returns ``(tuple(range(len(active_candidates))), post_worlds)``.
    For shops or campfires where specific candidates have immediate stochastic/pickup
    side effects (e.g. ``has_upon_pickup_effect=True`` relics), isolates the verified
    deterministic subset ``D`` (provided ``|D| >= 2``) while leaving ``i not in D``
    untouched by subset-preserving MPO.
    """

    active_candidates = tuple(decision.candidates if candidates is None else candidates)
    if len(active_candidates) < 2:
        return None
    full = synthesize_deterministic_afterstates(
        encoder=encoder,
        observation=observation,
        decision=decision,
        candidates=active_candidates,
    )
    if full is not None:
        return tuple(range(len(active_candidates))), full

    if decision.surface not in {"shop", "rest"}:
        return None
    try:
        base_canonical = _canonical_model_observation(observation)
    except Exception:
        return None
    if "combat" in base_canonical:
        return None

    det_indices: list[int] = []
    det_worlds: list[EncodedWorldSnapshot] = []
    for idx, candidate in enumerate(active_candidates):
        if decision.surface == "shop":
            post_canonical = _synthesize_shop_candidate(
                base_canonical,
                observation,
                candidate,
            )
        else:
            post_canonical = _synthesize_rest_candidate(
                base_canonical,
                observation,
                candidate,
            )
        if post_canonical is None:
            continue
        try:
            encoded = encoder.encode_canonical_world_snapshot(post_canonical)
        except Exception:
            continue
        det_indices.append(idx)
        det_worlds.append(encoded)

    if len(det_indices) < 2:
        return None
    return tuple(det_indices), tuple(det_worlds)


__all__ = [
    "AFTERSTATE_SYNTHESIZER_VERSION",
    "synthesize_deterministic_afterstates",
    "synthesize_partial_deterministic_afterstates",
]
