"""Generic combat hand-mutation affordance extraction.

This module deliberately models *classes* of action->hand effects rather than
single card names.  Cards like Armaments should be detected because their text
says "upgrade ... hand", while cost reducers / discard / exhaust / transform /
copy / draw-to-hand effects use the same representation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import observation_common as obs_common


def _card_text(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    parts: list[str] = []
    for key in ("title", "name", "id", "description", "text", "canonical_text"):
        value = str(card.get(key) or "").strip()
        if value:
            parts.append(value)
    for key in ("keywords", "semantic_tags"):
        values = card.get(key)
        if isinstance(values, list):
            parts.extend(str(v or "").strip() for v in values if str(v or "").strip())
    signals = card.get("semantic_signals")
    if isinstance(signals, dict):
        parts.extend(str(k) for k, v in signals.items() if v)
    return " | ".join(parts).lower()


def _norm_count(value: float, denom: float = 12.0) -> float:
    return min(max(float(value), 0.0) / max(denom, 1.0), 1.0)


def _signed_norm(value: float, denom: float = 10.0) -> float:
    return max(min(float(value) / max(denom, 1.0), 1.0), -1.0)


def _cost(card: dict[str, Any] | None) -> float:
    if not isinstance(card, dict):
        return 0.0
    return max(obs_common._runtime_spend_cost(card), 0.0)


def _upgrade_level(card: dict[str, Any] | None) -> int:
    if not isinstance(card, dict):
        return 0
    return max(obs_common._infer_upgrade_level(card), 0)


def _preview(card: dict[str, Any] | None, key: str) -> float:
    if not isinstance(card, dict):
        return 0.0
    if key in {"damage", "block"}:
        bundle = obs_common._build_card_preview_bundle(card)
        return float(bundle.get(f"preview_{key}", 0.0))
    return obs_common._preview_metric(card, key)


def _card_type(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    return str(card.get("type") or "").strip().lower()


def _is_upgradeable(card: dict[str, Any] | None) -> bool:
    if not isinstance(card, dict):
        return False
    if bool(card.get("upgraded")) or bool(card.get("is_upgraded")):
        return False
    if _upgrade_level(card) > 0:
        return False
    if card.get("can_upgrade") is not None:
        return bool(card.get("can_upgrade"))
    text = _card_text(card)
    return "unupgradable" not in text and "cannot upgrade" not in text


def _infer_scope(text: str, *, default_one: bool = False) -> tuple[str, bool, bool, bool, bool]:
    """Return scope plus one/all/random/choice flags."""
    all_scope = bool(re.search(r"\b(all|each|every)\b.*\b(hand|cards? in your hand)\b", text))
    if not all_scope:
        all_scope = "all cards in your hand" in text or "your hand" in text and "all" in text
    random_scope = "random" in text
    one_scope = bool(re.search(r"\b(a|one|1)\b.*\b(card|attack|skill|power)\b.*\b(hand|your hand)\b", text))
    choice = one_scope or "choose" in text or "select" in text
    if all_scope:
        return "all", False, True, random_scope, False
    if random_scope:
        return "random", False, False, True, False
    if one_scope or default_one:
        return "one", True, False, False, choice
    return "hand", False, False, False, choice


def _type_filter(text: str) -> set[str]:
    filters: set[str] = set()
    for kind in ("attack", "skill", "power"):
        if kind in text:
            filters.add(kind)
    return filters


def _matches_filter(card: dict[str, Any], filters: set[str]) -> bool:
    if not filters:
        return True
    return _card_type(card) in filters


@dataclass(slots=True)
class MutationTarget:
    index: int
    card: dict[str, Any]
    affected: bool = False
    upgradeable: bool = False
    would_upgrade: bool = False
    would_cost_modify: bool = False
    would_discard: bool = False
    would_exhaust: bool = False
    would_transform: bool = False
    would_copy: bool = False
    would_gain_modifier: bool = False
    damage_delta: float = 0.0
    block_delta: float = 0.0
    cost_delta: float = 0.0
    draw_delta: float = 0.0
    energy_delta: float = 0.0
    playable_now: bool = False
    playable_after: bool = False
    followup_value: float = 0.0
    uncertain: bool = False


@dataclass(slots=True)
class HandMutationPlan:
    will_mutate_hand: bool = False
    upgrade_hand: bool = False
    upgrade_one: bool = False
    upgrade_all: bool = False
    upgrade_random: bool = False
    cost_modify_hand: bool = False
    set_cost_zero: bool = False
    draw_to_hand: bool = False
    add_to_hand: bool = False
    return_to_hand: bool = False
    discard_from_hand: bool = False
    exhaust_from_hand: bool = False
    transform_hand: bool = False
    copy_hand: bool = False
    modifier_hand: bool = False
    random_target: bool = False
    choice_required: bool = False
    temporary: bool = False
    affected_count: float = 0.0
    upgradeable_count: float = 0.0
    playable_followup_count: float = 0.0
    current_hand_damage: float = 0.0
    expected_damage_delta: float = 0.0
    current_hand_block: float = 0.0
    expected_block_delta: float = 0.0
    expected_cost_delta: float = 0.0
    best_single_target_value: float = 0.0
    same_turn_followup_possible: bool = False
    source_upgraded_scope_bonus: bool = False
    post_hand_count: float = 0.0
    post_upgrade_count: float = 0.0
    post_total_damage: float = 0.0
    post_total_block: float = 0.0
    post_total_draw: float = 0.0
    post_playable_cards: float = 0.0
    post_zero_cost_cards: float = 0.0
    post_energy_spendable_value: float = 0.0
    mutation_uncertainty: float = 0.0
    targets: list[MutationTarget] = field(default_factory=list)
    text: str = ""


def infer_hand_mutation(
    source: dict[str, Any] | None,
    hand: list[Any] | None,
    draw_pile: list[Any] | None = None,
    discard_pile: list[Any] | None = None,
    exhaust_pile: list[Any] | None = None,
    *,
    current_energy: float = 0.0,
) -> HandMutationPlan:
    plan = HandMutationPlan()
    if not isinstance(source, dict):
        return plan
    hand_cards = [c for c in (hand or []) if isinstance(c, dict)]
    text = _card_text(source)
    plan.text = text
    if not text:
        return plan

    source_level = _upgrade_level(source)
    mentions_hand = "hand" in text or "cards in your hand" in text or "your cards" in text
    upgrade = "upgrade" in text or "smith" in text
    cost_mod = any(t in text for t in ("cost 0", "costs 0", "cost to 0", "reduce the cost", "lower the cost", "set its cost", "set the cost"))
    set_zero = any(t in text for t in ("cost 0", "costs 0", "cost to 0", "cost is 0", "set its cost to 0", "set the cost to 0"))
    draw = bool(re.search(r"\bdraw\s+\d+", text) or "draw a card" in text or "draw cards" in text)
    add_to_hand = any(t in text for t in ("add", "create", "put")) and "hand" in text
    return_to_hand = any(t in text for t in ("return", "retrieve")) and "hand" in text
    discard = "discard" in text and mentions_hand
    exhaust = ("exhaust" in text or "consume" in text) and mentions_hand
    transform = ("transform" in text or "mutate" in text or "change" in text) and mentions_hand
    copy = ("copy" in text or "duplicate" in text) and mentions_hand
    modifier = any(t in text for t in ("retain", "ethereal", "bound", "temporary", "enchant", "afflict")) and mentions_hand
    temporary = "this turn" in text or "temporary" in text or "until played" in text

    if not any((upgrade and mentions_hand, cost_mod and mentions_hand, draw, add_to_hand, return_to_hand, discard, exhaust, transform, copy, modifier)):
        return plan

    # Upgraded cards often change one-card hand effects into all-hand effects.
    scope, one_scope, all_scope, random_scope, choice = _infer_scope(text, default_one=bool(mentions_hand and (upgrade or cost_mod or discard or exhaust or transform or copy or modifier)))
    if upgrade and mentions_hand and source_level > 0 and not random_scope and not all_scope:
        # Generic upgraded-scope exposure: if live text already says all, the text
        # path sets this.  If source only has title+/upgrade_level but stale text,
        # expose a soft "scope bonus" instead of hardcoding Armaments.
        plan.source_upgraded_scope_bonus = True

    filters = _type_filter(text)
    upgradeable_cards = [c for c in hand_cards if _is_upgradeable(c) and _matches_filter(c, filters)]
    candidate_cards = [c for c in hand_cards if _matches_filter(c, filters)] or hand_cards
    if all_scope:
        affected_indices = set(range(len(hand_cards))) if not filters else {i for i, c in enumerate(hand_cards) if _matches_filter(c, filters)}
    elif random_scope or one_scope:
        # For choice/random single-target effects, expose all plausible target
        # cards; aggregate count remains expected one target.
        affected_indices = {i for i, c in enumerate(hand_cards) if c in candidate_cards}
    else:
        affected_indices = {i for i, c in enumerate(hand_cards) if c in candidate_cards}

    plan.will_mutate_hand = True
    plan.upgrade_hand = bool(upgrade and mentions_hand)
    plan.upgrade_one = bool(plan.upgrade_hand and one_scope and not all_scope and not random_scope)
    plan.upgrade_all = bool(plan.upgrade_hand and all_scope)
    plan.upgrade_random = bool(plan.upgrade_hand and random_scope)
    plan.cost_modify_hand = bool(cost_mod and mentions_hand)
    plan.set_cost_zero = bool(plan.cost_modify_hand and set_zero)
    plan.draw_to_hand = bool(draw)
    plan.add_to_hand = bool(add_to_hand)
    plan.return_to_hand = bool(return_to_hand)
    plan.discard_from_hand = bool(discard)
    plan.exhaust_from_hand = bool(exhaust)
    plan.transform_hand = bool(transform)
    plan.copy_hand = bool(copy)
    plan.modifier_hand = bool(modifier)
    plan.random_target = bool(random_scope)
    plan.choice_required = bool(choice)
    plan.temporary = bool(temporary)

    plan.current_hand_damage = sum(_preview(c, "damage") for c in hand_cards)
    plan.current_hand_block = sum(_preview(c, "block") for c in hand_cards)
    plan.upgradeable_count = float(len(upgradeable_cards))

    effective_expected_count = 1.0 if (one_scope or random_scope) and not all_scope else float(len(affected_indices))
    if plan.draw_to_hand:
        m = re.search(r"draw\s+(\d+)", text)
        draw_count = float(m.group(1)) if m else 1.0
        effective_expected_count += draw_count
    if plan.add_to_hand or plan.return_to_hand:
        effective_expected_count += 1.0
    if plan.discard_from_hand or plan.exhaust_from_hand or plan.transform_hand:
        effective_expected_count = max(effective_expected_count, 1.0)
    plan.affected_count = effective_expected_count

    target_rows: list[MutationTarget] = []
    for i, card in enumerate(hand_cards):
        affected = i in affected_indices
        upg = _is_upgradeable(card)
        base_damage = _preview(card, "damage")
        base_block = _preview(card, "block")
        base_draw = _preview(card, "draw")
        base_energy = obs_common._get_card_extra_metrics(card)[2]
        base_cost = _cost(card)
        dmg_delta = 0.0
        block_delta = 0.0
        draw_delta = 0.0
        energy_delta = 0.0
        cost_delta = 0.0
        if affected and plan.upgrade_hand and upg:
            # Generic upgrade preview when exact upgraded payload is unavailable.
            dmg_delta = max(base_damage * 0.25, 3.0 if base_damage > 0 else 0.0)
            block_delta = max(base_block * 0.25, 2.0 if base_block > 0 else 0.0)
            draw_delta = 1.0 if "draw" in _card_text(card) else 0.0
        if affected and plan.cost_modify_hand:
            if plan.set_cost_zero:
                cost_delta = -base_cost
            else:
                cost_delta = -1.0 if base_cost > 0 else 0.0
        playable_now = base_cost <= max(current_energy, 0.0)
        playable_after = max(base_cost + cost_delta, 0.0) <= max(current_energy, 0.0)
        followup = max(base_damage + dmg_delta, 0.0) + 0.75 * max(base_block + block_delta, 0.0) + 2.0 * max(base_draw + draw_delta, 0.0) + 2.0 * max(base_energy + energy_delta, 0.0) - max(base_cost + cost_delta, 0.0)
        target_rows.append(
            MutationTarget(
                index=i,
                card=card,
                affected=affected,
                upgradeable=upg,
                would_upgrade=bool(affected and plan.upgrade_hand and upg),
                would_cost_modify=bool(affected and plan.cost_modify_hand),
                would_discard=bool(affected and plan.discard_from_hand),
                would_exhaust=bool(affected and plan.exhaust_from_hand),
                would_transform=bool(affected and plan.transform_hand),
                would_copy=bool(affected and plan.copy_hand),
                would_gain_modifier=bool(affected and plan.modifier_hand),
                damage_delta=dmg_delta,
                block_delta=block_delta,
                cost_delta=cost_delta,
                draw_delta=draw_delta,
                energy_delta=energy_delta,
                playable_now=playable_now,
                playable_after=playable_after,
                followup_value=followup,
                uncertain=bool(random_scope or (one_scope and len(affected_indices) > 1)),
            )
        )

    affected_targets = [t for t in target_rows if t.affected]
    if one_scope or random_scope:
        scale = 1.0 / max(len(affected_targets), 1)
    else:
        scale = 1.0
    plan.expected_damage_delta = sum(t.damage_delta for t in affected_targets) * scale
    plan.expected_block_delta = sum(t.block_delta for t in affected_targets) * scale
    plan.expected_cost_delta = sum(t.cost_delta for t in affected_targets) * scale
    plan.best_single_target_value = max((t.followup_value for t in affected_targets), default=0.0)
    plan.playable_followup_count = sum(1.0 for t in target_rows if t.playable_after and not (t.would_discard or t.would_exhaust))
    plan.same_turn_followup_possible = any(t.affected and t.playable_after and not (t.would_discard or t.would_exhaust) for t in target_rows)
    plan.mutation_uncertainty = 1.0 if (random_scope or one_scope or choice) else 0.0

    post_hand_count = float(len(hand_cards))
    if plan.draw_to_hand:
        post_hand_count += max(plan.affected_count - (1.0 if (one_scope or random_scope) else float(len(affected_indices))), 1.0)
    if plan.discard_from_hand or plan.exhaust_from_hand:
        post_hand_count -= effective_expected_count if (one_scope or random_scope) else min(effective_expected_count, len(hand_cards))
    if plan.copy_hand or plan.add_to_hand or plan.return_to_hand:
        post_hand_count += 1.0
    plan.post_hand_count = max(post_hand_count, 0.0)
    plan.post_upgrade_count = sum(1.0 for c in hand_cards if not _is_upgradeable(c)) + sum(1.0 for t in target_rows if t.would_upgrade) * (scale if (one_scope or random_scope) else 1.0)
    plan.post_total_damage = max(plan.current_hand_damage + plan.expected_damage_delta, 0.0)
    plan.post_total_block = max(plan.current_hand_block + plan.expected_block_delta, 0.0)
    plan.post_total_draw = sum(_preview(c, "draw") for c in hand_cards) + sum(t.draw_delta for t in affected_targets) * scale
    plan.post_playable_cards = plan.playable_followup_count
    plan.post_zero_cost_cards = sum(1.0 for c in hand_cards if _cost(c) <= 0.0) + sum(1.0 for t in affected_targets if t.cost_delta < 0 and max(_cost(t.card) + t.cost_delta, 0.0) == 0.0) * scale
    plan.post_energy_spendable_value = sum(max(_preview(t.card, "damage"), _preview(t.card, "block")) for t in target_rows if t.playable_after) / 20.0

    # Keep valuable targets first; encoder can then budget the most useful rows.
    target_rows.sort(key=lambda t: (float(t.affected), float(t.would_upgrade or t.would_cost_modify), t.followup_value), reverse=True)
    plan.targets = target_rows
    return plan


def mutation_summary_numeric(plan: HandMutationPlan) -> list[float]:
    row = [0.0] * 96
    row[0] = float(plan.will_mutate_hand)
    row[1] = float(plan.upgrade_hand)
    row[2] = float(plan.upgrade_one)
    row[3] = float(plan.upgrade_all)
    row[4] = float(plan.upgrade_random)
    row[5] = float(plan.cost_modify_hand)
    row[6] = float(plan.set_cost_zero)
    row[7] = float(plan.draw_to_hand)
    row[8] = float(plan.add_to_hand)
    row[9] = float(plan.return_to_hand)
    row[10] = float(plan.discard_from_hand)
    row[11] = float(plan.exhaust_from_hand)
    row[12] = float(plan.transform_hand)
    row[13] = float(plan.copy_hand)
    row[14] = float(plan.modifier_hand)
    row[15] = float(plan.random_target)
    row[16] = float(plan.choice_required)
    row[17] = float(plan.temporary)
    row[18] = _norm_count(plan.affected_count)
    row[19] = _norm_count(plan.upgradeable_count)
    row[20] = _norm_count(plan.playable_followup_count)
    row[21] = min(plan.current_hand_damage / 80.0, 1.0)
    row[22] = _signed_norm(plan.expected_damage_delta, 30.0)
    row[23] = min(plan.current_hand_block / 80.0, 1.0)
    row[24] = _signed_norm(plan.expected_block_delta, 30.0)
    row[25] = _signed_norm(plan.expected_cost_delta, 5.0)
    row[26] = min(max(plan.best_single_target_value, 0.0) / 40.0, 1.0)
    row[27] = float(plan.same_turn_followup_possible)
    row[28] = float(plan.source_upgraded_scope_bonus)
    return row


def mutation_target_numeric(target: MutationTarget) -> list[float]:
    row = [0.0] * 96
    row[0] = float(target.affected)
    row[1] = float(target.upgradeable)
    row[2] = min(_upgrade_level(target.card) / 3.0, 1.0)
    row[3] = float(target.would_upgrade)
    row[4] = float(target.would_cost_modify)
    row[5] = float(target.would_discard)
    row[6] = float(target.would_exhaust)
    row[7] = float(target.would_transform)
    row[8] = float(target.would_copy)
    row[9] = float(target.would_gain_modifier)
    row[10] = _signed_norm(target.damage_delta, 30.0)
    row[11] = _signed_norm(target.block_delta, 30.0)
    row[12] = _signed_norm(target.cost_delta, 5.0)
    row[13] = _signed_norm(target.draw_delta, 5.0)
    row[14] = _signed_norm(target.energy_delta, 5.0)
    row[15] = min(_cost(target.card) / 5.0, 1.0)
    row[16] = float(target.playable_now)
    row[17] = float(target.playable_after)
    row[18] = min(max(target.followup_value, 0.0) / 40.0, 1.0)
    row[19] = float(target.uncertain)
    row[20] = _norm_count(target.index + 1)
    return row


def post_hand_preview_numeric(plan: HandMutationPlan) -> list[float]:
    row = [0.0] * 96
    row[0] = _norm_count(plan.post_hand_count)
    row[1] = _norm_count(plan.post_upgrade_count)
    row[2] = min(plan.post_total_damage / 80.0, 1.0)
    row[3] = min(plan.post_total_block / 80.0, 1.0)
    row[4] = min(plan.post_total_draw / 10.0, 1.0)
    row[5] = _norm_count(plan.post_playable_cards)
    row[6] = _norm_count(plan.post_zero_cost_cards)
    row[7] = min(max(plan.post_energy_spendable_value, 0.0), 1.0)
    row[8] = min(max(plan.expected_damage_delta, 0.0) / max(plan.current_hand_damage, 1.0), 1.0)
    row[9] = min(max(plan.expected_block_delta, 0.0) / max(plan.current_hand_block, 1.0), 1.0)
    row[10] = min(max(plan.mutation_uncertainty, 0.0), 1.0)
    return row
