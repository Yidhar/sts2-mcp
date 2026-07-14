"""Fail-closed audit for the native card-facts transport contract.

This does not maintain a card-effect database.  It asks the pinned simulator
for the facts produced by every CardModel and checks that no DynamicVar,
keyword, tag, or lifecycle field is malformed or dropped.  A small set of
representative cards guards the semantic families required by the learner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

_REQUIRED_CARD_FIELDS = frozenset(
    {
        "card_id",
        "base_cost",
        "card_type",
        "is_x_cost",
        "rarity",
        "target_type",
        "gains_block",
        "has_turn_end_in_hand_effect",
        "has_on_draw_effect",
        "exhaust_on_next_play",
        "keywords",
        "tags",
        "hover_tip_ids",
        "dynamic_vars",
    }
)
_REQUIRED_DYNAMIC_VAR_FIELDS = frozenset(
    {
        "name",
        "var_type",
        "family",
        "base_value",
        "enchanted_value",
        "preview_value",
        "int_value",
        "was_just_upgraded",
    }
)
_PRODUCTION_MAX_WORLD_TOKENS = 512
_PRODUCTION_MAX_CANDIDATE_LOCAL_TOKENS = 24


def _dynamic_var(card: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    values = card.get("dynamic_vars")
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise AssertionError(f"{card.get('card_id')}: dynamic_vars is not a sequence")
    matches = [
        item
        for item in values
        if isinstance(item, Mapping) and str(item.get("name")) == name
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"{card.get('card_id')}: expected one DynamicVar {name!r}, got {len(matches)}"
        )
    return matches[0]


def _assert_effect(
    cards: Mapping[str, Mapping[str, Any]],
    card_id: str,
    *,
    card_type: str,
    var_name: str,
    family: str,
    value: int,
) -> None:
    card = cards[card_id]
    if str(card.get("card_type", card.get("type", ""))).lower() != card_type:
        raise AssertionError(f"{card_id}: card_type mismatch")
    dynamic_var = _dynamic_var(card, var_name)
    if dynamic_var.get("family") != family:
        raise AssertionError(f"{card_id}/{var_name}: family mismatch")
    if int(dynamic_var.get("int_value", -999999)) != value:
        raise AssertionError(f"{card_id}/{var_name}: value mismatch")


def audit_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    raw_cards = catalog.get("cards")
    if not isinstance(raw_cards, Sequence) or isinstance(raw_cards, str | bytes):
        raise AssertionError("game_catalog.cards must be a sequence")
    cards: dict[str, Mapping[str, Any]] = {}
    families: dict[str, int] = {}
    dynamic_var_count = 0
    for index, raw_card in enumerate(raw_cards):
        if not isinstance(raw_card, Mapping):
            raise AssertionError(f"cards[{index}] is not a mapping")
        missing = _REQUIRED_CARD_FIELDS.difference(raw_card)
        if missing:
            raise AssertionError(
                f"cards[{index}] is missing card-facts fields: {sorted(missing)}"
            )
        card_id = str(raw_card["card_id"]).strip().upper()
        if not card_id or card_id in cards:
            raise AssertionError(f"invalid or duplicate card_id: {card_id!r}")
        cards[card_id] = raw_card
        for list_field in ("keywords", "tags", "hover_tip_ids", "dynamic_vars"):
            value = raw_card[list_field]
            if not isinstance(value, Sequence) or isinstance(value, str | bytes):
                raise AssertionError(f"{card_id}: {list_field} must be a sequence")
        for var_index, dynamic_var in enumerate(raw_card["dynamic_vars"]):
            if not isinstance(dynamic_var, Mapping):
                raise AssertionError(
                    f"{card_id}.dynamic_vars[{var_index}] is not a mapping"
                )
            missing = _REQUIRED_DYNAMIC_VAR_FIELDS.difference(dynamic_var)
            if missing:
                raise AssertionError(
                    f"{card_id}.dynamic_vars[{var_index}] missing {sorted(missing)}"
                )
            for numeric_field in (
                "base_value",
                "enchanted_value",
                "preview_value",
                "int_value",
            ):
                value = dynamic_var[numeric_field]
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise AssertionError(
                        f"{card_id}.dynamic_vars[{var_index}].{numeric_field} "
                        "must be numeric"
                    )
                if not math.isfinite(float(value)):
                    raise AssertionError(
                        f"{card_id}.dynamic_vars[{var_index}].{numeric_field} "
                        "must be finite"
                    )
            family = str(dynamic_var["family"])
            families[family] = families.get(family, 0) + 1
            dynamic_var_count += 1

    if len(cards) < 500:
        raise AssertionError(f"unexpectedly small native card catalog: {len(cards)}")

    _assert_effect(
        cards,
        "STRIKE_IRONCLAD",
        card_type="attack",
        var_name="Damage",
        family="damage",
        value=6,
    )
    _assert_effect(
        cards,
        "BASH",
        card_type="attack",
        var_name="Damage",
        family="damage",
        value=8,
    )
    _assert_effect(
        cards,
        "BASH",
        card_type="attack",
        var_name="VulnerablePower",
        family="power",
        value=2,
    )
    vulnerable = _dynamic_var(cards["BASH"], "VulnerablePower")
    if vulnerable.get("power_type") != "VulnerablePower":
        raise AssertionError("BASH must expose VulnerablePower exactly")
    _assert_effect(
        cards,
        "DEFEND_IRONCLAD",
        card_type="skill",
        var_name="Block",
        family="block",
        value=5,
    )
    if cards["DEFEND_IRONCLAD"].get("gains_block") is not True:
        raise AssertionError("DEFEND_IRONCLAD must expose gains_block")
    _assert_effect(
        cards,
        "BURN",
        card_type="status",
        var_name="Damage",
        family="damage",
        value=2,
    )
    burning_pact_tips = {
        str(value).lower() for value in cards["BURNING_PACT"]["hover_tip_ids"]
    }
    if not any("exhaust" in value for value in burning_pact_tips):
        raise AssertionError(
            "BURNING_PACT must expose the game-owned Exhaust hover-tip identity"
        )
    if cards["BURN"].get("has_turn_end_in_hand_effect") is not True:
        raise AssertionError("BURN must expose its end-of-turn-in-hand trigger")
    _assert_effect(
        cards,
        "VOID",
        card_type="status",
        var_name="Energy",
        family="energy",
        value=1,
    )
    if cards["VOID"].get("has_on_draw_effect") is not True:
        raise AssertionError("VOID must expose its on-draw trigger")
    _assert_effect(
        cards,
        "SHAME",
        card_type="curse",
        var_name="Frail",
        family="value",
        value=1,
    )
    _assert_effect(
        cards,
        "DEADLY_POISON",
        card_type="skill",
        var_name="PoisonPower",
        family="power",
        value=5,
    )
    poison = _dynamic_var(cards["DEADLY_POISON"], "PoisonPower")
    if poison.get("power_type") != "PoisonPower":
        raise AssertionError("DEADLY_POISON must expose PoisonPower exactly")
    _assert_effect(
        cards,
        "ADRENALINE",
        card_type="skill",
        var_name="Energy",
        family="energy",
        value=1,
    )
    _assert_effect(
        cards,
        "ADRENALINE",
        card_type="skill",
        var_name="Cards",
        family="cards",
        value=2,
    )
    _assert_effect(
        cards,
        "BURNING_PACT",
        card_type="skill",
        var_name="Cards",
        family="cards",
        value=2,
    )

    return {
        "cards": len(cards),
        "cards_with_dynamic_vars": sum(
            bool(card["dynamic_vars"]) for card in cards.values()
        ),
        "dynamic_vars": dynamic_var_count,
        "dynamic_var_families": dict(sorted(families.items())),
        "cards_with_keywords": sum(bool(card["keywords"]) for card in cards.values()),
        "cards_with_tags": sum(bool(card["tags"]) for card in cards.values()),
        "cards_with_hover_tips": sum(
            bool(card["hover_tip_ids"]) for card in cards.values()
        ),
        "cards_with_turn_end_in_hand_effect": sum(
            card["has_turn_end_in_hand_effect"] is True for card in cards.values()
        ),
        "cards_with_on_draw_effect": sum(
            card["has_on_draw_effect"] is True for card in cards.values()
        ),
        "cards_with_only_identity_type_cost_target": sum(
            not card["dynamic_vars"]
            and not card["keywords"]
            and not card["tags"]
            and not card["hover_tip_ids"]
            and card["gains_block"] is not True
            and card["has_turn_end_in_hand_effect"] is not True
            and card["has_on_draw_effect"] is not True
            and card["exhaust_on_next_play"] is not True
            for card in cards.values()
        ),
    }


def audit_encoder_capacity(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Run every catalog card through the production candidate-local ABI."""

    from sts2_rl.encoding.grounded import (
        GroundedEncodingConfig,
        GroundedObservationEncoder,
    )
    from sts2_rl.models.grounded_candidate import GroundedCandidateConfig

    raw_cards = catalog["cards"]
    model = GroundedCandidateConfig()
    encoder = GroundedObservationEncoder(
        GroundedEncodingConfig.from_model_config(
            model,
            max_world_tokens=_PRODUCTION_MAX_WORLD_TOKENS,
            max_candidates=1,
            max_candidate_local_tokens=_PRODUCTION_MAX_CANDIDATE_LOCAL_TOKENS,
        )
    )
    observation = {"phase": "combat", "decision_domain": "combat"}
    local_counts: list[tuple[int, str]] = []
    for raw_card in raw_cards:
        card_id = str(raw_card["card_id"])
        action = {
            "action_handle": "card-facts-capacity-audit",
            "kind": "play_card",
            "model_action_kind": "play_card",
            "is_enabled": True,
            "card": raw_card,
        }
        try:
            encoded = encoder.encode(observation, [action])
        except ValueError as exc:
            raise AssertionError(
                f"{card_id}: production grounded encoder rejected card facts: {exc}"
            ) from exc
        local_count = int(encoded.batch.candidates.local_mask[0, 0].sum().item())
        local_counts.append((local_count, card_id))
    maximum, card_id = max(local_counts)
    return {
        "candidate_local_capacity": _PRODUCTION_MAX_CANDIDATE_LOCAL_TOKENS,
        "max_candidate_local_tokens": maximum,
        "max_candidate_local_tokens_card": card_id,
    }


def audit_runtime_hand(client: Any) -> dict[str, Any]:
    """Verify that facts survive the real full-run state/action transport."""

    state = client.reset(character="IRONCLAD", seed="CARDFACTAUDIT")
    play_actions: list[Mapping[str, Any]] = []
    for _ in range(32):
        raw_actions = state.get("legal_actions")
        if not isinstance(raw_actions, Sequence) or isinstance(
            raw_actions, str | bytes
        ):
            raise AssertionError("translated full-run state has no legal actions")
        play_actions = [
            action
            for action in raw_actions
            if isinstance(action, Mapping)
            and action.get("model_action_kind") == "play_card"
        ]
        if play_actions:
            break
        enabled = [
            index
            for index, action in enumerate(raw_actions)
            if isinstance(action, Mapping)
            and action.get("is_enabled", action.get("enabled", True)) is True
        ]
        if not enabled:
            raise AssertionError("full-run bootstrap has no enabled action")
        state = client.step(state["episode_id"], action_index=enabled[0])
    if not play_actions:
        raise AssertionError("full-run bootstrap did not reach a combat hand")

    runtime_cards: dict[str, Mapping[str, Any]] = {}
    for action in play_actions:
        card = action.get("card")
        if not isinstance(card, Mapping):
            raise AssertionError("play-card candidate is missing its card facts")
        card_id = str(card.get("id", "")).removeprefix("CARD.").upper()
        runtime_cards[card_id] = card
    for card_id in ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"):
        if card_id not in runtime_cards:
            raise AssertionError(f"deterministic audit hand is missing {card_id}")
    _assert_effect(
        runtime_cards,
        "STRIKE_IRONCLAD",
        card_type="attack",
        var_name="Damage",
        family="damage",
        value=6,
    )
    _assert_effect(
        runtime_cards,
        "DEFEND_IRONCLAD",
        card_type="skill",
        var_name="Block",
        family="block",
        value=5,
    )
    _assert_effect(
        runtime_cards,
        "BASH",
        card_type="attack",
        var_name="VulnerablePower",
        family="power",
        value=2,
    )
    return {
        "runtime_hand_card_ids": sorted(runtime_cards),
        "runtime_hand_play_candidates": len(play_actions),
    }


def main() -> None:
    from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-exe", type=Path, default=None)
    args = parser.parse_args()
    with HeadlessSimBridgeClient(exe_path=args.sim_exe) as client:
        catalog = client._rpc("game_catalog")
        runtime_summary = audit_runtime_hand(client)
    summary = audit_catalog(catalog)
    summary.update(audit_encoder_capacity(catalog))
    summary.update(runtime_summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
