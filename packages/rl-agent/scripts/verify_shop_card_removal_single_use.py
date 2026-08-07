"""E2E regression for the merchant card-removal service's single-use contract.

The deterministic action prefix reaches the Act 1 floor-4 shop for seed
12600011, buys one potion, then completes exactly one card removal.  A valid
simulator must retire the removal entry immediately: a second removal purchase
must not appear in the next legal-action surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient  # noqa: E402

SEED = 12_600_011

# Frozen from the deterministic gate-zero trajectory for SEED.  This prefix
# ends after purchasing SPEED_POTION and before opening card removal.
SHOP_PREFIX_ACTION_IDS = (
    "sim:2:choose_event_option",
    "sim:0:proceed",
    "sim:2:choose_map_node",
    "sim:1:play_card",
    "sim:1:play_card",
    "sim:0:play_card",
    "sim:0:end_turn",
    "sim:0:play_card",
    "sim:2:play_card",
    "sim:1:play_card",
    "sim:0:end_turn",
    "sim:2:play_card",
    "sim:0:play_card",
    "sim:0:play_card",
    "sim:0:end_turn",
    "sim:0:play_card",
    "sim:0:play_card",
    "sim:0:claim_reward",
    "sim:0:claim_reward",
    "sim:2:select_card_reward",
    "sim:0:proceed",
    "sim:0:choose_map_node",
    "sim:0:play_card",
    "sim:2:play_card",
    "sim:0:play_card",
    "sim:0:end_turn",
    "sim:0:play_card",
    "sim:0:play_card",
    "sim:0:play_card",
    "sim:0:end_turn",
    "sim:2:play_card",
    "sim:2:play_card",
    "sim:1:play_card",
    "sim:0:play_card",
    "sim:0:play_card",
    "sim:0:end_turn",
    "sim:1:play_card",
    "sim:2:play_card",
    "sim:2:claim_reward",
    "sim:1:select_card_reward",
    "sim:0:claim_reward",
    "sim:0:claim_reward",
    "sim:0:proceed",
    "sim:0:choose_map_node",
    "sim:10:shop_purchase",
)

REMOVAL_TRANSACTION_ACTION_IDS = (
    "sim:12:shop_purchase",
    "sim:5:select_card",
    "sim:1:confirm_selection",
)


def _removal_actions(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        action
        for action in response.get("legal_actions", [])
        if isinstance(action, dict)
        and isinstance(action.get("item"), dict)
        and action["item"].get("category") == "card_removal"
    ]


def _player(response: dict[str, Any]) -> dict[str, Any]:
    observation = response.get("obs")
    if not isinstance(observation, dict) or not isinstance(observation.get("player"), dict):
        raise AssertionError("simulator response has no authoritative obs.player object")
    return observation["player"]


def _deck_count(player: dict[str, Any]) -> int:
    deck = player.get("deck", [])
    if not isinstance(deck, list):
        raise AssertionError("simulator response has no authoritative player.deck list")
    return sum(
        int(card.get("quantity", 1))
        for card in deck
        if isinstance(card, dict)
    )


def _step_exact(
    client: HeadlessSimBridgeClient,
    response: dict[str, Any],
    episode_id: str,
    action_id: str,
) -> dict[str, Any]:
    legal = response.get("legal_actions", [])
    if not any(
        isinstance(action, dict) and action.get("action_id") == action_id
        for action in legal
    ):
        surface = [
            (action.get("action_id"), action.get("label"))
            for action in legal
            if isinstance(action, dict)
        ]
        raise AssertionError(
            f"deterministic replay expected {action_id!r}, legal surface is {surface!r}"
        )
    return client.step(episode_id, action_id=action_id)


def verify(executable: Path | None) -> dict[str, Any]:
    with HeadlessSimBridgeClient(exe_path=executable) as client:
        response = client.reset(
            character="IRONCLAD",
            seed=SEED,
            ascension_level=0,
            training_revival_budget=64,
        )
        episode_id = str(response["episode_id"])
        for action_id in SHOP_PREFIX_ACTION_IDS:
            response = _step_exact(client, response, episode_id, action_id)

        removal_before = _removal_actions(response)
        if len(removal_before) != 1:
            raise AssertionError(
                f"expected exactly one stocked card-removal service before purchase, got {len(removal_before)}"
            )
        before_player = _player(response)
        gold_before = int(before_player["gold"])
        deck_before = _deck_count(before_player)
        price = int(removal_before[0]["item"]["price"])
        stale_purchase = removal_before[0].get("_sim_raw")
        if not isinstance(stale_purchase, dict):
            raise AssertionError("card-removal legal action has no raw simulator dispatch")

        for action_id in REMOVAL_TRANSACTION_ACTION_IDS:
            response = _step_exact(client, response, episode_id, action_id)

        after_player = _player(response)
        gold_after = int(after_player["gold"])
        deck_after = _deck_count(after_player)
        removal_after = _removal_actions(response)

        if gold_after != gold_before - price:
            raise AssertionError(
                f"card removal gold delta is {gold_before - gold_after}, expected {price}"
            )
        if deck_after != deck_before - 1:
            raise AssertionError(
                f"card removal deck delta is {deck_before - deck_after}, expected 1"
            )
        if removal_after:
            raise AssertionError(
                "single-use violation: card-removal service is still legal after a committed removal"
            )

        # Legal-surface retirement is not enough: an already-issued/stale
        # action must also fail closed at the authoritative dispatcher.
        stale_result = client._rpc("step", stale_purchase)
        if stale_result.get("accepted") is not False:
            raise AssertionError(
                "single-use violation: stale second card-removal purchase was accepted"
            )

        return {
            "ok": True,
            "seed": SEED,
            "gold_before": gold_before,
            "gold_after": gold_after,
            "deck_before": deck_before,
            "deck_after": deck_after,
            "removal_candidates_before": len(removal_before),
            "removal_candidates_after": len(removal_after),
            "stale_second_purchase_accepted": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-exe", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = verify(args.sim_exe)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
