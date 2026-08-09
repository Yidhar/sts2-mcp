"""E2E proof that native forge previews reach the grounded model input.

The frozen prefix replays held-out seed 12600013 to its first Act-1 forge
selection.  It was captured from v44's deterministic 25k evaluation and uses
only authoritative action IDs.  The verifier checks three independent
contracts on the currently built pinned HeadlessSim binary:

1. every legal upgrade target carries the native alternative card state;
2. the alternative is relation-bound to the source card and is factually
   different after one canonical upgrade; and
3. removing only ``upgrade_preview`` changes the grounded tensor payload,
   proving the field is not merely transported as dead JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient  # noqa: E402
from sts2_rl.encoding import GroundedObservationEncoder  # noqa: E402

SEED = 12_600_013

FORGE_PREFIX_ACTION_IDS = (
    "sim:1:choose_event_option",
    "sim:0:proceed",
    "sim:0:choose_map_node",
    "sim:1:play_card",
    "sim:1:play_card",
    "sim:1:play_card",
    "sim:0:end_turn",
    "sim:2:play_card",
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
    "sim:2:play_card",
    "sim:2:play_card",
    "sim:0:end_turn",
    "sim:0:play_card",
    "sim:2:claim_reward",
    "sim:2:select_card_reward",
    "sim:0:claim_reward",
    "sim:0:claim_reward",
    "sim:0:proceed",
    "sim:0:choose_map_node",
    "sim:0:choose_event_option",
    "sim:0:proceed",
    "sim:0:choose_map_node",
    "sim:0:choose_event_option",
    "sim:0:proceed",
    "sim:0:choose_map_node",
    "sim:0:play_card",
    "sim:0:use_potion",
    "sim:0:end_turn",
    "sim:1:play_card",
    "sim:1:play_card",
    "sim:1:play_card",
    "sim:0:end_turn",
    "sim:0:play_card",
    "sim:0:claim_reward",
    "sim:0:claim_reward",
    "sim:2:select_card_reward",
    "sim:0:proceed",
    "sim:0:choose_map_node",
    "sim:1:choose_rest_option",
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
    return cast(dict[str, Any], client.step(episode_id, action_id=action_id))


def _strip_upgrade_previews(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_upgrade_previews(child)
            for key, child in value.items()
            if key != "upgrade_preview" and key != "upgrade_previews"
        }
    if isinstance(value, list):
        return [_strip_upgrade_previews(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_strip_upgrade_previews(child) for child in value)
    return value


def _table_payload(table: Any) -> tuple[npt.NDArray[Any], ...]:
    return (
        table.feature_indptr,
        table.feature_indices,
        table.feature_values,
        table.ids,
    )


def _snapshot_differs(left: Any, right: Any) -> bool:
    arrays = (
        *_table_payload(left.world),
        *_table_payload(left.candidates),
        *_table_payload(left.locals),
        left.local_offsets,
        left.action_mask,
    )
    other = (
        *_table_payload(right.world),
        *_table_payload(right.candidates),
        *_table_payload(right.locals),
        right.local_offsets,
        right.action_mask,
    )
    return any(
        first.shape != second.shape or not np.array_equal(first, second)
        for first, second in zip(arrays, other, strict=True)
    )


def verify(executable: Path | None) -> dict[str, Any]:
    with HeadlessSimBridgeClient(exe_path=executable) as client:
        response = client.reset(
            character="IRONCLAD",
            seed=SEED,
            ascension_level=0,
            training_revival_budget=40,
        )
        episode_id = str(response["episode_id"])
        for action_id in FORGE_PREFIX_ACTION_IDS:
            response = _step_exact(client, response, episode_id, action_id)

    observation = response.get("obs")
    legal_actions = response.get("legal_actions")
    if not isinstance(observation, dict) or not isinstance(legal_actions, list):
        raise AssertionError("forge response has no translated observation/action surface")
    selection_actions = [
        action
        for action in legal_actions
        if isinstance(action, dict) and action.get("action") == "select_card"
    ]
    if not selection_actions:
        raise AssertionError("frozen forge prefix did not reach a card selection")

    changed_cards = 0
    for action in selection_actions:
        card = action.get("card")
        preview = action.get("upgrade_preview")
        if not isinstance(card, dict) or not isinstance(preview, dict):
            raise AssertionError("upgrade candidate lost its card or root preview")
        nested = card.get("upgrade_preview")
        if not isinstance(nested, dict) or nested != preview or nested is preview:
            raise AssertionError("root/nested upgrade previews are not exact isolated copies")
        if card.get("id") != preview.get("id"):
            raise AssertionError("upgrade preview changed the card definition")
        if card.get("source_pile") != preview.get("source_pile"):
            raise AssertionError("upgrade preview lost the physical source relation")
        if card.get("instance_id") is not None or preview.get("instance_id") is not None:
            if card.get("instance_id") != preview.get("instance_id"):
                raise AssertionError("upgrade preview lost the physical instance relation")
        source_facts = {
            key: value
            for key, value in card.items()
            if key not in {"upgrade_preview", "selection_membership", "is_selected"}
        }
        preview_facts = {
            key: value
            for key, value in preview.items()
            if key not in {"selection_membership", "is_selected"}
        }
        changed_cards += int(source_facts != preview_facts)
    if changed_cards != len(selection_actions):
        raise AssertionError("one or more native upgrade previews are factually unchanged")

    encoder = GroundedObservationEncoder()
    with_preview = encoder.encode(observation, tuple(legal_actions), device="cpu").snapshot
    without_preview = encoder.encode(
        _strip_upgrade_previews(deepcopy(observation)),
        tuple(_strip_upgrade_previews(deepcopy(legal_actions))),
        device="cpu",
    ).snapshot
    if with_preview.candidate_count != without_preview.candidate_count:
        raise AssertionError("preview facts unexpectedly changed legal candidate cardinality")
    if not _snapshot_differs(with_preview, without_preview):
        raise AssertionError("grounded tensor is blind to native upgrade previews")

    return {
        "ok": True,
        "seed": SEED,
        "prefix_steps": len(FORGE_PREFIX_ACTION_IDS),
        "upgrade_candidates": len(selection_actions),
        "factually_changed_previews": changed_cards,
        "candidate_count": with_preview.candidate_count,
        "world_tokens_with_preview": with_preview.world.token_count,
        "world_tokens_without_preview": without_preview.world.token_count,
        "local_tokens_with_preview": with_preview.locals.token_count,
        "local_tokens_without_preview": without_preview.locals.token_count,
        "encoding_fingerprint": with_preview.encoding_fingerprint,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-exe", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = verify(args.sim_exe)
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
