from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reward_card(
    card_id: str,
    *,
    title: str | None = None,
    type_: str = "Attack",
    cost: int = 1,
    damage: float = 0.0,
    block: float = 0.0,
    draw: float = 0.0,
) -> dict:
    return {
        "id": card_id,
        "title": title or card_id,
        "type": type_,
        "cost": cost,
        "card_effect_profile": {
            "semantic_signals": {"damage": damage, "block": block, "draw": draw},
            "semantic_tags": [],
        },
    }


def _thin_low_block_reward_raw() -> dict:
    deck = [_reward_card(f"strike_{idx}", type_="Attack", damage=6) for idx in range(5)]
    deck.append(_reward_card("weak_defend", type_="Skill", block=5))
    return {
        "run": {"current_floor": 4, "act_id": 0},
        "player": {"hp": 70, "max_hp": 80, "deck_cards": deck},
    }


def _reward_skip_action() -> dict:
    return {
        "kind": "skip_card_reward",
        "selection": "skip",
        "surface": "card_reward",
        "action_id": "reward.skip_card_reward",
    }


def _reward_pick_action(card: dict, idx: int) -> dict:
    return {
        "kind": "card_reward",
        "selection": "pick",
        "surface": "card_reward",
        "action_id": f"reward.pick_card:{idx}",
        "card": card,
    }


def test_final_action_is_skip_uses_final_action_fields_not_surface() -> None:
    from muzero.training.post_search_policy_retarget import final_action_is_skip

    pick = {
        "kind": "card_reward",
        "surface": "card_reward",
        "selection": "pick",
        "action_id": "reward.pick_card:0",
    }
    skip = {
        "kind": "skip_card_reward",
        "surface": "card_reward",
        "selection": "skip",
        "action_id": "reward.skip_card_reward",
    }

    assert final_action_is_skip(pick) is False
    assert final_action_is_skip(skip) is True


def test_annotate_card_reward_final_selection_rewrites_selected_is_skip_to_final_pick() -> None:
    from muzero.training.post_search_policy_retarget import annotate_card_reward_final_selection

    payload = {
        "selected_is_skip": True,
        "skip_blocked": True,
        "decision_reason": "skip_blocked",
    }
    pick = {
        "kind": "card_reward",
        "surface": "card_reward",
        "selection": "pick",
        "action_id": "reward.pick_card:0",
        "card": {"id": "good_block", "title": "Good Block", "type": "Skill", "cost": 1},
    }

    annotate_card_reward_final_selection(
        payload,
        chosen_action=pick,
        final_action_idx=0,
        final_selected_family="reward_pick",
        phase="card_reward",
        decision_domain="build",
        policy_retargeted=True,
    )

    assert payload["original_selected_is_skip"] is True
    assert payload["final_selected_is_skip"] is False
    assert payload["selected_is_skip"] is False
    assert payload["policy_retargeted"] is True
    assert payload["final_action_idx"] == 0
    assert payload["final_action"]["selection"] == "pick"
    assert payload["final_action"]["card"]["id"] == "good_block"


def test_retarget_search_policy_after_hard_guard_one_hots_final_action() -> None:
    from muzero.training.post_search_policy_retarget import retarget_search_policy_after_hard_guard

    original = np.array([0.05, 0.10, 0.15, 0.70], dtype=np.float32)

    retargeted, changed = retarget_search_policy_after_hard_guard(
        original,
        original_action_idx=3,
        final_action_idx=1,
        max_actions=4,
    )

    assert changed is True
    np.testing.assert_allclose(retargeted, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))


def test_retarget_search_policy_after_hard_guard_keeps_policy_when_unchanged() -> None:
    from muzero.training.post_search_policy_retarget import retarget_search_policy_after_hard_guard

    original = np.array([0.2, 0.8], dtype=np.float32)

    retargeted, changed = retarget_search_policy_after_hard_guard(
        original,
        original_action_idx=1,
        final_action_idx=1,
        max_actions=2,
    )

    assert changed is False
    np.testing.assert_allclose(retargeted, original)


def test_card_reward_guard_override_retarget_diagnostic_and_tracker_flow() -> None:
    """Lock the full Act1 anti-skip path to the final post-guard pick.

    The regression we care about is subtle: search/policy can prefer
    ``skip``, the card-reward guard can correctly override it to a useful
    card, but replay/JSONL/episode metrics must all learn/report the *final*
    pick instead of the original skip.  This test exercises that whole
    no-environment mini-flow.
    """

    from muzero.diagnostics.deck_build_metrics import CardRewardEpisodeTracker
    from muzero.training.card_reward_guard import apply_card_reward_guard
    from muzero.training.post_search_policy_retarget import (
        annotate_card_reward_final_selection,
        retarget_search_policy_after_hard_guard,
    )

    attack_card = _reward_card("ok_attack", type_="Attack", damage=6)
    block_card = _reward_card("good_block", type_="Skill", block=12)
    actions = [
        _reward_skip_action(),
        _reward_pick_action(attack_card, 0),
        _reward_pick_action(block_card, 1),
    ]
    stats: dict = {}

    final_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1, 1], dtype=np.float32),
        raw_obs=_thin_low_block_reward_raw(),
        search_stats=stats,
    )

    assert final_idx == 2
    assert stats["card_reward_guard_skip_blocked"] == 1.0
    assert stats["card_reward_guard_override"] == 1.0

    original_policy = np.array([0.90, 0.05, 0.05], dtype=np.float32)
    retargeted_policy, policy_retargeted = retarget_search_policy_after_hard_guard(
        original_policy,
        original_action_idx=0,
        final_action_idx=final_idx,
        max_actions=3,
    )

    assert policy_retargeted is True
    np.testing.assert_allclose(retargeted_policy, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    payload = stats["_card_reward_choice_diagnostic"]
    assert payload["original_selected_is_skip"] is True
    assert payload["selected_is_skip"] is True
    assert payload["final_selected_is_skip"] is None

    annotate_card_reward_final_selection(
        payload,
        chosen_action=actions[final_idx],
        final_action_idx=final_idx,
        final_selected_family="card_reward",
        phase="card_reward",
        decision_domain="build",
        policy_retargeted=policy_retargeted,
    )

    assert payload["original_selected_is_skip"] is True
    assert payload["final_selected_is_skip"] is False
    assert payload["selected_is_skip"] is False
    assert payload["policy_retargeted"] is True
    assert payload["final_action_idx"] == 2
    assert payload["final_action"]["selection"] == "pick"
    assert payload["final_action"]["card"]["id"] == "good_block"

    tracker = CardRewardEpisodeTracker()
    tracker.update(
        decision_domain="build",
        phase="card_reward",
        legal_actions=actions,
        chosen_action=actions[final_idx],
        chosen_signature=actions[final_idx],
        selected_family="card_reward",
        selection="pick",
    )
    metadata = tracker.as_metadata()

    assert metadata["card_reward_seen_count"] == 1.0
    assert metadata["card_reward_pick_count"] == 1.0
    assert metadata["card_reward_skip_count"] == 0.0
    assert metadata["card_reward_pick_rate"] == 1.0
    assert metadata["card_reward_skip_rate"] == 0.0
    assert metadata["card_reward_consecutive_skip_current"] == 0.0
