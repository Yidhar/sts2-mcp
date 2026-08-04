from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_rl.training.config import RuntimeConfig
from sts2_rl.training.evaluation_liveness import (
    evaluate_liveness_guard,
    summarize_greedy_liveness_journal,
)
from sts2_rl.training.seeding import (
    final_audit_evaluation_seeds,
    held_out_evaluation_seeds,
)


def _decision(
    episode: int,
    step: int,
    *,
    selected: str,
    confirm_ready: bool = False,
    end_turn_choice: bool = False,
    reward_hub: bool = False,
    terminal_cycle: bool = False,
) -> dict[str, object]:
    if end_turn_choice:
        legal_kinds = {"end_turn": 1, "play_card": 5}
        topk = [
            {"action": {"action": "end_turn"}, "probability": 0.19},
            {"action": {"action": "play_card"}, "probability": 0.17},
        ]
    elif reward_hub:
        legal_kinds = {"proceed": 1, "reward": 2}
        topk = [
            {"action": {"action": "proceed"}, "probability": 0.50},
            {"action": {"action": "claim_reward"}, "probability": 0.26},
            {"action": {"action": "claim_reward"}, "probability": 0.24},
        ]
    else:
        legal_kinds = {"card_selection": 3}
        topk = [
            {"action": {"action": "deselect_card"}, "probability": 0.34},
            {"action": {"action": "cancel_selection"}, "probability": 0.33},
            {"action": {"action": "confirm_selection"}, "probability": 0.33},
        ]
    return {
        "event": "decision",
        "record_kind": "summary",
        "episode_id": f"validation-{episode}",
        "step_index": step,
        "observation_summary": {
            "card_selection": {"can_confirm": confirm_ready},
        },
        "legal_action_kinds": legal_kinds,
        "selected_action": {"action": selected},
        "policy_topk": topk,
        "deadlock": (
            {"cycle_span": 2, "occurrences": 8}
            if terminal_cycle
            else None
        ),
    }


def test_liveness_summary_and_guard_detect_only_obvious_greedy_failures(
    tmp_path: Path,
) -> None:
    records: list[dict[str, object]] = []
    for episode in range(4):
        for step in range(8):
            records.append(
                _decision(
                    episode,
                    step,
                    selected="select_card" if step % 2 == 0 else "deselect_card",
                    confirm_ready=True,
                    terminal_cycle=step == 7,
                )
            )
        for step in range(8, 18):
            records.append(
                _decision(
                    episode,
                    step,
                    selected="end_turn",
                    end_turn_choice=True,
                )
            )
        records.append(
            _decision(
                episode,
                18,
                selected="proceed",
                reward_hub=True,
            )
        )
        records.append(
            _decision(
                episode,
                19,
                selected="deselect_card",
                confirm_ready=True,
                terminal_cycle=True,
            )
        )
    journal = tmp_path / "evaluation.jsonl"
    journal.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_greedy_liveness_journal(journal)
    assert summary["episode_count"] == 4
    assert summary["confirm_ready_decisions"] == 36
    assert summary["confirm_ready_greedy_rate"] == 0.0
    assert summary["multi_action_end_turn_decisions"] == 40
    assert summary["multi_action_end_turn_greedy_rate"] == 1.0
    assert summary["selection_cycle_episode_rate"] == 1.0
    assert summary["reward_hub_decisions"] == 4
    assert summary["reward_proceed_minus_best_claim_mean_margin"] == 0.24

    config = RuntimeConfig(
        early_evaluation_steps=(5_000,),
        early_evaluation_episodes=4,
        evaluation_liveness_guard_enabled=True,
    )
    guard = evaluate_liveness_guard(summary, config)
    assert guard["stop_requested"] is True
    assert {
        item["kind"] for item in guard["violations"]
    } == {
        "confirm_ready_greedy_failure",
        "avoidable_end_turn_collapse",
        "selection_cycle_collapse",
    }
    assert guard["outcome_metrics_used_for_stop"] is False


def test_final_audit_seed_range_is_odd_and_disjoint_from_repeated_validation() -> None:
    validation = held_out_evaluation_seeds(6, 32)
    final_audit = final_audit_evaluation_seeds(6, 32)
    assert set(validation).isdisjoint(final_audit)
    assert all(seed % 2 == 1 for seed in validation)
    assert all(seed % 2 == 1 for seed in final_audit)


def test_combat_selection_operation_names_use_the_same_liveness_semantics(
    tmp_path: Path,
) -> None:
    records = [
        {
            **_decision(
                0,
                step,
                selected=(
                    "combat_select_card"
                    if step % 2 == 0
                    else "combat_deselect_card"
                ),
                terminal_cycle=step == 7,
            ),
            "policy_topk": [
                {
                    "action": {"action": "combat_confirm_selection"},
                    "probability": 0.6,
                },
                {
                    "action": {"action": "combat_deselect_card"},
                    "probability": 0.4,
                },
            ],
        }
        for step in range(8)
    ]
    records.append(
        {
            **_decision(
                1,
                0,
                selected="combat_confirm_selection",
                confirm_ready=True,
            ),
            "policy_topk": [
                {
                    "action": {"action": "combat_confirm_selection"},
                    "probability": 1.0,
                }
            ],
        }
    )
    journal = tmp_path / "combat-selection.jsonl"
    journal.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_greedy_liveness_journal(journal)
    assert summary["selection_cycle_episode_count"] == 1
    assert summary["confirm_ready_decisions"] == 9
    assert summary["confirm_selected_decisions"] == 1


def test_optional_cancel_is_a_successful_selection_exit(
    tmp_path: Path,
) -> None:
    records = []
    for step in range(8):
        record = _decision(
            0,
            step,
            selected="combat_cancel_selection",
            confirm_ready=True,
        )
        record["observation_summary"] = {
            "card_selection": {
                "can_confirm": True,
                "can_cancel": True,
                "min_select": 0,
                "max_select": 1,
            }
        }
        records.append(record)

    journal = tmp_path / "optional-cancel.jsonl"
    journal.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_greedy_liveness_journal(journal)
    # The legacy metric remains confirm-only for report compatibility.
    assert summary["confirm_ready_decisions"] == 8
    assert summary["confirm_selected_decisions"] == 0
    assert summary["confirm_ready_failure_rate"] == 1.0
    # The guard asks whether the policy legally exited the window.
    assert summary["selection_exit_ready_decisions"] == 8
    assert summary["selection_exit_selected_decisions"] == 8
    assert summary["selection_exit_greedy_rate"] == 1.0
    assert summary["selection_exit_failure_rate"] == 0.0

    config = RuntimeConfig(
        early_evaluation_steps=(5_000,),
        early_evaluation_episodes=4,
        evaluation_liveness_guard_enabled=True,
    )
    guard = evaluate_liveness_guard(summary, config)
    assert guard["stop_requested"] is False
    assert guard["violations"] == []


def test_required_selection_without_exit_still_triggers_guard(
    tmp_path: Path,
) -> None:
    records = []
    for step in range(8):
        record = _decision(
            0,
            step,
            selected=(
                "combat_select_card"
                if step % 2 == 0
                else "combat_deselect_card"
            ),
            confirm_ready=True,
        )
        record["observation_summary"] = {
            "card_selection": {
                "can_confirm": True,
                "can_cancel": False,
                "min_select": 1,
                "max_select": 2,
            }
        }
        records.append(record)

    journal = tmp_path / "required-selection-no-exit.jsonl"
    journal.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_greedy_liveness_journal(journal)
    assert summary["selection_exit_ready_decisions"] == 8
    assert summary["selection_exit_selected_decisions"] == 0
    assert summary["selection_exit_failure_rate"] == 1.0

    config = RuntimeConfig(
        early_evaluation_steps=(5_000,),
        early_evaluation_episodes=4,
        evaluation_liveness_guard_enabled=True,
    )
    guard = evaluate_liveness_guard(summary, config)
    assert guard["stop_requested"] is True
    assert {
        item["kind"] for item in guard["violations"]
    } == {"confirm_ready_greedy_failure"}


def test_guard_accepts_legacy_confirm_only_telemetry() -> None:
    config = RuntimeConfig(
        early_evaluation_steps=(5_000,),
        early_evaluation_episodes=4,
        evaluation_liveness_guard_enabled=True,
    )
    guard = evaluate_liveness_guard(
        {
            "confirm_ready_decisions": 8,
            "confirm_ready_failure_rate": 1.0,
            "multi_action_end_turn_decisions": 0,
            "selection_cycle_episode_rate": 0.0,
        },
        config,
    )
    assert guard["stop_requested"] is True
    assert guard["violations"] == [
        {
            "kind": "confirm_ready_greedy_failure",
            "observed": 1.0,
            "threshold": 0.95,
            "exposures": 8,
        }
    ]


def test_card_removal_telemetry_requires_observed_deck_mutation(
    tmp_path: Path,
) -> None:
    records: list[dict[str, object]] = []

    def append(
        episode: int,
        step: int,
        action: dict[str, object],
        *,
        deck_count: int,
        removal_is_legal: bool = False,
    ) -> None:
        record = _decision(
            episode,
            step,
            selected=str(action["action"]),
        )
        record["selected_action"] = action
        record["observation_summary"] = {
            "player": {"deck_count": deck_count},
            "card_selection": {},
        }
        record["legal_action_semantics"] = (
            {"shop_purchase:card_removal": 1}
            if removal_is_legal
            else {}
        )
        records.append(record)

    purchase = {
        "action": "shop_purchase",
        "item": {"category": "card_removal"},
    }
    # Completed: confirmation is followed by an authoritative deck decrement.
    append(0, 0, purchase, deck_count=37, removal_is_legal=True)
    append(0, 1, {"action": "select_card"}, deck_count=37)
    append(0, 2, {"action": "confirm_selection"}, deck_count=37)
    append(0, 3, {"action": "proceed"}, deck_count=36)

    # Cancelled: it is not allowed to count as completion.
    append(1, 0, purchase, deck_count=30, removal_is_legal=True)
    append(1, 1, {"action": "select_card"}, deck_count=30)
    append(1, 2, {"action": "cancel_selection"}, deck_count=30)

    # Unresolved: merely entering the selection surface is not success.
    append(2, 0, purchase, deck_count=25, removal_is_legal=True)
    append(2, 1, {"action": "select_card"}, deck_count=25)

    journal = tmp_path / "shop-removal-outcomes.jsonl"
    journal.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_greedy_liveness_journal(journal)
    assert summary["card_removal_legal_decision_count"] == 3
    assert summary["card_removal_legal_candidate_count"] == 3
    assert summary["card_removal_purchase_attempt_count"] == 3
    assert summary["card_removal_completed_count"] == 1
    assert summary["card_removal_cancelled_count"] == 1
    assert summary["card_removal_unresolved_count"] == 1
    assert summary["card_removal_completion_rate"] == 1 / 3
    assert summary["card_removal_cancel_rate"] == 1 / 3


def test_three_step_shop_removal_cancel_cycle_is_a_liveness_failure(
    tmp_path: Path,
) -> None:
    records: list[dict[str, object]] = []
    for episode in range(16):
        deadlocked = episode < 11
        selected_actions = (
            (
                {
                    "action": "shop_purchase",
                    "item": {"category": "card_removal"},
                },
                {"action": "select_card"},
                {"action": "cancel_selection"},
            )
            * 8
            if deadlocked
            else ({"action": "proceed"},)
        )
        for step, selected_action in enumerate(selected_actions):
            record = _decision(
                episode,
                step,
                selected=str(selected_action["action"]),
            )
            record["selected_action"] = selected_action
            if deadlocked and step == len(selected_actions) - 1:
                record["deadlock"] = {
                    "cycle_span": 3,
                    "occurrences": 8,
                }
            records.append(record)

    journal = tmp_path / "shop-removal-cycle.jsonl"
    journal.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_greedy_liveness_journal(journal)
    assert summary["episode_count"] == 16
    assert summary["cycle_episode_count"] == 11
    assert summary["liveness_failure_episode_count"] == 11
    assert summary["liveness_failure_episode_rate"] == 11 / 16
    assert summary["card_removal_cancel_cycle_episode_count"] == 11
    # The old <=2 selection-only detector intentionally does not own this
    # cross-surface shop -> select -> cancel transaction cycle.
    assert summary["selection_cycle_episode_count"] == 0

    guard = evaluate_liveness_guard(
        summary,
        RuntimeConfig(
            evaluation_steps=(150_000,),
            evaluation_episodes=16,
            evaluation_liveness_guard_enabled=True,
        ),
    )
    assert guard["stop_requested"] is True
    assert any(
        item["kind"] == "liveness_failure_collapse"
        for item in guard["violations"]
    )


def test_liveness_guard_uses_frozen_baseline_before_stopping() -> None:
    config = RuntimeConfig(
        evaluation_steps=(5_000,),
        evaluation_episodes=16,
        evaluation_liveness_guard_enabled=True,
        evaluation_guard_liveness_baseline_failures=9,
        evaluation_guard_liveness_baseline_episodes=16,
        evaluation_guard_min_liveness_regression_rate=0.20,
    )

    equivalent_sample = evaluate_liveness_guard(
        {
            "episode_count": 8,
            "liveness_failure_episode_count": 5,
            "liveness_failure_episode_rate": 5 / 8,
            "selection_cycle_episode_rate": 0.0,
        },
        config,
    )
    assert equivalent_sample["stop_requested"] is False
    assert (
        equivalent_sample["thresholds"][
            "effective_liveness_failure_episode_rate"
        ]
        == pytest.approx(0.7625)
    )

    collapsed = evaluate_liveness_guard(
        {
            "episode_count": 16,
            "liveness_failure_episode_count": 13,
            "liveness_failure_episode_rate": 13 / 16,
            "selection_cycle_episode_rate": 0.0,
        },
        config,
    )
    assert collapsed["stop_requested"] is True
    violation = next(
        item
        for item in collapsed["violations"]
        if item["kind"] == "liveness_failure_collapse"
    )
    assert violation["threshold"] == pytest.approx(0.7625)
    assert violation["baseline"] == {
        "episodes": 16,
        "failures": 9,
        "rate": 9 / 16,
        "minimum_regression_rate": 0.20,
    }


def test_liveness_baseline_configuration_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="failures cannot exceed episodes"):
        RuntimeConfig(
            evaluation_guard_liveness_baseline_failures=10,
            evaluation_guard_liveness_baseline_episodes=8,
        )
    with pytest.raises(ValueError, match="require baseline episodes"):
        RuntimeConfig(
            evaluation_guard_liveness_baseline_failures=1,
            evaluation_guard_liveness_baseline_episodes=0,
        )
