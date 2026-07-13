from __future__ import annotations

import json
from pathlib import Path

from sts2_rl.training.trajectory import (
    SemanticDeadlockDetector,
    TrajectoryJournal,
    semantic_decision_fingerprint,
)


def test_semantic_fingerprint_ignores_transport_ids_and_candidate_order() -> None:
    observation_a = {
        "episode_id": "one",
        "state_version": 10,
        "player": {"hp": 30, "max_hp": 50},
    }
    observation_b = {
        "episode_id": "two",
        "state_version": 99,
        "player": {"hp": 30, "max_hp": 50},
    }
    actions_a = [
        {"action_handle": "uuid-a", "kind": "a"},
        {"action_handle": "uuid-b", "kind": "b"},
    ]
    actions_b = [
        {"action_handle": "different-b", "kind": "b"},
        {"action_handle": "different-a", "kind": "a"},
    ]
    assert semantic_decision_fingerprint(
        observation_a, actions_a
    ) == semantic_decision_fingerprint(observation_b, actions_b)


def test_semantic_deadlock_requires_exact_recurrent_state_action_pair() -> None:
    detector = SemanticDeadlockDetector(window_size=8, repeat_threshold=3)
    action = {"kind": "continue", "action_handle": "volatile"}
    evidence = None
    for step in range(3):
        evidence = detector.observe(
            step_index=step,
            observation={"player": {"hp": 10}, "state_version": step},
            legal_actions=(action,),
            selected_action=action,
        )
    assert evidence is not None
    assert evidence.occurrences == 3
    assert evidence.first_step == 0
    assert evidence.last_step == 2
    assert evidence.cycle_span == 1

    detector.reset()
    for step in range(6):
        evidence = detector.observe(
            step_index=step,
            observation={"player": {"hp": 10 - step}},
            legal_actions=(action,),
            selected_action=action,
        )
        assert evidence is None


def test_trajectory_journal_writes_versioned_compact_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    with TrajectoryJournal(path) as journal:
        journal.write(
            {
                "event": "decision",
                "episode_id": "volatile",
                "step_index": 4,
                "observation": {"player": {"hp": 7}},
            }
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["journal_version"] == "sts2-trajectory-journal-v2"
    assert payload["event"] == "decision"
    assert payload["episode_id"] == "volatile"
    assert payload["step_index"] == 4
    assert payload["observation"]["player"]["hp"] == 7
