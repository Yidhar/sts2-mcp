from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_rl.training.trajectory import (
    SemanticDeadlockDetector,
    TrajectoryJournal,
    _collection_count,
    semantic_decision_fingerprint,
)


def test_journal_collection_count_preserves_and_validates_multiplicity() -> None:
    assert _collection_count([{"id": "CARD.WOUND", "quantity": 50_000}]) == 50_000
    assert _collection_count({"cards": [{"id": "CARD.WOUND", "quantity": 7}]}) == 7
    with pytest.raises(ValueError, match="quantity"):
        _collection_count([{"id": "CARD.WOUND", "quantity": 0}])


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
    assert semantic_decision_fingerprint(observation_a, actions_a) == semantic_decision_fingerprint(
        observation_b, actions_b
    )


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


def test_semantic_deadlock_post_step_keeps_known_cycle_but_clears_novel_exit() -> None:
    action = {"kind": "continue", "action_handle": "volatile"}
    actions = (action,)

    known_cycle = SemanticDeadlockDetector(window_size=8, repeat_threshold=2)
    first = known_cycle.observe(
        step_index=0,
        observation={"state": "A"},
        legal_actions=actions,
        selected_action=action,
    )
    assert known_cycle.confirm_after_step(
        first,
        observation={"state": "B"},
        legal_actions=actions,
    ) is None
    middle = known_cycle.observe(
        step_index=1,
        observation={"state": "B"},
        legal_actions=actions,
        selected_action=action,
    )
    assert known_cycle.confirm_after_step(
        middle,
        observation={"state": "A"},
        legal_actions=actions,
    ) is None
    repeated = known_cycle.observe(
        step_index=2,
        observation={"state": "A"},
        legal_actions=actions,
        selected_action=action,
    )
    assert repeated is not None
    assert known_cycle.confirm_after_step(
        repeated,
        observation={"state": "B"},
        legal_actions=actions,
    ) is repeated

    novel_exit = SemanticDeadlockDetector(window_size=8, repeat_threshold=2)
    for step, before, after in ((0, "A", "B"), (1, "B", "A")):
        evidence = novel_exit.observe(
            step_index=step,
            observation={"state": before},
            legal_actions=actions,
            selected_action=action,
        )
        assert novel_exit.confirm_after_step(
            evidence,
            observation={"state": after},
            legal_actions=actions,
        ) is None
    repeated = novel_exit.observe(
        step_index=2,
        observation={"state": "A"},
        legal_actions=actions,
        selected_action=action,
    )
    assert repeated is not None
    assert novel_exit.confirm_after_step(
        repeated,
        observation={"state": "EXIT"},
        legal_actions=actions,
    ) is None

def _journal_decision(
    *,
    episode_id: str,
    step_index: int,
    outcome: str = "ongoing",
    deadlock: dict[str, object] | None = None,
    verbose_description: str = "verbose card text",
    candidate_count: int = 2,
) -> dict[str, object]:
    legal_actions = [
        {
            "kind": "choose_event_option",
            "model_action_kind": "event_option",
            "action_handle": f"volatile-{step_index}-hold",
            "index": 0,
            "label": "Hold on",
            "option": {
                "index": 0,
                "text_key": "SLIPPERY_BRIDGE.HOLD_ON_LOOP",
                "description": verbose_description,
            },
        },
        {
            "kind": "choose_event_option",
            "model_action_kind": "event_option",
            "action_handle": f"volatile-{step_index}-overcome",
            "index": 1,
            "label": "Overcome",
        },
    ]
    legal_actions.extend(
        {
            "kind": "choose_event_option",
            "model_action_kind": "event_option",
            "action_handle": f"volatile-{step_index}-extra-{index}",
            "index": index,
            "label": f"Extra option {index}",
            "option": {
                "index": index,
                "text_key": f"SLIPPERY_BRIDGE.EXTRA_{index}",
            },
        }
        for index in range(2, candidate_count)
    )
    return {
        "event": "decision",
        "episode_id": episode_id,
        "reset_seed": 23,
        "step_index": step_index,
        "observation": {
            "phase": "event",
            "decision_domain": "event",
            "semantic_state_hash": f"state-{step_index}",
            "run": {
                "act": 1,
                "floor": 9,
                "room_model_id": "SLIPPERY_BRIDGE",
                "room_type": "event",
            },
            "player": {
                "hp": 33,
                "max_hp": 67,
                "gold": 6,
                "deck": [
                    {
                        "id": "CARD.STRIKE_IRONCLAD",
                        "description": verbose_description,
                        "dynamic_vars": [{"name": "Damage", "value": 6}],
                    }
                    for _ in range(16)
                ],
                "hand": [],
                "relics": [{"id": "RELIC.LIZARD_TAIL"}] * 4,
                "potions": [{"id": "POTION.BLOCK"}],
            },
            "combat": {"in_progress": False, "enemies": []},
            "event": {
                "id": "SLIPPERY_BRIDGE",
                "page": "HOLD_ON_LOOP",
                "dynamic_vars": [{"name": "HpLoss", "value": step_index}],
            },
        },
        "legal_actions": legal_actions,
        "selected_index": 0,
        "selected_action": legal_actions[0],
        "policy_topk": [
            {"index": 0, "probability": 0.506},
            {"index": 1, "probability": 0.494},
        ],
        "value": -0.85,
        "reward": -0.01,
        "player_hp_lost": step_index,
        "revivals_used": step_index,
        "outcome": outcome,
        "deadlock": deadlock,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_trajectory_journal_writes_compact_steps_and_bounded_rich_snapshots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journal.jsonl"
    with TrajectoryJournal(path, snapshot_interval=3) as journal:
        for step in range(5):
            journal.write(
                _journal_decision(
                    episode_id="episode-one",
                    step_index=step,
                    outcome="success" if step == 4 else "ongoing",
                )
            )

    records = _read_jsonl(path)
    summaries = [item for item in records if item["record_kind"] == "summary"]
    snapshots = [item for item in records if item["record_kind"] == "rich_snapshot"]
    assert len(summaries) == 5
    assert {item["step_index"] for item in snapshots} == {0, 2, 4}
    assert all(item["journal_version"] == "sts2-trajectory-journal-v5" for item in records)

    ordinary = summaries[1]
    assert "observation" not in ordinary
    assert "legal_actions" not in ordinary
    assert ordinary["observation_summary"]["run"] == {
        "act": 1,
        "floor": 9,
        "room_model_id": "SLIPPERY_BRIDGE",
        "room_type": "event",
    }
    assert ordinary["observation_summary"]["player"]["deck_count"] == 16
    assert ordinary["legal_action_count"] == 2
    assert ordinary["legal_action_kinds"] == {"event_option": 2}
    assert "legal_action_semantics" not in ordinary
    assert ordinary["selected_action"]["label"] == "Hold on"
    assert ordinary["selected_action_fingerprint"]
    assert ordinary["policy_topk"][1]["action"]["label"] == "Overcome"
    assert ordinary["policy_topk"][1]["action_fingerprint"]

    first = next(item for item in snapshots if item["step_index"] == 0)
    periodic = next(item for item in snapshots if item["step_index"] == 2)
    last = next(item for item in snapshots if item["step_index"] == 4)
    assert first["snapshot_reasons"] == ["episode_first"]
    assert periodic["snapshot_reasons"] == ["periodic"]
    assert last["snapshot_reasons"] == ["episode_last"]
    assert len(first["observation"]["player"]["deck"]) == 16
    assert "action_handle" not in first["selected_action"]


def test_compact_action_preserves_transaction_entity_identity(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    event = _journal_decision(
        episode_id="transaction-episode",
        step_index=1,
    )
    action = {
        "action": "select_card",
        "kind": "select_card",
        "action_index": 14,
        "card_index": 14,
        "selection_operation": "select",
        "card": {
            # Prove the reviewed identity overlay, rather than alphabetic width
            # order, keeps the operator-facing identity.
            **{f"a_field_{index}": index for index in range(16)},
            "id": "CARD.SETUP_STRIKE",
            "name": "SETUP_STRIKE.title",
            "type": "Attack",
            "is_upgraded": False,
            "floor_added_to_deck": 6,
        },
        "selection": {
            "operation_type": "select",
            "mode": "DeckUpgrade",
            "prompt_id": "card_selection.TO_UPGRADE",
        },
    }
    event["legal_actions"] = [action]
    event["selected_action"] = action
    event["selected_index"] = 0
    event["policy_topk"] = [{"index": 0, "probability": 1.0}]

    with TrajectoryJournal(path) as journal:
        journal.write(event)

    summary = next(
        item for item in _read_jsonl(path) if item["record_kind"] == "summary"
    )
    selected = summary["selected_action"]
    assert selected["card_index"] == 14
    assert selected["card"] == {
        "floor_added_to_deck": 6,
        "id": "CARD.SETUP_STRIKE",
        "is_upgraded": False,
        "name": "SETUP_STRIKE.title",
        "type": "Attack",
    }
    assert selected["selection"] == {
        "mode": "DeckUpgrade",
        "operation_type": "select",
        "prompt_id": "card_selection.TO_UPGRADE",
    }


def test_trajectory_summary_keeps_parameterized_shop_action_surface(
    tmp_path: Path,
) -> None:
    event = _journal_decision(
        episode_id="shop-surface",
        step_index=0,
        outcome="success",
    )
    legal_actions = [
        {
            "action": "shop_purchase",
            "kind": "shop_purchase",
            "model_action_kind": "shop",
            "item": {"category": "card_removal", "slot_index": 13},
        },
        {
            "action": "shop_purchase",
            "kind": "shop_purchase",
            "model_action_kind": "shop",
            "item": {"category": "card", "slot_index": 2},
        },
        {
            "action": "shop_skip",
            "kind": "shop_skip",
            "model_action_kind": "shop",
        },
    ]
    event["legal_actions"] = legal_actions
    event["selected_action"] = legal_actions[0]
    event["policy_topk"] = [{"index": 0, "probability": 1.0}]

    path = tmp_path / "shop-surface.jsonl"
    with TrajectoryJournal(path) as journal:
        journal.write(event)

    summary = next(
        item
        for item in _read_jsonl(path)
        if item["record_kind"] == "summary"
    )
    assert summary["legal_action_kinds"] == {"shop": 3}
    assert summary["legal_action_semantics"] == {
        "shop_purchase:card": 1,
        "shop_purchase:card_removal": 1,
    }
    assert summary["selected_action"]["item"]["category"] == "card_removal"


def test_trajectory_journal_separates_model_candidate_and_raw_dispatch_indexes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "grouped-indexes.jsonl"
    event = _journal_decision(
        episode_id="grouped-episode",
        step_index=0,
        outcome="success",
    )
    raw_actions = [
        {
            "action_handle": f"select-{index}",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "card": {"id": "CARD.WOUND", "pile": "Discard"},
        }
        for index in range(3)
    ]
    event.update(
        {
            "legal_actions": raw_actions,
            "raw_legal_action_count": 3,
            "semantic_candidate_count": 1,
            "selected_index": 0,
            "selected_candidate_index": 0,
            "selected_dispatch_index": 2,
            "selected_action": raw_actions[2],
            "selected_action_multiplicity": 3,
            "selected_action_equivalence_fingerprint": "strict-wound-group",
            "policy_topk": [
                {
                    "index": 0,
                    "candidate_index": 0,
                    "dispatch_index": 2,
                    "multiplicity": 3,
                    "equivalence_fingerprint": "strict-wound-group",
                    "probability": 1.0,
                }
            ],
        }
    )

    with TrajectoryJournal(path) as journal:
        journal.write(event)

    records = _read_jsonl(path)
    summary = next(item for item in records if item["record_kind"] == "summary")
    assert summary["selected_index"] == 0
    assert summary["selected_candidate_index"] == 0
    assert summary["selected_dispatch_index"] == 2
    assert summary["selected_action_multiplicity"] == 3
    assert summary["selected_action"]["card"]["id"] == "CARD.WOUND"
    assert summary["raw_legal_action_count"] == 3
    assert summary["semantic_candidate_count"] == 1
    assert len(summary["policy_topk"]) == 1
    top = summary["policy_topk"][0]
    assert top["candidate_index"] == 0
    assert top["dispatch_index"] == 2
    assert top["multiplicity"] == 3
    assert top["equivalence_fingerprint"] == "strict-wound-group"
    assert top["action"]["action_handle"] == "select-2"
    assert top["action"]["card"]["id"] == "CARD.WOUND"
    assert top["action_fingerprint"]


def test_trajectory_journal_retains_bounded_rich_anomaly_context(
    tmp_path: Path,
) -> None:
    path = tmp_path / "anomaly.jsonl"
    with TrajectoryJournal(
        path,
        snapshot_interval=100,
        anomaly_context_steps=2,
    ) as journal:
        for step in range(5):
            journal.write(
                _journal_decision(
                    episode_id="event-loop",
                    step_index=step,
                    outcome="deadlock" if step == 4 else "ongoing",
                    deadlock=({"kind": "noncombat_no_durable_progress", "window": 256} if step == 4 else None),
                )
            )

    records = _read_jsonl(path)
    summaries = [item for item in records if item["record_kind"] == "summary"]
    snapshots = [item for item in records if item["record_kind"] == "rich_snapshot"]
    assert len(summaries) == 5
    assert {item["step_index"] for item in snapshots} == {0, 2, 3, 4}
    contexts = [item for item in snapshots if item["snapshot_reasons"] == ["anomaly_context"]]
    assert [item["step_index"] for item in contexts] == [2, 3]
    anomaly = next(item for item in snapshots if item["step_index"] == 4)
    assert anomaly["snapshot_reasons"] == ["anomaly", "episode_last"]
    assert anomaly["deadlock"]["kind"] == "noncombat_no_durable_progress"
    assert summaries[-1]["deadlock"]["window"] == 256


def test_trajectory_journal_volume_scales_with_compact_records_not_full_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "volume.jsonl"
    steps = 1_000
    verbose_description = "x" * 4_096
    example = _journal_decision(
        episode_id="long-loop",
        step_index=0,
        verbose_description=verbose_description,
    )
    old_full_volume = len(json.dumps(example, separators=(",", ":"))) * steps

    with TrajectoryJournal(
        path,
        snapshot_interval=128,
        anomaly_context_steps=4,
    ) as journal:
        for step in range(steps):
            event = _journal_decision(
                episode_id="long-loop",
                step_index=step,
                verbose_description=verbose_description,
            )
            journal.write(event)

    records = _read_jsonl(path)
    summaries = [item for item in records if item["record_kind"] == "summary"]
    snapshots = [item for item in records if item["record_kind"] == "rich_snapshot"]
    assert len(summaries) == steps
    assert len(snapshots) == 9  # first + seven periodic + explicit final
    assert path.stat().st_size < old_full_volume // 10


def test_production_journal_bounds_111_candidates_and_eight_step_anomaly_context(
    tmp_path: Path,
) -> None:
    path = tmp_path / "production-shape.jsonl"
    steps = 521
    long_label = "选择这一个很长的事件选项" * 1_024
    with TrajectoryJournal(path) as journal:
        for step in range(steps):
            event = _journal_decision(
                episode_id="candidate-111-loop",
                step_index=step,
                candidate_count=111,
                outcome="deadlock" if step == steps - 1 else "ongoing",
                deadlock=(
                    {
                        "kind": "noncombat_no_durable_progress",
                        "window": 256,
                    }
                    if step == steps - 1
                    else None
                ),
            )
            if step == 255:
                legal_actions = event["legal_actions"]
                assert isinstance(legal_actions, list)
                selected = legal_actions[0]
                assert isinstance(selected, dict)
                selected["label"] = long_label
            journal.write(event)

    records = _read_jsonl(path)
    summaries = [item for item in records if item["record_kind"] == "summary"]
    snapshots = [item for item in records if item["record_kind"] == "rich_snapshot"]
    assert len(summaries) == steps
    assert all(item["legal_action_count"] == 111 for item in summaries)

    # Production defaults are first, each 256th decision (zero-based 255/511),
    # eight steps preceding the anomaly, and the anomaly/episode-last itself.
    assert {item["step_index"] for item in snapshots} == {
        0,
        255,
        511,
        512,
        513,
        514,
        515,
        516,
        517,
        518,
        519,
        520,
    }
    assert len(snapshots) == 12
    contexts = [item for item in snapshots if item["snapshot_reasons"] == ["anomaly_context"]]
    assert [item["step_index"] for item in contexts] == list(range(512, 520))

    # An unexpected long scalar cannot inflate an ordinary record. Its full
    # value remains available in a rich snapshot when that step is sampled.
    bounded_label = summaries[255]["selected_action"]["label"]
    assert bounded_label["utf8_bytes"] == len(long_label.encode("utf-8"))
    assert len(bounded_label["sha256"]) == 64
    assert len(bounded_label["string_prefix"].encode("utf-8")) <= 160
    periodic_255 = next(item for item in snapshots if item["step_index"] == 255)
    assert periodic_255["selected_action"]["label"] == long_label

    summary_sizes = [
        len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for item in summaries
    ]
    rich_sizes = [
        len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for item in snapshots
    ]
    assert max(summary_sizes) < 16 * 1_024
    # A 30k-step anomaly has at most first + 117 periodic + 8 context +
    # anomaly/last rich records under the production 256/8 defaults.
    extrapolated_30k_bytes = (sum(summary_sizes) / len(summary_sizes)) * 30_000 + max(rich_sizes) * 127
    assert extrapolated_30k_bytes < 128 * 1_024 * 1_024

    # Context is appended only when the anomaly appears, so append order is
    # intentionally not chronological; consumers must sort by episode/step.
    summary_519_index = next(
        index for index, item in enumerate(records) if item["record_kind"] == "summary" and item["step_index"] == 519
    )
    context_512_index = next(
        index
        for index, item in enumerate(records)
        if item["record_kind"] == "rich_snapshot" and item["step_index"] == 512
    )
    assert context_512_index > summary_519_index
    assert records[context_512_index - 1]["step_index"] == 519


def test_compact_mapping_width_is_strictly_bounded_and_reports_omissions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wide-mapping.jsonl"
    event = _journal_decision(
        episode_id="wide-mapping",
        step_index=0,
        outcome="success",
    )
    legal_actions = event["legal_actions"]
    assert isinstance(legal_actions, list)
    selected = legal_actions[0]
    assert isinstance(selected, dict)
    selected["wide_payload"] = {f"field_{index:04d}": index for index in range(1_000)}

    with TrajectoryJournal(path) as journal:
        journal.write(event)

    records = _read_jsonl(path)
    summary = next(item for item in records if item["record_kind"] == "summary")
    compact_wide = summary["selected_action"]["wide_payload"]
    assert compact_wide["_mapping_width"] == {
        "key_count": 1_000,
        "omitted_count": 992,
    }
    assert [key for key in compact_wide if key.startswith("field_")] == [f"field_{index:04d}" for index in range(8)]
    assert len(compact_wide) == 9  # eight projected keys plus fixed width metadata
    assert len(json.dumps(summary, separators=(",", ":"))) < 8 * 1_024

    snapshot = next(item for item in records if item["record_kind"] == "rich_snapshot")
    assert len(snapshot["selected_action"]["wide_payload"]) == 1_000


def test_compact_observation_preserves_live_combat_and_deck_shape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "live-shape.jsonl"
    event = _journal_decision(
        episode_id="live-combat-shape",
        step_index=7,
        outcome="success",
    )
    event["observation"] = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {
            "active": True,
            "floor": 7,
            "room_type": "Monster",
            "room_model": "ROOM.TEST",
        },
        "player": {
            "character_id": "CHARACTER.IRONCLAD",
            "hp": 61,
            "max_hp": 80,
            "block": 3,
            "gold": 99,
            "deck": 2,
            "deck_cards": [
                {"id": "CARD.STRIKE", "type": "Attack", "cost": 1},
                {"id": "CARD.DEFEND", "type": "Skill", "cost": 1},
            ],
            "relics": [{"id": "RELIC.BURNING_BLOOD"}],
            "potions": [{"id": "POTION.FIRE"}],
        },
        "combat": {
            "round": 2,
            "energy": 2,
            "max_energy": 3,
            "hand": [
                {"id": "CARD.STRIKE", "type": "Attack", "cost": 1},
            ],
            "draw": 2,
            "discard": 1,
            "exhaust": 0,
            "draw_pile": {
                "count": 2,
                "cards": [
                    {"id": "CARD.SECRET_A"},
                    {"id": "CARD.SECRET_B"},
                ],
            },
            "discard_pile": {
                "count": 1,
                "cards": [{"id": "CARD.SECRET_C"}],
            },
            "exhaust_pile": {"count": 0, "cards": []},
            "play_pile": {"count": 1, "cards": [{"id": "CARD.IN_PLAY"}]},
            "enemies": [{"model_id": "MONSTER.CULTIST", "hp": 31, "max_hp": 48}],
        },
    }

    with TrajectoryJournal(path) as journal:
        journal.write(event)

    records = _read_jsonl(path)
    summary = next(item for item in records if item["record_kind"] == "summary")
    observation = summary["observation_summary"]
    assert observation["run"]["room_model"] == "ROOM.TEST"
    assert observation["player"]["character_id"] == "CHARACTER.IRONCLAD"
    assert observation["player"]["deck_count"] == 2
    assert observation["player"]["deck_cards_count"] == 2
    assert observation["combat"]["energy"] == 2
    assert observation["combat"]["max_energy"] == 3
    assert observation["combat"]["hand_count"] == 1
    assert observation["combat"]["hand"] == [
        {"cost": 1, "id": "CARD.STRIKE", "type": "Attack"},
    ]
    assert observation["combat"]["draw_pile_count"] == 2
    assert observation["combat"]["discard_pile_count"] == 1
    assert observation["combat"]["exhaust_pile_count"] == 0
    assert observation["combat"]["play_pile_count"] == 1
