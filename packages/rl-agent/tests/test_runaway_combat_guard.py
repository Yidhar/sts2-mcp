from __future__ import annotations

from typing import Any

from sts2_rl.training import build_training_resources
from sts2_rl.training.episode_replay import BoundaryOutcome
from sts2_rl.training.runaway_combat import (
    RunawayCombatGuard,
    combat_card_burden,
)
from tests.test_episodic_collector import (
    _AuditableReward,
    _observation,
    _run_config,
    _ScriptedRunBackend,
)


def _status_card(quantity: int) -> dict[str, object]:
    return {
        "id": "CARD.SYNTHETIC_STATUS",
        "type": "Status",
        "quantity": quantity,
    }


def _ordinary_card(quantity: int = 1) -> dict[str, object]:
    return {
        "id": "CARD.SYNTHETIC_ATTACK",
        "type": "Attack",
        "quantity": quantity,
    }


def _runaway_observation(*, status_cards: int = 35_505) -> dict[str, Any]:
    observation = _observation(
        act=3,
        floor=46,
        combat=True,
        revivals=11_980,
        hp_loss=312_510.0,
        room_model_id="TEST_SUBJECT_BOSS",
    )
    # Mirrors the isolated failure's compact translated shape: 35,510 cards
    # across draw/discard plus five Wounds in hand and one in exhaust, with quantities rather than
    # materialising tens of thousands of equal mappings.
    hand_status = min(5, status_cards)
    draw_status = min(666, status_cards - hand_status)
    remaining_status = status_cards - hand_status - draw_status
    exhaust_status = min(1, remaining_status)
    discard_status = remaining_status - exhaust_status
    player = observation["player"]
    player.update(
        {
            "hand": ([_status_card(hand_status)] if hand_status else []),
            "draw_pile": (
                ([_status_card(draw_status)] if draw_status else [])
                + [_ordinary_card(4)]
            ),
            "draw_pile_count": draw_status + 4,
            "discard_pile": (
                ([_status_card(discard_status)] if discard_status else [])
                + [_ordinary_card(7)]
            ),
            "discard_pile_count": discard_status + 7,
            # The authoritative count includes three ordinary cards. The
            # compact visible list only needs the one factual Status entry for
            # this detector regression and keeps the tiny test encoder bounded.
            "exhaust_pile": (
                [_status_card(exhaust_status)] if exhaust_status else []
            ),
            "exhaust_pile_count": 4,
        }
    )
    return observation


def _end_turn_actions() -> tuple[dict[str, object], ...]:
    return (
        {
            "action_handle": "sim:0:end_turn",
            "action_id": "sim:0:end_turn",
            "kind": "end_turn",
            "model_action_kind": "end_turn",
        },
    )


def test_incident_scale_status_burden_is_detected_from_compressed_quantities() -> None:
    observation = _runaway_observation()
    burden = combat_card_burden(observation)
    guard = RunawayCombatGuard()

    guard.observe(
        observation=observation,
        legal_actions=_end_turn_actions(),
        no_net_progress_steps=0,
    )
    status = guard.observe(
        observation=observation,
        legal_actions=_end_turn_actions(),
        no_net_progress_steps=236,
    )

    assert burden.total_cards == 35_519
    assert burden.status_cards == 35_505
    assert status.triggered
    assert status.reason == "catastrophic_status_card_burden"
    assert status.only_end_turn
    assert status.status_fraction > 0.99


def test_guard_does_not_kill_an_ordinary_forced_end_turn() -> None:
    observation = _runaway_observation(status_cards=20)
    guard = RunawayCombatGuard()

    status = guard.observe(
        observation=observation,
        legal_actions=_end_turn_actions(),
        no_net_progress_steps=236,
    )

    assert not status.triggered
    assert status.status_cards == 20


def test_guard_requires_the_authoritative_surface_to_be_only_end_turn() -> None:
    observation = _runaway_observation()
    guard = RunawayCombatGuard()
    actions = (
        *_end_turn_actions(),
        {
            "action_handle": "sim:1:play_card",
            "kind": "play_card",
            "model_action_kind": "play_card",
        },
    )

    status = guard.observe(
        observation=observation,
        legal_actions=actions,
        no_net_progress_steps=236,
    )

    assert not status.triggered
    assert not status.only_end_turn


def test_guard_requires_a_no_net_enemy_hp_progress_age() -> None:
    observation = _runaway_observation()
    guard = RunawayCombatGuard()

    status = guard.observe(
        observation=observation,
        legal_actions=_end_turn_actions(),
        no_net_progress_steps=0,
    )

    assert not status.triggered


def test_guard_can_detect_rapid_status_growth_before_catastrophic_scale() -> None:
    guard = RunawayCombatGuard()
    before = _runaway_observation(status_cards=700)
    after = _runaway_observation(status_cards=1_300)
    guard.observe(
        observation=before,
        legal_actions=_end_turn_actions(),
        no_net_progress_steps=15,
    )

    status = guard.observe(
        observation=after,
        legal_actions=_end_turn_actions(),
        no_net_progress_steps=16,
    )

    assert status.triggered
    assert status.reason == "rapid_status_card_growth"
    assert status.observed_status_growth == 600


class _ForcedEndTurnRunBackend(_ScriptedRunBackend):
    @staticmethod
    def _actions() -> tuple[dict[str, Any], ...]:
        return _end_turn_actions()


def test_collector_stops_before_the_next_rpc_and_commits_policy_failure_replay() -> None:
    runaway = _runaway_observation()
    backend = _ForcedEndTurnRunBackend(
        [runaway for _ in range(65)],
        terminal_result=None,
    )
    resources = build_training_resources(
        _run_config(max_episode_steps=64, combat_window=128),
        backend=backend,
    )
    resources.collector.reward_calculator = _AuditableReward()
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
        )
    finally:
        resources.close()

    # The 16th accepted transition supplies the factual terminal prefix. The
    # collector never sends the otherwise-forced 17th end_turn RPC.
    assert backend._step == 16
    assert episode.metrics.steps == 16
    assert episode.metrics.combat_progress_stalled
    assert episode.metrics.combat_policy_failed
    assert episode.metrics.terminal_reason == "combat_progress_stall"
    evidence = episode.metrics.stall_evidence
    assert evidence is not None
    assert evidence["kind"] == "combat_runaway_status_burden"
    assert evidence["visible_status_cards"] == 35_505
    assert evidence["legal_action_kinds"] == {"end_turn": 1}

    completed = episode.completed_episode
    assert completed is not None
    assert completed.completion.authoritative
    assert completed.won is False
    assert completed.completion.terminal_reason == "combat_progress_stall"
    assert completed.steps[-1].decision.combat_boundary is BoundaryOutcome.FAILED
    assert completed.steps[-1].decision.act_boundary is BoundaryOutcome.FAILED
