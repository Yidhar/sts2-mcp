from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sts2_rl.training import build_training_resources, summarize_evaluation
from sts2_rl.training import runtime as runtime_module
from sts2_rl.training.collector import EpisodeMetrics, _is_forge_selection_surface
from sts2_rl.training.config import FailureCreditConfig, TransactionLearningConfig
from sts2_rl.training.episode_replay import EpisodeDecisionStep
from sts2_rl.training.failure_credit import (
    DirectPolicyTarget,
    EvidenceStratum,
    FailureOutcome,
)
from sts2_rl.training.runtime import (
    _checkpoint_role_for_prefix,
    _deadlock_streak_transition,
    _is_healthy_rollback_checkpoint,
)
from sts2_rl.training.trajectory import TrajectoryJournal
from tests.test_v2_training_pipeline import (
    FakeCombatBackend,
    StaticEventLoopBackend,
    TerminalWithoutObservationFlagsBackend,
    _config,
    _event_loop_config,
)


class _RestForgeSelectionCycleBackend(TerminalWithoutObservationFlagsBackend):
    """Open an upgrade grid from a rest surface, then toggle forever."""

    def __init__(self) -> None:
        super().__init__(terminal_step=100)

    @staticmethod
    def _card(card_id: str, instance_id: str) -> dict[str, object]:
        return {
            "id": card_id,
            "instance_id": instance_id,
            "source_pile": "Deck",
            "is_upgraded": False,
        }

    def _actions(self) -> tuple[dict[str, object], ...]:
        if self._step == 0:
            return (
                {
                    "action_handle": "rest-forge",
                    "action": "choose_rest_option",
                    "kind": "choose_rest_option",
                    "model_action_kind": "rest_site",
                    "model_action_variant": "forge",
                    "index": 0,
                },
                {
                    "action_handle": "rest-alternate",
                    "action": "choose_rest_option",
                    "kind": "choose_rest_option",
                    "model_action_kind": "rest_site",
                    "model_action_variant": "rest",
                    "index": 1,
                },
            )
        selected = bool((self._step - 1) % 2)
        strike = self._card("CARD.STRIKE", "strike-1")
        defend = self._card("CARD.DEFEND", "defend-1")
        return (
            {
                "action_handle": f"toggle:{self._step}",
                "action": "deselect_card" if selected else "select_card",
                "kind": "deselect_card" if selected else "select_card",
                "model_action_kind": "card_selection",
                "model_action_variant": "deselect" if selected else "select",
                "selection_operation": "deselect" if selected else "select",
                "card": {
                    **strike,
                    "selection_membership": (
                        "selected" if selected else "selectable"
                    ),
                    "is_selected": selected,
                },
            },
            {
                "action_handle": f"alternate:{self._step}",
                "action": "select_card",
                "kind": "select_card",
                "model_action_kind": "card_selection",
                "model_action_variant": "select",
                "selection_operation": "select",
                "card": {
                    **defend,
                    "selection_membership": "selectable",
                    "is_selected": False,
                },
            },
        )

    def _observation(self, *, terminal: bool = False) -> dict[str, object]:
        strike = self._card("CARD.STRIKE", "strike-1")
        defend = self._card("CARD.DEFEND", "defend-1")
        observation: dict[str, object] = {
            "phase": "rest" if self._step == 0 else "selection",
            "decision_domain": "resource" if self._step == 0 else "build",
            "state_type": "rest_site" if self._step == 0 else "card_select",
            "screen": "REST_SITE" if self._step == 0 else "SELECTION",
            "player": {
                "character": "IRONCLAD",
                "hp": 40,
                "max_hp": 70,
                "gold": 99,
                "deck": [strike, defend],
                "relics": [],
                "potions": [],
            },
            "combat": {"in_progress": False, "enemies": []},
            "run": {
                "active": not terminal,
                "act": 1,
                "floor": 9,
                "room_type": "rest",
                "room_model_id": "REST_SITE",
            },
        }
        if self._step == 0:
            return observation
        selected = bool((self._step - 1) % 2)
        observation["card_selection"] = {
            "mode": "SimpleGrid",
            "prompt_id": "card_selection.TO_UPGRADE",
            "operation_type": "upgrade",
            "source_zone": "Deck",
            "destination_zone": "Deck",
            "min_select": 1,
            "max_select": 1,
            "cards": [defend] if selected else [strike, defend],
            "selected_cards": [strike] if selected else [],
            "selected_count": int(selected),
            "remaining_picks": 1 - int(selected),
            "requires_manual_confirmation": True,
            "can_confirm": selected,
        }
        return observation


class _RestForgeSelectionSuccessBackend(_RestForgeSelectionCycleBackend):
    """Complete one rest-site forge transaction and then win the run."""

    def __init__(self) -> None:
        super().__init__()
        self.result_terminal_step = 3

    def _actions(self) -> tuple[dict[str, object], ...]:
        if self._step <= 1:
            return super()._actions()
        return (
            {
                "action_handle": "confirm-upgrade",
                "action": "confirm_selection",
                "kind": "confirm_selection",
                "model_action_kind": "card_selection",
                "model_action_variant": "confirm",
                "selection_operation": "confirm",
            },
            {
                "action_handle": "deselect-upgrade",
                "action": "deselect_card",
                "kind": "deselect_card",
                "model_action_kind": "card_selection",
                "model_action_variant": "deselect",
                "selection_operation": "deselect",
                "card": {
                    **self._card("CARD.STRIKE", "strike-1"),
                    "selection_membership": "selected",
                    "is_selected": True,
                },
            },
        )

    def _observation(self, *, terminal: bool = False) -> dict[str, object]:
        observation = super()._observation(terminal=terminal)
        if not terminal:
            return observation
        observation.pop("card_selection", None)
        observation.update(
            phase="map",
            decision_domain="map",
            state_type="map",
            screen="MAP",
        )
        player = observation["player"]
        assert isinstance(player, dict)
        player["deck"] = [
            {
                **self._card("CARD.STRIKE", "strike-1"),
                "is_upgraded": True,
                "upgrade_level": 1,
            },
            self._card("CARD.DEFEND", "defend-1"),
        ]
        run = observation["run"]
        assert isinstance(run, dict)
        run.update(active=False, room_type="map", room_model_id="MAP")
        return observation


def _recovery_config(*, max_steps: int, repeat_threshold: int):
    base = _event_loop_config(durable_window=max_steps)
    return replace(
        base,
        environment=replace(base.environment, max_episode_steps=max_steps),
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=max(8, repeat_threshold * 2),
            deadlock_repeat_threshold=repeat_threshold,
            noncombat_durable_progress_window=max_steps,
        ),
        curriculum=replace(
            base.curriculum,
            epsilon_start=0.15,
            epsilon_end=0.05,
            selection_surface_epsilon_floor=0.25,
        ),
        transaction_learning=TransactionLearningConfig(enabled=False),
        episodic_learning=replace(base.episodic_learning, enabled=True),
    )


def test_selection_epsilon_floor_is_targeted_and_behavior_is_journaled(
    tmp_path: Path,
) -> None:
    config = _recovery_config(max_steps=4, repeat_threshold=8)
    resources = build_training_resources(
        config,
        backend=_RestForgeSelectionCycleBackend(),
    )
    journal_path = tmp_path / "selection-floor.jsonl"
    try:
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        with TrajectoryJournal(journal_path) as journal:
            episode = resources.collector.collect_episode(
                epsilon=0.05,
                deterministic=False,
                record=False,
                trajectory_journal=journal,
            )
    finally:
        resources.close()

    assert episode.metrics.targeted_selection_exploration_decisions == 4
    assert episode.metrics.maximum_effective_collection_epsilon == pytest.approx(
        0.25
    )
    decisions = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("event") == "decision"
    ]
    assert len(decisions) == 4
    assert all(item["targeted_selection_exploration"] is True for item in decisions)
    assert all(item["effective_collection_epsilon"] == pytest.approx(0.25) for item in decisions)
    assert all(math.isfinite(float(item["behavior_log_probability"])) for item in decisions)


def test_selection_cycle_exempts_entrance_policy_without_legacy_transaction_replay() -> None:
    config = _recovery_config(max_steps=20, repeat_threshold=3)
    resources = build_training_resources(
        config,
        backend=_RestForgeSelectionCycleBackend(),
    )
    try:
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(
            epsilon=0.0,
            deterministic=True,
            record=True,
        )
    finally:
        resources.close()

    assert resources.transaction_replay is None
    assert episode.metrics.deadlocked
    assert episode.metrics.selection_action_cycle
    assert episode.metrics.trusted_policy_failure
    assert episode.metrics.selection_transactions_started == 1
    assert episode.metrics.selection_transactions_completed == 0
    assert episode.metrics.rest_site_selection_transactions_started == 1
    assert episode.metrics.rest_site_selection_transactions_completed == 0
    assert episode.metrics.forge_selection_transactions_started == 1
    assert episode.metrics.forge_selection_transactions_completed == 0
    assert episode.transaction_traces == ()

    completed = episode.completed_episode
    assert completed is not None
    entrance: EpisodeDecisionStep = completed.steps[0].decision
    assert entrance.decision_surface == "rest_site"
    assert entrance.policy_decision is False
    assert any(step.decision.policy_decision for step in completed.steps[1:-1])
    assert next(
        step for unroll in episode.unrolls for step in unroll.steps
    ).policy_decision is False


def test_forge_surface_and_evaluation_metric_are_exact() -> None:
    backend = _RestForgeSelectionCycleBackend()
    backend._step = 1
    observation = backend._observation()
    actions = backend._actions()
    assert _is_forge_selection_surface(observation, actions)

    base = _recovery_config(max_steps=4, repeat_threshold=8)
    resources = build_training_resources(base, backend=backend)
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=False,
            maximum_steps=1,
        )
    finally:
        resources.close()
    metric = replace(
        episode.metrics,
        forge_selection_transactions_started=2,
        forge_selection_transactions_completed=1,
    )
    summary = summarize_evaluation([metric], objective="run")
    assert summary["forge_transactions_started"] == 2
    assert summary["forge_transactions_completed"] == 1
    assert summary["forge_transaction_completion_rate"] == pytest.approx(0.5)


def test_forge_selection_clean_exit_counts_and_retains_positive_credit() -> None:
    base = _recovery_config(max_steps=8, repeat_threshold=8)
    config = replace(
        base,
        failure_credit=FailureCreditConfig(
            mode="shadow",
            burn_in_steps=1,
            maximum_context_steps=16,
        ),
    )
    resources = build_training_resources(
        config,
        backend=_RestForgeSelectionSuccessBackend(),
    )
    try:
        resources.collector.bind_failure_credit_run_id("forge-success-path")
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(
            epsilon=0.0,
            deterministic=True,
            record=True,
        )
    finally:
        resources.close()

    assert episode.metrics.run_won
    assert not episode.metrics.deadlocked
    assert episode.metrics.selection_transactions_started == 1
    assert episode.metrics.selection_transactions_completed == 1
    assert episode.metrics.rest_site_selection_transactions_started == 1
    assert episode.metrics.rest_site_selection_transactions_completed == 1
    assert episode.metrics.forge_selection_transactions_started == 1
    assert episode.metrics.forge_selection_transactions_completed == 1

    funnel = episode.failure_credit_shadow_metrics
    assert funnel is not None
    assert funnel.completion_controls >= 1
    assert funnel.actor_actionable_records >= 1
    completions = tuple(
        record
        for record in episode.failure_credit_records
        if record.incident.outcome is FailureOutcome.COMPLETED
    )
    assert completions
    assert all(
        EvidenceStratum.COMPLETION_CONTROL in record.plan.strata
        for record in completions
    )
    preferred = tuple(
        target
        for record in completions
        for target in record.plan.direct_policy_targets
        if target.target is DirectPolicyTarget.PREFER
    )
    assert preferred


def test_deadlock_streak_alerts_once_per_consecutive_run() -> None:
    current = 0
    triggers: list[bool] = []
    for deadlocked in (True, True, True, True, True, False, True):
        current, triggered = _deadlock_streak_transition(
            current,
            deadlocked=deadlocked,
            alert_threshold=4,
        )
        triggers.append(triggered)
    assert triggers == [False, False, False, True, False, False, False]
    assert current == 1


def test_rollback_anchor_requires_hashed_health_role_not_directory_name() -> None:
    path = Path("healthy-validation-step-000025000")
    assert not _is_healthy_rollback_checkpoint(path, {"checkpoint_role": "ordinary"})
    assert not _is_healthy_rollback_checkpoint(
        Path("periodic-step-000025000"),
        {"checkpoint_role": "ordinary"},
    )
    assert _is_healthy_rollback_checkpoint(
        Path("renamed-anywhere"),
        {"checkpoint_role": "healthy_evaluation_anchor"},
    )
    assert _checkpoint_role_for_prefix("healthy-validation") == (
        "healthy_evaluation_anchor"
    )
    assert _checkpoint_role_for_prefix("guard-alert") == "guard_failure_evidence"
    assert _checkpoint_role_for_prefix("periodic") == "ordinary"


def _evaluation_metric(index: int) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode_id=f"heldout-{index}",
        reset_seed=1_000_001 + index * 2,
        steps=2,
        reward_total=1.0,
        terminal_reason="run_victory",
        truncated=False,
        run_won=True,
        combat_won=True,
        act1_cleared=True,
        max_act=3,
        max_floor=46,
        policy_decisions=1,
        forced_decisions=1,
        maximum_observed_candidates=2,
        deadlocked=False,
        combat_progress_stalled=False,
        maximum_combat_no_net_progress_steps=0,
        noncombat_progress_stalled=False,
        maximum_noncombat_no_durable_progress_steps=0,
        revivals_used=0,
        revival_free_combat_win=True,
        revival_free_act1_clear=True,
        revival_free_run_win=True,
        player_hp_lost=0.0,
        stall_evidence=None,
    )


def test_guard_failure_rolls_back_to_hashed_health_anchor_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _config(total_steps=4)
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            log_dir="runs/v33-guard-rollback",
            checkpoint_dir="checkpoints/v33-guard-rollback",
            checkpoint_interval_steps=100,
            evaluation_steps=(0, 2),
            evaluation_episodes=16,
            early_evaluation_steps=(),
            final_audit_steps=(),
            evaluation_liveness_guard_enabled=True,
            evaluation_guard_min_liveness_episodes=16,
            evaluation_guard_liveness_baseline_failures=0,
            evaluation_guard_liveness_baseline_episodes=16,
            evaluation_guard_min_liveness_regression_rate=0.20,
            evaluation_guard_failure_action="rollback_continue",
            evaluation_guard_max_rollbacks=1,
        ),
    )
    failure_counts = iter((0, 16, 0))

    def fake_evaluation(
        _resources: object,
        *,
        episodes: int,
        **_kwargs: object,
    ) -> tuple[list[EpisodeMetrics], dict[str, object]]:
        failures = next(failure_counts)
        results = [_evaluation_metric(index) for index in range(episodes)]
        return results, {
            "greedy_liveness": {
                "episode_count": episodes,
                "selection_cycle_episode_rate": 0.0,
                "liveness_failure_episode_count": failures,
                "liveness_failure_episode_rate": failures / episodes,
            }
        }

    monkeypatch.setattr(
        runtime_module,
        "_evaluate_training_gate",
        fake_evaluation,
    )

    state = runtime_module.run_training(
        config,
        backend=FakeCombatBackend(),
    )

    assert state.environment_steps == 4
    assert state.evaluation_guard_rollbacks == 1
    assert state.evaluation_episodes == 32
    metrics_path = next(
        (tmp_path / "runs" / "v33-guard-rollback").glob("run-*/metrics.jsonl")
    )
    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    rollback = next(
        event for event in events if event["event"] == "evaluation_guard_rollback"
    )
    assert rollback["rollback_number"] == 1
    assert "healthy-gate-zero-step-000000000" in rollback["healthy_checkpoint"]
    assert "guard-alert-step-000000002" in rollback["failed_checkpoint"]
    assert "guard-rollback-restored-step-000000000" in rollback[
        "restored_checkpoint"
    ]

    checkpoint_root = next(
        (tmp_path / "checkpoints" / "v33-guard-rollback").glob("run-*")
    )
    restored_metadata = json.loads(
        (
            checkpoint_root
            / "guard-rollback-restored-step-000000000"
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert restored_metadata["checkpoint_role"] == "healthy_evaluation_anchor"
    assert restored_metadata["provenance"]["checkpoint_load_mode"] == (
        "in_process_rollback"
    )
    assert restored_metadata["training_state"]["evaluation_guard_rollbacks"] == 1


def test_four_training_deadlocks_publish_one_immediate_alert_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _event_loop_config(durable_window=4)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=4),
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=8,
            deadlock_repeat_threshold=2,
        ),
        runtime=replace(
            base.runtime,
            total_environment_steps=20,
            log_dir="runs/v33-deadlock-alert",
            checkpoint_dir="checkpoints/v33-deadlock-alert",
            checkpoint_interval_steps=100,
            evaluation_steps=(),
            early_evaluation_steps=(),
            final_audit_steps=(),
            training_deadlock_streak_alert_episodes=4,
        ),
    )

    state = runtime_module.run_training(
        config,
        backend=StaticEventLoopBackend(),
    )

    assert state.environment_steps == 20
    assert state.episodes >= 5
    assert state.training_deadlock_alerts == 1
    checkpoints = tuple(
        (tmp_path / "checkpoints" / "v33-deadlock-alert").glob(
            "run-*/deadlock-streak-alert-step-*"
        )
    )
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    metadata = json.loads(
        (checkpoint / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["checkpoint_role"] == "deadlock_alert_evidence"
    assert metadata["training_state"]["training_deadlock_streak"] == 4
    metrics_path = next(
        (tmp_path / "runs" / "v33-deadlock-alert").glob(
            "run-*/metrics.jsonl"
        )
    )
    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    alerts = [
        event
        for event in events
        if event["event"] == "training_deadlock_streak_alert"
    ]
    assert len(alerts) == 1
    assert alerts[0]["training_deadlock_streak"] == 4
    assert alerts[0]["checkpoint"] == str(checkpoint)
