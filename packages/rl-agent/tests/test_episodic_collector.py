from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sts2_baseline import TaskReward
from sts2_rl.contracts import (
    BackendCapabilities,
    CombatResetRequest,
    EnvironmentResult,
    EnvironmentTransition,
    ResetRequest,
    StepRequest,
)
from sts2_rl.training import (
    DiagnosticsConfig,
    EpisodeMetrics,
    build_training_resources,
    run_training,
    summarize_evaluation,
)
from sts2_rl.training.collector import (
    CollectionProtocolError,
    _act_exit_hp_ratio,
    _episodic_decision_surface,
    _extended_transaction_option_endpoint,
    _macro_return_diagnostics,
    _noncombat_durable_projections,
)
from sts2_rl.training.episode_replay import BoundaryOutcome
from tests.test_v2_training_pipeline import _config


def test_macro_return_diagnostics_group_factual_returns_without_learning_labels() -> None:
    diagnostics = _macro_return_diagnostics(
        rewards=(0.0, 0.5, 1.0),
        discounts=(1.0, 1.0, 0.0),
        decisions=(
            (0, "rest_site", "choose_rest_option:rest", "low_25_50", 1),
            (1, "rest_site", "choose_rest_option:rest", "low_25_50", 1),
            (1, "rest_site", "choose_rest_option:smith", "low_25_50", 1),
        ),
        authoritative=True,
    )
    by_action = {item.action_key: item for item in diagnostics}
    assert by_action["choose_rest_option:rest"].count == 2
    assert by_action["choose_rest_option:rest"].mean_return == pytest.approx(1.5)
    assert by_action["choose_rest_option:smith"].count == 1
    assert by_action["choose_rest_option:smith"].mean_return == pytest.approx(1.5)
    assert _macro_return_diagnostics(
        rewards=(1.0,),
        discounts=(0.0,),
        decisions=((0, "map", "choose_map_node", "high_75_100", 1),),
        authoritative=False,
    ) == ()


def test_final_act_health_uses_last_player_receipt_when_victory_terminal_is_sparse() -> None:
    before = _observation(act=3, floor=46, combat=False)
    before["player"]["hp"] = 28
    terminal = {
        "phase": "terminal",
        "run": {"active": False, "act": 3, "floor": 46},
    }

    assert _act_exit_hp_ratio(
        before_observation=before,
        after_observation=terminal,
        authoritative_run_result="victory",
    ) == pytest.approx(28.0 / 80.0)


def test_missing_player_receipt_on_nonterminal_act_boundary_remains_fail_closed() -> None:
    before = _observation(act=1, floor=17, combat=False)
    malformed_after = {
        "phase": "map",
        "run": {"active": True, "act": 2, "floor": 18},
    }

    with pytest.raises(
        CollectionProtocolError,
        match="no positive player max HP",
    ):
        _act_exit_hp_ratio(
            before_observation=before,
            after_observation=malformed_after,
            authoritative_run_result=None,
        )


def test_extended_transaction_endpoint_stops_before_next_rest_choice_or_at_act_boundary() -> None:
    trace = SimpleNamespace(
        start_step=0,
        lifecycle=SimpleNamespace(
            support_eligible=True,
            entry_step_index=0,
            exit_step_index=0,
        ),
    )
    step = lambda floor, surface, boundary=BoundaryOutcome.NONE: SimpleNamespace(  # noqa: E731
        floor=floor,
        decision_surface=surface,
        act_boundary=boundary,
    )

    next_rest = _extended_transaction_option_endpoint(
        trace,
        episodic_steps=(
            step(4, "shop"),
            step(5, "combat"),
            step(6, "card_reward"),
            step(9, "rest_site"),
        ),
        authoritative_outcome=False,
    )
    assert next_rest == (2, "next_rest_site")

    next_act = _extended_transaction_option_endpoint(
        trace,
        episodic_steps=(
            step(4, "shop"),
            step(5, "combat"),
            step(17, "combat", BoundaryOutcome.SUCCEEDED),
            step(18, "rest_site"),
        ),
        authoritative_outcome=False,
    )
    assert next_act == (2, "act_boundary")


def _counter(observation: dict[str, Any], name: str) -> float:
    training = observation.get("_training")
    assert isinstance(training, dict)
    return float(training.get(name, 0.0))


def _observation(
    *,
    act: int,
    floor: int,
    combat: bool,
    revivals: int = 0,
    hp_loss: float = 0.0,
    room_model_id: str | None = None,
    encounter_id: str | None = None,
) -> dict[str, Any]:
    return {
        "phase": "combat" if combat else "map",
        "decision_domain": "combat" if combat else "route",
        "player": {
            "id": "player",
            "hp": 50,
            "max_hp": 80,
            "gold": 99,
            "deck": [
                {"id": "CARD.STRIKE", "is_upgraded": False},
                {"id": "CARD.DEFEND", "is_upgraded": False},
            ],
        },
        "combat": {
            "in_progress": combat,
            "encounter_id": encounter_id,
            "enemies": ([{"id": "enemy", "hp": 20, "max_hp": 20}] if combat else []),
        },
        "run": {
            "active": True,
            "act": act,
            "floor": floor,
            "room_type": "combat" if combat else "map",
            "room_model_id": room_model_id,
        },
        "_training": {
            "revivals_used": revivals,
            "player_hp_lost": hp_loss,
        },
    }


class _ScriptedRunBackend:
    def __init__(
        self,
        states: list[dict[str, Any]],
        *,
        terminal_result: str | None,
    ) -> None:
        assert len(states) >= 2
        assert terminal_result in {None, "victory", "defeat"}
        self.states = states
        self.terminal_result = terminal_result
        self._capabilities = BackendCapabilities(
            backend_name="episodic-test",
            session_id="episodic-test-session",
        )
        self._state_version = 0
        self._step = 0
        self._episode = 0
        self.closed = False

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def session_id(self) -> str:
        return self._capabilities.session_id

    @property
    def is_connected(self) -> bool:
        return not self.closed

    def health(self) -> dict[str, Any]:
        return {"ok": True}

    def get_spec(self) -> dict[str, Any]:
        return {"ok": True}

    def get_state(self) -> dict[str, Any]:
        return {"ok": True, "state_version": self._state_version}

    @staticmethod
    def _actions() -> tuple[dict[str, Any], ...]:
        # One action exercises forced-decision metadata without any sampling
        # ambiguity in the assertions below.
        return (
            {
                "action_handle": "advance",
                "kind": "proceed",
                "model_action_kind": "proceed",
            },
        )

    def reset(self, request: ResetRequest) -> EnvironmentResult:
        assert request.expected_state_version == self._state_version
        before = self._state_version
        self._state_version += 1
        self._step = 0
        self._episode += 1
        episode_id = f"scripted-run-{self._episode}"
        return EnvironmentResult(
            episode_id=episode_id,
            step_index=0,
            observation=self.states[0],
            legal_actions=self._actions(),
            transition=EnvironmentTransition(
                episode_id=episode_id,
                step_index=0,
                before_state_version=before,
                after_state_version=self._state_version,
                facts={
                    "combat_result": "none",
                    "run_result": "none",
                    "terminal_reason": None,
                },
            ),
            info={"reward_authority": "external-rl"},
        )

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult:
        raise AssertionError("scripted full run must use reset")

    def step(self, request: StepRequest) -> EnvironmentResult:
        assert request.expected_step_index == self._step
        before_observation = self.states[min(self._step, len(self.states) - 1)]
        before_version = self._state_version
        self._state_version += 1
        self._step += 1
        state_index = min(self._step, len(self.states) - 1)
        after_observation = self.states[state_index]
        terminal = bool(self.terminal_result is not None and self._step >= len(self.states) - 1)
        before_combat = bool(before_observation["combat"]["in_progress"])
        after_combat = bool(after_observation["combat"]["in_progress"])
        combat_result = (
            "victory"
            if before_combat and not after_combat and not (terminal and self.terminal_result == "defeat")
            else "none"
        )
        terminal_reason = f"run_{self.terminal_result}" if terminal else None
        episode_id = f"scripted-run-{self._episode}"
        return EnvironmentResult(
            episode_id=episode_id,
            step_index=self._step,
            observation=after_observation,
            legal_actions=() if terminal else self._actions(),
            transition=EnvironmentTransition(
                episode_id=episode_id,
                step_index=self._step,
                before_state_version=before_version,
                after_state_version=self._state_version,
                facts={
                    "combat_result": combat_result,
                    "run_result": self.terminal_result if terminal else "none",
                    "terminal_reason": terminal_reason,
                },
            ),
            terminated=terminal,
            terminal_reason=terminal_reason,
            info={"reward_authority": "external-rl"},
        )

    def close(self) -> None:
        self.closed = True


class _AuditableReward:
    """Expose exact counters while making shaped reward visibly different."""

    def __init__(self, *, progress_by_step: dict[int, float] | None = None) -> None:
        self.progress_by_step = progress_by_step or {}

    def evaluate(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
        *,
        deadlock: bool = False,
        horizon_exhausted: bool = False,
    ) -> TaskReward:
        before_revivals = int(_counter(before.observation, "revivals_used"))
        after_revivals = int(_counter(after.observation, "revivals_used"))
        before_hp = _counter(before.observation, "player_hp_lost")
        after_hp = _counter(after.observation, "player_hp_lost")
        facts = after.transition.facts if after.transition is not None else {}
        run_result = facts.get("run_result")
        task_terminal = bool(after.terminated or deadlock or horizon_exhausted)
        terminal_reward = (
            1.0 if run_result == "victory" else -1.0 if run_result == "defeat" or deadlock or horizon_exhausted else 0.0
        )
        progress_reward = float(self.progress_by_step.get(after.step_index, 0.0))
        # The +37 shaped component must never enter EpisodeDecisionStep.task_reward.
        shaped_component = 37.0
        return TaskReward(
            reward=terminal_reward + progress_reward + shaped_component,
            discount=0.0 if task_terminal else 1.0,
            terminal_reward=terminal_reward,
            potential_reward=progress_reward,
            task_terminal=task_terminal,
            outcome=(
                "success"
                if run_result == "victory"
                else "failure"
                if run_result == "defeat"
                else "deadlock"
                if deadlock
                else "horizon"
                if horizon_exhausted
                else "ongoing"
            ),
            revival_penalty=20.0,
            pace_penalty=10.0,
            hp_loss_penalty=7.0,
            progress_reward=progress_reward,
            revivals_used_delta=after_revivals - before_revivals,
            player_hp_lost_delta=after_hp - before_hp,
        )


def _run_config(
    *,
    max_episode_steps: int = 32,
    durable_window: int = 32,
    combat_window: int = 128,
    combat_room_windows: dict[str, int] | None = None,
    combat_encounter_windows: dict[str, int] | None = None,
):
    base = _config(total_steps=max_episode_steps)
    return replace(
        base,
        environment=replace(
            base.environment,
            scenario="full-run",
            max_episode_steps=max_episode_steps,
        ),
        curriculum=replace(base.curriculum, reward_objective="run"),
        episodic_learning=replace(base.episodic_learning, enabled=True),
        diagnostics=DiagnosticsConfig(
            deadlock_window=128,
            deadlock_repeat_threshold=64,
            combat_net_progress_window=combat_window,
            combat_net_progress_room_windows=combat_room_windows or {},
            combat_net_progress_encounter_windows=(combat_encounter_windows or {}),
            noncombat_durable_progress_window=durable_window,
            combat_min_net_hp_fraction=0.05,
            journal_policy_topk=5,
        ),
    )


def test_decision_surface_uses_factual_state_identity_and_canonical_combat() -> None:
    assert (
        _episodic_decision_surface(
            {
                "state_type": " Card-Select ",
                "screen": "SELECTION",
                "decision_domain": "build",
            },
            combat_in_progress=False,
        )
        == "card_select"
    )
    assert (
        _episodic_decision_surface(
            {"screen": " SHOP SCREEN ", "decision_domain": "build"},
            combat_in_progress=False,
        )
        == "shop_screen"
    )
    assert (
        _episodic_decision_surface(
            {"run": {"room_type": " Treasure-Room "}},
            combat_in_progress=False,
        )
        == "treasure_room"
    )
    assert (
        _episodic_decision_surface(
            {"state_type": "map", "screen": "MAP"},
            combat_in_progress=True,
        )
        == "combat"
    )
    assert _episodic_decision_surface({}, combat_in_progress=False) == "noncombat"


def test_streamed_full_run_backfills_boundaries_exact_costs_and_primary_reward() -> None:
    states = [
        _observation(act=0, floor=0, combat=False),
        _observation(act=1, floor=1, combat=False),
        _observation(act=1, floor=1, combat=True),
        # A single native-revival transition can advance the exact counter by
        # more than one while combat remains active.  It is not a boundary.
        _observation(act=1, floor=1, combat=True, revivals=2, hp_loss=7.0),
        _observation(act=1, floor=2, combat=False, revivals=2, hp_loss=7.0),
        _observation(act=2, floor=16, combat=False, revivals=2, hp_loss=7.0),
        _observation(act=2, floor=16, combat=True, revivals=2, hp_loss=7.0),
        _observation(act=2, floor=31, combat=False, revivals=3, hp_loss=10.0),
    ]
    backend = _ScriptedRunBackend(states, terminal_result="victory")
    resources = build_training_resources(_run_config(), backend=backend)
    resources.collector.reward_calculator = _AuditableReward(progress_by_step={3: 0.25})
    streamed = []

    def sink(unroll):
        streamed.append(unroll)
        return unroll.policy_version + 1

    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
            policy_version=5,
            unroll_sink=sink,
        )
    finally:
        resources.close()

    assert episode.unrolls == ()
    assert sum(len(unroll.steps) for unroll in streamed) == 7
    completed = episode.completed_episode
    assert completed is not None
    assert completed.completion.authoritative
    assert completed.won is True
    assert len(completed.steps) == 7
    decisions = [step.decision for step in completed.steps]
    assert [step.policy_version for step in decisions] == [5, 5, 6, 6, 7, 7, 8]
    assert all(not step.policy_decision for step in decisions)
    assert [step.decision_surface for step in decisions] == [
        "map",
        "map",
        "combat",
        "combat",
        "map",
        "map",
        "combat",
    ]
    assert episode.metrics.forced_decisions == 7

    assert decisions[0].act == 0
    assert decisions[0].act_boundary is BoundaryOutcome.SUCCEEDED
    assert decisions[2].exact_revivals_delta == 2
    assert decisions[2].combat_id == decisions[3].combat_id == "combat:0"
    assert decisions[2].combat_boundary is BoundaryOutcome.NONE
    assert decisions[3].combat_boundary is BoundaryOutcome.SUCCEEDED
    assert decisions[5].combat_id is None
    assert decisions[6].combat_id == "combat:1"
    assert decisions[6].combat_boundary is BoundaryOutcome.SUCCEEDED
    assert decisions[4].act_boundary is BoundaryOutcome.SUCCEEDED
    assert decisions[6].act_boundary is BoundaryOutcome.SUCCEEDED

    # Only terminal + factual progress is retained for long primary credit.
    assert decisions[2].task_reward == pytest.approx(0.25)
    assert decisions[0].task_reward == 0.0
    assert decisions[-1].task_reward == 1.0
    assert episode.metrics.reward_total > 250.0  # shaped online reward is distinct

    assert episode.metrics.act_revival_counts == (2, 3)
    assert episode.metrics.act_hp_loss_counts == (7.0, 10.0)
    assert not episode.metrics.revival_free_act1_clear
    assert all(step.run.observed and step.run.success for step in completed.steps)
    assert completed.steps[0].run.future_revivals == 3


def test_collector_aggregates_bounded_hash_collision_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        _observation(act=0, floor=0, combat=False),
        _observation(act=1, floor=1, combat=False),
        _observation(act=1, floor=2, combat=False),
    ]
    resources = build_training_resources(
        _run_config(),
        backend=_ScriptedRunBackend(states, terminal_result="victory"),
    )
    original_encode = resources.collector.encoder.encode

    def annotated_encode(*args: Any, **kwargs: Any) -> Any:
        return replace(
            original_encode(*args, **kwargs),
            definition_hash_collisions=3,
            relation_hash_collisions=5,
        )

    monkeypatch.setattr(resources.collector.encoder, "encode", annotated_encode)
    progress: list[Any] = []
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
            progress_sink=progress.append,
        )
    finally:
        resources.close()

    assert episode.metrics.steps == 2
    assert episode.metrics.maximum_definition_hash_collisions_per_decision == 3
    assert episode.metrics.maximum_relation_hash_collisions_per_decision == 5
    assert episode.metrics.definition_hash_collisions_total == 6
    assert episode.metrics.relation_hash_collisions_total == 10
    assert progress
    assert progress[-1].maximum_definition_hash_collisions_per_decision == 3
    assert progress[-1].maximum_relation_hash_collisions_per_decision == 5
    assert progress[-1].definition_hash_collisions_total == 6
    assert progress[-1].relation_hash_collisions_total == 10


def test_authoritative_run_defeat_closes_active_combat_and_act_as_failure() -> None:
    states = [
        _observation(act=1, floor=7, combat=True),
        _observation(act=1, floor=7, combat=True, revivals=4, hp_loss=12.0),
        _observation(act=1, floor=7, combat=False, revivals=4, hp_loss=12.0),
    ]
    resources = build_training_resources(
        _run_config(),
        backend=_ScriptedRunBackend(states, terminal_result="defeat"),
    )
    resources.collector.reward_calculator = _AuditableReward()
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
        )
    finally:
        resources.close()

    completed = episode.completed_episode
    assert completed is not None
    assert completed.completion.authoritative
    assert completed.won is False
    assert completed.steps[0].decision.exact_revivals_delta == 4
    assert completed.steps[-1].decision.combat_boundary is BoundaryOutcome.FAILED
    assert completed.steps[-1].decision.act_boundary is BoundaryOutcome.FAILED
    assert all(step.run.success is False for step in completed.steps)
    assert all(not step.run.efficiency_eligible for step in completed.steps)
    assert episode.metrics.act_revival_counts == ()


def test_revival_free_act1_uses_act_boundary_not_later_run_total() -> None:
    states = [
        _observation(act=1, floor=15, combat=True),
        _observation(act=1, floor=15, combat=False),
        _observation(act=2, floor=16, combat=False),
        _observation(act=2, floor=20, combat=True),
        _observation(act=2, floor=20, combat=True, revivals=2, hp_loss=8.0),
        _observation(act=2, floor=31, combat=False, revivals=2, hp_loss=8.0),
    ]
    resources = build_training_resources(
        _run_config(),
        backend=_ScriptedRunBackend(states, terminal_result="victory"),
    )
    resources.collector.reward_calculator = _AuditableReward()
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
        )
    finally:
        resources.close()

    assert episode.metrics.revivals_used == 2
    assert episode.metrics.act_revival_counts == (0, 2)
    assert episode.metrics.revival_free_act1_clear
    assert not episode.metrics.revival_free_run_win


def test_collection_budget_is_censored_and_does_not_fabricate_targets() -> None:
    states = [
        _observation(act=1, floor=3, combat=False),
        _observation(act=1, floor=3, combat=False, revivals=2, hp_loss=5.0),
        _observation(act=1, floor=3, combat=False, revivals=2, hp_loss=5.0),
    ]
    resources = build_training_resources(
        _run_config(max_episode_steps=10),
        backend=_ScriptedRunBackend(states, terminal_result=None),
    )
    resources.collector.reward_calculator = _AuditableReward()
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
            maximum_steps=1,
        )
    finally:
        resources.close()

    completed = episode.completed_episode
    assert completed is not None
    assert not completed.completion.authoritative
    assert completed.won is None
    assert episode.metrics.terminal_reason == "collection_budget"
    only = completed.steps[0]
    assert only.decision.act_boundary is BoundaryOutcome.CENSORED
    assert not only.run.observed
    assert not only.act.observed
    assert not only.combat.observed


def test_durable_deck_fingerprint_is_multiplicity_aware_and_bounded() -> None:
    card = {"id": "CARD.STRIKE_IRONCLAD", "upgrade_level": 0}
    expanded = {"player": {"deck_cards": [dict(card) for _ in range(7)]}}
    compressed = {"player": {"deck_cards": [{**card, "quantity": 7}]}}
    changed = {"player": {"deck_cards": [{**card, "quantity": 8}]}}

    _, expanded_resources = _noncombat_durable_projections(expanded)
    _, compressed_resources = _noncombat_durable_projections(compressed)
    _, changed_resources = _noncombat_durable_projections(changed)

    assert expanded_resources == compressed_resources
    assert changed_resources != compressed_resources
    deck = compressed_resources["player"]["deck"]
    assert len(deck) == 1
    assert deck[0][-1] == 7


def test_noncombat_progress_projection_ignores_vitality_and_revival_churn() -> None:
    before = _observation(
        act=1,
        floor=9,
        combat=False,
        revivals=0,
        hp_loss=0.0,
        room_model_id="EVENT.VITALITY",
    )
    after = _observation(
        act=1,
        floor=9,
        combat=False,
        revivals=999,
        hp_loss=1_000_000.0,
        room_model_id="EVENT.VITALITY",
    )
    before_player = before["player"]
    after_player = after["player"]
    assert isinstance(before_player, dict)
    assert isinstance(after_player, dict)
    before_player.update({"hp": 1, "max_hp": 80})
    after_player.update({"hp": 500, "max_hp": 2_000})
    before_run = before["run"]
    after_run = after["run"]
    assert isinstance(before_run, dict)
    assert isinstance(after_run, dict)
    # Native death/revival can transiently toggle run-lifecycle flags without
    # leaving the current Act/floor/room/event.  Terminality is handled by the
    # accepted transition, not by treating these volatile flags as progress.
    before_run.update({"active": True, "game_over": False})
    after_run.update({"active": False, "game_over": True})

    before_locus, before_resources = _noncombat_durable_projections(before)
    after_locus, after_resources = _noncombat_durable_projections(after)
    assert before_locus == after_locus
    assert before_resources == after_resources

    after_player["gold"] = int(after_player.get("gold", 0)) + 1
    _, gold_resources = _noncombat_durable_projections(after)
    assert gold_resources != before_resources


def test_combat_stall_is_observed_policy_failure_with_full_run_credit() -> None:
    stable = _observation(act=1, floor=17, combat=True, revivals=12, hp_loss=40.0)
    resources = build_training_resources(
        _run_config(max_episode_steps=10, combat_window=2),
        backend=_ScriptedRunBackend([stable, stable, stable], terminal_result=None),
    )
    resources.collector.reward_calculator = _AuditableReward()
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
        )
    finally:
        resources.close()

    completed = episode.completed_episode
    assert completed is not None
    assert episode.metrics.combat_progress_stalled
    assert episode.metrics.combat_policy_failed
    assert episode.metrics.terminal_reason == "combat_progress_stall"
    assert completed.completion.authoritative
    assert completed.won is False
    assert completed.completion.terminal_reason == "combat_progress_stall"
    assert completed.steps[-1].decision.combat_boundary is BoundaryOutcome.FAILED
    assert completed.steps[-1].decision.act_boundary is BoundaryOutcome.FAILED
    assert all(step.combat.observed and step.combat.success is False for step in completed.steps)
    assert all(step.act.observed and step.act.success is False for step in completed.steps)
    assert all(step.run.observed and step.run.success is False for step in completed.steps)
    assert all(not step.run.efficiency_eligible for step in completed.steps)


@pytest.mark.parametrize(
    (
        "room_model_id",
        "encounter_id",
        "room_windows",
        "encounter_windows",
        "source",
        "match_id",
        "window",
    ),
    (
        (
            "test_subject_boss",
            None,
            {"TEST_SUBJECT_BOSS": 2},
            {},
            "room_model_id",
            "TEST_SUBJECT_BOSS",
            2,
        ),
        (
            None,
            "encounter.test_subject",
            {},
            {"ENCOUNTER.TEST_SUBJECT": 3},
            "encounter_id",
            "ENCOUNTER.TEST_SUBJECT",
            3,
        ),
    ),
)
def test_combat_stall_uses_scoped_room_or_encounter_window_and_records_source(
    room_model_id: str | None,
    encounter_id: str | None,
    room_windows: dict[str, int],
    encounter_windows: dict[str, int],
    source: str,
    match_id: str,
    window: int,
) -> None:
    stable = _observation(
        act=3,
        floor=46,
        combat=True,
        revivals=12,
        hp_loss=40.0,
        room_model_id=room_model_id,
        encounter_id=encounter_id,
    )
    resources = build_training_resources(
        _run_config(
            max_episode_steps=10,
            combat_window=8,
            combat_room_windows=room_windows,
            combat_encounter_windows=encounter_windows,
        ),
        backend=_ScriptedRunBackend([stable, stable], terminal_result=None),
    )
    resources.collector.reward_calculator = _AuditableReward()
    progress = []
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
            progress_sink=progress.append,
        )
    finally:
        resources.close()

    assert episode.metrics.combat_progress_stalled
    assert episode.metrics.steps == window
    evidence = episode.metrics.stall_evidence
    assert evidence is not None
    assert evidence["kind"] == "combat_no_net_progress"
    assert evidence["window"] == window
    assert evidence["default_window"] == 8
    assert evidence["window_source"] == source
    assert evidence["window_match_id"] == match_id
    assert evidence["room_model_id"] == (room_model_id.strip().upper() if room_model_id is not None else None)
    assert evidence["encounter_id"] == (encounter_id.strip().upper() if encounter_id is not None else None)
    assert progress
    assert progress[-1].combat_net_progress_window == window
    assert progress[-1].combat_progress_window_source == source
    assert progress[-1].combat_progress_window_match_id == match_id


def test_collection_budget_censors_a_simultaneous_combat_stall() -> None:
    stable = _observation(act=1, floor=17, combat=True, revivals=12, hp_loss=40.0)
    resources = build_training_resources(
        _run_config(max_episode_steps=2, combat_window=2),
        backend=_ScriptedRunBackend([stable, stable, stable], terminal_result=None),
    )
    resources.collector.reward_calculator = _AuditableReward()
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
        )
    finally:
        resources.close()

    completed = episode.completed_episode
    assert completed is not None
    assert episode.metrics.truncated
    assert not episode.metrics.combat_progress_stalled
    assert not episode.metrics.combat_policy_failed
    assert episode.metrics.terminal_reason == "collection_budget"
    assert not completed.completion.authoritative
    assert completed.won is None
    assert completed.steps[-1].decision.combat_boundary is BoundaryOutcome.CENSORED
    assert all(not step.run.observed for step in completed.steps)


def test_noncombat_stall_is_censored_not_a_run_loss() -> None:
    stable = _observation(act=1, floor=3, combat=False)
    resources = build_training_resources(
        _run_config(max_episode_steps=10, durable_window=2),
        backend=_ScriptedRunBackend([stable, stable], terminal_result=None),
    )
    resources.collector.reward_calculator = _AuditableReward()
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
        )
    finally:
        resources.close()

    completed = episode.completed_episode
    assert completed is not None
    assert episode.metrics.noncombat_progress_stalled
    assert episode.metrics.terminal_reason == "noncombat_progress_stall"
    assert not completed.completion.authoritative
    assert all(not step.run.observed for step in completed.steps)
    assert completed.steps[-1].decision.act_boundary is BoundaryOutcome.CENSORED


def test_evaluation_does_not_emit_training_episode_replay() -> None:
    states = [
        _observation(act=1, floor=1, combat=False),
        _observation(act=1, floor=2, combat=False),
    ]
    resources = build_training_resources(
        _run_config(),
        backend=_ScriptedRunBackend(states, terminal_result="victory"),
    )
    resources.collector.reward_calculator = _AuditableReward()
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=False,
            evaluation_seed=1,
        )
    finally:
        resources.close()

    assert episode.completed_episode is None
    assert episode.metrics.run_won


def test_recorded_collection_rejects_held_out_evaluation_seed() -> None:
    states = [
        _observation(act=1, floor=1, combat=False),
        _observation(act=2, floor=16, combat=False),
    ]
    resources = build_training_resources(
        _run_config(),
        backend=_ScriptedRunBackend(states, terminal_result="victory"),
    )
    try:
        with pytest.raises(
            ValueError,
            match="recorded training collection cannot use a held-out evaluation seed",
        ):
            resources.collector.collect_episode(
                deterministic=True,
                record=True,
                evaluation_seed=1,
            )
    finally:
        resources.close()


def test_disabled_episodic_learning_does_not_retain_complete_run() -> None:
    states = [
        _observation(act=1, floor=1, combat=False),
        _observation(act=1, floor=2, combat=False),
    ]
    enabled = _run_config()
    config = replace(
        enabled,
        episodic_learning=replace(enabled.episodic_learning, enabled=False),
    )
    resources = build_training_resources(
        config,
        backend=_ScriptedRunBackend(states, terminal_result="victory"),
    )
    resources.collector.reward_calculator = _AuditableReward()
    try:
        episode = resources.collector.collect_episode(
            deterministic=True,
            record=True,
        )
    finally:
        resources.close()

    assert not resources.collector.episodic_learning_enabled
    assert episode.completed_episode is None
    assert episode.metrics.run_won


def _evaluation_metric(
    *,
    episode_id: str,
    run_won: bool,
    revivals_used: int,
    hp_lost: float,
    act_revivals: tuple[int, ...],
    act_hp_loss: tuple[float, ...],
    maximum_definition_hash_collisions: int = 0,
    maximum_relation_hash_collisions: int = 0,
    definition_hash_collisions_total: int = 0,
    relation_hash_collisions_total: int = 0,
) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode_id=episode_id,
        reset_seed=1,
        steps=100,
        reward_total=1.0 if run_won else -1.0,
        terminal_reason="run_victory" if run_won else "run_defeat",
        truncated=False,
        run_won=run_won,
        combat_won=False,
        act1_cleared=bool(act_revivals),
        max_act=3 if run_won else 2,
        max_floor=46 if run_won else 20,
        policy_decisions=90,
        forced_decisions=10,
        maximum_observed_candidates=16,
        deadlocked=False,
        combat_progress_stalled=False,
        maximum_combat_no_net_progress_steps=0,
        noncombat_progress_stalled=False,
        maximum_noncombat_no_durable_progress_steps=0,
        revivals_used=revivals_used,
        revival_free_combat_win=False,
        revival_free_act1_clear=bool(act_revivals and act_revivals[0] == 0),
        revival_free_run_win=bool(run_won and revivals_used == 0),
        player_hp_lost=hp_lost,
        stall_evidence=None,
        act_revival_counts=act_revivals,
        act_hp_loss_counts=act_hp_loss,
        maximum_definition_hash_collisions_per_decision=(
            maximum_definition_hash_collisions
        ),
        maximum_relation_hash_collisions_per_decision=(
            maximum_relation_hash_collisions
        ),
        definition_hash_collisions_total=definition_hash_collisions_total,
        relation_hash_collisions_total=relation_hash_collisions_total,
    )


def test_evaluation_keeps_completion_primary_and_efficiency_conditional() -> None:
    summary = summarize_evaluation(
        [
            _evaluation_metric(
                episode_id="win",
                run_won=True,
                revivals_used=1,
                hp_lost=20.0,
                act_revivals=(0, 1, 1),
                act_hp_loss=(5.0, 12.0, 20.0),
                maximum_definition_hash_collisions=2,
                maximum_relation_hash_collisions=3,
                definition_hash_collisions_total=7,
                relation_hash_collisions_total=11,
            ),
            _evaluation_metric(
                episode_id="loss",
                run_won=False,
                revivals_used=0,
                hp_lost=30.0,
                act_revivals=(0,),
                act_hp_loss=(10.0,),
                maximum_definition_hash_collisions=5,
                maximum_relation_hash_collisions=4,
                definition_hash_collisions_total=13,
                relation_hash_collisions_total=17,
            ),
        ],
        objective="run",
    )

    assert summary["evaluation_objective"] == "run"
    assert summary["combat_win_rate_applicable"] is False
    assert summary["combat_win_rate"] is None
    assert summary["revival_free_combat_win_rate"] is None
    assert summary["run_win_rate"] == 0.5
    assert summary["maximum_definition_hash_collisions_per_decision"] == 5
    assert summary["maximum_relation_hash_collisions_per_decision"] == 4
    assert summary["definition_hash_collisions_total"] == 20
    assert summary["relation_hash_collisions_total"] == 28
    assert summary["run_win_at_most_one_revival_count"] == 1
    assert summary["run_win_at_most_one_revival_rate"] == 0.5
    assert summary["successful_run_count"] == 1
    assert summary["successful_run_mean_revivals"] == 1.0
    assert summary["successful_run_mean_hp_lost"] == 20.0
    assert summary["act1_clear_at_most_one_revival_rate"] == 1.0
    assert summary["act1_boundary_count"] == 2
    assert summary["act1_boundary_mean_revivals"] == 0.0
    assert summary["act1_boundary_mean_hp_lost"] == 7.5


def test_combat_evaluation_reports_terminal_combat_success_as_applicable() -> None:
    metric = replace(
        _evaluation_metric(
            episode_id="combat-win",
            run_won=False,
            revivals_used=0,
            hp_lost=0.0,
            act_revivals=(),
            act_hp_loss=(),
        ),
        terminal_reason="combat_victory",
        combat_won=True,
        revival_free_combat_win=True,
    )

    summary = summarize_evaluation([metric], objective="combat")

    assert summary["evaluation_objective"] == "combat"
    assert summary["combat_win_rate_applicable"] is True
    assert summary["combat_win_rate"] == 1.0
    assert summary["revival_free_combat_win_rate"] == 1.0


def test_one_episode_runtime_learns_after_storing_complete_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _run_config(max_episode_steps=4)
    config = replace(
        base,
        episodic_learning=replace(
            base.episodic_learning,
            enabled=True,
            replay_capacity_episodes=4,
            replay_capacity_bytes=20_000_000,
            per_episode_capacity_bytes=10_000_000,
            max_segments_per_episode=2,
            sample_sequences=1,
            burn_in_steps=1,
            learn_steps=2,
        ),
        runtime=replace(
            base.runtime,
            total_environment_steps=2,
            log_dir="episodic-e2e/logs",
            checkpoint_dir="episodic-e2e/checkpoints",
            checkpoint_interval_steps=100,
            evaluation_steps=(),
            evaluation_episodes=0,
        ),
    )
    backend = _ScriptedRunBackend(
        [
            _observation(act=1, floor=1, combat=False),
            _observation(act=1, floor=2, combat=False),
            _observation(act=2, floor=16, combat=False),
        ],
        terminal_result="victory",
    )

    state = run_training(config, backend=backend)

    assert state.environment_steps == 2
    assert state.episodes == 1
    assert state.learner_updates == 1
    assert state.consumed_unrolls == 1
    metadata_paths = tuple(tmp_path.rglob("metadata.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    replay_spec = metadata["episodic_replay_spec"]
    assert replay_spec["put_count"] == 1
    assert replay_spec["size"] == 1
    assert replay_spec["sample_count"] >= 1
    assert (metadata_paths[0].parent / "episodic_replay.pkl").is_file()
    metrics_path = next(tmp_path.rglob("metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    learner_updates = [item for item in events if item["event"] == "learner_update"]
    assert len(learner_updates) == 1
    assert learner_updates[0]["unrolls"] == 1
    assert learner_updates[0]["batch_environment_steps"] == 2
    assert learner_updates[0]["episodic_sequences"] == 1
    assert learner_updates[0]["episodic_task_value_labels"] > 0
