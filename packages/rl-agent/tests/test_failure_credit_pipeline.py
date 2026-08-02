from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.semantics import (
    DecisionSemanticsKernel,
    ProgressKind,
)
from sts2_rl.training.failure_credit import (
    DirectPolicyTarget,
    EvidenceStratum,
    FailureCreditEpisodePipeline,
    FailureCreditPipelineConfig,
    FailureOutcome,
)


def _event(
    page: str,
    *,
    hp: int = 60,
    revivals: int = 0,
    selected_count: int | None = None,
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "phase": "event",
        "decision_domain": "event",
        "run": {"act": 1, "floor": 8},
        "room": {
            "room_type": "event",
            "room_model_id": "ROOM_FULL_OF_CHEESE",
        },
        "event": {
            "event_id": "LINGER9",
            "page_id": page,
            "is_finished": False,
        },
        "player": {"hp": hp, "max_hp": 80},
        "_training": {
            "revivals_used": revivals,
            "player_hp_lost": max(0, 60 - hp),
        },
    }
    if selected_count is not None:
        observation["phase"] = "card_selection"
        observation["card_selection"] = {
            "prompt_id": "ROOM_FULL_OF_CHEESE.MULTI",
            "min_select": 0,
            "max_select": 2,
            "selected_count": selected_count,
            "remaining_select": 2 - selected_count,
            "selected_cards": ([] if selected_count == 0 else [{"card_id": "CARD.STRIKE", "upgrade_level": 0}]),
        }
    return observation


def _event_actions() -> tuple[dict[str, Any], ...]:
    return (
        {
            "action_handle": "loop",
            "model_action_kind": "event_option",
            "kind": "event_option",
            "option_id": "LOOP",
        },
        {
            "action_handle": "exit",
            "model_action_kind": "event_option",
            "kind": "event_option",
            "option_id": "EXIT",
        },
    )


def _forced_proceed() -> tuple[dict[str, Any], ...]:
    return (
        {
            "action_handle": "proceed",
            "model_action_kind": "proceed",
            "kind": "proceed",
        },
    )


def _selection_actions(operation: str) -> tuple[dict[str, Any], ...]:
    inverse = "deselect" if operation == "select" else "select"
    return (
        {
            "action_handle": f"{operation}:strike",
            "model_action_kind": "card_selection",
            "model_action_variant": operation,
            "selection_operation": operation,
            "kind": f"{operation}_card",
            "card": {
                "card_id": "CARD.STRIKE",
                "card_instance_id": "strike-instance",
                "upgrade_level": 0,
            },
        },
        {
            "action_handle": f"{inverse}:defend",
            "model_action_kind": "card_selection",
            "model_action_variant": inverse,
            "selection_operation": inverse,
            "kind": f"{inverse}_card",
            "card": {
                "card_id": "CARD.DEFEND",
                "card_instance_id": "defend-instance",
                "upgrade_level": 0,
            },
        },
    )


def _pipeline(
    *,
    detector_window_steps: int = 32,
    context_burn_in_steps: int = 1,
    learning_tail_steps: int = 8,
    maximum_completion_controls: int = 32,
    maximum_completion_bytes: int = 134_217_728,
) -> tuple[FailureCreditEpisodePipeline, GroundedObservationEncoder]:
    kernel = DecisionSemanticsKernel()
    provenance = FailureCreditEpisodePipeline.build_provenance(
        run_id="run-shadow-v4",
        game_version="test-game",
        environment_schema_version="test-environment",
        policy_version=11,
        kernel=kernel,
    )
    return (
        FailureCreditEpisodePipeline(
            episode_id="seed-2:episode-v4",
            provenance=provenance,
            kernel=kernel,
            config=FailureCreditPipelineConfig(
                detector_window_steps=detector_window_steps,
                context_burn_in_steps=context_burn_in_steps,
                learning_tail_steps=learning_tail_steps,
                maximum_completion_controls=maximum_completion_controls,
                maximum_completion_bytes=maximum_completion_bytes,
            ),
        ),
        GroundedObservationEncoder(),
    )


def _observe(
    pipeline: FailureCreditEpisodePipeline,
    encoder: GroundedObservationEncoder,
    *,
    step: int,
    before: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    selected: int,
    after: Mapping[str, Any],
    after_actions: Sequence[Mapping[str, Any]],
    terminal: bool = False,
) -> ProgressKind:
    snapshot = encoder.encode(before, actions, device="cpu").snapshot
    receipt = pipeline.observe_transition(
        episode_step=step,
        before_observation=before,
        before_legal_actions=actions,
        after_observation=after,
        after_legal_actions=after_actions,
        snapshot=snapshot,
        action_index=selected,
        behavior_log_probability=(0.0 if int(np.count_nonzero(snapshot.action_mask)) == 1 else float(np.log(0.5))),
        policy_version=11,
        pre_recurrent_state=np.full(8, step, dtype=np.float32),
        terminal=terminal,
    )
    return receipt.kind


def test_event_two_state_loop_captures_detector_time_direct_witness_and_forced_suffix() -> None:
    pipeline, encoder = _pipeline()
    linger = _event("LINGER9", hp=60, revivals=0)
    warning = _event("DEATH_WARNING", hp=0, revivals=1)

    _observe(
        pipeline,
        encoder,
        step=0,
        before=linger,
        actions=_event_actions(),
        selected=0,
        after=warning,
        after_actions=_forced_proceed(),
    )
    _observe(
        pipeline,
        encoder,
        step=1,
        before=warning,
        actions=_forced_proceed(),
        selected=0,
        after=linger,
        after_actions=_event_actions(),
    )
    _observe(
        pipeline,
        encoder,
        step=2,
        before=linger,
        actions=_event_actions(),
        selected=0,
        after=warning,
        after_actions=_forced_proceed(),
    )
    _observe(
        pipeline,
        encoder,
        step=3,
        before=warning,
        actions=_forced_proceed(),
        selected=0,
        after=linger,
        after_actions=_event_actions(),
    )

    result = pipeline.finalize(
        failure_kind="noncombat_event_action_cycle",
        local_failure=True,
        terminal_succeeded=False,
    )
    failure = result.records[-1]

    assert failure.incident.outcome is FailureOutcome.DEADLOCK_CYCLE
    assert EvidenceStratum.DIRECT_WITNESS in failure.plan.strata
    assert EvidenceStratum.RISK_SEQUENCE in failure.plan.strata
    assert len(failure.plan.direct_policy_targets) == 1
    direct = failure.plan.direct_policy_targets[0]
    assert direct.target is DirectPolicyTarget.AVOID
    # The second LOOP policy choice owns actor credit.  Its forced PROCEED
    # suffix remains value/Q context but is never blamed directly.
    assert failure.plan.context.steps[direct.step_index].episode_step == 2
    forced_step = next(index for index, step in enumerate(failure.plan.context.steps) if step.episode_step == 3)
    assert failure.plan.context.steps[forced_step].forced
    assert forced_step not in {target.step_index for target in failure.plan.direct_policy_targets}
    witness = failure.incident.witnesses[0]
    assert witness.supporting_episode_steps == (0, 1, 2, 3)
    assert witness.loop_edges[0].supporting_episode_steps == (0, 2)
    assert result.metrics.detected_direct_cycles == 1
    assert result.metrics.detected_multi_edge_cycles == 0


def test_detector_confirmed_cycle_is_drainable_before_episode_boundary_once() -> None:
    pipeline, encoder = _pipeline()
    linger = _event("LINGER9", hp=60, revivals=0)
    warning = _event("DEATH_WARNING", hp=0, revivals=1)

    for step, (before, actions, selected, after, after_actions) in enumerate(
        (
            (linger, _event_actions(), 0, warning, _forced_proceed()),
            (warning, _forced_proceed(), 0, linger, _event_actions()),
            (linger, _event_actions(), 0, warning, _forced_proceed()),
            (warning, _forced_proceed(), 0, linger, _event_actions()),
        )
    ):
        _observe(
            pipeline,
            encoder,
            step=step,
            before=before,
            actions=actions,
            selected=selected,
            after=after,
            after_actions=after_actions,
        )

    streamed = pipeline.drain_ready_records()
    assert len(streamed) == 1
    assert streamed[0].incident.outcome is FailureOutcome.DEADLOCK_CYCLE
    assert EvidenceStratum.DIRECT_WITNESS in streamed[0].plan.strata
    assert pipeline.drain_ready_records() == ()

    # The terminal detector may report the same still-active cycle, but the
    # already-streamed local incident must not be duplicated at the boundary.
    result = pipeline.finalize(
        failure_kind="noncombat_event_action_cycle",
        local_failure=True,
        terminal_succeeded=False,
    )
    assert not any(record.incident.outcome is FailureOutcome.DEADLOCK_CYCLE for record in result.records)
    assert result.metrics.streamed_records == 1
    assert result.metrics.streamed_actor_actionable_records == 1
    assert result.metrics.detected_direct_cycles == 1
    assert result.metrics.detected_multi_edge_cycles == 0


def test_multiselect_select_deselect_cycle_is_multi_edge_not_last_action_blame() -> None:
    pipeline, encoder = _pipeline()
    empty = _event("SELECT", selected_count=0)
    selected = _event("SELECT", selected_count=1)

    for step, (before, actions, after, after_actions) in enumerate(
        (
            (
                empty,
                _selection_actions("select"),
                selected,
                _selection_actions("deselect"),
            ),
            (
                selected,
                _selection_actions("deselect"),
                empty,
                _selection_actions("select"),
            ),
            (
                empty,
                _selection_actions("select"),
                selected,
                _selection_actions("deselect"),
            ),
            (
                selected,
                _selection_actions("deselect"),
                empty,
                _selection_actions("select"),
            ),
        )
    ):
        _observe(
            pipeline,
            encoder,
            step=step,
            before=before,
            actions=actions,
            selected=0,
            after=after,
            after_actions=after_actions,
        )

    result = pipeline.finalize(
        failure_kind="selection_action_cycle",
        local_failure=True,
        terminal_succeeded=False,
    )
    failure = result.records[-1]

    assert failure.incident.outcome is FailureOutcome.DEADLOCK_CYCLE
    assert EvidenceStratum.MULTI_EDGE_CYCLE in failure.plan.strata
    assert not failure.plan.direct_policy_targets
    assert len(failure.plan.cycle_policy_targets) == 1
    attributed = tuple(
        failure.plan.context.steps[index].episode_step for index in failure.plan.cycle_policy_targets[0].step_indices
    )
    assert attributed == (2, 3)
    assert len(failure.incident.witnesses[0].loop_edges) == 2
    assert result.metrics.detected_direct_cycles == 0
    assert result.metrics.detected_multi_edge_cycles == 1


def test_unique_unresolved_stall_has_risk_value_credit_but_no_avoid_target() -> None:
    pipeline, encoder = _pipeline()
    actions = _event_actions()
    for step in range(4):
        before = _event(f"UNIQUE-{step}")
        after = _event(f"UNIQUE-{step + 1}")
        _observe(
            pipeline,
            encoder,
            step=step,
            before=before,
            actions=actions,
            selected=step % 2,
            after=after,
            after_actions=actions,
        )

    result = pipeline.finalize(
        failure_kind="noncombat_no_durable_progress",
        local_failure=True,
        terminal_succeeded=False,
    )
    failure = result.records[-1]

    assert failure.incident.outcome is FailureOutcome.DEADLOCK_STALL
    assert EvidenceStratum.RISK_SEQUENCE in failure.plan.strata
    assert EvidenceStratum.UNRESOLVED_STALL in failure.plan.strata
    assert failure.plan.liveness_value_targets
    assert failure.plan.liveness_q_targets
    assert not failure.plan.direct_policy_targets
    assert not failure.plan.cycle_policy_targets
    assert failure.plan.direct_actor_label_count == 0
    assert failure.plan.risk_actor_candidate_count == 4
    assert failure.plan.actor_label_count == 4


def test_flow_completion_is_zero_liveness_control_without_generic_prefer() -> None:
    pipeline, encoder = _pipeline()
    event = _event("CHOICE")
    map_observation = {
        "phase": "map",
        "decision_domain": "map",
        "run": {"act": 1, "floor": 9},
        "map": {"current_coordinate": {"x": 1, "y": 9}},
        "player": {"hp": 60, "max_hp": 80},
    }
    map_actions = (
        {
            "action_handle": "map-next",
            "model_action_kind": "map_node",
            "kind": "map_node",
            "coordinate": {"x": 2, "y": 10},
        },
    )
    kind = _observe(
        pipeline,
        encoder,
        step=0,
        before=event,
        actions=_event_actions(),
        selected=1,
        after=map_observation,
        after_actions=map_actions,
    )

    result = pipeline.finalize(
        failure_kind=None,
        local_failure=False,
        terminal_succeeded=True,
    )
    completion = result.records[0]

    assert kind is ProgressKind.FLOW_ADVANCE
    assert completion.incident.outcome is FailureOutcome.COMPLETED
    assert completion.plan.strata == (EvidenceStratum.COMPLETION_CONTROL,)
    assert completion.plan.liveness_value_targets
    assert all(target.target == 0.0 for target in completion.plan.liveness_value_targets)
    assert not completion.plan.direct_policy_targets
    assert completion.plan.actor_label_count == 0


def test_combat_net_damage_resets_stall_epoch_without_completion_or_prefer() -> None:
    pipeline, encoder = _pipeline()
    before = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"act": 1, "floor": 4},
        "combat": {
            "in_progress": True,
            "encounter_id": "CULTIST",
            "round": 1,
            "enemies": [{"id": "CULTIST", "hp": 50, "max_hp": 50}],
        },
        "player": {"hp": 70, "max_hp": 80, "energy": 3, "hand": []},
    }
    after = {
        **before,
        "combat": {
            **before["combat"],
            "enemies": [{"id": "CULTIST", "hp": 44, "max_hp": 50}],
        },
    }
    actions = (
        {
            "action_handle": "strike",
            "model_action_kind": "play_card",
            "kind": "play_card",
            "card": {"card_id": "CARD.STRIKE", "damage": 6},
            "target": {"id": "CULTIST"},
        },
        {
            "action_handle": "end",
            "model_action_kind": "end_turn",
            "kind": "end_turn",
        },
    )

    kind = _observe(
        pipeline,
        encoder,
        step=0,
        before=before,
        actions=actions,
        selected=0,
        after=after,
        after_actions=actions,
    )
    result = pipeline.finalize(
        failure_kind=None,
        local_failure=False,
        terminal_succeeded=True,
    )

    assert kind is ProgressKind.FLOW_ADVANCE
    assert result.records == ()
    assert result.metrics.completion_controls == 0


def test_shop_durable_commit_has_zero_liveness_control_but_no_prefer() -> None:
    pipeline, encoder = _pipeline()
    actions = (
        {
            "action_handle": "buy-card",
            "model_action_kind": "shop",
            "kind": "shop",
            "operation": "buy",
            "item": {"id": "CARD.OFFER", "price": 50},
        },
        {
            "action_handle": "leave",
            "model_action_kind": "shop",
            "kind": "shop",
            "operation": "leave",
        },
    )
    before = {
        "phase": "shop",
        "decision_domain": "shop",
        "run": {"act": 1, "floor": 7},
        "room": {"room_type": "shop", "room_model_id": "SHOP.1"},
        "shop": {
            "shop_id": "SHOP.1",
            "stock": [{"id": "CARD.OFFER", "price": 50}],
            "is_open": True,
        },
        "player": {"hp": 60, "max_hp": 80, "gold": 100, "deck": []},
    }
    after = {
        **before,
        "shop": {**before["shop"], "stock": [], "is_open": True},
        "player": {
            **before["player"],
            "gold": 50,
            "deck": [{"card_id": "CARD.OFFER"}],
        },
    }

    kind = _observe(
        pipeline,
        encoder,
        step=0,
        before=before,
        actions=actions,
        selected=0,
        after=after,
        after_actions=actions,
    )
    result = pipeline.finalize(
        failure_kind=None,
        local_failure=False,
        terminal_succeeded=True,
    )
    completion = result.records[0]

    assert kind is ProgressKind.DURABLE_COMMIT
    assert completion.incident.failure_kind == "verified_durable_commit"
    assert completion.plan.liveness_q_targets
    assert not completion.plan.direct_policy_targets
    assert completion.plan.actor_label_count == 0


def test_completion_staging_is_count_bounded_and_context_is_critic_minimal() -> None:
    pipeline, encoder = _pipeline(
        context_burn_in_steps=2,
        learning_tail_steps=8,
        maximum_completion_controls=2,
    )

    def map_state(floor: int) -> dict[str, Any]:
        return {
            "phase": "map",
            "decision_domain": "map",
            "run": {"act": 1, "floor": floor},
            "map": {"current_coordinate": {"x": floor % 3, "y": floor}},
            "player": {"hp": 60, "max_hp": 80},
        }

    def map_action(floor: int) -> tuple[dict[str, Any], ...]:
        return (
            {
                "action_handle": f"map-{floor + 1}",
                "model_action_kind": "map",
                "kind": "map",
                "coordinate": {"x": (floor + 1) % 3, "y": floor + 1},
            },
            {
                "action_handle": f"map-alt-{floor + 1}",
                "model_action_kind": "map",
                "kind": "map",
                "coordinate": {"x": (floor + 2) % 3, "y": floor + 1},
            },
        )

    for step in range(5):
        _observe(
            pipeline,
            encoder,
            step=step,
            before=map_state(step),
            actions=map_action(step),
            selected=0,
            after=map_state(step + 1),
            after_actions=map_action(step + 1),
        )

    result = pipeline.finalize(
        failure_kind="boundary",
        local_failure=False,
        terminal_succeeded=False,
    )
    completions = [record for record in result.records if record.incident.outcome is FailureOutcome.COMPLETED]

    assert len(completions) == 2
    assert result.metrics.completion_controls == 2
    assert result.metrics.completion_controls_observed == 5
    assert result.metrics.completion_controls_dropped == 3
    assert result.metrics.completion_storage_nbytes <= 134_217_728
    for completion in completions:
        assert len(completion.plan.context.learn_step_indices) == 1
        assert len(completion.plan.context.steps) <= 3
        assert completion.plan.actor_label_count == 0


def test_hp_death_and_revival_are_cost_only_and_do_not_reset_loop_epoch() -> None:
    pipeline, encoder = _pipeline()
    before = _event("LINGER9", hp=60, revivals=0)
    after = _event("LINGER9", hp=20, revivals=1)

    kind = _observe(
        pipeline,
        encoder,
        step=0,
        before=before,
        actions=_event_actions(),
        selected=0,
        after=after,
        after_actions=_event_actions(),
    )
    _observe(
        pipeline,
        encoder,
        step=1,
        before=after,
        actions=_event_actions(),
        selected=0,
        after=before,
        after_actions=_event_actions(),
    )
    result = pipeline.finalize(
        failure_kind="noncombat_event_action_cycle",
        local_failure=True,
        terminal_succeeded=False,
    )

    assert kind is ProgressKind.COST_ONLY
    assert result.records[-1].incident.outcome is FailureOutcome.DEADLOCK_CYCLE
    assert result.metrics.progress_receipts == ((ProgressKind.COST_ONLY, 2),)


def test_cycle_followed_by_a_long_unique_tail_loses_direct_actor_blame() -> None:
    pipeline, encoder = _pipeline(
        detector_window_steps=32,
        context_burn_in_steps=1,
        learning_tail_steps=8,
    )
    actions = _event_actions()
    cycle_a = _event("CYCLE-A")
    cycle_b = _event("CYCLE-B")
    current = cycle_a
    for step, after in enumerate((cycle_b, cycle_a, cycle_b, cycle_a)):
        _observe(
            pipeline,
            encoder,
            step=step,
            before=current,
            actions=actions,
            selected=0,
            after=after,
            after_actions=actions,
        )
        current = after

    # The factual cycle remains in detector provenance, but nine subsequent
    # unique choices put it outside the eight-step learning tail.  Blaming the
    # old cycle would be a stale causal inference.
    for step in range(4, 13):
        after = _event(f"UNIQUE-TAIL-{step}")
        _observe(
            pipeline,
            encoder,
            step=step,
            before=current,
            actions=actions,
            selected=step % 2,
            after=after,
            after_actions=actions,
        )
        current = after

    failure = pipeline.finalize(
        failure_kind="noncombat_no_durable_progress",
        local_failure=True,
        terminal_succeeded=False,
    ).records[-1]

    assert failure.incident.outcome is FailureOutcome.DEADLOCK_STALL
    assert EvidenceStratum.UNRESOLVED_STALL in failure.plan.strata
    assert EvidenceStratum.MULTI_EDGE_CYCLE not in failure.plan.strata
    assert not failure.plan.direct_policy_targets
    assert not failure.plan.cycle_policy_targets
    assert len(failure.plan.context.steps) == 9
    assert failure.plan.context.burn_in_steps == 1


def test_abandoned_cycle_inside_learning_tail_cannot_blame_later_stall() -> None:
    pipeline, encoder = _pipeline(
        detector_window_steps=32,
        context_burn_in_steps=1,
        learning_tail_steps=16,
    )
    actions = _event_actions()
    repeated = _event("P0")

    # Two exact self-loop macros establish a valid direct witness at step 1.
    for step in range(2):
        _observe(
            pipeline,
            encoder,
            step=step,
            before=repeated,
            actions=actions,
            selected=0,
            after=repeated,
            after_actions=actions,
        )

    # The policy then leaves that loop and traverses unique no-progress pages.
    # The old cycle remains inside both detector and recurrent windows, but it
    # is no longer a suffix-contiguous cause of the final stall.
    current = repeated
    for step in range(2, 10):
        after = _event(f"UNIQUE-SUFFIX-{step}")
        _observe(
            pipeline,
            encoder,
            step=step,
            before=current,
            actions=actions,
            selected=step % 2,
            after=after,
            after_actions=actions,
        )
        current = after

    failure = pipeline.finalize(
        failure_kind="noncombat_no_durable_progress",
        local_failure=True,
        terminal_succeeded=False,
    ).records[-1]

    assert failure.incident.outcome is FailureOutcome.DEADLOCK_STALL
    assert EvidenceStratum.UNRESOLVED_STALL in failure.plan.strata
    assert EvidenceStratum.DIRECT_WITNESS not in failure.plan.strata
    assert EvidenceStratum.MULTI_EDGE_CYCLE not in failure.plan.strata
    assert not failure.plan.direct_policy_targets
    assert not failure.plan.cycle_policy_targets
    assert failure.plan.direct_actor_label_count == 0
    assert failure.plan.risk_actor_candidate_count == 10


def test_detector_window_never_expands_recurrent_context_past_hard_bound() -> None:
    pipeline, encoder = _pipeline(
        detector_window_steps=256,
        context_burn_in_steps=32,
        learning_tail_steps=224,
    )
    actions = _event_actions()

    # Two identical 112-edge periods end exactly at the failure boundary.  The
    # witness is suffix-contiguous, yet its detector window is much larger than
    # the recurrent burn-in requirement.
    pages = tuple(f"BOUND-CYCLE-{index}" for index in range(112))
    current = _event(pages[0])
    for step in range(224):
        after = _event(pages[(step + 1) % len(pages)])
        _observe(
            pipeline,
            encoder,
            step=step,
            before=current,
            actions=actions,
            selected=0,
            after=after,
            after_actions=actions,
        )
        current = after

    failure = pipeline.finalize(
        failure_kind="noncombat_no_durable_progress",
        local_failure=True,
        terminal_succeeded=False,
    ).records[-1]

    assert failure.incident.outcome is FailureOutcome.DEADLOCK_CYCLE
    assert EvidenceStratum.MULTI_EDGE_CYCLE in failure.plan.strata
    assert len(failure.plan.context.steps) <= 256
    assert failure.plan.context.burn_in_steps == 32
    assert failure.plan.context.steps[0].episode_step == 80
    assert failure.plan.context.steps[-1].episode_step == 223
    attributed = tuple(
        failure.plan.context.steps[index].episode_step for index in failure.plan.cycle_policy_targets[0].step_indices
    )
    assert attributed == tuple(range(112, 224))
