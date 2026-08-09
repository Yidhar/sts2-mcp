from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from sts2_rl.contracts import StepRequest
from sts2_rl.training import (
    FailureCreditConfig,
    TransactionExplorationConfig,
    TransactionLearningConfig,
    TransactionLifecycleOutcome,
    TransactionPolicyTarget,
    build_training_resources,
    factual_transaction_policy_targets,
    one_sided_policy_support_loss,
    summarize_evaluation,
    two_sided_policy_support_loss,
)
from sts2_rl.training.collector import (
    _branch_balanced_epsilon_behavior,
    _deck_card_removal_committed,
    _transaction_completion_guided_behavior,
    _transaction_entry_exploration_branch_ids,
)
from sts2_rl.training.failure_credit import (
    DirectPolicyTarget,
    FailureOutcome,
)
from tests.test_v2_training_pipeline import TerminalWithoutObservationFlagsBackend
from tests.test_v33_recovery_semantics import (
    _recovery_config,
    _RestForgeCancelThenSuccessBackend,
    _RestForgeSelectionSuccessBackend,
    _ScriptedChoiceRng,
)


def _exploration_config(*, max_steps: int):
    base = _recovery_config(max_steps=max_steps, repeat_threshold=8)
    return replace(
        base,
        failure_credit=FailureCreditConfig(
            mode="shadow",
            burn_in_steps=1,
            maximum_context_steps=16,
        ),
        transaction_exploration=TransactionExplorationConfig(
            enabled=True,
            operations=("upgrade", "remove"),
            entry_epsilon_floor=0.50,
            completion_guidance_probability=0.95,
        ),
    )


def _lifecycle_learning_config(*, max_steps: int):
    base = _exploration_config(max_steps=max_steps)
    return replace(
        base,
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=64,
            replay_byte_capacity=64 * 1024 * 1024,
            sample_traces=4,
            burn_in_steps=1,
            effect_weight=0.0,
            transaction_q_weight=0.0,
            completion_policy_weight=0.0,
            lifecycle_entry_support_weight=0.25,
            lifecycle_entry_support_probability_floor=0.05,
            lifecycle_smdp_q_weight=0.25,
            pairwise_ranking_weight=0.0,
        ),
    )


class _ShopRemovalSuccessBackend(TerminalWithoutObservationFlagsBackend):
    """Purchase removal, select one deck card, confirm, and prove mutation."""

    def __init__(self) -> None:
        super().__init__(terminal_step=100)
        self.result_terminal_step = 3
        self.step_requests: list[StepRequest] = []

    @staticmethod
    def _card(card_id: str, instance_id: str) -> dict[str, object]:
        return {
            "id": card_id,
            "instance_id": instance_id,
            "source_pile": "Deck",
            "is_upgraded": False,
        }

    def _actions(self) -> tuple[dict[str, object], ...]:
        strike = self._card("CARD.STRIKE", "strike-1")
        defend = self._card("CARD.DEFEND", "defend-1")
        if self._step == 0:
            return (
                {
                    "action_handle": "shop-remove",
                    "action": "shop_purchase",
                    "kind": "shop_purchase",
                    "model_action_kind": "shop",
                    "model_action_variant": "purchase",
                    "item": {
                        "category": "card_removal",
                        "index": 13,
                        "cost": 75,
                        "can_afford": True,
                    },
                },
                {
                    "action_handle": "shop-card",
                    "action": "shop_purchase",
                    "kind": "shop_purchase",
                    "model_action_kind": "shop",
                    "model_action_variant": "purchase",
                    "item": {
                        "category": "card",
                        "index": 2,
                        "cost": 50,
                        "can_afford": True,
                    },
                },
                {
                    "action_handle": "shop-leave",
                    "action": "shop_skip",
                    "kind": "shop_skip",
                    "model_action_kind": "shop",
                    "model_action_variant": "leave",
                },
            )
        if self._step == 1:
            return (
                {
                    "action_handle": "select-strike",
                    "action": "select_card",
                    "kind": "select_card",
                    "model_action_kind": "card_selection",
                    "model_action_variant": "select",
                    "selection_operation": "select",
                    "card": strike,
                },
                {
                    "action_handle": "select-defend",
                    "action": "select_card",
                    "kind": "select_card",
                    "model_action_kind": "card_selection",
                    "model_action_variant": "select",
                    "selection_operation": "select",
                    "card": defend,
                },
                {
                    "action_handle": "cancel-removal",
                    "action": "cancel_selection",
                    "kind": "cancel_selection",
                    "model_action_kind": "card_selection",
                    "model_action_variant": "cancel_prompt",
                    "selection_operation": "cancel_prompt",
                },
            )
        return (
            {
                "action_handle": "deselect-strike",
                "action": "deselect_card",
                "kind": "deselect_card",
                "model_action_kind": "card_selection",
                "model_action_variant": "deselect",
                "selection_operation": "deselect",
                "card": {**strike, "is_selected": True},
            },
            {
                "action_handle": "confirm-removal",
                "action": "confirm_selection",
                "kind": "confirm_selection",
                "model_action_kind": "card_selection",
                "model_action_variant": "confirm",
                "selection_operation": "confirm",
            },
            {
                "action_handle": "cancel-removal",
                "action": "cancel_selection",
                "kind": "cancel_selection",
                "model_action_kind": "card_selection",
                "model_action_variant": "cancel_prompt",
                "selection_operation": "cancel_prompt",
            },
        )

    def _observation(self, *, terminal: bool = False) -> dict[str, object]:
        strike = self._card("CARD.STRIKE", "strike-1")
        defend = self._card("CARD.DEFEND", "defend-1")
        at_shop = self._step == 0 and not terminal
        observation: dict[str, object] = {
            "phase": "map" if terminal else "shop" if at_shop else "selection",
            "decision_domain": "map" if terminal else "resource" if at_shop else "build",
            "state_type": "map" if terminal else "shop" if at_shop else "card_select",
            "screen": "MAP" if terminal else "SHOP" if at_shop else "SELECTION",
            "player": {
                "character": "IRONCLAD",
                "hp": 40,
                "max_hp": 70,
                "gold": 24 if terminal else 99,
                "deck": [defend] if terminal else [strike, defend],
                "relics": [],
                "potions": [],
            },
            "combat": {"in_progress": False, "enemies": []},
            "run": {
                "active": not terminal,
                "act": 1,
                "floor": 4,
                "room_type": "map" if terminal else "shop",
                "room_model_id": "MAP" if terminal else "SHOP",
            },
        }
        if at_shop:
            observation["shop"] = {
                "is_open": True,
                "items": [action["item"] for action in self._actions() if "item" in action],
            }
        elif not terminal:
            selected = self._step >= 2
            observation["card_selection"] = {
                "mode": "SimpleGrid",
                "prompt_id": "card_selection.TO_REMOVE",
                "operation_type": "remove",
                "source_zone": "Deck",
                "destination_zone": "Removed",
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

    def step(self, request: StepRequest):  # type: ignore[no-untyped-def]
        self.step_requests.append(request)
        return super().step(request)


def test_transaction_entry_explorer_removes_shop_candidate_cardinality_bias() -> None:
    actions = (
        {
            "model_action_kind": "shop",
            "item": {"category": "card", "index": 0},
        },
        {
            "model_action_kind": "shop",
            "item": {"category": "card_removal", "index": 1},
        },
        {
            "model_action_kind": "shop",
            "model_action_variant": "leave",
        },
    )
    branch_ids, targeted = _transaction_entry_exploration_branch_ids(
        policy_branch_ids=np.asarray([7, 7, 7], dtype=np.int64),
        semantic_actions=actions,
        enabled_operations=frozenset({"remove"}),
    )
    assert targeted
    assert branch_ids.tolist() == [7, 8, 7]
    behavior = _branch_balanced_epsilon_behavior(
        policy=np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float32),
        valid=np.asarray([True, True, True], dtype=np.bool_),
        policy_branch_ids=branch_ids,
        epsilon=1.0,
    )
    assert behavior[1] == pytest.approx(0.5)
    assert behavior[[0, 2]].sum() == pytest.approx(0.5)


def test_completion_guidance_preserves_support_and_learned_card_ranking() -> None:
    actions = (
        {"model_action_kind": "card_selection", "selection_operation": "select"},
        {"model_action_kind": "card_selection", "selection_operation": "select"},
        {"model_action_kind": "card_selection", "selection_operation": "cancel_prompt"},
    )
    policy = np.asarray([0.2, 0.6, 0.2], dtype=np.float32)
    base = np.asarray([0.3, 0.3, 0.4], dtype=np.float64)
    behavior, forward, fallback = _transaction_completion_guided_behavior(
        policy=policy,
        valid=np.asarray([True, True, True], dtype=np.bool_),
        semantic_actions=actions,
        base_behavior=base,
        operation="remove",
        guidance_probability=0.90,
    )
    assert not fallback
    assert forward == frozenset({0, 1})
    assert behavior[2] > 0.0  # Cancel remains in behavior support for V-trace.
    guided_component = (behavior[:2] - 0.10 * base[:2]) / 0.90
    assert guided_component[1] / guided_component[0] == pytest.approx(3.0)


def test_card_removal_commit_requires_exact_authoritative_deck_decrease() -> None:
    before = (("CARD.DEFEND", 1, 0), ("CARD.STRIKE", 1, 0))

    def observation(*card_ids: str) -> dict[str, object]:
        return {
            "player": {
                "deck": [
                    {
                        "id": card_id,
                        "instance_id": f"{card_id}:{index}",
                        "source_pile": "Deck",
                        "is_upgraded": False,
                    }
                    for index, card_id in enumerate(card_ids)
                ]
            }
        }

    assert _deck_card_removal_committed(before, observation("CARD.DEFEND"))
    assert not _deck_card_removal_committed(
        before,
        observation("CARD.DEFEND", "CARD.STRIKE"),
    )
    assert not _deck_card_removal_committed(
        before,
        observation("CARD.DEFEND", "CARD.NEW"),
    )


def test_guided_forge_completes_and_keeps_verified_positive_credit() -> None:
    config = _exploration_config(max_steps=8)
    resources = build_training_resources(
        config,
        backend=_RestForgeSelectionSuccessBackend(),
    )
    try:
        resources.collector.bind_failure_credit_run_id("v34-guided-forge")
        resources.collector._rng = _ScriptedChoiceRng((0, 0, 1))  # type: ignore[assignment]
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(
            epsilon=0.05,
            deterministic=False,
            record=True,
        )
    finally:
        resources.close()

    assert episode.metrics.run_won
    assert episode.metrics.forge_selection_transactions_completed == 1
    assert episode.metrics.targeted_transaction_entry_exploration_decisions == 1
    assert episode.metrics.transaction_completion_guidance_decisions == 2
    assert episode.metrics.transaction_completion_forward_decisions == 2
    assert episode.metrics.transaction_completion_guidance_fallbacks == 0
    assert any(
        target.target is DirectPolicyTarget.PREFER
        for record in episode.failure_credit_records
        if record.incident.outcome is FailureOutcome.COMPLETED
        for target in record.plan.direct_policy_targets
    )


def test_guided_shop_removal_requires_real_deck_mutation_and_reports_it() -> None:
    config = _exploration_config(max_steps=8)
    backend = _ShopRemovalSuccessBackend()
    resources = build_training_resources(config, backend=backend)
    try:
        resources.collector.bind_failure_credit_run_id("v34-guided-shop-removal")
        resources.collector._rng = _ScriptedChoiceRng((0, 0, 1))  # type: ignore[assignment]
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(
            epsilon=0.05,
            deterministic=False,
            record=True,
        )
    finally:
        resources.close()

    assert [request.action_id for request in backend.step_requests] == [
        "shop-remove",
        "select-strike",
        "confirm-removal",
    ]
    assert episode.metrics.run_won
    assert episode.metrics.shop_card_removal_transactions_started == 1
    assert episode.metrics.shop_card_removal_transactions_closed == 1
    assert episode.metrics.shop_card_removal_transactions_completed == 1
    assert episode.metrics.shop_card_removal_transactions_cancelled == 0
    assert episode.metrics.shop_card_removal_transactions_unresolved == 0
    assert episode.metrics.targeted_transaction_entry_exploration_decisions == 1
    assert episode.metrics.transaction_completion_guidance_decisions == 2
    assert episode.metrics.transaction_completion_forward_decisions == 2

    summary = summarize_evaluation([episode.metrics], objective="run")
    assert summary["shop_card_removal_transactions_started"] == 1
    assert summary["shop_card_removal_transactions_committed"] == 1
    assert summary["shop_card_removal_transaction_completion_rate"] == pytest.approx(1.0)


def test_transaction_exploration_never_changes_deterministic_evaluation() -> None:
    config = _exploration_config(max_steps=8)
    resources = build_training_resources(config, backend=_ShopRemovalSuccessBackend())
    try:
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(
            epsilon=1.0,
            deterministic=True,
            record=False,
            maximum_steps=1,
        )
    finally:
        resources.close()

    assert episode.metrics.targeted_transaction_entry_exploration_decisions == 0
    assert episode.metrics.transaction_completion_guidance_decisions == 0
    assert episode.metrics.transaction_completion_forward_decisions == 0


def test_one_sided_entry_support_recovers_from_softmax_absorption_and_stops_at_floor() -> None:
    saturated_logits = torch.tensor([80.0, -80.0], requires_grad=True)
    saturated_log_probability = torch.log_softmax(saturated_logits, dim=0)[1]
    saturated_loss, saturated_gap = one_sided_policy_support_loss(
        saturated_log_probability,
        probability_floor=0.05,
    )
    saturated_loss.backward()

    assert saturated_loss.detach().item() > 100.0
    assert saturated_gap.detach().item() > 100.0
    assert saturated_logits.grad is not None
    assert saturated_logits.grad.tolist() == pytest.approx([1.0, -1.0])

    supported_logits = torch.tensor([0.0, math.log(0.1 / 0.9)], requires_grad=True)
    supported_log_probability = torch.log_softmax(supported_logits, dim=0)[1]
    supported_loss, supported_gap = one_sided_policy_support_loss(
        supported_log_probability,
        probability_floor=0.05,
    )
    supported_loss.backward()

    assert supported_loss.detach().item() == pytest.approx(0.0)
    assert supported_gap.detach().item() == pytest.approx(0.0)
    assert supported_logits.grad is not None
    assert supported_logits.grad.tolist() == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    ("logits", "selected_index", "expected_gradient"),
    (
        ([80.0, -80.0], 1, [1.0, -1.0]),
        ([80.0, -80.0], 0, [1.0, -1.0]),
    ),
)
def test_two_sided_entry_support_recovers_either_absorbed_branch(
    logits: list[float],
    selected_index: int,
    expected_gradient: list[float],
) -> None:
    values = torch.tensor(logits, requires_grad=True)
    log_probabilities = torch.log_softmax(values, dim=0)
    alternative_index = 1 - selected_index
    loss, gap = two_sided_policy_support_loss(
        log_probabilities[selected_index],
        log_probabilities[alternative_index],
        probability_floor=0.05,
    )
    loss.backward()

    assert loss.detach().item() > 100.0
    assert gap.detach().item() > 100.0
    assert values.grad is not None
    assert values.grad.tolist() == pytest.approx(expected_gradient)

    supported = torch.tensor([0.0, 0.0], requires_grad=True)
    supported_log_probabilities = torch.log_softmax(supported, dim=0)
    supported_loss, supported_gap = two_sided_policy_support_loss(
        supported_log_probabilities[0],
        supported_log_probabilities[1],
        probability_floor=0.05,
    )
    supported_loss.backward()
    assert supported_loss.detach().item() == pytest.approx(0.0)
    assert supported_gap.detach().item() == pytest.approx(0.0)
    assert supported.grad is not None
    assert supported.grad.tolist() == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    ("backend", "choices", "operation"),
    (
        (_RestForgeSelectionSuccessBackend, (0, 0, 1), "upgrade"),
        (_ShopRemovalSuccessBackend, (0, 0, 1), "remove"),
    ),
)
def test_verified_transaction_lifecycle_links_entry_to_terminal_commit(
    backend: type[TerminalWithoutObservationFlagsBackend],
    choices: tuple[int, ...],
    operation: str,
) -> None:
    config = _lifecycle_learning_config(max_steps=8)
    resources = build_training_resources(config, backend=backend())
    try:
        resources.collector.bind_failure_credit_run_id(f"lifecycle-{operation}")
        resources.collector._rng = _ScriptedChoiceRng(choices)  # type: ignore[assignment]
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
            for parameter in resources.collector_model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(
            epsilon=0.05,
            deterministic=False,
            record=True,
        )
        lifecycle_traces = tuple(
            trace for trace in episode.transaction_traces if trace.lifecycle is not None
        )
        assert len(lifecycle_traces) == 1
        trace = lifecycle_traces[0]
        lifecycle = trace.lifecycle
        assert lifecycle is not None
        assert lifecycle.operation == operation
        assert lifecycle.outcome is TransactionLifecycleOutcome.COMMITTED
        assert lifecycle.effect_verified
        assert lifecycle.support_eligible
        assert lifecycle.entry_step_index == trace.burn_in_steps
        assert lifecycle.exit_step_index == len(trace.steps) - 1
        assert lifecycle.post_terminal
        assert lifecycle.post_snapshot is None
        assert lifecycle.option_target_observed
        assert lifecycle.option_discount == pytest.approx(0.0)
        assert lifecycle.option_steps == len(trace.steps)
        assert all(
            target.step_index != lifecycle.entry_step_index
            for target in factual_transaction_policy_targets(trace)
        )

        losses = resources.learner._transaction_losses((trace,))
        assert losses.entry_support_labels == 1
        assert losses.smdp_q_labels == 1
        assert losses.entry_support_satisfied_labels == 1
        assert losses.entry_support_loss.detach().item() == pytest.approx(0.0)
        assert losses.entry_model_probability_mean >= 0.05
        assert losses.smdp_q_loss.detach().item() > 0.0
        if operation == "upgrade":
            assert losses.upgrade_entry_support_labels == 1
            assert losses.remove_entry_support_labels == 0
        else:
            assert losses.upgrade_entry_support_labels == 0
            assert losses.remove_entry_support_labels == 1
        resources.learner.optimizer.zero_grad(set_to_none=True)
        losses.smdp_q_loss.backward()
        q_gradients = tuple(
            parameter.grad
            for name, parameter in resources.model.named_parameters()
            if name.startswith("transaction_q_head.") and parameter.grad is not None
        )
        assert q_gradients
        assert any(bool(torch.count_nonzero(gradient)) for gradient in q_gradients)
    finally:
        resources.close()


def test_cancelled_transaction_lifecycle_never_creates_positive_entry_credit() -> None:
    config = _lifecycle_learning_config(max_steps=12)
    resources = build_training_resources(
        config,
        backend=_RestForgeCancelThenSuccessBackend(),
    )
    try:
        resources.collector.bind_failure_credit_run_id("lifecycle-cancel-then-commit")
        resources.collector._rng = _ScriptedChoiceRng(  # type: ignore[assignment]
            (0, 0, 1, 0, 0, 1)
        )
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
            for parameter in resources.collector_model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(
            epsilon=0.05,
            deterministic=False,
            record=True,
        )
        lifecycle_traces = tuple(
            trace for trace in episode.transaction_traces if trace.lifecycle is not None
        )
        assert len(lifecycle_traces) == 2
        cancelled = next(
            trace
            for trace in lifecycle_traces
            if trace.lifecycle is not None
            and trace.lifecycle.outcome is TransactionLifecycleOutcome.CANCELLED
        )
        committed = next(
            trace
            for trace in lifecycle_traces
            if trace.lifecycle is not None
            and trace.lifecycle.outcome is TransactionLifecycleOutcome.COMMITTED
        )
        assert cancelled.lifecycle is not None
        assert not cancelled.lifecycle.support_eligible
        assert not cancelled.lifecycle.option_target_observed
        assert committed.lifecycle is not None
        assert committed.lifecycle.support_eligible
        losses = resources.learner._transaction_losses(lifecycle_traces)
        assert losses.lifecycle_cancelled == 1
        assert losses.lifecycle_committed == 1
        assert losses.entry_support_labels == 1
        assert losses.smdp_q_labels == 1
    finally:
        resources.close()


def test_forced_singleton_lifecycle_entry_suppresses_support_corridor() -> None:
    """A committed lifecycle whose entry was a forced singleton must not
    produce a two-sided support label (there is no legal alternative to keep
    inside the corridor), must not crash the learner, and must keep its
    factual SMDP entry value labels."""

    config = _lifecycle_learning_config(max_steps=8)
    resources = build_training_resources(
        config,
        backend=_RestForgeSelectionSuccessBackend(),
    )
    try:
        resources.collector.bind_failure_credit_run_id("lifecycle-singleton")
        resources.collector._rng = _ScriptedChoiceRng((0, 0, 1))  # type: ignore[assignment]
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
            for parameter in resources.collector_model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(
            epsilon=0.05,
            deterministic=False,
            record=True,
        )
        trace = next(
            trace
            for trace in episode.transaction_traces
            if trace.lifecycle is not None
        )
        lifecycle = trace.lifecycle
        assert lifecycle is not None
        assert lifecycle.support_eligible

        entry_index = lifecycle.entry_step_index
        entry_step = trace.steps[entry_index]
        singleton_mask = np.zeros_like(entry_step.snapshot.action_mask)
        singleton_mask[entry_step.action_index] = True
        singleton_step = replace(
            entry_step,
            snapshot=replace(entry_step.snapshot, action_mask=singleton_mask),
        )
        singleton_trace = replace(
            trace,
            steps=tuple(
                singleton_step if index == entry_index else step
                for index, step in enumerate(trace.steps)
            ),
        )

        losses = resources.learner._transaction_losses((singleton_trace,))
        assert losses.entry_support_singleton_suppressed_labels == 1
        assert losses.entry_support_labels == 0
        assert losses.upgrade_entry_support_labels == 0
        assert losses.entry_support_loss.detach().item() == pytest.approx(0.0)
        assert losses.smdp_q_labels == 1
        assert losses.lifecycle_committed == 1

        baseline = resources.learner._transaction_losses((trace,))
        assert baseline.entry_support_singleton_suppressed_labels == 0
        assert baseline.entry_support_labels == 1
    finally:
        resources.close()


@pytest.mark.parametrize(
    ("backend", "choices"),
    (
        (_RestForgeSelectionSuccessBackend, (0, 0, 1)),
        (_ShopRemovalSuccessBackend, (0, 0, 1)),
    ),
)
def test_completion_ce_excludes_card_target_steps_in_verified_lifecycles(
    backend: type[TerminalWithoutObservationFlagsBackend],
    choices: tuple[int, ...],
) -> None:
    """Inside a verified upgrade/removal lifecycle the card-target select
    steps receive neither PREFER nor AVOID from the completed-path CE; the
    forward (confirm/proceed) steps keep their PREFER labels."""

    config = _lifecycle_learning_config(max_steps=8)
    resources = build_training_resources(config, backend=backend())
    try:
        resources.collector.bind_failure_credit_run_id("target-ce-exclusion")
        resources.collector._rng = _ScriptedChoiceRng(choices)  # type: ignore[assignment]
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
            for parameter in resources.collector_model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(
            epsilon=0.05,
            deterministic=False,
            record=True,
        )
        trace = next(
            trace
            for trace in episode.transaction_traces
            if trace.lifecycle is not None
        )
        assert trace.lifecycle is not None and trace.lifecycle.support_eligible

        labels = factual_transaction_policy_targets(trace)
        target_steps = {
            index
            for index, step in enumerate(trace.steps)
            if step.selected_count_delta > 0
        }
        assert target_steps, "fixture must contain a card-target select step"
        assert all(label.step_index not in target_steps for label in labels)
        assert any(
            label.target is TransactionPolicyTarget.PREFER for label in labels
        ), "forward completion steps must keep PREFER labels"
    finally:
        resources.close()
