from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from sts2_baseline import RolloutStep, SequenceUnroll
from sts2_rl.checkpoints import ValidatedResumeCheckpoint
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.encoding import (
    EncodedDecisionSnapshot,
    GroundedEncodingConfig,
    GroundedObservationEncoder,
    grounding_encoding_identity,
)
from sts2_rl.encoding.snapshot import sparse_token_table
from sts2_rl.models import GroundedCandidateConfig, RecurrentCandidateModel
from sts2_rl.training import (
    BoundedTransactionReplay,
    TrainingConfig,
    TrainingState,
    TransactionEffect,
    TransactionLearningConfig,
    TransactionStep,
    TransactionTrace,
    backfill_factual_monte_carlo_returns,
    build_training_resources,
    initialize_model_from_checkpoint,
    load_training_checkpoint,
    observed_outcome_pairs,
    save_training_checkpoint,
)
from sts2_rl.training import checkpointing as checkpointing_module
from sts2_rl.training.collector import (
    _classify_transaction_transition,
    _transaction_node_key,
    _transaction_surface_key,
)
from sts2_rl.training.learner import VTraceLearner
from sts2_rl.training.runtime import run_training


class _NoopBackend:
    def close(self) -> None:
        pass


def test_transaction_surface_and_node_are_order_independent_but_membership_sensitive() -> None:
    first_observation = {
        "run": {"act": 1, "floor": 7, "room_type": "combat"},
        "card_selection": {
            "mode": "SimpleGrid",
            "prompt_id": "card.TEST.discard",
            "operation_type": "discard",
            "source_zone": "Hand",
            "min_select": 1,
            "max_select": 2,
            "selected_count": 1,
            "remaining_picks": 1,
            "can_confirm": True,
            "options": [
                {
                    "option_index": 0,
                    "is_selected": True,
                    "preview": {"damage": 9},
                    "card": {
                        "id": "CARD.STRIKE",
                        "instance_id": "strike-1",
                        "pile": "Selected",
                        "cost": 0,
                    },
                },
                {
                    "option_index": 1,
                    "is_selected": False,
                    "preview": {"damage": 100},
                    "card": {
                        "id": "CARD.DEFEND",
                        "instance_id": "defend-1",
                        "pile": "Hand",
                        "cost": 1,
                    },
                },
            ],
        },
    }
    first_actions = (
        {
            "action_handle": "volatile-a",
            "kind": "deselect_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "deselect",
            "card": {
                "id": "CARD.STRIKE",
                "instance_id": "strike-1",
                "pile": "Selected",
                "cost": 0,
            },
        },
        {
            "action_handle": "volatile-b",
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "card": {
                "id": "CARD.DEFEND",
                "instance_id": "defend-1",
                "pile": "Hand",
                "cost": 1,
            },
        },
        {
            "action_handle": "volatile-c",
            "kind": "confirm_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "confirm",
        },
    )
    reordered_observation = {
        **first_observation,
        "card_selection": {
            **first_observation["card_selection"],
            "options": list(reversed(first_observation["card_selection"]["options"])),
        },
    }
    reordered_actions = tuple(reversed(first_actions))

    first_surface = _transaction_surface_key(first_observation, first_actions)
    reordered_surface = _transaction_surface_key(reordered_observation, reordered_actions)
    assert first_surface is not None
    assert reordered_surface == first_surface
    assert _transaction_node_key(first_surface, first_observation, first_actions) == _transaction_node_key(
        reordered_surface,
        reordered_observation,
        reordered_actions,
    )

    changed_membership = {
        **first_observation,
        "card_selection": {
            **first_observation["card_selection"],
            "options": [
                {
                    **first_observation["card_selection"]["options"][0],
                    "is_selected": False,
                    "card": {
                        **first_observation["card_selection"]["options"][0]["card"],
                        "pile": "Hand",
                        "cost": 2,
                    },
                },
                {
                    **first_observation["card_selection"]["options"][1],
                    "is_selected": True,
                    "card": {
                        **first_observation["card_selection"]["options"][1]["card"],
                        "pile": "Selected",
                        "cost": 0,
                    },
                },
            ],
        },
    }
    changed_surface = _transaction_surface_key(changed_membership, reordered_actions)
    assert changed_surface == first_surface
    assert _transaction_node_key(
        changed_surface,
        changed_membership,
        reordered_actions,
    ) != _transaction_node_key(first_surface, first_observation, first_actions)


def test_transaction_node_separates_reward_relevant_world_context() -> None:
    actions = (
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "card": {"id": "CARD.STRIKE", "instance_id": "strike-1"},
        },
    )
    selection = {
        "mode": "SimpleGrid",
        "prompt_id": "test.discard",
        "operation_type": "discard",
        "min_select": 1,
        "max_select": 1,
        "cards": [{"id": "CARD.STRIKE", "instance_id": "strike-1"}],
        "selected_cards": [],
    }
    low_hp = {
        "run": {"act": 1, "floor": 3, "room_type": "combat"},
        "player": {"hp": 10, "deck_cards": [{"id": "CARD.STRIKE"}]},
        "_training": {"revivals_used": 2, "player_hp_lost": 90},
        "card_selection": selection,
    }
    high_hp = {
        **low_hp,
        "player": {"hp": 80, "deck_cards": [{"id": "CARD.BASH"}]},
        "_training": {"revivals_used": 0, "player_hp_lost": 0},
    }

    surface = _transaction_surface_key(low_hp, actions)
    assert surface == _transaction_surface_key(high_hp, actions)
    assert surface is not None
    assert _transaction_node_key(surface, low_hp, actions) != _transaction_node_key(
        surface,
        high_hp,
        actions,
    )


def test_action_only_selection_surface_is_captured_without_observation_dto() -> None:
    observation = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"act": 1, "floor": 3, "room_type": "combat"},
    }
    actions = (
        {
            "action_handle": "select-1",
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "prompt_id": "legacy.discard",
            "operation_type": "discard",
            "source_zone": "Discard",
            "card": {
                "id": "CARD.STRIKE",
                "instance_id": "strike-1",
                "pile": "Discard",
                "selection_membership": "selectable",
            },
        },
        {
            "action_handle": "select-2",
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "prompt_id": "legacy.discard",
            "operation_type": "discard",
            "source_zone": "Discard",
            "card": {
                "id": "CARD.DEFEND",
                "instance_id": "defend-1",
                "pile": "Discard",
                "selection_membership": "selectable",
            },
        },
    )

    surface = _transaction_surface_key(observation, actions)
    assert surface is not None
    assert _transaction_surface_key(observation, tuple(reversed(actions))) == surface


def test_canonical_cards_universe_does_not_duplicate_selected_membership_subset() -> None:
    """The simulator exposes ``cards`` as a full universe plus a selected subset."""

    base = {
        "run": {"act": 1, "floor": 3, "room_type": "combat"},
        "card_selection": {
            "mode": "SimpleGrid",
            "prompt_id": "card.HEADBUTT.selection",
            "operation_type": "select",
            "source_zone": "Discard",
            "min_select": 1,
            "max_select": 2,
            "cards": [
                {
                    "id": "CARD.STRIKE",
                    "instance_id": "strike-1",
                    "source_pile": "Discard",
                },
                {
                    "id": "CARD.DEFEND",
                    "instance_id": "defend-1",
                    "source_pile": "Discard",
                },
            ],
            "selected_cards": [
                {
                    "id": "CARD.STRIKE",
                    "instance_id": "strike-1",
                    "source_pile": "Discard",
                }
            ],
            "selected_count": 1,
            "remaining_picks": 1,
            "can_confirm": True,
        },
    }
    actions = (
        {
            "kind": "deselect_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "deselect",
            "card": {
                "id": "CARD.STRIKE",
                "instance_id": "strike-1",
                "source_pile": "Discard",
            },
        },
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "card": {
                "id": "CARD.DEFEND",
                "instance_id": "defend-1",
                "source_pile": "Discard",
            },
        },
    )
    deselected = {
        **base,
        "card_selection": {
            **base["card_selection"],
            "selected_cards": [],
            "selected_count": 0,
            "remaining_picks": 2,
            "can_confirm": False,
        },
    }
    deselected_actions = (
        {
            **actions[0],
            "kind": "select_card",
            "model_action_variant": "select",
        },
        actions[1],
    )

    surface = _transaction_surface_key(base, actions)
    deselected_surface = _transaction_surface_key(deselected, deselected_actions)
    assert surface is not None
    assert deselected_surface == surface
    assert _transaction_node_key(surface, base, actions) != _transaction_node_key(
        deselected_surface,
        deselected,
        deselected_actions,
    )


def test_transaction_exit_is_not_mislabeled_as_selection_teardown_delta() -> None:
    effect, delta = _classify_transaction_transition(
        current_surface="selection-surface",
        next_surface=None,
        current_node="selected-one",
        next_node="reward-screen",
        seen_nodes={"selected-zero", "selected-one"},
        before_selected_count=1,
        after_selected_count=0,
    )
    assert effect is TransactionEffect.EXIT
    assert delta == 0


def _model_config() -> GroundedCandidateConfig:
    return GroundedCandidateConfig(
        token_feature_dim=224,
        d_model=16,
        n_heads=4,
        ffn_dim=32,
        world_layers=1,
        latent_slots=2,
        latent_layers=1,
        local_layers=1,
        candidate_layers=1,
        recurrent_hidden_dim=32,
        dropout=0.0,
        domain_count=8,
        type_vocab_size=16,
        role_vocab_size=12,
        owner_vocab_size=16,
        entity_vocab_size=64,
        zone_vocab_size=10,
        order_vocab_size=16,
    )


def _snapshot(config: GroundedEncodingConfig) -> EncodedDecisionSnapshot:
    feature_dim = config.feature_dim
    world_feature = tuple([1.0] + [0.0] * (feature_dim - 1))
    candidate_a = tuple([0.0, 1.0] + [0.0] * (feature_dim - 2))
    candidate_b = tuple([0.0, 0.0, 1.0] + [0.0] * (feature_dim - 3))
    return EncodedDecisionSnapshot(
        config=config,
        encoding_fingerprint=grounding_encoding_identity()["fingerprint_sha256"],
        world=sparse_token_table(
            features=(world_feature,),
            ids=((2, 2, 2, 2, 2, 2, 0),),
            feature_dim=feature_dim,
            id_width=7,
        ),
        candidates=sparse_token_table(
            features=(candidate_a, candidate_b),
            ids=(
                (2, 2, 2, 3, 3, 2, 2, 4, 4),
                (3, 3, 2, 4, 4, 2, 2, 3, 3),
            ),
            feature_dim=feature_dim,
            id_width=9,
        ),
        locals=sparse_token_table(
            features=(),
            ids=(),
            feature_dim=feature_dim,
            id_width=7,
        ),
        local_offsets=np.asarray([0, 0, 0], dtype=np.uint32),
        action_mask=np.asarray([True, True], dtype=np.bool_),
        domain_id=1,
    )


def _trace(
    snapshot: EncodedDecisionSnapshot,
    *,
    trace_id: str,
    action_index: int,
    action_fingerprint: str,
    effect: TransactionEffect,
    transaction_return: float | None,
    node_key: str = "same-node",
    partition: str = "training",
) -> TransactionTrace:
    return TransactionTrace(
        trace_id=trace_id,
        episode_id=f"episode-{trace_id}",
        surface_key="selection-surface",
        start_step=10,
        policy_version=3,
        initial_recurrent_state=np.zeros(32, dtype=np.float32),
        steps=(
            TransactionStep(
                snapshot=snapshot,
                action_index=action_index,
                node_key=node_key,
                next_node_key=f"next-{trace_id}",
                action_fingerprint=action_fingerprint,
                effect=effect,
                selected_count_delta=0,
                transaction_return=transaction_return,
                return_steps=1 if transaction_return is not None else None,
            ),
        ),
        data_partition=partition,
    )


def test_transaction_heads_are_opt_in_and_candidate_equivariant() -> None:
    config = _model_config()
    disabled = RecurrentCandidateModel(config)
    assert not any("transaction" in key or "effect_head" in key for key in disabled.state_dict())

    enabled = RecurrentCandidateModel(config, enable_transaction_heads=True).eval()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    encoder = GroundedObservationEncoder(encoding)
    batch = encoder.collate_snapshots((_snapshot(encoding),))
    output = enabled(batch)

    assert output.candidate_effect_logits is not None
    assert output.selection_delta_logits is not None
    assert output.transaction_q_values is not None
    assert output.candidate_effect_logits.shape == (1, 2, 4)
    assert output.selection_delta_logits.shape == (1, 2, 3)
    assert output.transaction_q_values.shape == (1, 2)

    permutation = torch.tensor([1, 0])
    permuted = enabled(batch.permute_candidates(permutation))
    torch.testing.assert_close(
        permuted.candidate_effect_logits,
        output.candidate_effect_logits[:, permutation],
    )
    torch.testing.assert_close(
        permuted.selection_delta_logits,
        output.selection_delta_logits[:, permutation],
    )
    torch.testing.assert_close(
        permuted.transaction_q_values,
        output.transaction_q_values[:, permutation],
    )


def test_replay_is_bounded_checkpointable_and_rejects_heldout() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding)
    first = _trace(
        snapshot,
        trace_id="first",
        action_index=0,
        action_fingerprint="select-a",
        effect=TransactionEffect.EXIT,
        transaction_return=1.0,
    )
    second = _trace(
        snapshot,
        trace_id="second",
        action_index=1,
        action_fingerprint="deselect-a",
        effect=TransactionEffect.REVISIT,
        transaction_return=-1.0,
    )
    with pytest.raises(ValueError, match="held-out"):
        _trace(
            snapshot,
            trace_id="heldout",
            action_index=0,
            action_fingerprint="confirm",
            effect=TransactionEffect.EXIT,
            transaction_return=1.0,
            partition="evaluation",
        )

    replay = BoundedTransactionReplay(
        capacity=1,
        byte_capacity=max(first.storage_nbytes(), second.storage_nbytes()) + 128,
        seed=7,
    )
    assert replay.put(first)
    assert not replay.put(first)
    assert replay.put(second)
    assert replay.snapshot() == (second,)
    assert replay.metrics()["eviction_count"] == 1

    payload = replay.state_dict()
    restored = BoundedTransactionReplay(
        capacity=1,
        byte_capacity=replay.byte_capacity,
        seed=999,
    )
    restored.load_state_dict(payload)
    assert restored.snapshot() == replay.snapshot()
    assert restored.metrics() == replay.metrics()
    assert restored.sample(1)[0].trace_id == "second"


def test_pairwise_labels_require_same_node_distinct_factual_actions() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding)
    better = _trace(
        snapshot,
        trace_id="better",
        action_index=0,
        action_fingerprint="actual-a",
        effect=TransactionEffect.EXIT,
        transaction_return=2.0,
    )
    worse = _trace(
        snapshot,
        trace_id="worse",
        action_index=1,
        action_fingerprint="actual-b",
        effect=TransactionEffect.REVISIT,
        transaction_return=-1.0,
    )
    censored = _trace(
        snapshot,
        trace_id="censored",
        action_index=1,
        action_fingerprint="actual-c",
        effect=TransactionEffect.MOVE,
        transaction_return=None,
    )
    other_node = _trace(
        snapshot,
        trace_id="other-node",
        action_index=1,
        action_fingerprint="actual-d",
        effect=TransactionEffect.EXIT,
        transaction_return=10.0,
        node_key="different-node",
    )

    pairs = observed_outcome_pairs((better, worse, censored, other_node))
    assert len(pairs) == 1
    assert pairs[0].better_trace == 0
    assert pairs[0].worse_trace == 1


def test_factual_mc_backfill_masks_infrastructure_aborts() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    pending = _trace(
        _snapshot(encoding),
        trace_id="pending",
        action_index=0,
        action_fingerprint="factual-action",
        effect=TransactionEffect.EXIT,
        transaction_return=None,
    )
    rewards = (0.0,) * 10 + (0.25, 1.0)
    discounts = (0.9,) * 11 + (0.0,)

    completed = backfill_factual_monte_carlo_returns(
        pending,
        episode_rewards=rewards,
        episode_discounts=discounts,
        authoritative_outcome=True,
    )
    assert completed.steps[0].transaction_return == pytest.approx(1.15)
    assert completed.steps[0].return_steps == 2

    infrastructure_abort = backfill_factual_monte_carlo_returns(
        completed,
        episode_rewards=rewards,
        episode_discounts=discounts,
        authoritative_outcome=False,
    )
    assert infrastructure_abort.steps[0].transaction_return is None
    assert infrastructure_abort.steps[0].return_steps is None


def test_transaction_burn_in_recomputes_context_and_excludes_it_from_labels() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding_config)
    model = RecurrentCandidateModel(model_config, enable_transaction_heads=True)
    learner = VTraceLearner(
        model=model,
        encoder=GroundedObservationEncoder(encoding_config),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1.0e-3),
        config=TrainingConfig().optimization,
        maximum_unroll_length=4,
        maximum_policy_lag=10,
        transaction_config=TransactionLearningConfig(
            enabled=True,
            replay_byte_capacity=10_000_000,
            burn_in_steps=1,
        ),
    )
    context = TransactionStep(
        snapshot=snapshot,
        action_index=0,
        node_key="context-node",
        next_node_key="transaction-node",
        action_fingerprint="context-factual-action",
        effect=TransactionEffect.MOVE,
        selected_count_delta=0,
        transaction_return=None,
        return_steps=None,
    )
    factual = TransactionStep(
        snapshot=snapshot,
        action_index=1,
        node_key="transaction-node",
        next_node_key="exited",
        action_fingerprint="transaction-factual-action",
        effect=TransactionEffect.EXIT,
        selected_count_delta=0,
        transaction_return=1.0,
        return_steps=1,
    )
    trace = TransactionTrace(
        trace_id="burn-in",
        episode_id="burn-in-episode",
        surface_key="burn-in-surface",
        start_step=7,
        policy_version=0,
        initial_recurrent_state=np.zeros(32, dtype=np.float32),
        steps=(context, factual),
        burn_in_steps=1,
    )
    forward_calls = 0

    def count_forward(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal forward_calls
        forward_calls += 1

    handle = model.register_forward_hook(count_forward)
    try:
        losses = learner._transaction_losses((trace,))
    finally:
        handle.remove()

    assert forward_calls == 2
    assert losses[4] == 1
    assert losses[5] == 1
    with pytest.raises(ValueError, match="zero initial state"):
        replace(
            trace,
            initial_recurrent_state=np.ones(32, dtype=np.float32),
        )


def test_learner_uses_only_factual_effect_q_and_pairwise_targets() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding_config)
    model = RecurrentCandidateModel(model_config, enable_transaction_heads=True)
    encoder = GroundedObservationEncoder(encoding_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    transaction_config = TransactionLearningConfig(
        enabled=True,
        replay_byte_capacity=10_000_000,
        burn_in_steps=0,
        pairwise_ranking_weight=0.10,
    )
    learner = VTraceLearner(
        model=model,
        encoder=encoder,
        optimizer=optimizer,
        config=TrainingConfig().optimization,
        maximum_unroll_length=4,
        maximum_policy_lag=10,
        transaction_config=transaction_config,
    )
    # A zero-burn lineage is intentional and valid; it must not silently
    # inherit the default burn-in requirement from another configuration.
    assert transaction_config.burn_in_steps == 0
    unroll = SequenceUnroll(
        episode_id="main-vtrace",
        start_step=0,
        policy_version=0,
        initial_recurrent_state=np.zeros(32, dtype=np.float32),
        steps=(
            RolloutStep(
                snapshot=snapshot,
                action_index=0,
                behavior_log_probability=math.log(0.5),
                reward=0.0,
                discount=0.0,
                policy_decision=True,
            ),
        ),
        bootstrap_snapshot=None,
    )
    better = _trace(
        snapshot,
        trace_id="better",
        action_index=0,
        action_fingerprint="observed-a",
        effect=TransactionEffect.EXIT,
        transaction_return=1.0,
    )
    worse = _trace(
        snapshot,
        trace_id="worse",
        action_index=1,
        action_fingerprint="observed-b",
        effect=TransactionEffect.REVISIT,
        transaction_return=-1.0,
    )
    censored = _trace(
        snapshot,
        trace_id="censored",
        action_index=1,
        action_fingerprint="observed-c",
        effect=TransactionEffect.MOVE,
        transaction_return=None,
        node_key="censored-node",
    )

    metrics = learner.update(
        (unroll,),
        current_policy_version=0,
        transaction_traces=(better, worse, censored),
    )

    assert metrics.transaction_traces == 3
    assert metrics.transaction_effect_labels == 3
    assert metrics.transaction_q_labels == 2
    assert metrics.transaction_pairs == 1
    assert metrics.transaction_effect_loss > 0.0
    assert metrics.transaction_delta_loss > 0.0
    assert metrics.transaction_q_loss > 0.0
    assert metrics.transaction_pairwise_ranking_loss > 0.0


def test_training_collector_emits_factual_trace_but_evaluation_does_not() -> None:
    from tests.test_v2_training_pipeline import (  # local import avoids fixture coupling
        DynamicPreviewLoopBackend,
        _event_loop_config,
    )

    base = _event_loop_config(durable_window=4)
    config = replace(
        base,
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            sample_traces=2,
            burn_in_steps=2,
        ),
    )
    resources = build_training_resources(
        config,
        backend=DynamicPreviewLoopBackend(alternate_ui_actions=True),
    )
    try:
        training_episode = resources.collector.collect_episode(record=True)
        assert training_episode.metrics.noncombat_progress_stalled
        assert len(training_episode.transaction_traces) == 1
        trace = training_episode.transaction_traces[0]
        assert trace.data_partition == "training"
        assert trace.start_step == 0
        assert trace.burn_in_steps == 0
        assert all(step.transaction_return is not None for step in trace.learn_steps)

        evaluation_episode = resources.collector.collect_episode(
            record=False,
            evaluation_seed=9,
            deterministic=True,
        )
        assert evaluation_episode.transaction_traces == ()
    finally:
        resources.close()


def test_model_initialization_adds_only_new_heads_and_replay_roundtrips(
    tmp_path: Path,
) -> None:
    base = TrainingConfig()
    base = replace(
        base,
        model=replace(
            base.model,
            d_model=16,
            n_heads=4,
            ffn_dim=32,
            world_layers=1,
            latent_slots=2,
            latent_layers=1,
            local_layers=1,
            candidate_layers=1,
            recurrent_hidden_dim=32,
        ),
        runtime=replace(base.runtime, device="cpu", collector_device="cpu"),
    )
    source = build_training_resources(
        base,
        backend=cast(EnvironmentBackend, _NoopBackend()),
    )
    try:
        source_state = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
        checkpoint = save_training_checkpoint(
            tmp_path / "v10-source",
            config=base,
            resources=source,
            state=TrainingState(environment_steps=70_679, policy_version=1_111),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    v11 = replace(
        base,
        transaction_learning=replace(
            base.transaction_learning,
            enabled=True,
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            sample_traces=2,
        ),
    )
    target = build_training_resources(
        v11,
        backend=cast(EnvironmentBackend, _NoopBackend()),
    )
    try:
        new_head_before = {
            key: value.detach().clone() for key, value in target.model.state_dict().items() if key not in source_state
        }
        parent = initialize_model_from_checkpoint(
            checkpoint,
            config=v11,
            resources=target,
        )
        assert parent == checkpoint.resolve()
        assert len(target.optimizer.state) == 0
        assert target.transaction_replay is not None
        for key, expected in source_state.items():
            assert torch.equal(target.model.state_dict()[key], expected), key
        for key, expected in new_head_before.items():
            assert torch.equal(target.model.state_dict()[key], expected), key

        snapshot = _snapshot(v11.model.to_encoding_config())
        target.transaction_replay.put(
            TransactionTrace(
                trace_id="checkpoint-trace",
                episode_id="checkpoint-episode",
                surface_key="checkpoint-surface",
                start_step=0,
                policy_version=0,
                initial_recurrent_state=np.zeros(32, dtype=np.float32),
                steps=(
                    TransactionStep(
                        snapshot=snapshot,
                        action_index=0,
                        node_key="checkpoint-node",
                        next_node_key="checkpoint-next",
                        action_fingerprint="factual-action",
                        effect=TransactionEffect.EXIT,
                        selected_count_delta=0,
                        transaction_return=1.0,
                        return_steps=1,
                    ),
                ),
            )
        )
        migrated = save_training_checkpoint(
            tmp_path / "v11-lineage",
            config=v11,
            resources=target,
            state=TrainingState(),
            parent_checkpoint=checkpoint,
            checkpoint_load_mode="model_initialization",
            parent_relation="model_parameter_initialization",
        )
        assert (migrated / "transaction_replay.pkl").is_file()
        metadata = json.loads((migrated / "metadata.json").read_text("utf-8"))
        assert metadata["training_state"] == asdict(TrainingState())
        assert metadata["provenance"]["checkpoint_load_mode"] == "model_initialization"
        parent_metadata = metadata["provenance"]["parent_checkpoint"]
        assert parent_metadata["relation"] == "model_parameter_initialization"
        assert parent_metadata["training_state"]["policy_version"] == 1_111
    finally:
        target.close()

    restored = build_training_resources(
        v11,
        backend=cast(EnvironmentBackend, _NoopBackend()),
    )
    try:
        loaded = load_training_checkpoint(migrated, config=v11, resources=restored)
        assert loaded == TrainingState()
        assert restored.transaction_replay is not None
        assert restored.transaction_replay.snapshot()[0].trace_id == "checkpoint-trace"
    finally:
        restored.close()


def test_parameter_initialization_overlay_is_strict_except_for_new_heads() -> None:
    config = _model_config()
    source = RecurrentCandidateModel(config).state_dict()
    target = RecurrentCandidateModel(
        config,
        enable_transaction_heads=True,
    ).state_dict()

    migrated = checkpointing_module._model_parameter_initialization_state(
        source,
        target_state=target,
        allow_missing_transaction_heads=True,
    )
    assert set(migrated) == set(target)
    assert all(torch.equal(migrated[key], value) for key, value in source.items())

    partial_heads = dict(target)
    del partial_heads["transaction_q_head.3.bias"]
    with pytest.raises(ValueError, match="either all or none"):
        checkpointing_module._model_parameter_initialization_state(
            partial_heads,
            target_state=target,
            allow_missing_transaction_heads=True,
        )

    shared_key = "policy_head.0.weight"
    missing_shared = dict(source)
    del missing_shared[shared_key]
    with pytest.raises(ValueError, match="missing shared tensors"):
        checkpointing_module._model_parameter_initialization_state(
            missing_shared,
            target_state=target,
            allow_missing_transaction_heads=True,
        )

    unexpected = dict(source)
    unexpected["unsupported.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unsupported tensors"):
        checkpointing_module._model_parameter_initialization_state(
            unexpected,
            target_state=target,
            allow_missing_transaction_heads=True,
        )

    wrong_shape = dict(source)
    wrong_shape[shared_key] = source[shared_key][:-1]
    with pytest.raises(ValueError, match="tensor ABI differs"):
        checkpointing_module._model_parameter_initialization_state(
            wrong_shape,
            target_state=target,
            allow_missing_transaction_heads=True,
        )


def test_legacy_v10_lineage_without_transaction_section_exact_resumes_only_disabled(
    tmp_path: Path,
) -> None:
    config = TrainingConfig()
    legacy_lineage = config.lineage_mapping()
    del legacy_lineage["transaction_learning"]
    validated = ValidatedResumeCheckpoint(
        root=tmp_path,
        manifest={},
        metadata={
            "format": "sts2-recurrent-vtrace-checkpoint-v3",
            "model_config": asdict(config.model.to_model_config()),
            "encoding_contract": grounding_encoding_identity(),
            "model_state_spec": {},
            "training_state": asdict(TrainingState()),
            "lineage_config": legacy_lineage,
            "resolved_device": "cpu",
            "resolved_collector_device": "cpu",
            "queue_spec": {},
        },
    )

    checkpointing_module._validate_metadata(
        validated,
        config=config,
        resolved_device="cpu",
        resolved_collector_device="cpu",
        model_only=False,
    )

    enabled = replace(
        config,
        transaction_learning=replace(config.transaction_learning, enabled=True),
    )
    with pytest.raises(ValueError, match="identical immutable v2 lineage"):
        checkpointing_module._validate_metadata(
            validated,
            config=enabled,
            resolved_device="cpu",
            resolved_collector_device="cpu",
            model_only=False,
        )


def test_runtime_commits_completed_traces_then_samples_and_checkpoints_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_v2_training_pipeline import (
        DynamicPreviewLoopBackend,
        _event_loop_config,
    )

    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _event_loop_config(durable_window=4)
    config = replace(
        base,
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=16,
            replay_byte_capacity=10_000_000,
            sample_traces=4,
            burn_in_steps=2,
        ),
        runtime=replace(
            base.runtime,
            total_environment_steps=8,
            log_dir="runs/transaction-runtime",
            checkpoint_dir="checkpoints/transaction-runtime",
            checkpoint_interval_steps=100,
            evaluation_steps=(),
            evaluation_episodes=0,
        ),
    )

    state = run_training(
        config,
        backend=DynamicPreviewLoopBackend(alternate_ui_actions=True),
    )

    assert state.environment_steps == 8
    assert state.episodes == 2
    metrics_path = next(
        (tmp_path / "runs" / "transaction-runtime").glob("run-*/metrics.jsonl")
    )
    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    train_episodes = [event for event in events if event["event"] == "train_episode"]
    assert len(train_episodes) == 2
    assert all(event["transaction_traces_emitted"] == 1 for event in train_episodes)
    assert all(event["transaction_traces_stored"] == 1 for event in train_episodes)
    learner_updates = [event for event in events if event["event"] == "learner_update"]
    assert any(event["transaction_traces"] >= 1 for event in learner_updates)
    assert any(event["transaction_effect_labels"] >= 1 for event in learner_updates)

    checkpoint = next(
        (tmp_path / "checkpoints" / "transaction-runtime").glob("run-*/final-*")
    )
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["transaction_heads_enabled"] is True
    assert metadata["transaction_replay_spec"]["size"] == 2
    assert (checkpoint / "transaction_replay.pkl").is_file()
