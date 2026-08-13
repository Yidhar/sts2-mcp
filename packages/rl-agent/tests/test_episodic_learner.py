from __future__ import annotations

import os

# This suite is deliberately CPU-only.  In particular it must remain runnable
# while the host's discrete AMD adapter is disabled or recovering from a TDR.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import torch

from sts2_rl.encoding import (
    EncodedDecisionSnapshot,
    GroundedEncodingConfig,
    GroundedObservationEncoder,
    grounding_encoding_identity,
)
from sts2_rl.encoding.snapshot import collate_encoded_snapshots, sparse_token_table
from sts2_rl.models import GroundedCandidateConfig, RecurrentCandidateModel
from sts2_rl.training.config import EpisodicLearningConfig, OptimizationConfig
from sts2_rl.training.episode_replay import (
    BoundaryOutcome,
    BoundedEpisodicReplay,
    CompletedEpisode,
    EpisodeCompletion,
    EpisodeDecisionStep,
    HorizonTargets,
    ReplaySequence,
    backfill_completed_episode,
)
from sts2_rl.training.learner import (
    VTraceLearner,
    _annealed_entropy_weight,
)

CPU = torch.device("cpu")


def test_entropy_weight_anneals_to_explicit_nonzero_floor() -> None:
    config = OptimizationConfig(
        entropy_weight=0.02,
        entropy_weight_end=0.004,
        entropy_decay_updates=2_000,
    )

    assert _annealed_entropy_weight(config, policy_version=0) == pytest.approx(
        0.02
    )
    assert _annealed_entropy_weight(
        config,
        policy_version=1_000,
    ) == pytest.approx(0.012)
    assert _annealed_entropy_weight(
        config,
        policy_version=2_000,
    ) == pytest.approx(0.004)
    assert _annealed_entropy_weight(
        config,
        policy_version=20_000,
    ) == pytest.approx(0.004)


def test_learner_dynamics_v3_round_trips_and_refuses_retired_breaker_payloads() -> None:
    learner, _ = _learner()
    state = learner.dynamics_state_dict()
    assert state == {"version": "sts2-vtrace-learner-dynamics-v3"}
    assert VTraceLearner.validate_dynamics_state_dict(state) == state

    restored, _ = _learner()
    restored.load_dynamics_state_dict(state)
    assert restored.dynamics_state_dict() == state

    # A v2 payload still carries the retired entropy-breaker counters. Exact
    # resume must refuse it outright rather than silently dropping state.
    legacy = {
        "version": "sts2-vtrace-learner-dynamics-v2",
        "collapse_batch_streak": 0,
        "entropy_breaker_remaining_updates": 7,
        "entropy_breaker_triggers": 1,
    }
    with pytest.raises(ValueError, match="learner dynamics state keys mismatch"):
        restored.load_dynamics_state_dict(legacy)
    # Even a key-stripped v2 payload fails on its version string.
    with pytest.raises(ValueError, match="unsupported learner dynamics state"):
        restored.load_dynamics_state_dict(
            {"version": "sts2-vtrace-learner-dynamics-v2"}
        )
    # Extra keys on a v3 payload fail closed as well.
    with pytest.raises(ValueError, match="learner dynamics state keys mismatch"):
        restored.load_dynamics_state_dict({**state, "entropy_breaker_triggers": 0})
    with pytest.raises(TypeError, match="learner dynamics state must be an object"):
        restored.load_dynamics_state_dict(None)


def _model_config(*, dropout: float = 0.0) -> GroundedCandidateConfig:
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
        dropout=dropout,
        domain_count=8,
        type_vocab_size=16,
        role_vocab_size=12,
        owner_vocab_size=16,
        entity_vocab_size=64,
        zone_vocab_size=10,
        order_vocab_size=16,
    )


def _encoding_config(model: GroundedCandidateConfig) -> GroundedEncodingConfig:
    return GroundedEncodingConfig.from_model_config(
        model,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )


def _snapshot(
    config: GroundedEncodingConfig,
    *,
    domain_id: int,
    singleton: bool = False,
    feature_slot: int = 0,
) -> EncodedDecisionSnapshot:
    feature_dim = config.feature_dim
    world = [0.0] * feature_dim
    world[feature_slot % feature_dim] = 1.0
    action_a = [0.0] * feature_dim
    action_b = [0.0] * feature_dim
    action_a[(feature_slot + 1) % feature_dim] = 1.0
    action_b[(feature_slot + 2) % feature_dim] = 1.0
    return EncodedDecisionSnapshot(
        config=config,
        encoding_fingerprint=grounding_encoding_identity()["fingerprint_sha256"],
        world=sparse_token_table(
            features=(tuple(world),),
            ids=((2, 2, 2, 2, 2, 2, 2, 2, 0),),
            feature_dim=feature_dim,
            id_width=9,
        ),
        candidates=sparse_token_table(
            features=(tuple(action_a), tuple(action_b)),
            ids=(
                (2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4),
                (3, 3, 2, 4, 4, 4, 4, 2, 2, 3, 3, 3, 3),
            ),
            feature_dim=feature_dim,
            id_width=13,
        ),
        locals=sparse_token_table(
            features=(),
            ids=(),
            feature_dim=feature_dim,
            id_width=9,
        ),
        local_offsets=np.asarray([0, 0, 0], dtype=np.uint32),
        action_mask=np.asarray([True, not singleton], dtype=np.bool_),
        domain_id=domain_id,
    )


def _episode(
    snapshots: tuple[EncodedDecisionSnapshot, ...],
    *,
    episode_id: str,
    won: bool,
    final_revivals: int = 0,
    policy_decisions: tuple[bool, ...] | None = None,
    policy_versions: tuple[int, ...] | None = None,
    decision_surfaces: tuple[str, ...] | None = None,
) -> CompletedEpisode:
    if policy_decisions is None:
        policy_decisions = (True,) * len(snapshots)
    if policy_versions is None:
        policy_versions = (0,) * len(snapshots)
    if decision_surfaces is None:
        decision_surfaces = tuple(
            "combat" if snapshot.domain_id == 1 else "other"
            for snapshot in snapshots
        )
    assert len(policy_decisions) == len(snapshots)
    assert len(policy_versions) == len(snapshots)
    assert len(decision_surfaces) == len(snapshots)
    steps = tuple(
        EpisodeDecisionStep(
            snapshot=snapshot,
            step_index=index,
            action_index=0,
            behavior_log_probability=-0.6931471805599453,
            policy_decision=policy_decisions[index],
            policy_version=policy_versions[index],
            act=1,
            combat_id=None,
            task_reward=(1.0 if won else -1.0)
            if index == len(snapshots) - 1
            else 0.0,
            discount=0.0 if index == len(snapshots) - 1 else 1.0,
            revivals_before=0,
            revivals_after=final_revivals if index == len(snapshots) - 1 else 0,
            hp_loss_before=0.0,
            hp_loss_after=0.0,
            decision_surface=decision_surfaces[index],
            act_boundary=(
                BoundaryOutcome.SUCCEEDED if won else BoundaryOutcome.FAILED
            )
            if index == len(snapshots) - 1
            else BoundaryOutcome.NONE,
        )
        for index, snapshot in enumerate(snapshots)
    )
    return backfill_completed_episode(
        episode_id=episode_id,
        steps=steps,
        completion=EpisodeCompletion(
            authoritative=True,
            won=won,
            final_revivals=final_revivals,
            final_hp_loss=0.0,
            terminal_reason="run_victory" if won else "run_defeat",
        ),
    )


def _sequence(
    episode: CompletedEpisode,
    *,
    learn_start: int = 0,
    learn_steps: int = 1,
    prefix_indexes: tuple[int, ...] = (),
    configured_burn_in_steps: int = 0,
) -> ReplaySequence:
    prefix = tuple(episode.steps[index] for index in prefix_indexes)
    learning = episode.steps[learn_start : learn_start + learn_steps]
    steps = prefix + learning
    return ReplaySequence(
        episode_id=episode.episode_id,
        start_step=steps[0].step_index,
        learn_start_step=learn_start,
        steps=steps,
        burn_in_steps=len(prefix),
        configured_burn_in_steps=configured_burn_in_steps,
        source_episode_won=episode.won,
        source_episode_authoritative=episode.completion.authoritative,
    )


def _learner(
    *,
    burn_in_steps: int = 0,
    learn_steps: int = 4,
    task_value_weight: float = 0.25,
    revival_value_weight: float = 0.10,
    combat_hp_loss_value_weight: float = 0.0,
    combat_hp_loss_reference: float = 80.0,
    dropout: float = 0.0,
) -> tuple[VTraceLearner, GroundedEncodingConfig]:
    torch.manual_seed(11)
    model_config = _model_config(dropout=dropout)
    encoding = _encoding_config(model_config)
    model = RecurrentCandidateModel(model_config).to(CPU)
    learner = VTraceLearner(
        model=model,
        encoder=GroundedObservationEncoder(encoding),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        config=OptimizationConfig(),
        maximum_unroll_length=4,
        maximum_policy_lag=100,
        episodic_config=EpisodicLearningConfig(
            enabled=True,
            burn_in_steps=burn_in_steps,
            learn_steps=learn_steps,
            task_value_weight=task_value_weight,
            revival_value_weight=revival_value_weight,
            combat_hp_loss_value_weight=combat_hp_loss_value_weight,
            combat_hp_loss_reference=combat_hp_loss_reference,
        ),
    )
    assert learner.device == CPU
    return learner, encoding


def _horizon(
    *,
    success: bool,
    task_return: float,
    return_steps: int,
) -> HorizonTargets:
    return HorizonTargets(
        success=success,
        future_revivals=0,
        future_hp_loss=0.0,
        task_return=task_return,
        return_steps=return_steps,
    )


@pytest.mark.parametrize(
    ("success", "terminal_return"),
    ((True, 1.0), (False, -1.0)),
)
def test_run_task_target_consumes_factual_terminal_outcome_exactly_once(
    success: bool,
    terminal_return: float,
) -> None:
    target = _horizon(
        success=success,
        task_return=terminal_return,
        return_steps=4,
    )

    assert VTraceLearner._episodic_task_target(
        "run",
        target,
        run_target=target,
    ) == pytest.approx(terminal_return)


@pytest.mark.parametrize("horizon", ("combat", "act"))
def test_local_task_target_adds_one_boundary_unit_unless_run_terminal_is_shared(
    horizon: str,
) -> None:
    run_target = _horizon(success=True, task_return=1.5, return_steps=8)
    earlier_local_boundary = _horizon(
        success=True,
        task_return=0.25,
        return_steps=3,
    )
    terminal_local_boundary = _horizon(
        success=True,
        task_return=1.5,
        return_steps=8,
    )

    assert VTraceLearner._episodic_task_target(
        horizon,
        earlier_local_boundary,
        run_target=run_target,
    ) == pytest.approx(1.25)
    assert VTraceLearner._episodic_task_target(
        horizon,
        terminal_local_boundary,
        run_target=run_target,
    ) == pytest.approx(1.5)


def test_local_boundary_cannot_conflict_with_shared_run_terminal_outcome() -> None:
    run_failure = _horizon(success=False, task_return=-1.0, return_steps=2)
    conflicting_local_success = _horizon(
        success=True,
        task_return=-1.0,
        return_steps=2,
    )

    with pytest.raises(ValueError, match="conflicting outcome"):
        VTraceLearner._episodic_task_target(
            "combat",
            conflicting_local_success,
            run_target=run_failure,
        )


def test_failed_combat_hp_loss_is_bounded_factual_value_supervision() -> None:
    learner, encoding = _learner(
        learn_steps=1,
        task_value_weight=0.0,
        revival_value_weight=0.0,
        combat_hp_loss_value_weight=0.10,
        combat_hp_loss_reference=80.0,
    )
    snapshot = _snapshot(encoding, domain_id=1)
    episode = backfill_completed_episode(
        episode_id="failed-combat-hp-loss",
        steps=(
            EpisodeDecisionStep(
                snapshot=snapshot,
                step_index=0,
                action_index=0,
                behavior_log_probability=-0.6931471805599453,
                policy_decision=True,
                policy_version=0,
                act=1,
                combat_id="combat-1",
                task_reward=-1.0,
                discount=0.0,
                revivals_before=0,
                revivals_after=1,
                hp_loss_before=0.0,
                hp_loss_after=20.0,
                combat_boundary=BoundaryOutcome.FAILED,
                act_boundary=BoundaryOutcome.FAILED,
                decision_surface="combat",
            ),
        ),
        completion=EpisodeCompletion(
            authoritative=True,
            won=False,
            final_revivals=1,
            final_hp_loss=20.0,
            terminal_reason="run_defeat",
        ),
    )

    losses = learner._episodic_losses((_sequence(episode),))
    learner.model.zero_grad(set_to_none=True)
    losses.total_loss.backward()

    assert losses.combat_hp_loss_value_labels == 1
    assert losses.combat_hp_loss_value_loss.detach().item() > 0.0
    assert torch.isfinite(losses.combat_hp_loss_value_loss)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in learner.model.combat_hp_loss_value_head.parameters()
    )


def test_forced_singleton_trains_values_without_any_policy_label() -> None:
    learner, encoding = _learner(learn_steps=1)
    singleton = _snapshot(encoding, domain_id=0, singleton=True)
    episode = _episode(
        (singleton,),
        episode_id="forced-success",
        won=True,
        policy_decisions=(False,),
    )
    sequence = _sequence(episode)
    losses = learner._episodic_losses((sequence,))
    learner.model.zero_grad(set_to_none=True)
    losses.total_loss.backward()

    assert losses.task_value_labels == 2
    assert losses.revival_value_labels == 2
    assert losses.task_value_loss.detach().item() > 0.0
    assert losses.revival_value_loss.detach().item() > 0.0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for name, parameter in learner.model.named_parameters()
        if name.startswith("policy_head.")
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for name, parameter in learner.model.named_parameters()
        if name.startswith("run_task_value_head.")
    )


def test_revival_value_log_observation_bounds_unlimited_revival_tail() -> None:
    learner, encoding = _learner(learn_steps=1)
    snapshot = _snapshot(encoding, domain_id=0, singleton=True)
    episode = _episode(
        (snapshot,),
        episode_id="thousand-revival-success",
        won=True,
        final_revivals=1_000,
        policy_decisions=(False,),
    )

    losses = learner._episodic_losses((_sequence(episode),))

    assert losses.revival_value_labels == 2
    assert torch.isfinite(losses.revival_value_loss)
    # Raw-count smooth-L1 would be approximately one thousand here and dominate
    # every shared representation gradient. The monotone log1p observation
    # model keeps this auxiliary target on a single-digit scale.
    assert 0.0 < float(losses.revival_value_loss.detach().item()) < 10.0
    learner.model.zero_grad(set_to_none=True)
    losses.total_loss.backward()
    finite_gradients = [
        parameter.grad
        for parameter in learner.model.parameters()
        if parameter.grad is not None
    ]
    assert finite_gradients
    assert all(torch.isfinite(gradient).all() for gradient in finite_gradients)


def test_sparse_exact_burn_in_matches_full_split_recurrent_history() -> None:
    learner, encoding = _learner(burn_in_steps=1, learn_steps=1)
    domains = (0, 1, 1, 0, 1, 1)
    snapshots = tuple(
        _snapshot(encoding, domain_id=domain, feature_slot=index)
        for index, domain in enumerate(domains)
    )
    episode = _episode(snapshots, episode_id="split-memory", won=False)
    size = episode.storage_nbytes()
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=size,
        episode_byte_capacity=size,
        max_segments_per_episode=6,
        seed=7,
    )
    assert replay.put(episode)
    sequences = replay.sample(6, learn_steps=1, burn_in_steps=1)
    sequence = next(item for item in sequences if item.learn_start_step == 5)
    assert [step.step_index for step in sequence.burn_in] == [0, 3, 4]

    def recurrent_state(steps: tuple) -> torch.Tensor:
        state = learner.model.initial_state(1, device=CPU)
        with torch.no_grad():
            for step in steps:
                encoded = collate_encoded_snapshots(
                    (step.snapshot,),
                    expected_config=encoding,
                    expected_fingerprint=grounding_encoding_identity()[
                        "fingerprint_sha256"
                    ],
                    device=CPU,
                )
                state = learner.model(
                    encoded,
                    state,
                    validate=False,
                ).recurrent_state
        return state

    full = recurrent_state(episode.steps[: sequence.learn_start_step])
    sparse = recurrent_state(sequence.burn_in)
    torch.testing.assert_close(sparse, full, atol=1e-6, rtol=1e-6)
    losses = learner._episodic_losses((sequence,))
    assert losses.burn_in_steps == 3
    assert losses.learn_steps == 1


def test_dropout_replay_is_deterministic_exact_and_restores_model_mode() -> None:
    learner, encoding = _learner(burn_in_steps=1, learn_steps=1, dropout=0.35)
    domains = (0, 1, 1, 0, 1, 1)
    snapshots = tuple(
        _snapshot(encoding, domain_id=domain, feature_slot=index)
        for index, domain in enumerate(domains)
    )
    episode = _episode(snapshots, episode_id="dropout-split-memory", won=False)
    size = episode.storage_nbytes()
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=size,
        episode_byte_capacity=size,
        max_segments_per_episode=6,
        seed=7,
    )
    assert replay.put(episode)
    sampled = replay.sample(6, learn_steps=1, burn_in_steps=1)
    sparse = next(item for item in sampled if item.learn_start_step == 5)
    assert [step.step_index for step in sparse.burn_in] == [0, 3, 4]
    full = ReplaySequence(
        episode_id=episode.episode_id,
        start_step=0,
        learn_start_step=5,
        steps=episode.steps[:6],
        burn_in_steps=5,
        configured_burn_in_steps=1,
        source_episode_won=episode.won,
        source_episode_authoritative=episode.completion.authoritative,
    )

    learner.model.train()
    first = learner._episodic_losses((sparse,))
    assert learner.model.training
    second = learner._episodic_losses((sparse,))
    assert learner.model.training
    complete_history = learner._episodic_losses((full,))
    assert learner.model.training

    for name in (
        "total_loss",
        "task_value_loss",
        "revival_value_loss",
    ):
        torch.testing.assert_close(
            getattr(first, name),
            getattr(second, name),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            getattr(first, name),
            getattr(complete_history, name),
            atol=1e-6,
            rtol=1e-6,
        )

    learner.model.eval()
    learner._episodic_losses((sparse,))
    assert not learner.model.training

    learner.model.train()
    mismatched = ReplaySequence(
        episode_id=episode.episode_id,
        start_step=0,
        learn_start_step=5,
        steps=episode.steps[:6],
        burn_in_steps=5,
        configured_burn_in_steps=3,
        source_episode_won=episode.won,
        source_episode_authoritative=episode.completion.authoritative,
    )
    with pytest.raises(ValueError, match="burn-in configuration"):
        learner._episodic_losses((mismatched,))
    assert learner.model.training


def test_ten_thousand_step_source_keeps_graph_suffix_bounded() -> None:
    _, encoding = _learner(burn_in_steps=4, learn_steps=32)
    noncombat = _snapshot(encoding, domain_id=0)
    episode = _episode(
        (noncombat,) * 10_000,
        episode_id="ten-thousand-source",
        won=False,
    )
    size = episode.storage_nbytes()
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=size,
        episode_byte_capacity=size,
        max_segments_per_episode=1,
        seed=31,
    )
    assert replay.put(episode)
    sequence = replay.sample(1, learn_steps=32, burn_in_steps=4)[0]

    assert replay.metrics()["maximum_observed_episode_steps"] == 10_000
    assert sequence.learn_start_step > 1_000
    assert len(sequence.burn_in) == sequence.learn_start_step
    assert sequence.burn_in_no_grad
    assert sequence.exact_recurrent_reconstruction
    assert len(sequence.learn_steps) == 32
    assert len(sequence.learn_steps) <= 32