from __future__ import annotations

import os

# This suite is deliberately CPU-only.  In particular it must remain runnable
# while the host's discrete AMD adapter is disabled or recovering from a TDR.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

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
            ids=((2, 2, 2, 2, 2, 2, 0),),
            feature_dim=feature_dim,
            id_width=7,
        ),
        candidates=sparse_token_table(
            features=(tuple(action_a), tuple(action_b)),
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
) -> CompletedEpisode:
    if policy_decisions is None:
        policy_decisions = (True,) * len(snapshots)
    if policy_versions is None:
        policy_versions = (0,) * len(snapshots)
    assert len(policy_decisions) == len(snapshots)
    assert len(policy_versions) == len(snapshots)
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
            task_reward=0.0,
            discount=1.0,
            revivals_before=0,
            revivals_after=final_revivals if index == len(snapshots) - 1 else 0,
            hp_loss_before=0.0,
            hp_loss_after=0.0,
            decision_surface=(
                "combat" if snapshot.domain_id == 1 else "other"
            ),
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
    primary_policy_weight: float = 0.25,
    task_value_weight: float = 0.25,
    revival_value_weight: float = 0.10,
    revival_policy_weight: float = 0.05,
    secondary_advantage_fraction: float = 0.25,
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
            primary_policy_weight=primary_policy_weight,
            task_value_weight=task_value_weight,
            revival_value_weight=revival_value_weight,
            revival_policy_weight=revival_policy_weight,
            secondary_advantage_fraction=secondary_advantage_fraction,
        ),
    )
    assert learner.device == CPU
    return learner, encoding


def _policy_gradient(model: RecurrentCandidateModel) -> torch.Tensor:
    gradients = [
        parameter.grad.detach().flatten()
        for name, parameter in model.named_parameters()
        if name.startswith("policy_head.") and parameter.grad is not None
    ]
    assert gradients
    return torch.cat(gradients)


def _zero_multiscale_heads(model: RecurrentCandidateModel) -> None:
    for name in (
        "combat_task_value_head",
        "act_task_value_head",
        "run_task_value_head",
        "combat_revival_cost_value_head",
        "act_revival_cost_value_head",
        "run_revival_cost_value_head",
    ):
        for parameter in getattr(model, name).parameters():
            torch.nn.init.zeros_(parameter)


def _constant_head_output(head: torch.nn.Sequential, value: float) -> None:
    for parameter in head.parameters():
        torch.nn.init.zeros_(parameter)
    final_linear = head[-1]
    assert isinstance(final_linear, torch.nn.Linear)
    torch.nn.init.constant_(final_linear.bias, value)


def test_early_action_beyond_online_unroll_gets_run_policy_gradient() -> None:
    learner, encoding = _learner(learn_steps=1)
    snapshot = _snapshot(encoding, domain_id=0)
    episode = _episode(
        (snapshot,) * 40,
        episode_id="long-run-success",
        won=True,
    )
    assert len(episode.steps) > learner.maximum_unroll_length
    assert episode.steps[0].run.success is True
    assert episode.steps[0].run.return_steps == 40

    sequence = _sequence(episode)
    losses = learner._episodic_losses((sequence,), current_policy_version=0)
    learner.model.zero_grad(set_to_none=True)
    losses.total_loss.backward()

    assert losses.policy_labels == 1
    assert losses.task_value_labels == 2  # Act and complete-run boundaries.
    gradient = _policy_gradient(learner.model)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_failed_horizon_is_value_only_without_anti_imitation_policy_label() -> None:
    learner, encoding = _learner(learn_steps=1)
    snapshot = _snapshot(encoding, domain_id=0)
    episode = _episode((snapshot,) * 12, episode_id="long-run-failure", won=False)

    losses = learner._episodic_losses(
        (_sequence(episode),),
        current_policy_version=0,
    )

    assert losses.task_value_labels == 2
    assert losses.success_policy_candidate_labels == 0
    assert losses.policy_labels == 0
    assert losses.policy_active_sequences == 0
    assert losses.failure_policy_suppressed_labels == 1
    assert losses.policy_lag_suppressed_labels == 0
    assert losses.revival_value_labels == 0
    assert losses.efficiency_policy_labels == 0
    assert losses.primary_policy_loss.detach().item() == 0.0
    assert losses.revival_value_loss.detach().item() == 0.0
    assert losses.revival_policy_loss.detach().item() == 0.0

    learner.model.zero_grad(set_to_none=True)
    losses.total_loss.backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for name, parameter in learner.model.named_parameters()
        if name.startswith("policy_head.")
    )


def test_stale_episode_keeps_value_labels_but_suppresses_policy_gradient() -> None:
    learner, encoding = _learner(learn_steps=1)
    snapshot = _snapshot(encoding, domain_id=0)
    episode = _episode((snapshot,), episode_id="stale-success", won=True)

    losses = learner._episodic_losses(
        (_sequence(episode),),
        current_policy_version=(
            learner.episodic_config.policy_gradient_max_lag + 1
        ),
    )

    assert losses.success_policy_candidate_labels == 1
    assert losses.policy_labels == 0
    assert losses.policy_active_sequences == 0
    assert losses.policy_lag_suppressed_labels == 1
    assert losses.failure_policy_suppressed_labels == 0
    assert losses.task_value_labels == 2
    assert losses.revival_value_labels == 2
    assert losses.task_value_loss.detach().item() > 0.0


def test_policy_label_classification_separates_fresh_stale_and_failed_sequences() -> None:
    learner, encoding = _learner(learn_steps=1)
    snapshot = _snapshot(encoding, domain_id=0)
    current_policy_version = learner.episodic_config.policy_gradient_max_lag + 1
    fresh_success = _episode(
        (snapshot,),
        episode_id="fresh-success",
        won=True,
        policy_versions=(current_policy_version,),
    )
    stale_success = _episode(
        (snapshot,),
        episode_id="stale-success",
        won=True,
        policy_versions=(0,),
    )
    stale_failure = _episode(
        (snapshot,),
        episode_id="stale-failure",
        won=False,
        policy_versions=(0,),
    )

    losses = learner._episodic_losses(
        (
            _sequence(fresh_success),
            _sequence(stale_success),
            _sequence(stale_failure),
        ),
        current_policy_version=current_policy_version,
    )

    assert losses.success_policy_candidate_labels == 2
    assert losses.policy_labels == 1
    assert losses.policy_active_sequences == 1
    assert losses.policy_lag_suppressed_labels == 1
    assert losses.failure_policy_suppressed_labels == 1
    assert losses.task_value_labels == 6
    assert losses.revival_value_labels == 4


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
    losses = learner._episodic_losses((sequence,), current_policy_version=0)
    learner.model.zero_grad(set_to_none=True)
    losses.total_loss.backward()

    assert losses.task_value_labels == 2
    assert losses.revival_value_labels == 2
    assert losses.success_policy_candidate_labels == 0
    assert losses.policy_labels == 0
    assert losses.policy_active_sequences == 0
    assert losses.failure_policy_suppressed_labels == 0
    assert losses.policy_lag_suppressed_labels == 0
    assert losses.efficiency_policy_labels == 0
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

    losses = learner._episodic_losses(
        (_sequence(episode),),
        current_policy_version=0,
    )

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
    losses = learner._episodic_losses((sequence,), current_policy_version=0)
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
    first = learner._episodic_losses((sparse,), current_policy_version=0)
    assert learner.model.training
    second = learner._episodic_losses((sparse,), current_policy_version=0)
    assert learner.model.training
    complete_history = learner._episodic_losses((full,), current_policy_version=0)
    assert learner.model.training

    for name in (
        "total_loss",
        "primary_policy_loss",
        "task_value_loss",
        "revival_value_loss",
        "revival_policy_loss",
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
    learner._episodic_losses((sparse,), current_policy_version=0)
    assert not learner.model.training

    learner.model.train()
    with pytest.raises(ValueError, match="newer than the learner"):
        learner._episodic_losses((sparse,), current_policy_version=-1)
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


def test_secondary_revival_signal_cannot_reverse_or_exceed_primary_fraction() -> None:
    primary_weight = 0.4
    secondary_fraction = 0.25
    learner, encoding = _learner(
        learn_steps=1,
        primary_policy_weight=primary_weight,
        task_value_weight=0.0,
        revival_value_weight=0.0,
        revival_policy_weight=10.0,
        secondary_advantage_fraction=secondary_fraction,
    )
    _zero_multiscale_heads(learner.model)
    snapshot = _snapshot(encoding, domain_id=0)
    episode = _episode(
        (snapshot,),
        episode_id="successful-but-costly",
        won=True,
        final_revivals=100,
    )
    losses = learner._episodic_losses(
        (_sequence(episode),),
        current_policy_version=0,
    )
    assert losses.efficiency_policy_labels == 1

    learner.model.zero_grad(set_to_none=True)
    losses.primary_policy_loss.backward(retain_graph=True)
    primary_gradient = _policy_gradient(learner.model)
    learner.model.zero_grad(set_to_none=True)
    losses.total_loss.backward()
    combined_gradient = _policy_gradient(learner.model)

    assert torch.dot(primary_gradient, combined_gradient) > 0.0
    ratio = combined_gradient.norm() / primary_gradient.norm()
    lower = primary_weight * (1.0 - secondary_fraction)
    upper = primary_weight * (1.0 + secondary_fraction)
    assert float(ratio) >= lower - 1e-5
    assert float(ratio) <= upper + 1e-5


def test_success_tie_keeps_revival_preference_after_primary_calibration() -> None:
    def update_probability(final_revivals: int) -> tuple[float, float, float]:
        learner, encoding = _learner(
            learn_steps=1,
            primary_policy_weight=1.0,
            task_value_weight=0.0,
            revival_value_weight=0.0,
            revival_policy_weight=10.0,
            secondary_advantage_fraction=0.25,
        )
        snapshot = _snapshot(encoding, domain_id=0)
        encoded = collate_encoded_snapshots(
            (snapshot,),
            expected_config=encoding,
            expected_fingerprint=grounding_encoding_identity()[
                "fingerprint_sha256"
            ],
            device=CPU,
        )
        # The helper trajectory has task_return=0 and an explicit successful
        # boundary outcome of +1.  Make the run value exactly calibrated so
        # its policy advantage is zero; only the success-stratum revival
        # tie-break may distinguish the two otherwise successful paths.
        _constant_head_output(learner.model.run_task_value_head, 1.0)
        with torch.no_grad():
            before = float(
                torch.softmax(learner.model(encoded).policy_logits, dim=-1)[0, 0]
            )
        episode = _episode(
            (snapshot,),
            episode_id=f"calibrated-success-{final_revivals}",
            won=True,
            final_revivals=final_revivals,
        )
        losses = learner._episodic_losses(
            (_sequence(episode),),
            current_policy_version=0,
        )
        assert losses.efficiency_policy_labels == 1
        assert losses.primary_policy_loss.detach().item() == pytest.approx(0.0)
        learner.optimizer.zero_grad(set_to_none=True)
        losses.total_loss.backward()
        learner.optimizer.step()
        with torch.no_grad():
            after = float(
                torch.softmax(learner.model(encoded).policy_logits, dim=-1)[0, 0]
            )
        return before, after, float(losses.total_loss.detach().item())

    low_before, low_after, low_loss = update_probability(0)
    high_before, high_after, high_loss = update_probability(100)

    assert low_before == pytest.approx(high_before)
    assert low_after > low_before
    assert high_after < high_before
    assert low_loss != pytest.approx(0.0)
    assert high_loss != pytest.approx(0.0)


def test_revival_cost_cannot_override_primary_outside_success_tie_band() -> None:
    learner, encoding = _learner(
        learn_steps=1,
        primary_policy_weight=1.0,
        task_value_weight=0.0,
        revival_value_weight=0.0,
        revival_policy_weight=10.0,
        secondary_advantage_fraction=0.25,
    )
    snapshot = _snapshot(encoding, domain_id=0)
    encoded = collate_encoded_snapshots(
        (snapshot,),
        expected_config=encoding,
        expected_fingerprint=grounding_encoding_identity()["fingerprint_sha256"],
        device=CPU,
    )
    tolerance = learner.episodic_config.primary_success_tie_tolerance
    # Successful target is +1. Set the task value just outside the explicit
    # tie band, leaving a small but material positive completion advantage.
    _constant_head_output(
        learner.model.run_task_value_head,
        1.0 - tolerance - 0.01,
    )
    with torch.no_grad():
        before = float(
            torch.softmax(learner.model(encoded).policy_logits, dim=-1)[0, 0]
        )
    costly_success = _episode(
        (snapshot,),
        episode_id="outside-tie-costly-success",
        won=True,
        final_revivals=100,
    )
    losses = learner._episodic_losses(
        (_sequence(costly_success),),
        current_policy_version=0,
    )
    learner.optimizer.zero_grad(set_to_none=True)
    losses.total_loss.backward()
    learner.optimizer.step()
    with torch.no_grad():
        after = float(
            torch.softmax(learner.model(encoded).policy_logits, dim=-1)[0, 0]
        )

    # High revival cost may weaken but cannot reverse the material primary
    # completion update outside the explicitly declared success-tie stratum.
    assert after > before


def test_importance_ratio_is_detached_so_rare_good_action_is_encouraged() -> None:
    learner, encoding = _learner(
        learn_steps=1,
        primary_policy_weight=1.0,
        task_value_weight=0.0,
        revival_value_weight=0.0,
        revival_policy_weight=0.0,
        secondary_advantage_fraction=0.0,
    )
    snapshot = _snapshot(encoding, domain_id=0)
    encoded = collate_encoded_snapshots(
        (snapshot,),
        expected_config=encoding,
        expected_fingerprint=grounding_encoding_identity()["fingerprint_sha256"],
        device=CPU,
    )

    # Make factual action 0 rare under the current policy while its recorded
    # behavior probability remains 0.5.  With a differentiable rho, the
    # product-rule term reverses the update once log(pi(a)) < -1.
    preparation = torch.optim.Adam(learner.model.parameters(), lr=0.002)
    for _ in range(100):
        logits = learner.model(encoded).policy_logits
        probability = float(torch.softmax(logits.detach(), dim=-1)[0, 0])
        if probability < 0.2:
            break
        make_action_one_likely = -F.log_softmax(logits, dim=-1)[0, 1]
        preparation.zero_grad(set_to_none=True)
        make_action_one_likely.backward()
        preparation.step()
    with torch.no_grad():
        before = float(
            torch.softmax(learner.model(encoded).policy_logits, dim=-1)[0, 0]
        )
    assert 0.01 < before < 0.2

    _zero_multiscale_heads(learner.model)
    episode = _episode((snapshot,), episode_id="rare-good-action", won=True)
    losses = learner._episodic_losses(
        (_sequence(episode),),
        current_policy_version=0,
    )
    assert float(losses.importance_ratios[0]) < 0.4
    learner.optimizer.zero_grad(set_to_none=True)
    losses.total_loss.backward()
    learner.optimizer.step()
    with torch.no_grad():
        after = float(
            torch.softmax(learner.model(encoded).policy_logits, dim=-1)[0, 0]
        )

    assert after > before
