from __future__ import annotations

import pickle
import threading
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

import sts2_rl.training.episode_replay as episode_replay_module
from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedEncodingConfig
from sts2_rl.encoding.snapshot import sparse_token_table
from sts2_rl.training.episode_replay import (
    BoundaryOutcome,
    BoundedEpisodicReplay,
    CompletedEpisode,
    EpisodeCompletion,
    EpisodeDecisionStep,
    ReplaySequence,
    backfill_completed_episode,
)


def _snapshot(*, domain_id: int = 0) -> EncodedDecisionSnapshot:
    config = GroundedEncodingConfig(
        max_world_tokens=4,
        max_candidates=4,
        max_candidate_local_tokens=2,
    )
    feature_dim = config.feature_dim
    world = tuple([1.0] + [0.0] * (feature_dim - 1))
    action_a = tuple([0.0, 1.0] + [0.0] * (feature_dim - 2))
    action_b = tuple([0.0, 0.0, 1.0] + [0.0] * (feature_dim - 3))
    return EncodedDecisionSnapshot(
        config=config,
        encoding_fingerprint="0" * 64,
        world=sparse_token_table(
            features=(world,),
            ids=((2, 2, 2, 2, 2, 2, 2, 2, 0),),
            feature_dim=feature_dim,
            id_width=9,
        ),
        candidates=sparse_token_table(
            features=(action_a, action_b),
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
        action_mask=np.asarray([True, True], dtype=np.bool_),
        domain_id=domain_id,
    )


def _step(
    snapshot: EncodedDecisionSnapshot,
    index: int,
    *,
    act: int,
    combat_id: str | None,
    reward: float,
    before_revivals: int,
    after_revivals: int,
    before_hp: float,
    after_hp: float,
    combat_boundary: BoundaryOutcome = BoundaryOutcome.NONE,
    act_boundary: BoundaryOutcome = BoundaryOutcome.NONE,
    policy_decision: bool = True,
    decision_surface: str | None = None,
) -> EpisodeDecisionStep:
    return EpisodeDecisionStep(
        snapshot=snapshot,
        step_index=index,
        action_index=index % 2,
        behavior_log_probability=-0.5,
        policy_decision=policy_decision,
        policy_version=7,
        act=act,
        combat_id=combat_id,
        task_reward=reward,
        discount=1.0,
        revivals_before=before_revivals,
        revivals_after=after_revivals,
        hp_loss_before=before_hp,
        hp_loss_after=after_hp,
        combat_boundary=combat_boundary,
        act_boundary=act_boundary,
        decision_surface=(
            "combat"
            if snapshot.domain_id == 1
            else "other"
            if decision_surface is None
            else decision_surface
        ),
    )


def test_complete_episode_backfill_propagates_exact_combat_act_and_run_targets() -> None:
    snapshot = _snapshot()
    steps = (
        _step(
            snapshot,
            0,
            act=1,
            combat_id="act1-combat",
            reward=1.0,
            before_revivals=0,
            after_revivals=0,
            before_hp=0.0,
            after_hp=4.0,
        ),
        _step(
            snapshot,
            1,
            act=1,
            combat_id="act1-combat",
            reward=2.0,
            before_revivals=0,
            after_revivals=1,
            before_hp=4.0,
            after_hp=20.0,
            combat_boundary=BoundaryOutcome.SUCCEEDED,
        ),
        _step(
            snapshot,
            2,
            act=1,
            combat_id=None,
            reward=3.0,
            before_revivals=1,
            after_revivals=1,
            before_hp=20.0,
            after_hp=20.0,
            act_boundary=BoundaryOutcome.SUCCEEDED,
        ),
        _step(
            snapshot,
            3,
            act=2,
            combat_id="act2-combat",
            reward=4.0,
            before_revivals=1,
            after_revivals=1,
            before_hp=20.0,
            after_hp=25.0,
        ),
        _step(
            snapshot,
            4,
            act=2,
            combat_id="act2-combat",
            reward=5.0,
            before_revivals=1,
            after_revivals=3,
            before_hp=25.0,
            after_hp=55.0,
            combat_boundary=BoundaryOutcome.FAILED,
        ),
        _step(
            snapshot,
            5,
            act=2,
            combat_id=None,
            reward=6.0,
            before_revivals=3,
            after_revivals=3,
            before_hp=55.0,
            after_hp=60.0,
            act_boundary=BoundaryOutcome.FAILED,
        ),
    )
    episode = backfill_completed_episode(
        episode_id="episode-long-credit",
        steps=steps,
        completion=EpisodeCompletion(
            authoritative=True,
            won=False,
            final_revivals=3,
            final_hp_loss=60.0,
            terminal_reason="run_defeat",
        ),
    )

    first = episode.steps[0]
    assert first.combat.success is True
    assert first.combat.future_revivals == 1
    assert first.combat.future_hp_loss == 20.0
    assert first.combat.task_return == 3.0
    assert first.combat.return_steps == 2
    assert first.combat.efficiency_eligible
    assert first.act.success is True
    assert first.act.future_revivals == 1
    assert first.act.future_hp_loss == 20.0
    assert first.act.task_return == 6.0
    assert first.act.return_steps == 3
    assert first.run.success is False
    assert first.run.future_revivals == 3
    assert first.run.future_hp_loss == 60.0
    assert first.run.task_return == 21.0
    assert first.run.return_steps == 6
    assert not first.run.efficiency_eligible

    second_combat = episode.steps[3].combat
    assert second_combat.success is False
    assert second_combat.future_revivals == 2
    assert second_combat.future_hp_loss == 35.0
    assert second_combat.task_return == 9.0
    assert episode.steps[2].combat.observed is False

    second_act = episode.steps[3].act
    assert second_act.success is False
    assert second_act.future_revivals == 2
    assert second_act.future_hp_loss == 40.0
    assert second_act.task_return == 15.0


def test_censored_boundaries_never_fabricate_long_horizon_targets() -> None:
    snapshot = _snapshot()
    steps = (
        _step(
            snapshot,
            0,
            act=1,
            combat_id="censored-combat",
            reward=1.0,
            before_revivals=0,
            after_revivals=2,
            before_hp=0.0,
            after_hp=50.0,
            combat_boundary=BoundaryOutcome.CENSORED,
            act_boundary=BoundaryOutcome.CENSORED,
        ),
    )
    episode = backfill_completed_episode(
        episode_id="episode-censored",
        steps=steps,
        completion=EpisodeCompletion(
            authoritative=False,
            won=None,
            final_revivals=2,
            final_hp_loss=50.0,
            terminal_reason="transport_abort",
        ),
    )

    target = episode.steps[0]
    assert not target.combat.observed
    assert not target.act.observed
    assert not target.run.observed


def _linear_episode(
    snapshot: EncodedDecisionSnapshot,
    *,
    episode_id: str,
    length: int,
    won: bool | None,
    partition: str = "training",
) -> CompletedEpisode:
    steps = tuple(
        _step(
            snapshot,
            index,
            act=1,
            combat_id=None,
            reward=float(index + 1),
            before_revivals=index,
            after_revivals=index + 1,
            before_hp=float(index * 2),
            after_hp=float((index + 1) * 2),
            act_boundary=(
                BoundaryOutcome.CENSORED
                if won is None
                else BoundaryOutcome.SUCCEEDED
                if won
                else BoundaryOutcome.FAILED
            )
            if index == length - 1
            else BoundaryOutcome.NONE,
        )
        for index in range(length)
    )
    return backfill_completed_episode(
        episode_id=episode_id,
        steps=steps,
        completion=EpisodeCompletion(
            authoritative=won is not None,
            won=won,
            final_revivals=length,
            final_hp_loss=float(length * 2),
            terminal_reason=(
                "transport_abort"
                if won is None
                else "run_victory"
                if won
                else "run_defeat"
            ),
        ),
        data_partition=partition,
    )


def _with_policy_versions(
    episode: CompletedEpisode,
    versions: tuple[int, ...],
    *,
    policy_decisions: tuple[bool, ...] | None = None,
) -> CompletedEpisode:
    assert len(versions) == len(episode.steps)
    decisions = (
        tuple(step.decision.policy_decision for step in episode.steps)
        if policy_decisions is None
        else policy_decisions
    )
    assert len(decisions) == len(episode.steps)
    return backfill_completed_episode(
        episode_id=episode.episode_id,
        steps=tuple(
            replace(
                step.decision,
                policy_version=versions[index],
                policy_decision=decisions[index],
            )
            for index, step in enumerate(episode.steps)
        ),
        completion=episode.completion,
        data_partition=episode.data_partition,
    )


def _surface_episode(
    *,
    episode_id: str,
    surfaces: tuple[str, ...],
    policy_decisions: tuple[bool, ...] | None = None,
    won: bool | None = True,
) -> CompletedEpisode:
    """Build one factual non-combat episode with explicit replay surfaces."""

    snapshot = _snapshot(domain_id=0)
    decisions = (
        (True,) * len(surfaces)
        if policy_decisions is None
        else policy_decisions
    )
    assert len(decisions) == len(surfaces)
    steps = tuple(
        _step(
            snapshot,
            index,
            act=1,
            combat_id=None,
            reward=0.0,
            before_revivals=0,
            after_revivals=0,
            before_hp=0.0,
            after_hp=0.0,
            act_boundary=(
                BoundaryOutcome.CENSORED
                if won is None and index == len(surfaces) - 1
                else BoundaryOutcome.SUCCEEDED
                if won is True and index == len(surfaces) - 1
                else BoundaryOutcome.FAILED
                if won is False and index == len(surfaces) - 1
                else BoundaryOutcome.NONE
            ),
            policy_decision=decisions[index],
            decision_surface=surface,
        )
        for index, surface in enumerate(surfaces)
    )
    return backfill_completed_episode(
        episode_id=episode_id,
        steps=steps,
        completion=EpisodeCompletion(
            authoritative=won is not None,
            won=won,
            final_revivals=0,
            final_hp_loss=0.0,
            terminal_reason=(
                "transport_abort"
                if won is None
                else "run_victory"
                if won
                else "run_defeat"
            ),
        ),
    )


def test_replay_is_byte_bounded_episode_balanced_and_samples_no_grad_burn_in() -> None:
    snapshot = _snapshot()
    first = _linear_episode(snapshot, episode_id="first", length=8, won=False)
    second = _linear_episode(snapshot, episode_id="second", length=9, won=True)
    third = _linear_episode(snapshot, episode_id="third", length=13, won=True)
    byte_capacity = second.storage_nbytes() + third.storage_nbytes() + 16
    replay = BoundedEpisodicReplay(
        capacity=2,
        byte_capacity=byte_capacity,
        episode_byte_capacity=max(item.storage_nbytes() for item in (first, second, third)),
        max_segments_per_episode=2,
        seed=31,
    )

    assert replay.put(first)
    assert replay.put(second)
    assert replay.put(third)
    assert [episode.episode_id for episode in replay.snapshot()] == ["second", "third"]
    assert replay.metrics()["storage_nbytes"] == second.storage_nbytes() + third.storage_nbytes()

    sequences = replay.sample(20, learn_steps=3, burn_in_steps=2)
    counts = Counter(sequence.episode_id for sequence in sequences)
    assert counts == {"second": 2, "third": 2}
    assert sequences[0].episode_id != sequences[1].episode_id
    for sequence in sequences:
        assert sequence.burn_in_no_grad
        assert sequence.exact_recurrent_reconstruction
        assert sequence.configured_burn_in_steps == 2
        assert sequence.burn_in_steps >= min(sequence.learn_start_step, 2)
        assert 1 <= len(sequence.learn_steps) <= 3
        assert len(sequence.burn_in) == sequence.burn_in_steps
        assert all(isinstance(step.snapshot.world.ids, np.ndarray) for step in sequence.steps)
        assert all(not step.snapshot.world.ids.flags.writeable for step in sequence.steps)
        assert not any(
            isinstance(value, torch.Tensor)
            for step in sequence.steps
            for value in (step.snapshot.world.ids, step.decision.behavior_log_probability)
        )

    metrics = replay.metrics()
    assert metrics["eviction_count"] == 1
    assert metrics["maximum_observed_episode_steps"] == 13
    assert metrics["sample_count"] == 4


def test_replay_rejects_oversize_episode_and_evaluation_partition() -> None:
    snapshot = _snapshot()
    training = _linear_episode(snapshot, episode_id="large", length=3, won=True)
    held_out = replace(training, episode_id="held-out", data_partition="held_out")
    replay = BoundedEpisodicReplay(
        capacity=4,
        byte_capacity=training.storage_nbytes() * 2,
        episode_byte_capacity=training.storage_nbytes() - 1,
        max_segments_per_episode=1,
        seed=0,
    )

    assert replay.put(training) is False
    with pytest.raises(ValueError, match="held-out"):
        replay.put(held_out)
    assert replay.metrics()["oversize_count"] == 1


def test_storage_accounting_includes_exact_snapshot_and_utf8_payload_once() -> None:
    snapshot = _snapshot()
    ascii_step = _step(
        snapshot,
        0,
        act=1,
        combat_id="a",
        reward=0.0,
        before_revivals=0,
        after_revivals=0,
        before_hp=0.0,
        after_hp=0.0,
        combat_boundary=BoundaryOutcome.SUCCEEDED,
        act_boundary=BoundaryOutcome.SUCCEEDED,
    )
    utf8_step = replace(ascii_step, combat_id="战")
    assert utf8_step.storage_nbytes() - ascii_step.storage_nbytes() == len("战".encode()) - 1
    utf8_surface = replace(ascii_step, decision_surface="牌")
    assert utf8_surface.storage_nbytes() - ascii_step.storage_nbytes() == (
        len("牌".encode()) - len(ascii_step.decision_surface.encode())
    )
    assert ascii_step.storage_nbytes() > snapshot.storage_nbytes()

    episode = backfill_completed_episode(
        episode_id="bytes",
        steps=(ascii_step,),
        completion=EpisodeCompletion(
            authoritative=True,
            won=True,
            final_revivals=0,
            final_hp_loss=0.0,
            terminal_reason="run_victory",
        ),
    )
    expected = (
        len(episode.episode_id.encode())
        + len(episode.data_partition.encode())
        + len(episode.version.encode())
        + episode.completion.storage_nbytes()
        + sum(step.storage_nbytes() for step in episode.steps)
    )
    assert episode.storage_nbytes() == expected


def test_step_rejects_torch_scalar_so_no_grad_fn_can_enter_replay() -> None:
    snapshot = _snapshot()
    with pytest.raises(TypeError, match="behavior_log_probability"):
        EpisodeDecisionStep(
            snapshot=snapshot,
            step_index=0,
            action_index=0,
            behavior_log_probability=torch.tensor(-0.5, requires_grad=True),  # type: ignore[arg-type]
            policy_decision=True,
            policy_version=0,
            act=1,
            combat_id=None,
            task_reward=0.0,
            discount=1.0,
            revivals_before=0,
            revivals_after=0,
            hp_loss_before=0.0,
            hp_loss_after=0.0,
        )


def test_act_zero_and_exact_policy_decision_flag_are_supported() -> None:
    snapshot = _snapshot()
    step = _step(
        snapshot,
        0,
        act=0,
        combat_id=None,
        reward=0.0,
        before_revivals=0,
        after_revivals=0,
        before_hp=0.0,
        after_hp=0.0,
        act_boundary=BoundaryOutcome.CENSORED,
    )
    assert step.act == 0
    assert step.policy_decision is True
    with pytest.raises(TypeError, match="policy_decision"):
        replace(step, policy_decision=1)  # type: ignore[arg-type]


def test_decision_surfaces_fail_closed_and_combat_is_canonical() -> None:
    noncombat = _step(
        _snapshot(domain_id=0),
        0,
        act=1,
        combat_id=None,
        reward=0.0,
        before_revivals=0,
        after_revivals=0,
        before_hp=0.0,
        after_hp=0.0,
        decision_surface="card_reward",
    )
    combat = _step(
        _snapshot(domain_id=1),
        0,
        act=1,
        combat_id="combat-one",
        reward=0.0,
        before_revivals=0,
        after_revivals=0,
        before_hp=0.0,
        after_hp=0.0,
    )

    assert noncombat.decision_surface == "card_reward"
    assert combat.decision_surface == "combat"
    with pytest.raises(ValueError, match="cannot claim the combat surface"):
        replace(noncombat, decision_surface="combat")
    with pytest.raises(ValueError, match="canonical combat surface"):
        replace(combat, decision_surface="map")
    with pytest.raises(ValueError, match="non-empty"):
        replace(noncombat, decision_surface=" ")


@pytest.mark.parametrize(
    "prior_version",
    ("sts2-episodic-replay-v1", "sts2-episodic-replay-v2"),
)
def test_v3_replay_rejects_prior_exact_resume_sidecars(
    prior_version: str,
) -> None:
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=1024,
        episode_byte_capacity=1024,
        max_segments_per_episode=1,
        seed=7,
    )
    payload = replay.state_dict()
    payload["version"] = prior_version

    with pytest.raises(ValueError, match="unsupported episodic replay"):
        replay.load_state_dict(payload)


def test_sampling_round_robins_win_failure_and_censored_strata() -> None:
    snapshot = _snapshot()
    episodes = tuple(
        _linear_episode(snapshot, episode_id=f"win-{index}", length=2, won=True)
        for index in range(6)
    ) + tuple(
        _linear_episode(snapshot, episode_id=f"failure-{index}", length=2, won=False)
        for index in range(2)
    ) + (
        _linear_episode(snapshot, episode_id="censored", length=2, won=None),
    )
    total_bytes = sum(episode.storage_nbytes() for episode in episodes)
    replay = BoundedEpisodicReplay(
        capacity=len(episodes),
        byte_capacity=total_bytes,
        episode_byte_capacity=max(episode.storage_nbytes() for episode in episodes),
        max_segments_per_episode=1,
        seed=41,
    )
    assert all(replay.put(episode) for episode in episodes)

    first_round = replay.sample(3, learn_steps=2, burn_in_steps=1)
    strata = {
        "censored"
        if not sequence.source_episode_authoritative
        else "win"
        if sequence.source_episode_won
        else "failure"
        for sequence in first_round
    }
    assert strata == {"win", "failure", "censored"}
    assert len({sequence.episode_id for sequence in first_round}) == 3


def test_fresh_policy_reservation_keeps_one_global_value_slot() -> None:
    snapshot = _snapshot()
    fresh = _with_policy_versions(
        _linear_episode(snapshot, episode_id="fresh-win", length=2, won=True),
        (0, 100),
    )
    stale_failure = _with_policy_versions(
        _linear_episode(snapshot, episode_id="stale-failure", length=2, won=False),
        (0, 0),
    )
    episodes = (fresh, stale_failure)
    replay = BoundedEpisodicReplay(
        capacity=2,
        byte_capacity=sum(episode.storage_nbytes() for episode in episodes),
        episode_byte_capacity=max(episode.storage_nbytes() for episode in episodes),
        max_segments_per_episode=1,
        seed=11,
    )
    assert all(replay.put(episode) for episode in episodes)

    sample = replay.sample_for_learning(
        2,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=0.0,
        current_policy_version=100,
        policy_gradient_max_lag=5,
        fresh_policy_sequences=1,
    )

    assert len(sample.sequences) == 2
    assert sample.sequences[0].episode_id == "fresh-win"
    assert sample.sequences[0].learn_start_step == 1
    assert sample.sequences[1].episode_id == "stale-failure"
    assert sample.diagnostics.to_mapping() == {
        "fresh_policy_quota_requested": 1,
        "fresh_policy_quota_filled": 1,
        "fresh_policy_quota_missed": 0,
        "fresh_policy_candidate_episodes": 1,
        "fresh_policy_candidate_decisions": 1,
        "sampled_fresh_policy_lag_min": 0,
        "sampled_fresh_policy_lag_mean": 0.0,
        "sampled_fresh_policy_lag_max": 0,
    }


@pytest.mark.parametrize(
    ("behavior_version", "filled"),
    ((72, 1), (71, 0)),
)
def test_fresh_policy_lag_boundary_is_closed(
    behavior_version: int,
    filled: int,
) -> None:
    snapshot = _snapshot()
    episode = _with_policy_versions(
        _linear_episode(snapshot, episode_id="lag-boundary", length=2, won=True),
        (0, behavior_version),
        policy_decisions=(False, True),
    )
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=episode.storage_nbytes(),
        episode_byte_capacity=episode.storage_nbytes(),
        max_segments_per_episode=1,
        seed=13,
    )
    assert replay.put(episode)

    sample = replay.sample_for_learning(
        1,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=0.0,
        current_policy_version=200,
        policy_gradient_max_lag=128,
        fresh_policy_sequences=1,
    )

    assert len(sample.sequences) == 1
    assert sample.diagnostics.fresh_policy_quota_filled == filled
    if filled:
        assert sample.sequences[0].learn_start_step == 1
        assert sample.diagnostics.sampled_fresh_policy_lag_max == 128


def test_fresh_policy_reservation_requires_an_actual_learner_label() -> None:
    snapshot = _snapshot()
    forced_win = _with_policy_versions(
        _linear_episode(snapshot, episode_id="forced-win", length=2, won=True),
        (100, 100),
        policy_decisions=(False, False),
    )
    failed = _with_policy_versions(
        _linear_episode(snapshot, episode_id="fresh-failure", length=2, won=False),
        (100, 100),
    )
    stale_win = _with_policy_versions(
        _linear_episode(snapshot, episode_id="stale-win", length=2, won=True),
        (0, 0),
    )
    episodes = (forced_win, failed, stale_win)
    replay = BoundedEpisodicReplay(
        capacity=3,
        byte_capacity=sum(episode.storage_nbytes() for episode in episodes),
        episode_byte_capacity=max(episode.storage_nbytes() for episode in episodes),
        max_segments_per_episode=1,
        seed=17,
    )
    assert all(replay.put(episode) for episode in episodes)

    sample = replay.sample_for_learning(
        2,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=0.0,
        current_policy_version=100,
        policy_gradient_max_lag=5,
        fresh_policy_sequences=1,
    )

    assert len(sample.sequences) == 2
    assert sample.diagnostics.fresh_policy_candidate_episodes == 0
    assert sample.diagnostics.fresh_policy_candidate_decisions == 0
    assert sample.diagnostics.fresh_policy_quota_filled == 0
    assert sample.diagnostics.fresh_policy_quota_missed == 1


def test_censored_run_with_successful_act_is_fresh_policy_eligible() -> None:
    snapshot = _snapshot()
    decisions = (
        _step(
            snapshot,
            0,
            act=1,
            combat_id=None,
            reward=0.0,
            before_revivals=0,
            after_revivals=0,
            before_hp=0.0,
            after_hp=0.0,
            act_boundary=BoundaryOutcome.SUCCEEDED,
        ),
    )
    episode = backfill_completed_episode(
        episode_id="censored-after-act-success",
        steps=decisions,
        completion=EpisodeCompletion(
            authoritative=False,
            won=None,
            final_revivals=0,
            final_hp_loss=0.0,
            terminal_reason="transport_abort",
        ),
    )
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=episode.storage_nbytes(),
        episode_byte_capacity=episode.storage_nbytes(),
        max_segments_per_episode=1,
        seed=19,
    )
    assert replay.put(episode)

    sample = replay.sample_for_learning(
        1,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=0.0,
        current_policy_version=7,
        policy_gradient_max_lag=0,
        fresh_policy_sequences=1,
    )

    assert sample.diagnostics.fresh_policy_quota_filled == 1
    assert sample.sequences[0].learn_start_step == 0


def test_fresh_macro_sequence_satisfies_macro_reservation() -> None:
    fresh_map = _with_policy_versions(
        _surface_episode(
            episode_id="fresh-map",
            surfaces=("forced", "map"),
            policy_decisions=(False, True),
            won=True,
        ),
        (0, 100),
    )
    stale_failure = _with_policy_versions(
        _surface_episode(
            episode_id="stale-event",
            surfaces=("event",),
            won=False,
        ),
        (0,),
    )
    episodes = (fresh_map, stale_failure)
    replay = BoundedEpisodicReplay(
        capacity=2,
        byte_capacity=sum(episode.storage_nbytes() for episode in episodes),
        episode_byte_capacity=max(episode.storage_nbytes() for episode in episodes),
        max_segments_per_episode=1,
        seed=23,
    )
    assert all(replay.put(episode) for episode in episodes)

    sample = replay.sample_for_learning(
        2,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=0.5,
        current_policy_version=100,
        policy_gradient_max_lag=5,
        fresh_policy_sequences=1,
    )

    assert len(sample.sequences) == 2
    assert sample.sequences[0].episode_id == "fresh-map"
    assert sample.sequences[0].learn_start_step == 1
    assert replay.metrics()["macro_sample_count"] == 1


def test_fresh_reservation_prefers_macro_when_combat_is_also_eligible() -> None:
    base_episode = backfill_completed_episode(
        episode_id="fresh-mixed",
        steps=(
            _step(
                _snapshot(domain_id=1),
                0,
                act=1,
                combat_id="combat-1",
                reward=0.0,
                before_revivals=0,
                after_revivals=0,
                before_hp=0.0,
                after_hp=0.0,
                combat_boundary=BoundaryOutcome.SUCCEEDED,
            ),
            _step(
                _snapshot(domain_id=0),
                1,
                act=1,
                combat_id=None,
                reward=0.0,
                before_revivals=0,
                after_revivals=0,
                before_hp=0.0,
                after_hp=0.0,
                act_boundary=BoundaryOutcome.SUCCEEDED,
                decision_surface="map",
            ),
        ),
        completion=EpisodeCompletion(
            authoritative=True,
            won=True,
            final_revivals=0,
            final_hp_loss=0.0,
            terminal_reason="run_victory",
        ),
    )
    mixed_success = _with_policy_versions(
        base_episode,
        (100, 100),
    )
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=mixed_success.storage_nbytes(),
        episode_byte_capacity=mixed_success.storage_nbytes(),
        max_segments_per_episode=1,
        seed=25,
    )
    assert replay.put(mixed_success)

    sample = replay.sample_for_learning(
        1,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=1.0,
        current_policy_version=100,
        policy_gradient_max_lag=5,
        fresh_policy_sequences=1,
    )

    assert sample.diagnostics.fresh_policy_quota_filled == 1
    assert sample.sequences[0].learn_start_step == 1
    assert replay.metrics()["macro_sample_count"] == 1


def test_future_behavior_policy_in_fresh_view_fails_closed() -> None:
    snapshot = _snapshot()
    episode = _with_policy_versions(
        _linear_episode(snapshot, episode_id="future", length=1, won=True),
        (101,),
    )
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=episode.storage_nbytes(),
        episode_byte_capacity=episode.storage_nbytes(),
        max_segments_per_episode=1,
        seed=29,
    )
    assert replay.put(episode)

    with pytest.raises(ValueError, match="newer than the learner"):
        replay.sample_for_learning(
            1,
            learn_steps=1,
            burn_in_steps=0,
            macro_sample_fraction=0.0,
            current_policy_version=100,
            policy_gradient_max_lag=5,
            fresh_policy_sequences=1,
        )


def test_macro_fraction_reserves_exact_noncombat_policy_decisions_by_surface() -> None:
    episodes = (
        _surface_episode(
            episode_id="macro-map",
            surfaces=(
                "forced_event",
                "map",
                "forced_event",
                "forced_event",
                "forced_event",
                "forced_event",
            ),
            policy_decisions=(False, True, False, False, False, False),
        ),
        _surface_episode(
            episode_id="macro-card-reward",
            surfaces=(
                "forced_event",
                "card_reward",
                "forced_event",
                "forced_event",
                "forced_event",
                "forced_event",
            ),
            policy_decisions=(False, True, False, False, False, False),
        ),
    )
    total_bytes = sum(episode.storage_nbytes() for episode in episodes)
    replay = BoundedEpisodicReplay(
        capacity=2,
        byte_capacity=total_bytes,
        episode_byte_capacity=max(
            episode.storage_nbytes() for episode in episodes
        ),
        max_segments_per_episode=2,
        seed=101,
    )
    assert all(replay.put(episode) for episode in episodes)

    sequences = replay.sample(
        4,
        learn_steps=2,
        burn_in_steps=1,
        macro_sample_fraction=0.5,
    )

    assert len(sequences) == 4
    reserved = tuple(
        sequence
        for sequence in sequences
        if sequence.learn_steps[0].decision.policy_decision
    )
    assert len(reserved) == 2
    reserved_decisions = tuple(
        sequence.learn_steps[0].decision for sequence in reserved
    )
    assert {decision.decision_surface for decision in reserved_decisions} == {
        "map",
        "card_reward",
    }
    assert all(decision.policy_decision for decision in reserved_decisions)
    assert all(decision.snapshot.domain_id != 1 for decision in reserved_decisions)
    for sequence, decision in zip(reserved, reserved_decisions, strict=True):
        source = next(
            episode
            for episode in episodes
            if episode.episode_id == sequence.episode_id
        )
        assert decision is source.steps[sequence.learn_start_step].decision
    assert replay.metrics()["macro_sample_count"] == 2
    assert replay.metrics()["sample_count"] == 4


def test_macro_reservation_excludes_combat_and_forced_decisions() -> None:
    combat_snapshot = _snapshot(domain_id=1)
    combat_steps = tuple(
        _step(
            combat_snapshot,
            index,
            act=1,
            combat_id="combat-one",
            reward=0.0,
            before_revivals=0,
            after_revivals=0,
            before_hp=0.0,
            after_hp=0.0,
            combat_boundary=(
                BoundaryOutcome.SUCCEEDED
                if index == 1
                else BoundaryOutcome.NONE
            ),
            act_boundary=(
                BoundaryOutcome.SUCCEEDED
                if index == 1
                else BoundaryOutcome.NONE
            ),
        )
        for index in range(2)
    )
    combat = backfill_completed_episode(
        episode_id="combat-only",
        steps=combat_steps,
        completion=EpisodeCompletion(
            authoritative=True,
            won=True,
            final_revivals=0,
            final_hp_loss=0.0,
            terminal_reason="run_victory",
        ),
    )
    forced = _surface_episode(
        episode_id="forced-noncombat",
        surfaces=("map", "card_reward"),
        policy_decisions=(False, False),
    )
    episodes = (combat, forced)
    total_bytes = sum(episode.storage_nbytes() for episode in episodes)
    replay = BoundedEpisodicReplay(
        capacity=2,
        byte_capacity=total_bytes,
        episode_byte_capacity=max(
            episode.storage_nbytes() for episode in episodes
        ),
        max_segments_per_episode=1,
        seed=37,
    )
    assert all(replay.put(episode) for episode in episodes)

    sequences = replay.sample(
        2,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=1.0,
    )

    assert len(sequences) == 2
    assert replay.metrics()["macro_sample_count"] == 0


def test_zero_macro_fraction_preserves_default_sampling_and_rng_contract() -> None:
    episodes = (
        _surface_episode(
            episode_id="zero-win",
            surfaces=("map", "card_reward", "shop", "rest"),
            won=True,
        ),
        _surface_episode(
            episode_id="zero-loss",
            surfaces=("event", "map", "event", "map"),
            won=False,
        ),
    )
    total_bytes = sum(episode.storage_nbytes() for episode in episodes)

    def build() -> BoundedEpisodicReplay:
        replay = BoundedEpisodicReplay(
            capacity=2,
            byte_capacity=total_bytes,
            episode_byte_capacity=max(
                episode.storage_nbytes() for episode in episodes
            ),
            max_segments_per_episode=2,
            seed=59,
        )
        assert all(replay.put(episode) for episode in episodes)
        return replay

    implicit = build()
    explicit = build()
    implicit_sequences = implicit.sample(4, learn_steps=1, burn_in_steps=1)
    explicit_sequences = explicit.sample(
        4,
        learn_steps=1,
        burn_in_steps=1,
        macro_sample_fraction=0.0,
    )

    def projection(sequence: ReplaySequence) -> tuple[object, ...]:
        return (
            sequence.episode_id,
            sequence.learn_start_step,
            tuple(step.step_index for step in sequence.burn_in),
            tuple(step.step_index for step in sequence.learn_steps),
        )

    assert [projection(sequence) for sequence in explicit_sequences] == [
        projection(sequence) for sequence in implicit_sequences
    ]
    assert explicit.state_dict()["rng_state"] == implicit.state_dict()["rng_state"]
    assert explicit.metrics()["macro_sample_count"] == 0
    assert implicit.metrics()["macro_sample_count"] == 0


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (-0.01, ValueError),
        (1.01, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("0.5", TypeError),
    ),
)
def test_macro_sample_fraction_is_strictly_validated(
    value: object,
    error: type[Exception],
) -> None:
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=1024,
        episode_byte_capacity=1024,
        max_segments_per_episode=1,
        seed=0,
    )
    with pytest.raises(error, match="macro_sample_fraction"):
        replay.sample(
            1,
            learn_steps=1,
            burn_in_steps=0,
            macro_sample_fraction=value,  # type: ignore[arg-type]
        )


def test_macro_sampling_keeps_every_episode_ahead_of_any_second_segment() -> None:
    macro_rich = _surface_episode(
        episode_id="macro-rich",
        surfaces=("map", "card_reward", "shop", "rest"),
    )
    forced_only = _surface_episode(
        episode_id="forced-only",
        surfaces=("event", "event", "event", "event"),
        policy_decisions=(False, False, False, False),
    )
    episodes = (macro_rich, forced_only)
    total_bytes = sum(episode.storage_nbytes() for episode in episodes)
    replay = BoundedEpisodicReplay(
        capacity=2,
        byte_capacity=total_bytes,
        episode_byte_capacity=max(
            episode.storage_nbytes() for episode in episodes
        ),
        max_segments_per_episode=2,
        seed=3,
    )
    assert all(replay.put(episode) for episode in episodes)

    sequences = replay.sample(
        4,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=0.5,
    )
    first_positions = {
        episode.episode_id: next(
            index
            for index, sequence in enumerate(sequences)
            if sequence.episode_id == episode.episode_id
        )
        for episode in episodes
    }
    second_positions = [
        index
        for episode in episodes
        for index, sequence in enumerate(sequences)
        if sequence.episode_id == episode.episode_id
        and sum(
            prior.episode_id == episode.episode_id
            for prior in sequences[:index]
        )
        == 1
    ]

    assert len(sequences) == 4
    assert second_positions
    assert min(second_positions) > max(first_positions.values())


def test_macro_sampling_preserves_outcome_strata_round_robin() -> None:
    episodes = (
        _surface_episode(
            episode_id="macro-win-a",
            surfaces=("map",),
            won=True,
        ),
        _surface_episode(
            episode_id="macro-win-b",
            surfaces=("card_reward",),
            won=True,
        ),
        _surface_episode(
            episode_id="macro-failure",
            surfaces=("shop",),
            won=False,
        ),
        _surface_episode(
            episode_id="macro-censored",
            surfaces=("event",),
            won=None,
        ),
    )
    total_bytes = sum(episode.storage_nbytes() for episode in episodes)
    replay = BoundedEpisodicReplay(
        capacity=len(episodes),
        byte_capacity=total_bytes,
        episode_byte_capacity=max(
            episode.storage_nbytes() for episode in episodes
        ),
        max_segments_per_episode=1,
        seed=17,
    )
    assert all(replay.put(episode) for episode in episodes)

    sequences = replay.sample(
        3,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=1.0,
    )
    strata = {
        "censored"
        if not sequence.source_episode_authoritative
        else "win"
        if sequence.source_episode_won
        else "failure"
        for sequence in sequences
    }

    assert len(sequences) == 3
    assert strata == {"win", "failure", "censored"}
    assert replay.metrics()["macro_sample_count"] == 3


def test_split_gru_prefix_is_sparse_but_exact_and_learning_suffix_is_contiguous() -> None:
    noncombat = _snapshot(domain_id=0)
    combat = _snapshot(domain_id=1)
    domains = (noncombat, combat, combat, noncombat, combat, combat)
    steps = tuple(
        _step(
            snapshot,
            index,
            act=1,
            combat_id=("combat-one" if index in {1, 2} else "combat-two" if index in {4, 5} else None),
            reward=0.0,
            before_revivals=0,
            after_revivals=0,
            before_hp=0.0,
            after_hp=0.0,
            combat_boundary=(
                BoundaryOutcome.SUCCEEDED
                if index == 2
                else BoundaryOutcome.FAILED
                if index == 5
                else BoundaryOutcome.NONE
            ),
            act_boundary=BoundaryOutcome.FAILED if index == 5 else BoundaryOutcome.NONE,
        )
        for index, snapshot in enumerate(domains)
    )
    episode = backfill_completed_episode(
        episode_id="split-gru",
        steps=steps,
        completion=EpisodeCompletion(
            authoritative=True,
            won=False,
            final_revivals=0,
            final_hp_loss=0.0,
            terminal_reason="run_defeat",
        ),
    )
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=episode.storage_nbytes(),
        episode_byte_capacity=episode.storage_nbytes(),
        max_segments_per_episode=6,
        seed=7,
    )
    assert replay.put(episode)
    sequences = replay.sample(6, learn_steps=1, burn_in_steps=1)
    target = next(sequence for sequence in sequences if sequence.learn_start_step == 5)

    # All historical non-combat decisions (0, 3), plus current-combat state 4.
    assert [step.step_index for step in target.burn_in] == [0, 3, 4]
    assert [step.step_index for step in target.learn_steps] == [5]
    assert target.exact_recurrent_reconstruction
    assert target.burn_in_no_grad


def test_ten_thousand_step_episode_only_expands_no_grad_prefix() -> None:
    snapshot = _snapshot(domain_id=0)
    episode = _linear_episode(
        snapshot,
        episode_id="ten-thousand",
        length=10_000,
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

    assert sequence.learn_start_step > 1_000
    assert len(sequence.burn_in) == sequence.learn_start_step
    assert sequence.burn_in_no_grad
    assert sequence.exact_recurrent_reconstruction
    # Full run-memory history affects only no-grad compute.  Graph-bearing
    # learning length remains the configured short suffix.
    assert len(sequence.learn_steps) == 32
    assert [step.step_index for step in sequence.learn_steps] == list(
        range(sequence.learn_start_step, sequence.learn_start_step + 32)
    )


def test_replay_state_dict_round_trips_rng_items_counters_and_rejects_atomically() -> None:
    snapshot = _snapshot()
    episodes = (
        _linear_episode(snapshot, episode_id="state-win", length=5, won=True),
        _linear_episode(snapshot, episode_id="state-loss", length=7, won=False),
        _linear_episode(snapshot, episode_id="state-censored", length=6, won=None),
    )
    total_bytes = sum(episode.storage_nbytes() for episode in episodes)

    def replay() -> BoundedEpisodicReplay:
        return BoundedEpisodicReplay(
            capacity=3,
            byte_capacity=total_bytes,
            episode_byte_capacity=max(episode.storage_nbytes() for episode in episodes),
            max_segments_per_episode=2,
            seed=73,
        )

    original = replay()
    assert all(original.put(episode) for episode in episodes)
    original.sample(
        2,
        learn_steps=2,
        burn_in_steps=1,
        macro_sample_fraction=0.5,
    )
    payload = original.state_dict()
    assert payload["version"] == "sts2-episodic-replay-v3"
    assert payload["sample_count"] == 2
    assert payload["macro_sample_count"] == 1
    assert set(payload) == {
        "version",
        "capacity",
        "byte_capacity",
        "episode_byte_capacity",
        "max_segments_per_episode",
        "items",
        "rng_state",
        "put_count",
        "sample_count",
        "macro_sample_count",
        "eviction_count",
        "duplicate_count",
        "oversize_count",
        "maximum_observed_episode_steps",
    }

    restored = replay()
    restored.load_state_dict(payload)
    assert restored.metrics() == original.metrics()
    assert [item.episode_id for item in restored.snapshot()] == [
        item.episode_id for item in original.snapshot()
    ]

    expected = original.sample(5, learn_steps=2, burn_in_steps=1)
    actual = restored.sample(5, learn_steps=2, burn_in_steps=1)

    def projection(sequence: ReplaySequence) -> object:
        return (
            sequence.episode_id,
            sequence.learn_start_step,
            tuple(step.step_index for step in sequence.burn_in),
            tuple(step.step_index for step in sequence.learn_steps),
        )
    assert [projection(sequence) for sequence in actual] == [
        projection(sequence) for sequence in expected
    ]

    expected_fresh = original.sample_for_learning(
        2,
        learn_steps=2,
        burn_in_steps=1,
        macro_sample_fraction=0.5,
        current_policy_version=7,
        policy_gradient_max_lag=128,
        fresh_policy_sequences=1,
    )
    actual_fresh = restored.sample_for_learning(
        2,
        learn_steps=2,
        burn_in_steps=1,
        macro_sample_fraction=0.5,
        current_policy_version=7,
        policy_gradient_max_lag=128,
        fresh_policy_sequences=1,
    )
    assert actual_fresh.diagnostics == expected_fresh.diagnostics
    assert [projection(sequence) for sequence in actual_fresh.sequences] == [
        projection(sequence) for sequence in expected_fresh.sequences
    ]
    assert restored.state_dict()["rng_state"] == original.state_dict()["rng_state"]

    before = restored.metrics()
    invalid = dict(payload)
    invalid["items"] = (episodes[0], episodes[0])
    with pytest.raises(ValueError, match="duplicate"):
        restored.load_state_dict(invalid)
    assert restored.metrics() == before

    extra_key = dict(payload)
    extra_key["unexpected"] = 1
    with pytest.raises(ValueError, match="schema"):
        restored.load_state_dict(extra_key)

    impossible_counter = dict(payload)
    sample_count = payload["sample_count"]
    assert isinstance(sample_count, int)
    impossible_counter["macro_sample_count"] = sample_count + 1
    with pytest.raises(ValueError, match="macro_sample_count"):
        restored.load_state_dict(impossible_counter)
    assert restored.metrics() == before


def test_replay_protocol4_payload_is_copied_and_refrozen_without_repairing_shapes() -> None:
    snapshot = _snapshot()
    episode = _linear_episode(snapshot, episode_id="protocol-four", length=3, won=True)
    size = episode.storage_nbytes()

    def replay() -> BoundedEpisodicReplay:
        return BoundedEpisodicReplay(
            capacity=1,
            byte_capacity=size,
            episode_byte_capacity=size,
            max_segments_per_episode=1,
            seed=19,
        )

    source = replay()
    assert source.put(episode)
    legacy = pickle.loads(pickle.dumps(source.state_dict(), protocol=4))
    legacy_snapshot = legacy["items"][0].steps[0].snapshot
    assert legacy_snapshot.world.ids.flags.writeable
    assert legacy_snapshot.action_mask.flags.writeable

    restored = replay()
    restored.load_state_dict(legacy)
    stored_snapshot = restored.snapshot()[0].steps[0].snapshot
    assert not stored_snapshot.world.ids.flags.writeable
    assert not stored_snapshot.action_mask.flags.writeable
    # Loading canonicalizes a detached copy; it does not mutate caller-owned
    # protocol-4 payloads merely to satisfy the replay invariant.
    assert legacy_snapshot.world.ids.flags.writeable
    assert legacy_snapshot.action_mask.flags.writeable

    malformed = pickle.loads(pickle.dumps(source.state_dict(), protocol=4))
    malformed_snapshot = malformed["items"][0].steps[0].snapshot
    object.__setattr__(
        malformed_snapshot,
        "action_mask",
        np.asarray([True], dtype=np.bool_),
    )
    before = restored.state_dict()
    with pytest.raises(ValueError, match="action mask length"):
        restored.load_state_dict(malformed)
    assert restored.state_dict() == before


def test_replay_rejects_hash_consistent_accounting_edits_atomically() -> None:
    snapshot = _snapshot()
    episode = _linear_episode(snapshot, episode_id="accounting", length=2, won=False)
    size = episode.storage_nbytes()
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=size,
        episode_byte_capacity=size,
        max_segments_per_episode=1,
        seed=23,
    )
    assert replay.put(episode)
    before = replay.state_dict()
    edited = dict(before)
    put_count = edited["put_count"]
    assert isinstance(put_count, int)
    edited["put_count"] = put_count + 1

    with pytest.raises(ValueError, match="put/eviction accounting"):
        replay.load_state_dict(edited)
    assert replay.state_dict() == before


def test_fresh_sampling_reuses_put_time_index_without_reclassifying_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A learner update must not scan every stored step for fresh candidates."""

    episode = _surface_episode(
        episode_id="indexed-fresh",
        surfaces=("event", "card_reward", "map"),
        won=True,
    )
    size = episode.storage_nbytes()
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=size,
        episode_byte_capacity=size,
        max_segments_per_episode=3,
        seed=29,
    )
    assert replay.put(episode)

    def unexpected_reclassification(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fresh sampling rescanned an indexed episode")

    monkeypatch.setattr(
        episode_replay_module,
        "_primary_policy_target",
        unexpected_reclassification,
    )
    sample = replay.sample_for_learning(
        1,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=0.0,
        current_policy_version=7,
        policy_gradient_max_lag=0,
        fresh_policy_sequences=1,
    )

    assert len(sample.sequences) == 1
    assert sample.sequences[0].episode_id == episode.episode_id
    assert sample.diagnostics.fresh_policy_candidate_decisions == 3
    assert sample.diagnostics.fresh_policy_quota_filled == 1


def test_indexed_fresh_sampling_preserves_legacy_step_order_for_version_runs() -> None:
    episode = _with_policy_versions(
        _surface_episode(
            episode_id="indexed-version-runs",
            surfaces=("event", "event", "event"),
            won=True,
        ),
        (7, 6, 7),
    )
    sampling_index = episode_replay_module._build_episode_sampling_index(
        episode,
        storage_nbytes=episode.storage_nbytes(),
    )

    for seed in range(16):
        expected_rng = np.random.default_rng(seed)
        # The legacy implementation consumed one draw to resolve the sole
        # surface tie, then indexed the eligible decisions in episode order.
        expected_rng.integers(0, 1)
        expected_step = int(expected_rng.integers(0, 3))
        actual, candidate_episodes, candidate_decisions = (
            episode_replay_module._fresh_policy_candidates(
                (episode,),
                (sampling_index,),
                episode_order=(0,),
                current_policy_version=7,
                maximum_policy_lag=1,
                rng=np.random.default_rng(seed),
            )
        )

        assert actual == ((0, expected_step, 7 - (7, 6, 7)[expected_step]),)
        assert candidate_episodes == 1
        assert candidate_decisions == 3


def test_concurrent_put_completes_while_sampling_materializes_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow learner sample must not retain the replay item/deque lock."""

    snapshot = _snapshot()
    first = _linear_episode(
        snapshot,
        episode_id="concurrent-first",
        length=4,
        won=True,
    )
    second = _linear_episode(
        snapshot,
        episode_id="concurrent-second",
        length=5,
        won=True,
    )
    byte_capacity = first.storage_nbytes() + second.storage_nbytes()
    replay = BoundedEpisodicReplay(
        capacity=2,
        byte_capacity=byte_capacity,
        episode_byte_capacity=max(
            first.storage_nbytes(),
            second.storage_nbytes(),
        ),
        max_segments_per_episode=1,
        seed=31,
    )
    assert replay.put(first)

    sampling_started = threading.Event()
    release_sampling = threading.Event()
    put_finished = threading.Event()
    errors: list[BaseException] = []
    original = episode_replay_module._fresh_policy_candidates

    def paused_fresh_candidates(*args: object, **kwargs: object) -> object:
        sampling_started.set()
        if not release_sampling.wait(timeout=5.0):
            raise TimeoutError("test did not release paused sampling")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        episode_replay_module,
        "_fresh_policy_candidates",
        paused_fresh_candidates,
    )

    def sample_worker() -> None:
        try:
            replay.sample_for_learning(
                1,
                learn_steps=1,
                burn_in_steps=0,
                macro_sample_fraction=0.0,
                current_policy_version=7,
                policy_gradient_max_lag=0,
                fresh_policy_sequences=1,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def put_worker() -> None:
        try:
            assert replay.put(second)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            put_finished.set()

    sampler = threading.Thread(target=sample_worker, daemon=True)
    putter = threading.Thread(target=put_worker, daemon=True)
    sampler.start()
    assert sampling_started.wait(timeout=2.0)
    putter.start()
    # Event ordering, rather than elapsed-duration comparison, proves that put
    # can acquire the deque lock while the sampler remains deliberately paused.
    assert put_finished.wait(timeout=2.0)
    assert sampler.is_alive()
    release_sampling.set()
    sampler.join(timeout=5.0)
    putter.join(timeout=5.0)

    assert not sampler.is_alive()
    assert not putter.is_alive()
    assert errors == []
    assert [item.episode_id for item in replay.snapshot()] == [
        first.episode_id,
        second.episode_id,
    ]


def test_sampling_index_tracks_eviction_duplicate_bytes_and_future_failure() -> None:
    snapshot = _snapshot()
    first = _linear_episode(
        snapshot,
        episode_id="indexed-evicted",
        length=2,
        won=True,
    )
    second = _linear_episode(
        snapshot,
        episode_id="indexed-retained",
        length=3,
        won=True,
    )
    capacity = max(first.storage_nbytes(), second.storage_nbytes())
    replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=capacity,
        episode_byte_capacity=capacity,
        max_segments_per_episode=1,
        seed=37,
    )
    assert replay.put(first)
    assert replay.put(second)
    assert not replay.put(second)

    sample = replay.sample_for_learning(
        1,
        learn_steps=1,
        burn_in_steps=0,
        macro_sample_fraction=0.0,
        current_policy_version=7,
        policy_gradient_max_lag=0,
        fresh_policy_sequences=1,
    )
    assert [sequence.episode_id for sequence in sample.sequences] == [second.episode_id]
    metrics = replay.metrics()
    assert metrics["size"] == 1
    assert metrics["storage_nbytes"] == second.storage_nbytes()
    assert metrics["put_count"] == 2
    assert metrics["eviction_count"] == 1
    assert metrics["duplicate_count"] == 1

    failed_future = _with_policy_versions(
        _linear_episode(
            snapshot,
            episode_id="indexed-failed-future",
            length=1,
            won=False,
        ),
        (8,),
    )
    failed_size = failed_future.storage_nbytes()
    future_replay = BoundedEpisodicReplay(
        capacity=1,
        byte_capacity=failed_size,
        episode_byte_capacity=failed_size,
        max_segments_per_episode=1,
        seed=41,
    )
    assert future_replay.put(failed_future)
    with pytest.raises(ValueError, match="newer than the learner"):
        future_replay.sample_for_learning(
            1,
            learn_steps=1,
            burn_in_steps=0,
            macro_sample_fraction=0.0,
            current_policy_version=7,
            policy_gradient_max_lag=128,
            fresh_policy_sequences=1,
        )
