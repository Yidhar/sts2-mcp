from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
import torch

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
            ids=((2, 2, 2, 2, 2, 2, 0),),
            feature_dim=feature_dim,
            id_width=7,
        ),
        candidates=sparse_token_table(
            features=(action_a, action_b),
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
) -> EpisodeDecisionStep:
    return EpisodeDecisionStep(
        snapshot=snapshot,
        step_index=index,
        action_index=index % 2,
        behavior_log_probability=-0.5,
        policy_decision=True,
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
    original.sample(2, learn_steps=2, burn_in_steps=1)
    payload = original.state_dict()
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
    edited["put_count"] = int(edited["put_count"]) + 1

    with pytest.raises(ValueError, match="put/eviction accounting"):
        replay.load_state_dict(edited)
    assert replay.state_dict() == before
