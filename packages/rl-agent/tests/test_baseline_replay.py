from __future__ import annotations

import pickle
from collections import Counter

import numpy as np
import pytest

from sts2_baseline import (
    BaselineTargets,
    BaselineTransition,
    PotentialState,
    ReplayMix,
    ReplaySample,
    ReplayStratum,
    StratifiedReplayBuffer,
)


def _sample(*, episode: str, stratum: ReplayStratum) -> ReplaySample:
    transition = BaselineTransition(
        episode_id=episode,
        step_index=0,
        action_handle=f"action:{episode}",
        before=PotentialState(player_hp=80, player_max_hp=80),
        after=PotentialState(player_hp=80, player_max_hp=80),
    )
    return ReplaySample(
        transition=transition,
        targets=BaselineTargets(reward=0.0),
        stratum=stratum,
    )


def test_priority_refresh_changes_per_distribution_and_batch_returns_stable_indices() -> None:
    buffer = StratifiedReplayBuffer(
        16,
        mix=ReplayMix(coverage=0.0, recent=0.0, per=1.0),
        alpha=1.0,
        beta=1.0,
        seed=7,
    )
    stratum = ReplayStratum(domain="combat", tier="normal", encounter_id="same")
    first = buffer.add(_sample(episode="first", stratum=stratum), priority=1.0)
    second = buffer.add(_sample(episode="second", stratum=stratum), priority=1.0)

    before = buffer.source_probabilities("per")
    assert before[first] == pytest.approx(0.5)
    assert before[second] == pytest.approx(0.5)

    buffer.update_priorities([first, second], [99.0, 1.0])
    after = buffer.source_probabilities("per")
    assert after[first] == pytest.approx(0.99)
    assert after[second] == pytest.approx(0.01)

    batch = buffer.sample(2_048)
    assert set(batch.indices.tolist()) <= {first, second}
    assert Counter(batch.indices.tolist())[first] > Counter(batch.indices.tolist())[second]
    high_probability_weights = batch.importance_weights[batch.indices == first]
    low_probability_weights = batch.importance_weights[batch.indices == second]
    assert high_probability_weights.size > 0
    assert low_probability_weights.size > 0
    assert float(high_probability_weights.max()) < float(low_probability_weights.min())


def test_coverage_source_balances_strata_not_raw_bucket_sizes() -> None:
    buffer = StratifiedReplayBuffer(
        256,
        mix=ReplayMix(coverage=1.0, recent=0.0, per=0.0),
        seed=11,
    )
    strata = [
        ReplayStratum(domain="combat", tier="weak", encounter_id="weak"),
        ReplayStratum(domain="combat", tier="normal", encounter_id="normal"),
        ReplayStratum(domain="combat", tier="elite", encounter_id="elite"),
        ReplayStratum(domain="combat", tier="boss", encounter_id="boss"),
    ]
    for index in range(100):
        buffer.add(_sample(episode=f"weak-{index}", stratum=strata[0]))
    rare_ids = [
        buffer.add(_sample(episode=f"rare-{index}", stratum=stratum))
        for index, stratum in enumerate(strata[1:], start=1)
    ]

    batch = buffer.sample(4)
    sampled_strata = {sample.stratum for sample in batch.samples}

    assert sampled_strata == set(strata)
    probabilities = buffer.source_probabilities("coverage")
    assert all(probabilities[replay_id] == pytest.approx(0.25) for replay_id in rare_ids)


def test_recent_source_only_uses_recent_window_and_source_draws_are_multinomial() -> None:
    buffer = StratifiedReplayBuffer(
        32,
        recent_window=4,
        mix=ReplayMix(coverage=0.5, recent=0.25, per=0.25),
        seed=3,
    )
    stratum = ReplayStratum(domain="combat", tier="normal", encounter_id="normal")
    ids = [buffer.add(_sample(episode=f"episode-{index}", stratum=stratum)) for index in range(10)]

    recent_probabilities = buffer.source_probabilities("recent")
    assert set(recent_probabilities) == set(ids[-4:])

    batch = buffer.sample(8_000)
    source_counts = Counter(batch.sources)
    assert source_counts["coverage"] / len(batch.sources) == pytest.approx(0.50, abs=0.025)
    assert source_counts["recent"] / len(batch.sources) == pytest.approx(0.25, abs=0.025)
    assert source_counts["per"] / len(batch.sources) == pytest.approx(0.25, abs=0.025)
    assert np.all(batch.importance_weights > 0.0)
    assert np.all(batch.importance_weights <= 1.0)

    # Small batches no longer deterministically starve PER.  Source choices are
    # independent across calls instead of repeating a largest-remainder quota.
    small_sources = Counter(
        source
        for _ in range(256)
        for source in buffer.sample(2).sources
    )
    assert small_sources["per"] > 0
    assert small_sources["recent"] > 0


def test_batch_probabilities_and_is_weights_use_exact_marginal_distribution() -> None:
    buffer = StratifiedReplayBuffer(
        16,
        recent_window=1,
        mix=ReplayMix(coverage=0.5, recent=0.25, per=0.25),
        alpha=0.6,
        beta=0.7,
        seed=19,
    )
    stratum = ReplayStratum(domain="combat", tier="normal", encounter_id="same")
    ids = [
        buffer.add(
            _sample(episode=f"episode-{index}", stratum=stratum),
            priority=priority,
        )
        for index, priority in enumerate((1_000.0, 1.0, 1.0))
    ]

    marginal = buffer.mixture_probabilities()
    assert marginal[ids[0]] != pytest.approx(1.0 / 3.0)
    batch = buffer.sample(20_000)
    empirical = Counter(batch.indices.tolist())
    for replay_id in ids:
        assert empirical[replay_id] / len(batch.samples) == pytest.approx(
            marginal[replay_id],
            abs=0.02,
        )

    support_minimum = min(probability for probability in marginal.values() if probability > 0)
    normalizer = (len(buffer) * support_minimum) ** (-buffer.beta)
    for replay_id, probability, weight in zip(
        batch.indices,
        batch.probabilities,
        batch.importance_weights,
        strict=True,
    ):
        assert probability == pytest.approx(marginal[int(replay_id)])
        expected = (len(buffer) * float(probability)) ** (-buffer.beta) / normalizer
        assert float(weight) == pytest.approx(expected, rel=1e-5)


def test_replay_sample_is_pickleable_for_checkpoint_storage() -> None:
    sample = _sample(
        episode="pickle-roundtrip",
        stratum=ReplayStratum(domain="combat", tier="elite", encounter_id="guardian"),
    )

    restored = pickle.loads(pickle.dumps(sample))

    assert restored == sample
    assert restored.transition.metadata == {}


def test_transition_metadata_and_replay_payload_own_nested_copies() -> None:
    source_metadata = {"audit": {"tags": ["original"]}}
    transition = BaselineTransition(
        episode_id="owned-nested-state",
        step_index=0,
        action_handle="action:owned",
        before=PotentialState(player_hp=80, player_max_hp=80),
        after=PotentialState(player_hp=80, player_max_hp=80),
        metadata=source_metadata,
    )
    source_metadata["audit"]["tags"].append("caller-mutation")
    assert transition.metadata["audit"]["tags"] == ["original"]

    payload = {"nested": {"values": [1]}}
    sample = ReplaySample(
        transition=transition,
        targets=BaselineTargets(reward=0.0),
        stratum=ReplayStratum(domain="combat"),
        payload=payload,
    )
    buffer = StratifiedReplayBuffer(4, seed=9)
    buffer.add(sample)

    payload["nested"]["values"].append(2)
    transition.metadata["audit"]["tags"].append("post-add-mutation")
    stored = buffer.sample(1).samples[0]
    assert stored.payload == {"nested": {"values": [1]}}
    assert stored.transition.metadata["audit"]["tags"] == ["original"]
