from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sts2_rl.joint import (  # noqa: E402
    DomainExplorationSchedule,
    FactualCoverageSchedule,
    HybridCollectionAuthority,
)
from tests.test_macro_authority import (  # noqa: E402
    _REST_ACTIONS,
    _rest_observation,
    _snapshot,
)


def _authority(*, evaluation: bool = False) -> HybridCollectionAuthority:
    def forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        return torch.tensor([0.0, 2.0])[: len(snapshot.action_mask)], hidden

    schedule = DomainExplorationSchedule(start=0.4, end=0.1, decay_decisions=10)
    return HybridCollectionAuthority(
        forward_q=forward,
        initial_state=lambda: None,
        epsilon=0.0,
        seed=4,
        evaluation_ownership=evaluation,
        macro_exploration=schedule,
        combat_exploration=schedule,
    )


def test_hybrid_authority_records_only_factual_terminal_runs() -> None:
    authority = _authority()
    authority.begin_episode("censored")
    authority.choose(
        observation=_rest_observation(),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    assert authority.finish_run() is None

    authority.begin_episode("terminal")
    authority.choose(
        observation=_rest_observation(),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    authority.observe_step(
        reward=1.0,
        revivals_delta=2,
        floor=7,
        terminal=True,
    )
    run = authority.finish_run(environment_steps=17)
    assert run is not None
    assert run.steps[-1].terminal is True
    assert run.steps[-1].revivals_delta == 2
    assert run.environment_steps == 17


def test_infrastructure_discard_cannot_leak_a_partial_run() -> None:
    authority = _authority()
    authority.begin_episode("interrupted")
    authority.choose(
        observation=_rest_observation(),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    authority.observe_step(reward=0.25, floor=7, terminal=False)
    authority.discard_run()

    authority.begin_episode("replacement-session")
    authority.choose(
        observation=_rest_observation(),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    authority.observe_step(reward=1.0, floor=8, terminal=True)
    run = authority.finish_run(environment_steps=9)

    assert run is not None
    assert run.run_id == "replacement-session"
    assert len(run.steps) == 1
    assert run.steps[0].reward == 1.0


def test_domain_clocks_roundtrip_and_evaluation_forces_zero_epsilon() -> None:
    authority = _authority()
    authority.load_exploration_state_dict({"domain_decisions": {"macro": 5, "combat": 8}, "policy_version": 12})
    assert authority.exploration_state_dict() == {
        "domain_decisions": {"macro": 5, "combat": 8},
        "policy_version": 12,
    }
    evaluation = _authority(evaluation=True)
    evaluation.begin_episode("held-out")
    evaluation.choose(
        observation=_rest_observation(),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    assert evaluation.epsilon == 0.0


def test_sparse_multibranch_macro_surface_uses_collection_only_coverage_floor() -> None:
    authority = _authority()
    authority.begin_episode("coverage")
    authority.choose(
        observation=_rest_observation(),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    assert authority.epsilon == pytest.approx(0.50)
    coverage = authority.metrics()["factual_coverage"]
    assert coverage["eligible_decisions"] == 1
    assert sum(coverage["branch_counts"].values()) == 1

    evaluation = _authority(evaluation=True)
    evaluation.begin_episode("heldout-coverage")
    evaluation.choose(
        observation=_rest_observation(),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    assert evaluation.epsilon == 0.0
    assert evaluation.metrics()["factual_coverage"]["eligible_decisions"] == 0


def test_factual_coverage_collects_the_non_greedy_rest_branch() -> None:
    authority = _authority()
    selected: list[int] = []
    for episode in range(400):
        authority.begin_episode(f"coverage-{episode}")
        chosen = authority.choose(
            observation=_rest_observation(),
            semantic_actions=_REST_ACTIONS,
            snapshot=_snapshot(candidate_count=2),
            valid=np.ones(2, dtype=np.bool_),
        )
        assert chosen is not None
        selected.append(chosen)
    # Greedy always selects native index 1.  A 0.50 branch-balanced coverage
    # floor should supply about 25% factual index-0 trials without preferring
    # it in the learned policy or in held-out evaluation.
    non_greedy = selected.count(0) / len(selected)
    assert 0.18 <= non_greedy <= 0.32


def test_hybrid_authority_publishes_candidate_q_for_exclusive_collector() -> None:
    authority = _authority()
    authority.begin_episode("q")
    selected = authority.choose(
        observation=_rest_observation(),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    assert selected in {0, 1}
    assert authority.last_q_values is not None
    np.testing.assert_allclose(authority.last_q_values, np.asarray([0.0, 2.0]))
    assert authority.last_native_q_values is not None
    np.testing.assert_allclose(
        authority.last_native_q_values,
        np.asarray([0.0, 2.0]),
    )


def test_hybrid_authority_reports_nonfinite_control_scores_at_the_source() -> None:
    authority = HybridCollectionAuthority(
        forward_q=lambda snapshot, hidden: (
            torch.tensor([0.0, float("nan")])[: len(snapshot.action_mask)],
            hidden,
        ),
        initial_state=lambda: None,
        epsilon=0.0,
        evaluation_ownership=True,
    )
    authority.begin_episode("nonfinite")
    with pytest.raises(
        FloatingPointError,
        match=r"non-finite control scores: finite=1/2",
    ):
        authority.choose(
            observation=_rest_observation(),
            semantic_actions=_REST_ACTIONS,
            snapshot=_snapshot(candidate_count=2),
            valid=np.ones(2, dtype=np.bool_),
        )


def test_hybrid_authority_retries_transient_forward_without_poisoning_recurrence() -> None:
    initial = torch.tensor([[3.0]])
    calls: list[torch.Tensor] = []

    def transient_forward(snapshot: Any, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(hidden.clone())
        if len(calls) == 1:
            return (
                torch.full((len(snapshot.action_mask),), float("nan")),
                torch.full_like(hidden, float("nan")),
            )
        return torch.tensor([0.0, 2.0])[: len(snapshot.action_mask)], hidden + 1.0

    authority = HybridCollectionAuthority(
        forward_q=transient_forward,
        initial_state=lambda: initial.clone(),
        epsilon=0.0,
        evaluation_ownership=True,
    )
    authority.begin_episode("transient-forward")
    selected = authority.choose(
        observation=_rest_observation(),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )

    assert selected == 1
    assert len(calls) == 2
    torch.testing.assert_close(calls[0], initial)
    torch.testing.assert_close(calls[1], initial)
    torch.testing.assert_close(authority._hidden, initial + 1.0)
    assert authority.metrics()["forward_retries"] == 1


def test_semantic_targets_map_back_to_one_native_entry() -> None:
    deck = [
        {
            "id": f"CARD.TEST_{index}",
            "instance_id": f"CARD.TEST_{index}-1",
            "index": index,
            "is_upgradable": True,
            "is_removable": True,
        }
        for index in range(2)
    ]

    def forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        assert len(snapshot.action_mask) == 3  # rest + two smith targets
        return torch.tensor([0.0, 1.0, 2.0]), hidden

    authority = HybridCollectionAuthority(
        forward_q=forward,
        initial_state=lambda: None,
        epsilon=0.0,
        macro_exploration=DomainExplorationSchedule(0.0, 0.0, 1),
        combat_exploration=DomainExplorationSchedule(0.0, 0.0, 1),
        factual_coverage=FactualCoverageSchedule(0.0, 0.0, 1),
    )
    authority.begin_episode("native-map")
    assert (
        authority.choose(
            observation=_rest_observation(deck=deck),
            semantic_actions=_REST_ACTIONS,
            snapshot=_snapshot(candidate_count=2),
            valid=np.ones(2, dtype=np.bool_),
        )
        == 1
    )
    assert authority.last_native_q_values is not None
    np.testing.assert_allclose(
        authority.last_native_q_values,
        np.asarray([0.0, 2.0]),
    )


def test_publication_reload_does_not_rewind_domain_clocks() -> None:
    authority = _authority()
    authority.domain_decisions.update({"macro": 11, "combat": 13})
    authority.policy_version = 9
    authority.load_exploration_state_dict({"domain_decisions": {"macro": 5, "combat": 8}, "policy_version": 4})
    assert authority.exploration_state_dict() == {
        "domain_decisions": {"macro": 11, "combat": 13},
        "policy_version": 9,
    }
