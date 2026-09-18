from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from sts2_rl.joint.macro_credit_collection import (  # noqa: E402
    COMPONENTS,
    StochasticBranchAuthority,
    TrainingEvidenceTrace,
    collect_macro_credit_root,
)
from sts2_rl.joint.macro_credit_data import (  # noqa: E402
    MacroCreditCohort,
    snapshot_equal,
    training_seed,
    trajectory_storage_nbytes,
)
from sts2_rl.macro_branch_diagnostic import BranchLimits  # noqa: E402
from sts2_rl.macro_root_diagnostic import ProbeBudgetExhausted, ProbeMismatch  # noqa: E402
from sts2_rl.semantics.clock import DECISION_CLOCK_BASE, decision_discount  # noqa: E402
from sts2_rl.training.config import load_training_config  # noqa: E402
from tests.test_macro_branch_diagnostic import BranchBackend  # noqa: E402
from tests.test_macro_root_diagnostic import CountingPolicy, _choose, _reward_actions  # noqa: E402


@pytest.fixture
def config() -> Any:
    path = Path(__file__).resolve().parents[1] / "config/experiments/hybrid_vtrace_v1_reward_v9_budget64_on_policy.toml"
    return load_training_config(profile="preheat", config_path=path)


def collect(tmp_path: Path, config: Any, *, backends: list[Any] | None = None, **kwargs: Any) -> Any:
    pending = iter(backends or [BranchBackend()])
    return collect_macro_credit_root(
        backend_factory=lambda: next(pending),
        policy=CountingPolicy(),
        config=config,
        seed=13400000,
        teacher_version=0,
        limits=kwargs.pop("limits", BranchLimits(continuation_macro_decisions=0)),
        output_dir=tmp_path / "root",
        artifact_root=tmp_path,
        **kwargs,
    )


def test_real_protocol_collects_stochastic_complete_group_and_actual_bootstrap(tmp_path: Path, config: Any) -> None:
    backends = [BranchBackend()]
    root, report = collect(tmp_path, config, backends=backends)
    assert report["status"] == "complete" and report["same_action_gate"] == "matched"
    assert all(backend.closed for backend in backends)
    assert len(root.candidates) == 2 and root.root_step_index == 0
    assert len(root.reference.steps) == 3
    assert [step.control_domain for step in root.reference.steps] == ["macro", "combat", "combat"]
    assert root.reference.bootstrap_snapshot is not None and root.reference.bootstrap_domain == "macro"
    assert root.reference.ending == "macro_horizon" and not root.reference.steps[-1].terminal
    natural = root.reference.steps[0].action_index
    assert root.teacher_log_probabilities == pytest.approx((-math.log1p(math.exp(-2)), -2 - math.log1p(math.exp(-2))))
    for index, candidate in enumerate(root.candidates):
        assert candidate.root_intervened == (index != natural)
        assert candidate.steps[0].behavior_log_probability == (
            0.0 if index != natural else root.teacher_log_probabilities[index]
        )
        assert candidate.steps[0].snapshot is root.reference.steps[0].snapshot
        assert candidate.data_partition == "training" and candidate.continuation_policy == "frozen_stochastic"
    assert (tmp_path / "root/reference.jsonl").is_file()
    assert not (tmp_path / "root/same-action-control.jsonl").exists()
    assert not (tmp_path / "root/candidate-001.jsonl").exists()
    cohort = MacroCreditCohort("cohort-0", 0, (root,))
    assert cohort.storage_nbytes() > 0
    assert all(item["policy_mode"] == "frozen_stochastic" for item in report["passes"])


def test_factual_terminal_has_no_bootstrap(tmp_path: Path, config: Any) -> None:
    root, _ = collect(tmp_path, config, limits=BranchLimits(terminal_only=True))
    assert root.reference.ending == "factual_terminal" and root.reference.bootstrap_snapshot is None
    assert all(candidate.steps[-1].terminal and candidate.steps[-1].discount == 0 for candidate in root.candidates)


def test_later_root_retains_and_reuses_exact_cpu_prefix(tmp_path: Path, config: Any) -> None:
    class LaterRootBackend(BranchBackend):
        def _state(self, step: int, **kwargs: Any) -> Any:
            result = super()._state(step, **kwargs)
            if step == 3:
                result = replace(
                    result, observation={**result.observation, "phase": "reward"}, legal_actions=_reward_actions()
                )
            return result

    root, _ = collect(
        tmp_path,
        config,
        backends=[LaterRootBackend()],
        limits=BranchLimits(root_occurrence=2, continuation_macro_decisions=0),
    )
    assert root.root_step_index == 3 and root.reference.prefix == ()
    for candidate in root.candidates:
        assert len(candidate.prefix) == 3
        for actual, expected in zip(candidate.prefix, root.reference.steps[:3], strict=True):
            assert actual is expected.snapshot


@pytest.mark.parametrize("seed", [1, 13400001, -2, 2**31, True, 2.0])
def test_training_partition_rejects_invalid_or_heldout_seed(seed: Any) -> None:
    with pytest.raises(ValueError):
        training_seed(seed)


def test_snapshot_and_external_path_limits_fail_closed(tmp_path: Path, config: Any) -> None:
    with pytest.raises(ProbeBudgetExhausted, match="snapshot"):
        collect(tmp_path, config, maximum_snapshot_bytes=1)
    with pytest.raises(ValueError, match="artifact root"):
        collect_macro_credit_root(
            backend_factory=BranchBackend,
            policy=CountingPolicy(),
            config=config,
            seed=2,
            teacher_version=0,
            limits=BranchLimits(),
            output_dir=tmp_path.parent / "not-in-root",
            artifact_root=tmp_path,
        )


def test_resource_guard_runs_inside_a_pass_and_closes_backend(tmp_path: Path, config: Any) -> None:
    backend = BranchBackend()
    count = 0

    def guard() -> None:
        nonlocal count
        count += 1
        if count >= 6:
            raise ProbeBudgetExhausted("resource_guard")

    with pytest.raises(ProbeBudgetExhausted, match="resource_guard"):
        collect(tmp_path, config, backends=[backend], resource_guard=guard)
    assert count == 6 and backend.closed



def test_data_refuses_diagnostic_and_false_prior_and_replaced_reference(tmp_path: Path, config: Any) -> None:
    root, _ = collect(tmp_path, config)
    with pytest.raises(ValueError, match="training-partition"):
        replace(root.reference, continuation_policy="greedy")
    with pytest.raises(ValueError, match="prior differs"):
        replace(root, teacher_log_probabilities=(math.log(0.5), math.log(0.5)))
    with pytest.raises(ValueError, match="partial"):
        replace(root, candidates=root.candidates[:1])
    index = root.reference.steps[0].action_index
    candidate = root.candidates[index]
    steps = (*candidate.steps[:-1], replace(candidate.steps[-1], behavior_log_probability=-0.1))
    candidates = list(root.candidates)
    candidates[index] = replace(candidate, steps=steps)
    with pytest.raises(ValueError, match="reference suffix steps"):
        replace(root, candidates=tuple(candidates))
    with pytest.raises(ValueError, match="mixes teacher"):
        MacroCreditCohort("cohort", 1, (root,))


def test_sampling_is_not_greedy_and_intervention_burns_same_root_rng_draw(config: Any) -> None:
    seen: set[int] = set()
    for seed in range(0, 60, 2):
        authority = StochasticBranchAuthority(
            CountingPolicy(), seed, BranchLimits(), None, teacher_version=0, config=config
        )
        authority.begin_episode("test")
        index = _choose(authority, {"phase": "reward", "run": {"floor": 1}}, _reward_actions())
        seen.add(index)
        forced = StochasticBranchAuthority(
            CountingPolicy(), seed, BranchLimits(), 1 - index, teacher_version=0, config=config
        )
        forced.begin_episode("forced")
        _choose(forced, {"phase": "reward", "run": {"floor": 1}}, _reward_actions())
        assert forced._sampling_rng.random() == authority._sampling_rng.random()
    assert seen == {0, 1}


def test_training_edge_folds_native_reward_prefix_discount_not_bare_sum(config: Any) -> None:
    authority = StochasticBranchAuthority(CountingPolicy(), 2, BranchLimits(), None, teacher_version=0, config=config)
    authority.begin_episode("oracle")
    _choose(authority, {"phase": "reward", "run": {"floor": 1}}, _reward_actions())
    trace = TrainingEvidenceTrace(
        io.StringIO(), authority=authority, limits=BranchLimits(), maximum_snapshot_bytes=10**8
    )

    def state(floor: int) -> dict[str, Any]:
        return {"observation": {"run": {"floor": floor}}}

    trace.record("decision", {"semantic": authority.event, "state": state(1)})
    for before, after, reward in [(1, 2, 2.0), (2, 3, 3.0), (3, 3, 5.0)]:
        trace.record(
            "transition",
            {
                "state": state(after),
                "reward_breakdown": {
                    **dict.fromkeys(COMPONENTS, 0.0),
                    "progress_reward": reward,
                    "revivals_used_delta": 0,
                    "player_hp_lost_delta": 2,
                    "task_terminal": False,
                },
                "native_discount": {
                    "floor_before": before,
                    "floor_after": after,
                    "terminal": False,
                    "discount": decision_discount(floor_before=before, floor_after=after, terminal=False),
                },
            },
        )
    trace._close_edge(3, False)
    assert trace.steps[0].reward == pytest.approx(2 + DECISION_CLOCK_BASE * 3 + DECISION_CLOCK_BASE**2 * 5)
    assert trace.steps[0].discount == pytest.approx(DECISION_CLOCK_BASE**2)
    assert trace.reward_facts[0].player_hp_lost_delta == 6
    assert snapshot_equal(trace.steps[0].snapshot, authority.current_snapshot)
