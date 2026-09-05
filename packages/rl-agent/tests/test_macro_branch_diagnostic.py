from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from sts2_rl import macro_branch_diagnostic as branch  # noqa: E402
from sts2_rl.contracts import EnvironmentResult, StepRequest  # noqa: E402
from sts2_rl.macro_root_diagnostic import EvidenceTrace, ProbeLimits  # noqa: E402
from sts2_rl.semantics.clock import DECISION_CLOCK_BASE  # noqa: E402
from sts2_rl.training.config import load_training_config  # noqa: E402
from tests.test_macro_authority import _PICKER_ACTIONS, _REST_ACTIONS, _rest_observation  # noqa: E402
from tests.test_macro_root_diagnostic import (  # noqa: E402
    CountingPolicy,
    FakeProbeBackend,
    _choose,
    _reward_actions,
)
from tests.test_task_reward import _result  # noqa: E402


class BranchBackend(FakeProbeBackend):
    """Reward root, two combats separated by map nodes, then factual run end."""

    def __init__(
        self, *, replay_hp: int = 50, truncate_after: int | None = None, boundary_outcome_none: bool = False
    ) -> None:
        super().__init__()
        self.branch = 0
        self.replay_hp = replay_hp
        self.truncate_after = truncate_after
        self.boundary_outcome_none = boundary_outcome_none

    def _state(self, step: int, *, terminal: bool = False, truncated: bool = False) -> EnvironmentResult:
        terminal = terminal or step >= 7
        truncated = truncated or step == self.truncate_after
        combat = step in {1, 2, 4, 5}
        loss = float(10 * self.branch if step >= 2 else 0)
        previous_loss = float(10 * self.branch if step > 2 else 0)
        base = _result(
            step=step,
            floor=1 + int(step >= 1) + int(step >= 4),
            hp=self.replay_hp - int(loss),
            revival_budget=64,
            player_hp_lost=loss,
            player_hp_lost_delta=loss - previous_loss,
            terminated=terminal,
            truncated=truncated,
            combat_result="victory" if step in {3, 6} and not self.boundary_outcome_none else "none",
            run_result="victory" if terminal else "none",
        )
        actions: tuple[dict[str, Any], ...]
        if step == 0:
            actions = _reward_actions()
        elif combat:
            actions = ({"kind": "end_turn", "model_action_kind": "end_turn", "action_handle": "end-current"},)
        else:
            actions = (
                {
                    "kind": "choose_map_node",
                    "model_action_kind": "map",
                    "action_handle": "map-current",
                    "map_node": {"index": 0, "coord": {"x": 0, "y": step}, "point_type": "monster"},
                },
            )
        return replace(
            base,
            observation={
                **base.observation,
                "phase": "combat" if combat else "reward" if step == 0 else "map",
                "combat": {"in_progress": combat},
            },
            legal_actions=() if terminal or truncated else actions,
            info={"reward_authority": "external-rl"},
        )

    def step(self, request: StepRequest) -> EnvironmentResult:
        if not self.step_requests:
            self.branch = int(request.action_id == "skip-current")
        self.step_requests.append(request)
        return self._state(len(self.step_requests))


@pytest.fixture
def config() -> Any:
    path = Path(__file__).resolve().parents[1] / "config/experiments/hybrid_vtrace_v1_reward_v9_budget64_on_policy.toml"
    return load_training_config(profile="preheat", config_path=path)


def _run(
    tmp_path: Path, config: Any, backends: list[BranchBackend], limits: branch.BranchLimits | None = None
) -> dict[str, Any]:
    pending = iter(backends)
    return branch.run_branch_probe(
        backend_factory=lambda: next(pending),
        policy=CountingPolicy(),
        config=config,
        seed=17,
        limits=limits or branch.BranchLimits(continuation_macro_decisions=0),
        output_dir=tmp_path,
    )


def test_b1_runs_fresh_same_action_gate_then_all_candidates_and_compares_complete_group(
    tmp_path: Path,
    config: Any,
) -> None:
    backends = [BranchBackend() for _ in range(3)]
    result = _run(tmp_path, config, backends)
    assert result["status"] == "complete", result
    assert result["same_action_gate"] == "matched"
    assert len(result["passes"]) == 3 and result["complete_candidate_count"] == 2
    assert result["independent_root_samples"] == 1
    assert result["training_label_eligible"] is False and result["tail_value_used"] is False
    assert [backend.branch for backend in backends] == [0, 0, 1]
    assert all(backend.closed and len(backend.reset_requests) == 1 for backend in backends)
    assert result["comparisons"][0]["difference_from_greedy_control"] == 0
    assert result["comparisons"][1]["difference_from_greedy_control"] < 0
    assert all(item["ending"] == "macro_horizon" for item in result["comparisons"])
    assert result["control_scores"] == "policy_logits_not_Q"
    assert [item["policy_logit"] for item in result["comparisons"]] == [2.0, 0.0]
    assert all(item["root_policy_logits"] == [2.0, 0.0] for item in result["passes"])
    assert (tmp_path / "reference.jsonl").read_bytes() == (tmp_path / "same-action-control.jsonl").read_bytes()
    assert (tmp_path / "reference.prefix.jsonl").read_bytes() == (tmp_path / "candidate-001.prefix.jsonl").read_bytes()


def test_macro_horizon_finishes_combat_instead_of_counting_combat_decisions(tmp_path: Path, config: Any) -> None:
    result = _run(tmp_path, config, [BranchBackend() for _ in range(3)])
    assert result["status"] == "complete", result
    for outcome in result["passes"]:
        assert outcome["native_steps"] == 3
        assert outcome["continuation_macro_decisions"] == 0
        assert outcome["continuation_combat_decisions"] == 2
        assert outcome["post_root_combats_started"] == outcome["post_root_combats_completed"] == 1
        assert outcome["post_root_combats_exited"] == 1
        assert outcome["post_root_floor_span"] == 1
        assert outcome["tail_discount"] == pytest.approx(DECISION_CLOCK_BASE)


def test_observed_combat_exit_does_not_invent_authoritative_outcome(tmp_path: Path, config: Any) -> None:
    result = _run(tmp_path, config, [BranchBackend(boundary_outcome_none=True) for _ in range(3)])
    assert result["status"] == "complete", result
    assert "no victory/defeat inference" in result["combat_coverage_definitions"]["exited"]
    for outcome in result["passes"]:
        assert outcome["post_root_combats_started"] == 1
        assert outcome["post_root_combats_exited"] == 1
        assert outcome["post_root_combats_completed"] == 0


def test_one_following_macro_decision_includes_its_next_full_combat(tmp_path: Path, config: Any) -> None:
    result = _run(
        tmp_path, config, [BranchBackend() for _ in range(3)], branch.BranchLimits(continuation_macro_decisions=1)
    )
    assert result["status"] == "complete", result
    for outcome in result["passes"]:
        assert outcome["native_steps"] == 6
        assert outcome["continuation_macro_decisions"] == 1
        assert outcome["continuation_combat_decisions"] == 4
        assert outcome["post_root_combats_completed"] == 2
        assert outcome["tail_discount"] == pytest.approx(DECISION_CLOCK_BASE**2)


def test_terminal_only_reaches_factual_end_without_inventing_act_terminal(tmp_path: Path, config: Any) -> None:
    result = _run(tmp_path, config, [BranchBackend() for _ in range(3)], branch.BranchLimits(terminal_only=True))
    assert result["status"] == "complete", result
    assert all(
        item["ending"] == "factual_terminal" and item["native_steps"] == 7 and item["tail_discount"] == 0.0
        for item in result["passes"]
    )


def test_failed_same_action_control_prevents_any_alternative_candidate(tmp_path: Path, config: Any) -> None:
    backends = [BranchBackend(), BranchBackend(replay_hp=49), BranchBackend()]
    result = _run(tmp_path, config, backends)
    assert result["status"] == "mismatch" and result["same_action_gate"] == "mismatch"
    assert result["comparisons"] is None
    assert not backends[1].step_requests and not backends[2].reset_requests
    assert backends[0].closed and backends[1].closed


def test_each_new_candidate_must_verify_its_own_prefix_before_root_dispatch(tmp_path: Path, config: Any) -> None:
    backends = [BranchBackend(), BranchBackend(), BranchBackend(replay_hp=49)]
    result = _run(tmp_path, config, backends)
    assert result["status"] == "mismatch" and result["same_action_gate"] == "matched"
    assert result["comparisons"] is None and not backends[2].step_requests
    assert all(backend.closed for backend in backends)


@pytest.mark.parametrize(
    "limits",
    [
        branch.BranchLimits(max_native_steps=2),
        branch.BranchLimits(max_candidates=1),
        branch.BranchLimits(max_trace_bytes=16),
    ],
)
def test_administrative_limits_do_not_publish_partial_rankings(
    tmp_path: Path,
    config: Any,
    limits: branch.BranchLimits,
) -> None:
    backend = BranchBackend()
    result = _run(tmp_path, config, [backend], limits)
    assert result["status"] == "inconclusive", result
    assert result["comparisons"] is None and result["training_label_eligible"] is False
    assert backend.closed


def test_truncation_of_one_candidate_invalidates_entire_comparison_group(tmp_path: Path, config: Any) -> None:
    backends = [BranchBackend(), BranchBackend(), BranchBackend(truncate_after=2)]
    result = _run(tmp_path, config, backends)
    assert result["status"] == "inconclusive" and result["same_action_gate"] == "matched"
    assert result["comparisons"] is None and all(backend.closed for backend in backends)
    lines = [json.loads(line) for line in (tmp_path / "candidate-001.jsonl").read_text().splitlines()]
    truncated = next(row for row in lines if row["kind"] == "environment_truncated")
    assert truncated["facts"]["reward_not_evaluated"] is True


def test_snapshot_export_is_raw_observations_candidates_and_indices_not_saved_training_hidden(
    tmp_path: Path,
    config: Any,
) -> None:
    result = _run(tmp_path, config, [BranchBackend() for _ in range(3)])
    assert result["status"] == "complete", result
    rows = [json.loads(line) for line in (tmp_path / "candidate-001.jsonl").read_text().splitlines()]
    snapshots = [row["facts"] for row in rows if row["kind"] == "semantic_snapshot"]
    assert snapshots[0]["selected_index"] == 1 and snapshots[0]["is_root"]
    assert snapshots[0]["observation"]["run"]["floor"] == 1
    assert len(snapshots[0]["semantic_actions"]) == 2
    assert all("hidden" not in snapshot and "saved_hidden" not in snapshot for snapshot in snapshots)
    edges = [row["facts"] for row in rows if row["kind"] == "semantic_edge"]
    assert sum(edge["native_steps"] for edge in edges) == 3
    weighted = sum(edge["prefix_discount"] * edge["reward_components"]["reward"] for edge in edges)
    assert weighted == pytest.approx(result["comparisons"][1]["observed_discounted_return"])


def test_forced_smith_target_is_one_intervention_with_original_mechanical_suffix() -> None:
    authority = branch.BranchAuthority(CountingPolicy(), 17, branch.BranchLimits(root_surface="rest"), 1)
    authority.begin_episode("smith")
    assert _choose(authority, _rest_observation(), _REST_ACTIONS) == 1
    assert authority.greedy_root_index == 0 and authority.root_evidence["selected_index"] == 1
    assert _choose(authority, {"phase": "card_selection", "run": {"floor": 7}}, _PICKER_ACTIONS) == 0
    assert _choose(authority, {"phase": "card_selection", "run": {"floor": 7}}, _PICKER_ACTIONS) == 1
    assert authority.semantic_count == 1 and authority.mechanical_dispatches == 2


def test_semantic_return_aggregates_suffix_rewards_before_floor_discount() -> None:
    returns = branch._SemanticReturn()
    trace = EvidenceTrace(io.StringIO(), ProbeLimits())
    values = dict.fromkeys(branch.REWARD_COMPONENTS, 0.0)
    returns.open(1)
    returns.add({**values, "reward": 2.0}, floor_before=1, floor_after=1)
    returns.add({**values, "reward": 3.0}, floor_before=1, floor_after=2)
    returns.close(2, False, trace)
    returns.open(2)
    returns.add({**values, "reward": 7.0}, floor_before=2, floor_after=3, terminal=True)
    returns.close(3, True, trace)
    assert returns.discounted["reward"] == pytest.approx(5.0 + DECISION_CLOCK_BASE * 7.0)
    assert returns.weight == 0.0 and returns.undiscounted["reward"] == 12.0


def test_semantic_return_matches_native_oracle_when_suffix_crosses_floors_early() -> None:
    returns = branch._SemanticReturn()
    output = io.StringIO()
    trace = EvidenceTrace(output, ProbeLimits())
    values = dict.fromkeys(branch.REWARD_COMPONENTS, 0.0)
    returns.open(1)
    for before, after, reward in [(1, 2, 2.0), (2, 3, 3.0), (3, 3, 5.0)]:
        returns.add({**values, "reward": reward}, floor_before=before, floor_after=after)
    returns.close(3, False, trace)
    returns.open(3)
    returns.add({**values, "reward": 7.0}, floor_before=3, floor_after=4, terminal=True)
    returns.close(4, True, trace)
    oracle = 2.0 + DECISION_CLOCK_BASE * 3.0 + DECISION_CLOCK_BASE**2 * (5.0 + 7.0)
    assert returns.discounted["reward"] == pytest.approx(oracle)
    assert returns.undiscounted["reward"] == 17.0 and returns.weight == 0.0
    edges = [json.loads(row)["facts"] for row in output.getvalue().splitlines()]
    assert edges[0]["reward_components_semantics"] == "edge-local native-prefix-discounted"
    assert edges[0]["reward_components"]["reward"] == pytest.approx(
        2.0 + DECISION_CLOCK_BASE * 3.0 + DECISION_CLOCK_BASE**2 * 5.0
    )
    assert edges[0]["raw_reward_components"]["reward"] == 10.0
    assert edges[0]["discount"] == pytest.approx(DECISION_CLOCK_BASE**2)


def test_same_floor_mechanical_suffix_retains_raw_reward_sum() -> None:
    returns = branch._SemanticReturn()
    values = dict.fromkeys(branch.REWARD_COMPONENTS, 0.0)
    returns.open(9)
    for reward in [2.0, -1.0, 3.0]:
        returns.add({**values, "reward": reward}, floor_before=9, floor_after=9)
    returns.close(9, False, EvidenceTrace(io.StringIO(), ProbeLimits()))
    assert returns.discounted["reward"] == returns.undiscounted["reward"] == 4.0
    assert returns.weight == 1.0


@pytest.mark.parametrize("before,after", [(5, 4), (6, 7)])
def test_semantic_return_rejects_regressing_or_discontinuous_native_floor(before: int, after: int) -> None:
    returns = branch._SemanticReturn()
    returns.open(5)
    with pytest.raises(ValueError, match="continuous and nondecreasing"):
        returns.add(dict.fromkeys(branch.REWARD_COMPONENTS, 1.0), floor_before=before, floor_after=after)
    assert returns.undiscounted["reward"] == 0.0


def test_explicit_authority_is_used_without_rebuilding_and_reports_custom_policy_mode(
    config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CustomAuthority(branch.BranchAuthority):
        policy_mode = "test_custom_sampler"

    policy = CountingPolicy()
    limits = branch.BranchLimits(continuation_macro_decisions=0)
    authority = CustomAuthority(policy, 17, limits)

    def forbidden_rebuild(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an explicitly supplied authority must not be rebuilt")

    monkeypatch.setattr(branch, "BranchAuthority", forbidden_rebuild)
    backend = BranchBackend()
    try:
        result = branch.run_branch_pass(
            backend=backend,
            policy=policy,
            config=config,
            seed=17,
            limits=limits,
            trace=EvidenceTrace(io.StringIO(), limits.trace_limits()),
            prefix=EvidenceTrace(io.StringIO(), limits.trace_limits()),
            authority=authority,
        )
    finally:
        backend.close()
    assert result["policy_mode"] == "test_custom_sampler"
    assert result["root_found"] and authority.semantic_count == 3
    assert result["training_label_eligible"] is False


@pytest.mark.parametrize("mismatch", ["policy", "seed", "limits", "intervention"])
def test_explicit_authority_binding_mismatch_fails_before_reset(config: Any, mismatch: str) -> None:
    policy = CountingPolicy()
    limits = branch.BranchLimits(continuation_macro_decisions=0)
    authority = branch.BranchAuthority(
        CountingPolicy() if mismatch == "policy" else policy,
        19 if mismatch == "seed" else 17,
        replace(limits, continuation_macro_decisions=1) if mismatch == "limits" else limits,
        1 if mismatch == "intervention" else None,
    )
    backend = BranchBackend()
    with pytest.raises(ValueError, match="does not match"):
        branch.run_branch_pass(
            backend=backend,
            policy=policy,
            config=config,
            seed=17,
            limits=limits,
            trace=EvidenceTrace(io.StringIO(), limits.trace_limits()),
            prefix=EvidenceTrace(io.StringIO(), limits.trace_limits()),
            authority=authority,
        )
    assert not backend.reset_requests


@pytest.mark.parametrize("surface", branch.SURFACES)
def test_branch_limits_accept_all_supported_generic_surfaces(surface: str) -> None:
    assert branch.BranchLimits(root_surface=surface).root_surface == surface
