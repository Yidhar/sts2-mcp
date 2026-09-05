from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sts2_rl import macro_root_diagnostic as probe  # noqa: E402
from sts2_rl.contracts import EnvironmentResult, ResetRequest, StepRequest  # noqa: E402
from sts2_rl.encoding import GroundedObservationEncoder  # noqa: E402
from sts2_rl.training.config import load_training_config  # noqa: E402
from tests.test_macro_authority import (  # noqa: E402
    _PICKER_ACTIONS,
    _REST_ACTIONS,
    _rest_observation,
    _snapshot,
)
from tests.test_task_reward import _result  # noqa: E402


class CountingPolicy:
    def __init__(self, selected: int = 0) -> None:
        self.selected = selected
        self.calls: list[torch.Tensor] = []

    def initial_state(self) -> torch.Tensor:
        return torch.zeros((1, 2))

    def forward(self, snapshot: Any, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        before = self.initial_state() if hidden is None else hidden
        self.calls.append(before.clone())
        logits = torch.zeros(len(snapshot.action_mask))
        logits[min(self.selected, len(logits) - 1)] = 2.0
        return logits, before + 1.0


def _reward_actions() -> tuple[dict[str, Any], ...]:
    return (
        {
            "kind": "select_card_reward",
            "model_action_kind": "card_reward",
            "idx": 0,
            "action_handle": "take-current",
            "card": {"id": "CARD.HAVOC", "cost": 1},
        },
        {"kind": "skip_card_reward", "model_action_kind": "card_reward", "idx": 1,
         "action_handle": "skip-current"},
    )


def _choose(authority: probe.ProbeAuthority, observation: dict[str, Any], actions: Any) -> int | None:
    authority.event = None
    authority.policy_event = None
    return authority.choose(
        observation=observation,
        semantic_actions=actions,
        snapshot=_snapshot(candidate_count=len(actions)),
        valid=np.ones(len(actions), dtype=np.bool_),
    )


def test_probe_authority_counts_reward_root_and_stops_before_following_dispatch() -> None:
    policy = CountingPolicy()
    authority = probe.ProbeAuthority(policy, 17, probe.ProbeLimits(continuation_decisions=0))
    authority.begin_episode("reference")
    assert _choose(authority, {"phase": "reward", "run": {"floor": 1}}, _reward_actions()) == 0
    assert authority.root_ordinal == 1
    assert authority.root_evidence is not None and authority.root_evidence["is_root"]
    assert authority.semantic_count == 1
    assert len(policy.calls) == 1
    with pytest.raises(probe._FrontierReached):
        _choose(authority, _rest_observation(), _REST_ACTIONS)
    assert authority.semantic_count == 2
    assert len(policy.calls) == 2  # Frontier is observed/scored, not dispatched.
    assert authority._open is None
    assert authority._hybrid_steps == []


def test_probe_authority_waits_for_requested_reward_occurrence() -> None:
    authority = probe.ProbeAuthority(CountingPolicy(), 17, probe.ProbeLimits(root_occurrence=2))
    authority.begin_episode("reference")
    for expected in (1, 2):
        assert _choose(authority, {"phase": "reward", "run": {"floor": expected}}, _reward_actions()) == 0
        assert authority.reward_roots_seen == expected
        assert (authority.root_ordinal is not None) == (expected == 2)


def test_probe_authority_mechanical_suffix_does_not_advance_hidden_or_semantic_clock() -> None:
    policy = CountingPolicy(selected=1)
    authority = probe.ProbeAuthority(policy, 17, probe.ProbeLimits())
    authority.begin_episode("smith")
    assert _choose(authority, _rest_observation(), _REST_ACTIONS) == 1
    hidden_after_entry = authority._hidden.clone()
    assert authority.semantic_count == 1
    assert _choose(authority, {"phase": "card_selection", "run": {"floor": 7}}, _PICKER_ACTIONS) == 0
    assert authority.event is None and authority.policy_event is None
    assert _choose(authority, {"phase": "card_selection", "run": {"floor": 7}}, _PICKER_ACTIONS) == 1
    assert authority.semantic_count == 1
    assert authority.reward_roots_seen == 0
    assert len(policy.calls) == 1
    assert authority.mechanical_dispatches == 2
    torch.testing.assert_close(authority._hidden, hidden_after_entry, rtol=0, atol=0)
    assert authority._hybrid_steps == [] and authority._open is None


def _reference_row(facts: Any, policy: Any = None) -> io.StringIO:
    output = io.StringIO()
    probe.EvidenceTrace(output, probe.ProbeLimits()).record("decision", facts, policy)
    output.seek(0)
    return output


def test_evidence_trace_requires_exact_facts_but_tolerates_policy_roundoff() -> None:
    reference = _reference_row({"hp": 30.0}, {"logits": [1.0], "hidden": [0.5]})
    actual = probe.EvidenceTrace(io.StringIO(), probe.ProbeLimits(), reference)
    actual.record("decision", {"hp": 30.0}, {"logits": [1.0 + 1e-7], "hidden": [0.5 + 1e-7]})
    actual.finish()
    reference = _reference_row({"hp": 30.0}, {"logits": [1.0]})
    with pytest.raises(probe.ProbeMismatch) as error:
        probe.EvidenceTrace(io.StringIO(), probe.ProbeLimits(), reference).record(
            "decision", {"hp": 30.0 + 1e-7}, {"logits": [1.0]}
        )
    assert error.value.detail["section"] == "facts"
    assert error.value.detail["path"] == '$["hp"]'


def test_evidence_trace_policy_tolerance_does_not_hide_shape_or_large_value_changes() -> None:
    for actual in ({"shape": [2], "values": [1.0]}, {"shape": [1], "values": [1.1]}):
        reference = _reference_row({}, {"shape": [1], "values": [1.0]})
        with pytest.raises(probe.ProbeMismatch) as error:
            probe.EvidenceTrace(io.StringIO(), probe.ProbeLimits(), reference).record("decision", {}, actual)
        assert error.value.detail["section"] == "policy"


def test_evidence_trace_detects_missing_extra_and_mismatching_rows() -> None:
    reference = _reference_row({"hp": 30})
    trace = probe.EvidenceTrace(io.StringIO(), probe.ProbeLimits(), reference)
    with pytest.raises(probe.ProbeMismatch, match="missing_expected_row"):
        trace.finish()
    with pytest.raises(probe.ProbeMismatch, match="unexpected_extra_row"):
        probe.EvidenceTrace(io.StringIO(), probe.ProbeLimits(), io.StringIO()).record("decision", {})
    output = io.StringIO()
    with pytest.raises(probe.ProbeMismatch):
        probe.EvidenceTrace(output, probe.ProbeLimits(), _reference_row({"hp": 30})).record(
            "decision", {"hp": 29}
        )
    assert json.loads(output.getvalue())["facts"]["hp"] == 29


def test_evidence_trace_caps_utf8_bytes_before_write() -> None:
    output = io.StringIO()
    trace = probe.EvidenceTrace(output, probe.ProbeLimits(max_trace_bytes=16))
    with pytest.raises(probe.ProbeBudgetExhausted, match="max_trace_bytes"):
        trace.record("decision", {"label": "中文证据"})
    assert output.getvalue() == ""
    assert trace.rows == trace.bytes_written == 0


def test_state_evidence_ignores_only_known_transport_locations() -> None:
    base = _result(step=0, revival_budget=64)
    assert base.transition is not None
    state = replace(
        base,
        observation={
            **base.observation,
            "state_version": 100,
            "_sim_raw": {"state_version": 200, "entity": {"state_version": 201}},
            "player": {"state_version": 300},
            "_training": {"state_version": 400},
        },
        transition=replace(base.transition, facts={"state_version": 500}),
        info={"reward_authority": "external-rl"},
    )
    evidence = probe.canonical_facts(probe._state_facts(state))
    observation = evidence["observation"]
    assert "state_version" not in observation
    assert "state_version" not in observation["_sim_raw"]
    assert observation["_sim_raw"]["entity"]["state_version"] == 201
    assert observation["player"]["state_version"] == 300
    assert observation["_training"]["state_version"] == 400
    assert evidence["transition"]["facts"]["state_version"] == 500
    assert "episode_id" not in evidence["transition"]
    assert "before_state_version" not in evidence["transition"]


def _grouped_state(*, handles: bool = True) -> EnvironmentResult:
    actions = []
    for card, enabled in (("CARD.WOUND", False), ("CARD.BURN", True)):
        for copy in range(3):
            action = {
                "kind": "combat_select_card", "model_action_kind": "card_selection",
                "selection_operation": "select", "enabled": enabled,
                "card_index": len(actions), "card": {"id": card, "pile": "Discard", "cost": -2},
            }
            if handles:
                action["action_handle"] = f"select-{card}-{copy}"
            actions.append(action)
    return EnvironmentResult(episode_id="current-episode", step_index=7,
                             observation={"phase": "selection"}, legal_actions=tuple(actions))


@pytest.mark.parametrize("handles", [True, False])
def test_dispatch_request_resolves_group_index_to_current_raw_representative(handles: bool) -> None:
    state = _grouped_state(handles=handles)
    encoded = GroundedObservationEncoder().encode(state.observation, state.legal_actions)
    assert len(encoded.actions) == 2 and encoded.action(1).position == 3
    backend = type("BackendIdentity", (), {"session_id": "current-session"})()
    request = probe.dispatch_request(backend, state, encoded, 1, timeout_ms=19)
    assert request.episode_id == "current-episode" and request.expected_step_index == 7
    assert request.session_id == "current-session" and request.timeout_ms == 19
    assert request.action_id == ("select-CARD.BURN-0" if handles else None)
    assert request.action_index == (None if handles else 3)
    assert request.request_id != probe.dispatch_request(backend, state, encoded, 1, timeout_ms=19).request_id
    with pytest.raises(ValueError, match="not enabled"):
        probe.dispatch_request(backend, state, encoded, 0, timeout_ms=19)


def test_dispatch_request_rejects_stale_encoded_handle() -> None:
    state = _grouped_state()
    encoded = GroundedObservationEncoder().encode(state.observation, state.legal_actions)
    actions = [dict(action) for action in state.legal_actions]
    actions[3]["action_handle"] = "new-current-handle"
    current = replace(state, legal_actions=tuple(actions))
    backend = type("BackendIdentity", (), {"session_id": "current-session"})()
    with pytest.raises(ValueError, match="handle|representative|current"):
        probe.dispatch_request(backend, current, encoded, 1, timeout_ms=19)


class FakeProbeBackend:
    def __init__(self, *, hp: int = 50, initial_terminal: bool = False,
                 step_truncated: bool = False, fail_step: bool = False) -> None:
        self.session_id = "fake-session"
        self.hp = hp
        self.initial_terminal = initial_terminal
        self.step_truncated = step_truncated
        self.fail_step = fail_step
        self.closed = False
        self.reset_requests: list[ResetRequest] = []
        self.step_requests: list[StepRequest] = []

    def get_state(self) -> dict[str, int]:
        return {"state_version": 41}

    def _state(self, step: int, *, terminal: bool = False, truncated: bool = False) -> EnvironmentResult:
        base = _result(step=step, hp=self.hp, revival_budget=64, floor=step + 1,
                       terminated=terminal, truncated=truncated,
                       run_result="victory" if terminal else "none")
        observation = {**base.observation, "phase": "reward", "combat": {"in_progress": False}}
        return replace(base, observation=observation,
                       legal_actions=() if terminal or truncated else _reward_actions(),
                       info={"reward_authority": "external-rl"})

    def reset(self, request: ResetRequest) -> EnvironmentResult:
        self.reset_requests.append(request)
        return self._state(0, terminal=self.initial_terminal)

    def step(self, request: StepRequest) -> EnvironmentResult:
        self.step_requests.append(request)
        if self.fail_step:
            raise RuntimeError("synthetic transport failure")
        return self._state(len(self.step_requests), truncated=self.step_truncated)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def config() -> Any:
    path = Path(__file__).resolve().parents[1] / "config/experiments/hybrid_vtrace_v1_reward_v9_budget64_on_policy.toml"
    return load_training_config(profile="preheat", config_path=path)


def _run(tmp_path: Path, config: Any, backends: list[FakeProbeBackend],
         limits: probe.ProbeLimits | None = None) -> dict[str, Any]:
    pending = iter(backends)
    return probe.run_probe(backend_factory=lambda: next(pending), policy=CountingPolicy(),
                           config=config, seed=17, limits=limits or probe.ProbeLimits(continuation_decisions=0),
                           output_dir=tmp_path)


def test_run_probe_reconstructs_reward_and_policy_and_closes_two_backends(tmp_path: Path, config: Any) -> None:
    backends = [FakeProbeBackend(), FakeProbeBackend()]
    result = _run(tmp_path, config, backends)
    assert result["status"] == "matched", result
    assert len(result["passes"]) == 2
    assert result["independent_statistical_samples"] == 1
    assert result["training_writes"] is False and result["hidden_rng_clone_verified"] is False
    for backend in backends:
        assert backend.closed and len(backend.step_requests) == 1
        reset = backend.reset_requests[0]
        assert reset.expected_state_version == 41 and reset.seed == 17
        assert reset.training_revival_budget == 64
        assert reset.character == config.environment.character
        assert reset.force_fresh and not reset.rebind_active_run and not reset.defensive_buffs
        assert reset.additional_relics is None
    assert result["passes"][0]["reward_state"] == result["passes"][1]["reward_state"]
    assert (tmp_path / "reference.jsonl").read_text() == (tmp_path / "replay.jsonl").read_text()


def test_run_probe_stops_mismatching_replay_before_dispatch(tmp_path: Path, config: Any) -> None:
    backends = [FakeProbeBackend(), FakeProbeBackend(hp=49)]
    result = _run(tmp_path, config, backends)
    assert result["status"] == "mismatch", result
    assert result["difference"]["section"] == "facts"
    assert all(backend.closed for backend in backends)
    assert not backends[1].step_requests
    assert len(result["passes"]) == 1


def test_run_probe_policy_divergence_is_detected_before_replay_dispatch(tmp_path: Path, config: Any) -> None:
    class ChangedReplayPolicy(CountingPolicy):
        def forward(self, snapshot: Any, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
            logits, after = super().forward(snapshot, hidden)
            if len(self.calls) > 2:
                logits = logits + 0.25  # Same argmax, different policy evidence.
            return logits, after

    backends = [FakeProbeBackend(), FakeProbeBackend()]
    pending = iter(backends)
    result = probe.run_probe(
        backend_factory=lambda: next(pending), policy=ChangedReplayPolicy(),
        config=config, seed=17, limits=probe.ProbeLimits(continuation_decisions=0), output_dir=tmp_path,
    )
    assert result["status"] == "mismatch", result
    assert result["difference"]["section"] == "policy"
    assert len(backends[0].step_requests) == 1 and not backends[1].step_requests
    assert all(backend.closed for backend in backends)


@pytest.mark.parametrize("limits,reason", [
    (probe.ProbeLimits(max_native_steps=1, continuation_decisions=2), "max_native_steps"),
    (probe.ProbeLimits(max_trace_bytes=16), "max_trace_bytes"),
])
def test_run_probe_budget_is_inconclusive_and_closes_backend(
    tmp_path: Path, config: Any, limits: probe.ProbeLimits, reason: str,
) -> None:
    backend = FakeProbeBackend()
    result = _run(tmp_path, config, [backend], limits)
    assert result["status"] == "inconclusive" and result["reason"] == reason
    assert backend.closed


def test_run_probe_no_root_is_inconclusive_not_a_matched_terminal(tmp_path: Path, config: Any) -> None:
    backend = FakeProbeBackend(initial_terminal=True)
    result = _run(tmp_path, config, [backend])
    assert result["status"] == "inconclusive", result
    assert not result["passes"][0]["root_found"]
    assert backend.closed and not backend.step_requests


def test_run_probe_environment_truncation_is_inconclusive_not_task_failure(tmp_path: Path, config: Any) -> None:
    backend = FakeProbeBackend(step_truncated=True)
    result = _run(tmp_path, config, [backend])
    assert result["status"] == "inconclusive", result
    assert backend.closed


def test_run_probe_transport_exception_closes_backend_without_retry(tmp_path: Path, config: Any) -> None:
    backend = FakeProbeBackend(fail_step=True)
    result = _run(tmp_path, config, [backend])
    assert result["status"] == "failed" and result["error_type"] == "RuntimeError"
    assert backend.closed and len(backend.step_requests) == 1


def test_run_probe_timeout_is_inconclusive_and_closes_backend(
    tmp_path: Path, config: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0.0, 181.0])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(clock))
    backend = FakeProbeBackend()
    result = _run(tmp_path, config, [backend])
    assert result["status"] == "inconclusive" and result["reason"] == "timeout_seconds"
    assert backend.closed and not backend.reset_requests
