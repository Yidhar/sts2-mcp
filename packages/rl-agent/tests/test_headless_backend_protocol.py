from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.backends import (
    HeadlessBackend,
    HeadlessBackendPoisonedError,
    HeadlessProtocolError,
    HeadlessRecoverableProtocolError,
)
from sts2_rl.backends.headless import DEFAULT_REQUEST_CACHE_SIZE, DEFAULT_REQUEST_CACHE_TTL_S
from sts2_rl.contracts import ResetRequest, StepRequest

RESET_ID = "11111111-1111-4111-8111-111111111111"
STEP_ID = "22222222-2222-4222-8222-222222222222"


class FakeHeadlessClient:
    session_id = "headless-session"

    def __init__(self) -> None:
        self.is_connected = True
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.step_payload: dict[str, Any] | None = None

    @staticmethod
    def _payload(*, after: bool, reward: float) -> dict[str, Any]:
        obs = {
            "state_version": 8 if after else 7,
            "phase": "combat",
            "player": {
                "hp": 48 if after else 50,
                "gold": 12 if after else 10,
                "deck": [{"id": "CARD.A"}, *([{"id": "CARD.B"}] if after else [])],
                "potions": ([{"id": "POTION.X"}] if after else []),
            },
            "run": {"floor": 3, "room_type": "Monster"},
            "combat": {"enemies": [{"hp": 5 if after else 10}]},
        }
        return {
            "ok": True,
            "episode_id": "sim-episode",
            "step_index": 99,
            "reward": reward,
            "done": False,
            "truncated": False,
            "obs": obs,
            "legal_actions": [{"action_id": "end_turn"}],
            "info": {},
        }

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        self.reset_calls += 1
        return self._payload(after=False, reward=10.0)

    def step(self, **kwargs: Any) -> dict[str, Any]:
        self.step_calls += 1
        return self.step_payload or self._payload(after=True, reward=99.0)

    def combat_reset(self, **kwargs: Any) -> dict[str, Any]:
        return self.reset(**kwargs)

    def health(self) -> dict[str, Any]:
        return {"ok": True}

    def get_state(self) -> dict[str, Any]:
        return self._payload(after=False, reward=0.0)["obs"]

    def combat_catalog(self) -> dict[str, Any]:
        return {"encounters": []}

    def close(self) -> None:
        self.close_calls += 1
        self.is_connected = False


def reset_request(**overrides: Any) -> ResetRequest:
    values = {
        "request_id": RESET_ID,
        "session_id": "headless-session",
        "scenario": "full-run",
        "expected_state_version": 0,
        "character": "IRONCLAD",
        "timeout_ms": 100,
    }
    values.update(overrides)
    return ResetRequest(**values)


def test_headless_request_id_replays_and_conflicts_fail_closed() -> None:
    client = FakeHeadlessClient()
    backend = HeadlessBackend(client=client)
    request = reset_request()
    first = backend.reset(request)
    replay = backend.reset(request)
    assert replay is first
    assert client.reset_calls == 1
    with pytest.raises(HeadlessProtocolError, match="different request body"):
        backend.reset(reset_request(character="SILENT"))
    assert client.reset_calls == 1


def test_default_request_store_is_sized_for_sustained_online_collection() -> None:
    client = FakeHeadlessClient()
    backend = HeadlessBackend(client=client)
    spec = backend.get_spec()

    assert DEFAULT_REQUEST_CACHE_SIZE == 65_536
    assert DEFAULT_REQUEST_CACHE_TTL_S == 600.0
    assert spec["request_id_dedupe_capacity"] == DEFAULT_REQUEST_CACHE_SIZE
    assert spec["request_id_dedupe_ttl_s"] == DEFAULT_REQUEST_CACHE_TTL_S

    # The previous 2,048-entry default failed after only 1,971 environment
    # steps in a real online run, long before the ten-minute TTL could expire.
    for index in range(2_049):
        request_id = f"{index:08x}-0000-4000-8000-000000000000"
        backend.reset(
            reset_request(
                request_id=request_id,
                expected_state_version=index,
            )
        )

    assert client.reset_calls == 2_049


def test_headless_session_and_expected_step_are_enforced_before_mutation() -> None:
    client = FakeHeadlessClient()
    backend = HeadlessBackend(client=client)
    reset = backend.reset(reset_request())
    with pytest.raises(HeadlessProtocolError, match="session"):
        backend.step(
            StepRequest(
                request_id=STEP_ID,
                session_id="wrong",
                episode_id=reset.episode_id,
                expected_step_index=0,
                action_id="end_turn",
            )
        )
    with pytest.raises(HeadlessProtocolError, match="expected_step_index"):
        backend.step(
            StepRequest(
                request_id=STEP_ID,
                session_id=backend.session_id,
                episode_id=reset.episode_id,
                expected_step_index=7,
                action_id="end_turn",
            )
        )
    assert client.step_calls == 0


def test_headless_projects_canonical_facts_and_ignores_adapter_scalar() -> None:
    client = FakeHeadlessClient()
    backend = HeadlessBackend(client=client)
    reset = backend.reset(reset_request())
    request = StepRequest(
        request_id=STEP_ID,
        session_id=backend.session_id,
        episode_id=reset.episode_id,
        expected_step_index=reset.step_index,
        action_id="end_turn",
        timeout_ms=100,
    )
    result = backend.step(request)
    replay = backend.step(request)

    assert replay is result
    assert client.step_calls == 1
    assert result.step_index == 1
    assert result.reward == 0.0
    assert result.raw["reward"] is None
    assert result.raw["reward_authority"] == "external-rl"
    assert result.info["sim_backend_reward_ignored"] == 99.0
    assert result.transition is not None
    assert result.transition.facts["hp_delta"] == -2.0
    assert result.transition.facts["enemy_hp_delta"] == 5.0
    assert result.transition.facts["gold_delta"] == 2.0
    assert result.transition.facts["cards_added"] == ["CARD.B"]
    assert result.transition.facts["potions_added"] == ["POTION.X"]


def test_headless_request_store_refuses_capacity_without_evicting_unexpired_ids() -> None:
    clock = {"now": 100.0}
    client = FakeHeadlessClient()
    backend = HeadlessBackend(
        client=client,
        request_cache_size=2,
        request_cache_ttl_s=10.0,
        clock=lambda: clock["now"],
    )
    first = reset_request()
    first_result = backend.reset(first)
    second = reset_request(
        request_id="33333333-3333-4333-8333-333333333333",
        expected_state_version=1,
    )
    backend.reset(second)
    assert client.reset_calls == 2

    third = reset_request(
        request_id="44444444-4444-4444-8444-444444444444",
        expected_state_version=2,
    )
    with pytest.raises(HeadlessProtocolError, match="capacity exceeded"):
        backend.reset(third)
    assert client.reset_calls == 2

    # The first identity remains replayable and cannot execute twice merely
    # because newer requests filled the store.
    assert backend.reset(first) is first_result
    assert client.reset_calls == 2

    # Once the advertised TTL expires, completed identities may be purged and
    # a new request can execute.
    clock["now"] = 111.0
    backend.reset(third)
    assert client.reset_calls == 3


def test_invalid_post_mutation_surface_is_quarantined_without_revision_or_cache_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(artifact_root))
    client = FakeHeadlessClient()
    backend = HeadlessBackend(client=client)
    reset = backend.reset(reset_request())
    client.step_payload = {
        "ok": True,
        "episode_id": reset.episode_id,
        "step_index": 999,
        "reward": 0.0,
        "done": False,
        "truncated": False,
        "obs": {
            "state_type": "combat_start_pending",
            # Ensure the incident crosses the compression threshold and the
            # raw response remains available for post-mortem inspection.
            "diagnostic_blob": "x" * 70_000,
        },
        "legal_actions": [],
        "info": {},
    }
    request = StepRequest(
        request_id=STEP_ID,
        session_id=backend.session_id,
        episode_id=reset.episode_id,
        expected_step_index=reset.step_index,
        action_id="end_turn",
        timeout_ms=100,
    )

    with pytest.raises(HeadlessRecoverableProtocolError) as caught:
        backend.step(request)

    incident = caught.value
    assert incident.recoverable is True
    assert incident.poisoned is True
    assert incident.operation == "step"
    assert len(incident.fingerprint) == 24
    assert incident.quarantine_path is not None
    evidence_path = Path(incident.quarantine_path)
    assert evidence_path.is_file()
    assert evidence_path.name.endswith(".json.gz")
    with gzip.open(evidence_path, "rt", encoding="utf-8") as handle:
        evidence = json.load(handle)
    assert evidence["fingerprint"] == incident.fingerprint
    assert evidence["incident"]["legal_action_count"] == 0
    assert evidence["raw_response"]["obs"]["diagnostic_blob"] == "x" * 70_000

    # The invalid result never becomes a typed transition, revision, cached
    # request result, or new active observation.
    assert backend._state_version == 1
    assert backend._logical_step_index == 0
    assert STEP_ID not in backend._request_cache
    assert backend._observation == dict(reset.observation)
    assert client.step_calls == 1
    assert backend.is_connected is False
    assert backend.health()["poisoned"] is True

    with pytest.raises(HeadlessBackendPoisonedError) as reused:
        backend.step(request)
    assert reused.value.fingerprint == incident.fingerprint
    assert client.step_calls == 1
    with pytest.raises(HeadlessBackendPoisonedError):
        backend.get_state()


def test_predispatch_ordering_errors_are_not_recoverable_or_poisoning() -> None:
    client = FakeHeadlessClient()
    backend = HeadlessBackend(client=client)
    reset = backend.reset(reset_request())
    request = StepRequest(
        request_id=STEP_ID,
        session_id=backend.session_id,
        episode_id=reset.episode_id,
        expected_step_index=99,
        action_id="end_turn",
        timeout_ms=100,
    )

    with pytest.raises(HeadlessProtocolError) as caught:
        backend.step(request)

    assert not isinstance(caught.value, HeadlessRecoverableProtocolError)
    assert caught.value.recoverable is False
    assert backend.is_connected is True
    assert client.step_calls == 0


@pytest.mark.parametrize("outcome", ["victory", "defeat"])
def test_full_run_terminal_projects_typed_run_result_without_player_block(outcome: str) -> None:
    client = FakeHeadlessClient()
    backend = HeadlessBackend(client=client)
    reset = backend.reset(reset_request())
    client.step_payload = {
        "ok": True,
        "episode_id": reset.episode_id,
        "step_index": 1,
        "reward": 0.0,
        "done": True,
        "truncated": False,
        "terminal_reason": f"run_{outcome}",
        "obs": {
            "state_type": "game_over",
            "terminated": True,
            "run": {"act": 3, "floor": 46},
        },
        "legal_actions": [],
        "info": {"sim_run_outcome": outcome},
    }

    result = backend.step(
        StepRequest(
            request_id=STEP_ID,
            session_id=backend.session_id,
            episode_id=reset.episode_id,
            expected_step_index=reset.step_index,
            action_id="end_turn",
            timeout_ms=100,
        )
    )

    assert result.terminated
    assert result.terminal_reason == f"run_{outcome}"
    assert result.transition is not None
    assert result.transition.facts["run_result"] == outcome
    assert result.transition.facts["combat_result"] == "none"
    assert result.observation.get("player") is None
    assert backend._state_version == 2
    assert STEP_ID in backend._request_cache

def test_terminal_surface_with_actions_uses_plain_json_incident_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    client = FakeHeadlessClient()
    backend = HeadlessBackend(client=client)
    reset = backend.reset(reset_request())
    client.step_payload = {
        "ok": True,
        "episode_id": reset.episode_id,
        "step_index": 1,
        "reward": 0.0,
        "done": True,
        "truncated": False,
        "terminal_reason": "victory",
        "obs": {"state_type": "game_over", "terminated": True},
        "legal_actions": [{"action_id": "must-not-exist"}],
        "info": {},
    }
    request = StepRequest(
        request_id=STEP_ID,
        session_id=backend.session_id,
        episode_id=reset.episode_id,
        expected_step_index=reset.step_index,
        action_id="end_turn",
        timeout_ms=100,
    )

    with pytest.raises(HeadlessRecoverableProtocolError) as caught:
        backend.step(request)

    evidence_path = Path(str(caught.value.quarantine_path))
    assert evidence_path.name.endswith(".json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["incident"]["terminal"] is True
    assert evidence["incident"]["legal_action_count"] == 1
    assert backend._state_version == 1
    assert STEP_ID not in backend._request_cache
