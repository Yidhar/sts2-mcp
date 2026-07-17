from __future__ import annotations

from typing import Any

import pytest

from sts2_rl.backends import (
    EnvironmentCommandError,
    HeadlessBackend,
    LegacyClientBackend,
    LiveBackend,
)
from sts2_rl.contracts import (
    CombatResetRequest,
    EnvironmentBackend,
    EnvironmentResult,
    ResetRequest,
    StepRequest,
)


class FakeLegacyClient:
    def __init__(self) -> None:
        self.is_connected = True
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.close_count = 0

    def health(self) -> dict[str, Any]:
        return {"ok": True}

    def get_spec(self) -> dict[str, Any]:
        return {"ok": True, "source": "fake"}

    def get_state(self) -> dict[str, Any]:
        return {"phase": "combat"}

    def _result(self, reward: float = 0.0) -> dict[str, Any]:
        return {
            "ok": True,
            "episode_id": "episode-1",
            "step_index": 2,
            "reward": reward,
            "done": False,
            "truncated": False,
            "obs": {"phase": "combat"},
            "legal_actions": [{"action_id": "end_turn"}],
            "info": {},
        }

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("reset", kwargs))
        return self._result()

    def step(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("step", kwargs))
        return self._result(1.5)

    def combat_reset(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("combat_reset", kwargs))
        return self._result()

    def close(self) -> None:
        self.close_count += 1
        self.is_connected = False


def test_step_request_requires_v2_identity_and_one_action_selector() -> None:
    with pytest.raises(TypeError):
        StepRequest(episode_id="ep", action_index=0)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        StepRequest.from_legacy(
            episode_id="ep",
            expected_step_index=0,
            action_index=0,
            action_id="a",
        )
    request = StepRequest.from_legacy(
        episode_id="ep",
        expected_step_index=3,
        action_index=0,
    )
    assert request.action_index == 0
    assert request.request_id
    assert request.expected_step_index == 3


def test_environment_result_round_trips_legacy_payload() -> None:
    typed = EnvironmentResult.from_legacy(FakeLegacyClient()._result(2.5))
    assert typed.reward == 2.5
    assert typed.observation["phase"] == "combat"
    assert typed.legal_actions[0]["action_id"] == "end_turn"
    assert typed.to_legacy()["done"] is False


@pytest.mark.parametrize("backend_type", [LiveBackend, HeadlessBackend])
def test_live_and_headless_adapters_share_protocol(backend_type: type[LegacyClientBackend]) -> None:
    client = FakeLegacyClient()
    backend = backend_type(
        client=client,
        session_id="test-session",
        **({"allow_legacy_fallback": True} if backend_type is LiveBackend else {}),
    )
    assert isinstance(backend, EnvironmentBackend)

    reset = backend.reset(
        ResetRequest.from_legacy(
            session_id=backend.session_id,
            expected_state_version=0,
            character="IRONCLAD",
            seed="ABC123",
        )
    )
    stepped = backend.step(
        StepRequest.from_legacy(
            session_id=backend.session_id,
            episode_id=reset.episode_id,
            expected_step_index=reset.step_index,
            action_id="end_turn",
        )
    )
    combat = backend.combat_reset(
        CombatResetRequest.from_legacy(
            session_id=backend.session_id,
            expected_state_version=2,
            encounter_id="TEST",
            deck=("CARD.A",),
        )
    )

    assert stepped.reward == (1.5 if backend_type is LiveBackend else 0.0)
    if backend_type is HeadlessBackend:
        assert stepped.raw["reward_authority"] == "external-rl"
        assert stepped.transition is not None
    assert combat.episode_id == "episode-1"
    assert client.calls[0][1]["seed"] == "ABC123"
    assert client.calls[1][1]["action_id"] == "end_turn"
    assert client.calls[2][1]["deck"] == ["CARD.A"]

    backend.close()
    backend.close()
    assert client.close_count == 1
    with pytest.raises(RuntimeError):
        backend.get_state()


def test_legacy_backend_context_manager_closes_client() -> None:
    client = FakeLegacyClient()
    with LegacyClientBackend(client, backend_name="fake", session_id="fake-session") as backend:
        assert backend.health()["ok"]
    assert client.close_count == 1


class FakeV2Client(FakeLegacyClient):
    session_id = "v2-session"

    def __init__(self) -> None:
        super().__init__()
        self.v2_calls: list[tuple[str, dict[str, Any]]] = []

    def _result(self, reward: float = 0.0) -> dict[str, Any]:
        transition = {
            "episode_id": "episode-1",
            "step_index": 2,
            "before_state_version": 1,
            "after_state_version": 2,
            "facts": {"enemy_hp_delta": reward, "combat_result": "none"},
        }
        return {
            "ok": True,
            "episode_id": "episode-1",
            "step_index": 2,
            "reward": None,
            "reward_authority": "external-rl",
            "terminated": False,
            "truncated": False,
            "observation": {"phase": "combat"},
            "legal_actions": [
                {
                    "idx": 0,
                    "action_handle": "end_turn",
                    "kind": "end_turn",
                    "diagnostic": {"retained": True},
                }
            ],
            "transition": transition,
            "transition_facts": transition,
            "info": {"reward_authority": "external-rl"},
        }

    @staticmethod
    def _envelope(result: dict[str, Any], request_id: str) -> dict[str, Any]:
        return {
            "ok": True,
            "api_version": "2.0.0",
            "schema_version": "2026-07-17.1",
            "request_id": request_id,
            "status": "committed",
            "replayed_result": False,
            "result": result,
            "error": None,
        }

    def reset_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.v2_calls.append(("reset_v2", kwargs))
        return self._envelope(self._result(), str(kwargs["request_id"]))

    def get_state_v2(self) -> dict[str, Any]:
        return {
            "ok": True,
            "capability": "training",
            "state_version": 42,
            "legal_actions": [
                {
                    "idx": 0,
                    "action_handle": "end_turn",
                    "kind": "end_turn",
                    "diagnostic": {"retained": True},
                }
            ],
        }

    def step_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.v2_calls.append(("step_v2", kwargs))
        return self._envelope(self._result(2.0), str(kwargs["request_id"]))


def test_live_backend_prefers_scoped_v2_and_forwards_identity() -> None:
    client = FakeV2Client()
    backend = LiveBackend(client=client)
    reset_request = ResetRequest.from_legacy(
        session_id=backend.session_id,
        expected_state_version=42,
        character="IRONCLAD",
        seed="SEED",
    )
    reset = backend.reset(reset_request)
    state = backend.get_state()
    step_request = StepRequest.from_legacy(
        session_id=backend.session_id,
        episode_id=reset.episode_id,
        expected_step_index=reset.step_index,
        action_id="end_turn",
    )
    result = backend.step(step_request)

    assert result.reward == 0.0
    assert result.transition is not None
    assert result.transition.facts["enemy_hp_delta"] == 2.0
    assert reset.legal_actions[0]["action_id"] == "end_turn"
    assert "action_handle" not in reset.legal_actions[0]
    assert result.legal_actions[0]["action_id"] == "end_turn"
    assert result.legal_actions[0]["diagnostic"] == {"retained": True}
    assert "action_handle" not in result.legal_actions[0]
    assert state["legal_actions"][0]["action_id"] == "end_turn"
    assert state["legal_actions"][0]["diagnostic"] == {"retained": True}
    assert "action_handle" not in state["legal_actions"][0]
    assert [name for name, _ in client.v2_calls] == ["reset_v2", "step_v2"]
    reset_call = client.v2_calls[0][1]
    assert reset_call["request_id"] == reset_request.request_id
    assert reset_call["session_id"] == "v2-session"
    assert reset_call["expected_state_version"] == 42
    assert reset_call["scenario"] == "full-run"
    step_call = client.v2_calls[1][1]
    assert step_call["expected_step_index"] == reset.step_index
    assert step_call["action"]["action_handle"] == "end_turn"
    assert client.calls == []


def test_live_backend_does_not_fallback_on_v2_rejection() -> None:
    client = FakeV2Client()
    client.reset_v2 = lambda **kwargs: {
        "ok": False,
        "api_version": "2.0.0",
        "schema_version": "2026-07-17.1",
        "request_id": kwargs["request_id"],
        "status": "rejected_before_execution",
        "replayed_result": False,
        "error": {"message": "step conflict"},
    }
    backend = LiveBackend(client=client)
    request = ResetRequest.from_legacy(
        session_id=backend.session_id,
        expected_state_version=42,
    )
    with pytest.raises(EnvironmentCommandError, match="step conflict"):
        backend.reset(request)
    assert client.calls == []


def test_live_backend_rejects_legacy_action_id_on_v2_state_wire() -> None:
    client = FakeV2Client()
    client.get_state_v2 = lambda: {
        "ok": True,
        "capability": "training",
        "state_version": 42,
        "legal_actions": [{"idx": 0, "action_id": "end_turn", "kind": "end_turn"}],
    }
    backend = LiveBackend(client=client)

    with pytest.raises(EnvironmentCommandError, match="forbidden action_id"):
        backend.get_state()


def test_live_backend_rejects_legacy_action_id_on_v2_result_wire() -> None:
    client = FakeV2Client()
    malformed_result = client._result()
    malformed_result["legal_actions"] = [
        {"idx": 0, "action_id": "end_turn", "kind": "end_turn"}
    ]
    client.reset_v2 = lambda **kwargs: client._envelope(
        malformed_result,
        str(kwargs["request_id"]),
    )
    backend = LiveBackend(client=client)
    request = ResetRequest.from_legacy(
        session_id=backend.session_id,
        expected_state_version=42,
    )

    with pytest.raises(EnvironmentCommandError, match="forbidden action_id"):
        backend.reset(request)


def test_live_backend_rejects_stale_reset_revision_before_mutation() -> None:
    client = FakeV2Client()
    backend = LiveBackend(client=client)
    request = ResetRequest.from_legacy(
        session_id=backend.session_id,
        expected_state_version=41,
    )
    with pytest.raises(EnvironmentCommandError, match="revision is stale"):
        backend.reset(request)
    assert client.v2_calls == []


def test_live_backend_rejects_missing_typed_reset_revision() -> None:
    client = FakeV2Client()
    backend = LiveBackend(client=client)
    request = ResetRequest.from_legacy(session_id=backend.session_id)
    with pytest.raises(EnvironmentCommandError, match="missing expected_state_version"):
        backend.reset(request)
    assert client.v2_calls == []


def test_live_backend_rejects_cross_session_request() -> None:
    backend = LiveBackend(client=FakeV2Client())
    request = ResetRequest.from_legacy(
        session_id="wrong-session",
        expected_state_version=42,
    )
    with pytest.raises(ValueError, match="does not match"):
        backend.reset(request)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_version", "9.9.9"),
        ("schema_version", "unknown"),
        ("request_id", "different-request"),
        ("replayed_result", "false"),
    ],
)
def test_live_backend_rejects_mismatched_command_envelope_identity(
    field: str,
    value: object,
) -> None:
    client = FakeV2Client()

    def malformed(**kwargs: Any) -> dict[str, Any]:
        envelope = client._envelope(client._result(), str(kwargs["request_id"]))
        envelope[field] = value
        return envelope

    client.reset_v2 = malformed  # type: ignore[method-assign]
    backend = LiveBackend(client=client)
    request = ResetRequest.from_legacy(
        session_id=backend.session_id,
        expected_state_version=42,
    )
    with pytest.raises(EnvironmentCommandError, match=field):
        backend.reset(request)
