from __future__ import annotations

from typing import Any

import pytest

from muzero.training import env_factory
from sts2_env.bridge_client import BridgeError
from sts2_env.environment_runtime import EnvironmentRuntimeMixin
from sts2_rl.backends import LiveBackend


class FakeV2Client:
    session_id = "session-v2"

    def __init__(self) -> None:
        self.v2_calls: list[tuple[str, dict[str, Any]]] = []
        self.legacy_calls: list[str] = []
        self.is_connected = True

    @staticmethod
    def _result(*, step_index: int, reward: Any = None) -> dict[str, Any]:
        facts = {
            "hp_delta": -2.0 if step_index else 0.0,
            "enemy_hp_delta": 10.0 if step_index else 0.0,
            "combat_result": "none",
        }
        transition = {
            "episode_id": "episode-v2",
            "step_index": step_index,
            "before_state_version": max(step_index - 1, 0),
            "after_state_version": step_index,
            "facts": facts,
        }
        return {
            "episode_id": "episode-v2",
            "step_index": step_index,
            "observation": {"state_version": step_index, "phase": "combat"},
            "legal_actions": [
                {
                    "idx": 0,
                    "action_handle": "end_turn",
                    "kind": "end_turn",
                    "diagnostic": {"retained": True},
                }
            ],
            "terminated": False,
            "truncated": False,
            "transition": transition,
            "transition_facts": transition,
            "reward": reward,
            "reward_authority": "external-rl",
            "info": {"reward_authority": "external-rl"},
        }

    @staticmethod
    def _envelope(result: dict[str, Any], request_id: str) -> dict[str, Any]:
        return {
            "ok": True,
            "api_version": "2.0.0",
            "schema_version": "2026-07-11.1",
            "request_id": request_id,
            "status": "committed",
            "replayed_result": False,
            "result": result,
            "error": None,
        }

    def reset_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.v2_calls.append(("reset_v2", kwargs))
        return self._envelope(self._result(step_index=0), str(kwargs["request_id"]))

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
        return self._envelope(self._result(step_index=1), str(kwargs["request_id"]))

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        self.legacy_calls.append("reset")
        raise AssertionError("legacy reset must not be called")

    def step(self, **kwargs: Any) -> dict[str, Any]:
        self.legacy_calls.append("step")
        raise AssertionError("legacy step must not be called")

    def combat_reset(self, **kwargs: Any) -> dict[str, Any]:
        self.legacy_calls.append("combat_reset")
        raise AssertionError("legacy combat_reset must not be called")

    def health(self) -> dict[str, Any]:
        return {"ok": True}

    def get_state(self) -> dict[str, Any]:
        return {"phase": "combat"}

    def close(self) -> None:
        self.is_connected = False


class RuntimeHarness(EnvironmentRuntimeMixin):
    pass


def test_mainline_factory_injects_strict_live_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: dict[str, Any] = {}

    class Backend:
        def __init__(self, **kwargs: Any) -> None:
            constructed["backend_kwargs"] = kwargs
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class Encoder:
        def __init__(self, **kwargs: Any) -> None:
            constructed["encoder_kwargs"] = kwargs

    class Env:
        def __init__(self, **kwargs: Any) -> None:
            constructed["env_kwargs"] = kwargs

    monkeypatch.setattr(env_factory, "LiveBackend", Backend)
    monkeypatch.setattr(env_factory, "WorldTokenObservationEncoder", Encoder)
    monkeypatch.setattr(env_factory, "SlayTheSpire2EnvV2", Env)

    built = env_factory.build_train_env(
        env_index=0,
        session_file="session.json",
        combat_sandbox=False,
        combat_sandbox_potions=True,
        character=None,
        defensive_buffs=False,
        encounter_id=None,
        encounter_pool=[],
        snapshot_pool=None,
        reset_timeout_ms=100,
        step_timeout_ms=100,
        obs_mode="token_v3",
        environment_backend="live",
    )

    assert isinstance(built, Env)
    assert constructed["backend_kwargs"] == {
        "session_path": "session.json",
        "allow_legacy_fallback": False,
    }
    assert isinstance(constructed["env_kwargs"]["backend"], Backend)
    assert "bridge" not in constructed["env_kwargs"]


def test_live_runtime_uses_v2_identity_and_external_fact_reward() -> None:
    client = FakeV2Client()
    backend = LiveBackend(client=client, allow_legacy_fallback=False)
    runtime = RuntimeHarness()
    runtime._initialize_environment_runtime(backend=backend)

    reset = runtime._backend_reset(character="IRONCLAD", timeout_ms=100)
    stepped = runtime._backend_step(
        episode_id=reset["episode_id"],
        action_id="end_turn",
        timeout_ms=100,
    )
    breakdown = runtime._record_backend_transition(stepped, action_handle="end_turn")

    assert [name for name, _ in client.v2_calls] == ["reset_v2", "step_v2"]
    assert client.legacy_calls == []
    reset_call = client.v2_calls[0][1]
    step_call = client.v2_calls[1][1]
    assert reset_call["session_id"] == "session-v2"
    assert reset_call["expected_state_version"] == 42
    assert step_call["episode_id"] == "episode-v2"
    assert step_call["expected_step_index"] == 0
    assert step_call["action"] == {"action_handle": "end_turn", "timeout_ms": 100}
    assert reset["legal_actions"][0]["action_id"] == "end_turn"
    assert "action_handle" not in reset["legal_actions"][0]
    assert stepped["legal_actions"][0]["action_id"] == "end_turn"
    assert stepped["legal_actions"][0]["diagnostic"] == {"retained": True}
    assert "action_handle" not in stepped["legal_actions"][0]
    # v2 reward is external: -2 hp * .02 + 10 enemy hp * .01 = .06.
    assert breakdown.total == pytest.approx(0.06)
    assert breakdown.components["legacy_backend_reward"] == 0.0
    assert breakdown.spec_version != "legacy-backend-reward-v1"
    assert stepped["info"]["canonical_reward"]["authority"] == "external-rl"


def test_strict_live_backend_fails_closed_when_v2_is_missing() -> None:
    client = FakeV2Client()

    def missing(**kwargs: Any) -> dict[str, Any]:
        raise BridgeError("missing", status_code=404)

    client.reset_v2 = missing  # type: ignore[method-assign]
    backend = LiveBackend(client=client, allow_legacy_fallback=False)
    runtime = RuntimeHarness()
    runtime._initialize_environment_runtime(backend=backend)

    with pytest.raises(RuntimeError, match="legacy mutation fallback is disabled"):
        runtime._backend_reset(timeout_ms=100)
    assert client.legacy_calls == []
