from __future__ import annotations

import json
from unittest import mock

import pytest

from sts2_env.bridge_client import BridgeClient, BridgeError

PLAYER_TOKEN = "player-token-" + "p" * 32
TRAINING_TOKEN = "training-token-" + "t" * 32


def _game_compatibility():
    return {
        "health": "ready",
        "startup_allowed": True,
        "error_code": "",
        "error_message": "",
        "profile_id": "retail-test-profile",
        "assembly": {
            "name": "sts2",
            "assembly_version": "0.1.0.0",
            "informational_version": "0.1.0+test",
            "module_version_id": "97f10687-c306-4798-ab75-8b9f23f34dfb",
        },
        "probes": [
            {
                "capability": "NGame._Ready",
                "passed": True,
                "code": "capability_present",
                "detail": "method present",
            }
        ],
    }


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload=None):
        self._payload = payload or {"ok": True}

    def json(self):
        return self._payload


def _session_file(tmp_path, *, base_url="http://127.0.0.1:27100/", training=TRAINING_TOKEN):
    path = tmp_path / "session.json"
    capabilities = {"player-control": PLAYER_TOKEN}
    if training is not None:
        capabilities["training"] = training
    path.write_text(
        json.dumps(
            {
                "session_id": "example-session-0001",
                "pid": 123,
                "base_url": base_url,
                "capability_tokens": capabilities,
                "api_versions": ["2.0.0"],
                "schema_version": "2026-07-11.1",
                "action_schema_version": "2.0.0",
                "legal_action_ordering_version": "2.0.0",
                "capabilities": list(capabilities),
                "game_assembly_version": "0.1.0.0",
                "game_compatibility": _game_compatibility(),
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("method_name", "expected_path"),
    [
        ("health_v2", "/v2/health"),
        ("get_spec_v2", "/v2/env/spec"),
        ("get_state_v2", "/v2/env/state"),
        ("combat_catalog_v2", "/v2/env/combat_catalog"),
    ],
)
def test_v2_reads_use_training_scope_and_exact_route(tmp_path, method_name, expected_path):
    client = BridgeClient(session_path=_session_file(tmp_path))
    client._session.request = mock.Mock(return_value=_Response())

    assert getattr(client, method_name)() == {"ok": True}

    call = client._session.request.call_args
    assert call.args[:2] == ("GET", f"http://127.0.0.1:27100{expected_path}")
    assert call.kwargs["headers"] == {"Authorization": f"Bearer {TRAINING_TOKEN}"}


def test_v2_session_needs_no_top_level_legacy_token(tmp_path):
    client = BridgeClient(session_path=_session_file(tmp_path))
    client._session.request = mock.Mock(return_value=_Response({"status": "ok"}))

    assert client.health_v2() == {"status": "ok"}

    with pytest.raises(BridgeError) as raised:
        client.health()
    assert raised.value.status_code == 403


def test_reset_v2_serializes_required_revision_with_training_scope(tmp_path):
    client = BridgeClient(session_path=_session_file(tmp_path))
    client._session.request = mock.Mock(return_value=_Response({"status": "accepted"}))

    client.reset_v2(
        request_id="d2f69d93-8f27-4541-bab7-2e0be9fe3120",
        session_id="example-session-0001",
        expected_state_version=42,
        scenario="full-run",
        seed=None,
        options={"timeout_ms": 100},
        timeout_ms=100,
    )

    call = client._session.request.call_args
    assert call.args[:2] == ("POST", "http://127.0.0.1:27100/v2/env/reset")
    assert call.kwargs["headers"] == {"Authorization": f"Bearer {TRAINING_TOKEN}"}
    assert call.kwargs["json"]["expected_state_version"] == 42


def test_reset_v2_rejects_invalid_revision_before_http(tmp_path):
    client = BridgeClient(session_path=_session_file(tmp_path))
    client._session.request = mock.Mock()
    with pytest.raises(ValueError, match="expected_state_version"):
        client.reset_v2(
            request_id="d2f69d93-8f27-4541-bab7-2e0be9fe3120",
            session_id="example-session-0001",
            expected_state_version=-1,
            scenario="full-run",
            seed=None,
            options={},
            timeout_ms=100,
        )
    client._session.request.assert_not_called()


def test_missing_training_capability_fails_before_http(tmp_path):
    client = BridgeClient(session_path=_session_file(tmp_path, training=None))
    client._session.request = mock.Mock()

    with pytest.raises(BridgeError) as raised:
        client.get_state_v2()

    assert raised.value.status_code == 403
    client._session.request.assert_not_called()


def test_non_loopback_session_authority_is_rejected(tmp_path):
    path = _session_file(tmp_path, base_url="http://192.168.2.1:27100/")

    with pytest.raises(BridgeError, match="Non-loopback"):
        BridgeClient(session_path=path)


def test_explicit_relay_requires_exact_allowlist(tmp_path, monkeypatch):
    import sts2_env.bridge_client as module

    path = _session_file(tmp_path)
    relay = "http://192.168.2.1:27200"
    monkeypatch.setattr(module, "_resolve_session_path", lambda: path)
    monkeypatch.setenv("STS2_BRIDGE_BASE_URL", relay)

    with pytest.raises(BridgeError, match="STS2_BRIDGE_RELAY_ALLOWLIST"):
        BridgeClient()

    monkeypatch.setenv("STS2_BRIDGE_RELAY_ALLOWLIST", relay)
    client = BridgeClient()
    assert client.base_url == relay


def test_allowlist_does_not_authorize_session_descriptor_non_loopback(tmp_path, monkeypatch):
    relay = "http://192.168.2.1:27200"
    monkeypatch.setenv("STS2_BRIDGE_RELAY_ALLOWLIST", relay)
    path = _session_file(tmp_path, base_url=relay)

    with pytest.raises(BridgeError, match="Non-loopback"):
        BridgeClient(session_path=path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "stale", "schema_version"),
        ("action_schema_version", "stale", "action_schema_version"),
        ("legal_action_ordering_version", "stale", "legal_action_ordering_version"),
        ("api_versions", ["legacy-v1"], "supported API"),
        ("session_id", "short", "session_id"),
        ("pid", 0, "pid"),
    ],
)
def test_session_descriptor_identity_mismatch_fails_closed(
    tmp_path,
    field,
    value,
    message,
):
    path = _session_file(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BridgeError, match=message):
        BridgeClient(session_path=path)


def test_session_capability_names_and_credentials_must_match(tmp_path):
    path = _session_file(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["capabilities"] = ["player-control"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BridgeError, match="do not match"):
        BridgeClient(session_path=path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("game_compatibility"), "game_compatibility"),
        (
            lambda payload: payload["game_compatibility"].update(
                {"health": "degraded", "startup_allowed": False}
            ),
            "incompatible game assembly",
        ),
        (
            lambda payload: payload["game_compatibility"]["assembly"].update(
                {"module_version_id": "not-a-uuid"}
            ),
            "module_version_id",
        ),
        (
            lambda payload: payload["game_compatibility"]["probes"][0].update(
                {"passed": False, "code": "missing_capability"}
            ),
            "failed compatibility probe",
        ),
        (
            lambda payload: payload["game_compatibility"].update({"unexpected": True}),
            "game_compatibility",
        ),
        (
            lambda payload: payload.update({"game_assembly_version": "0.2.0.0"}),
            "internally inconsistent",
        ),
    ],
)
def test_game_compatibility_evidence_fails_closed(tmp_path, mutate, message):
    path = _session_file(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BridgeError, match=message):
        BridgeClient(session_path=path)
