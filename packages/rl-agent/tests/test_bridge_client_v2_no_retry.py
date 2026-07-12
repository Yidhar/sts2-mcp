from __future__ import annotations

from unittest import mock

import pytest
import requests

from sts2_env.bridge_client import BridgeClient, BridgeError


def _client(session: object) -> BridgeClient:
    client = BridgeClient.__new__(BridgeClient)
    client._session = session
    client._base_url = "http://127.0.0.1:1234"
    client._session_base_url = client._base_url
    client._is_connected = True
    client._capability_tokens = {"training": "training-token"}
    return client


def test_v2_mutation_timeout_is_never_retried() -> None:
    session = mock.Mock()
    session.request.side_effect = requests.Timeout("timeout")
    client = _client(session)

    with pytest.raises(BridgeError) as raised:
        client.step_v2(
            request_id="4f81ef77-e3b8-4707-8eec-8cf0e10e0286",
            session_id="session",
            episode_id="episode",
            expected_step_index=1,
            action={"action_handle": "end_turn"},
            timeout_ms=100,
        )

    assert session.request.call_count == 1
    assert raised.value.response_body["outcome_unknown"] is True


def test_v2_http_error_preserves_status_without_retry() -> None:
    response = mock.Mock(status_code=404)
    response.json.return_value = {"error": "not_found"}
    session = mock.Mock()
    session.request.return_value = response
    client = _client(session)

    with pytest.raises(BridgeError) as raised:
        client.reset_v2(
            request_id="4f81ef77-e3b8-4707-8eec-8cf0e10e0286",
            session_id="session",
            expected_state_version=42,
            scenario="full-run",
            seed=None,
            options={},
            timeout_ms=100,
        )

    assert raised.value.status_code == 404
    assert session.request.call_count == 1
