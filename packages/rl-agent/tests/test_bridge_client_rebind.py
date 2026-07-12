"""Tests for BridgeClient auto-rebind after a launcher-side instance restart.

Scenario: launcher watchdog kills a hung bridge, relaunches the game process,
and the bridge mod writes a FRESH session_N.json with a new token + new port.
The BridgeClient instance owned by the (still-alive) RL training worker must
detect the file change and swap to the new credentials mid-request.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

import sts2_env.bridge_client as bridge_client

# ``sts2_env`` exposes a lazy package surface, so importing this pure HTTP
# client does not import Torch. Use the canonical module rather than replacing
# ``sys.modules["sts2_env.bridge_client"]`` during collection: that replacement
# leaked into unrelated tests and made outcomes depend on collection order and
# any real session descriptor on the host.
BridgeClient = bridge_client.BridgeClient
BridgeError = bridge_client.BridgeError

OLD_TOKEN = "OLD_TOKEN_" + "o" * 32
NEW_TOKEN = "NEW_TOKEN_" + "n" * 32


def _write_session(path: Path, *, token: str, port: int) -> None:
    path.write_text(
        json.dumps({
            "session_id": f"test-session-{port:05d}",
            "base_url": f"http://127.0.0.1:{port}/",
            "token": token,
            "capability_tokens": {
                "player-control": "p" * 32,
                "training": "t" * 32,
                "legacy-privileged": token,
            },
            "api_versions": ["2.0.0", "legacy-v1"],
            "schema_version": "2026-07-13.1",
            "action_schema_version": "2.1.0",
            "legal_action_ordering_version": "2.0.0",
            "capabilities": ["player-control", "training", "legacy-privileged"],
            "pid": 12345,
            "game_assembly_version": "0.1.0.0",
            "game_compatibility": {
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
            },
        }),
        encoding="utf-8",
    )


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class BridgeClientRebindTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = Path(tempfile.mkdtemp(prefix="bridge_client_rebind_"))
        self._session_path = self._tmp / "session_0.json"
        _write_session(self._session_path, token=OLD_TOKEN, port=27100)

    def tearDown(self):
        # Best-effort cleanup.
        try:
            if self._session_path.exists():
                self._session_path.unlink()
            self._tmp.rmdir()
        except OSError:
            pass

    def _force_mtime_bump(self) -> None:
        # Windows filesystems sometimes round mtime; advance explicitly so
        # the test is deterministic regardless of clock resolution.
        st = self._session_path.stat()
        os.utime(self._session_path, (st.st_atime, st.st_mtime + 2.0))

    def test_load_session_caches_mtime(self) -> None:
        client = BridgeClient(session_path=self._session_path)
        self.assertEqual(client._token, OLD_TOKEN)
        self.assertGreater(client._session_mtime, 0.0)

    def test_rebind_noop_when_file_unchanged(self) -> None:
        client = BridgeClient(session_path=self._session_path)
        rebound = client._maybe_rebind_session(reason="unit_test")
        self.assertFalse(rebound)
        self.assertEqual(client._token, OLD_TOKEN)

    def test_rebind_picks_up_new_token_after_file_rewrite(self) -> None:
        client = BridgeClient(session_path=self._session_path)
        _write_session(self._session_path, token=NEW_TOKEN, port=27200)
        self._force_mtime_bump()
        rebound = client._maybe_rebind_session(reason="unit_test")
        self.assertTrue(rebound)
        self.assertEqual(client._token, NEW_TOKEN)
        self.assertIn("27200", client._base_url)

    def test_rebind_log_never_exposes_token_or_token_prefix(self) -> None:
        old_token = "OLD_TOKEN_SUPER_SECRET_" + "o" * 32
        new_token = "NEW_TOKEN_SUPER_SECRET_" + "n" * 32
        _write_session(self._session_path, token=old_token, port=27100)
        client = BridgeClient(session_path=self._session_path)
        _write_session(self._session_path, token=new_token, port=27200)
        self._force_mtime_bump()

        with patch("builtins.print") as mocked_print:
            self.assertTrue(client._maybe_rebind_session(reason="unit_test"))

        rendered = " ".join(
            " ".join(str(value) for value in call.args)
            for call in mocked_print.call_args_list
        )
        for secret in (old_token, new_token, old_token[:6], new_token[:6]):
            self.assertNotIn(secret, rendered)
        self.assertNotIn("token=", rendered.casefold())
        self.assertIn("session=", rendered)
        self.assertIn("port=27100", rendered)
        self.assertIn("port=27200", rendered)

    def test_connection_error_then_rebind_then_success(self) -> None:
        """First attempt gets ConnectionError; meanwhile launcher rewrites
        session_N.json with a new token; next attempt succeeds against the
        new token.
        """
        client = BridgeClient(session_path=self._session_path)
        call_count = {"n": 0}

        def fake_request(method, url, json=None, timeout=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Simulate launcher killing+restarting between call 1 and 2.
                _write_session(self._session_path, token=NEW_TOKEN, port=27200)
                self._force_mtime_bump()
                raise requests.ConnectionError("connection refused")
            # Call 2 should be against the new base_url + carry the new token.
            self.assertIn("27200", url)
            return _FakeResponse(status_code=200, payload={"ok": True})

        with patch.object(client._session, "request", side_effect=fake_request):
            # Patch sleep to keep the test fast.
            with patch.object(bridge_client.time, "sleep", lambda *_: None):
                result = client._request("GET", "health")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(client._token, NEW_TOKEN)
        self.assertEqual(call_count["n"], 2)

    def test_401_triggers_rebind_when_file_changed(self) -> None:
        """Bridge returns a non-specific 401 (e.g. generic auth rejection).
        If the session file has been rewritten, we should hot-reload and
        retry rather than give up after the legacy narrow error-code check.
        """
        client = BridgeClient(session_path=self._session_path)
        call_count = {"n": 0}

        def fake_request(method, url, json=None, timeout=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                _write_session(self._session_path, token=NEW_TOKEN, port=27200)
                self._force_mtime_bump()
                # Return a 401 with a generic body — NOT the legacy
                # "missing_or_invalid_token" code. Legacy code would have
                # raised; new code should rebind + retry.
                return _FakeResponse(
                    status_code=401,
                    payload={"error": "unauthorized"},
                )
            self.assertIn("27200", url)
            return _FakeResponse(status_code=200, payload={"ok": True})

        with patch.object(client._session, "request", side_effect=fake_request):
            with patch.object(bridge_client.time, "sleep", lambda *_: None):
                result = client._request("GET", "health")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(client._token, NEW_TOKEN)
        self.assertEqual(call_count["n"], 2)

    def test_connection_error_without_rewrite_uses_old_token(self) -> None:
        """Transient network blip with no session file change should simply
        retry with the existing credentials. No rebind, no regression.
        """
        client = BridgeClient(session_path=self._session_path)
        call_count = {"n": 0}
        original_mtime = client._session_mtime

        def fake_request(method, url, json=None, timeout=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise requests.ConnectionError("blip")
            self.assertIn("27100", url)  # still old port
            return _FakeResponse(status_code=200, payload={"ok": True})

        with patch.object(client._session, "request", side_effect=fake_request):
            with patch.object(bridge_client.time, "sleep", lambda *_: None):
                result = client._request("GET", "health")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(client._token, OLD_TOKEN)
        self.assertEqual(client._session_mtime, original_mtime)

    def test_gives_up_after_max_retries_even_with_rebind_attempts(self) -> None:
        """If rebind happens but the new bridge is also unreachable, we
        still eventually raise BridgeError (don't retry forever).
        """
        client = BridgeClient(session_path=self._session_path)
        # Disable outage-recovery wait so the test doesn't block for 180s.
        client.MAX_BRIDGE_OUTAGE_S = 0.0

        def always_fail(method, url, json=None, timeout=None):
            raise requests.ConnectionError("permanent")

        with patch.object(client._session, "request", side_effect=always_fail):
            with patch.object(bridge_client.time, "sleep", lambda *_: None):
                with self.assertRaises(BridgeError):
                    client._request("GET", "health")

    def test_outage_recovery_retries_past_max_retries_when_bridge_down(self) -> None:
        """2026-04-27: when the game crashes mid-training, watchdog needs
        ~30-60s to relaunch + write a new session.json. The 3x1s burst is
        too short to ride that out, so _request enters an outage-recovery
        wait that polls + retries until MAX_BRIDGE_OUTAGE_S elapses.
        """
        client = BridgeClient(session_path=self._session_path)
        client.MAX_BRIDGE_OUTAGE_S = 30.0  # cap test wall-clock cost
        client.BRIDGE_OUTAGE_POLL_S = 0.0  # no real sleep, advance via fake monotonic

        # Simulate: bridge down for first 6 attempts, then watchdog rewrites
        # session.json and bridge comes back. With burst=3, the only way to
        # reach attempt 7 is via the outage-recovery loop.
        call_count = {"n": 0}

        def fake_request(method, url, json=None, timeout=None):
            call_count["n"] += 1
            if call_count["n"] < 7:
                if call_count["n"] == 6:
                    # Watchdog finishes restart on attempt 6 — new session emitted.
                    _write_session(self._session_path, token=NEW_TOKEN, port=27200)
                    self._force_mtime_bump()
                raise requests.ConnectionError("connection refused")
            self.assertIn("27200", url)
            return _FakeResponse(status_code=200, payload={"ok": True})

        # monotonic_time advances 1s per call so deadline math reaches 30s
        # in finite iterations even with sleep stubbed out.
        fake_clock = {"t": 0.0}
        def fake_monotonic():
            fake_clock["t"] += 1.0
            return fake_clock["t"]

        with patch.object(client._session, "request", side_effect=fake_request):
            with patch.object(bridge_client.time, "sleep", lambda *_: None):
                with patch.object(bridge_client.time, "monotonic", fake_monotonic):
                    result = client._request("GET", "health")
        self.assertEqual(result, {"ok": True})
        self.assertGreaterEqual(call_count["n"], 7)
        self.assertEqual(client._token, NEW_TOKEN)

    def test_http_502_then_success_is_retried(self) -> None:
        """Combat sandbox resets can transiently return an empty HTTP 502.
        The trainer should ride out the blip instead of exiting.
        """
        client = BridgeClient(session_path=self._session_path)
        call_count = {"n": 0}

        def fake_request(method, url, json=None, timeout=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _FakeResponse(status_code=502, payload=None, text="")
            return _FakeResponse(status_code=200, payload={"ok": True})

        with patch.object(client._session, "request", side_effect=fake_request):
            with patch.object(bridge_client.time, "sleep", lambda *_: None):
                result = client._request("POST", "env/combat_reset", body={"timeout_ms": 1})

        self.assertEqual(result, {"ok": True})
        self.assertEqual(call_count["n"], 2)

    def test_repeated_http_502_eventually_raises_bridge_error(self) -> None:
        """Transient HTTP retry must be bounded; repeated gateway errors
        should raise BridgeError instead of spinning forever.
        """
        client = BridgeClient(session_path=self._session_path)
        client.MAX_BRIDGE_OUTAGE_S = 0.0
        call_count = {"n": 0}

        def always_502(method, url, json=None, timeout=None):
            call_count["n"] += 1
            return _FakeResponse(status_code=502, payload=None, text="bad gateway")

        with patch.object(client._session, "request", side_effect=always_502):
            with patch.object(bridge_client.time, "sleep", lambda *_: None):
                with self.assertRaises(BridgeError) as ctx:
                    client._request("POST", "env/combat_reset", body={"timeout_ms": 1})

        self.assertIn("Bridge unreachable", str(ctx.exception))
        self.assertGreaterEqual(call_count["n"], 13)


if __name__ == "__main__":
    unittest.main()
