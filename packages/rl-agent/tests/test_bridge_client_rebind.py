"""Tests for BridgeClient auto-rebind after a launcher-side instance restart.

Scenario: launcher watchdog kills a hung bridge, relaunches the game process,
and the bridge mod writes a FRESH session_N.json with a new token + new port.
The BridgeClient instance owned by the (still-alive) RL training worker must
detect the file change and swap to the new credentials mid-request.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import requests


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))


# Load bridge_client.py directly, bypassing sts2_env/__init__.py which pulls
# in torch-dependent modules (AuxMaskablePPO) that aren't available in the
# Windows test venv. This mirrors the pattern used by other tests that want
# to exercise pure-Python logic in sts2_env/ without the ML stack.
_BRIDGE_CLIENT_SRC = RL_AGENT_ROOT / "sts2_env" / "bridge_client.py"
_spec = importlib.util.spec_from_file_location(
    "bridge_client_under_test", str(_BRIDGE_CLIENT_SRC),
)
# The module imports from sibling `path_utils` via `.path_utils`; to satisfy
# the relative import without triggering __init__.py, install a minimal
# sts2_env package shim before loading.
if "sts2_env" not in sys.modules:
    pkg = types.ModuleType("sts2_env")
    pkg.__path__ = [str(RL_AGENT_ROOT / "sts2_env")]
    sys.modules["sts2_env"] = pkg
_path_utils_src = RL_AGENT_ROOT / "sts2_env" / "path_utils.py"
_pu_spec = importlib.util.spec_from_file_location(
    "sts2_env.path_utils", str(_path_utils_src),
)
_pu_mod = importlib.util.module_from_spec(_pu_spec)
sys.modules["sts2_env.path_utils"] = _pu_mod
_pu_spec.loader.exec_module(_pu_mod)
_bc_spec = importlib.util.spec_from_file_location(
    "sts2_env.bridge_client", str(_BRIDGE_CLIENT_SRC),
)
bridge_client = importlib.util.module_from_spec(_bc_spec)
sys.modules["sts2_env.bridge_client"] = bridge_client
_bc_spec.loader.exec_module(bridge_client)
BridgeClient = bridge_client.BridgeClient
BridgeError = bridge_client.BridgeError


def _write_session(path: Path, *, token: str, port: int) -> None:
    path.write_text(
        json.dumps({
            "base_url": f"http://127.0.0.1:{port}/",
            "token": token,
            "pid": 12345,
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
        _write_session(self._session_path, token="OLD_TOKEN", port=27100)

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
        self.assertEqual(client._token, "OLD_TOKEN")
        self.assertGreater(client._session_mtime, 0.0)

    def test_rebind_noop_when_file_unchanged(self) -> None:
        client = BridgeClient(session_path=self._session_path)
        rebound = client._maybe_rebind_session(reason="unit_test")
        self.assertFalse(rebound)
        self.assertEqual(client._token, "OLD_TOKEN")

    def test_rebind_picks_up_new_token_after_file_rewrite(self) -> None:
        client = BridgeClient(session_path=self._session_path)
        _write_session(self._session_path, token="NEW_TOKEN", port=27200)
        self._force_mtime_bump()
        rebound = client._maybe_rebind_session(reason="unit_test")
        self.assertTrue(rebound)
        self.assertEqual(client._token, "NEW_TOKEN")
        self.assertIn("27200", client._base_url)

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
                _write_session(self._session_path, token="NEW_TOKEN", port=27200)
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
        self.assertEqual(client._token, "NEW_TOKEN")
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
                _write_session(self._session_path, token="NEW_TOKEN", port=27200)
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
        self.assertEqual(client._token, "NEW_TOKEN")
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
        self.assertEqual(client._token, "OLD_TOKEN")
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
                    _write_session(self._session_path, token="NEW_TOKEN", port=27200)
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
        self.assertEqual(client._token, "NEW_TOKEN")

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
