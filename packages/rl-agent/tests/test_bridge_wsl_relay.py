from __future__ import annotations

import http.client
import importlib.util
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bridge_wsl_relay.py"
SPEC = importlib.util.spec_from_file_location("bridge_wsl_relay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
relay_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relay_module)
RelayServer = relay_module.RelayServer


class UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return None

    def _reply(self, payload: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        if self.path == "/large":
            self._reply(b"x" * 512)
            return
        if self.path == "/slow":
            time.sleep(0.3)
        self._reply(json.dumps({"path": self.path}).encode())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self._reply(body)


@contextmanager
def running_server(server: ThreadingHTTPServer) -> Iterator[ThreadingHTTPServer]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(
    relay: RelayServer,
    method: str,
    path: str,
    *,
    token: str = "base-token",
    body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, bytes]:
    host, port = relay.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=2)
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    status = response.status
    connection.close()
    return status, payload


@contextmanager
def relay_for(
    upstream: ThreadingHTTPServer,
    **overrides: object,
) -> Iterator[RelayServer]:
    host, port = upstream.server_address[:2]
    kwargs: dict[str, object] = {
        "allowed_tokens": frozenset({"base-token", "training-token"}),
        "training_tokens": frozenset({"training-token"}),
        "allowed_client_cidrs": ("127.0.0.0/8",),
        "max_body_bytes": 32,
        "max_response_bytes": 128,
        "max_concurrency": 2,
        "instance_id": "instance",
    }
    kwargs.update(overrides)
    relay = RelayServer(
        ("127.0.0.1", 0),
        target_base_url=f"http://{host}:{port}",
        target_timeout_s=float(kwargs.pop("target_timeout_s", 1.0)),
        state_file=None,
        quiet=True,
        **kwargs,
    )
    with running_server(relay):
        yield relay


@pytest.fixture
def upstream() -> Iterator[ThreadingHTTPServer]:
    with running_server(ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)) as server:
        yield server


def test_relay_allows_only_explicit_routes_methods_and_capabilities(upstream: ThreadingHTTPServer) -> None:
    with relay_for(upstream) as relay:
        status, _ = request(relay, "GET", "/v2/health")
        assert status == 200
        status, _ = request(relay, "GET", "/not-allowed")
        assert status == 404
        status, _ = request(relay, "PUT", "/v2/health")
        assert status == 405
        status, _ = request(
            relay,
            "POST",
            "/v2/env/step",
            token="base-token",
            body=b"{}",
            content_type="application/json",
        )
        assert status == 403
        status, payload = request(
            relay,
            "POST",
            "/v2/env/step",
            token="training-token",
            body=b"{}",
            content_type="application/json",
        )
        assert status == 200
        assert payload == b"{}"


def test_relay_rejects_oversized_or_wrong_type_bodies(upstream: ThreadingHTTPServer) -> None:
    with relay_for(upstream, max_body_bytes=4) as relay:
        status, _ = request(
            relay,
            "POST",
            "/v2/env/reset",
            token="training-token",
            body=b"12345",
            content_type="application/json",
        )
        assert status == 413
        status, _ = request(
            relay,
            "POST",
            "/v2/env/reset",
            token="training-token",
            body=b"{}",
            content_type="text/plain",
        )
        assert status == 400


def test_relay_rejects_disallowed_clients_and_excess_concurrency(upstream: ThreadingHTTPServer) -> None:
    with relay_for(upstream, allowed_client_cidrs=("192.0.2.0/24",)) as relay:
        status, _ = request(relay, "GET", "/v2/health")
        assert status == 403

    with relay_for(upstream, max_concurrency=1) as relay:
        assert relay._request_slots.acquire(blocking=False)
        try:
            status, _ = request(relay, "GET", "/v2/health")
            assert status == 503
        finally:
            relay._request_slots.release()


def test_relay_bounds_upstream_response_and_timeout(upstream: ThreadingHTTPServer) -> None:
    # Temporarily expose test-only paths to exercise the generic bound logic.
    original = relay_module.ALLOWED_ROUTES
    relay_module.ALLOWED_ROUTES = {
        **original,
        "GET": original["GET"] | {"/large", "/slow"},
    }
    try:
        with relay_for(upstream, max_response_bytes=64) as relay:
            status, _ = request(relay, "GET", "/large")
            assert status == 502
        with relay_for(upstream, target_timeout_s=0.1) as relay:
            status, _ = request(relay, "GET", "/slow")
            assert status == 504
    finally:
        relay_module.ALLOWED_ROUTES = original


def test_relay_state_has_identity_and_no_token(tmp_path: Path, upstream: ThreadingHTTPServer) -> None:
    state_file = tmp_path / "relay.json"
    host, port = upstream.server_address[:2]
    relay = RelayServer(
        ("127.0.0.1", 0),
        target_base_url=f"http://{host}:{port}",
        target_timeout_s=1.0,
        state_file=state_file,
        quiet=True,
        allowed_tokens=frozenset({"secret"}),
        training_tokens=frozenset({"secret"}),
        instance_id="nonce",
    )
    try:
        relay.write_state()
        payload = json.loads(state_file.read_text(encoding="utf-8"))
        assert payload["pid"] > 0
        assert payload["instance_id"] == "nonce"
        assert payload["process_started_at_utc"]
        assert Path(payload["executable_path"]).is_absolute()
        assert Path(payload["script_path"]).resolve() == SCRIPT.resolve()
        assert "secret" not in state_file.read_text(encoding="utf-8")
    finally:
        relay.server_close()


def test_target_must_remain_loopback() -> None:
    with pytest.raises(ValueError, match="loopback"):
        RelayServer(
            ("127.0.0.1", 0),
            target_base_url="http://192.0.2.1:9000",
            target_timeout_s=1.0,
            state_file=None,
            quiet=True,
            allowed_tokens=frozenset({"token"}),
            training_tokens=frozenset({"token"}),
        )
