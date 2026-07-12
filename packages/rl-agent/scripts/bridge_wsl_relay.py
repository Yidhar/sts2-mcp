#!/usr/bin/env python3
"""Bounded, authenticated relay from a WSL-only interface to the loopback bridge."""

from __future__ import annotations

import argparse
import hmac
import http.client
import ipaddress
import json
import os
import socket
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

# The relay is a training transport, not a general-purpose proxy.  Legacy
# mutation endpoints and static export are deliberately absent.
ALLOWED_ROUTES: dict[str, frozenset[str]] = {
    "GET": frozenset({
        "/v2/health",
        "/v2/env/state",
        "/v2/env/spec",
        "/v2/env/combat_catalog",
    }),
    "POST": frozenset({
        "/v2/env/reset",
        "/v2/env/step",
    }),
}
TRAINING_ROUTES = frozenset({"/v2/env/reset", "/v2/env/step"})


def _filtered_headers(header_items: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        key: value
        for key, value in header_items
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
    }


def _bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    scheme, separator, token = value.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _token_allowed(token: str | None, allowed: frozenset[str]) -> bool:
    if token is None:
        return False
    return any(hmac.compare_digest(token, candidate) for candidate in allowed)


def _load_session_tokens(path: Path) -> tuple[frozenset[str], frozenset[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("auth session descriptor must be a JSON object")
    base_token = str(payload.get("token") or "").strip()
    capabilities = payload.get("capability_tokens")
    capability_tokens = (
        {str(key): str(value) for key, value in capabilities.items() if str(value).strip()}
        if isinstance(capabilities, dict)
        else {}
    )
    all_tokens = {base_token, *capability_tokens.values()}
    all_tokens.discard("")
    training_token = capability_tokens.get("training", "").strip()
    if not all_tokens:
        raise ValueError("auth session descriptor contains no bearer tokens")
    return frozenset(all_tokens), frozenset({training_token} if training_token else set())


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    request_queue_size = 16

    def __init__(
        self,
        server_address: tuple[str, int],
        target_base_url: str,
        target_timeout_s: float,
        state_file: Path | None,
        quiet: bool,
        *,
        allowed_tokens: frozenset[str],
        training_tokens: frozenset[str],
        allowed_client_cidrs: Iterable[str] = ("127.0.0.0/8", "::1/128"),
        max_body_bytes: int = 4 * 1024 * 1024,
        max_response_bytes: int = 32 * 1024 * 1024,
        max_concurrency: int = 8,
        instance_id: str = "",
    ) -> None:
        parsed = urlsplit(target_base_url)
        target_host = parsed.hostname or ""
        if parsed.scheme not in {"http", "https"} or target_host not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("target-base-url must be an HTTP(S) loopback endpoint")
        if not (0.1 <= float(target_timeout_s) <= 120.0):
            raise ValueError("target-timeout-s must be between 0.1 and 120 seconds")
        if max_body_bytes <= 0 or max_response_bytes <= 0:
            raise ValueError("body/response limits must be positive")
        if not (1 <= max_concurrency <= 64):
            raise ValueError("max-concurrency must be between 1 and 64")
        networks = tuple(ipaddress.ip_network(value, strict=False) for value in allowed_client_cidrs)
        if not networks:
            raise ValueError("at least one allowed client CIDR is required")
        if not allowed_tokens:
            raise ValueError("at least one relay bearer token is required")

        self.target_base_url = target_base_url.rstrip("/")
        self.target_timeout_s = float(target_timeout_s)
        self.state_file = state_file
        self.quiet = quiet
        self.target_scheme = parsed.scheme
        self.target_host = target_host
        self.target_port = parsed.port or (443 if self.target_scheme == "https" else 80)
        self.target_path_prefix = parsed.path.rstrip("/")
        self.allowed_tokens = frozenset(allowed_tokens)
        self.training_tokens = frozenset(training_tokens)
        self.allowed_client_networks = networks
        self.max_body_bytes = int(max_body_bytes)
        self.max_response_bytes = int(max_response_bytes)
        self.max_concurrency = int(max_concurrency)
        self._request_slots = threading.BoundedSemaphore(self.max_concurrency)
        self.instance_id = str(instance_id)
        self.process_started_at_utc = datetime.now(timezone.utc).isoformat()
        super().__init__(server_address, RelayHandler)

    @property
    def relay_base_url(self) -> str:
        host, port = self.server_address[:2]
        bracketed = f"[{host}]" if ":" in str(host) else host
        return f"http://{bracketed}:{port}/"

    def client_allowed(self, host: str) -> bool:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        return any(address in network for network in self.allowed_client_networks)

    def write_state(self) -> None:
        if self.state_file is None:
            return
        payload = {
            "ok": True,
            "pid": os.getpid(),
            "instance_id": self.instance_id,
            "process_started_at_utc": self.process_started_at_utc,
            "executable_path": str(Path(sys.executable).resolve()),
            "script_path": str(Path(__file__).resolve()),
            "base_url": self.relay_base_url,
            "listen_host": self.server_address[0],
            "listen_port": self.server_address[1],
            "target_base_url": self.target_base_url + "/",
            "allowed_client_cidrs": [str(network) for network in self.allowed_client_networks],
            "max_body_bytes": self.max_body_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_concurrency": self.max_concurrency,
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        temporary.replace(self.state_file)


class RelayHandler(BaseHTTPRequestHandler):
    server: RelayServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        if not self.server.quiet:
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def do_PATCH(self) -> None:
        self._method_not_allowed()

    def do_OPTIONS(self) -> None:
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self._send_json(405, {"ok": False, "error": "relay_method_not_allowed"})

    def _handle(self) -> None:
        if not self.server.client_allowed(self.client_address[0]):
            self._send_json(403, {"ok": False, "error": "relay_client_not_allowed"})
            return

        path = urlsplit(self.path).path
        if path in {"/_relay/health", "/_relay/health/"}:
            token = _bearer_token(self.headers.get("Authorization"))
            if not _token_allowed(token, self.server.allowed_tokens):
                self._send_json(401, {"ok": False, "error": "relay_unauthorized"})
                return
            self._send_json(
                200,
                {
                    "ok": True,
                    "pid": os.getpid(),
                    "instance_id": self.server.instance_id,
                    "relay_base_url": self.server.relay_base_url,
                    "target_base_url": self.server.target_base_url + "/",
                },
            )
            return

        if path not in ALLOWED_ROUTES.get(self.command, frozenset()):
            self._send_json(404, {"ok": False, "error": "relay_route_not_allowed"})
            return

        token = _bearer_token(self.headers.get("Authorization"))
        required_tokens = (
            self.server.training_tokens if path in TRAINING_ROUTES else self.server.allowed_tokens
        )
        if not _token_allowed(token, required_tokens):
            self._send_json(403, {"ok": False, "error": "relay_capability_not_allowed"})
            return

        if not self.server._request_slots.acquire(blocking=False):
            self._send_json(503, {"ok": False, "error": "relay_concurrency_limit"})
            return
        try:
            self._proxy(path)
        finally:
            self.server._request_slots.release()

    def _read_body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("chunked_request_not_supported")
        raw_length = self.headers.get("Content-Length")
        if self.command == "POST" and raw_length is None:
            raise ValueError("content_length_required")
        try:
            content_length = int(raw_length or "0")
        except ValueError as exc:
            raise ValueError("invalid_content_length") from exc
        if content_length < 0 or content_length > self.server.max_body_bytes:
            raise OverflowError("request_body_too_large")
        if self.command == "POST":
            content_type = self.headers.get_content_type()
            if content_type != "application/json":
                raise TypeError("content_type_must_be_application_json")
        return self.rfile.read(content_length) if content_length else None

    def _proxy(self, path: str) -> None:
        try:
            body = self._read_body()
        except OverflowError:
            self._send_json(413, {"ok": False, "error": "relay_request_too_large"})
            return
        except (TypeError, ValueError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
            return

        query = urlsplit(self.path).query
        upstream_path = f"{self.server.target_path_prefix}{path}"
        if query:
            upstream_path += f"?{query}"
        if not upstream_path.startswith("/"):
            upstream_path = "/" + upstream_path

        headers = _filtered_headers(self.headers.items())
        headers["Host"] = f"{self.server.target_host}:{self.server.target_port}"
        headers["X-Forwarded-For"] = self.client_address[0]
        headers["X-Forwarded-Proto"] = "http"
        headers["Connection"] = "close"

        connection_cls = (
            http.client.HTTPSConnection
            if self.server.target_scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_cls(
            self.server.target_host,
            self.server.target_port,
            timeout=self.server.target_timeout_s,
        )
        try:
            connection.request(self.command, upstream_path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read(self.server.max_response_bytes + 1)
            if len(payload) > self.server.max_response_bytes:
                self._send_json(502, {"ok": False, "error": "relay_upstream_response_too_large"})
                return

            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in HOP_BY_HOP_HEADERS or key.lower() == "content-length":
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            if payload:
                self.wfile.write(payload)
        except (socket.timeout, TimeoutError):
            self._send_json(504, {"ok": False, "error": "relay_upstream_timeout"})
        except Exception:
            self._send_json(502, {"ok": False, "error": "relay_upstream_error"})
        finally:
            connection.close()
            self.close_connection = True

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-base-url", required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-timeout-s", type=float, default=30.0)
    parser.add_argument("--state-file", type=str, default=None)
    parser.add_argument("--auth-session-file", type=str, required=True)
    parser.add_argument("--allow-client-cidr", action="append", default=[])
    parser.add_argument("--max-body-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-response-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_tokens, training_tokens = _load_session_tokens(
        resolve_external_input_path(args.auth_session_file)
    )
    state_file = resolve_artifact_path(args.state_file) if args.state_file else None
    client_cidrs = args.allow_client_cidr or ["127.0.0.0/8", "::1/128"]
    server = RelayServer(
        (args.listen_host, args.listen_port),
        target_base_url=args.target_base_url,
        target_timeout_s=args.target_timeout_s,
        state_file=state_file,
        quiet=args.quiet,
        allowed_tokens=allowed_tokens,
        training_tokens=training_tokens,
        allowed_client_cidrs=client_cidrs,
        max_body_bytes=args.max_body_bytes,
        max_response_bytes=args.max_response_bytes,
        max_concurrency=args.max_concurrency,
        instance_id=args.instance_id,
    )
    server.write_state()
    if not args.quiet:
        print(
            json.dumps(
                {
                    "ok": True,
                    "pid": os.getpid(),
                    "instance_id": args.instance_id,
                    "base_url": server.relay_base_url,
                    "target_base_url": server.target_base_url + "/",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
