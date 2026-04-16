#!/usr/bin/env python3
"""Expose the Windows-only STS2 loopback bridge to WSL via a user-space relay.

The bridge mod currently writes ``http://127.0.0.1:<port>/`` into session.json.
When WSL localhost forwarding is unavailable, Linux-side training cannot reach
that loopback listener. This relay binds a normal Windows TCP socket on a
WSL-reachable host IP/port and forwards HTTP requests to the local bridge.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

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


def _filtered_headers(header_items: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        key: value
        for key, value in header_items
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
    }


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        target_base_url: str,
        target_timeout_s: float,
        state_file: Path | None,
        quiet: bool,
    ):
        self.target_base_url = target_base_url.rstrip("/")
        self.target_timeout_s = target_timeout_s
        self.state_file = state_file
        self.quiet = quiet
        parsed = urlsplit(self.target_base_url)
        self.target_scheme = parsed.scheme or "http"
        self.target_host = parsed.hostname or "127.0.0.1"
        self.target_port = parsed.port or (443 if self.target_scheme == "https" else 80)
        self.target_path_prefix = parsed.path.rstrip("/")
        super().__init__(server_address, RelayHandler)

    @property
    def relay_base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}/"

    def write_state(self) -> None:
        if self.state_file is None:
            return
        payload = {
            "ok": True,
            "pid": os.getpid(),
            "base_url": self.relay_base_url,
            "listen_host": self.server_address[0],
            "listen_port": self.server_address[1],
            "target_base_url": self.target_base_url + "/",
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class RelayHandler(BaseHTTPRequestHandler):
    server: RelayServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        if not self.server.quiet:
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def do_OPTIONS(self) -> None:
        self._handle()

    def _handle(self) -> None:
        if self.path in {"/_relay/health", "/_relay/health/"}:
            self._send_json(
                200,
                {
                    "ok": True,
                    "pid": os.getpid(),
                    "relay_base_url": self.server.relay_base_url,
                    "target_base_url": self.server.target_base_url + "/",
                },
            )
            return

        content_length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(content_length) if content_length > 0 else None

        upstream_path = f"{self.server.target_path_prefix}{self.path}"
        if not upstream_path.startswith("/"):
            upstream_path = "/" + upstream_path

        headers = _filtered_headers(self.headers.items())
        headers["Host"] = f"{self.server.target_host}:{self.server.target_port}"
        headers["X-Forwarded-For"] = self.client_address[0]
        headers["X-Forwarded-Proto"] = "http"

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
            payload = response.read()

            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in HOP_BY_HOP_HEADERS:
                    continue
                if key.lower() == "content-length":
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if payload:
                self.wfile.write(payload)
        except Exception as exc:
            self._send_json(
                502,
                {
                    "ok": False,
                    "error": "relay_upstream_error",
                    "detail": str(exc),
                    "target_base_url": self.server.target_base_url + "/",
                },
            )
        finally:
            connection.close()

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSL relay for the Windows-only STS2 loopback bridge.")
    parser.add_argument("--target-base-url", required=True, help="Bridge base URL from session.json")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-timeout-s", type=float, default=95.0)
    parser.add_argument("--state-file", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_file = Path(args.state_file) if args.state_file else None
    server = RelayServer(
        (args.listen_host, args.listen_port),
        target_base_url=args.target_base_url,
        target_timeout_s=args.target_timeout_s,
        state_file=state_file,
        quiet=args.quiet,
    )
    server.write_state()
    if not args.quiet:
        print(
            json.dumps(
                {
                    "ok": True,
                    "pid": os.getpid(),
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


if __name__ == "__main__":
    main()
