"""Loopback-only HTTP server for the read-only STS2 training dashboard."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import threading
import webbrowser
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from sts2_rl.artifacts import ARTIFACT_ROOT_ENV, artifact_root
from sts2_rl.monitoring import DashboardStore

_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)
_MAX_QUERY_LENGTH = 2048


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_supported_bind_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.version == 4 and address.is_loopback


def _host_without_port(value: str) -> str:
    text = value.strip()
    if text.startswith("["):
        closing = text.find("]")
        return text[1:closing] if closing > 0 else ""
    if text.count(":") == 1:
        return text.rsplit(":", 1)[0]
    return text


def _dashboard_html() -> bytes:
    return (
        resources.files("sts2_rl")
        .joinpath("dashboard_static")
        .joinpath("index.html")
        .read_bytes()
    )


def _json_safe(value: object) -> object:
    """Replace non-standard JSON numbers before writing an HTTP response."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return str(value)


class DashboardHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying the immutable dashboard store."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        store: DashboardStore,
        html: bytes,
    ) -> None:
        self.store = store
        self.dashboard_html = html
        super().__init__(server_address, DashboardRequestHandler)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Serve a small same-origin UI and bounded JSON snapshots."""

    server: DashboardHTTPServer
    server_version = "STS2TrainingMonitor/1"
    sys_version = ""

    def version_string(self) -> str:
        return self.server_version

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the launcher terminal useful without printing a line for every
        # short polling request.
        if args and str(args[1]) not in {"200", "204"}:
            super().log_message(format, *args)

    def _host_allowed(self) -> bool:
        supplied = self.headers.get("Host", "")
        host = _host_without_port(supplied)
        return bool(host) and _is_loopback_host(host)

    def _common_headers(self, *, content_type: str, content_length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", _CSP)

    def _send_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        *,
        content_type: str,
        include_body: bool,
    ) -> None:
        self.send_response(status)
        self._common_headers(content_type=content_type, content_length=len(payload))
        self.end_headers()
        if include_body and payload:
            self.wfile.write(payload)

    def _send_json(
        self,
        status: HTTPStatus,
        value: object,
        *,
        include_body: bool,
    ) -> None:
        payload = json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(
            status,
            payload,
            content_type="application/json; charset=utf-8",
            include_body=include_body,
        )

    def _send_error_json(
        self,
        status: HTTPStatus,
        message: str,
        *,
        include_body: bool,
    ) -> None:
        self._send_json(
            status,
            {"error": status.phrase, "message": message},
            include_body=include_body,
        )

    def _dispatch(self, *, include_body: bool) -> None:
        if not self._host_allowed():
            self._send_error_json(
                HTTPStatus.FORBIDDEN,
                "Host must be localhost or a loopback address",
                include_body=include_body,
            )
            return
        if len(self.path) > _MAX_QUERY_LENGTH:
            self._send_error_json(
                HTTPStatus.REQUEST_URI_TOO_LONG,
                "request target is too long",
                include_body=include_body,
            )
            return
        parsed = urlsplit(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_bytes(
                HTTPStatus.OK,
                self.server.dashboard_html,
                content_type="text/html; charset=utf-8",
                include_body=include_body,
            )
            return
        if parsed.path == "/favicon.ico":
            self._send_bytes(
                HTTPStatus.NO_CONTENT,
                b"",
                content_type="image/x-icon",
                include_body=False,
            )
            return
        if parsed.path in {"/health", "/api/v1/health"}:
            runs_payload = self.server.store.list_runs()
            runs = runs_payload.get("runs")
            run_count = len(runs) if isinstance(runs, list) else 0
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "schema": "sts2-training-dashboard-v1",
                    "read_only": True,
                    "run_count": run_count,
                    "artifact_root": str(self.server.store.artifact_root),
                },
                include_body=include_body,
            )
            return
        if parsed.path == "/api/v1/runs":
            self._send_json(
                HTTPStatus.OK,
                self.server.store.list_runs(),
                include_body=include_body,
            )
            return
        if parsed.path == "/api/v1/snapshot":
            query = parse_qs(parsed.query, keep_blank_values=True)
            unknown_parameters = set(query).difference({"run"})
            if unknown_parameters or len(query.get("run", [])) > 1:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "snapshot accepts at most one server-issued run key",
                    include_body=include_body,
                )
                return
            run_values = query.get("run")
            run_key = run_values[0] if run_values and run_values[0] else None
            try:
                snapshot = self.server.store.snapshot(run_key)
            except KeyError:
                self._send_error_json(
                    HTTPStatus.NOT_FOUND,
                    "unknown run key; refresh /api/v1/runs",
                    include_body=include_body,
                )
                return
            self._send_json(HTTPStatus.OK, snapshot, include_body=include_body)
            return
        self._send_error_json(
            HTTPStatus.NOT_FOUND,
            "route not found",
            include_body=include_body,
        )

    def do_GET(self) -> None:
        self._dispatch(include_body=True)

    def do_HEAD(self) -> None:
        self._dispatch(include_body=False)

    def do_POST(self) -> None:
        self._send_error_json(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "the training dashboard is read-only",
            include_body=True,
        )

    def do_PUT(self) -> None:
        self.do_POST()

    def do_PATCH(self) -> None:
        self.do_POST()

    def do_DELETE(self) -> None:
        self.do_POST()

    def do_OPTIONS(self) -> None:
        self.do_POST()


def make_server(
    artifact_directory: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    stale_seconds: float = 15.0 * 60.0,
) -> DashboardHTTPServer:
    """Build a loopback-only server; useful for both CLI and tests."""

    if not _is_supported_bind_host(host):
        raise ValueError("dashboard host must be localhost or an IPv4 loopback address")
    store = DashboardStore(artifact_directory, stale_seconds=stale_seconds)
    return DashboardHTTPServer((host, port), store, _dashboard_html())


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the read-only local STS2 training dashboard.",
    )
    parser.add_argument(
        "--artifact-root",
        help=f"absolute runtime artifact root (or set {ARTIFACT_ROOT_ENV})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="loopback bind address")
    parser.add_argument("--port", type=int, default=8765, help="local TCP port")
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=15.0 * 60.0,
        help="seconds without metrics/evaluation/checkpoint activity before stale_unknown",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="open the dashboard in the default browser after binding",
    )
    return parser.parse_args(argv)


def _resolve_artifact_root(configured: str | None) -> Path:
    values = dict(os.environ)
    if configured:
        values[ARTIFACT_ROOT_ENV] = configured
    return artifact_root(environ=values)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not isinstance(args.host, str) or not _is_supported_bind_host(args.host):
        raise SystemExit("--host must be localhost or an IPv4 loopback address")
    if not isinstance(args.port, int) or not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    if not isinstance(args.stale_seconds, float) or args.stale_seconds < 60.0:
        raise SystemExit("--stale-seconds must be at least 60")
    try:
        root = _resolve_artifact_root(args.artifact_root)
        server = make_server(
            root,
            host=args.host,
            port=args.port,
            stale_seconds=args.stale_seconds,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot start dashboard: {exc}") from exc
    actual_host_value, actual_port = server.server_address[:2]
    actual_host = (
        actual_host_value.decode("ascii", errors="replace")
        if isinstance(actual_host_value, bytes)
        else str(actual_host_value)
    )
    display_host = "127.0.0.1" if actual_host == "localhost" else actual_host
    url = f"http://{display_host}:{actual_port}/"
    print(f"STS2 training dashboard: {url}", flush=True)
    print(f"Read-only artifact root: {root}", flush=True)
    if args.open_browser:
        opener = threading.Timer(0.2, webbrowser.open, args=(url,))
        opener.daemon = True
        opener.start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping dashboard.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


__all__ = [
    "DashboardHTTPServer",
    "DashboardRequestHandler",
    "main",
    "make_server",
]
