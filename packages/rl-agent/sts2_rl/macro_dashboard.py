"""Loopback-only, read-only monitor for the synchronous macro-credit course.

No torch import, checkpoint deserialization, simulator, or training control.
Unlike the legacy dashboard this reads the selected course/status/cohort JSON
directly; it never synthesizes old metrics.jsonl or silently selects another run.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import urlsplit

from sts2_rl.artifacts import resolve_artifact_path

SCHEMA = "sts2-macro-credit-dashboard-v1"
COURSE_VERSION = "sts2-synchronous-macro-credit-course-v1"
MAX_JSON_BYTES = 1024 * 1024
MAX_TAIL_BYTES = 1024 * 1024
MAX_ATTEMPTS = 10000
MAX_HISTORY = 200
_CSP = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _open_artifact_read(path: Path) -> BinaryIO:
    """Do not block the trainer's atomic replacement of observed metadata.

    Ordinary Windows Python open() does not share delete access. An otherwise
    read-only dashboard could therefore make a concurrent os.replace fail.
    Explicitly share read/write/delete while requesting read access only.
    This supports the actual WSL/POSIX trainer's rename semantics. Native
    Windows MoveFileEx replacement of an open destination still differs and
    is not certified by this WSL-course adapter.
    """
    if os.name != "nt":
        return path.open("rb")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close = kernel.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        close(handle)
        raise
    try:
        return cast(BinaryIO, os.fdopen(descriptor, "rb"))
    except BaseException:
        os.close(descriptor)
        raise


class MacroDashboardStore:
    def __init__(self, run_dir: Path, *, artifact_root: Path, invocation_dir: Path | None = None) -> None:
        self.run_dir = resolve_artifact_path(run_dir, root=artifact_root)
        self.artifact_root = artifact_root.resolve()
        if not self.run_dir.is_dir():
            raise ValueError("macro dashboard requires an existing selected run directory")
        self.invocation_dir = resolve_artifact_path(invocation_dir, root=artifact_root) if invocation_dir else None
        self._lock = threading.RLock()
        course = self._read_json(self.run_dir / "course.json", self.run_dir)
        if _mapping(course.get("course_config")).get("version") != COURSE_VERSION:
            raise ValueError("selected run is not a synchronous macro-credit course")
        if not isinstance(course.get("lineage_id"), str) or not course["lineage_id"]:
            raise ValueError("macro course has no lineage_id")
        self.lineage_id = str(course["lineage_id"])

    @staticmethod
    def _safe(path: Path, boundary: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(boundary) or path.is_symlink():
            raise ValueError("dashboard refuses a path outside its fixed artifact boundary")
        return resolved

    def _read_json(self, path: Path, boundary: Path) -> dict[str, Any]:
        resolved = self._safe(path, boundary)
        with _open_artifact_read(resolved) as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("dashboard JSON size limit exceeded")
        parsed = json.loads(raw, parse_constant=_reject_constant)
        if not isinstance(parsed, dict):
            raise ValueError("dashboard metadata must be a JSON object")
        return parsed

    def _optional_json(self, path: Path, boundary: Path, warnings: list[str]) -> dict[str, Any]:
        try:
            return self._read_json(path, boundary)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            warnings.append(f"{path.name}: {type(error).__name__}; metadata not used")
            return {}

    def _tail_progress(self, path: Path) -> dict[str, Any]:
        resolved = self._safe(path, self.run_dir)
        with _open_artifact_read(resolved) as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - MAX_TAIL_BYTES)
            handle.seek(start)
            raw = handle.read(MAX_TAIL_BYTES)
        lines = raw.splitlines()
        if start and lines:
            lines.pop(0)
        if raw and not raw.endswith(b"\n") and lines:
            lines.pop()
        for line in reversed(lines):
            try:
                row = json.loads(line, parse_constant=_reject_constant)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(row, dict):
                continue
            facts = _mapping(row.get("facts"))
            run = _mapping(_mapping(_mapping(facts.get("state")).get("observation")).get("run"))
            semantic = _mapping(facts.get("semantic"))
            return {"record_kind": row.get("kind"), "floor": _integer(facts.get("ending_floor", run.get("floor"))),
                    "native_steps": _integer(facts.get("native_steps")),
                    "surface": semantic.get("surface"), "domain": semantic.get("domain")}
        return {}

    def _invocation(self, warnings: list[str]) -> dict[str, Any]:
        directory = self.invocation_dir
        if directory is None:
            return {}
        process = self._optional_json(directory / "process.json", directory, warnings)
        launch = self._optional_json(directory / "launch.json", directory, warnings)
        exited = self._optional_json(directory / "exit.json", directory, warnings)
        argv = process.get("argv")
        # An invocation must explicitly bind to this lineage. Do not show an
        # unrelated pilot's target/PIDs when a caller selects the wrong folder.
        if not isinstance(argv, list) or "--lineage-id" not in argv:
            warnings.append("invocation identity unavailable; invocation target not used")
            return {}
        index = argv.index("--lineage-id")
        if index + 1 >= len(argv) or argv[index + 1] != self.lineage_id:
            warnings.append("invocation lineage differs; invocation target not used")
            return {}
        logs: dict[str, Any] = {}
        for name in ("stdout.log", "stderr.log"):
            path = self._safe(directory / name, directory)
            if path.is_file():
                stat = path.stat()
                logs[name] = {"bytes": stat.st_size, "modified_unix_s": stat.st_mtime}
        return {"directory": str(directory), "starting_update": _integer(process.get("starting_update")),
                "target_update": _integer(process.get("maximum_target_update", launch.get("target_update"))),
                "maximum_additional_updates": _integer(process.get("maximum_additional_updates")),
                "wsl_training_pid_recorded": _integer(process.get("pid")),
                "windows_wsl_pid_recorded": _integer(launch.get("windows_wsl_pid")),
                "started_utc": process.get("started_utc"), "exit_record_present": bool(exited),
                "exit": {key: exited[key] for key in ("returncode", "exit_code", "ended_utc", "exited_utc", "status") if key in exited},
                "process_liveness_verified": False, "logs": logs}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            warnings: list[str] = []
            course = self._read_json(self.run_dir / "course.json", self.run_dir)
            if course.get("lineage_id") != self.lineage_id or _mapping(course.get("course_config")).get("version") != COURSE_VERSION:
                raise ValueError("selected course identity changed; refusing old/new-run confusion")
            config = _mapping(course.get("course_config"))
            status = self._optional_json(self.run_dir / "status.json", self.run_dir, warnings)
            attempts = list(islice((self.run_dir / "cohorts").glob("update-*-attempt-*"), MAX_ATTEMPTS + 1))
            if len(attempts) > MAX_ATTEMPTS:
                warnings.append("attempt discovery limit exceeded; history is incomplete")
            attempts = [path for path in attempts[:MAX_ATTEMPTS] if path.is_dir() and not path.is_symlink()]
            attempts.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            history: list[dict[str, Any]] = []
            for attempt in attempts[:MAX_HISTORY]:
                update = self._optional_json(attempt / "update.json", self.run_dir, warnings)
                if not update:
                    continue
                metrics = _mapping(update.get("learner"))
                version = _integer(update.get("committed_update"))
                if version is None or metrics.get("version") != "sts2-fresh-macro-credit-learner-v1":
                    warnings.append(f"{attempt.name}: incompatible update metrics ignored")
                    continue
                scalar_keys = (
                    "loss", "macro_loss", "combat_loss", "parameter_delta_norm", "gradient_norm",
                    "maximum_likelihood_error", "max_autograd_steps", "backward_chunks", "policy_lag",
                    "macro_steps", "combat_steps", "cohort_storage_bytes", "elapsed_seconds",
                    "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "process_peak_rss_bytes",
                    "mean_updated_policy_kl_to_teacher", "mean_root_kl",
                )
                history.append({"update": version, "next_seed_index": _integer(update.get("next_seed_index")),
                                "attempt": attempt.name, "coverage": _mapping(update.get("coverage")),
                                "metrics": {key: _number(metrics.get(key)) for key in scalar_keys},
                                "root_targets": metrics.get("root_targets", [])})
            history.sort(key=lambda item: item["update"])
            raw_attempt = status.get("attempt")
            attempt_name = Path(raw_attempt.replace("\\", "/")).name if isinstance(raw_attempt, str) else None
            active = next((path for path in attempts if path.name == attempt_name), attempts[0] if attempts else None)
            progress: dict[str, Any] = {}
            latest_activity = (self.run_dir / "status.json").stat().st_mtime if (self.run_dir / "status.json").is_file() else 0.0
            if active:
                traces = [path for path in islice(active.glob("root-*/*.jsonl"), 1024)
                          if path.is_file() and not path.is_symlink() and ".prefix." not in path.name]
                if traces:
                    newest = max(traces, key=lambda path: path.stat().st_mtime)
                    stat = newest.stat()
                    latest_activity = max(latest_activity, stat.st_mtime)
                    progress = {"attempt": active.name, "trace": str(newest.relative_to(self.run_dir)),
                                "trace_bytes": stat.st_size, "trace_modified_unix_s": stat.st_mtime,
                                "trace_count": len(traces)}
                    try:
                        progress.update(self._tail_progress(newest))
                    except (OSError, ValueError):
                        warnings.append("latest trace is unavailable or oversized; progress inferred from file stat only")
            latest = history[-1] if history else None
            committed = _integer(status.get("committed_update"))
            if committed is None and latest:
                committed = latest["update"]
            cursor = _integer(status.get("next_seed_index"))
            if cursor is None and latest:
                cursor = latest["next_seed_index"]
            writer = (self.run_dir / ".writer.lock").exists()
            phase = str(status.get("state", "status_unavailable"))
            if phase in {"collecting", "updating", "initializing"} and not writer:
                warnings.append("active phase is recorded but writer lock is absent; process may have stopped")
            if committed is not None and latest and committed > latest["update"]:
                warnings.append("committed status is ahead of visible update metrics; checkpoint remains recovery authority")
            invocation = self._invocation(warnings)
            return {"schema": SCHEMA, "read_only": True, "server_pid": os.getpid(),
                    "observed_at_utc": datetime.now(UTC).isoformat(), "run_dir": str(self.run_dir),
                    "lineage_id": self.lineage_id, "course_version": COURSE_VERSION,
                    "state": phase, "committed_update": committed, "next_seed_index": cursor,
                    "next_training_seed": 2 * (6700000 + cursor) if cursor is not None else None,
                    "writer_lock_present": writer, "process_liveness_verified": False,
                    "activity_age_seconds": max(0.0, now - latest_activity) if latest_activity else None,
                    "error_type": status.get("error_type"),
                    "error": str(status["error"])[:4096] if status.get("error") is not None else None,
                    "invocation": invocation,
                    "root_surface": _mapping(config.get("branches")).get("root_surface"),
                    "macro_horizon": _mapping(config.get("branches")).get("continuation_macro_decisions"),
                    "revival_budget": _mapping(course.get("reward_identity")).get("revival_budget"),
                    "device": _mapping(_mapping(config.get("training")).get("runtime")).get("device"),
                    "progress": progress, "latest_update": latest, "history": history,
                    "history_limit": MAX_HISTORY, "attempts_observed": len(attempts), "warnings": warnings,
                    "measurement_scope": {"gpu_peaks": "learner window only; not full-course peak",
                                          "rss": "process historical high-water mark", "quality_certified": False}}


class MacroDashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: MacroDashboardStore) -> None:
        self.store = store
        self.html = (Path(__file__).parent / "dashboard_static/macro-credit.html").read_bytes()
        super().__init__(address, MacroDashboardHandler)


class MacroDashboardHandler(BaseHTTPRequestHandler):
    server: MacroDashboardServer
    server_version = "STS2MacroCreditMonitor/1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # No per-poll output or raw request logging.

    def _send(self, status: int, value: dict[str, Any] | bytes, *, head: bool = False) -> None:
        is_html = isinstance(value, bytes)
        content = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8" if is_html else "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", _CSP)
        self.end_headers()
        if not head:
            self.wfile.write(content)

    def _get(self, *, head: bool = False) -> None:
        try:
            host = urlsplit("//" + self.headers.get("Host", "")).hostname or ""
        except ValueError:
            host = ""
        if not _loopback(host):
            self._send(HTTPStatus.FORBIDDEN, {"error": "loopback Host required"}, head=head)
            return
        route = urlsplit(self.path)
        if len(self.path) > 2048 or route.query:
            self._send(HTTPStatus.BAD_REQUEST, {"error": "no query or file selection is supported"}, head=head)
        elif route.path in {"/", "/index.html"}:
            self._send(HTTPStatus.OK, self.server.html, head=head)
        elif route.path in {"/health", "/api/v1/health"}:
            self._send(HTTPStatus.OK, {"schema": SCHEMA, "status": "ok", "read_only": True,
                                      "run_dir": str(self.server.store.run_dir), "server_pid": os.getpid(),
                                      "lineage_id": self.server.store.lineage_id}, head=head)
        elif route.path == "/api/v1/snapshot":
            try:
                self._send(HTTPStatus.OK, self.server.store.snapshot(), head=head)
            except (OSError, ValueError) as error:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"schema": SCHEMA, "error": type(error).__name__,
                                                          "message": "selected-run metadata is unavailable; no fallback run"}, head=head)
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "route not found"}, head=head)

    def do_GET(self) -> None:
        self._get()

    def do_HEAD(self) -> None:
        self._get(head=True)

    def do_POST(self) -> None:
        self._send(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "read-only dashboard; no training controls"})

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST
    do_OPTIONS = do_POST


def make_server(run_dir: Path, *, artifact_root: Path, invocation_dir: Path | None = None,
                host: str = "127.0.0.1", port: int = 8766) -> MacroDashboardServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("macro dashboard binds IPv4 loopback only")
    return MacroDashboardServer((host, port), MacroDashboardStore(run_dir, artifact_root=artifact_root,
                                                                invocation_dir=invocation_dir))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--invocation-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    server = make_server(args.run_dir, artifact_root=args.artifact_root, invocation_dir=args.invocation_dir,
                         host=args.host, port=args.port)
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}/", "schema": SCHEMA,
                      "run_dir": str(server.store.run_dir), "pid": os.getpid(), "read_only": True}), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
