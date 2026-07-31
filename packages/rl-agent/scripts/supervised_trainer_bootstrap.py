#!/usr/bin/env python3
"""Fail-closed Linux bootstrap for a supervised trainer process.

The bootstrap deliberately imports only the Python standard library.  It arms a
Linux parent-death signal before acknowledging readiness, then waits for one
canonical GO frame on an inherited pipe.  Only after revalidating the parent
process identity does it exec the exact trainer argv in-place.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import re
import selectors
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - exercised only by Windows collection.
    fcntl = None  # type: ignore[assignment]

PROTOCOL_VERSION = "sts2-supervised-trainer-bootstrap-v1"
READY_KIND = "READY"
GO_KIND = "GO"
ERROR_KIND = "ERROR"

MAX_FRAME_BYTES = 4096
_NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
_PR_SET_PDEATHSIG = 1
_PR_GET_PDEATHSIG = 2
_SIGKILL = int(getattr(signal, "SIGKILL", 9))


class BootstrapError(RuntimeError):
    """A fail-closed bootstrap or protocol validation failure."""


def canonical_frame(mapping: Mapping[str, object]) -> bytes:
    """Return the one-line canonical JSON wire representation of *mapping*."""

    if not isinstance(mapping, Mapping):
        raise BootstrapError("protocol frame must be a mapping")
    try:
        payload = (
            json.dumps(
                dict(mapping),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BootstrapError("protocol frame is not canonical-JSON encodable") from exc
    if len(payload) > MAX_FRAME_BYTES:
        raise BootstrapError(f"protocol frame exceeds {MAX_FRAME_BYTES} bytes")
    return payload


def _strict_json_object(payload: bytes) -> dict[str, object]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BootstrapError(f"duplicate protocol field: {key}")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload.decode("ascii"), object_pairs_hook=reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("protocol frame is not valid ASCII JSON") from exc
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise BootstrapError("protocol frame must decode to an object")
    result = dict(decoded)
    if canonical_frame(result) != payload + b"\n":
        raise BootstrapError("protocol frame is not in canonical encoding")
    return result


def read_bounded_frame(fd: int, timeout_s: float, *, require_eof: bool = False) -> dict[str, object]:
    """Read exactly one bounded canonical frame from *fd*.

    ``require_eof`` is used for the one-shot GO pipe.  It prevents a valid
    prefix followed by delayed extra data from authorizing exec.  READY readers
    leave it false because the status pipe intentionally stays open until exec.
    """

    if isinstance(fd, bool) or not isinstance(fd, int) or fd < 0:
        raise BootstrapError("protocol fd must be a non-negative integer")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float) or not (0.0 < timeout_s <= 300.0):
        raise BootstrapError("protocol timeout must be in (0, 300] seconds")

    deadline = time.monotonic() + float(timeout_s)
    data = bytearray()
    newline_seen = False
    selector = selectors.DefaultSelector()
    try:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BootstrapError("protocol frame timed out")
            if not selector.select(remaining):
                raise BootstrapError("protocol frame timed out")
            try:
                chunk = os.read(fd, min(1024, MAX_FRAME_BYTES + 1 - len(data)))
            except InterruptedError:
                continue
            except OSError as exc:
                raise BootstrapError(f"protocol read failed: errno={exc.errno}") from exc
            if not chunk:
                if not newline_seen:
                    raise BootstrapError("protocol pipe reached EOF before a complete frame")
                break
            data.extend(chunk)
            if len(data) > MAX_FRAME_BYTES:
                raise BootstrapError(f"protocol frame exceeds {MAX_FRAME_BYTES} bytes")
            newline_index = data.find(b"\n")
            if newline_index >= 0:
                newline_seen = True
                if newline_index != len(data) - 1:
                    raise BootstrapError("protocol pipe contained trailing or multiple frames")
                if not require_eof:
                    break
                # In EOF-required mode, continue waiting.  Any further byte is
                # rejected above; only closure authorizes parsing the frame.
    finally:
        selector.close()

    if not data or data[-1:] != b"\n":
        raise BootstrapError("protocol frame is incomplete")
    return _strict_json_object(bytes(data[:-1]))


def process_start_ticks(pid: int) -> int:
    """Return Linux ``/proc/<pid>/stat`` field 22 (process start ticks)."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise BootstrapError("pid must be a positive integer")
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            stat = handle.read()
    except (OSError, UnicodeError) as exc:
        raise BootstrapError(f"cannot read process identity for pid {pid}") from exc
    right_paren = stat.rfind(")")
    if right_paren < 0:
        raise BootstrapError(f"malformed /proc/{pid}/stat")
    fields_after_comm = stat[right_paren + 2 :].split()
    # The first entry after ``comm`` is field 3; starttime is field 22.
    if len(fields_after_comm) <= 19:
        raise BootstrapError(f"malformed /proc/{pid}/stat")
    try:
        value = int(fields_after_comm[19])
    except ValueError as exc:
        raise BootstrapError(f"malformed start ticks for pid {pid}") from exc
    if value <= 0:
        raise BootstrapError(f"invalid start ticks for pid {pid}")
    return value


def _prctl(option: int, argument: int = 0) -> int:
    if sys.platform != "linux":
        raise BootstrapError("supervised trainer bootstrap requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    result = int(prctl(option, argument, 0, 0, 0))
    if result != 0:
        error_number = ctypes.get_errno()
        raise BootstrapError(f"prctl({option}) failed: errno={error_number}")
    return result


def _get_parent_death_signal() -> int:
    value = ctypes.c_int(0)
    if sys.platform != "linux":
        raise BootstrapError("supervised trainer bootstrap requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    result = int(prctl(_PR_GET_PDEATHSIG, ctypes.byref(value), 0, 0, 0))
    if result != 0:
        error_number = ctypes.get_errno()
        raise BootstrapError(f"prctl({_PR_GET_PDEATHSIG}) failed: errno={error_number}")
    return int(value.value)


def _require_parent_identity(expected_pid: int, expected_start_ticks: int) -> None:
    actual_pid = os.getppid()
    if actual_pid != expected_pid:
        raise BootstrapError(f"parent pid changed: expected={expected_pid}, actual={actual_pid}")
    actual_ticks = process_start_ticks(expected_pid)
    if actual_ticks != expected_start_ticks:
        raise BootstrapError(
            f"parent process identity changed: expected_start_ticks={expected_start_ticks}, actual={actual_ticks}"
        )


def arm_parent_death_guard(expected_parent_pid: int, expected_parent_start_ticks: int) -> None:
    """Arm SIGKILL-on-parent-death and close the set/check race fail-closed."""

    _prctl(_PR_SET_PDEATHSIG, _SIGKILL)
    if _get_parent_death_signal() != _SIGKILL:
        raise BootstrapError("kernel did not retain SIGKILL parent-death signal")
    # Linux does not retroactively deliver PDEATHSIG when the parent exits
    # between fork/exec and PR_SET_PDEATHSIG.  Checking both PID and start ticks
    # after arming closes that race and also prevents PID-reuse confusion.
    _require_parent_identity(expected_parent_pid, expected_parent_start_ticks)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        except OSError as exc:
            raise BootstrapError(f"protocol write failed: errno={exc.errno}") from exc
        if written <= 0:
            raise BootstrapError("protocol write made no progress")
        view = view[written:]


def _validate_nonce(nonce: str) -> str:
    if not _NONCE_RE.fullmatch(nonce):
        raise BootstrapError("nonce must be exactly 64 lowercase hexadecimal characters")
    return nonce


def _validate_go(frame: Mapping[str, object], nonce: str) -> None:
    expected = {"protocol": PROTOCOL_VERSION, "kind": GO_KIND, "nonce": nonce}
    if dict(frame) != expected:
        raise BootstrapError("GO frame did not exactly match protocol, kind, and nonce")


def _set_close_on_exec(fd: int) -> None:
    if fcntl is None:
        raise BootstrapError("close-on-exec setup requires Linux fcntl")
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFD)
        fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
    except OSError as exc:
        raise BootstrapError(f"cannot mark fd {fd} close-on-exec: errno={exc.errno}") from exc


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    parser.add_argument("--expected-parent-start-ticks", type=int, required=True)
    parser.add_argument("--gate-fd", type=int, required=True)
    parser.add_argument("--status-fd", type=int, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    result = parser.parse_args(argv)
    if not result.command:
        parser.error("an exact trainer argv is required after --")
    if result.command[0] == "--":
        result.command = result.command[1:]
    if not result.command or not result.command[0]:
        parser.error("trainer argv[0] must be non-empty")
    return result


def _error_frame(exc: BaseException) -> bytes:
    return canonical_frame(
        {
            "kind": ERROR_KIND,
            "message": str(exc)[:1024],
            "nonce": "unavailable",
            "pid": os.getpid(),
            "protocol": PROTOCOL_VERSION,
        }
    )


def run_bootstrap(args: argparse.Namespace) -> NoReturn:
    gate_fd = int(args.gate_fd)
    status_fd = int(args.status_fd)
    if gate_fd < 3 or status_fd < 3 or gate_fd == status_fd:
        raise BootstrapError("gate/status fds must be distinct inherited descriptors >= 3")
    nonce = _validate_nonce(str(args.nonce))
    timeout_s = float(args.timeout_seconds)
    if not (0.0 < timeout_s <= 300.0):
        raise BootstrapError("timeout must be in (0, 300] seconds")
    expected_parent_pid = int(args.expected_parent_pid)
    expected_parent_start_ticks = int(args.expected_parent_start_ticks)
    if expected_parent_pid <= 0 or expected_parent_start_ticks <= 0:
        raise BootstrapError("expected parent pid/start ticks must be positive")

    arm_parent_death_guard(expected_parent_pid, expected_parent_start_ticks)
    bootstrap_start_ticks = process_start_ticks(os.getpid())
    ready = {
        "kind": READY_KIND,
        "nonce": nonce,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "proc_start_ticks": bootstrap_start_ticks,
        "protocol": PROTOCOL_VERSION,
    }
    _write_all(status_fd, canonical_frame(ready))

    go = read_bounded_frame(gate_fd, timeout_s, require_eof=True)
    _validate_go(go, nonce)
    _require_parent_identity(expected_parent_pid, expected_parent_start_ticks)
    if _get_parent_death_signal() != _SIGKILL:
        raise BootstrapError("SIGKILL parent-death guard was lost before exec")
    if process_start_ticks(os.getpid()) != bootstrap_start_ticks:
        raise BootstrapError("bootstrap process identity changed before exec")

    # Both descriptors were inherited explicitly by the bootstrap.  Closing the
    # gate now and marking status CLOEXEC guarantees the final trainer receives
    # neither protocol descriptor.  Status-pipe EOF is the supervisor's exec ACK.
    try:
        os.close(gate_fd)
    except OSError as exc:
        if exc.errno != errno.EBADF:
            raise BootstrapError(f"cannot close gate fd: errno={exc.errno}") from exc
    _set_close_on_exec(status_fd)
    os.execve(str(args.command[0]), [str(item) for item in args.command], dict(os.environ))
    raise AssertionError("os.execve unexpectedly returned")


def main(argv: Sequence[str] | None = None) -> int:
    status_fd: int | None = None
    try:
        args = _parse_args(argv)
        status_fd = int(args.status_fd)
        run_bootstrap(args)
    except (BootstrapError, OSError, ValueError) as exc:
        if status_fd is not None and status_fd >= 0:
            try:
                _write_all(status_fd, _error_frame(exc))
            except (BootstrapError, OSError):
                pass
        print(f"supervised trainer bootstrap refused exec: {exc}", file=sys.stderr, flush=True)
        return 70
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
