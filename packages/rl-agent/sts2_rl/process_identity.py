"""Fail-closed process identity capture and termination helpers.

PID values are recyclable and therefore are never sufficient authority for a
destructive operation.  This module captures a platform process-start marker
and executable path alongside the PID.  A process is signalled only after the
live process has been re-opened and all three values still match.

The implementation deliberately has no third-party dependency so that it can
be used by launch/bootstrap utilities before the RL environment is installed.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import signal
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ProcessIdentityError(RuntimeError):
    """Raised when a stable process identity cannot be captured."""


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Stable identity for one OS process lifetime."""

    pid: int
    creation_marker: str
    executable: str
    started_at_unix_ns: int | None = None
    command_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProcessIdentity:
        try:
            pid = int(value["pid"])
            creation_marker = str(value["creation_marker"])
            executable = str(value["executable"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProcessIdentityError("Invalid persisted process identity.") from exc
        if pid <= 0 or not creation_marker or not executable:
            raise ProcessIdentityError("Persisted process identity is incomplete.")
        raw_started = value.get("started_at_unix_ns")
        started_at_unix_ns = int(raw_started) if raw_started is not None else None
        raw_fingerprint = value.get("command_fingerprint")
        command_fingerprint = str(raw_fingerprint) if raw_fingerprint else None
        return cls(
            pid=pid,
            creation_marker=creation_marker,
            executable=executable,
            started_at_unix_ns=started_at_unix_ns,
            command_fingerprint=command_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class TerminationResult:
    terminated: bool
    reason: str


def _normalized_executable(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(value)))


if sys.platform == "win32":
    from ctypes import wintypes

    _PROCESS_TERMINATE = 0x0001
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _SYNCHRONIZE = 0x00100000
    _WAIT_OBJECT_0 = 0
    _WAIT_TIMEOUT = 258
    _WINDOWS_EPOCH_OFFSET_100NS = 116_444_736_000_000_000

    class _FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]  # noqa: RUF012


def _filetime_value(value: _FILETIME) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def _windows_snapshot_from_handle(handle: int, pid: int) -> ProcessIdentity:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    creation = _FILETIME()
    exit_time = _FILETIME()
    kernel = _FILETIME()
    user = _FILETIME()
    if not kernel32.GetProcessTimes(
        handle,
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise ProcessIdentityError(f"GetProcessTimes failed for pid={pid}.")

    capacity = wintypes.DWORD(32_768)
    buffer = ctypes.create_unicode_buffer(capacity.value)
    if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(capacity)):
        raise ProcessIdentityError(f"QueryFullProcessImageNameW failed for pid={pid}.")

    created_100ns = _filetime_value(creation)
    return ProcessIdentity(
        pid=pid,
        creation_marker=f"win-filetime:{created_100ns}",
        executable=_normalized_executable(buffer.value),
        started_at_unix_ns=max(0, created_100ns - _WINDOWS_EPOCH_OFFSET_100NS) * 100,
    )


def _open_windows_process(pid: int, access: int) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(access, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        raise ProcessIdentityError(f"OpenProcess failed for pid={pid} (winerror={error}).")
    return int(handle)


def _linux_command_fingerprint(pid: int) -> str | None:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return hashlib.sha256(command).hexdigest() if command else None


def _linux_started_at_ns(start_ticks: int) -> int | None:
    try:
        clock_ticks = int(os.sysconf("SC_CLK_TCK"))  # type: ignore[attr-defined]
        boot_line = next(
            line for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines() if line.startswith("btime ")
        )
        boot_seconds = int(boot_line.split()[1])
    except (OSError, StopIteration, ValueError):
        return None
    return boot_seconds * 1_000_000_000 + (start_ticks * 1_000_000_000 // clock_ticks)


def _linux_snapshot(pid: int) -> ProcessIdentity:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # The comm field is parenthesized and may contain spaces or ')'.  The
        # last ')' is therefore the only safe boundary before field 3.
        remainder = stat_text[stat_text.rfind(")") + 2 :].split()
        start_ticks = int(remainder[19])  # field 22, with remainder[0] == field 3
        executable = os.readlink(f"/proc/{pid}/exe")
    except (OSError, IndexError, ValueError) as exc:
        raise ProcessIdentityError(f"Could not inspect /proc identity for pid={pid}.") from exc
    return ProcessIdentity(
        pid=pid,
        creation_marker=f"linux-start-ticks:{start_ticks}",
        executable=_normalized_executable(executable),
        started_at_unix_ns=_linux_started_at_ns(start_ticks),
        command_fingerprint=_linux_command_fingerprint(pid),
    )


def capture_process_identity(pid: int) -> ProcessIdentity:
    """Capture the current process lifetime identity, or fail closed."""

    if pid <= 0:
        raise ProcessIdentityError(f"Invalid pid={pid}.")
    if sys.platform == "win32":
        access = _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE
        handle = _open_windows_process(pid, access)
        try:
            return _windows_snapshot_from_handle(handle, pid)
        finally:
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
    if sys.platform.startswith("linux"):
        return _linux_snapshot(pid)
    raise ProcessIdentityError(f"Process identity is unsupported on platform {sys.platform!r}.")


def same_process(expected: ProcessIdentity, actual: ProcessIdentity) -> bool:
    """Return whether two snapshots identify the exact same process lifetime."""

    if expected.pid != actual.pid:
        return False
    if expected.creation_marker != actual.creation_marker:
        return False
    if _normalized_executable(expected.executable) != _normalized_executable(actual.executable):
        return False
    if expected.command_fingerprint and actual.command_fingerprint:
        return expected.command_fingerprint == actual.command_fingerprint
    return True


def verify_process_identity(expected: ProcessIdentity) -> TerminationResult:
    """Verify a persisted identity without sending any signal."""

    try:
        actual = capture_process_identity(expected.pid)
    except ProcessIdentityError:
        return TerminationResult(terminated=False, reason="process_not_found_or_uninspectable")
    if not same_process(expected, actual):
        return TerminationResult(terminated=False, reason="identity_mismatch")
    return TerminationResult(terminated=False, reason="identity_verified")


def _terminate_windows(expected: ProcessIdentity, timeout_s: float) -> TerminationResult:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    access = _PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE | _SYNCHRONIZE
    try:
        handle = _open_windows_process(expected.pid, access)
    except ProcessIdentityError:
        return TerminationResult(terminated=True, reason="already_exited")
    try:
        try:
            actual = _windows_snapshot_from_handle(handle, expected.pid)
        except ProcessIdentityError:
            return TerminationResult(terminated=False, reason="identity_uninspectable")
        if not same_process(expected, actual):
            return TerminationResult(terminated=False, reason="identity_mismatch")
        if not kernel32.TerminateProcess(handle, 1):
            return TerminationResult(terminated=False, reason="terminate_failed")
        wait_ms = max(0, min(int(timeout_s * 1_000), 2_147_483_647))
        wait_result = int(kernel32.WaitForSingleObject(handle, wait_ms))
        if wait_result == _WAIT_OBJECT_0:
            return TerminationResult(terminated=True, reason="terminated_owned_process")
        if wait_result == _WAIT_TIMEOUT:
            return TerminationResult(terminated=False, reason="termination_timeout")
        return TerminationResult(terminated=False, reason="wait_failed")
    finally:
        kernel32.CloseHandle(handle)


def _linux_alive(expected: ProcessIdentity) -> bool:
    try:
        return same_process(expected, capture_process_identity(expected.pid))
    except ProcessIdentityError:
        return False


def _send_linux_signal(expected: ProcessIdentity, sig: signal.Signals) -> bool:
    # pidfd makes the signal immune to PID reuse between verification and
    # delivery.  Older Python/kernel combinations fall back to a second
    # identity check immediately before os.kill.
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if callable(pidfd_open) and callable(pidfd_send_signal):
        try:
            pidfd = pidfd_open(expected.pid)
        except OSError:
            return False
        try:
            if not _linux_alive(expected):
                return False
            pidfd_send_signal(pidfd, sig)
            return True
        except OSError:
            return False
        finally:
            os.close(pidfd)
    if not _linux_alive(expected):
        return False
    try:
        os.kill(expected.pid, sig)
    except ProcessLookupError:
        return False
    return True


def _terminate_linux(expected: ProcessIdentity, timeout_s: float) -> TerminationResult:
    try:
        actual = capture_process_identity(expected.pid)
    except ProcessIdentityError:
        return TerminationResult(terminated=True, reason="already_exited")
    if not same_process(expected, actual):
        return TerminationResult(terminated=False, reason="identity_mismatch")
    if not _send_linux_signal(expected, signal.SIGTERM):
        return TerminationResult(terminated=False, reason="signal_failed")

    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if not _linux_alive(expected):
            return TerminationResult(terminated=True, reason="terminated_owned_process")
        time.sleep(0.05)
    if not _linux_alive(expected):
        return TerminationResult(terminated=True, reason="terminated_owned_process")
    if not _send_linux_signal(expected, getattr(signal, "SIGKILL", signal.SIGTERM)):
        return TerminationResult(terminated=False, reason="kill_failed")
    for _ in range(100):
        if not _linux_alive(expected):
            return TerminationResult(terminated=True, reason="killed_owned_process")
        time.sleep(0.01)
    return TerminationResult(terminated=False, reason="termination_timeout")


def terminate_verified_process(expected: ProcessIdentity, *, timeout_s: float = 10.0) -> TerminationResult:
    """Terminate only the exact process represented by ``expected``.

    A mismatched, incomplete, or uninspectable identity is never signalled.
    """

    if sys.platform == "win32":
        return _terminate_windows(expected, timeout_s)
    if sys.platform.startswith("linux"):
        return _terminate_linux(expected, timeout_s)
    return TerminationResult(terminated=False, reason="unsupported_platform")


__all__ = [
    "ProcessIdentity",
    "ProcessIdentityError",
    "TerminationResult",
    "capture_process_identity",
    "same_process",
    "terminate_verified_process",
    "verify_process_identity",
]
