from __future__ import annotations

import importlib.util
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

if sys.platform != "linux":
    pytest.skip("supervised trainer bootstrap is Linux-only", allow_module_level=True)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "supervised_trainer_bootstrap.py"


def _load_bootstrap() -> ModuleType:
    spec = importlib.util.spec_from_file_location("supervised_trainer_bootstrap_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = _load_bootstrap()


_PARENT_DEATH_SUPERVISOR = r"""
import ctypes
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

mode, expected_parent_raw, helper_path, gate_read_raw, gate_write_raw, status_read_raw, status_write_raw, exec_marker, done_marker = sys.argv[1:]
expected_parent = int(expected_parent_raw)
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "test-supervisor PR_SET_PDEATHSIG failed")
if os.getppid() != expected_parent:
    raise RuntimeError("test harness exited before supervisor armed PDEATHSIG")
gate_read = int(gate_read_raw)
gate_write = int(gate_write_raw)
status_read = int(status_read_raw)
status_write = int(status_write_raw)
spec = importlib.util.spec_from_file_location("bootstrap_for_parent_death_supervisor", helper_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
nonce = "c" * 64
trainer = (
    "import os, pathlib, sys, time; "
    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii'); "
    "time.sleep(60); "
    "pathlib.Path(sys.argv[2]).write_text('unexpected-completion', encoding='ascii')"
)
command = [
    sys.executable,
    helper_path,
    "--expected-parent-pid", str(os.getpid()),
    "--expected-parent-start-ticks", str(module.process_start_ticks(os.getpid())),
    "--gate-fd", str(gate_read),
    "--status-fd", str(status_write),
    "--nonce", nonce,
    "--timeout-seconds", "30",
    "--",
    sys.executable, "-c", trainer, exec_marker, done_marker,
]
child = subprocess.Popen(
    command,
    close_fds=True,
    pass_fds=(gate_read, status_write),
    stderr=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
)
os.close(gate_read)
os.close(status_write)
ready = module.read_bounded_frame(status_read, 5.0)
print(json.dumps({"phase": "ready", "ready": ready}, sort_keys=True), flush=True)
if mode == "exec":
    os.write(
        gate_write,
        module.canonical_frame({"kind": module.GO_KIND, "nonce": nonce, "protocol": module.PROTOCOL_VERSION}),
    )
    os.close(gate_write)
    gate_write = -1
    print(json.dumps({"phase": "go_written"}, sort_keys=True), flush=True)
    trailing = os.read(status_read, 4096)
    if trailing:
        raise RuntimeError(f"bootstrap emitted unexpected post-READY status: {trailing!r}")
    print(json.dumps({"phase": "exec_ack"}, sort_keys=True), flush=True)
time.sleep(60)
"""


_PARENT_DEATH_HARNESS = r"""
import ctypes
import json
import os
import pathlib
import selectors
import signal
import subprocess
import sys
import time

PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37
mode, supervisor_program, helper_path, exec_marker_raw, done_marker_raw = sys.argv[1:]
exec_marker = pathlib.Path(exec_marker_raw)
done_marker = pathlib.Path(done_marker_raw)
libc = ctypes.CDLL(None, use_errno=True)
old_subreaper = ctypes.c_int(0)
if libc.prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(old_subreaper), 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")

gate_read, gate_write = os.pipe()
status_read, status_write = os.pipe()
supervisor = None
child_pid = None
child_reaped = False

def read_line_with_timeout(stream, timeout):
    selector = selectors.DefaultSelector()
    payload = bytearray()
    try:
        selector.register(stream.fileno(), selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError("supervisor protocol timed out")
            byte = os.read(stream.fileno(), 1)
            if not byte:
                raise RuntimeError("supervisor protocol reached EOF")
            if byte == b"\n":
                return json.loads(payload.decode("ascii"))
            payload.extend(byte)
            if len(payload) > 4096:
                raise RuntimeError("supervisor protocol line exceeded 4096 bytes")
    finally:
        selector.close()

try:
    supervisor = subprocess.Popen(
        [
            sys.executable, "-c", supervisor_program, mode, str(os.getpid()), helper_path,
            str(gate_read), str(gate_write), str(status_read), str(status_write),
            str(exec_marker), str(done_marker),
        ],
        close_fds=True,
        pass_fds=(gate_read, gate_write, status_read, status_write),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    os.close(gate_read)
    gate_read = -1
    os.close(status_read)
    status_read = -1
    os.close(status_write)
    status_write = -1

    first = read_line_with_timeout(supervisor.stdout, 5.0)
    if first.get("phase") != "ready":
        raise RuntimeError(f"unexpected supervisor phase: {first!r}")
    ready = first["ready"]
    child_pid = int(ready["pid"])
    if int(ready["ppid"]) != supervisor.pid:
        raise RuntimeError("bootstrap READY did not name its direct supervisor parent")

    if mode == "exec":
        second = read_line_with_timeout(supervisor.stdout, 5.0)
        if second != {"phase": "go_written"}:
            raise RuntimeError(f"unexpected GO phase: {second!r}")
        # The harness deliberately retained a duplicate gate writer.  Closing
        # it only after GO_WRITTEN produces the EOF required to authorize exec.
        os.close(gate_write)
        gate_write = -1
        third = read_line_with_timeout(supervisor.stdout, 5.0)
        if third != {"phase": "exec_ack"}:
            raise RuntimeError(f"unexpected exec phase: {third!r}")
        deadline = time.monotonic() + 5.0
        while not exec_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not exec_marker.exists():
            raise RuntimeError("trainer did not publish its exec sentinel")

    os.kill(supervisor.pid, signal.SIGKILL)
    supervisor.wait(timeout=5.0)
    reaped_pid, child_status = os.waitpid(child_pid, 0)
    child_reaped = True
    if reaped_pid != child_pid:
        raise RuntimeError("reaped a process other than the bootstrap/trainer")
    result = {
        "bootstrap_pid": child_pid,
        "child_signaled": os.WIFSIGNALED(child_status),
        "child_signal": os.WTERMSIG(child_status) if os.WIFSIGNALED(child_status) else None,
        "done_marker_exists": done_marker.exists(),
        "exec_marker_exists": exec_marker.exists(),
        "exec_marker_pid": int(exec_marker.read_text(encoding="ascii")) if exec_marker.exists() else None,
        "mode": mode,
        "proc_entry_exists_after_reap": pathlib.Path(f"/proc/{child_pid}").exists(),
        "supervisor_returncode": supervisor.returncode,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
finally:
    for fd in (gate_read, gate_write, status_read, status_write):
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
    if supervisor is not None and supervisor.poll() is None:
        supervisor.kill()
        supervisor.wait(timeout=5.0)
    if child_pid is not None and not child_reaped:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(child_pid, 0)
        except ChildProcessError:
            pass
    libc.prctl(PR_SET_CHILD_SUBREAPER, int(old_subreaper.value), 0, 0, 0)
"""


def _nonce(character: str = "a") -> str:
    return character * 64


def _go(nonce: str | None = None) -> dict[str, object]:
    return {
        "kind": bootstrap.GO_KIND,
        "nonce": nonce or _nonce(),
        "protocol": bootstrap.PROTOCOL_VERSION,
    }


def _pipe_with(payload: bytes) -> tuple[int, int]:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload)
    os.close(write_fd)
    return read_fd, write_fd


def test_canonical_frame_and_bounded_reader_round_trip() -> None:
    payload = bootstrap.canonical_frame(_go())
    assert payload.endswith(b"\n")
    assert payload == b'{"kind":"GO","nonce":"' + b"a" * 64 + (
        b'","protocol":"sts2-supervised-trainer-bootstrap-v1"}\n'
    )
    read_fd, _ = _pipe_with(payload)
    try:
        assert bootstrap.read_bounded_frame(read_fd, 1.0, require_eof=True) == _go()
    finally:
        os.close(read_fd)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"kind":"GO"}',  # EOF before newline.
        bootstrap.canonical_frame(_go()) + bootstrap.canonical_frame(_go()),
        b'{"kind":"GO", "nonce":"' + b"a" * 64 + b'","protocol":"sts2-supervised-trainer-bootstrap-v1"}\n',
        b'{"kind":"GO","kind":"GO","nonce":"' + b"a" * 64 + b'","protocol":"sts2-supervised-trainer-bootstrap-v1"}\n',
    ],
)
def test_bounded_reader_rejects_partial_extra_noncanonical_and_duplicate_frames(payload: bytes) -> None:
    read_fd, _ = _pipe_with(payload)
    try:
        with pytest.raises(bootstrap.BootstrapError):
            bootstrap.read_bounded_frame(read_fd, 1.0, require_eof=True)
    finally:
        os.close(read_fd)


def test_bounded_reader_times_out() -> None:
    read_fd, write_fd = os.pipe()
    try:
        started = time.monotonic()
        with pytest.raises(bootstrap.BootstrapError, match="timed out"):
            bootstrap.read_bounded_frame(read_fd, 0.05)
        assert time.monotonic() - started < 1.0
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_process_start_ticks_identifies_current_process() -> None:
    if sys.platform != "linux":
        pytest.skip("/proc start-tick identity is Linux-specific")
    assert bootstrap.process_start_ticks(os.getpid()) > 0
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.process_start_ticks(0)


def _spawn_bootstrap(
    tmp_path: Path, *, timeout_s: float = 2.0
) -> tuple[subprocess.Popen[bytes], dict[str, object], int, int, Path]:
    if sys.platform != "linux":
        pytest.skip("PDEATHSIG bootstrap integration is Linux-specific")
    marker = tmp_path / "trainer-exec.json"
    gate_read, gate_write = os.pipe()
    status_read, status_write = os.pipe()
    nonce = _nonce()
    trainer_program = """
import errno, json, os, sys
gate_fd, status_fd = int(sys.argv[2]), int(sys.argv[3])
def descriptor_is_open(fd):
    try:
        os.fstat(fd)
    except OSError as exc:
        if exc.errno == errno.EBADF:
            return False
        raise
    return True
result = {
    "pid": os.getpid(),
    "gate_fd_open": descriptor_is_open(gate_fd),
    "status_fd_open": descriptor_is_open(status_fd),
    "environment": os.environ.get("STS2_BOOTSTRAP_TEST_ENV"),
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(result, handle, sort_keys=True)
"""
    command = [
        sys.executable,
        str(SCRIPT),
        "--expected-parent-pid",
        str(os.getpid()),
        "--expected-parent-start-ticks",
        str(bootstrap.process_start_ticks(os.getpid())),
        "--gate-fd",
        str(gate_read),
        "--status-fd",
        str(status_write),
        "--nonce",
        nonce,
        "--timeout-seconds",
        str(timeout_s),
        "--",
        sys.executable,
        "-c",
        trainer_program,
        str(marker),
        str(gate_read),
        str(status_write),
    ]
    env = dict(os.environ)
    env["STS2_BOOTSTRAP_TEST_ENV"] = "inherited-exactly"
    process = subprocess.Popen(
        command,
        close_fds=True,
        env=env,
        pass_fds=(gate_read, status_write),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    os.close(gate_read)
    os.close(status_write)
    try:
        ready = bootstrap.read_bounded_frame(status_read, 2.0)
    except BaseException:
        os.close(gate_write)
        os.close(status_read)
        process.kill()
        process.wait(timeout=2.0)
        raise
    return process, ready, gate_write, status_read, marker


def _wait_for_status_eof(status_read: int, timeout_s: float = 2.0) -> None:
    selector = selectors.DefaultSelector()
    try:
        selector.register(status_read, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            events = selector.select(deadline - time.monotonic())
            if not events:
                break
            data = os.read(status_read, 4096)
            if not data:
                return
        raise AssertionError("status fd did not close on exec")
    finally:
        selector.close()


def _assert_ready(ready: dict[str, object], process: subprocess.Popen[bytes]) -> None:
    assert ready == {
        "kind": bootstrap.READY_KIND,
        "nonce": _nonce(),
        "pid": process.pid,
        "ppid": os.getpid(),
        "proc_start_ticks": bootstrap.process_start_ticks(process.pid),
        "protocol": bootstrap.PROTOCOL_VERSION,
    }


def test_go_execs_in_place_with_inherited_environment_and_no_protocol_fds(tmp_path: Path) -> None:
    process, ready, gate_write, status_read, marker = _spawn_bootstrap(tmp_path)
    try:
        _assert_ready(ready, process)
        os.write(gate_write, bootstrap.canonical_frame(_go()))
        os.close(gate_write)
        gate_write = -1
        _wait_for_status_eof(status_read)
        stdout, stderr = process.communicate(timeout=3.0)
        assert process.returncode == 0, (stdout, stderr)
        result = json.loads(marker.read_text(encoding="utf-8"))
        assert result == {
            "environment": "inherited-exactly",
            "gate_fd_open": False,
            "pid": ready["pid"],
            "status_fd_open": False,
        }
    finally:
        if gate_write >= 0:
            os.close(gate_write)
        os.close(status_read)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)


@pytest.mark.parametrize("failure_mode", ["eof", "partial", "wrong_nonce", "extra", "timeout"])
def test_missing_or_malformed_go_never_execs(tmp_path: Path, failure_mode: str) -> None:
    timeout_s = 0.15 if failure_mode == "timeout" else 2.0
    process, ready, gate_write, status_read, marker = _spawn_bootstrap(tmp_path, timeout_s=timeout_s)
    try:
        _assert_ready(ready, process)
        if failure_mode == "partial":
            os.write(gate_write, bootstrap.canonical_frame(_go())[:-1])
        elif failure_mode == "wrong_nonce":
            os.write(gate_write, bootstrap.canonical_frame(_go(_nonce("b"))))
        elif failure_mode == "extra":
            os.write(gate_write, bootstrap.canonical_frame(_go()) + bootstrap.canonical_frame(_go()))
        if failure_mode != "timeout":
            os.close(gate_write)
            gate_write = -1
        stdout, stderr = process.communicate(timeout=3.0)
        assert process.returncode == 70, (stdout, stderr)
        assert not marker.exists()
    finally:
        if gate_write >= 0:
            os.close(gate_write)
        os.close(status_read)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)


def test_wrong_parent_start_ticks_never_reaches_ready_or_exec(tmp_path: Path) -> None:
    if sys.platform != "linux":
        pytest.skip("PDEATHSIG bootstrap integration is Linux-specific")
    marker = tmp_path / "must-not-exist"
    gate_read, gate_write = os.pipe()
    status_read, status_write = os.pipe()
    command = [
        sys.executable,
        str(SCRIPT),
        "--expected-parent-pid",
        str(os.getpid()),
        "--expected-parent-start-ticks",
        str(bootstrap.process_start_ticks(os.getpid()) + 1),
        "--gate-fd",
        str(gate_read),
        "--status-fd",
        str(status_write),
        "--nonce",
        _nonce(),
        "--timeout-seconds",
        "1",
        "--",
        sys.executable,
        "-c",
        f"open({str(marker)!r}, 'w').close()",
    ]
    process = subprocess.Popen(
        command,
        close_fds=True,
        pass_fds=(gate_read, status_write),
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    os.close(gate_read)
    os.close(status_write)
    os.close(gate_write)
    try:
        frame = bootstrap.read_bounded_frame(status_read, 2.0)
        assert frame["kind"] == bootstrap.ERROR_KIND
        assert frame["protocol"] == bootstrap.PROTOCOL_VERSION
        stdout, stderr = process.communicate(timeout=2.0)
        assert process.returncode == 70, (stdout, stderr)
        assert not marker.exists()
    finally:
        os.close(status_read)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)


@pytest.mark.parametrize("mode", ["pre_go", "exec"])
def test_parent_death_sigkill_has_no_orphan_before_or_after_exec(tmp_path: Path, mode: str) -> None:
    """Prove PDEATHSIG both before GO and after the in-place trainer exec.

    A disposable harness process becomes a Linux child subreaper.  That lets it
    strictly reap the bootstrap/trainer after killing the direct supervisor,
    rather than merely polling an orphan under ``/proc``.
    """

    if sys.platform != "linux":
        pytest.skip("PDEATHSIG/subreaper integration is Linux-specific")
    exec_marker = tmp_path / f"{mode}-trainer-exec"
    done_marker = tmp_path / f"{mode}-trainer-completed"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PARENT_DEATH_HARNESS,
            mode,
            _PARENT_DEATH_SUPERVISOR,
            str(SCRIPT),
            str(exec_marker),
            str(done_marker),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["mode"] == mode
    assert result["supervisor_returncode"] == -int(signal.SIGKILL)
    assert result["child_signaled"] is True
    assert result["child_signal"] == int(signal.SIGKILL)
    assert result["done_marker_exists"] is False
    assert result["proc_entry_exists_after_reap"] is False
    if mode == "pre_go":
        assert result["exec_marker_exists"] is False
        assert result["exec_marker_pid"] is None
    else:
        assert result["exec_marker_exists"] is True
        assert result["exec_marker_pid"] == result["bootstrap_pid"]
