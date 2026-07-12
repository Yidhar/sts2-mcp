"""Launch and manage only launcher-owned STS2 game processes.

The ownership registry stores PID + native process creation marker + executable
path.  A later --kill invocation validates all fields against the live process
before termination, so stale state and PID reuse fail closed.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from launcher_watchdog import InstanceAction, Watchdog, WatchdogInstanceView
from sts2_env.path_utils import default_bridge_session_dir
from sts2_rl.process_identity import (
    ProcessIdentity,
    ProcessIdentityError,
    TerminationResult,
    capture_process_identity,
    same_process,
    terminate_verified_process,
)

_REGISTRY_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class LauncherProcessRecord:
    identity: ProcessIdentity
    instance_id: int
    session_path: str
    launcher_run_id: str
    launched_at_unix_ns: int
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["identity"] = self.identity.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LauncherProcessRecord:
        raw_identity = value.get("identity")
        if not isinstance(raw_identity, Mapping):
            raise ValueError("Launcher record is missing process identity.")
        return cls(
            identity=ProcessIdentity.from_dict(raw_identity),
            instance_id=int(value["instance_id"]),
            session_path=str(value["session_path"]),
            launcher_run_id=str(value["launcher_run_id"]),
            launched_at_unix_ns=int(value["launched_at_unix_ns"]),
            session_id=str(value["session_id"]) if value.get("session_id") else None,
        )


@dataclass(slots=True)
class GameInstance:
    instance_id: int
    process: subprocess.Popen[Any]
    session_path: Path
    ownership: LauncherProcessRecord
    base_url: str | None = None
    token: str | None = None
    session_id: str | None = None
    ready: bool = False


def session_dir() -> Path:
    return default_bridge_session_dir()


def session_path_for(instance_id: int) -> Path:
    return session_dir() / f"session_{instance_id}.json"


def launcher_registry_path() -> Path:
    explicit = os.environ.get("STS2_LAUNCHER_STATE_FILE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if local := os.environ.get("LOCALAPPDATA"):
        return Path(local) / "STS2Launcher" / "owned_instances.json"
    state_root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_root / "sts2-launcher" / "owned_instances.json"


def _read_registry() -> list[LauncherProcessRecord]:
    path = launcher_registry_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read ownership registry {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _REGISTRY_SCHEMA:
        raise RuntimeError(f"Invalid ownership registry schema: {path}")
    raw_records = payload.get("processes")
    if not isinstance(raw_records, list):
        raise RuntimeError(f"Invalid ownership registry payload: {path}")
    try:
        return [
            LauncherProcessRecord.from_dict(item)
            for item in raw_records
            if isinstance(item, Mapping)
        ]
    except (KeyError, TypeError, ValueError, ProcessIdentityError) as exc:
        raise RuntimeError(f"Invalid ownership registry record: {path}") from exc


def _write_registry(records: Sequence[LauncherProcessRecord]) -> None:
    path = launcher_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "schema_version": _REGISTRY_SCHEMA,
        "processes": [record.to_dict() for record in records],
    }
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _record_key(record: LauncherProcessRecord) -> tuple[int, str]:
    return record.identity.pid, record.identity.creation_marker


def _upsert_record(record: LauncherProcessRecord) -> None:
    records = [item for item in _read_registry() if _record_key(item) != _record_key(record)]
    _write_registry([*records, record])


def _remove_record(record: LauncherProcessRecord) -> None:
    _write_registry(
        [item for item in _read_registry() if _record_key(item) != _record_key(record)]
    )


def _capture_child(proc: subprocess.Popen[Any], executable: Path) -> ProcessIdentity:
    last_error: Exception | None = None
    for _ in range(20):
        if proc.poll() is not None:
            raise RuntimeError(f"Game exited before identity capture: {proc.returncode}")
        try:
            identity = capture_process_identity(proc.pid)
        except ProcessIdentityError as exc:
            last_error = exc
            time.sleep(0.05)
            continue
        expected = os.path.normcase(os.path.realpath(executable))
        actual = os.path.normcase(os.path.realpath(identity.executable))
        if actual != expected:
            terminate_verified_process(identity, timeout_s=5.0)
            raise RuntimeError(
                f"Launched executable mismatch: expected={executable}, actual={identity.executable}"
            )
        return identity
    # This Popen handle refers to the exact child we just created. It is not a
    # PID lookup and is safe for failure cleanup.
    proc.terminate()
    raise RuntimeError(f"Could not capture child identity: {last_error}")


def launch_instance(
    game_exe: str,
    instance_id: int,
    *,
    launcher_run_id: str | None = None,
) -> GameInstance:
    executable = Path(game_exe).expanduser().resolve(strict=True)
    if not executable.is_file():
        raise ValueError(f"Game executable is not a file: {executable}")
    env = os.environ.copy()
    env["STS2_BRIDGE_INSTANCE_ID"] = str(instance_id)
    env.setdefault("STS2_BRIDGE_ENABLE_TRAINING_V2", "1")
    proc = subprocess.Popen(
        [str(executable)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    identity = _capture_child(proc, executable)
    record = LauncherProcessRecord(
        identity=identity,
        instance_id=instance_id,
        session_path=str(session_path_for(instance_id)),
        launcher_run_id=launcher_run_id or uuid.uuid4().hex,
        launched_at_unix_ns=time.time_ns(),
    )
    _upsert_record(record)
    return GameInstance(instance_id, proc, session_path_for(instance_id), record)


def _loopback_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
    return bool(
        parsed.scheme == "http"
        and loopback
        and parsed.port is not None
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _timestamp_ns(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1e9)
    except ValueError:
        return None


def _session_matches(data: Mapping[str, Any], record: LauncherProcessRecord) -> bool:
    try:
        if int(data.get("pid")) != record.identity.pid:
            return False
        actual = capture_process_identity(record.identity.pid)
    except (TypeError, ValueError, ProcessIdentityError):
        return False
    if not same_process(record.identity, actual):
        return False
    descriptor_start = _timestamp_ns(data.get("process_started_at_utc"))
    native_start = record.identity.started_at_unix_ns
    if descriptor_start is not None and native_start is not None:
        if abs(descriptor_start - native_start) > 2_000_000_000:
            return False
    return bool(str(data.get("session_id") or ""))


def _read_owned_session(instance: GameInstance) -> tuple[str, str, str] | None:
    try:
        data = json.loads(instance.session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, Mapping) or not _session_matches(data, instance.ownership):
        return None
    base_url = str(data.get("base_url") or "").rstrip("/")
    tokens = data.get("capability_tokens")
    if not _loopback_http_url(base_url) or not isinstance(tokens, Mapping):
        return None
    token = str(tokens.get("training") or "")
    session_id = str(data.get("session_id") or "")
    return (base_url, token, session_id) if token and session_id else None


def wait_for_ready(instance: GameInstance, timeout_s: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if instance.process.poll() is not None:
            print(f"  [!] Instance {instance.instance_id} exited prematurely")
            return False
        session = _read_owned_session(instance) if instance.session_path.exists() else None
        if session:
            instance.base_url, instance.token, instance.session_id = session
            instance.ownership = replace(instance.ownership, session_id=instance.session_id)
            _upsert_record(instance.ownership)
            break
        time.sleep(1.0)
    else:
        print(f"  [!] Instance {instance.instance_id} session file timeout")
        return False

    while time.monotonic() < deadline:
        if instance.process.poll() is not None:
            return False
        try:
            response = requests.get(
                f"{instance.base_url}/v2/health",
                headers={"Authorization": f"Bearer {instance.token}"},
                timeout=5.0,
            )
            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, Mapping):
                    if payload.get("process_id") is not None:
                        if int(payload["process_id"]) != instance.process.pid:
                            time.sleep(1.0)
                            continue
                    if payload.get("session_id") is not None:
                        if str(payload["session_id"]) != instance.session_id:
                            time.sleep(1.0)
                            continue
                instance.ready = True
                return True
        except (requests.ConnectionError, requests.Timeout, requests.JSONDecodeError, ValueError):
            pass
        time.sleep(2.0)
    print(f"  [!] Instance {instance.instance_id} v2 health timeout")
    return False


def launch_all(game_exe: str, count: int) -> list[GameInstance]:
    if count <= 0:
        raise ValueError("--instances must be positive")
    for instance_id in range(count):
        session_path_for(instance_id).unlink(missing_ok=True)
    run_id = uuid.uuid4().hex
    instances: list[GameInstance] = []
    try:
        for instance_id in range(count):
            instance = launch_instance(game_exe, instance_id, launcher_run_id=run_id)
            instances.append(instance)
            print(f"[launcher] Started instance {instance_id} (pid={instance.process.pid})")
            time.sleep(3.0)
    except Exception:
        kill_all(instances)
        raise
    for instance in instances:
        print(f"[launcher] Waiting for instance {instance.instance_id}...")
        if wait_for_ready(instance):
            print(f"[launcher] Instance {instance.instance_id} ready at {instance.base_url}")
        else:
            print(f"[launcher] Instance {instance.instance_id} FAILED")
    print(f"\n[launcher] {sum(item.ready for item in instances)}/{count} instances ready")
    return instances


def _remove_owned_session(record: LauncherProcessRecord) -> None:
    path = Path(record.session_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pid_matches = isinstance(data, Mapping) and int(data.get("pid")) == record.identity.pid
        session_matches = not record.session_id or data.get("session_id") == record.session_id
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return
    if pid_matches and session_matches:
        path.unlink(missing_ok=True)


def _terminate_record(record: LauncherProcessRecord) -> TerminationResult:
    return terminate_verified_process(record.identity, timeout_s=10.0)


def kill_instance(instance: GameInstance, reason: str) -> bool:
    print(f"[launcher] terminating owned instance {instance.instance_id}: {reason}")
    result = (
        _terminate_record(instance.ownership)
        if instance.process.poll() is None
        else TerminationResult(True, "already_exited")
    )
    if not result.terminated:
        print(f"[launcher] REFUSED instance {instance.instance_id}: {result.reason}")
        return False
    try:
        instance.process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        return False
    _remove_owned_session(instance.ownership)
    _remove_record(instance.ownership)
    print(f"[launcher] instance {instance.instance_id}: {result.reason}")
    return True


def kill_all(instances: Sequence[GameInstance] | None = None) -> bool:
    """Terminate only validated in-memory or persisted launcher-owned processes."""
    if instances is not None:
        succeeded = True
        for instance in instances:
            succeeded = kill_instance(instance, "launcher_shutdown") and succeeded
        return succeeded

    succeeded = True
    retained: list[LauncherProcessRecord] = []
    for record in _read_registry():
        result = _terminate_record(record)
        if result.terminated:
            print(
                f"[launcher] instance {record.instance_id} pid={record.identity.pid}: "
                f"{result.reason}"
            )
            _remove_owned_session(record)
        else:
            succeeded = False
            print(
                f"[launcher] REFUSED pid={record.identity.pid} "
                f"instance={record.instance_id}: {result.reason}"
            )
            # An identity mismatch means the recorded process lifetime is gone.
            # Drop it so a recycled PID can never become authorized later.
            if result.reason != "identity_mismatch":
                retained.append(record)
    _write_registry(retained)
    return succeeded


def restart_instance(old: GameInstance, game_exe: str) -> GameInstance:
    if not kill_instance(old, "watchdog_restart"):
        raise RuntimeError(
            f"Refusing replacement for instance {old.instance_id}; ownership check failed."
        )
    fresh = launch_instance(
        game_exe,
        old.instance_id,
        launcher_run_id=old.ownership.launcher_run_id,
    )
    print(f"[launcher] restarted instance {fresh.instance_id} (pid={fresh.process.pid})")
    if wait_for_ready(fresh):
        print(f"[launcher] instance {fresh.instance_id} ready again at {fresh.base_url}")
    return fresh


def get_session_files(n_instances: int) -> list[str]:
    return [str(session_path_for(index)) for index in range(n_instances)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="STS2 owned multi-instance launcher")
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--game-exe", help="Path to Slay the Spire 2 executable")
    parser.add_argument("--kill", action="store_true", help="Kill registry-owned instances")
    parser.add_argument("--watchdog", action="store_true")
    parser.add_argument("--watchdog-strike-threshold", type=int, default=3)
    parser.add_argument("--watchdog-interval-s", type=float, default=15.0)
    parser.add_argument("--watchdog-probe-timeout-s", type=float, default=5.0)
    parser.add_argument("--watchdog-log-flood-mb-per-s", type=float, default=10.0)
    parser.add_argument("--watchdog-state-stall-s", type=float, default=90.0)
    parser.add_argument("--watchdog-state-probe-timeout-s", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.kill:
        return 0 if kill_all() else 2
    if not args.game_exe:
        parser.error("--game-exe is required unless --kill is used")

    instances = launch_all(args.game_exe, args.instances)
    if not any(instance.ready for instance in instances):
        kill_all(instances)
        return 1
    for instance in instances:
        if instance.ready:
            print(f"Instance {instance.instance_id}: {instance.session_path}")

    watchdog: Watchdog | None = None
    if args.watchdog:
        watchdog = Watchdog(
            strike_threshold=args.watchdog_strike_threshold,
            probe_timeout_s=args.watchdog_probe_timeout_s,
            log_flood_bytes_per_sec=args.watchdog_log_flood_mb_per_s * 1024 * 1024,
            state_stall_threshold_s=args.watchdog_state_stall_s,
            state_probe_timeout_s=args.watchdog_state_probe_timeout_s,
            logs_dir=session_dir().parent / "logs",
        )

    try:
        next_tick = time.monotonic() + args.watchdog_interval_s
        while True:
            time.sleep(5.0)
            if not watchdog or time.monotonic() < next_tick:
                continue
            next_tick = time.monotonic() + args.watchdog_interval_s
            views = [
                WatchdogInstanceView(
                    instance_id=item.instance_id,
                    process_alive=item.process.poll() is None,
                    base_url=item.base_url,
                    token=item.token,
                )
                for item in instances
            ]
            decision = watchdog.evaluate(views, now_unix_s=time.time())
            if decision.log_flood_detected:
                instances = [restart_instance(item, args.game_exe) for item in instances]
                continue
            for entry in decision.per_instance:
                if entry.action is not InstanceAction.KILL_AND_RESTART:
                    continue
                for index, instance in enumerate(instances):
                    if instance.instance_id == entry.instance_id:
                        instances[index] = restart_instance(instance, args.game_exe)
                        break
    except KeyboardInterrupt:
        return 0 if kill_all(instances) else 2


if __name__ == "__main__":
    raise SystemExit(main())
