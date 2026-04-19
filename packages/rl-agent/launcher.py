"""Launch and manage N parallel STS2 game instances for RL training.

Each instance gets a unique STS2_BRIDGE_INSTANCE_ID env var, causing the
bridge mod to write to session_{id}.json and bind to a unique port range.

Usage:
    python launcher.py --instances 2 --game-exe "E:/path/to/SlayTheSpire2.exe"
    python launcher.py --instances 4 --game-exe "E:/path/to/SlayTheSpire2.exe" --wait-ready
    python launcher.py --kill  # kill all managed instances
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from launcher_watchdog import (
    InstanceAction,
    Watchdog,
    WatchdogInstanceView,
)
from sts2_env.path_utils import default_bridge_session_dir


@dataclass
class GameInstance:
    instance_id: int
    process: subprocess.Popen
    session_path: Path
    base_url: str | None = None
    token: str | None = None
    ready: bool = False


def session_dir() -> Path:
    return default_bridge_session_dir()


def session_path_for(instance_id: int) -> Path:
    return session_dir() / f"session_{instance_id}.json"


def launch_instance(game_exe: str, instance_id: int) -> GameInstance:
    env = os.environ.copy()
    env["STS2_BRIDGE_INSTANCE_ID"] = str(instance_id)

    proc = subprocess.Popen(
        [game_exe],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return GameInstance(
        instance_id=instance_id,
        process=proc,
        session_path=session_path_for(instance_id),
    )


def wait_for_ready(instance: GameInstance, timeout_s: float = 120.0) -> bool:
    deadline = time.time() + timeout_s

    # Phase 1: Wait for session file
    while time.time() < deadline:
        if instance.process.poll() is not None:
            print(f"  [!] Instance {instance.instance_id} exited prematurely")
            return False
        if instance.session_path.exists():
            try:
                data = json.loads(instance.session_path.read_text())
                instance.base_url = data["base_url"].rstrip("/")
                instance.token = data["token"]
                break
            except (json.JSONDecodeError, KeyError):
                pass
        time.sleep(1.0)
    else:
        print(f"  [!] Instance {instance.instance_id} session file timeout")
        return False

    # Phase 2: Wait for bridge HTTP
    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{instance.base_url}/health",
                headers={"Authorization": f"Bearer {instance.token}"},
                timeout=5.0,
            )
            if resp.status_code == 200:
                instance.ready = True
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2.0)

    print(f"  [!] Instance {instance.instance_id} bridge health timeout")
    return False


def launch_all(game_exe: str, n_instances: int) -> list[GameInstance]:
    # Clean stale session files
    for i in range(n_instances):
        path = session_path_for(i)
        if path.exists():
            path.unlink()

    instances = []
    for i in range(n_instances):
        inst = launch_instance(game_exe, i)
        instances.append(inst)
        print(f"[launcher] Started instance {i} (pid={inst.process.pid})")
        time.sleep(3.0)  # Stagger to reduce disk I/O contention

    for inst in instances:
        print(f"[launcher] Waiting for instance {inst.instance_id}...")
        if wait_for_ready(inst):
            print(f"[launcher] Instance {inst.instance_id} ready at {inst.base_url}")
        else:
            print(f"[launcher] Instance {inst.instance_id} FAILED")

    ready = [i for i in instances if i.ready]
    print(f"\n[launcher] {len(ready)}/{n_instances} instances ready")
    return instances


def kill_all(instances: list[GameInstance] | None = None):
    if instances:
        for inst in instances:
            if inst.process.poll() is None:
                inst.process.terminate()
                print(f"[launcher] Terminated instance {inst.instance_id}")
    else:
        # Kill by finding game processes (fallback)
        if sys.platform == "win32":
            os.system('taskkill /F /IM "Slay the Spire 2.exe" 2>nul')
            os.system('taskkill /F /IM "SlayTheSpire2.exe" 2>nul')


def kill_instance(instance: GameInstance, reason: str) -> None:
    if instance.process.poll() is None:
        print(f"[launcher] killing instance {instance.instance_id}: {reason}")
        try:
            instance.process.kill()
        except Exception as exc:
            print(f"[launcher] kill failed for instance {instance.instance_id}: {exc}")
        try:
            instance.process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            pass
    # Clean up session file so the replacement process writes a fresh one.
    try:
        if instance.session_path.exists():
            instance.session_path.unlink()
    except OSError:
        pass


def restart_instance(old: GameInstance, game_exe: str) -> GameInstance:
    """Kill and relaunch one instance, preserving its instance_id / session slot."""
    kill_instance(old, reason="watchdog_restart")
    fresh = launch_instance(game_exe, old.instance_id)
    print(f"[launcher] restarted instance {fresh.instance_id} (pid={fresh.process.pid})")
    if wait_for_ready(fresh):
        print(f"[launcher] instance {fresh.instance_id} ready again at {fresh.base_url}")
    else:
        print(f"[launcher] instance {fresh.instance_id} FAILED to rebind after restart")
    return fresh


def get_session_files(n_instances: int) -> list[str]:
    """Return list of session file paths for N instances."""
    return [str(session_path_for(i)) for i in range(n_instances)]


def main():
    parser = argparse.ArgumentParser(description="STS2 Multi-Instance Launcher")
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--game-exe", type=str, required=True,
                        help="Path to Slay the Spire 2 executable")
    parser.add_argument("--kill", action="store_true", help="Kill all game instances")
    parser.add_argument("--watchdog", action="store_true",
                        help="Enable health/log watchdog to kill+restart hung instances")
    parser.add_argument("--watchdog-strike-threshold", type=int, default=3,
                        help="Consecutive /health failures before restart")
    parser.add_argument("--watchdog-interval-s", type=float, default=15.0,
                        help="Seconds between watchdog ticks")
    parser.add_argument("--watchdog-probe-timeout-s", type=float, default=5.0,
                        help="Per-probe HTTP timeout (hung instance = timeout)")
    parser.add_argument("--watchdog-log-flood-mb-per-s", type=float, default=10.0,
                        help="Aggregate godot logs growth rate ceiling")
    parser.add_argument("--watchdog-state-stall-s", type=float, default=90.0,
                        help="Seconds of frozen /state state_version before "
                             "silent-hang kill+restart (0 disables)")
    parser.add_argument("--watchdog-state-probe-timeout-s", type=float, default=10.0,
                        help="/state probe HTTP timeout for state-stall detection")
    args = parser.parse_args()

    if args.kill:
        kill_all()
        return

    instances = launch_all(args.game_exe, args.instances)
    ready = [i for i in instances if i.ready]

    if not ready:
        print("[launcher] No instances ready. Exiting.")
        kill_all(instances)
        return

    print("\nSession files:")
    for inst in ready:
        print(f"  Instance {inst.instance_id}: {inst.session_path}")

    print("\nInstances running. Press Ctrl+C to stop all.")

    watchdog = None
    watchdog_logs_dir = None
    if args.watchdog:
        from sts2_env.path_utils import default_bridge_session_dir as _session_dir
        # Bridge session dir is %APPDATA%/SlayTheSpire2/bridge — logs live at
        # %APPDATA%/SlayTheSpire2/logs, which is the parent's sibling.
        watchdog_logs_dir = _session_dir().parent / "logs"
        watchdog = Watchdog(
            strike_threshold=args.watchdog_strike_threshold,
            probe_timeout_s=args.watchdog_probe_timeout_s,
            log_flood_bytes_per_sec=args.watchdog_log_flood_mb_per_s * 1024 * 1024,
            state_stall_threshold_s=args.watchdog_state_stall_s,
            state_probe_timeout_s=args.watchdog_state_probe_timeout_s,
            logs_dir=watchdog_logs_dir,
        )
        print(
            f"[launcher] watchdog enabled: interval={args.watchdog_interval_s}s "
            f"strike_threshold={args.watchdog_strike_threshold} "
            f"probe_timeout={args.watchdog_probe_timeout_s}s "
            f"log_flood_trip={args.watchdog_log_flood_mb_per_s} MB/s "
            f"state_stall={args.watchdog_state_stall_s}s "
            f"state_probe_timeout={args.watchdog_state_probe_timeout_s}s "
            f"logs_dir={watchdog_logs_dir}"
        )

    try:
        next_watchdog_tick = time.time() + (args.watchdog_interval_s if watchdog else 0.0)
        while True:
            time.sleep(5.0)
            for inst in instances:
                if inst.process.poll() is not None:
                    print(f"[launcher] Instance {inst.instance_id} died (exit={inst.process.returncode})")

            if watchdog and time.time() >= next_watchdog_tick:
                next_watchdog_tick = time.time() + args.watchdog_interval_s
                views = [
                    WatchdogInstanceView(
                        instance_id=inst.instance_id,
                        process_alive=inst.process.poll() is None,
                        base_url=inst.base_url,
                        token=inst.token,
                    )
                    for inst in instances
                ]
                decision = watchdog.evaluate(views, now_unix_s=time.time())
                if decision.log_flood_detected:
                    print(
                        f"[launcher] watchdog LOG FLOOD detected "
                        f"({decision.log_flood_bytes_per_sec / 1024 / 1024:.1f} MB/s) — "
                        f"killing all instances to break the loop"
                    )
                    new_instances: list[GameInstance] = []
                    for inst in instances:
                        kill_instance(inst, reason="log_flood")
                        new_instances.append(restart_instance(inst, args.game_exe))
                    instances = new_instances
                    continue
                for entry in decision.per_instance:
                    if entry.action is not InstanceAction.KILL_AND_RESTART:
                        continue
                    # Find the live object for this instance_id, replace in list.
                    for idx, inst in enumerate(instances):
                        if inst.instance_id != entry.instance_id:
                            continue
                        print(
                            f"[launcher] watchdog → instance {inst.instance_id}: "
                            f"{entry.reason}"
                        )
                        instances[idx] = restart_instance(inst, args.game_exe)
                        break
    except KeyboardInterrupt:
        print("\n[launcher] Shutting down...")
        kill_all(instances)


if __name__ == "__main__":
    main()
