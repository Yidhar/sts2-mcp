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


def get_session_files(n_instances: int) -> list[str]:
    """Return list of session file paths for N instances."""
    return [str(session_path_for(i)) for i in range(n_instances)]


def main():
    parser = argparse.ArgumentParser(description="STS2 Multi-Instance Launcher")
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--game-exe", type=str, required=True,
                        help="Path to Slay the Spire 2 executable")
    parser.add_argument("--kill", action="store_true", help="Kill all game instances")
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
    try:
        while True:
            time.sleep(5.0)
            for inst in instances:
                if inst.process.poll() is not None:
                    print(f"[launcher] Instance {inst.instance_id} died (exit={inst.process.returncode})")
    except KeyboardInterrupt:
        print("\n[launcher] Shutting down...")
        kill_all(instances)


if __name__ == "__main__":
    main()
