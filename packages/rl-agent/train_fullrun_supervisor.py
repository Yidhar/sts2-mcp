"""Segmented full-run training supervisor with periodic game restarts.

This wrapper exists for the practical case where the live STS2 instance grows
slower over long sessions or occasionally exits, taking the bridge down with
it. Instead of relying on one long fragile training process, we train in
segments, restart the game between segments, and always resume from the newest
checkpoint directory.

Example:
    python train_fullrun_supervisor.py ^
      --game-exe "E:/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2/SlayTheSpire2.exe" ^
      --checkpoint-dir checkpoints/fullrun-supervised-01 ^
      --log-dir runs/fullrun-supervised-01 ^
      --total-timesteps 30000 ^
      --segment-timesteps 4000 ^
      --initial-checkpoint offline_runs/offline-build-v2-family-20260405-0020-broad-semantic-refresh-v2/best ^
      -- --learning-rate 1e-4 --batch-size 64 --n-steps 64 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import requests


def default_session_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "SlayTheSpire2" / "bridge" / "session.json"
    userprofile = os.environ.get("USERPROFILE", "")
    return Path(userprofile) / "AppData" / "Roaming" / "SlayTheSpire2" / "bridge" / "session.json"


def resolve_game_exe(path_or_dir: str) -> Path:
    candidate = Path(path_or_dir)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for name in ("SlayTheSpire2.exe", "Slay the Spire 2.exe"):
            exe = candidate / name
            if exe.exists():
                return exe
    raise FileNotFoundError(f"Could not resolve game executable from: {path_or_dir}")


def read_session(session_path: Path) -> dict | None:
    if not session_path.exists():
        return None
    try:
        return json.loads(session_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def wait_for_bridge_ready(session_path: Path, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    last_error = "session file not created"

    while time.time() < deadline:
        data = read_session(session_path)
        if data and data.get("base_url") and data.get("token"):
            base_url = str(data["base_url"]).rstrip("/")
            token = str(data["token"])
            try:
                resp = requests.get(
                    f"{base_url}/health",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    health = resp.json()
                    return {
                        "session": data,
                        "health": health,
                    }
                last_error = f"/health returned {resp.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
        time.sleep(1.0)

    raise RuntimeError(f"Bridge not ready after {timeout_s:.0f}s: {last_error}")


def terminate_process(proc: subprocess.Popen | None, timeout_s: float = 20.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout_s)
        return
    except Exception:
        pass
    try:
        proc.kill()
        proc.wait(timeout=5.0)
    except Exception:
        pass


def checkpoint_timesteps(checkpoint_dir: Path) -> int:
    metadata_path = checkpoint_dir / "metadata.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(data.get("timesteps", 0))


def iter_checkpoint_dirs(checkpoint_root: Path) -> Iterable[Path]:
    if not checkpoint_root.exists():
        return []
    return [
        path for path in checkpoint_root.iterdir()
        if path.is_dir()
        and (path / "metadata.json").exists()
        and (path / "model.safetensors").exists()
    ]


def latest_checkpoint(checkpoint_root: Path) -> Path | None:
    candidates = list(iter_checkpoint_dirs(checkpoint_root))
    if not candidates:
        return None
    candidates.sort(
        key=lambda path: (
            checkpoint_timesteps(path),
            path.stat().st_mtime,
            path.name,
        )
    )
    return candidates[-1]


def validate_passthrough_args(train_args: list[str]) -> None:
    forbidden = {
        "--total-timesteps",
        "--checkpoint-dir",
        "--log-dir",
        "--init-checkpoint",
    }
    conflicts = [arg for arg in train_args if arg in forbidden]
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(
            f"Do not pass {joined} after '--'; the supervisor owns those arguments."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-run RL training in restartable segments."
    )
    parser.add_argument("--game-exe", type=str, required=True,
                        help="Path to SlayTheSpire2.exe or its containing directory.")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, required=True,
                        help="Additional timesteps to train across all segments.")
    parser.add_argument("--segment-timesteps", type=int, default=4000,
                        help="Timesteps per training segment before restarting the game.")
    parser.add_argument("--initial-checkpoint", type=str, default=None,
                        help="Checkpoint used only when checkpoint-dir has no saved state yet.")
    parser.add_argument("--session-file", type=str, default=None,
                        help="Optional bridge session file path. Defaults to the standard session.json.")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=str, default="train_v2.py")
    parser.add_argument("--bridge-ready-timeout", type=float, default=180.0)
    parser.add_argument("--post-launch-delay", type=float, default=8.0)
    parser.add_argument("--post-exit-delay", type=float, default=3.0)
    parser.add_argument("--max-segment-retries", type=int, default=3)
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="Arguments forwarded to train_v2.py. Prefix with '--'.")
    args = parser.parse_args()

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    validate_passthrough_args(train_args)

    game_exe = resolve_game_exe(args.game_exe)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    train_script = Path(args.train_script)
    if not train_script.is_absolute():
        train_script = (script_dir / train_script).resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    session_path = Path(args.session_file).resolve() if args.session_file else default_session_path()

    current_checkpoint = latest_checkpoint(checkpoint_dir)
    if current_checkpoint is None and args.initial_checkpoint:
        current_checkpoint = Path(args.initial_checkpoint).resolve()

    current_timesteps = checkpoint_timesteps(current_checkpoint) if current_checkpoint else 0
    target_total = current_timesteps + int(args.total_timesteps)

    print(f"[supervisor] game_exe={game_exe}")
    print(f"[supervisor] checkpoint_dir={checkpoint_dir}")
    print(f"[supervisor] log_dir={log_dir}")
    print(f"[supervisor] start_timesteps={current_timesteps}")
    print(f"[supervisor] target_timesteps={target_total}")
    print(f"[supervisor] segment_timesteps={args.segment_timesteps}")
    if current_checkpoint:
        print(f"[supervisor] initial_resume_checkpoint={current_checkpoint}")

    segment_index = 0
    while current_timesteps < target_total:
        remaining = target_total - current_timesteps
        segment_timesteps = min(int(args.segment_timesteps), remaining)
        segment_index += 1
        attempt = 0
        segment_completed = False

        while not segment_completed:
            attempt += 1
            if attempt > int(args.max_segment_retries):
                raise RuntimeError(
                    f"Segment {segment_index} exceeded max retries ({args.max_segment_retries})."
                )

            if session_path.exists():
                try:
                    session_path.unlink()
                except OSError:
                    pass

            print(
                f"[supervisor] segment={segment_index} attempt={attempt} "
                f"start_timesteps={current_timesteps} train_for={segment_timesteps}"
            )

            game_proc: subprocess.Popen | None = None
            train_proc: subprocess.Popen | None = None
            segment_start_timesteps = current_timesteps
            try:
                game_proc = subprocess.Popen(
                    [str(game_exe)],
                    cwd=str(game_exe.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[supervisor] launched game pid={game_proc.pid}")
                time.sleep(max(float(args.post_launch_delay), 0.0))

                ready = wait_for_bridge_ready(session_path, timeout_s=float(args.bridge_ready_timeout))
                print(
                    "[supervisor] bridge ready "
                    f"session_id={ready['health'].get('session_id')} "
                    f"base_url={ready['health'].get('base_url')}"
                )

                cmd = [
                    str(Path(args.python_exe).resolve()),
                    str(train_script),
                    "--total-timesteps", str(segment_timesteps),
                    "--checkpoint-dir", str(checkpoint_dir),
                    "--log-dir", str(log_dir),
                ]
                if current_checkpoint:
                    cmd.extend(["--init-checkpoint", str(current_checkpoint)])
                if args.session_file:
                    cmd.extend(["--session-file", str(session_path)])
                cmd.extend(train_args)

                print("[supervisor] train cmd:", " ".join(cmd))
                train_proc = subprocess.Popen(cmd, cwd=str(script_dir))
                returncode = train_proc.wait()
                print(f"[supervisor] training exited code={returncode}")
            finally:
                terminate_process(train_proc, timeout_s=5.0)
                terminate_process(game_proc, timeout_s=20.0)
                time.sleep(max(float(args.post_exit_delay), 0.0))

            newest = latest_checkpoint(checkpoint_dir)
            if newest is None:
                raise RuntimeError("No checkpoint found after segment attempt.")

            newest_timesteps = checkpoint_timesteps(newest)
            delta = max(0, newest_timesteps - segment_start_timesteps)
            if delta > 0:
                current_checkpoint = newest
                current_timesteps = newest_timesteps
                print(
                    f"[supervisor] advanced +{delta} -> timesteps={current_timesteps} "
                    f"resume={current_checkpoint.name}"
                )
            else:
                print("[supervisor] no timestep advance detected in this attempt")

            if current_timesteps >= target_total:
                segment_completed = True
                continue

            if delta >= segment_timesteps:
                segment_completed = True
                continue

            print(
                f"[supervisor] segment incomplete (wanted {segment_timesteps}, got {delta}); "
                "restarting game and continuing."
            )

    print(f"[supervisor] done at timesteps={current_timesteps} checkpoint={current_checkpoint}")


if __name__ == "__main__":
    main()
