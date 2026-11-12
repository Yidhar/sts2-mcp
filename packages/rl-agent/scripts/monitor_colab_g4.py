"""Non-blocking local monitor for a remote Colab G4 Durable Afterstate campaign.

Each check executes a short one-shot Python inspection on the remote Colab VM
via ``colab exec`` and returns immediately, keeping any ``--watch`` sleep loop
strictly on the local machine so ``Ctrl+C`` never blocks the remote Jupyter
kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

_REMOTE_PROBE_TEMPLATE = """
import json, subprocess
from pathlib import Path

pid_path = Path(__PID_PATH__)
log_path = Path(__LOG_PATH__)
local_dir = Path(__LOCAL_DIR__)
drive_dir = Path(__DRIVE_DIR__)

def is_live(pid: int) -> bool:
    st = Path(f"/proc/{pid}/stat")
    if not st.is_file():
        return False
    try:
        return st.read_text().rsplit(")", 1)[-1].strip().split()[0] != "Z"
    except OSError:
        return False

pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
alive = bool(pid is not None and is_live(pid))
sim_proc = subprocess.run(["pgrep", "-fc", "HeadlessSim"], capture_output=True, text=True)
sim_workers = int(sim_proc.stdout.strip() or "0") if sim_proc.returncode == 0 else 0

gpu_out = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
gpu_raw = gpu_out.stdout.strip() if gpu_out.returncode == 0 else ""

status = json.loads((local_dir / "status.json").read_text()) if (local_dir / "status.json").is_file() else {}
sync_manifest = (
    json.loads((drive_dir / "sync_manifest.json").read_text())
    if (drive_dir / "sync_manifest.json").is_file()
    else {}
)

# Check in-flight episode progress in the active cohort attempt directory.
active_attempt = status.get("attempt")
inflight_completed = 0
inflight_aborted = 0
if isinstance(active_attempt, str) and Path(active_attempt).is_dir():
    for col_file in Path(active_attempt).glob("episode-*/collection.json"):
        try:
            st_val = json.loads(col_file.read_text()).get("status")
            if st_val == "complete":
                inflight_completed += 1
            elif st_val == "aborted":
                inflight_aborted += 1
        except Exception:
            pass

updates = []
if (local_dir / "cohorts").is_dir():
    for upd_path in sorted((local_dir / "cohorts").glob("update-*/update.json"), key=lambda p: p.parent.name)[-12:]:
        try:
            raw = json.loads(upd_path.read_text())
            lrn = raw.get("learner", {})
            eps = [e for e in raw.get("training_episodes", []) if isinstance(e, dict)]
            wins = sum(1 for e in eps if e.get("run_result") == "victory")
            floors = [float(e["max_floor"]) for e in eps if isinstance(e.get("max_floor"), int | float)]
            revivs = [float(e["revivals_used"]) for e in eps if isinstance(e.get("revivals_used"), int | float)]
            updates.append({
                "u": raw.get("committed_update"),
                "episodes": len(eps),
                "win_rate": wins / max(1, len(eps)),
                "mean_floor": sum(floors) / len(floors) if floors else None,
                "mean_revivals": sum(revivs) / len(revivs) if revivs else None,
                "v_dur_loss": lrn.get("durable_value_loss"),
                "v_post_loss": lrn.get("durable_post_value_loss"),
                "mpo_loss": lrn.get("macro_mpo_policy_loss"),
                "pg_loss": lrn.get("macro_pg_policy_loss"),
                "afterstate_proj": lrn.get("enabled_afterstate_projections"),
                "actor_on": lrn.get("durable_actor_active"),
                "combat_on": lrn.get("combat_actor_active"),
                "grad_norm": lrn.get("gradient_norm"),
            })
        except Exception:
            pass

history_rows = []
hist_path = local_dir / "campaign_history.jsonl"
if hist_path.is_file():
    for line in hist_path.read_text().splitlines()[-12:]:
        if line.strip():
            try:
                history_rows.append(json.loads(line))
            except Exception:
                pass

snapshots = (
    sorted(p.name for p in (drive_dir / "snapshots").iterdir() if p.is_dir())
    if (drive_dir / "snapshots").is_dir()
    else []
)
log_lines = log_path.read_text(errors="replace").splitlines()[-5:] if log_path.is_file() else []

print("MONITOR_PAYLOAD_BEGIN")
print(json.dumps({
    "pid": pid,
    "alive": alive,
    "sim_workers": sim_workers,
    "gpu_raw": gpu_raw,
    "status": status,
    "inflight_completed": inflight_completed,
    "inflight_aborted": inflight_aborted,
    "sync_manifest": sync_manifest,
    "updates": updates,
    "history": history_rows,
    "snapshots": snapshots,
    "log_tail": log_lines,
}))
print("MONITOR_PAYLOAD_END")
"""


def _build_remote_script(
    *,
    pid_path: str,
    log_path: str,
    local_dir: str,
    drive_dir: str,
) -> str:
    return (
        _REMOTE_PROBE_TEMPLATE.replace("__PID_PATH__", repr(pid_path))
        .replace("__LOG_PATH__", repr(log_path))
        .replace("__LOCAL_DIR__", repr(local_dir))
        .replace("__DRIVE_DIR__", repr(drive_dir))
    )


def fetch_remote_snapshot(
    *,
    session: str,
    use_wsl: bool,
    pid_path: str,
    log_path: str,
    local_dir: str,
    drive_dir: str,
    timeout_seconds: int = 45,
) -> dict[str, Any]:
    script_text = _build_remote_script(
        pid_path=pid_path,
        log_path=log_path,
        local_dir=local_dir,
        drive_dir=drive_dir,
    )
    if use_wsl:
        q_session = shlex.quote(session)
        bash_cmd = (
            'tmpf="/tmp/sts2_g4_probe_$$.py"; '
            'trap "rm -f $tmpf" EXIT; '
            'cat > "$tmpf" && '
            f'export PATH="$HOME/.local/bin:$PATH" && '
            f'colab exec -s {q_session} --timeout {int(timeout_seconds)} -f "$tmpf"'
        )
        completed = subprocess.run(
            ["wsl", "-e", "bash", "-lc", bash_cmd],
            input=script_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds + 15,
            check=False,
        )
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
            handle.write(script_text)
            temp_path = Path(handle.name)
        try:
            completed = subprocess.run(
                ["colab", "exec", "-s", session, "--timeout", str(timeout_seconds), "-f", str(temp_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds + 15,
                check=False,
            )
        finally:
            temp_path.unlink(missing_ok=True)
    combined = f"{completed.stdout}\n{completed.stderr}"
    if "MONITOR_PAYLOAD_BEGIN" not in combined or "MONITOR_PAYLOAD_END" not in combined:
        raise RuntimeError(f"colab exec failed (code {completed.returncode}): {combined.strip()[-800:]}")
    body = combined.split("MONITOR_PAYLOAD_BEGIN", 1)[1].split("MONITOR_PAYLOAD_END", 1)[0].strip()
    payload: dict[str, Any] = json.loads(body)
    return payload


def _phase_description(committed_update: int) -> str:
    next_upd = committed_update + 1
    if next_upd <= 6:
        return f"Phase 1/3 (Updates 1..6): V_durable Critic Calibration [next={next_upd}]"
    if next_upd <= 8:
        return f"Phase 2/3 (Updates 7..8): Macro Afterstate MPO + V-trace Actor Active [next={next_upd}]"
    return f"Phase 3/3 (Updates 9+): Joint Macro + Combat Co-Adaptation Active [next={next_upd}]"


def _estimate_minutes_per_update(history: Sequence[dict[str, Any]]) -> float | None:
    timestamps: list[tuple[int, float]] = []
    for row in history:
        upd = row.get("committed_update")
        ts_raw = row.get("completed_at_utc")
        if isinstance(upd, int) and isinstance(ts_raw, str):
            try:
                timestamps.append((upd, datetime.fromisoformat(ts_raw).timestamp()))
            except ValueError:
                continue
    if len(timestamps) < 2:
        return None
    u0, t0 = timestamps[0]
    u1, t1 = timestamps[-1]
    if u1 <= u0 or t1 <= t0:
        return None
    return (t1 - t0) / 60.0 / float(u1 - u0)


def format_dashboard(payload: dict[str, Any], *, session: str, target_updates: int) -> str:
    lines: list[str] = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alive = bool(payload.get("alive"))
    pid = payload.get("pid")
    sim_workers = int(payload.get("sim_workers") or 0)
    status = payload.get("status") or {}
    state = str(status.get("state") or "unknown")
    committed = int(status.get("committed_update") or 0)
    seed_cursor = status.get("next_seed_index")
    inflight_done = int(payload.get("inflight_completed") or 0)
    inflight_aborted = int(payload.get("inflight_aborted") or 0)

    gpu_parts = [p.strip() for p in str(payload.get("gpu_raw") or "").split(",")]
    gpu_summary = (
        f"{gpu_parts[0]} | Util {gpu_parts[1]}% | VRAM {gpu_parts[2]}/{gpu_parts[3]} MiB"
        if len(gpu_parts) >= 4
        else str(payload.get("gpu_raw") or "N/A")
    )

    history = payload.get("history") or []
    mins_per_upd = _estimate_minutes_per_update(history)
    remaining_updates = max(0, target_updates - committed)
    eta_str = (
        f"~{mins_per_upd * remaining_updates / 60.0:.1f}h ({mins_per_upd:.1f} min/upd)"
        if mins_per_upd is not None
        else "~7.5 min/upd (calibrating)"
    )

    lines.append(f"=== Colab G4 Durable Afterstate v2 Monitor [{now_str}] ===")
    lines.append(f"Session       : {session} | Daemon PID: {pid} ({'RUNNING' if alive else 'STOPPED'})")
    lines.append(f"GPU           : {gpu_summary}")
    lines.append(
        f"Live State    : {state} | HeadlessSim workers: {sim_workers} | "
        f"Current cohort episodes done: {inflight_done}/8 (retried: {inflight_aborted})"
    )
    lines.append(f"Progress      : {committed} / {target_updates} updates ({100.0 * committed / max(1, target_updates):.1f}%) | Seed cursor: {seed_cursor} | ETA: {eta_str}")
    lines.append(f"Schedule      : {_phase_description(committed)}")

    sync = payload.get("sync_manifest") or {}
    ckpt_info = sync.get("checkpoint") or {}
    sha_short = str(ckpt_info.get("sha256") or "")[:12]
    ckpt_mb = float(ckpt_info.get("bytes") or 0) / (1024 * 1024)
    snapshots = payload.get("snapshots") or []
    lines.append(
        f"Google Drive  : synced update={sync.get('committed_update', 0)} | "
        f"checkpoint={ckpt_mb:.1f} MiB (sha256:{sha_short or 'n/a'}) | snapshots={snapshots[-4:]}"
    )
    lines.append("")

    hist_by_upd = {int(r["committed_update"]): r for r in history if isinstance(r.get("committed_update"), int)}
    updates = payload.get("updates") or []
    if updates:
        lines.append(
            "Upd | Mode   |  Win% | Floor | Reviv | Skip% | Rest% | Rem/Shp | V_dur  | MPO    | Rest(L-H) | JSD(S/R)"
        )
        lines.append("-" * 105)
        for row in updates:
            u = int(row.get("u") or 0)
            h = hist_by_upd.get(u, {})
            disc = h.get("episode_macro_discipline") or {}
            mode_tag = "Joint" if row.get("combat_on") else ("Macro" if row.get("actor_on") else "Warmup")
            win = f"{100.0 * float(row['win_rate']):5.1f}" if isinstance(row.get("win_rate"), int | float) else "  n/a"
            flr = f"{row['mean_floor']:5.1f}" if isinstance(row.get("mean_floor"), int | float) else "  n/a"
            rev = f"{row['mean_revivals']:5.1f}" if isinstance(row.get("mean_revivals"), int | float) else "  n/a"
            skp = f"{100.0 * float(disc['card_reward_skip_rate']):5.1f}" if isinstance(disc.get("card_reward_skip_rate"), int | float) else "  n/a"
            rst = f"{100.0 * float(disc['campfire_rest_rate']):5.1f}" if isinstance(disc.get("campfire_rest_rate"), int | float) else "  n/a"
            rem = f"{float(disc['shop_remove_per_visit_rate']):7.2f}" if isinstance(disc.get("shop_remove_per_visit_rate"), int | float) else "    n/a"
            vd = f"{float(row['v_dur_loss']):6.4f}" if isinstance(row.get("v_dur_loss"), int | float) else "   n/a"
            mpo = f"{float(row['mpo_loss']):6.4f}" if isinstance(row.get("mpo_loss"), int | float) else "   n/a"
            rlh = (
                f"{float(h['campfire_low_minus_high_rest_prob']):+9.4f}"
                if isinstance(h.get("campfire_low_minus_high_rest_prob"), int | float)
                else "      n/a"
            )
            jsd = (
                f"{float(h['smith_vs_remove_jsd']):8.5f}"
                if isinstance(h.get("smith_vs_remove_jsd"), int | float)
                else "     n/a"
            )
            lines.append(f"{u:3d} | {mode_tag:<6} | {win} | {flr} | {rev} | {skp} | {rst} | {rem} | {vd} | {mpo} | {rlh} | {jsd}")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="sts2-macro-g4", help="Colab CLI session name")
    parser.add_argument("--target-updates", type=int, default=120)
    parser.add_argument("--watch", action="store_true", help="Refresh periodically on the local machine")
    parser.add_argument("--interval", type=float, default=30.0, help="Refresh interval in seconds when --watch is set")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON snapshot")
    parser.add_argument("--pid-path", default="/content/sts2_artifacts/campaign_pid.txt")
    parser.add_argument("--log-path", default="/content/sts2_artifacts/campaign_stdout.log")
    parser.add_argument("--local-dir", default="/content/sts2_artifacts/campaigns/durable-afterstate-v2-b64-g4")
    parser.add_argument(
        "--drive-dir",
        default="/content/drive/MyDrive/sts2_mcp_artifacts/campaigns/durable-afterstate-v2-b64-g4",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    use_wsl = os.name == "nt"
    while True:
        snapshot = fetch_remote_snapshot(
            session=args.session,
            use_wsl=use_wsl,
            pid_path=args.pid_path,
            log_path=args.log_path,
            local_dir=args.local_dir,
            drive_dir=args.drive_dir,
        )
        if args.json:
            print(json.dumps(snapshot, indent=2, ensure_ascii=False), flush=True)
        else:
            if args.watch:
                print("\033[2J\033[H", end="")
            print(format_dashboard(snapshot, session=args.session, target_updates=args.target_updates), flush=True)
            if args.watch:
                print(f"\n[Local watch mode: refreshing every {args.interval:.0f}s — press Ctrl+C to exit safely]", flush=True)
        if not args.watch:
            return 0
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0) from None
