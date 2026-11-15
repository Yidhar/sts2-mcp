"""Synchronize a remote Colab G4 Durable Afterstate campaign into the local 8765 dashboard.

This read-only local bridge queries the remote Colab G4 VM via short one-shot
``colab exec`` calls (never blocking the remote Jupyter kernel), streams
committed ``update.json`` + ``campaign_history.jsonl`` metrics into a canonical
local ``runs/<lineage>/run-<uuid>/metrics.jsonl``, and deterministically
reconstructs completed cohort episodes via the local verified ``HeadlessSim``
into canonical ``evaluation-step-XXXXXXXXX.jsonl`` journals.

That enables full drill-down in ``http://127.0.0.1:8765/`` including:
- Learner loss / gradient / episode floor / revival / win-rate curves
- Exact card reward picks & skips with offered alternatives
- Campfire Rest/Smith upgrades, Shop purchases & card removals, Event choices
- Final deck (with ``+`` upgrades), relics, and potions
- Interactive Act 1 / Act 2 / Act 3 map topologies via ``/api/v1/heldout-replay-map``
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_DNS, uuid4, uuid5

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sts2_env.headless_sim_bridge_client import resolve_headless_sim_exe  # noqa: E402
from sts2_rl.artifacts import artifact_root  # noqa: E402
from sts2_rl.backends import HeadlessBackend  # noqa: E402
from sts2_rl.contracts import ResetRequest, StepRequest  # noqa: E402
from sts2_rl.simulator_identity import VerifiedSimulatorIdentity, verify_headless_simulator  # noqa: E402
from sts2_rl.training.trajectory import TrajectoryJournal  # noqa: E402

_REMOTE_PULL_TEMPLATE = """
import json, subprocess
from pathlib import Path

pid_path = Path(__PID_PATH__)
local_dir = Path(__LOCAL_DIR__)
already_updates = set(__ALREADY_UPDATES__)
already_journals = set(__ALREADY_JOURNALS__)

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
status = json.loads((local_dir / "status.json").read_text()) if (local_dir / "status.json").is_file() else {}
remote_revival_budget = None
course_file = local_dir / "course.json"
if course_file.is_file():
    try:
        course_raw = json.loads(course_file.read_text())
        rb = (course_raw.get("reward_identity") or {}).get("revival_budget")
        if isinstance(rb, int) and not isinstance(rb, bool) and rb > 0:
            remote_revival_budget = int(rb)
    except Exception:
        pass

cohorts_dir = local_dir / "cohorts"
updates = []
cohort_replays = {}

if cohorts_dir.is_dir():
    replay_candidates = []
    for c_dir in sorted(cohorts_dir.glob("update-*"), key=lambda p: p.name):
        u_file = c_dir / "update.json"
        if not u_file.is_file():
            continue
        try:
            u = json.loads(u_file.read_text())
            upd_num = int(u["committed_update"])
            if upd_num not in already_updates:
                u["_mtime"] = u_file.stat().st_mtime
                updates.append(u)
        except Exception:
            continue
        if upd_num not in already_journals and any(c_dir.glob("episode-*/episode.jsonl")):
            replay_candidates.append((upd_num, c_dir))

    # Pick at most 1 cohort per sync tick (preferring milestone updates divisible by 5, else latest)
    # and cap to top 8 highlight episodes (all victories + deepest floor) so local replay stays fast.
    chosen_replays = [item for item in replay_candidates if item[0] % 5 == 0]
    selected_pair = chosen_replays[-1] if chosen_replays else (replay_candidates[-1] if replay_candidates else None)
    if selected_pair is not None:
        upd_num, c_dir = selected_pair
        ep_metas = []
        for ep_dir in sorted(c_dir.glob("episode-*"), key=lambda p: p.name):
            coll_file = ep_dir / "collection.json"
            ep_file = ep_dir / "episode.jsonl"
            if not coll_file.is_file() or not ep_file.is_file():
                continue
            try:
                coll = json.loads(coll_file.read_text())
                if coll.get("status") != "complete":
                    continue
                te = coll.get("training_episode", {})
                won = 1 if str(te.get("run_result") or "") == "victory" else 0
                max_fl = int(te.get("max_floor") or 0)
                rev_used = int(te.get("revivals_used") or 0)
                ep_metas.append(((won, max_fl, -rev_used), ep_dir, coll, te, ep_file))
            except Exception:
                continue
        ep_metas.sort(key=lambda item: item[0], reverse=True)
        eps = []
        for _, ep_dir, coll, te, ep_file in ep_metas[:4]:
            try:
                actions = []
                with ep_file.open() as f:
                    for line in f:
                        row = json.loads(line)
                        if row.get("kind") == "decision":
                            facts = row.get("facts", {})
                            ref = facts.get("selected_reference", {})
                            sem = facts.get("semantic", {})
                            actions.append([
                                int(facts.get("native_steps", len(actions))),
                                int(ref.get("position", 0)),
                                sem.get("behavior_log_probability"),
                                sem.get("candidate_count"),
                            ])
                eps.append({
                    "episode_dir": ep_dir.name,
                    "seed": coll.get("seed"),
                    "training_episode": te,
                    "actions": actions,
                })
            except Exception:
                continue
        if eps:
            cohort_replays[str(upd_num)] = eps

history = []
hist_path = local_dir / "campaign_history.jsonl"
if hist_path.is_file():
    for line in hist_path.read_text().splitlines():
        if line.strip():
            try:
                history.append(json.loads(line))
            except Exception:
                pass

print("PULLED_JSON_START")
print(json.dumps({
    "alive": alive,
    "pid": pid,
    "sim_workers": sim_workers,
    "status": status,
    "revival_budget": remote_revival_budget,
    "updates": updates,
    "history": history,
    "cohort_replays": cohort_replays,
}, separators=(",", ":")))
print("PULLED_JSON_END")
"""

_MACRO_SCREENS = frozenset(
    {
        "CARD_REWARD_SELECTION",
        "REST_SITE",
        "SHOP",
        "MAP",
        "EVENT",
        "CARD_SELECTION",
    }
)


def _extract_marked_json(stdout: str, begin_marker: str, end_marker: str) -> dict[str, Any]:
    if begin_marker not in stdout or end_marker not in stdout:
        tail = stdout.strip()[-600:]
        raise RuntimeError(f"remote payload markers missing: {tail}")
    body = stdout.split(begin_marker, 1)[1].split(end_marker, 1)[0].strip()
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("remote payload must be a JSON object")
    return parsed


_WSL_BRIDGE_SERVER_SCRIPT = """
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

colab_bin = os.environ.get("COLAB_BIN") or str(Path.home() / ".local" / "bin" / "colab")
if not Path(colab_bin).is_file():
    colab_bin = shutil.which("colab") or colab_bin

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    req = json.loads(line)
    session = str(req["session"])
    timeout_s = int(req["timeout_seconds"])
    script = str(req["script"])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(script)
        tmp_path = Path(handle.name)
    try:
        proc = subprocess.run(
            [colab_bin, "exec", "-s", session, "--timeout", str(timeout_s), "-f", str(tmp_path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s + 20,
            check=False,
        )
        resp = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        resp = {
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    finally:
        tmp_path.unlink(missing_ok=True)
    sys.stdout.write(json.dumps(resp, separators=(",", ":")) + "\\n")
    sys.stdout.flush()
"""


class _PersistentWslColabBridge:
    """Keep a single WSL worker alive on Windows to avoid WSL2 hvsocket exhaustion (0x80072747)."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._out_queue: queue.Queue[str | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self.close()
        self._out_queue = queue.Queue()
        proc = subprocess.Popen(
            ["wsl", "-e", "python3", "-u", "-c", _WSL_BRIDGE_SERVER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._proc = proc
        out_q = self._out_queue

        def _reader() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    out_q.put(line)
            finally:
                out_q.put(None)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        self._reader_thread = thread
        return proc

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def exec_script(self, *, session: str, script: str, timeout_seconds: int) -> tuple[int, str, str]:
        proc = self._ensure_started()
        assert proc.stdin is not None
        payload = json.dumps(
            {"session": session, "timeout_seconds": int(timeout_seconds), "script": script},
            separators=(",", ":"),
        )
        try:
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
            raw = self._out_queue.get(timeout=timeout_seconds + 25)
        except Exception:
            self.close()
            raise
        if raw is None:
            self.close()
            raise RuntimeError("persistent WSL colab bridge closed unexpectedly")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.close()
            raise RuntimeError(f"persistent WSL colab bridge non-JSON output: {raw.strip()[-400:]}") from exc
        return (
            int(parsed.get("returncode", -1)),
            str(parsed.get("stdout") or ""),
            str(parsed.get("stderr") or ""),
        )


_WSL_BRIDGE = _PersistentWslColabBridge()
atexit.register(_WSL_BRIDGE.close)


def _fetch_remote_bundle(
    *,
    session: str,
    pid_path: str,
    remote_local_dir: str,
    already_updates: Sequence[int],
    already_journals: Sequence[int],
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    script = (
        _REMOTE_PULL_TEMPLATE.replace("__PID_PATH__", json.dumps(pid_path))
        .replace("__LOCAL_DIR__", json.dumps(remote_local_dir))
        .replace("__ALREADY_UPDATES__", json.dumps([int(x) for x in already_updates]))
        .replace("__ALREADY_JOURNALS__", json.dumps([int(x) for x in already_journals]))
    )
    if os.name == "nt":
        returncode, stdout, stderr = _WSL_BRIDGE.exec_script(
            session=session,
            script=script,
            timeout_seconds=int(timeout_seconds),
        )
    else:
        colab_bin = os.environ.get("COLAB_BIN") or str(Path.home() / ".local" / "bin" / "colab")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(script)
            tmp_path = Path(handle.name)
        try:
            proc = subprocess.run(
                [colab_bin, "exec", "-s", session, "--timeout", str(int(timeout_seconds)), "-f", str(tmp_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds + 20,
                check=False,
            )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        finally:
            tmp_path.unlink(missing_ok=True)

    if returncode != 0:
        detail = (stderr or stdout).strip()[-600:]
        raise RuntimeError(f"colab exec failed (exit {returncode}): {detail}")
    return _extract_marked_json(stdout, "PULLED_JSON_START", "PULLED_JSON_END")


def _load_seed_pull_cache(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8", errors="replace")
    if "PULLED_JSON_START" in raw and "PULLED_JSON_END" in raw:
        return _extract_marked_json(raw, "PULLED_JSON_START", "PULLED_JSON_END")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    fval = float(value)
    return fval if math.isfinite(fval) else None


def _reconstruct_heldout_journal(
    *,
    journal_path: Path,
    gate: int,
    cohort_episodes: Sequence[Mapping[str, Any]],
    simulator: VerifiedSimulatorIdentity,
    revival_budget: int = 64,
) -> int:
    """Replay recorded ``(seed, position)`` streams through HeadlessBackend into TrajectoryJournal."""
    sim_identity_payload = json.loads(simulator.identity_path.read_text(encoding="utf-8"))
    seeds = [int(ep["seed"]) for ep in cohort_episodes if isinstance(ep.get("seed"), int)]
    tmp_path = journal_path.with_name(f".{journal_path.name}.tmp-{uuid4().hex[:8]}")
    reconstructed_episodes = 0

    backend: HeadlessBackend | None = HeadlessBackend(exe_path=simulator.executable)
    try:
        with TrajectoryJournal(tmp_path) as journal:
            journal.write_episode_boundary(
                {
                    "event": "evaluation_started",
                    "evaluation_gate": gate,
                    "evaluation_attempt": 1,
                    "gate_kind": "validation",
                    "data_partition": "validation",
                    "policy_version": gate,
                    "actor_policy_version": max(0, gate - 1),
                    "evaluation_seeds": seeds,
                    "epsilon": 0.05,
                    "deterministic": False,
                    "journal_version": "sts2-trajectory-journal-v2",
                    "simulator": {
                        "backend": "headless",
                        "simulator_identity": sim_identity_payload,
                    },
                }
            )

            for ep in cohort_episodes:
                raw_seed = ep.get("seed")
                raw_actions = ep.get("actions")
                if not isinstance(raw_seed, int) or not isinstance(raw_actions, list) or not raw_actions:
                    continue
                seed = int(raw_seed)
                te = ep.get("training_episode")
                te_map = te if isinstance(te, Mapping) else {}
                run_won = str(te_map.get("run_result") or "") == "victory"
                ep_prefix = f"heldout-seed-{seed}-attempt-1:"

                journal.write_episode_boundary(
                    {
                        "event": "evaluation_attempt_started",
                        "evaluation_seed": seed,
                        "attempt": 1,
                        "journal_episode_namespace": ep_prefix,
                        "data_partition": "validation",
                    }
                )

                if backend is None:
                    backend = HeadlessBackend(exe_path=simulator.executable)

                try:
                    revision = int(backend.get_state()["state_version"])
                    state = backend.reset(
                        ResetRequest(
                            request_id=str(uuid4()),
                            session_id=backend.session_id,
                            scenario="full-run",
                            expected_state_version=revision,
                            character="IRONCLAD",
                            seed=seed,
                            force_fresh=True,
                            defensive_buffs=False,
                            additional_relics=None,
                            training_revival_budget=revival_budget,
                            timeout_ms=45_000,
                        )
                    )
                    full_ep_id = f"{ep_prefix}{state.episode_id}"
                    steps_replayed = 0

                    for action_idx, item in enumerate(raw_actions):
                        if state.terminated or state.truncated:
                            break
                        if not isinstance(item, list) or len(item) < 2:
                            break
                        pos = int(item[1])
                        beh_logp = _finite(item[2]) if len(item) > 2 else None
                        legal_actions = state.legal_actions
                        if not 0 <= pos < len(legal_actions):
                            raise RuntimeError(
                                f"seed={seed} step={state.step_index} pos={pos} out of {len(legal_actions)} legal actions"
                            )
                        selected_action = legal_actions[pos]
                        raw_handle = selected_action.get("action_handle", selected_action.get("action_id"))
                        handle = str(raw_handle) if raw_handle is not None and str(raw_handle) else None

                        screen = str(state.observation.get("screen") or "")
                        max_topk = 8 if screen in _MACRO_SCREENS else 4
                        candidate_indices = [pos] + [i for i in range(len(legal_actions)) if i != pos]
                        candidate_indices = candidate_indices[:max_topk]

                        p_sel: float | None = None
                        if beh_logp is not None:
                            p_sel = min(1.0, max(0.0, math.exp(beh_logp)))
                        elif len(legal_actions) == 1:
                            p_sel = 1.0
                        other_prob: float | None = None
                        if p_sel is not None and len(legal_actions) > 1:
                            other_prob = max(0.0, 1.0 - p_sel) / float(len(legal_actions) - 1)

                        policy_topk = [
                            {
                                "index": idx,
                                "candidate_index": idx,
                                "dispatch_index": idx,
                                "multiplicity": 1,
                                "probability": p_sel if idx == pos else other_prob,
                            }
                            for idx in candidate_indices
                        ]

                        next_state = backend.step(
                            StepRequest(
                                request_id=str(uuid4()),
                                session_id=backend.session_id,
                                episode_id=state.episode_id,
                                expected_step_index=state.step_index,
                                action_id=handle,
                                action_index=None if handle else pos,
                                timeout_ms=20_000,
                            )
                        )
                        steps_replayed += 1

                        obs_training = state.observation.get("_training")
                        next_training = next_state.observation.get("_training")
                        tr_map = next_training if isinstance(next_training, Mapping) else (
                            obs_training if isinstance(obs_training, Mapping) else {}
                        )
                        hp_lost = _finite(tr_map.get("player_hp_lost")) or 0.0
                        revivals_used = int(_finite(tr_map.get("revivals_used")) or 0)

                        is_last = (
                            action_idx == len(raw_actions) - 1
                            or next_state.terminated
                            or next_state.truncated
                        )
                        outcome = ("success" if run_won else "failure") if is_last else "ongoing"

                        journal_event: dict[str, Any] = {
                            "event": "decision",
                            "episode_id": full_ep_id,
                            "reset_seed": seed,
                            "step_index": state.step_index,
                            "observation": state.observation,
                            "legal_actions": legal_actions,
                            "selected_index": pos,
                            "selected_candidate_index": pos,
                            "selected_dispatch_index": pos,
                            "selected_action_multiplicity": 1,
                            "semantic_candidate_count": len(legal_actions),
                            "raw_legal_action_count": len(legal_actions),
                            "maximum_candidate_multiplicity": 1,
                            "selected_action": selected_action,
                            "policy_topk": policy_topk,
                            "behavior_log_probability": beh_logp,
                            "effective_collection_epsilon": 0.05,
                            "player_hp_lost": hp_lost,
                            "revivals_used": revivals_used,
                            "outcome": outcome,
                        }
                        if is_last:
                            journal_event.update(
                                {
                                    "result_step_index": next_state.step_index,
                                    "result_observation": next_state.observation,
                                    "result_legal_actions": next_state.legal_actions,
                                    "result_terminated": next_state.terminated,
                                    "result_truncated": next_state.truncated,
                                    "result_terminal_reason": next_state.terminal_reason,
                                }
                            )
                        journal.write(journal_event)
                        state = next_state

                    journal.write_episode_boundary(
                        {
                            "event": "evaluation_attempt_completed",
                            "evaluation_seed": seed,
                            "attempt": 1,
                            "episode_id": full_ep_id,
                            "steps": steps_replayed,
                        }
                    )
                    reconstructed_episodes += 1
                except Exception as exc:
                    print(f"[sync] warning: replay failed on update={gate} seed={seed}: {exc}", flush=True)
                    journal.write_episode_boundary(
                        {
                            "event": "evaluation_attempt_aborted",
                            "evaluation_seed": seed,
                            "attempt": 1,
                            "exception_type": type(exc).__name__,
                            "fingerprint": "",
                            "message": str(exc)[:512],
                            "retryable": False,
                        }
                    )
                    backend.close()
                    backend = None
    finally:
        if backend is not None:
            backend.close()

    if reconstructed_episodes > 0:
        os.replace(tmp_path, journal_path)
    else:
        tmp_path.unlink(missing_ok=True)
    return reconstructed_episodes


def sync_once(
    *,
    root: Path,
    simulator: VerifiedSimulatorIdentity,
    session: str,
    lineage: str,
    run_uuid: str,
    pid_path: str,
    remote_local_dir: str,
    target_updates: int,
    revival_budget: int = 64,
    seed_pull_file: Path | None = None,
    offline_only: bool = False,
) -> dict[str, Any]:
    run_dir = root / "runs" / lineage / f"run-{run_uuid}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    sync_state_dir = root / "colab_g4_sync" / lineage
    sync_state_dir.mkdir(parents=True, exist_ok=True)
    state_file = sync_state_dir / "sync_state.json"

    state_data: dict[str, Any] = {
        "synced_updates": [],
        "synced_journals": [],
        "cumulative_env_steps": 0,
        "last_unix_s": 0.0,
    }
    if state_file.is_file():
        try:
            loaded_state = json.loads(state_file.read_text(encoding="utf-8"))
            if isinstance(loaded_state, dict):
                state_data.update(loaded_state)
        except Exception:
            pass

    synced_updates: set[int] = {int(x) for x in state_data.get("synced_updates", [])}
    synced_journals: set[int] = {int(x) for x in state_data.get("synced_journals", [])}
    # Also scan existing evaluation-step-*.jsonl files in run_dir
    for existing_j in run_dir.glob("evaluation-step-*.jsonl"):
        stem_digits = existing_j.stem.removeprefix("evaluation-step-")
        if stem_digits.isdigit():
            synced_journals.add(int(stem_digits))

    seed_bundle = _load_seed_pull_cache(seed_pull_file)
    if offline_only:
        remote_bundle = seed_bundle
    else:
        remote_bundle = _fetch_remote_bundle(
            session=session,
            pid_path=pid_path,
            remote_local_dir=remote_local_dir,
            already_updates=sorted(synced_updates),
            already_journals=sorted(synced_journals),
        )

    # Merge updates & cohort_replays from seed_bundle + remote_bundle
    updates_by_num: dict[int, dict[str, Any]] = {}
    for src in (seed_bundle.get("updates", []), remote_bundle.get("updates", [])):
        if isinstance(src, list):
            for u in src:
                if isinstance(u, dict) and isinstance(u.get("committed_update"), int):
                    updates_by_num[int(u["committed_update"])] = u

    cohort_replays: dict[int, list[dict[str, Any]]] = {}
    for src in (seed_bundle.get("cohort_replays", {}), remote_bundle.get("cohort_replays", {})):
        if isinstance(src, Mapping):
            for k, v in src.items():
                if str(k).isdigit() and isinstance(v, list) and v:
                    cohort_replays[int(k)] = [row for row in v if isinstance(row, dict)]

    cumulative_env_steps = int(state_data.get("cumulative_env_steps", 0))
    last_unix_s = float(state_data.get("last_unix_s", 0.0))

    def monotonic_ts(candidate: float | None) -> float:
        nonlocal last_unix_s
        ts = candidate if (candidate is not None and math.isfinite(candidate) and candidate > 0) else time.time()
        if ts <= last_unix_s:
            ts = last_unix_s + 0.001
        last_unix_s = ts
        return ts

    events_to_append: list[dict[str, Any]] = []

    effective_revival_budget = int(revival_budget)
    rb_remote = remote_bundle.get("revival_budget")
    if isinstance(rb_remote, int) and not isinstance(rb_remote, bool) and rb_remote > 0:
        effective_revival_budget = int(rb_remote)

    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        earliest_mtime = _finite(updates_by_num[min(updates_by_num)].get("_mtime")) if updates_by_num else None
        first_mtime = (earliest_mtime - 420.0) if earliest_mtime is not None else (time.time() - 60.0)
        start_ts = monotonic_ts(first_mtime)
        events_to_append.append(
            {
                "event": "run_start",
                "run_id": run_uuid,
                "pipeline": "durable-afterstate-macro-credit-v2",
                "unix_s": start_ts,
                "config": {
                    "profile": (
                        "durable_afterstate_v4"
                        if "v4" in lineage
                        else ("durable_afterstate_v3" if "v3" in lineage else "durable_afterstate_v2")
                    ),
                    "version": (
                        "sts2-durable-afterstate-v5-wide256"
                        if "v5" in lineage
                        else (
                            "sts2-durable-afterstate-v4"
                            if "v4" in lineage
                            else (
                                "sts2-durable-afterstate-v3"
                                if "v3" in lineage
                                else "sts2-durable-afterstate-v2"
                            )
                        )
                    ),
                    "curriculum": {
                        "mode": "native-revival-preheat",
                        "reward_objective": "run",
                        "revival_budget": effective_revival_budget,
                    },
                    "model": {
                        "max_candidates": 128,
                    },
                    "runtime": {
                        "total_environment_steps": target_updates * 48 * 450,
                        "device": "cuda:colab-g4",
                        "seed": 13440000,
                    },
                },
                "state": {
                    "environment_steps": 0,
                    "policy_version": 0,
                    "episodes": 0,
                    "evaluation_episodes": 0,
                },
            }
        )

    # Flush immediate live heartbeat BEFORE running local HeadlessSim journal replay so telemetry age resets right away.
    if bool(remote_bundle.get("alive")) and metrics_path.is_file() and metrics_path.stat().st_size > 0:
        remote_status_early = remote_bundle.get("status")
        stage_early = (
            str(remote_status_early.get("state") or "collecting")
            if isinstance(remote_status_early, Mapping)
            else "collecting"
        )
        early_hb = {
            "event": "learner_progress",
            "unix_s": monotonic_ts(time.time()),
            "environment_steps": cumulative_env_steps,
            "stage": stage_early,
        }
        with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(early_hb, ensure_ascii=False) + "\n")

    new_journals_built: list[int] = []
    # Reconstruct any available held-out journals first so evaluation events find complete journals on disk.
    for upd_num in sorted(cohort_replays):
        if upd_num in synced_journals:
            continue
        journal_path = run_dir / f"evaluation-step-{upd_num:09d}.jsonl"
        print(f"[sync] Reconstructing rich deck/route journal for update {upd_num} -> {journal_path.name} ...", flush=True)
        count = _reconstruct_heldout_journal(
            journal_path=journal_path,
            gate=upd_num,
            cohort_episodes=cohort_replays[upd_num],
            simulator=simulator,
            revival_budget=effective_revival_budget,
        )
        if count > 0:
            synced_journals.add(upd_num)
            new_journals_built.append(upd_num)

    new_updates_synced: list[int] = []
    for upd_num in sorted(updates_by_num):
        if upd_num in synced_updates:
            continue
        u = updates_by_num[upd_num]
        upd_mtime = _finite(u.get("_mtime"))
        base_ts = monotonic_ts(upd_mtime)
        lrn = u.get("learner") if isinstance(u.get("learner"), Mapping) else {}
        eps = [e for e in u.get("training_episodes", []) if isinstance(e, Mapping)]

        for ep_idx, ep in enumerate(eps):
            ep_steps = max(1, int(_finite(ep.get("native_steps")) or 600))
            cumulative_env_steps += ep_steps
            run_won = str(ep.get("run_result") or "") == "victory"
            max_floor = int(_finite(ep.get("max_floor")) or 1)
            max_act = 3 if max_floor >= 34 else (2 if max_floor >= 18 else 1)
            revivals = int(_finite(ep.get("revivals_used")) or 0)
            hp_lost = _finite(ep.get("player_hp_lost")) or 0.0
            reward_total = _finite(ep.get("total_reward")) or 0.0
            events_to_append.append(
                {
                    "event": "train_episode",
                    "unix_s": monotonic_ts(base_ts + 0.01 * ep_idx),
                    "environment_steps": cumulative_env_steps,
                    "episode_id": f"u{upd_num:04d}-ep{ep_idx:02d}",
                    "reset_seed": ep.get("seed"),
                    "max_act": max_act,
                    "max_floor": max_floor,
                    "revivals_used": revivals,
                    "player_hp_lost": hp_lost,
                    "steps": ep_steps,
                    "reward_total": reward_total,
                    "run_won": run_won,
                    "act1_cleared": max_floor >= 17 or run_won,
                    "terminal_reason": "run_victory" if run_won else str(ep.get("ending") or "defeat"),
                    "truncated": bool(ep.get("truncated", False)),
                    "deadlocked": False,
                }
            )

        v_dur = _finite(lrn.get("durable_value_loss")) or 0.0
        v_post = _finite(lrn.get("durable_post_value_loss")) or 0.0
        mpo_loss = _finite(lrn.get("macro_mpo_policy_loss")) or 0.0
        pg_loss = _finite(lrn.get("macro_pg_policy_loss")) or 0.0
        combat_loss = _finite(lrn.get("combat_policy_loss")) or 0.0
        distill_stage = str(lrn.get("distillation_stage") or "pure_rl")
        distill_mac = _finite(lrn.get("distill_macro_kl")) or 0.0
        distill_com = _finite(lrn.get("distill_combat_kl")) or 0.0
        distill_margin = _finite(lrn.get("distill_afterstate_margin_loss")) or 0.0
        ks_weight = _finite(lrn.get("effective_kickstart_kl_weight")) or 0.0
        eff_macro_actor = (
            distill_mac + distill_margin
            if distill_stage == "teacher_rollout_distill"
            else mpo_loss + pg_loss + ks_weight * distill_mac
        )
        eff_combat_actor = (
            distill_com
            if distill_stage == "teacher_rollout_distill"
            else combat_loss + ks_weight * distill_com
        )
        total_loss = v_dur + v_post + eff_macro_actor + eff_combat_actor
        grad_norm = _finite(lrn.get("gradient_norm"))

        events_to_append.append(
            {
                "event": "learner_update",
                "unix_s": monotonic_ts(base_ts + 0.2),
                "environment_steps": cumulative_env_steps,
                "policy_version": upd_num,
                "distillation_stage": distill_stage,
                "rollout_policy": u.get("rollout_policy"),
                "distill_macro_kl": distill_mac,
                "distill_combat_kl": distill_com,
                "distill_afterstate_margin_loss": distill_margin,
                "loss": total_loss,
                "online_objective_loss": total_loss,
                "value_loss": v_dur + v_post,
                "policy_loss": eff_macro_actor,
                "hybrid_macro_loss": v_dur + v_post + eff_macro_actor,
                "hybrid_combat_loss": eff_combat_actor,
                "hybrid_macro_value_loss": v_dur,
                "hybrid_macro_q_loss": v_post + distill_margin,
                "hybrid_macro_actor_loss": (
                    distill_mac
                    if distill_stage == "teacher_rollout_distill"
                    else (mpo_loss if mpo_loss != 0.0 else pg_loss)
                ),
                "hybrid_combat_actor_loss": eff_combat_actor,
                "gradient_norm": grad_norm,
            }
        )

        events_to_append.append(
            {
                "event": "checkpoint",
                "unix_s": monotonic_ts(base_ts + 0.3),
                "environment_steps": cumulative_env_steps,
                "policy_version": upd_num,
            }
        )

        n_eps = max(1, len(eps))
        wins = sum(1 for e in eps if str(e.get("run_result") or "") == "victory")
        act1_clears = sum(
            1
            for e in eps
            if int(_finite(e.get("max_floor")) or 0) >= 17 or str(e.get("run_result") or "") == "victory"
        )
        act3_reaches = sum(
            1
            for e in eps
            if int(_finite(e.get("max_floor")) or 0) >= 34 or str(e.get("run_result") or "") == "victory"
        )
        zero_rev_wins = sum(
            1
            for e in eps
            if str(e.get("run_result") or "") == "victory" and int(_finite(e.get("revivals_used")) or 0) == 0
        )
        floors = [int(_finite(e.get("max_floor")) or 1) for e in eps]
        revivs = [_finite(e.get("revivals_used")) or 0.0 for e in eps]
        hp_losses = [_finite(e.get("player_hp_lost")) or 0.0 for e in eps]

        eval_event: dict[str, Any] = {
            "event": "evaluation",
            "unix_s": monotonic_ts(base_ts + 0.4),
            "evaluation_gate": upd_num,
            "evaluation_attempt": 1,
            "gate_kind": "validation",
            "data_partition": "validation",
            "environment_steps": cumulative_env_steps,
            "policy_version": upd_num,
            "actor_policy_version": max(0, upd_num - 1),
            "episodes": len(eps),
            "act1_clear_count": act1_clears,
            "act1_clear_rate": act1_clears / n_eps,
            "act3_reach_count": act3_reaches,
            "act3_reach_rate": act3_reaches / n_eps,
            "run_win_rate": wins / n_eps,
            "successful_run_count": wins,
            "zero_revival_victories": zero_rev_wins,
            "zero_revival_act1_successes": zero_rev_wins,
            "revival_free_act1_clear_rate": zero_rev_wins / n_eps,
            "mean_revivals_used": sum(revivs) / n_eps,
            "mean_player_hp_lost": sum(hp_losses) / n_eps,
            "mean_max_floor": sum(floors) / n_eps if floors else 0.0,
            "maximum_floor": max(floors) if floors else 0,
            "deadlock_count": 0,
            "deadlock_rate": 0.0,
            "combat_progress_stall_rate": 0.0,
            "noncombat_progress_stall_rate": 0.0,
            "combat_policy_failure_count": 0,
            "evaluation_context": {
                "policy_version": upd_num,
                "actor_policy_version": max(0, upd_num - 1),
                "data_partition": "validation",
                "deterministic": False,
                "epsilon": 0.05,
            },
        }
        if upd_num in synced_journals:
            eval_event["journal_name"] = f"evaluation-step-{upd_num:09d}.jsonl"
        events_to_append.append(eval_event)

        synced_updates.add(upd_num)
        new_updates_synced.append(upd_num)

    # Emit live heartbeat when remote Colab G4 daemon is actively running so dashboard badge stays "训练中".
    terminal_emitted = bool(state_data.get("terminal_emitted", False))
    remote_status = remote_bundle.get("status")
    if bool(remote_bundle.get("alive")):
        terminal_emitted = False
        stage = (
            str(remote_status.get("state") or "collecting")
            if isinstance(remote_status, Mapping)
            else "collecting"
        )
        events_to_append.append(
            {
                "event": "learner_progress",
                "unix_s": monotonic_ts(time.time()),
                "environment_steps": cumulative_env_steps,
                "stage": stage,
            }
        )
    elif not terminal_emitted and isinstance(remote_status, Mapping):
        st_state = str(remote_status.get("state") or "")
        latest_ver = max(synced_updates, default=0)
        if st_state in {
            "completed",
            "campaign_complete",
            "archived_complete",
            "bounded_training_complete",
        }:
            events_to_append.append(
                {
                    "event": "run_complete",
                    "unix_s": monotonic_ts(time.time()),
                    "environment_steps": cumulative_env_steps,
                    "policy_version": latest_ver,
                    "completion_status": st_state,
                }
            )
            terminal_emitted = True
        elif st_state in {"aborted", "failed"}:
            err_type = str(remote_status.get("error_type") or "RuntimeError")
            err_msg = str(remote_status.get("error") or st_state)
            events_to_append.append(
                {
                    "event": "run_failed",
                    "unix_s": monotonic_ts(time.time()),
                    "environment_steps": cumulative_env_steps,
                    "policy_version": latest_ver,
                    "error_type": err_type,
                    "message": f"{err_type}: {err_msg}",
                }
            )
            terminal_emitted = True

    if events_to_append:
        with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            for ev in events_to_append:
                handle.write(json.dumps(ev, ensure_ascii=False) + "\n")

    new_state = {
        "synced_updates": sorted(synced_updates),
        "synced_journals": sorted(synced_journals),
        "cumulative_env_steps": cumulative_env_steps,
        "last_unix_s": last_unix_s,
        "terminal_emitted": terminal_emitted,
    }
    state_file.write_text(json.dumps(new_state, indent=2) + "\n", encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "alive": bool(remote_bundle.get("alive")),
        "new_updates_synced": new_updates_synced,
        "new_journals_built": new_journals_built,
        "total_updates": len(synced_updates),
        "total_journals": len(synced_journals),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync remote Colab G4 training into the local 8765 dashboard.")
    parser.add_argument("--session", default="sts2-macro-g4", help="Colab CLI session ID")
    parser.add_argument("--lineage", default="durable-afterstate-v2-b64-g4", help="local run lineage folder name")
    parser.add_argument("--run-id", default=None, help="optional explicit UUID for run-<uuid>")
    parser.add_argument("--artifact-root", type=Path, default=None, help="local runtime artifact root")
    parser.add_argument("--sim-exe", type=Path, default=None, help="pinned local HeadlessSim executable")
    parser.add_argument("--target-updates", type=int, default=200, help="planned total campaign updates")
    parser.add_argument("--revival-budget", type=int, default=64, help="fallback revival budget if course.json absent")
    parser.add_argument("--pid-path", default="/content/sts2_artifacts/campaign_pid.txt")
    parser.add_argument(
        "--remote-local-dir",
        default="/content/sts2_artifacts/campaigns/durable-afterstate-v2-b64-g4",
    )
    parser.add_argument("--seed-pull-file", type=Path, default=None, help="optional cached pull payload")
    parser.add_argument("--offline-only", action="store_true", help="ingest only from --seed-pull-file")
    parser.add_argument("--watch", action="store_true", help="poll Colab G4 continuously on a local timer")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between --watch syncs")
    args = parser.parse_args(argv)

    environ = dict(os.environ)
    if args.artifact_root is not None:
        environ["STS2_ARTIFACT_ROOT"] = str(args.artifact_root.resolve())
    root = artifact_root(environ=environ)
    simulator = verify_headless_simulator(resolve_headless_sim_exe(str(args.sim_exe) if args.sim_exe else None))

    lineage = args.lineage.strip().replace("_", "-")
    run_uuid = args.run_id or str(uuid5(NAMESPACE_DNS, f"sts2-colab-g4-{lineage}"))

    default_cache = root / "colab_g4_sync_cache" / "raw_pull.txt"
    seed_pull_file = args.seed_pull_file or (
        default_cache if lineage == "durable-afterstate-v2-b64-g4" and default_cache.is_file() else None
    )

    if not args.watch:
        result = sync_once(
            root=root,
            simulator=simulator,
            session=args.session,
            lineage=lineage,
            run_uuid=run_uuid,
            pid_path=args.pid_path,
            remote_local_dir=args.remote_local_dir,
            target_updates=args.target_updates,
            revival_budget=args.revival_budget,
            seed_pull_file=seed_pull_file,
            offline_only=args.offline_only,
        )
        print(f"[sync] {json.dumps(result, ensure_ascii=False)}", flush=True)
        return 0

    interval = max(15.0, float(args.interval))
    retry_interval = min(30.0, interval)
    print(f"[sync] Watching session={args.session} every {interval:.0f}s (Ctrl+C to stop)...", flush=True)
    first_pass = True
    try:
        while True:
            succeeded = False
            try:
                step_res = sync_once(
                    root=root,
                    simulator=simulator,
                    session=args.session,
                    lineage=lineage,
                    run_uuid=run_uuid,
                    pid_path=args.pid_path,
                    remote_local_dir=args.remote_local_dir,
                    target_updates=args.target_updates,
                    revival_budget=args.revival_budget,
                    seed_pull_file=seed_pull_file if first_pass else None,
                    offline_only=args.offline_only if first_pass else False,
                )
                first_pass = False
                succeeded = True
                print(f"[sync] {json.dumps(step_res, ensure_ascii=False)}", flush=True)
            except Exception as exc:
                print(f"[sync] transient poll warning (retrying in {retry_interval:.0f}s): {exc}", flush=True)
            time.sleep(interval if succeeded else retry_interval)
    except KeyboardInterrupt:
        print("\n[sync] Stopped.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
