"""Synchronize a remote Colab G4 Durable Afterstate campaign into the local 8765 dashboard.

This read-only local bridge queries the remote Colab G4 VM via short one-shot
``colab exec`` calls (never blocking the remote Jupyter kernel), streams
committed ``update.json`` + ``campaign_history.jsonl`` metrics into a canonical
local ``runs/<lineage>/run-<uuid>/metrics.jsonl``, and deterministically
reconstructs completed training-sample cohort episodes via the local verified
``HeadlessSim`` into ``train-cohort-step-XXXXXXXXX.jsonl`` journals (distinct
from odd-seed ``heldout_evaluations.jsonl`` validation gates).

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
import base64
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
wrapper_stage_f = local_dir / "campaign_wrapper_stage.json"
if wrapper_stage_f.is_file():
    try:
        ws = json.loads(wrapper_stage_f.read_text())
        if isinstance(ws, dict) and ws.get("stage"):
            status["state"] = str(ws["stage"])
    except Exception:
        pass
remote_revival_budget = None
remote_reward_objective = None
course_file = local_dir / "course.json"
if course_file.is_file():
    try:
        course_raw = json.loads(course_file.read_text())
        rb = (course_raw.get("reward_identity") or {}).get("revival_budget")
        if isinstance(rb, int) and not isinstance(rb, bool) and rb >= 0:
            remote_revival_budget = int(rb)
        ro = (
            ((course_raw.get("course_config") or {}).get("training") or {})
            .get("curriculum", {})
            .get("reward_objective")
        )
        if isinstance(ro, str) and ro in {"act1", "act2", "run"}:
            remote_reward_objective = ro
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

heldout_evaluations = []
heldout_path = local_dir / "heldout_evaluations.jsonl"
if heldout_path.is_file():
    for line in heldout_path.read_text().splitlines():
        if line.strip():
            try:
                heldout_evaluations.append(json.loads(line))
            except Exception:
                pass

extra_heldout_files = {}
for extra_p in sorted(local_dir.glob("heldout_evaluations_b*.jsonl")):
    rows_e = []
    for line in extra_p.read_text().splitlines():
        if line.strip():
            try:
                rows_e.append(json.loads(line))
            except Exception:
                pass
    extra_heldout_files[extra_p.name] = rows_e

print("PULLED_JSON_START")
print(json.dumps({
    "alive": alive,
    "pid": pid,
    "sim_workers": sim_workers,
    "status": status,
    "revival_budget": remote_revival_budget,
    "reward_objective": remote_reward_objective,
    "updates": updates,
    "history": history,
    "heldout_evaluations": heldout_evaluations,
    "extra_heldout_files": extra_heldout_files,
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
import contextlib, io, json, sys, time, uuid
from colab_cli.common import state
from colab_cli.runtime import ColabRuntime

real_stdout = sys.stdout
runtimes = {}

def close_cached(session):
    entry = runtimes.pop(session, None)
    if entry is not None:
        rt, _ = entry
        try:
            rt.stop()
        except Exception:
            pass

def get_runtime(session, force_refresh=False):
    if force_refresh:
        close_cached(session)
        state._sessions = None
        state._store = None
    s = state.get_session(session)
    cached = runtimes.get(session)
    if cached is not None:
        rt, tok = cached
        if tok == s.token:
            return rt
        close_cached(session)
    def _on_k(kid):
        s.kernel_id = kid
        state.store.add(s)
    rt = ColabRuntime(
        s.url,
        s.token,
        kernel_id=s.kernel_id,
        session_id=str(uuid.uuid4()),
        on_kernel_started=_on_k,
    )
    runtimes[session] = (rt, s.token)
    return rt

import re as _re

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    req = json.loads(line)
    session = str(req["session"])
    timeout_s = int(req["timeout_seconds"])
    script = str(req["script"])
    last_err = ""
    resp = None
    for attempt in range(3):
        cap_out = io.StringIO()
        cap_err = io.StringIO()
        try:
            with contextlib.redirect_stdout(cap_out), contextlib.redirect_stderr(cap_err):
                rt = get_runtime(session, force_refresh=(attempt > 0))
                outputs = rt.execute_code(script, timeout=timeout_s)
            out_parts = []
            err_parts = []
            has_err = False
            for out in outputs:
                if "text" in out:
                    out_parts.append(str(out["text"]))
                elif out.get("output_type") == "error":
                    has_err = True
                    ename = str(out.get("ename", "Error"))
                    evalue = str(out.get("evalue", ""))
                    tb = out.get("traceback") or []
                    err_parts.append("".join(tb) if tb else f"{ename}: {evalue}")
            resp = {
                "returncode": 1 if has_err else 0,
                "stdout": "".join(out_parts),
                "stderr": "\\n".join(err_parts),
            }
            break
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)) and not isinstance(exc, SystemExit):
                raise
            captured = _re.sub(r"https?://\\S+", "<REDACTED_URL>", (cap_out.getvalue() + " " + cap_err.getvalue()).strip())
            last_err = f"{type(exc).__name__}: {exc}" + (f" ({captured})" if captured else "")
            close_cached(session)
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    if resp is None:
        resp = {
            "returncode": -1,
            "stdout": "",
            "stderr": last_err,
        }
    real_stdout.write(json.dumps(resp, separators=(",", ":")) + "\\n")
    real_stdout.flush()
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
        b64_script = base64.b64encode(_WSL_BRIDGE_SERVER_SCRIPT.encode("utf-8")).decode("ascii")
        proc = subprocess.Popen(
            [
                "wsl",
                "-e",
                "bash",
                "-lc",
                (
                    "colab_bin=${COLAB_BIN:-$HOME/.local/bin/colab}; "
                    "py_bin=$(head -n 1 \"$colab_bin\" | cut -c 3-); "
                    f"exec \"$py_bin\" -u -c 'import base64; exec(base64.b64decode(\"{b64_script}\").decode(\"utf-8\"))'"
                ),
            ],
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


_REMOTE_RATCHET_STATE_TEMPLATE = """
import json
from pathlib import Path

p = Path(__RATCHET_STATE_PATH__)
data = json.loads(p.read_text()) if p.is_file() else None
print("RATCHET_JSON_START")
print(json.dumps({"ratchet": data}, separators=(",", ":")))
print("RATCHET_JSON_END")
"""


def _fetch_ratchet_state(
    *,
    session: str,
    ratchet_state_path: str,
    timeout_seconds: int = 30,
) -> dict[str, Any] | None:
    script = _REMOTE_RATCHET_STATE_TEMPLATE.replace("__RATCHET_STATE_PATH__", json.dumps(ratchet_state_path))
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
        raise RuntimeError(f"colab ratchet state pull failed (exit {returncode}): {detail}")
    wrapped = _extract_marked_json(stdout, "RATCHET_JSON_START", "RATCHET_JSON_END")
    inner = wrapped.get("ratchet")
    return dict(inner) if isinstance(inner, Mapping) else None


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
    gate_kind: str = "train_cohort",
    data_partition: str = "train_cohort",
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
                    "gate_kind": gate_kind,
                    "data_partition": data_partition,
                    "policy_version": gate,
                    "actor_policy_version": max(0, gate - 1),
                    "evaluation_seeds": seeds,
                    "epsilon": 0.05 if data_partition == "train_cohort" else 0.0,
                    "deterministic": data_partition != "train_cohort",
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
                prefix_kind = "train" if data_partition == "train_cohort" else "heldout"
                ep_prefix = f"{prefix_kind}-seed-{seed}-attempt-1:"

                journal.write_episode_boundary(
                    {
                        "event": "evaluation_attempt_started",
                        "evaluation_seed": seed,
                        "attempt": 1,
                        "journal_episode_namespace": ep_prefix,
                        "data_partition": data_partition,
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


_RESUMABLE_COMPLETION_STATUSES = frozenset({"paused_stage_cap_reached"})


def _strip_stale_terminal_events(metrics_path: Path) -> int:
    """Remove stale resumable terminal records from metrics.jsonl when a stage resumes live."""
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        return 0
    raw_bytes = metrics_path.read_bytes()
    if b'"run_complete"' not in raw_bytes and b'"run_failed"' not in raw_bytes:
        return 0
    kept: list[str] = []
    removed = 0
    for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped:
            try:
                obj = json.loads(stripped)
            except Exception:
                obj = None
            if isinstance(obj, Mapping):
                ev = obj.get("event")
                if ev == "run_failed" or (
                    ev == "run_complete"
                    and str(obj.get("completion_status") or "") in _RESUMABLE_COMPLETION_STATUSES
                ):
                    removed += 1
                    continue
        kept.append(line)
    if removed == 0:
        return 0
    tmp_path = metrics_path.with_name(f".{metrics_path.name}.tmp-{os.getpid()}")
    payload = ("\n".join(kept) + "\n") if kept else ""
    tmp_path.write_text(payload, encoding="utf-8", newline="\n")
    for attempt in range(3):
        try:
            os.replace(tmp_path, metrics_path)
            return removed
        except PermissionError:
            if attempt == 2:
                tmp_path.unlink(missing_ok=True)
                return 0
            time.sleep(0.05)
    return 0


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
    force_terminal_status: str | None = None,
    ratchet_state_label: str | None = None,
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
        "synced_heldout_gates": [],
        "update_env_steps": {},
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
    synced_heldout_gates: set[int] = {int(x) for x in state_data.get("synced_heldout_gates", [])}
    raw_update_steps = state_data.get("update_env_steps")
    update_env_steps: dict[int, int] = (
        {int(k): int(v) for k, v in raw_update_steps.items() if str(k).isdigit() and isinstance(v, int | float)}
        if isinstance(raw_update_steps, Mapping)
        else {}
    )
    # Also scan existing train-cohort-step-*.jsonl files in run_dir
    for existing_j in run_dir.glob("train-cohort-step-*.jsonl"):
        stem_digits = existing_j.stem.removeprefix("train-cohort-step-")
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
    if isinstance(rb_remote, int) and not isinstance(rb_remote, bool) and rb_remote >= 0:
        effective_revival_budget = int(rb_remote)
    ro_remote = remote_bundle.get("reward_objective")
    effective_reward_objective = (
        str(ro_remote)
        if isinstance(ro_remote, str) and ro_remote in {"act1", "act2", "run"}
        else ("act1" if "act1" in lineage else ("act2" if "act2" in lineage else "run"))
    )

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
                        "durable_afterstate_v7"
                        if "v7" in lineage
                        else "durable_afterstate_v6"
                        if "v6" in lineage
                        else (
                            "durable_afterstate_v4"
                            if "v4" in lineage
                            else ("durable_afterstate_v3" if "v3" in lineage else "durable_afterstate_v2")
                        )
                    ),
                    "version": (
                        "sts2-durable-afterstate-v7.5-active-vitals-b0"
                        if ("v7-5" in lineage or "v7.5" in lineage)
                        else "sts2-durable-afterstate-v7.4-joint-b0"
                        if ("v7-4" in lineage or "v7.4" in lineage)
                        else "sts2-durable-afterstate-v7.3-combat-vitals-b0"
                        if ("v7-3" in lineage or "v7.3" in lineage)
                        else "sts2-durable-afterstate-v7.2-turbo-b0"
                        if ("v7-2" in lineage or "v7.2" in lineage)
                        else "sts2-durable-afterstate-v7.1-combat-first-b0"
                        if ("v7-1" in lineage or "v7.1" in lineage)
                        else "sts2-durable-afterstate-v7.0-act-b0"
                        if ("v7-0" in lineage or "v7.0" in lineage or "v7" in lineage)
                        else "sts2-durable-afterstate-v6.6-ratchet"
                        if ("v6-6" in lineage or "v6.6" in lineage)
                        else "sts2-durable-afterstate-v6.4-ratchet"
                        if ("v6-4" in lineage or "v6.4" in lineage)
                        else "sts2-durable-afterstate-v6.3-ratchet"
                        if ("v6-3" in lineage or "v6.3" in lineage or "ratchet" in lineage)
                        else "sts2-durable-afterstate-v6.2-ablation"
                        if ("v6-2" in lineage or "v6.2" in lineage)
                        else "sts2-durable-afterstate-v6-wide256"
                        if "v6" in lineage
                        else "sts2-durable-afterstate-v5-wide256"
                        if "v5" in lineage
                        else "sts2-durable-afterstate-v4"
                        if "v4" in lineage
                        else "sts2-durable-afterstate-v3"
                        if "v3" in lineage
                        else "sts2-durable-afterstate-v2"
                    ),
                    "curriculum": {
                        "mode": "native-revival-preheat",
                        "reward_objective": effective_reward_objective,
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

    stage_alive = bool(remote_bundle.get("alive")) and force_terminal_status is None
    # If this stage was previously paused at cap (or failed) and has now resumed live, strip stale
    # terminal lines so the dashboard immediately transitions back to active training status.
    if stage_alive and metrics_path.is_file() and metrics_path.stat().st_size > 0:
        removed_stale = _strip_stale_terminal_events(metrics_path)
        if removed_stale > 0:
            print(
                f"[sync] Cleared {removed_stale} stale terminal event(s) for resumed live stage {lineage}",
                flush=True,
            )
    # Flush immediate live heartbeat BEFORE running local HeadlessSim journal replay so telemetry age resets right away.
    if stage_alive and metrics_path.is_file() and metrics_path.stat().st_size > 0:
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
    # Reconstruct any available training-sample replay journals first (explicitly tagged as train_cohort).
    for upd_num in sorted(cohort_replays):
        if upd_num in synced_journals:
            continue
        journal_path = run_dir / f"train-cohort-step-{upd_num:09d}.jsonl"
        print(
            f"[sync] Reconstructing rich deck/route training-sample journal for update {upd_num} -> {journal_path.name} ...",
            flush=True,
        )
        count = _reconstruct_heldout_journal(
            journal_path=journal_path,
            gate=upd_num,
            cohort_episodes=cohort_replays[upd_num],
            simulator=simulator,
            revival_budget=effective_revival_budget,
            gate_kind="train_cohort",
            data_partition="train_cohort",
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
            reward_val = _finite(ep.get("observed_discounted_return"))
            if reward_val is None:
                reward_val = _finite(ep.get("total_reward"))
            reward_total = reward_val if reward_val is not None else 0.0
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
        update_env_steps[upd_num] = cumulative_env_steps

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

        cohort_event: dict[str, Any] = {
            "event": "train_cohort_summary",
            "unix_s": monotonic_ts(base_ts + 0.4),
            "evaluation_gate": upd_num,
            "evaluation_attempt": 1,
            "gate_kind": "train_cohort",
            "data_partition": "train_cohort",
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
                "data_partition": "train_cohort",
                "deterministic": False,
                "epsilon": 0.05,
            },
        }
        if upd_num in synced_journals:
            cohort_event["journal_name"] = f"train-cohort-step-{upd_num:09d}.jsonl"
        events_to_append.append(cohort_event)

        synced_updates.add(upd_num)
        new_updates_synced.append(upd_num)

    # Emit genuine odd-seed held-out evaluation records ONLY from heldout_evaluations.jsonl.
    heldout_by_gate: dict[int, dict[str, Any]] = {}
    for src_h in (seed_bundle.get("heldout_evaluations", []), remote_bundle.get("heldout_evaluations", [])):
        if not isinstance(src_h, list):
            continue
        for h_row in src_h:
            if not isinstance(h_row, Mapping):
                continue
            if str(h_row.get("data_partition") or "validation") != "validation":
                continue
            rec_budget = _finite(h_row.get("revival_budget"))
            if rec_budget is not None and int(rec_budget) != effective_revival_budget:
                continue
            pv = _finite(h_row.get("policy_version", h_row.get("evaluation_gate")))
            if pv is None:
                continue
            heldout_by_gate[int(pv)] = dict(h_row)

    new_heldout_synced: list[int] = []
    for h_gate in sorted(heldout_by_gate):
        hr = heldout_by_gate[h_gate]
        raw_rep_eps = hr.get("replay_episodes")
        eval_journal_path = run_dir / f"evaluation-step-{h_gate:09d}.jsonl"
        if isinstance(raw_rep_eps, list) and raw_rep_eps and not eval_journal_path.is_file():
            rep_eps = [row for row in raw_rep_eps if isinstance(row, Mapping)]
            if rep_eps:
                print(
                    f"[sync] Reconstructing odd-seed validation journal for gate {h_gate} -> {eval_journal_path.name} ...",
                    flush=True,
                )
                _reconstruct_heldout_journal(
                    journal_path=eval_journal_path,
                    gate=h_gate,
                    cohort_episodes=rep_eps,
                    simulator=simulator,
                    revival_budget=effective_revival_budget,
                    gate_kind="validation",
                    data_partition="validation",
                )
        if h_gate in synced_heldout_gates:
            continue
        h_eps = max(1, int(_finite(hr.get("episodes")) or 48))
        h_a1_rate = _finite(hr.get("act1_clear_rate")) or 0.0
        h_a3_rate = _finite(hr.get("act3_reach_rate")) or 0.0
        h_win_rate = _finite(hr.get("run_win_rate")) or 0.0
        h_z_wins = int(_finite(hr.get("zero_revival_victories")) or 0)
        h_env_steps = 0 if h_gate == 0 else update_env_steps.get(h_gate, cumulative_env_steps)
        events_to_append.append(
            {
                "event": "evaluation",
                "unix_s": monotonic_ts(_finite(hr.get("unix_s"))),
                "evaluation_gate": h_gate,
                "evaluation_attempt": max(1, int(_finite(hr.get("evaluation_attempt")) or 1)),
                "gate_kind": "validation",
                "data_partition": "validation",
                "environment_steps": h_env_steps,
                "policy_version": h_gate,
                "actor_policy_version": h_gate,
                "episodes": h_eps,
                "act1_clear_count": int(_finite(hr.get("act1_clear_count")) or round(h_a1_rate * h_eps)),
                "act1_clear_rate": h_a1_rate,
                "act3_reach_count": int(_finite(hr.get("act3_reach_count")) or round(h_a3_rate * h_eps)),
                "act3_reach_rate": h_a3_rate,
                "run_win_rate": h_win_rate,
                "successful_run_count": int(_finite(hr.get("successful_run_count")) or round(h_win_rate * h_eps)),
                "zero_revival_victories": h_z_wins,
                "zero_revival_act1_successes": int(
                    _finite(hr.get("zero_revival_act1_successes")) or h_z_wins
                ),
                "revival_free_act1_clear_rate": (
                    _finite(hr.get("revival_free_act1_clear_rate"))
                    if _finite(hr.get("revival_free_act1_clear_rate")) is not None
                    else (h_z_wins / h_eps)
                ),
                "mean_revivals_used": _finite(hr.get("mean_revivals_used")),
                "mean_player_hp_lost": _finite(hr.get("mean_player_hp_lost")),
                "mean_max_floor": _finite(hr.get("mean_max_floor")),
                "maximum_floor": int(_finite(hr.get("maximum_floor")) or 50),
                "deadlock_count": int(_finite(hr.get("deadlock_count")) or 0),
                "deadlock_rate": _finite(hr.get("deadlock_rate")) or 0.0,
                "combat_progress_stall_rate": _finite(hr.get("combat_progress_stall_rate")) or 0.0,
                "noncombat_progress_stall_rate": _finite(hr.get("noncombat_progress_stall_rate")) or 0.0,
                "combat_policy_failure_count": int(_finite(hr.get("combat_policy_failure_count")) or 0),
                "evaluation_context": {
                    "policy_version": h_gate,
                    "actor_policy_version": h_gate,
                    "data_partition": "validation",
                    "seed_parity": str(hr.get("seed_parity") or "odd_held_out"),
                    "deterministic": True,
                    "epsilon": 0.0,
                },
            }
        )
        synced_heldout_gates.add(h_gate)
        new_heldout_synced.append(h_gate)

    extra_files_map = remote_bundle.get("extra_heldout_files")
    if isinstance(extra_files_map, Mapping):
        for ef_name, ef_rows in extra_files_map.items():
            if (
                isinstance(ef_name, str)
                and ef_name.startswith("heldout_evaluations_b")
                and ef_name.endswith(".jsonl")
                and "/" not in ef_name
                and "\\" not in ef_name
                and isinstance(ef_rows, list)
                and ef_rows
            ):
                ef_dst = run_dir / ef_name
                ef_text = "".join(
                    json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                    for row in ef_rows
                    if isinstance(row, Mapping)
                )
                ef_dst.write_text(ef_text, encoding="utf-8")

    # Emit live heartbeat when remote Colab G4 daemon is actively running so dashboard badge stays "训练中".
    terminal_emitted = bool(state_data.get("terminal_emitted", False))
    remote_status = remote_bundle.get("status")
    if stage_alive:
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
    elif not terminal_emitted:
        latest_ver = max(synced_updates, default=0)
        st_state = str(remote_status.get("state") or "") if isinstance(remote_status, Mapping) else ""
        if force_terminal_status is not None:
            events_to_append.append(
                {
                    "event": "run_complete",
                    "unix_s": monotonic_ts(time.time()),
                    "environment_steps": cumulative_env_steps,
                    "policy_version": latest_ver,
                    "completion_status": force_terminal_status,
                }
            )
            terminal_emitted = True
        elif ratchet_state_label in {"ratchet_complete", "stage_promoted", "paused_stage_cap_reached"}:
            events_to_append.append(
                {
                    "event": "run_complete",
                    "unix_s": monotonic_ts(time.time()),
                    "environment_steps": cumulative_env_steps,
                    "policy_version": latest_ver,
                    "completion_status": ratchet_state_label,
                }
            )
            terminal_emitted = True
        elif ratchet_state_label == "aborted":
            events_to_append.append(
                {
                    "event": "run_failed",
                    "unix_s": monotonic_ts(time.time()),
                    "environment_steps": cumulative_env_steps,
                    "policy_version": latest_ver,
                    "error_type": "RatchetAborted",
                    "message": "Ratchet orchestrator reported aborted state",
                }
            )
            terminal_emitted = True
        elif st_state in {
            "completed",
            "campaign_complete",
            "archived_complete",
        } or (st_state == "bounded_training_complete" and latest_ver >= target_updates):
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
        elif st_state in {"aborted", "failed"} and isinstance(remote_status, Mapping):
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
        "synced_heldout_gates": sorted(synced_heldout_gates),
        "update_env_steps": {str(k): v for k, v in sorted(update_env_steps.items())},
        "cumulative_env_steps": cumulative_env_steps,
        "last_unix_s": last_unix_s,
        "terminal_emitted": terminal_emitted,
    }
    state_file.write_text(json.dumps(new_state, indent=2) + "\n", encoding="utf-8")

    return {
        "lineage": lineage,
        "revival_budget": effective_revival_budget,
        "run_dir": str(run_dir),
        "alive": stage_alive,
        "new_updates_synced": new_updates_synced,
        "new_heldout_synced": new_heldout_synced,
        "new_journals_built": new_journals_built,
        "total_updates": len(synced_updates),
        "total_heldout_gates": len(synced_heldout_gates),
        "total_journals": len(synced_journals),
    }


def _is_local_stage_terminal(root: Path, stage_lineage: str) -> bool:
    st_file = root / "colab_g4_sync" / stage_lineage / "sync_state.json"
    if not st_file.is_file():
        return False
    try:
        data = json.loads(st_file.read_text(encoding="utf-8"))
        return isinstance(data, Mapping) and bool(data.get("terminal_emitted", False))
    except Exception:
        return False


def _sync_ratchet_tick(
    *,
    root: Path,
    simulator: VerifiedSimulatorIdentity,
    session: str,
    pid_path: str,
    ratchet_state_path: str,
) -> dict[str, Any]:
    ratchet_info = _fetch_ratchet_state(session=session, ratchet_state_path=ratchet_state_path)
    if ratchet_info is None:
        return {"status": "waiting_for_ratchet_state", "ratchet_state_path": ratchet_state_path}

    max_stage_updates = int(ratchet_info.get("max_updates_per_stage") or 80)
    cur_lineage = str(ratchet_info["current_lineage"]).strip().replace("_", "-")
    cur_local_dir = str(ratchet_info["current_local_dir"])
    raw_cur_budget = ratchet_info.get("current_budget")
    cur_budget = (
        int(raw_cur_budget)
        if isinstance(raw_cur_budget, int) and not isinstance(raw_cur_budget, bool)
        else 24
    )
    cur_state = str(ratchet_info.get("state") or "")
    campaigns_parent = str(Path(cur_local_dir).parent).replace("\\", "/")

    # Ensure every already-promoted stage in promoted_history has been finalized locally.
    prom_hist = ratchet_info.get("promoted_history")
    if isinstance(prom_hist, list):
        for entry in prom_hist:
            if not isinstance(entry, Mapping):
                continue
            p_lin = str(entry.get("lineage_id") or "").strip().replace("_", "-")
            raw_p_bud = entry.get("budget")
            p_bud = (
                int(raw_p_bud)
                if isinstance(raw_p_bud, int) and not isinstance(raw_p_bud, bool)
                else 24
            )
            p_upd = int(entry.get("promoted_at_update") or max_stage_updates)
            if not p_lin or p_lin == cur_lineage or _is_local_stage_terminal(root, p_lin):
                continue
            p_uuid = str(uuid5(NAMESPACE_DNS, f"sts2-colab-g4-{p_lin}"))
            p_remote_dir = f"{campaigns_parent}/{p_lin}"
            print(f"[sync] Finalizing promoted stage {p_lin} (u={p_upd}) -> run_complete ...", flush=True)
            sync_once(
                root=root,
                simulator=simulator,
                session=session,
                lineage=p_lin,
                run_uuid=p_uuid,
                pid_path=pid_path,
                remote_local_dir=p_remote_dir,
                target_updates=max(p_upd, 1),
                revival_budget=p_bud,
                force_terminal_status=f"stage_promoted_at_u{p_upd}",
            )

    cur_uuid = str(uuid5(NAMESPACE_DNS, f"sts2-colab-g4-{cur_lineage}"))
    force_term = "ratchet_complete" if cur_state == "ratchet_complete" else None
    res = sync_once(
        root=root,
        simulator=simulator,
        session=session,
        lineage=cur_lineage,
        run_uuid=cur_uuid,
        pid_path=pid_path,
        remote_local_dir=cur_local_dir,
        target_updates=max_stage_updates,
        revival_budget=cur_budget,
        force_terminal_status=force_term,
        ratchet_state_label=cur_state,
    )
    res["ratchet_state"] = cur_state
    res["stage_index"] = ratchet_info.get("stage_index")
    res["total_stages"] = ratchet_info.get("total_stages")
    return res


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
    parser.add_argument(
        "--follow-ratchet-state",
        default=None,
        help="remote path to active_ratchet_state.json for automatic multi-stage curriculum following",
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
        if args.follow_ratchet_state:
            result = _sync_ratchet_tick(
                root=root,
                simulator=simulator,
                session=args.session,
                pid_path=args.pid_path,
                ratchet_state_path=str(args.follow_ratchet_state),
            )
        else:
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
                if args.follow_ratchet_state:
                    step_res = _sync_ratchet_tick(
                        root=root,
                        simulator=simulator,
                        session=args.session,
                        pid_path=args.pid_path,
                        ratchet_state_path=str(args.follow_ratchet_state),
                    )
                else:
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
