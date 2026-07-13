"""Synchronous JSON-lines transport for the HeadlessSim process.

This client owns subprocess lifecycle, RPC framing, action-handle dispatch,
and structural DTO translation only.  Rewards and learning semantics are
owned by the typed ``sts2_rl`` boundary.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

# Use the shared transport exception so typed live/headless callers have one
# failure boundary without importing simulator-specific exception classes.
from sts2_env.bridge_client import BridgeError
from sts2_rl.artifacts import artifact_root, resolve_artifact_path, resolve_external_input_path
from sts2_rl.contracts.versions import API_VERSION, SCHEMA_VERSION


def _headless_sim_candidates() -> tuple[Path, ...]:
    """Return portable candidate paths without assuming a developer drive."""

    roots: list[Path] = []
    configured_root = os.environ.get("STS2_AI_ROOT")
    if configured_root:
        source_root = resolve_external_input_path(configured_root)
    else:
        source_root = artifact_root() / "dependencies" / "sts2-ai"
    roots.append(source_root / "STS2AI" / "ENV" / "Sim")

    relative_candidates = (
        Path("HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"),
        Path("HeadlessSim/bin/Debug/net9.0/HeadlessSim.exe"),
        Path("Runtime/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"),
        Path("Runtime/HeadlessSim/bin/Debug/net9.0/HeadlessSim.exe"),
    )
    return tuple(root / candidate for root in roots for candidate in relative_candidates)


def resolve_headless_sim_exe(exe_path: str | Path | None = None) -> Path:
    """Resolve the simulator executable from explicit/env/repository sources."""

    explicit = exe_path or os.environ.get("STS2_HEADLESS_SIM_EXE")
    if explicit:
        candidate = resolve_external_input_path(explicit)
        if candidate.is_file():
            return candidate
        raise HeadlessSimError(f"HeadlessSim exe not found: {candidate}")

    candidates = _headless_sim_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  - ".join(str(candidate) for candidate in candidates)
    raise HeadlessSimError(
        "HeadlessSim exe not found. Set STS2_HEADLESS_SIM_EXE or STS2_AI_ROOT. "
        f"Searched:\n  - {searched}"
    )


class HeadlessSimError(BridgeError):
    """Raised when the HeadlessSim subprocess returns an error, dies, or
    fails to respond to an RPC within the configured timeout.
    """


class HeadlessSimBridgeClient:
    """Drop-in replacement for BridgeClient that drives HeadlessSim.

    Not a subclass of BridgeClient because BridgeClient's __init__ assumes a
    session.json file — instead we expose the same public methods. Python is
    structurally typed so CombatSandboxEnv/SlayTheSpire2EnvV2 don't care.
    """

    # Compatibility attribute; resolution also checks release/runtime layouts
    # and environment overrides. This path is repository-relative, never tied
    # to a developer drive.
    DEFAULT_EXE_PATH = _headless_sim_candidates()[0]

    def __init__(
        self,
        *,
        exe_path: str | Path | None = None,
        protocol: str = "json",
        startup_timeout_s: float = 30.0,
        request_timeout_s: float = 30.0,
        # RPC timeout: how long we wait for a sim response line before
        # declaring the sim hung. Empirically sim steps take 3-22ms, so
        # 25s is ~1000x normal — safely above noise but well under
        # AsyncReadyCollector's 45s step-watchdog so we kill the sim and
        # raise a clean BridgeError before the collector hard-restarts
        # the worker (which otherwise leaves the sim subprocess orphaned).
        rpc_timeout_s: float = 25.0,
    ):
        self._exe_path = resolve_headless_sim_exe(exe_path)
        if protocol != "json":
            raise NotImplementedError("Only json protocol supported for now")

        self._request_timeout_s = float(request_timeout_s)
        self._rpc_timeout_s = float(rpc_timeout_s)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._episode_counter = 0
        self._current_episode_id: str = ""
        # Cache the bridge-shaped legal_actions from the most recent step/reset
        # response so step() can resolve an action_index back to the sim's
        # raw action dict when the env dispatches.
        self._last_legal_actions: list[dict[str, Any]] = []
        # Dedicated reader thread drains stdout into this queue so the main
        # thread can wait with a timeout. Without this, a hung sim leaves
        # our main thread blocked in readline() indefinitely.
        self._stdout_queue: queue.Queue[str | None] = queue.Queue(maxsize=8)
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._startup_ready = threading.Event()
        self._startup_stderr_tail: deque[str] = deque(maxlen=40)
        # Per-process hang-debug log. Each RPC starts by appending a line
        # here; on hang, the tail of this file identifies the exact action
        # that made the sim stop responding. File name includes the sim
        # PID once it's up so 4 parallel sims don't clobber each other.
        self._hang_log_path: Path | None = None
        self._hang_log_handle = None
        self._start_subprocess(startup_timeout_s)

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------
    def _start_subprocess(self, startup_timeout_s: float) -> None:
        env = os.environ.copy()
        env.setdefault("STS2_MIN_THREADS", "8")
        self._proc = subprocess.Popen(
            [str(self._exe_path), "--stdio", "--protocol", "json"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(self._exe_path.parent),
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._reader_stop.clear()
        self._startup_ready.clear()
        self._reader_thread = threading.Thread(
            target=self._stdout_reader_loop,
            name=f"sim-stdout-reader-pid{self._proc.pid}",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_reader_loop,
            name=f"sim-stderr-reader-pid{self._proc.pid}",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()

        # Never call stderr.readline() on the constructing thread: an otherwise
        # healthy but silent process would bypass startup_timeout_s forever.
        deadline = time.monotonic() + max(float(startup_timeout_s), 0.001)
        while time.monotonic() < deadline:
            if self._startup_ready.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic()))):
                self._open_hang_log()
                return
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return_code = None if proc is None else proc.returncode
                tail = " | ".join(self._startup_stderr_tail)
                self._kill_sim("process exited during startup")
                raise HeadlessSimError(
                    f"HeadlessSim exited during startup (rc={return_code}); stderr_tail={tail!r}"
                )

        tail = " | ".join(self._startup_stderr_tail)
        self._kill_sim(f"startup timeout after {startup_timeout_s}s")
        raise HeadlessSimError(
            f"HeadlessSim failed to become ready in {startup_timeout_s:.1f}s; stderr_tail={tail!r}"
        )

    def _open_hang_log(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            directory = resolve_artifact_path(
                os.environ.get("STS2_SIM_HANG_LOG_DIR") or None,
                default="logs/headless-sim",
            )
            directory.mkdir(parents=True, exist_ok=True)
            self._hang_log_path = directory / f"sim_hang_debug_pid{proc.pid}.log"
            self._hang_log_handle = self._hang_log_path.open("w", encoding="utf-8")
            self._hang_log_handle.write(
                f"# sim hang-debug log for pid={proc.pid} started at {time.time()}\n"
            )
            for line in self._startup_stderr_tail:
                self._hang_log("sim_startup_stderr", line=line)
            self._hang_log_handle.flush()
        except (OSError, ValueError):
            self._hang_log_path = None
            self._hang_log_handle = None

    def _stderr_reader_loop(self) -> None:
        """Drain stderr continuously and signal the startup banner."""

        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while not self._reader_stop.is_set():
                line = proc.stderr.readline()
                if not line:
                    return
                snippet = line.rstrip("\r\n")
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                self._startup_stderr_tail.append(snippet)
                if "stdio mode ready" in snippet:
                    self._startup_ready.set()
                self._hang_log("sim_stderr", line=snippet)
        except (OSError, ValueError) as exc:
            self._hang_log("stderr_reader_exception", error=repr(exc))

    def _enqueue_stdout(self, line: str | None) -> bool:
        """Bound stdout buffering while preserving response ordering."""

        while not self._reader_stop.is_set():
            try:
                self._stdout_queue.put(line, timeout=0.1)
                return True
            except queue.Full:
                self._hang_log("reader_queue_backpressure", maxsize=self._stdout_queue.maxsize)
        return False

    def _stdout_reader_loop(self) -> None:
        """Dedicated thread. Reads stdout lines one-at-a-time and feeds the
        queue. Exits when the pipe closes or stop is requested. Without this
        indirection, ``_rpc``'s ``readline()`` would block indefinitely when
        the sim hangs, stalling the worker and leaking the sim subprocess.
        """
        proc = self._proc
        if proc is None:
            return
        stdout = proc.stdout
        try:
            while not self._reader_stop.is_set():
                # Phase A instrumentation: record before/after each readline
                # so we can tell from the hang log whether reader thread is
                # healthy or stuck inside readline() itself.
                self._hang_log("reader_readline_begin")
                line = stdout.readline()
                if not line:
                    # EOF — sim died or closed its pipes.
                    self._hang_log("reader_eof")
                    self._enqueue_stdout(None)
                    break
                # text mode — len(line) is character count
                self._hang_log("reader_readline_done", nchars=len(line))
                if not self._enqueue_stdout(line):
                    return
        except Exception as exc:
            # Push a sentinel so waiters don't block forever. We swallow
            # the exception because the worker thread picking up the
            # sentinel will raise a clean HeadlessSimError.
            self._hang_log("reader_exception", error=repr(exc))
            self._enqueue_stdout(None)

    def _hang_log(self, event: str, **fields: Any) -> None:
        """Append a compact JSON line to the hang-debug log. Silent on
        errors — diagnostics must never break the training loop.
        """
        if self._hang_log_handle is None:
            return
        entry = {"t": time.time(), "event": event, **fields}
        try:
            self._hang_log_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._hang_log_handle.flush()
        except Exception:
            pass

    def _join_reader_threads(self) -> None:
        current = threading.current_thread()
        for thread in (self._reader_thread, self._stderr_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=1.0)
        self._reader_thread = None
        self._stderr_thread = None

    @staticmethod
    def _close_process_pipes(proc: subprocess.Popen[Any]) -> None:
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    def _drain_stdout_queue(self) -> None:
        while True:
            try:
                self._stdout_queue.get_nowait()
            except queue.Empty:
                return

    def _kill_sim(self, reason: str) -> None:
        """Force-terminate and reap the subprocess. Idempotent."""

        proc = self._proc
        self._proc = None
        self._reader_stop.set()
        if proc is not None:
            self._hang_log("process_kill", reason=reason, pid=proc.pid)
            try:
                if proc.poll() is None:
                    proc.kill()
            except (OSError, ProcessLookupError):
                pass
            try:
                proc.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, OSError):
                pass
            self._close_process_pipes(proc)
        self._join_reader_threads()
        self._drain_stdout_queue()

    def close(self) -> None:
        """Close stdin, then force-kill if needed; always reap and join readers."""

        with self._lock:
            proc = self._proc
            if proc is not None and proc.stdin is not None:
                try:
                    proc.stdin.close()
                except (OSError, ValueError):
                    pass
            if proc is not None:
                try:
                    proc.wait(timeout=1.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            self._kill_sim("client close")
        handle = self._hang_log_handle
        self._hang_log_handle = None
        if handle is not None:
            try:
                handle.close()
            except (OSError, ValueError):
                pass

    def __enter__(self) -> HeadlessSimBridgeClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Low-level RPC
    # ------------------------------------------------------------------
    def _rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        request = {"method": method}
        if params is not None:
            request["params"] = params
        # text mode — write str, not bytes
        payload = json.dumps(request, ensure_ascii=False) + "\n"
        sent_at = time.time()
        self._hang_log("sent", method=method, params=params, sent_at=sent_at)
        rpc_timeout_s = self._rpc_timeout_s if timeout_s is None else max(float(timeout_s), 0.001)
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise HeadlessSimError(
                    f"HeadlessSim subprocess is not running (method={method})"
                )
            # Phase A checkpoint A: about to write request payload to sim stdin.
            self._hang_log("rpc_stdin_write_begin", method=method, nbytes=len(payload))
            try:
                proc.stdin.write(payload)
                proc.stdin.flush()
            except Exception as exc:
                self._kill_sim(f"stdin write failed: {exc}")
                self._hang_log("stdin_write_error", method=method, error=repr(exc))
                raise HeadlessSimError(
                    f"HeadlessSim stdin write failed for {method}: {exc}"
                ) from exc
            # Phase A checkpoint B: stdin write+flush returned. If sim never
            # observes this byte stream, the hang is at OS pipe level.
            self._hang_log("rpc_stdin_write_done", method=method)
            # Phase A checkpoint C: about to block waiting for response line.
            self._hang_log("rpc_stdout_await_begin", method=method)
            try:
                line = self._stdout_queue.get(timeout=rpc_timeout_s)
            except queue.Empty:
                # This is THE hang-diagnosis signal. The hang log tail has
                # the exact request that went in without a response.
                waited = time.time() - sent_at
                # Report whether the reader thread is still alive — if it
                # died, that's where the hang is (not in sim).
                reader_alive = (
                    self._reader_thread is not None
                    and self._reader_thread.is_alive()
                )
                self._hang_log(
                    "rpc_timeout",
                    method=method,
                    params=params,
                    waited_s=waited,
                    rpc_timeout_s=rpc_timeout_s,
                    reader_thread_alive=reader_alive,
                    proc_poll=self._proc.poll() if self._proc else None,
                )
                self._kill_sim(f"rpc timeout {rpc_timeout_s}s on {method}")
                raise HeadlessSimError(
                    f"HeadlessSim RPC {method} timed out after "
                    f"{waited:.1f}s (sim hung on params={json.dumps(params or {}, ensure_ascii=False)[:200]} — "
                    f"killed; see {self._hang_log_path} for context)"
                ) from None
        elapsed = time.time() - sent_at
        # Phase A checkpoint D: got a line from reader queue.
        self._hang_log("rpc_stdout_await_done", method=method, elapsed_s=elapsed)
        self._hang_log("recv", method=method, elapsed_s=elapsed)
        if line is None:
            # Reader thread saw EOF or error — sim died.
            self._kill_sim(f"eof on {method}")
            raise HeadlessSimError(
                f"HeadlessSim closed stdout on {method} (process died)"
            )
        # `line` is already a str in text mode.
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise HeadlessSimError(
                f"HeadlessSim returned non-JSON for {method}: {line[:400]!r}"
            ) from exc

    # ------------------------------------------------------------------
    # BridgeClient-compatible public API
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def base_url(self) -> str:
        return f"headless_sim://{self._exe_path.name}"

    @property
    def session_base_url(self) -> str:
        return self.base_url

    @property
    def session_id(self) -> str:
        proc = self._proc
        return f"headless-sim-{proc.pid}" if proc is not None else "headless-sim-closed"

    def health(self) -> dict[str, Any]:
        return {"ok": self.is_connected, "backend": "headless_sim"}

    def get_spec(self) -> dict[str, Any]:
        """Return the shared typed backend capabilities without another RPC."""

        return {
            "ok": True,
            "backend": "headless_sim",
            "api_version": API_VERSION,
            "schema_version": SCHEMA_VERSION,
            "supports_full_run": True,
            "supports_combat_reset": True,
            "supports_seed": True,
        }

    def combat_catalog(self) -> dict[str, Any]:
        response = self._rpc("combat_catalog")
        return response

    def perf_stats(self) -> dict[str, Any]:
        """Pull FullRunSimulationDiagnostics counters from the sim.

        Useful for diagnosing sim hangs — in particular these counters
        after a session of training will tell you whether the hang-relief
        patches are actually firing:
          - combat_step.action_executor_timeout : WaitForSettledAndSnapshotAsync
            hit its 2s timeout (ActionExecutor was truly stuck)
          - settle.wait_combat_followup.*.scheduler_yield_deadline : the
            wall-clock budget inside TryAdvance* expired
          - settle.wait_combat_followup.*.scheduler_yield_hits : clean
            progress was detected after extra yields (patch doing its job)
        """
        return self._rpc("perf_stats")

    def get_state(self) -> dict[str, Any]:
        sim_state = self._rpc("state")
        from sts2_env._sim_translate import translate_to_bridge_shape
        return translate_to_bridge_shape(
            sim_state,
            episode_id=self._current_episode_id,
        )

    def reset(
        self,
        character: str | None = None,
        rebind_active_run: bool = False,
        force_fresh: bool = False,
        defensive_buffs: bool = False,
        seed: str | int | None = None,
        timeout_ms: int = 45_000,
    ) -> dict[str, Any]:
        timeout_s = max(float(timeout_ms) / 1000.0, 0.001)
        if rebind_active_run and self._current_episode_id:
            sim_state = self._rpc("state", timeout_s=timeout_s)
            return _build_bridge_step_response(
                self, sim_state, episode_started=False, reward=0.0,
                info_extra={"sim_rebound_active_run": True},
            )
        params: dict[str, Any] = {}
        if character is not None:
            params["character_id"] = _normalize_character(character)
        if seed is not None:
            params["seed"] = str(seed)
        if defensive_buffs:
            params["defensive_buffs"] = True
        sim_state = self._rpc("reset", params, timeout_s=timeout_s)
        return _build_bridge_step_response(
            self, sim_state, episode_started=True, reward=0.0,
        )

    def step(
        self,
        episode_id: str,
        action_index: int | None = None,
        action_id: str | None = None,
        timeout_ms: int = 20_000,
    ) -> dict[str, Any]:
        if str(episode_id) != self._current_episode_id:
            raise HeadlessSimError(
                f"step episode mismatch: requested={episode_id!r}, active={self._current_episode_id!r}"
            )
        # Sim's step() takes the action itself (action name + index/card_index/
        # target_id etc.), not a 0-based index into a previously-emitted
        # legal_actions list. We stored the raw sim action dict inside each
        # bridge legal_action under ``_sim_raw`` at translation time; look it
        # up and forward the fields the sim's ParseActionRequest understands.
        raw: dict[str, Any] | None = None
        if action_index is not None and self._last_legal_actions:
            idx = int(action_index)
            if 0 <= idx < len(self._last_legal_actions):
                raw = self._last_legal_actions[idx].get("_sim_raw")
        elif action_id is not None:
            # action_id is our synthetic "sim:<kind>:<index>" id
            for entry in self._last_legal_actions or []:
                if entry.get("action_id") == action_id:
                    raw = entry.get("_sim_raw")
                    break
        if not isinstance(raw, dict):
            raise HeadlessSimError(
                f"step: could not resolve action_index={action_index} / "
                f"action_id={action_id!r} against last legal_actions "
                f"({len(self._last_legal_actions)} entries)"
            )

        # Build the sim params dict from the raw action. Sim expects fields
        # named exactly as its legal_actions emit them.
        params: dict[str, Any] = {"action": str(raw.get("action") or "")}
        for field in ("index", "card_index", "target_id", "target", "value",
                      "type", "col", "row", "choice_index", "slot", "card_id"):
            if field in raw and raw[field] is not None:
                params[field] = raw[field]

        wrapped = self._rpc("step", params, timeout_s=max(float(timeout_ms) / 1000.0, 0.001))
        # Sim step wraps the new state under {accepted, state, error}. Unwrap.
        sim_state = wrapped.get("state") or {}
        accepted = bool(wrapped.get("accepted", True))
        reward = float(wrapped.get("reward", 0.0) or 0.0)
        info_extra: dict[str, Any] = {}
        if not accepted:
            info_extra["sim_step_rejected"] = True
            info_extra["sim_error"] = str(wrapped.get("error") or "")

        return _build_bridge_step_response(
            self, sim_state, episode_started=False, reward=reward,
            info_extra=info_extra,
        )

    def combat_reset(
        self,
        *,
        character: str | None = None,
        encounter_id: str | None = None,
        seed: int | None = None,
        current_hp: int | None = None,
        max_hp: int | None = None,
        max_energy: int | None = None,
        deck: list[str] | None = None,
        deck_entries: list[dict[str, Any]] | None = None,
        relics: list[str] | None = None,
        additional_relics: list[str] | None = None,
        potions: list[str] | None = None,
        gold: int | None = None,
        timeout_ms: int = 15_000,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if character is not None:
            params["character_id"] = _normalize_character(character)
        if encounter_id is not None:
            # Sim's CombatTrainingMode.ResolveEncounter re-prepends "ENCOUNTER."
            # via new ModelId(ModelId.SlugifyCategory<EncounterModel>(), value),
            # producing "ENCOUNTER.ENCOUNTER.<SUFFIX>" if we pass the fully
            # qualified id. Strip any leading "ENCOUNTER." (case-insensitive)
            # so the sim receives the raw suffix it expects.
            enc = str(encounter_id)
            if enc.upper().startswith("ENCOUNTER."):
                enc = enc[len("ENCOUNTER."):]
            params["encounter_id"] = enc
        if seed is not None:
            params["seed"] = str(seed)

        # Sim's CombatResetAsync only reads 4 flat params (character_id,
        # encounter_id, seed, ascension_level) + a nested `build` object of
        # type SimulationBuildSpec. Flat deck/relics/current_hp/etc. are
        # silently ignored — every combat_reset was running on the default
        # starter Ironclad configuration regardless of snapshot content.
        # Fix: pack snapshot fields into `build`. SimulationBuildSpec fields
        # per <STS2_AI_ROOT>/.../FullRunSimulationDtos.cs:
        #   deck: list[SimulationBuildCardSpec{id, upgrade_level}]
        #   relics: list[SimulationBuildRelicSpec{id}]
        #   current_hp / max_hp / max_energy / gold: int
        build: dict[str, Any] = {}
        if current_hp is not None:
            build["current_hp"] = int(current_hp)
        if max_hp is not None:
            build["max_hp"] = int(max_hp)
        if max_energy is not None:
            build["max_energy"] = int(max_energy)
        if gold is not None:
            build["gold"] = int(gold)
        # Sim's SimulationBuildSupport.ResolveCard re-prepends "CARD." (and
        # analogous for RELIC./POTION.), same bug pattern as encounter_id.
        # Strip the category prefix so we don't end up with CARD.CARD.ANGER.
        def _strip_prefix(raw: str, prefix: str) -> str:
            s = str(raw)
            if s.upper().startswith(prefix):
                s = s[len(prefix):]
            return s

        if deck_entries is not None and len(deck_entries) > 0:
            # Prefer upgrade-aware entries when we have them.
            build["deck"] = [
                {
                    "id": _strip_prefix(entry.get("id"), "CARD."),
                    "upgrade_level": int(entry.get("upgrade_level") or 0),
                }
                for entry in deck_entries
                if isinstance(entry, dict) and entry.get("id")
            ]
        elif deck is not None:
            build["deck"] = [
                {"id": _strip_prefix(cid, "CARD."), "upgrade_level": 0}
                for cid in deck if cid
            ]
        if relics is not None:
            build["relics"] = [
                {"id": _strip_prefix(rid, "RELIC.")} for rid in relics if rid
            ]
        if additional_relics is not None:
            build["additional_relics"] = [
                {"id": _strip_prefix(rid, "RELIC.")}
                for rid in additional_relics
                if rid
            ]
        if potions is not None:
            # SimulationBuildSpec may or may not have potions; include under
            # build to be future-proof; sim ignores unknown fields.
            build["potions"] = [_strip_prefix(pid, "POTION.") for pid in potions if pid]
        if build:
            params["build"] = build
        timeout_s = max(float(timeout_ms) / 1000.0, 0.001)
        combat_result = self._rpc("combat_reset", params, timeout_s=timeout_s)
        # The sim's combat_reset RPC returns a CombatTrainingStateSnapshot
        # (top-level keys: combat_active, enemies, hand, piles, ...). That
        # shape is different from FullRunSimulationStateSnapshot which our
        # translate_to_bridge_shape expects (battle.enemies, run.floor, ...).
        # After combat_reset, the sim's full-run state mirrors the active
        # combat, so re-fetch via the "state" RPC to get the shape the
        # translator knows how to read.
        if isinstance(combat_result, dict) and "error" in combat_result:
            raise HeadlessSimError(
                f"combat_reset failed: {combat_result.get('error_code')}: "
                f"{str(combat_result.get('error'))[:500]}"
            )
        sim_state = self._rpc("state", timeout_s=timeout_s)
        return _build_bridge_step_response(
            self, sim_state, episode_started=True, reward=0.0,
        )


def _normalize_character(character: str | None) -> str:
    """Sim expects ``IRONCLAD`` / ``SILENT`` / ..., our BridgeClient API gets
    things like ``CHARACTER.IRONCLAD`` or ``ironclad``. Normalize both."""
    if not character:
        return "IRONCLAD"
    s = str(character).strip().upper()
    if s.startswith("CHARACTER."):
        s = s.split(".", 1)[1]
    return s


def _build_bridge_step_response(
    client: HeadlessSimBridgeClient,
    sim_state: dict[str, Any],
    *,
    episode_started: bool,
    reward: float,
    info_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap a translated sim state into the shape our envs expect from
    ``BridgeClient.step/reset/combat_reset``.
    """
    if episode_started:
        client._episode_counter += 1
        client._current_episode_id = f"sim-ep-{client._episode_counter}"

    terminal = bool(sim_state.get("terminal", False))
    truncated = bool(sim_state.get("truncated", False))

    from sts2_env._sim_translate import translate_to_bridge_shape
    bridge_obs = translate_to_bridge_shape(
        sim_state,
        episode_id=client._current_episode_id,
    )
    legal_actions = bridge_obs.get("available_actions") or []
    # Cache so the next step() call can map an action_index back to the raw
    # sim action dict stored under ``_sim_raw``.
    client._last_legal_actions = legal_actions

    info = dict(sim_state.get("info") or {})
    if info_extra:
        info.update(info_extra)

    # This adapter preserves the simulator scalar as transport data only.
    # HeadlessBackend projects transition facts and replaces it before any
    # learner sees a reward.
    info["sim_backend_reward"] = float(reward)

    return {
        "ok": True,
        "episode_id": client._current_episode_id,
        "step_index": int(sim_state.get("step_index", 0) or 0),
        "reward": float(reward),
        "reward_status": "backend-diagnostic",
        "reward_authority": "backend-diagnostic",
        "done": terminal,
        "truncated": truncated,
        "obs": bridge_obs,
        "legal_actions": legal_actions,
        "info": info,
    }
