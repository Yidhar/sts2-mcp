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
from copy import deepcopy
from pathlib import Path
from typing import Any, TextIO, cast

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


class HeadlessSimError(BridgeError):  # type: ignore[misc]
    """Raised when the HeadlessSim subprocess returns an error, dies, or
    fails to respond to an RPC within the configured timeout.
    """


class HeadlessSimProtocolError(HeadlessSimError):
    """The simulator returned a response that cannot be committed safely.

    ``response`` and ``params`` are deliberately retained on the exception so
    the typed backend can write one bounded, local incident record before it
    poisons the process-local session.  They are transport evidence only and
    never enter an observation or learner sample.
    """

    def __init__(
        self,
        message: str,
        *,
        method: str,
        params: dict[str, Any] | None = None,
        response: Any = None,
    ) -> None:
        super().__init__(message)
        self.method = str(method)
        self.params = deepcopy(params) if params is not None else None
        self.response = deepcopy(response)


class HeadlessSimStepRejected(HeadlessSimProtocolError):
    """A mutation was authoritatively rejected by HeadlessSim."""


class HeadlessSimUnsettledError(HeadlessSimProtocolError):
    """A committed mutation did not reach an actionable/terminal surface."""


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
        transition_poll_budget_s: float = 5.0,
        transition_poll_max_attempts: int = 32,
    ):
        self._exe_path = resolve_headless_sim_exe(exe_path)
        if protocol != "json":
            raise NotImplementedError("Only json protocol supported for now")

        self._request_timeout_s = float(request_timeout_s)
        self._rpc_timeout_s = float(rpc_timeout_s)
        if float(transition_poll_budget_s) <= 0.0:
            raise ValueError("transition_poll_budget_s must be positive")
        if int(transition_poll_max_attempts) <= 0:
            raise ValueError("transition_poll_max_attempts must be positive")
        self._transition_poll_budget_s = float(transition_poll_budget_s)
        self._transition_poll_max_attempts = int(transition_poll_max_attempts)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._episode_counter = 0
        self._current_episode_id: str = ""
        # ``combat_reset`` creates a combat-scoped episode on top of the
        # full-run simulator.  The full-run RPC reports a won combat as the
        # non-interactive ``combat_post_end_pending`` boundary rather than as
        # a terminal episode, so the adapter must remember which reset mode
        # owns the current episode.
        self._combat_episode_active = False
        self._last_observation: dict[str, Any] | None = None
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
        self._hang_log_handle: TextIO | None = None
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
            handle = self._hang_log_path.open("w", encoding="utf-8")
            self._hang_log_handle = handle
            handle.write(
                f"# sim hang-debug log for pid={proc.pid} started at {time.time()}\n"
            )
            for line in self._startup_stderr_tail:
                self._hang_log("sim_startup_stderr", line=line)
            handle.flush()
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
        if stdout is None:
            return
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

    def __del__(self) -> None:
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
        request: dict[str, Any] = {"method": method}
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
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HeadlessSimError(
                f"HeadlessSim returned non-JSON for {method}: {line[:400]!r}"
            ) from exc
        if not isinstance(response, dict):
            raise HeadlessSimProtocolError(
                f"HeadlessSim returned a non-object JSON response for {method}",
                method=method,
                params=params,
                response=response,
            )
        return response

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
        return cast(
            dict[str, Any],
            translate_to_bridge_shape(
                sim_state,
                episode_id=self._current_episode_id,
            ),
        )

    def reset(
        self,
        character: str | None = None,
        rebind_active_run: bool = False,
        force_fresh: bool = False,
        defensive_buffs: bool = False,
        additional_relics: list[str] | None = None,
        training_revival_budget: int | None = None,
        seed: str | int | None = None,
        timeout_ms: int = 45_000,
    ) -> dict[str, Any]:
        self._combat_episode_active = False
        timeout_s = max(float(timeout_ms) / 1000.0, 0.001)
        if rebind_active_run and self._current_episode_id:
            sim_state = self._rpc("state", timeout_s=timeout_s)
            return _build_bridge_step_response(
                self, sim_state, episode_started=False, reward=0.0,
                info_extra={"sim_rebound_active_run": True},
                source_method="state",
            )
        params: dict[str, Any] = {}
        if character is not None:
            params["character_id"] = _normalize_character(character)
        if seed is not None:
            params["seed"] = str(seed)
        if defensive_buffs:
            params["defensive_buffs"] = True
        build: dict[str, Any] = {}
        if additional_relics is not None:
            build["additional_relics"] = [
                {"id": _strip_model_prefix(relic_id, "RELIC.")}
                for relic_id in additional_relics
                if relic_id
            ]
        if training_revival_budget is not None:
            budget = int(training_revival_budget)
            if budget < -1:
                raise ValueError(
                    "training_revival_budget must be -1 or a non-negative integer"
                )
            build["training_revival_budget"] = budget
        if build:
            params["build"] = build
        sim_state = self._rpc("reset", params, timeout_s=timeout_s)
        return _build_bridge_step_response(
            self, sim_state, episode_started=True, reward=0.0,
            source_method="reset",
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
        wrapped = self._resolve_authoritative_settlement(wrapped, step_params=params)
        # Sim step wraps the new state under {accepted, state, error}.  Never
        # coerce this authority bit with bool(...): strings such as "false"
        # are truthy in Python and used to turn a rejected action into an
        # apparently successful transition here.
        accepted_value = wrapped.get("accepted")
        if not isinstance(accepted_value, bool):
            raise HeadlessSimProtocolError(
                "HeadlessSim step response has no exact boolean accepted field",
                method="step",
                params=params,
                response=wrapped,
            )
        if not accepted_value:
            error = str(wrapped.get("error") or "unspecified simulator rejection")
            raise HeadlessSimStepRejected(
                f"HeadlessSim rejected step: {error[:500]}",
                method="step",
                params=params,
                response=wrapped,
            )

        sim_state_value = wrapped.get("state")
        if not isinstance(sim_state_value, dict):
            raise HeadlessSimProtocolError(
                "accepted HeadlessSim step response has no state object",
                method="step",
                params=params,
                response=wrapped,
            )
        sim_state = sim_state_value
        reward = float(wrapped.get("reward", 0.0) or 0.0)

        return _build_bridge_step_response(
            self, sim_state, episode_started=False, reward=reward,
            source_method="step",
        )

    def _resolve_authoritative_settlement(
        self,
        wrapped: dict[str, Any],
        *,
        step_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve an explicitly typed committed transition without retrying it.

        Older simulators do not expose settlement authority.  Their stable
        actionable/terminal responses continue through the legacy structural
        checks, but an actionless response is *not* guessed to be pending.
        Read-only polling is enabled only when the simulator supplies the
        authoritative ``settlement_status``, ``action_committed`` and
        ``transition_token`` protocol added for this purpose.
        """

        authority_keys = {"settlement_status", "action_committed", "transition_token"}
        if not authority_keys.intersection(wrapped):
            return wrapped

        status_value = wrapped.get("settlement_status")
        committed_value = wrapped.get("action_committed")
        accepted_value = wrapped.get("accepted")
        if not isinstance(status_value, str) or not status_value.strip():
            raise HeadlessSimProtocolError(
                "authoritative step response has no settlement_status",
                method="step",
                params=step_params,
                response=wrapped,
            )
        status = status_value.strip().lower()
        if status not in {"actionable", "terminal", "pending", "unsettled", "rejected"}:
            raise HeadlessSimProtocolError(
                f"authoritative step response has unknown settlement_status={status!r}",
                method="step",
                params=step_params,
                response=wrapped,
            )
        if not isinstance(committed_value, bool):
            raise HeadlessSimProtocolError(
                "authoritative step response has no exact boolean action_committed field",
                method="step",
                params=step_params,
                response=wrapped,
            )
        if not isinstance(accepted_value, bool):
            raise HeadlessSimProtocolError(
                "authoritative step response has no exact boolean accepted field",
                method="step",
                params=step_params,
                response=wrapped,
            )

        if not accepted_value:
            if committed_value or status != "rejected":
                raise HeadlessSimProtocolError(
                    "rejected step response has inconsistent settlement authority",
                    method="step",
                    params=step_params,
                    response=wrapped,
                )
            error = str(wrapped.get("error") or "unspecified simulator rejection")
            raise HeadlessSimStepRejected(
                f"HeadlessSim rejected step: {error[:500]}",
                method="step",
                params=step_params,
                response=wrapped,
            )

        if not committed_value or status == "rejected":
            raise HeadlessSimProtocolError(
                "accepted step response has inconsistent settlement authority",
                method="step",
                params=step_params,
                response=wrapped,
            )
        if status in {"actionable", "terminal"}:
            self._validate_authoritative_settled_surface(
                wrapped,
                status=status,
                method="step",
                params=step_params,
            )
            return wrapped

        state_value = wrapped.get("state")
        if (
            self._combat_episode_active
            and isinstance(state_value, dict)
            and str(state_value.get("state_type") or "").lower()
            == "combat_post_end_pending"
        ):
            # The simulator authority is scoped to the full run, while a
            # combat_reset episode intentionally terminates at the first
            # post-combat boundary.  Preserve that explicit local episode
            # contract instead of polling through it into full-run rewards.
            # Full-run episodes never take this branch.
            return wrapped

        token_value = wrapped.get("transition_token")
        if not isinstance(token_value, str) or not token_value.strip():
            raise HeadlessSimProtocolError(
                "committed pending step response has no transition_token",
                method="step",
                params=step_params,
                response=wrapped,
            )
        transition_token = token_value.strip()
        initial_reward = wrapped.get("reward", 0.0)
        deadline = time.monotonic() + self._transition_poll_budget_s
        last_response = wrapped
        for attempt in range(self._transition_poll_max_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            poll_params = {"transition_token": transition_token}
            polled = self._rpc(
                "poll_transition",
                poll_params,
                timeout_s=min(remaining, self._rpc_timeout_s),
            )
            last_response = polled
            poll_status_value = polled.get("settlement_status")
            poll_accepted = polled.get("accepted")
            poll_committed = polled.get("action_committed")
            poll_token = polled.get("transition_token")
            if (
                not isinstance(poll_status_value, str)
                or poll_status_value.strip().lower()
                not in {"actionable", "terminal", "pending", "unsettled"}
                or poll_accepted is not True
                or poll_committed is not True
                or poll_token != transition_token
            ):
                raise HeadlessSimProtocolError(
                    "poll_transition returned inconsistent settlement authority",
                    method="poll_transition",
                    params=poll_params,
                    response=polled,
                )
            poll_status = poll_status_value.strip().lower()
            if poll_status in {"actionable", "terminal"}:
                resolved = dict(polled)
                resolved.setdefault("reward", initial_reward)
                self._validate_authoritative_settled_surface(
                    resolved,
                    status=poll_status,
                    method="poll_transition",
                    params=poll_params,
                )
                return resolved

            # A read-only poll is deliberately not a mutation replay, but it
            # must also give the simulator's asynchronous continuation real
            # wall-clock time to run.  Without pacing, an in-process/fake-fast
            # transport can burn all attempts in effectively 0 ms even though
            # this contract advertises a multi-second settlement budget.
            # Spread remaining attempts across the remaining deadline.  The
            # first poll stays immediate and ordinary already-settled steps
            # never sleep, so throughput is affected only while authority is
            # explicitly pending.
            polls_left = self._transition_poll_max_attempts - attempt - 1
            remaining_after_poll = deadline - time.monotonic()
            if polls_left > 0 and remaining_after_poll > 0.0:
                time.sleep(
                    remaining_after_poll / float(polls_left + 1)
                )

        raise HeadlessSimUnsettledError(
            "committed HeadlessSim transition did not settle within the bounded read-only poll budget",
            method="poll_transition",
            params={"transition_token": transition_token},
            response=last_response,
        )

    @staticmethod
    def _validate_authoritative_settled_surface(
        wrapped: dict[str, Any],
        *,
        status: str,
        method: str,
        params: dict[str, Any],
    ) -> None:
        state = wrapped.get("state")
        if not isinstance(state, dict):
            raise HeadlessSimProtocolError(
                f"authoritative {status} response has no state object",
                method=method,
                params=params,
                response=wrapped,
            )
        is_actionable = state.get("is_actionable")
        if not isinstance(is_actionable, bool):
            raise HeadlessSimProtocolError(
                f"authoritative {status} state has no exact boolean is_actionable field",
                method=method,
                params=params,
                response=wrapped,
            )
        terminal = state.get("terminal") is True or state.get("truncated") is True
        actions = state.get("legal_actions")
        if not isinstance(actions, list):
            raise HeadlessSimProtocolError(
                f"authoritative {status} state has no legal_actions array",
                method=method,
                params=params,
                response=wrapped,
            )
        if status == "actionable" and (not is_actionable or terminal or not actions):
            raise HeadlessSimProtocolError(
                "settlement_status=actionable is inconsistent with the state surface",
                method=method,
                params=params,
                response=wrapped,
            )
        if status == "terminal" and (is_actionable or not terminal or actions):
            raise HeadlessSimProtocolError(
                "settlement_status=terminal is inconsistent with the state surface",
                method=method,
                params=params,
                response=wrapped,
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
        training_revival_budget: int | None = None,
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
        if deck_entries is not None and len(deck_entries) > 0:
            # Prefer upgrade-aware entries when we have them.
            build["deck"] = [
                {
                    "id": _strip_model_prefix(entry.get("id"), "CARD."),
                    "upgrade_level": int(entry.get("upgrade_level") or 0),
                }
                for entry in deck_entries
                if isinstance(entry, dict) and entry.get("id")
            ]
        elif deck is not None:
            build["deck"] = [
                {"id": _strip_model_prefix(cid, "CARD."), "upgrade_level": 0}
                for cid in deck if cid
            ]
        if relics is not None:
            build["relics"] = [
                {"id": _strip_model_prefix(rid, "RELIC.")} for rid in relics if rid
            ]
        if additional_relics is not None:
            build["additional_relics"] = [
                {"id": _strip_model_prefix(rid, "RELIC.")}
                for rid in additional_relics
                if rid
            ]
        if training_revival_budget is not None:
            budget = int(training_revival_budget)
            if budget < -1:
                raise ValueError(
                    "training_revival_budget must be -1 or a non-negative integer"
                )
            build["training_revival_budget"] = budget
        if potions is not None:
            # SimulationBuildSpec may or may not have potions; include under
            # build to be future-proof; sim ignores unknown fields.
            build["potions"] = [
                _strip_model_prefix(pid, "POTION.") for pid in potions if pid
            ]
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
        self._combat_episode_active = True
        sim_state = self._rpc("state", timeout_s=timeout_s)
        return _build_bridge_step_response(
            self, sim_state, episode_started=True, reward=0.0,
            source_method="combat_reset",
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


def _strip_model_prefix(raw: Any, prefix: str) -> str:
    """Strip a category prefix before sending a model id to HeadlessSim."""

    value = str(raw)
    if value.upper().startswith(prefix):
        value = value[len(prefix):]
    return value


def _build_bridge_step_response(
    client: HeadlessSimBridgeClient,
    sim_state: dict[str, Any],
    *,
    episode_started: bool,
    reward: float,
    info_extra: dict[str, Any] | None = None,
    source_method: str = "state_projection",
) -> dict[str, Any]:
    """Wrap a translated sim state into the shape our envs expect from
    ``BridgeClient.step/reset/combat_reset``.
    """
    if episode_started:
        client._episode_counter += 1
        client._current_episode_id = f"sim-ep-{client._episode_counter}"

    sim_state = dict(sim_state)
    # CombatTrainingEnvService is hosted inside the full-run runtime.  After
    # its encounter is won, the full-run state advances to this internal
    # transition boundary with no legal actions, but the combat-scoped RL
    # episode is complete.  Treating it as non-terminal made every real win
    # look like a deadlocked episode and was the reason preheat reported zero
    # victories even under unlimited native revival.
    combat_victory_boundary = (
        client._combat_episode_active
        and str(sim_state.get("state_type") or "").lower()
        == "combat_post_end_pending"
    )
    if combat_victory_boundary:
        sim_state["terminal"] = True
        sim_state["run_outcome"] = "victory"
        sim_state["legal_actions"] = []

    for flag_name in ("terminal", "truncated"):
        flag_value = sim_state.get(flag_name, False)
        if not isinstance(flag_value, bool):
            raise HeadlessSimProtocolError(
                f"HeadlessSim state {flag_name} flag must be an exact boolean",
                method=source_method,
                response=sim_state,
            )
    terminal = bool(sim_state.get("terminal", False))
    truncated = bool(sim_state.get("truncated", False))

    from sts2_env._sim_translate import translate_to_bridge_shape
    translated_obs = translate_to_bridge_shape(
        sim_state,
        episode_id=client._current_episode_id,
    )
    if combat_victory_boundary and client._last_observation is not None:
        # The post-end DTO has already discarded the battle/player blocks.
        # Preserve the last factual combat snapshot for terminal HP/fact
        # derivation while taking counters and state identity from the actual
        # post-end state. No model action is ever chosen from this snapshot.
        bridge_obs = deepcopy(client._last_observation)
        for key in (
            "state_version",
            "state_hash",
            "semantic_state_hash",
            "_training",
            "_sim_raw",
        ):
            bridge_obs[key] = translated_obs.get(key)
        bridge_obs.update(
            {
                "state_type": "combat_victory",
                "terminated": True,
                "truncated": False,
                "available_actions": [],
            }
        )
        combat = bridge_obs.get("combat")
        if isinstance(combat, dict):
            combat["in_progress"] = False
    else:
        bridge_obs = translated_obs
    legal_actions = bridge_obs.get("available_actions") or []
    # A successful public environment result is either terminal or
    # actionable, never both and never neither.  Validate before mutating the
    # client's action/observation cache: once an invalid post-mutation surface
    # is seen the owning typed backend must poison and replace this client.
    if (terminal or truncated) and legal_actions:
        raise HeadlessSimProtocolError(
            "terminal HeadlessSim surface returned legal actions",
            method=source_method,
            response=sim_state,
        )
    if not terminal and not truncated and not legal_actions:
        raise HeadlessSimUnsettledError(
            "non-terminal HeadlessSim surface returned zero legal actions",
            method=source_method,
            response=sim_state,
        )
    if "is_actionable" in sim_state:
        is_actionable = sim_state["is_actionable"]
        if not isinstance(is_actionable, bool):
            raise HeadlessSimProtocolError(
                "HeadlessSim state is_actionable flag must be an exact boolean",
                method=source_method,
                response=sim_state,
            )
        expected_actionable = not terminal and not truncated and bool(legal_actions)
        if is_actionable is not expected_actionable:
            raise HeadlessSimProtocolError(
                "HeadlessSim state is_actionable flag contradicts terminal/action surface",
                method=source_method,
                response=sim_state,
            )
    # Cache so the next step() call can map an action_index back to the raw
    # sim action dict stored under ``_sim_raw``.
    client._last_legal_actions = legal_actions
    client._last_observation = deepcopy(bridge_obs)

    info = dict(sim_state.get("info") or {})
    if info_extra:
        info.update(info_extra)
    terminal_reason = (
        "combat_victory"
        if combat_victory_boundary
        else sim_state.get("terminal_reason", info.get("terminal_reason"))
    )
    if terminal_reason is not None:
        info["terminal_reason"] = str(terminal_reason)

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
        "terminal_reason": (
            str(terminal_reason) if terminal_reason is not None else None
        ),
        "obs": bridge_obs,
        "legal_actions": legal_actions,
        "info": info,
    }
