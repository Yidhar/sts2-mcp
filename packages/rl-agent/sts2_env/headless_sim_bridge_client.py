"""BridgeClient-compatible adapter that talks to frankqwang/sts2-ai's
HeadlessSim instead of the real Godot bridge HTTP mod.

The HeadlessSim is a C# process that links against the decompiled STS2 game
code minus rendering — so game logic is identical to the real game but it
runs ~100-1000× faster, can be parallelized to many processes, and doesn't
crash from PunchOff / silent-hang / save-file contention issues.

Architectural notes:
- This client IS a BridgeClient subclass API-wise (same method signatures)
  so CombatSandboxEnv and SlayTheSpire2EnvV2 don't need changes.
- Sim speaks line-delimited JSON over stdio (`{"method": ..., "params": ...}`
  per line). Much lower latency than HTTP.
- Sim's state dict format is STRUCTURALLY DIFFERENT from our bridge's. All
  the translation happens in ``_translate_to_bridge_shape``.
- This is Phase 2 of the sim migration — focused on getting training running
  today. Many aux fields (self_inflicted_hp_loss_cumulative,
  incoming_damage_multiplier, power.id, event_option.effect_deltas) are
  zero-filled for now; Phase 3 will port those from our mod into the sim.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

# Inherit from BridgeError so the combat_env narrow `except BridgeError`
# path catches sim failures too. env_v2 already uses `except Exception` so
# it doesn't care about the MRO.
from sts2_env.bridge_client import BridgeError


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

    DEFAULT_EXE_PATH = Path(
        "E:/game/project/sts2_mcp/third_party/sts2-ai/STS2AI/ENV/Sim/Host/"
        "bin/Debug/net9.0/headless_sim_host_0991.exe"
    )

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
        self._exe_path = Path(exe_path or self.DEFAULT_EXE_PATH)
        if not self._exe_path.exists():
            raise HeadlessSimError(f"HeadlessSim exe not found: {self._exe_path}")
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
        self._stdout_queue: queue.Queue = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        # Per-process hang-debug log. Each RPC starts by appending a line
        # here; on hang, the tail of this file identifies the exact action
        # that made the sim stop responding. File name includes the sim
        # PID once it's up so 4 parallel sims don't clobber each other.
        self._hang_log_path: Path | None = None
        self._hang_log_handle = None
        # Tracks cumulative self-inflicted HP loss across combat turns.
        # Reset at combat entry, advanced when a known self-damage card is
        # played and HP subsequently drops. Mirrors the bridge mod's
        # BuildEnvCombatPayload.self_inflicted_hp_loss_cumulative field.
        from sts2_env._sim_translate import SelfInflictedHpTracker  # noqa: PLC0415
        self._self_inflicted_tracker = SelfInflictedHpTracker()
        self._last_in_combat: bool = False
        # Combat-sandbox terminal-reward parity. Sim's C# side emits a flat
        # -1.0 / +1.0 terminal reward which does NOT match live bridge mod's
        # breakdown (BridgeGameApi.EnvHelpers.BuildEnvCombatSandboxReward-
        # Breakdown). Live formula for sandbox loss: death(-2.0) +
        # room_hp_delta_normalized*1.5 = -3.5 on full-HP-loss boss defeat.
        # We track per-combat start state at combat_reset and override sim's
        # terminal reward with the live-parity formula.
        self._sandbox_start_hp: float | None = None
        self._sandbox_max_hp: float | None = None
        self._sandbox_encounter_id: str | None = None
        self._start_subprocess(startup_timeout_s)

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------
    def _start_subprocess(self, startup_timeout_s: float) -> None:
        env = os.environ.copy()
        # Silence ThreadPool tuning noise; we don't need the stderr chatter.
        env.setdefault("STS2_MIN_THREADS", "8")
        # PATCH 2026-04-18 (downstream fork): reconfigured subprocess pipe
        # params after evidence-driven debugging pinpointed the hang to
        # Python-side pipe reading (sim C# tracing showed stdout flushed
        # in 8ms, Python timed out 27.8s later waiting on readline).
        #
        # Old: bufsize=0 + text=False → raw FileIO; readline() falls back
        #   to byte-by-byte syscall loop. Flaky for Windows pipes with
        #   responses > ~4KB (observed failure mode: 7680-byte end_turn
        #   response sat in OS pipe buffer but readline never returned).
        #
        # New: bufsize=1 + text=True → line-buffered TextIOWrapper over
        #   BufferedReader. readline() uses the buffer and only returns
        #   a full line. Universal newline handling ('' default) normalizes
        #   Windows \r\n to \n transparently.
        self._proc = subprocess.Popen(
            [str(self._exe_path), "--stdio", "--protocol", "json"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        # Wait for the "stdio mode ready" banner on stderr so we know sim is up.
        # stderr is now text mode too — read as str.
        deadline = time.time() + startup_timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise HeadlessSimError(
                    f"HeadlessSim exited during startup (rc={self._proc.returncode})"
                )
            line = self._proc.stderr.readline()
            if not line:
                time.sleep(0.05)
                continue
            decoded = line
            # Sim writes "HeadlessSim: stdio mode ready" once booted.
            if "stdio mode ready" in decoded:
                # Spin up the reader thread NOW so the first _rpc call
                # doesn't race against stdout buffering.
                self._reader_stop.clear()
                self._reader_thread = threading.Thread(
                    target=self._stdout_reader_loop,
                    name=f"sim-stdout-reader-pid{self._proc.pid}",
                    daemon=True,
                )
                self._reader_thread.start()
                # PATCH 2026-04-18: sim writes to stderr periodically
                # (bootstrap banner, watchdog heartbeat, Godot INFO/WARN,
                # Sentry telemetry). The stderr pipe buffer is tiny
                # (~4-64KB on Windows) — if nobody drains it, sim's next
                # Console.Error.Write blocks, which can cascade into the
                # stdio handler thread stalling and rpc hangs we've
                # misdiagnosed for 10+ rounds. Always drain stderr.
                self._stderr_thread = threading.Thread(
                    target=self._stderr_reader_loop,
                    name=f"sim-stderr-reader-pid{self._proc.pid}",
                    daemon=True,
                )
                self._stderr_thread.start()
                # Open hang-debug log for this sim instance. Kept small
                # (truncated on reopen); we only need the last ~200 actions
                # to backtrack a hang.
                try:
                    self._hang_log_path = Path(
                        os.environ.get("STS2_SIM_HANG_LOG_DIR") or "."
                    ) / f"sim_hang_debug_pid{self._proc.pid}.log"
                    self._hang_log_handle = open(self._hang_log_path, "w", encoding="utf-8")
                    self._hang_log_handle.write(
                        f"# sim hang-debug log for pid={self._proc.pid} started at {time.time()}\n"
                    )
                    self._hang_log_handle.flush()
                except Exception:
                    self._hang_log_path = None
                    self._hang_log_handle = None
                return
        raise HeadlessSimError("HeadlessSim failed to become ready in time")

    def _stderr_reader_loop(self) -> None:
        """Dedicated thread that drains sim stderr into the hang-debug log.
        Prevents stderr pipe buffer from filling up (which would block
        sim's next Console.Error.Write and cascade into stdio hangs).
        """
        proc = self._proc
        if proc is None:
            return
        stderr = proc.stderr
        try:
            while not self._reader_stop.is_set():
                line = stderr.readline()
                if not line:
                    return
                # Keep the last N stderr lines in the hang-debug log; cap
                # per-line length so pathological output doesn't explode.
                snippet = line.rstrip("\r\n")
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                self._hang_log("sim_stderr", line=snippet)
        except Exception as exc:
            self._hang_log("stderr_reader_exception", error=repr(exc))

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
                    self._stdout_queue.put(None)
                    break
                # text mode — len(line) is character count
                self._hang_log("reader_readline_done", nchars=len(line))
                self._stdout_queue.put(line)
        except Exception as exc:  # noqa: BLE001
            # Push a sentinel so waiters don't block forever. We swallow
            # the exception because the worker thread picking up the
            # sentinel will raise a clean HeadlessSimError.
            self._hang_log("reader_exception", error=repr(exc))
            self._stdout_queue.put(None)

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

    def _kill_sim(self, reason: str) -> None:
        """Force-terminate the sim subprocess. Called when we detect a hang
        via RPC timeout or when close() is invoked. Idempotent.
        """
        if self._proc is None:
            return
        # Signal reader thread to stop so we don't race against the pipe
        # closure (it will also notice EOF and push a sentinel).
        self._reader_stop.set()
        try:
            self._proc.kill()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5.0)
        except Exception:
            pass
        self._proc = None
        # Drain the queue so a subsequent _rpc call doesn't pick up stale
        # bytes from a dead sim.
        try:
            while True:
                self._stdout_queue.get_nowait()
        except queue.Empty:
            pass

    def close(self) -> None:
        # Public lifecycle entry. Must always leave _proc=None and sim
        # process reaped even on errors — otherwise we leak zombies when
        # AsyncReadyCollector restarts an env.
        if self._hang_log_handle is not None:
            try:
                self._hang_log_handle.close()
            except Exception:
                pass
            self._hang_log_handle = None
        with self._lock:
            if self._proc is None:
                return
            self._reader_stop.set()
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    self._proc.kill()
                except Exception:
                    pass
                try:
                    self._proc.wait(timeout=3.0)
                except Exception:
                    pass
            self._proc = None
        if self._reader_thread is not None:
            # Don't join the reader thread here — if close() is called from
            # a worker thread that's holding the lock, joining could
            # deadlock. The thread is daemon=True so it dies with the
            # process anyway.
            self._reader_thread = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Low-level RPC
    # ------------------------------------------------------------------
    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            raise HeadlessSimError(
                f"HeadlessSim subprocess is not running (method={method})"
            )
        request = {"method": method}
        if params is not None:
            request["params"] = params
        # text mode — write str, not bytes
        payload = json.dumps(request, ensure_ascii=False) + "\n"
        sent_at = time.time()
        self._hang_log("sent", method=method, params=params, sent_at=sent_at)
        with self._lock:
            # Phase A checkpoint A: about to write request payload to sim stdin.
            self._hang_log("rpc_stdin_write_begin", method=method, nbytes=len(payload))
            try:
                self._proc.stdin.write(payload)
                self._proc.stdin.flush()
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
                line = self._stdout_queue.get(timeout=self._rpc_timeout_s)
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
                    rpc_timeout_s=self._rpc_timeout_s,
                    reader_thread_alive=reader_alive,
                    proc_poll=self._proc.poll() if self._proc else None,
                )
                self._kill_sim(f"rpc timeout {self._rpc_timeout_s}s on {method}")
                raise HeadlessSimError(
                    f"HeadlessSim RPC {method} timed out after "
                    f"{waited:.1f}s (sim hung on params={json.dumps(params or {}, ensure_ascii=False)[:200]} — "
                    f"killed; see {self._hang_log_path} for context)"
                )
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

    def health(self) -> dict[str, Any]:
        return {"ok": self.is_connected, "backend": "headless_sim"}

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
        from sts2_env._sim_translate import translate_to_bridge_shape  # noqa: PLC0415
        return translate_to_bridge_shape(
            sim_state,
            episode_id=self._current_episode_id,
            self_inflicted_tracker=self._self_inflicted_tracker,
        )

    def reset(
        self,
        character: str | None = None,
        rebind_active_run: bool = False,
        force_fresh: bool = False,
        defensive_buffs: bool = False,
        timeout_ms: int = 45_000,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if character is not None:
            params["character_id"] = _normalize_character(character)
        sim_state = self._rpc("reset", params)
        # Full-run reset clears any combat_sandbox tracking — the sandbox
        # reward formula shouldn't fire during full-run episodes.
        self._sandbox_start_hp = None
        self._sandbox_max_hp = None
        self._sandbox_encounter_id = None
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

        # Inform the self-damage tracker about the action BEFORE dispatching,
        # so the post-step HP observation can attribute any drop.
        self._self_inflicted_tracker.note_action(raw)

        wrapped = self._rpc("step", params)
        # Sim step wraps the new state under {accepted, state, error}. Unwrap.
        sim_state = wrapped.get("state") or {}
        accepted = bool(wrapped.get("accepted", True))
        reward = float(wrapped.get("reward", 0.0) or 0.0)
        info_extra: dict[str, Any] = {}
        if not accepted:
            info_extra["sim_step_rejected"] = True
            info_extra["sim_error"] = str(wrapped.get("error") or "")

        # Deadlock workaround (2026-04-18): the C# sim's internal
        # auto-advance runs up to 30 iterations to settle a turn transition,
        # then returns control. Empirically, dense enemy-turn resolution
        # (multiple enemies × multiple intents × power triggers) can exceed
        # that budget, leaving the sim in a state where it THINKS the
        # player can act but internal action_queue / coroutines are still
        # draining. The next play_card then hangs waiting for a
        # "player_turn_ready" signal that never arrives in headless mode.
        #
        # Observed pattern (3/3 hangs in one training run):
        #   end_turn → sim returns ok → play_card → deadlock at 45s.
        #
        # Fix: after every successful end_turn that leaves us still in
        # combat (not game_over, not post-combat event), proactively send
        # one extra "wait" RPC to give sim another auto-advance cycle.
        # "wait" is the same action sim's internal auto-advance uses, so
        # it's a well-tested code path.
        action_kind = str(raw.get("action") or "")
        if accepted and action_kind == "end_turn":
            post_state_type = str(sim_state.get("state_type") or "").lower()
            if post_state_type in ("combat", "battle"):
                try:
                    wait_wrapped = self._rpc("step", {"action": "wait"})
                    wait_state = wait_wrapped.get("state")
                    if isinstance(wait_state, dict):
                        sim_state = wait_state
                        # Accumulate any reward that fired during the
                        # wait's extra auto-advance (enemy pokes,
                        # end-of-turn powers, etc.)
                        reward += float(wait_wrapped.get("reward", 0.0) or 0.0)
                        info_extra["sim_end_turn_settle_wait"] = True
                except HeadlessSimError as exc:
                    # Wait itself hung/died — this combat is probably
                    # unrecoverable anyway. Fall through with the
                    # original end_turn response; the next normal step
                    # will hit the same error and trigger worker restart
                    # via our standard error path.
                    info_extra["sim_settle_wait_failed"] = repr(exc)
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
        # per third_party/sts2-ai/.../FullRunSimulationDtos.cs:
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
        if potions is not None:
            # SimulationBuildSpec may or may not have potions; include under
            # build to be future-proof; sim ignores unknown fields.
            build["potions"] = [_strip_prefix(pid, "POTION.") for pid in potions if pid]
        if build:
            params["build"] = build
        combat_result = self._rpc("combat_reset", params)
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
        sim_state = self._rpc("state")
        # Capture starting HP for live-parity terminal reward computation.
        # max_hp in the sandbox reset request sets the new combat's max;
        # current_hp sets the starting HP. Fall back to observed player HP
        # if either is missing from the request.
        observed_hp = _extract_player_hp(sim_state)
        self._sandbox_start_hp = float(current_hp if current_hp is not None else (observed_hp or 0))
        self._sandbox_max_hp = float(max_hp if max_hp is not None else (observed_hp or 0))
        self._sandbox_encounter_id = str(encounter_id) if encounter_id else None
        return _build_bridge_step_response(
            self, sim_state, episode_started=True, reward=0.0,
        )


# ----------------------------------------------------------------------
# Translation: sim state dict → bridge-shaped observation
# ----------------------------------------------------------------------

def _extract_player_hp(sim_state: dict[str, Any]) -> int | None:
    """Dig out the player's current HP from whichever sub-state holds it."""
    for key in ("battle", "event", "map", "rest_site", "shop", "treasure",
                "rewards", "card_reward", "card_select", "hand_select",
                "relic_select", "game_over"):
        section = sim_state.get(key)
        if isinstance(section, dict):
            player = section.get("player")
            if isinstance(player, dict):
                hp = player.get("current_hp", player.get("hp"))
                if isinstance(hp, int):
                    return hp
    player = sim_state.get("player")
    if isinstance(player, dict):
        hp = player.get("current_hp", player.get("hp"))
        if isinstance(hp, int):
            return hp
    return None


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
        # Fresh episode: reset self-damage tracking to the starting HP.
        initial_hp = _extract_player_hp(sim_state)
        client._self_inflicted_tracker.reset(initial_hp)
        client._last_in_combat = False

    terminal = bool(sim_state.get("terminal", False))
    truncated = bool(sim_state.get("truncated", False))

    # If we just entered combat (wasn't combat last tick, is now), reset the
    # self-damage tracker at combat start so carry-over from previous combats
    # doesn't contaminate. Bridge mod resets per combat via
    # ResetSelfInflictedHpLossTrackerForNewCombat; mirror here.
    sim_state_type = str(sim_state.get("state_type") or "").lower()
    in_combat_now = sim_state_type in {"combat", "battle"}
    if in_combat_now and not client._last_in_combat:
        client._self_inflicted_tracker.reset(_extract_player_hp(sim_state))
    client._last_in_combat = in_combat_now

    # Advance tracker with the new player HP (attribution happens if the
    # last dispatched action was a known self-damage card).
    client._self_inflicted_tracker.observe_hp(_extract_player_hp(sim_state))

    from sts2_env._sim_translate import translate_to_bridge_shape  # noqa: PLC0415
    bridge_obs = translate_to_bridge_shape(
        sim_state,
        episode_id=client._current_episode_id,
        self_inflicted_tracker=client._self_inflicted_tracker,
    )
    legal_actions = bridge_obs.get("available_actions") or []
    # Cache so the next step() call can map an action_index back to the raw
    # sim action dict stored under ``_sim_raw``.
    client._last_legal_actions = legal_actions

    info = dict(sim_state.get("info") or {})
    if info_extra:
        info.update(info_extra)

    # Live-parity combat_sandbox terminal reward. Live bridge mod emits a
    # 5-term breakdown (BuildEnvCombatSandboxRewardBreakdown): combat_won
    # (+1.0), room_hp_quality (clamp((hp-start)/max,-1,1) * 1.5), boss_clear
    # (+1.5), elite_clear (+0.75), death (-2.0). Sim's C# side hardcodes
    # -1.0 / +1.0 which training treats as ~uniform signal. Override here
    # at terminal when we were running a sandbox combat. Clears sandbox
    # tracking after firing so stale state can't bleed into the next run.
    if terminal and client._sandbox_encounter_id is not None:
        cur_hp = float(_extract_player_hp(sim_state) or 0)
        start_hp = float(client._sandbox_start_hp or 0)
        max_hp = float(client._sandbox_max_hp or 0)
        enc = str(client._sandbox_encounter_id).upper()
        room_hp_delta_normalized = 0.0
        if max_hp > 0.0:
            room_hp_delta_normalized = max(-1.0, min(1.0, (cur_hp - start_hp) / max_hp))
        room_hp_quality_bonus = room_hp_delta_normalized * 1.5
        combat_won = cur_hp > 0.0
        combat_won_bonus = 1.0 if combat_won else 0.0
        boss_clear_bonus = 1.5 if combat_won and enc.endswith("_BOSS") else 0.0
        elite_clear_bonus = 0.75 if combat_won and enc.endswith("_ELITE") else 0.0
        death_penalty = -2.0 if cur_hp <= 0.0 else 0.0
        parity_reward = (
            combat_won_bonus
            + room_hp_quality_bonus
            + boss_clear_bonus
            + elite_clear_bonus
            + death_penalty
        )
        info["_sandbox_parity_reward_breakdown"] = {
            "sim_original_reward": float(reward),
            "combat_won_bonus": combat_won_bonus,
            "room_hp_quality_bonus": room_hp_quality_bonus,
            "boss_clear_bonus": boss_clear_bonus,
            "elite_clear_bonus": elite_clear_bonus,
            "death_penalty": death_penalty,
            "total": parity_reward,
        }
        reward = parity_reward
        client._sandbox_start_hp = None
        client._sandbox_max_hp = None
        client._sandbox_encounter_id = None

    return {
        "ok": True,
        "episode_id": client._current_episode_id,
        "step_index": int(sim_state.get("step_index", 0) or 0),
        "reward": float(reward),
        "done": terminal,
        "truncated": truncated,
        "obs": bridge_obs,
        "legal_actions": legal_actions,
        "info": info,
    }


