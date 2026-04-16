from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from typing import Any, Callable

import numpy as np


@dataclass(slots=True)
class AsyncReadyItem:
    env_id: int
    obs: dict[str, np.ndarray] | None
    info: dict[str, Any]
    episode_start: bool
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False
    transition_info: dict[str, Any] | None = None
    generation: int = 0
    ready_for_action: bool = True


@dataclass(slots=True)
class PrewarmedEnvSlot:
    env: object
    obs: dict[str, Any]
    info: dict[str, Any]


def stack_observations(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not observations:
        return {}
    keys = observations[0].keys()
    return {
        key: np.stack([np.asarray(obs[key]) for obs in observations], axis=0)
        for key in keys
    }


def resolve_action_mask(obs: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
    mask = info.get("action_mask") if isinstance(info, dict) else None
    if mask is None and isinstance(obs, dict):
        mask = obs.get("action_mask")
    if mask is None:
        raise KeyError("Missing action_mask in async ready item.")
    return np.asarray(mask, dtype=bool)


class AsyncReadyCollector:
    def __init__(
        self,
        env_factories: list[Callable[[], object]],
        *,
        prewarmed_slots: dict[int, PrewarmedEnvSlot] | None = None,
        step_watchdog_timeout_s: float = 45.0,
        reset_watchdog_timeout_s: float = 75.0,
        initial_reset_watchdog_timeout_s: float = 120.0,
        restart_cooldown_s: float = 15.0,
        event_log_path: str | None = None,
    ) -> None:
        self._env_factories = list(env_factories)
        self.num_envs = len(self._env_factories)
        self._ready_queue: Queue[AsyncReadyItem] = Queue()
        self._error_queue: Queue[BaseException] = Queue()
        self._restart_event_queue: Queue[dict[str, Any]] = Queue()
        self._stop_event = threading.Event()
        self._action_queues = [Queue(maxsize=1) for _ in range(self.num_envs)]
        self._threads: list[threading.Thread] = []
        self._prefetched_ready: dict[int, AsyncReadyItem] = {}
        self._prefetched_lock = threading.Lock()
        self._worker_status: dict[int, dict[str, Any]] = {}
        self._worker_status_lock = threading.Lock()
        self._worker_generation = {env_id: 0 for env_id in range(self.num_envs)}
        self._worker_restart_count = {env_id: 0 for env_id in range(self.num_envs)}
        self._last_restart_ts = {env_id: float("-inf") for env_id in range(self.num_envs)}
        self._worker_control_lock = threading.Lock()
        self._step_watchdog_timeout_s = max(float(step_watchdog_timeout_s), 1.0)
        self._reset_watchdog_timeout_s = max(float(reset_watchdog_timeout_s), 1.0)
        self._initial_reset_watchdog_timeout_s = max(float(initial_reset_watchdog_timeout_s), 1.0)
        self._restart_cooldown_s = max(float(restart_cooldown_s), 0.0)
        self._prewarmed_slots = dict(prewarmed_slots or {})
        self._event_log_path = Path(event_log_path).expanduser() if event_log_path else None
        self._event_log_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for env_id, env_factory in enumerate(self._env_factories):
            self._start_worker(env_id, env_factory, self._action_queues[env_id], self._prewarmed_slots.get(env_id))

    def close(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        for action_queue in self._action_queues:
            try:
                action_queue.put_nowait(None)
            except Exception:
                pass
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
        self._started = False

    def drain_ready(self, *, min_items: int = 0, timeout_s: float = 30.0, return_on_restart: bool = False) -> list[AsyncReadyItem]:
        restart_events = self.recover_stale_workers()
        self._raise_worker_error_if_any()
        items = self._drain_prefetched_ready()
        if return_on_restart and restart_events and min_items > 0 and not items:
            return items
        deadline = time.perf_counter() + max(float(timeout_s), 0.0)
        while len(items) < min_items:
            remaining_s = deadline - time.perf_counter()
            if remaining_s <= 0.0:
                self._raise_worker_error_if_any()
                status_summary = self._format_worker_status_summary()
                raise TimeoutError(
                    f"Async ready collector timed out after {float(timeout_s):.1f}s waiting for {min_items} ready item(s); "
                    f"received {len(items)}. Worker status: {status_summary}"
                )
            try:
                item = self._ready_queue.get(timeout=min(remaining_s, 1.0))
            except Empty:
                restart_events = self.recover_stale_workers()
                self._raise_worker_error_if_any()
                if return_on_restart and restart_events and not items:
                    return items
                continue
            if self._is_current_generation_item(item):
                items.append(item)
        while True:
            prefetched = self._drain_prefetched_ready()
            if prefetched:
                items.extend(prefetched)
                continue
            try:
                item = self._ready_queue.get_nowait()
            except Empty:
                break
            if self._is_current_generation_item(item):
                items.append(item)
        self._raise_worker_error_if_any()
        return items

    def push_ready_back(self, items: list[AsyncReadyItem]) -> None:
        if not items:
            return
        with self._prefetched_lock:
            for item in items:
                self._prefetched_ready[item.env_id] = item

    def dispatch_actions(self, env_ids: list[int], actions: np.ndarray | list[int]) -> None:
        for env_id, action in zip(env_ids, np.asarray(actions).reshape(-1), strict=True):
            self._action_queues[env_id].put(int(action))

    def pop_restart_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(self._restart_event_queue.get_nowait())
            except Empty:
                break
        return events

    def recover_stale_workers(self) -> list[dict[str, Any]]:
        if not self._started:
            return []
        now = time.perf_counter()
        with self._worker_status_lock:
            statuses = {env_id: dict(status) for env_id, status in self._worker_status.items()}
        events: list[dict[str, Any]] = []
        for env_id in range(self.num_envs):
            status = statuses.get(env_id)
            if not isinstance(status, dict):
                continue
            phase = str(status.get("phase") or "").strip()
            status_generation = int(status.get("generation", -1))
            age_s = max(now - float(status.get("ts") or now), 0.0)
            timeout_s: float | None = None
            if phase == "stepping":
                timeout_s = self._step_watchdog_timeout_s
            elif phase == "resetting_after_done":
                timeout_s = self._reset_watchdog_timeout_s
            elif phase == "resetting_initial":
                timeout_s = self._initial_reset_watchdog_timeout_s
            if timeout_s is None or age_s < timeout_s:
                continue
            with self._worker_control_lock:
                current_generation = int(self._worker_generation.get(env_id, 0))
                if status_generation != current_generation:
                    continue
                if (now - float(self._last_restart_ts.get(env_id, float("-inf")))) < self._restart_cooldown_s:
                    continue
                self._last_restart_ts[env_id] = now
                self._worker_generation[env_id] = current_generation + 1
                self._worker_restart_count[env_id] = int(self._worker_restart_count.get(env_id, 0)) + 1
                new_generation = self._worker_generation[env_id]
                self._action_queues[env_id] = Queue(maxsize=1)
                event = {
                    "env_id": env_id,
                    "stale_generation": current_generation,
                    "generation": new_generation,
                    "phase": phase,
                    "age_s": float(age_s),
                    "restart_count": int(self._worker_restart_count[env_id]),
                }
                self._restart_event_queue.put(event)
                self._log_event(
                    {
                        "event": "worker_restart",
                        "timestamp_unix_s": time.time(),
                        "env_id": env_id,
                        "stale_generation": current_generation,
                        "generation": new_generation,
                        "phase": phase,
                        "age_s": float(age_s),
                        "restart_count": int(self._worker_restart_count[env_id]),
                        "worker_restart_happened": True,
                    }
                )
                self._start_worker(
                    env_id,
                    self._env_factories[env_id],
                    self._action_queues[env_id],
                    None,
                    generation=new_generation,
                )
                self._update_worker_status(
                    env_id,
                    generation=new_generation,
                    phase="restarting",
                    restart_reason=phase,
                    restart_count=int(self._worker_restart_count[env_id]),
                )
                events.append(event)
        return events

    def _raise_worker_error_if_any(self) -> None:
        try:
            exc = self._error_queue.get_nowait()
        except Empty:
            return
        raise RuntimeError("Async ready collector worker failed") from exc

    def _drain_prefetched_ready(self) -> list[AsyncReadyItem]:
        with self._prefetched_lock:
            if not self._prefetched_ready:
                return []
            env_ids = sorted(self._prefetched_ready.keys())
            items = [self._prefetched_ready.pop(env_id) for env_id in env_ids]
        return items

    def _start_worker(
        self,
        env_id: int,
        env_factory: Callable[[], object],
        action_queue: Queue,
        prewarmed_slot: PrewarmedEnvSlot | None,
        *,
        generation: int | None = None,
    ) -> None:
        worker_generation = int(self._worker_generation.get(env_id, 0) if generation is None else generation)
        thread = threading.Thread(
            target=self._worker_loop,
            name=f"sts2-async-env-{env_id}-g{worker_generation}",
            args=(env_id, worker_generation, env_factory, action_queue, prewarmed_slot),
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def _worker_loop(
        self,
        env_id: int,
        generation: int,
        env_factory: Callable[[], object],
        action_queue: Queue,
        prewarmed_slot: PrewarmedEnvSlot | None,
    ) -> None:
        env = None
        try:
            if prewarmed_slot is not None:
                env = prewarmed_slot.env
                obs = prewarmed_slot.obs
                info = prewarmed_slot.info
                initial_reset_ms = float((info or {}).get("_serial_prewarm_reset_ms", 0.0))
                self._update_worker_status(
                    env_id,
                    generation=generation,
                    phase="publishing_initial_ready",
                    initial_reset_ms=initial_reset_ms,
                    prewarmed=True,
                )
            else:
                self._update_worker_status(env_id, generation=generation, phase="constructing_env")
                env = env_factory()
                self._update_worker_status(env_id, generation=generation, phase="resetting_initial")
                reset_started = time.perf_counter()
                obs, info = env.reset()
                initial_reset_ms = (time.perf_counter() - reset_started) * 1000.0
            self._update_worker_status(
                env_id,
                generation=generation,
                phase="publishing_initial_ready",
                initial_reset_ms=initial_reset_ms,
                prewarmed=bool(prewarmed_slot is not None),
            )
            self._ready_queue.put(
                AsyncReadyItem(
                    env_id=env_id,
                    obs=self._coerce_obs(obs),
                    info=dict(info) if isinstance(info, dict) else {},
                    episode_start=True,
                    generation=generation,
                )
            )
            self._update_worker_status(env_id, generation=generation, phase="waiting_action")

            while not self._stop_event.is_set():
                self._update_worker_status(env_id, generation=generation, phase="waiting_action")
                action = action_queue.get()
                if action is None or self._stop_event.is_set():
                    break

                self._update_worker_status(env_id, generation=generation, phase="stepping", action=int(action))
                step_started = time.perf_counter()
                next_obs, reward, terminated, truncated, step_info = env.step(int(action))
                transition_info = dict(step_info) if isinstance(step_info, dict) else {}
                step_ms = (time.perf_counter() - step_started) * 1000.0

                if terminated or truncated:
                    self._log_terminal_transition(
                        env_id=env_id,
                        generation=generation,
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        transition_info=transition_info,
                    )
                    self._ready_queue.put(
                        AsyncReadyItem(
                            env_id=env_id,
                            obs=None,
                            info={},
                            episode_start=False,
                            reward=float(reward),
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            transition_info=transition_info,
                            generation=generation,
                            ready_for_action=False,
                        )
                    )
                    self._update_worker_status(
                        env_id,
                        generation=generation,
                        phase="resetting_after_done",
                        last_step_ms=step_ms,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                    reset_started = time.perf_counter()
                    reset_obs, reset_info = env.reset()
                    self._update_worker_status(
                        env_id,
                        generation=generation,
                        phase="publishing_post_done_ready",
                        last_step_ms=step_ms,
                        post_done_reset_ms=(time.perf_counter() - reset_started) * 1000.0,
                    )
                    self._ready_queue.put(
                        AsyncReadyItem(
                            env_id=env_id,
                            obs=self._coerce_obs(reset_obs),
                            info=dict(reset_info) if isinstance(reset_info, dict) else {},
                            episode_start=True,
                            generation=generation,
                        )
                    )
                    continue

                self._update_worker_status(env_id, generation=generation, phase="publishing_ready", last_step_ms=step_ms)
                self._ready_queue.put(
                    AsyncReadyItem(
                        env_id=env_id,
                        obs=self._coerce_obs(next_obs),
                        info=transition_info,
                        episode_start=False,
                        reward=float(reward),
                        terminated=False,
                        truncated=False,
                        transition_info=transition_info,
                        generation=generation,
                    )
                )
        except BaseException as exc:  # noqa: BLE001
            self._update_worker_status(env_id, generation=generation, phase="failed", error=repr(exc))
            if self._is_current_generation(env_id, generation):
                self._error_queue.put(exc)
        finally:
            self._update_worker_status(env_id, generation=generation, phase="closed")
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    def _update_worker_status(self, env_id: int, *, generation: int, phase: str, **extra: Any) -> None:
        if not self._is_current_generation(env_id, generation):
            return
        payload: dict[str, Any] = {
            "phase": str(phase),
            "ts": time.perf_counter(),
            "generation": int(generation),
        }
        payload.update(extra)
        with self._worker_status_lock:
            self._worker_status[env_id] = payload

    def _log_event(self, payload: dict[str, Any]) -> None:
        if self._event_log_path is None:
            return
        try:
            self._event_log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            with self._event_log_lock:
                with self._event_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.write("\n")
        except Exception:
            return

    def _log_terminal_transition(
        self,
        *,
        env_id: int,
        generation: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        transition_info: dict[str, Any],
    ) -> None:
        bridge_info = transition_info.get("bridge_info") if isinstance(transition_info, dict) else None
        bridge_info = bridge_info if isinstance(bridge_info, dict) else {}
        bridge_action = bridge_info.get("action") if isinstance(bridge_info.get("action"), dict) else {}
        step_timing = bridge_info.get("step_timing_ms") if isinstance(bridge_info.get("step_timing_ms"), dict) else {}
        step_counts = bridge_info.get("step_timing_counts") if isinstance(bridge_info.get("step_timing_counts"), dict) else {}
        card_selection_before = (
            bridge_info.get("card_selection_before")
            if isinstance(bridge_info.get("card_selection_before"), dict)
            else {}
        )
        card_selection_after = (
            bridge_info.get("card_selection_after")
            if isinstance(bridge_info.get("card_selection_after"), dict)
            else {}
        )
        restart_count = int(self._worker_restart_count.get(env_id, 0))
        self._log_event(
            {
                "event": "episode_terminal",
                "timestamp_unix_s": time.time(),
                "env_id": int(env_id),
                "generation": int(generation),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "reward": float(reward),
                "encounter_id": transition_info.get("encounter_id"),
                "action_id": bridge_action.get("action_id"),
                "action_kind": bridge_action.get("kind"),
                "phase_before": bridge_info.get("phase_before"),
                "phase_after": bridge_info.get("phase_after"),
                "screen_before": bridge_info.get("screen_before"),
                "screen_after": bridge_info.get("screen_after"),
                "room_type_before": bridge_info.get("room_type_before"),
                "room_type_after": bridge_info.get("room_type_after"),
                "combat_in_progress_before": bridge_info.get("combat_in_progress_before"),
                "combat_in_progress_after": bridge_info.get("combat_in_progress_after"),
                "truncation_reason": bridge_info.get("truncation_reason"),
                "action_error": bridge_info.get("action_error"),
                "card_selection_screen_type_before": card_selection_before.get("screen_type"),
                "card_selection_screen_type_after": card_selection_after.get("screen_type"),
                "card_selection_selected_count_before": card_selection_before.get("selected_count"),
                "card_selection_selected_count_after": card_selection_after.get("selected_count"),
                "card_selection_confirm_ready_before": card_selection_before.get("confirm_ready"),
                "card_selection_confirm_ready_after": card_selection_after.get("confirm_ready"),
                "card_selection_selection_ready_before": card_selection_before.get("selection_ready"),
                "card_selection_selection_ready_after": card_selection_after.get("selection_ready"),
                "card_selection_opened_age_ms_before": card_selection_before.get("opened_age_ms"),
                "card_selection_opened_age_ms_after": card_selection_after.get("opened_age_ms"),
                "after_wait_ms": step_timing.get("after_wait"),
                "stable_iterations": step_counts.get("stable_iterations"),
                "worker_restart_happened": restart_count > 0,
                "worker_restart_count": restart_count,
            }
        )

    def _format_worker_status_summary(self) -> str:
        now = time.perf_counter()
        with self._worker_status_lock:
            statuses = {env_id: dict(status) for env_id, status in self._worker_status.items()}
        if not statuses:
            return "<no worker status>"
        parts: list[str] = []
        for env_id in sorted(statuses.keys()):
            status = statuses[env_id]
            phase = str(status.get("phase") or "unknown")
            age_s = max(now - float(status.get("ts") or now), 0.0)
            detail_parts = [f"phase={phase}", f"age_s={age_s:.1f}"]
            for key in ("generation", "restart_count", "initial_reset_ms", "post_done_reset_ms", "last_step_ms", "action", "error", "prewarmed", "restart_reason"):
                if key in status:
                    value = status[key]
                    if isinstance(value, float):
                        detail_parts.append(f"{key}={value:.1f}")
                    else:
                        detail_parts.append(f"{key}={value}")
            parts.append(f"env{env_id}[" + ", ".join(detail_parts) + "]")
        return "; ".join(parts)

    @staticmethod
    def _coerce_obs(obs: dict[str, Any] | None) -> dict[str, np.ndarray] | None:
        if obs is None:
            return None
        return {key: np.asarray(value) for key, value in (obs or {}).items()}

    def _is_current_generation_item(self, item: AsyncReadyItem) -> bool:
        return self._is_current_generation(item.env_id, item.generation)

    def _is_current_generation(self, env_id: int, generation: int) -> bool:
        return int(self._worker_generation.get(env_id, -1)) == int(generation)


__all__ = [
    "AsyncReadyCollector",
    "AsyncReadyItem",
    "PrewarmedEnvSlot",
    "resolve_action_mask",
    "stack_observations",
]
