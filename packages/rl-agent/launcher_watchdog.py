"""Per-instance watchdog for launcher-managed STS2 game processes.

Primary signal: HTTP ``/health`` probe per instance. N consecutive failures
inside a hung-but-alive window mean the game is locked up (observed: particle
null error loops in events like ``PunchOff`` spam gigabytes into godot.log and
stall the main thread — Sentry crashpad never fires because the process
doesn't raise, so there is no minidump to analyse after the fact).

Secondary signal: godot logs directory growth rate. Independent sanity check
for the log-flood failure mode specifically — if the whole logs dir is
growing >N MB in a short window, at least one instance is in a bad-state loop
even if ``/health`` still happens to respond briefly.

The watchdog is intentionally a pure data/decision module. Actually killing
and respawning game processes lives in ``launcher.py`` so the fault path
stays in one place (process ownership is there too).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class InstanceAction(Enum):
    HEALTHY = "healthy"
    KILL_AND_RESTART = "kill_and_restart"


@dataclass
class InstanceDecision:
    instance_id: int
    action: InstanceAction
    reason: str = ""


@dataclass
class WatchdogDecision:
    per_instance: list[InstanceDecision]
    log_flood_detected: bool = False
    log_flood_bytes_per_sec: float = 0.0


@dataclass(frozen=True)
class HealthSnapshot:
    transport_alive: bool
    game_thread_alive: bool | None = None
    pump_tick: int | None = None
    ms_since_last_pump: int | None = None
    active_operation: str | None = None
    source: str = "legacy"

    @property
    def is_v2(self) -> bool:
        return self.source == "v2"


@dataclass(frozen=True)
class StateSnapshot:
    state_version: int
    screen: str
    stable_decision: bool = False


class _HealthProbe(Protocol):
    def __call__(
        self, base_url: str, token: str, timeout_s: float,
    ) -> HealthSnapshot | Mapping[str, Any] | bool: ...


class _StateVersionProbe(Protocol):
    """Returns (state_version, screen) from the bridge, or None if unreachable.

    Used to catch silent hangs: game main thread frozen at a scene transition
    so /health still returns 200 but no new game state is ever committed. The
    PunchOff / CORPSE_SLUGS-style room init deadlocks manifest this way.

    The screen label lets the watchdog distinguish "idle but alive" (main
    menu, character select, game over) from "should be making progress"
    (combat, map, event, rewards). Without this split, a fresh instance
    sitting at the title screen waiting for the trainer's reset() would be
    killed by state-stall detection.
    """

    def __call__(
        self, base_url: str, token: str, timeout_s: float,
    ) -> StateSnapshot | tuple[int, str] | None: ...


# Screens where the game is legitimately not progressing state on its own
# and the trainer is responsible for kicking it forward. The stall timer
# MUST be reset (not accumulated) while in these screens so that a slow
# trainer startup, a launcher-restart, or an end-of-run pause doesn't trip
# the silent-hang guard. Values here mirror the `screen` string returned
# by the bridge /state endpoint.
_DEFAULT_IDLE_SCREENS: frozenset[str] = frozenset({
    "MAIN_MENU",
    "STARTUP_CHARACTER_SELECT",
    "CHARACTER_SELECT",
    # 2026-05-08: bridge actually reports the long form (RUN_MODE_SELECTION,
    # see BridgeGameApi.cs `case NSingleplayerSubmenu`). The original
    # ``RUN_MODE_SELECT`` entry was a typo that left the screen NOT idle —
    # so when the trainer paused on this screen for gradient updates the
    # silent-hang guard would (incorrectly) restart the game after
    # state_stall_threshold_s. Keep both spellings to stay safe across
    # bridge versions.
    "RUN_MODE_SELECT",
    "RUN_MODE_SELECTION",
    "GAME_OVER",
    "INTRO",
    "LOADING",
    "UNKNOWN",  # bridge sometimes reports UNKNOWN during very early init
})


@dataclass
class _InstanceState:
    """Internal per-instance state retained across watchdog ticks."""
    strike_count: int = 0
    main_thread_strike_count: int = 0
    last_probe_unix_s: float = 0.0
    last_pump_tick: int | None = None
    last_state_version: int | None = None
    last_screen: str | None = None
    # Unix time when the (state_version, screen) pair was first observed
    # (i.e., when the current stall started). Stall is computed from this in
    # seconds, not ticks, so the threshold stays meaningful even when the
    # tick interval changes.
    last_state_change_unix_s: float = 0.0


def _default_logs_total_bytes(logs_dir: Path) -> int:
    if not logs_dir.exists() or not logs_dir.is_dir():
        return 0
    total = 0
    for entry in logs_dir.iterdir():
        if entry.is_file() and entry.name.endswith(".log"):
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def _normalize_health_snapshot(value: HealthSnapshot | Mapping[str, Any] | bool) -> HealthSnapshot:
    if isinstance(value, HealthSnapshot):
        return value
    if isinstance(value, Mapping):
        pump_tick = value.get("pump_tick")
        pump_age = value.get("ms_since_last_pump")
        return HealthSnapshot(
            transport_alive=bool(value.get("transport_alive", value.get("ok", False))),
            game_thread_alive=(
                bool(value.get("game_thread_alive"))
                if value.get("game_thread_alive") is not None
                else None
            ),
            pump_tick=(
                int(pump_tick)
                if isinstance(pump_tick, int) and not isinstance(pump_tick, bool)
                else None
            ),
            ms_since_last_pump=(
                int(pump_age)
                if isinstance(pump_age, int) and not isinstance(pump_age, bool)
                else None
            ),
            active_operation=(
                str(value.get("active_operation"))
                if value.get("active_operation") is not None
                else None
            ),
            source="v2" if "game_thread_alive" in value else "legacy",
        )
    return HealthSnapshot(transport_alive=bool(value), source="legacy")


def _normalize_state_snapshot(
    value: StateSnapshot | tuple[int, str] | None,
) -> StateSnapshot | None:
    if value is None or isinstance(value, StateSnapshot):
        return value
    version, screen = value
    return StateSnapshot(
        state_version=int(version),
        screen=str(screen or "UNKNOWN").strip().upper() or "UNKNOWN",
    )


def _default_health_probe(
    base_url: str, token: str, timeout_s: float,
) -> HealthSnapshot:
    """Prefer contract-v2 health, falling back only for old bridge builds."""
    import requests  # local import so lightweight tests can inject probes

    headers = {"Authorization": f"Bearer {token}"}
    v2_url = f"{base_url.rstrip('/')}/v2/health"
    try:
        response = requests.get(v2_url, headers=headers, timeout=timeout_s)
    except (requests.ConnectionError, requests.Timeout):
        return HealthSnapshot(transport_alive=False, source="v2")
    except Exception:
        return HealthSnapshot(transport_alive=False, source="v2")

    if 200 <= response.status_code < 300:
        try:
            payload = response.json()
        except Exception:
            return HealthSnapshot(transport_alive=False, source="v2")
        if not isinstance(payload, dict):
            return HealthSnapshot(transport_alive=False, source="v2")
        return _normalize_health_snapshot(payload)

    if response.status_code != 404:
        return HealthSnapshot(transport_alive=False, source="v2")

    # Compatibility only: old bridges expose transport health but no main-loop
    # heartbeat.  Never treat a malformed v2 response as permission to downgrade.
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/health",
            headers=headers,
            timeout=timeout_s,
        )
    except (requests.ConnectionError, requests.Timeout):
        return HealthSnapshot(transport_alive=False, source="legacy")
    except Exception:
        return HealthSnapshot(transport_alive=False, source="legacy")
    return HealthSnapshot(
        transport_alive=200 <= response.status_code < 300,
        source="legacy",
    )


def _default_state_version_probe(
    base_url: str, token: str, timeout_s: float,
) -> StateSnapshot | None:
    """Read state version plus whether the game awaits a trainer decision."""
    import requests
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/state",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_s,
        )
    except (requests.ConnectionError, requests.Timeout):
        return None
    except Exception:
        return None
    if not (200 <= response.status_code < 300):
        return None
    try:
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("state_version")
    if isinstance(version, bool) or not isinstance(version, int):
        return None
    screen = str(payload.get("screen") or "UNKNOWN").strip().upper() or "UNKNOWN"
    actions = payload.get("available_actions")
    if not isinstance(actions, list):
        actions = payload.get("legal_actions")
    actionability = payload.get("actionability")
    frontier_stable = (
        actionability.get("frontier_stable")
        if isinstance(actionability, dict)
        else None
    )
    stable_decision = bool(actions) and frontier_stable is not False
    return StateSnapshot(
        state_version=version,
        screen=screen,
        stable_decision=stable_decision,
    )


@dataclass
class Watchdog:
    """Per-instance health + aggregate log-flood watchdog.

    Construct once, then call ``evaluate`` on each launcher monitor tick.

    Parameters
    ----------
    strike_threshold:
        Consecutive ``/health`` failures before ordering kill+restart.
    probe_timeout_s:
        Per-probe HTTP timeout. Any single probe exceeding this counts as a
        failure — so a hung instance doesn't block the watchdog itself.
    log_flood_bytes_per_sec:
        Aggregate across all ``*.log`` files under ``logs_dir``. Above this
        rate the watchdog emits a log-flood signal (kill everything — we
        cannot reliably attribute writes to a single PID without symlink
        games, and a real flood at this magnitude will already be dragging
        all instances down via shared disk I/O).
    logs_dir:
        Godot logs directory; typically ``%APPDATA%/SlayTheSpire2/logs``.
    """

    strike_threshold: int = 3
    probe_timeout_s: float = 5.0
    log_flood_bytes_per_sec: float = 10.0 * 1024 * 1024  # 10 MB/s sustained
    # "Silent hang" detection: if /health returns OK but /state's state_version
    # never advances for this many seconds, the game main thread is frozen at
    # a scene init / asset load deadlock (the known PunchOff + CORPSE_SLUGS
    # pattern). 0 or negative disables. Default 90s is enough to distinguish
    # brief game-over idle from a real hang at the watchdog's tick cadence.
    state_stall_threshold_s: float = 90.0
    state_probe_timeout_s: float = 10.0
    # Independent strikes for a responsive HTTP worker whose game main loop is
    # unhealthy, pump-stale, or cannot produce /state.
    main_thread_strike_threshold: int | None = None
    max_pump_age_ms: int = 30_000
    # Screens where the game is legitimately idle (waiting for the trainer
    # to act) and state_version won't advance on its own. The stall timer
    # is reset, not accumulated, while the game is on any of these.
    idle_screens: frozenset[str] = field(default_factory=lambda: _DEFAULT_IDLE_SCREENS)
    logs_dir: Path | None = None
    health_probe: _HealthProbe = field(default=_default_health_probe)
    state_version_probe: _StateVersionProbe = field(default=_default_state_version_probe)
    logs_total_bytes_fn: Callable[[Path], int] = field(default=_default_logs_total_bytes)

    _instance_state: dict[int, _InstanceState] = field(default_factory=dict)
    _last_logs_total_bytes: int | None = field(default=None)
    _last_logs_sample_unix_s: float | None = field(default=None)

    def evaluate(
        self,
        instances: Iterable[WatchdogInstanceView],
        now_unix_s: float,
    ) -> WatchdogDecision:
        per_instance: list[InstanceDecision] = []
        for view in instances:
            state = self._instance_state.setdefault(view.instance_id, _InstanceState())
            if not view.process_alive:
                state.strike_count = 0
                state.main_thread_strike_count = 0
                state.last_pump_tick = None
                state.last_state_version = None
                state.last_screen = None
                state.last_state_change_unix_s = 0.0
                per_instance.append(
                    InstanceDecision(
                        view.instance_id,
                        InstanceAction.KILL_AND_RESTART,
                        reason="process_exited",
                    )
                )
                continue
            if view.base_url is None or view.token is None:
                per_instance.append(
                    InstanceDecision(
                        view.instance_id,
                        InstanceAction.HEALTHY,
                        reason="not_bound_yet",
                    )
                )
                continue

            try:
                raw_health = self.health_probe(
                    view.base_url,
                    view.token,
                    self.probe_timeout_s,
                )
                health = _normalize_health_snapshot(raw_health)
            except Exception:
                health = HealthSnapshot(transport_alive=False, source="v2")
            state.last_probe_unix_s = now_unix_s

            if not health.transport_alive:
                state.strike_count += 1
                if state.strike_count >= self.strike_threshold:
                    per_instance.append(
                        InstanceDecision(
                            view.instance_id,
                            InstanceAction.KILL_AND_RESTART,
                            reason=f"health_failed_{state.strike_count}_consecutive",
                        )
                    )
                    state.strike_count = 0
                    state.main_thread_strike_count = 0
                else:
                    per_instance.append(
                        InstanceDecision(
                            view.instance_id,
                            InstanceAction.HEALTHY,
                            reason=f"strike_{state.strike_count}",
                        )
                    )
                continue
            state.strike_count = 0

            probed_state: StateSnapshot | None = None
            if self.state_stall_threshold_s > 0.0:
                try:
                    probed_state = _normalize_state_snapshot(
                        self.state_version_probe(
                            view.base_url,
                            view.token,
                            self.state_probe_timeout_s,
                        )
                    )
                except Exception:
                    probed_state = None

            main_thread_fault: str | None = None
            if health.is_v2:
                if health.game_thread_alive is not True:
                    main_thread_fault = "game_thread_not_alive"
                elif (
                    health.ms_since_last_pump is not None
                    and health.ms_since_last_pump > self.max_pump_age_ms
                ):
                    main_thread_fault = (
                        f"pump_stale_{health.ms_since_last_pump}ms"
                    )
                elif (
                    state.last_pump_tick is not None
                    and health.pump_tick is not None
                    and health.pump_tick < state.last_pump_tick
                ):
                    main_thread_fault = (
                        f"pump_tick_regressed_{state.last_pump_tick}_to_{health.pump_tick}"
                    )
                if health.pump_tick is not None:
                    state.last_pump_tick = health.pump_tick
                # A healthy HTTP worker with no /state is not proof the game
                # thread is healthy; this is a main-thread strike, not merely a
                # dropped diagnostic sample.
                if probed_state is None and main_thread_fault is None:
                    main_thread_fault = "state_probe_failed"

            if main_thread_fault is not None:
                state.main_thread_strike_count += 1
                threshold = self.main_thread_strike_threshold or self.strike_threshold
                operation = health.active_operation or "idle"
                if state.main_thread_strike_count >= threshold:
                    per_instance.append(
                        InstanceDecision(
                            view.instance_id,
                            InstanceAction.KILL_AND_RESTART,
                            reason=(
                                f"main_thread_failed_{state.main_thread_strike_count}_consecutive_"
                                f"{main_thread_fault}_operation_{operation}"
                            ),
                        )
                    )
                    state.main_thread_strike_count = 0
                    state.last_state_version = None
                    state.last_state_change_unix_s = 0.0
                else:
                    per_instance.append(
                        InstanceDecision(
                            view.instance_id,
                            InstanceAction.HEALTHY,
                            reason=(
                                f"main_thread_strike_{state.main_thread_strike_count}_"
                                f"{main_thread_fault}_operation_{operation}"
                            ),
                        )
                    )
                continue

            state.main_thread_strike_count = 0
            if probed_state is not None:
                stall_decision = self._evaluate_state_stall(
                    view,
                    state,
                    now_unix_s,
                    probed_state,
                    active_operation=health.active_operation,
                )
                if stall_decision is not None:
                    per_instance.append(stall_decision)
                    if stall_decision.action is InstanceAction.KILL_AND_RESTART:
                        state.last_state_version = None
                        state.last_state_change_unix_s = 0.0
                    continue
            per_instance.append(
                InstanceDecision(
                    view.instance_id,
                    InstanceAction.HEALTHY,
                    reason="probe_ok",
                )
            )

        log_flood, rate = self._evaluate_log_flood(now_unix_s)
        return WatchdogDecision(
            per_instance=per_instance,
            log_flood_detected=log_flood,
            log_flood_bytes_per_sec=rate,
        )

    def _evaluate_state_stall(
        self,
        view: WatchdogInstanceView,
        state: _InstanceState,
        now_unix_s: float,
        probed: StateSnapshot,
        *,
        active_operation: str | None,
    ) -> InstanceDecision | None:
        """Check for silent hangs where /health is OK but game state is frozen.

        Idle screens (main menu, character select, game over, ...) are
        exempt: state_version legitimately doesn't advance there, and the
        stall bookkeeping is reset so freshly-launched or just-restarted
        instances don't get killed during the handshake-to-trainer window.

        Returns a decision to OVERRIDE the default probe_ok path when the
        caller should act on state stall, or None if stall detection is
        disabled / not yet conclusive (in which case the caller stays on its
        normal ``probe_ok`` HEALTHY branch).
        """
        if self.state_stall_threshold_s <= 0.0:
            return None
        current_version = probed.state_version
        current_screen = probed.screen

        # A stable actionable frontier with no mutation in flight is waiting
        # for the trainer by design. It is idle even on COMBAT/MAP/EVENT, so an
        # unchanged state_version must never be interpreted as a game hang.
        if probed.stable_decision and not active_operation:
            state.last_state_version = current_version
            state.last_screen = current_screen
            state.last_state_change_unix_s = now_unix_s
            return InstanceDecision(
                view.instance_id,
                InstanceAction.HEALTHY,
                reason=f"stable_decision_idle_{current_screen}",
            )

        # Idle screen: game is alive but not supposed to be progressing on
        # its own. Reset stall bookkeeping so the clock starts only when
        # gameplay actually resumes. This covers: initial launch waiting
        # for trainer, end-of-run GAME_OVER waiting for reset, transient
        # main-menu passes after a watchdog-triggered restart.
        if current_screen in self.idle_screens:
            state.last_state_version = current_version
            state.last_screen = current_screen
            state.last_state_change_unix_s = now_unix_s
            return InstanceDecision(
                view.instance_id, InstanceAction.HEALTHY,
                reason=f"idle_screen_{current_screen}",
            )

        # Screen transition counts as progress even if state_version
        # happens to match (e.g. bridge snapshots the same counter across
        # two screens during a transition tick).
        if (
            state.last_state_version is None
            or current_version != state.last_state_version
            or current_screen != state.last_screen
        ):
            state.last_state_version = current_version
            state.last_screen = current_screen
            state.last_state_change_unix_s = now_unix_s
            return None

        stall_s = now_unix_s - state.last_state_change_unix_s
        if stall_s >= self.state_stall_threshold_s:
            return InstanceDecision(
                view.instance_id, InstanceAction.KILL_AND_RESTART,
                reason=(
                    f"silent_hang_state_version_frozen_at_{current_version}_"
                    f"screen_{current_screen}_for_{stall_s:.0f}s"
                ),
            )
        # Stall is building but not yet over threshold — still HEALTHY for now.
        return InstanceDecision(
            view.instance_id, InstanceAction.HEALTHY,
            reason=f"state_stall_{current_screen}_{stall_s:.0f}s",
        )

    def _evaluate_log_flood(self, now_unix_s: float) -> tuple[bool, float]:
        if self.logs_dir is None:
            return False, 0.0
        try:
            total = self.logs_total_bytes_fn(self.logs_dir)
        except Exception:
            return False, 0.0
        if self._last_logs_total_bytes is None or self._last_logs_sample_unix_s is None:
            self._last_logs_total_bytes = total
            self._last_logs_sample_unix_s = now_unix_s
            return False, 0.0
        dt = max(now_unix_s - self._last_logs_sample_unix_s, 1e-6)
        delta = max(total - self._last_logs_total_bytes, 0)
        rate = delta / dt
        self._last_logs_total_bytes = total
        self._last_logs_sample_unix_s = now_unix_s
        return rate >= self.log_flood_bytes_per_sec, rate


@dataclass
class WatchdogInstanceView:
    """Read-only slice of an instance the watchdog needs to reason about."""
    instance_id: int
    process_alive: bool
    base_url: str | None
    token: str | None


__all__ = [
    "InstanceAction",
    "InstanceDecision",
    "Watchdog",
    "WatchdogDecision",
    "WatchdogInstanceView",
]
