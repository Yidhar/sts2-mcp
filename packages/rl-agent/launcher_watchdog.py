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

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Protocol


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


class _HealthProbe(Protocol):
    def __call__(self, base_url: str, token: str, timeout_s: float) -> bool: ...


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
    ) -> tuple[int, str] | None: ...


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
    "RUN_MODE_SELECT",
    "GAME_OVER",
    "INTRO",
    "LOADING",
    "UNKNOWN",  # bridge sometimes reports UNKNOWN during very early init
})


@dataclass
class _InstanceState:
    """Internal per-instance state retained across watchdog ticks."""
    strike_count: int = 0
    last_probe_unix_s: float = 0.0
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


def _default_health_probe(base_url: str, token: str, timeout_s: float) -> bool:
    """Default HTTP /health probe. Return True on 200, False on any failure.

    Imported lazily — tests inject their own probe without pulling `requests`.
    """
    import requests  # local import so the tests don't need the dependency
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/health",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_s,
        )
    except (requests.ConnectionError, requests.Timeout):
        return False
    except Exception:
        return False
    return 200 <= resp.status_code < 300


def _default_state_version_probe(
    base_url: str, token: str, timeout_s: float,
) -> tuple[int, str] | None:
    """Default HTTP /state probe. Returns (state_version, screen) or None on
    any failure (unreachable, malformed payload).
    """
    import requests
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/state",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_s,
        )
    except (requests.ConnectionError, requests.Timeout):
        return None
    except Exception:
        return None
    if not (200 <= resp.status_code < 300):
        return None
    try:
        payload = resp.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    version = payload.get("state_version")
    if isinstance(version, bool) or not isinstance(version, int):
        return None
    screen = str(payload.get("screen") or "UNKNOWN").strip().upper() or "UNKNOWN"
    return version, screen


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
        instances: Iterable["WatchdogInstanceView"],
        now_unix_s: float,
    ) -> WatchdogDecision:
        per_instance: list[InstanceDecision] = []
        for view in instances:
            state = self._instance_state.setdefault(view.instance_id, _InstanceState())
            if not view.process_alive:
                # Process has exited (crashed, killed by OS, killed by the
                # user from the task manager, etc). The launcher's main
                # loop only restarts instances when watchdog emits
                # KILL_AND_RESTART — it has no separate dead-process
                # reaper — so we must emit the restart signal here or the
                # slot stays permanently dead.
                state.strike_count = 0
                state.last_state_version = None
                state.last_screen = None
                state.last_state_change_unix_s = 0.0
                per_instance.append(
                    InstanceDecision(
                        view.instance_id, InstanceAction.KILL_AND_RESTART,
                        reason="process_exited",
                    )
                )
                continue
            if view.base_url is None or view.token is None:
                # Not yet bound to bridge — watchdog abstains; launcher's
                # own readiness gate owns the not-yet-started case.
                per_instance.append(
                    InstanceDecision(
                        view.instance_id, InstanceAction.HEALTHY,
                        reason="not_bound_yet",
                    )
                )
                continue

            ok = self.health_probe(view.base_url, view.token, self.probe_timeout_s)
            state.last_probe_unix_s = now_unix_s
            if ok:
                state.strike_count = 0
                stall_decision = self._evaluate_state_stall(view, state, now_unix_s)
                if stall_decision is not None:
                    per_instance.append(stall_decision)
                    if stall_decision.action is InstanceAction.KILL_AND_RESTART:
                        # Reset the stall bookkeeping so the launcher's
                        # mid-restart window doesn't immediately re-trigger.
                        state.last_state_version = None
                        state.last_state_change_unix_s = 0.0
                    continue
                per_instance.append(
                    InstanceDecision(
                        view.instance_id, InstanceAction.HEALTHY,
                        reason="probe_ok",
                    )
                )
                continue
            state.strike_count += 1
            if state.strike_count >= self.strike_threshold:
                per_instance.append(
                    InstanceDecision(
                        view.instance_id, InstanceAction.KILL_AND_RESTART,
                        reason=f"health_failed_{state.strike_count}_consecutive",
                    )
                )
                # Reset so we don't re-flag the same instance every tick
                # while the launcher is mid-restart.
                state.strike_count = 0
            else:
                per_instance.append(
                    InstanceDecision(
                        view.instance_id, InstanceAction.HEALTHY,
                        reason=f"strike_{state.strike_count}",
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
        view: "WatchdogInstanceView",
        state: _InstanceState,
        now_unix_s: float,
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
        try:
            probed = self.state_version_probe(
                view.base_url or "", view.token or "", self.state_probe_timeout_s,
            )
        except Exception:
            probed = None
        if probed is None:
            # /state transiently unreachable while /health is OK — don't
            # penalize; wait for next tick. Leave bookkeeping intact.
            return None
        current_version, current_screen = probed

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
    "Watchdog",
    "WatchdogDecision",
    "WatchdogInstanceView",
    "InstanceAction",
    "InstanceDecision",
]
