"""Full-run action rules, timeouts, and scalar/act-id parsing helpers."""

from __future__ import annotations

from typing import Any

INVALID_ACTION_REASON = "invalid_action_index"
# These bridge actions are UI/automation controls, not choices the RL policy
# should learn or dispatch.  In particular, when the bridge is in a transient
# combat/window state it may expose only ``automation:unsupported_external_control``; treating
# that as a legal action lets training either stall on a non-game control or
# dispatch an unsupported external control.
#
# ``discard_potion`` is intentionally *not* in this hard-block list.  The live
# game can enter a forced potion-overflow modal where the only forward action is
# discarding a potion.  If EnvV2 filters that singleton away, the collector
# fabricates a terminal episode on floor 8/9 instead of continuing the run.  We
# still drop discard_potion when real game actions are also visible; only
# singleton/forced discard is allowed through as cleanup.
BLOCKED_ACTION_KINDS = {"automation"}
DISCARD_POTION_ACTION_KIND = "discard_potion"
EMPTY_POTION_NAMES = {"", "[empty]", "empty", "none", "null"}
RECOVERY_POLL_INTERVAL_S = 0.10
RECOVERY_MAX_WAIT_MS = 15_000
TRANSITION_RECOVERY_MAX_WAIT_MS = 60_000
RESET_READY_POLL_INTERVAL_S = 0.50
RESET_READY_MAX_WAIT_MS = 90_000
STEP_TRANSITION_RECOVERY_MAX_WAIT_MS = 5_000
ACTIONABILITY_FAST_WAIT_MS = 150
ACTIONABILITY_FAST_POLL_INTERVAL_S = 0.02
ACTIONABILITY_REBIND_TIMEOUT_MS = 1_000
STEP_RECOVERY_TRUNCATION_REASON = "bridge_episode_lost"
STARTUP_ACTION_PREFIXES = ("main_menu:", "run_mode:", "character_select:")
EVENT_COMBAT_LOW_HP_THRESHOLD = 0.70
EVENT_HP_LOSS_LOW_HP_THRESHOLD = 0.70


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_ACT_NAME_FRAGMENTS: tuple[tuple[str, float], ...] = (
    ("UNDERDOCKS", 1.0),
    ("UNDERDOCK", 1.0),
    ("ACT_ONE", 1.0),
    ("ACT_1", 1.0),
    ("ACT1", 1.0),
    ("HIVE", 2.0),
    ("ACT_TWO", 2.0),
    ("ACT_2", 2.0),
    ("ACT2", 2.0),
    ("GLORY", 3.0),
    ("ACT_THREE", 3.0),
    ("ACT_3", 3.0),
    ("ACT3", 3.0),
)


def _act_id_from_value(value: Any) -> float:
    """Parse STS2 act ids from both numeric and enum/name bridge payloads.

    Live bridge payloads have used strings like ``ACT.UNDERDOCKS`` instead
    of numeric ``1``.  Plain ``float(value)`` collapses those to 0, which
    makes downstream Act1-clear telemetry permanently false.  Keep this
    helper conservative: return 0 when unknown, but recognize the known act
    enum names plus strings carrying a trailing digit such as ``ACT.1``.
    """
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    for char in reversed(text):
        if char.isdigit():
            return float(char)
    upper = text.upper()
    for fragment, act_id in _ACT_NAME_FRAGMENTS:
        if fragment in upper:
            return float(act_id)
    return 0.0


def _parse_act_id(value: Any, run: dict[str, Any] | None = None) -> float:
    parsed = _act_id_from_value(value)
    if parsed > 0.0:
        return parsed
    if not isinstance(run, dict):
        return 0.0
    for key in ("act_id", "act_id_raw", "act", "act_number", "current_act", "act_name"):
        parsed = _act_id_from_value(run.get(key))
        if parsed > 0.0:
            return parsed
    for key in ("current_act_index", "act_index"):
        if key not in run:
            continue
        index = _float(run.get(key), default=-1.0)
        if index >= 0.0:
            # Godot / sim internals commonly store zero-based act indexes.
            return float(index + 1.0)
    return 0.0
