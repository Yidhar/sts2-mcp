"""Process-wide combat curriculum outcome tracker."""

from __future__ import annotations

import numpy as np

from .reward_constants import (
    CURRICULUM_HP_WEIGHT_LERP_MAX_TIER,
    CURRICULUM_HP_WEIGHT_LERP_MIN_TIER,
    CURRICULUM_MIN_EPISODES_FOR_PHASE,
    CURRICULUM_PHASE_WIN_RATE_THRESHOLDS,
    CURRICULUM_WIN_RATE_WINDOW,
)


class CurriculumTracker:
    """Sliding per-encounter win/loss tracker with phase classification."""

    __slots__ = ("_outcomes", "_phase_cache", "_window")

    def __init__(self, window: int = CURRICULUM_WIN_RATE_WINDOW) -> None:
        self._window = max(4, int(window))
        # encounter_id (str) -> list[int] of 0/1 outcomes, newest at end
        self._outcomes: dict[str, list[int]] = {}
        # encounter_id -> last reported phase (0..3).  Used to emit a one-line
        # "[curriculum] phase N→M" switch annotation when the phase changes.
        self._phase_cache: dict[str, int] = {}

    def record(self, encounter_id: str, win: bool) -> None:
        key = str(encounter_id or "unknown").strip().lower() or "unknown"
        bucket = self._outcomes.setdefault(key, [])
        bucket.append(1 if win else 0)
        if len(bucket) > self._window:
            del bucket[: len(bucket) - self._window]

    def win_rate(self, encounter_id: str) -> tuple[float, int]:
        key = str(encounter_id or "unknown").strip().lower() or "unknown"
        bucket = self._outcomes.get(key, [])
        if not bucket:
            return 0.0, 0
        return float(sum(bucket)) / float(len(bucket)), len(bucket)

    @staticmethod
    def _phase_for_win_rate(win_rate: float) -> int:
        p0_p1, p1_p2, p2_p3 = CURRICULUM_PHASE_WIN_RATE_THRESHOLDS
        if win_rate < p0_p1:
            return 0
        if win_rate < p1_p2:
            return 1
        if win_rate < p2_p3:
            return 2
        return 3

    def phase(self, encounter_id: str) -> int:
        wr, n = self.win_rate(encounter_id)
        if n < CURRICULUM_MIN_EPISODES_FOR_PHASE:
            return 0
        return self._phase_for_win_rate(wr)

    def progress(self, encounter_id: str) -> float:
        """0.0 at win_rate<=30%, 1.0 at win_rate>=80%, linear in between."""
        wr, n = self.win_rate(encounter_id)
        if n < CURRICULUM_MIN_EPISODES_FOR_PHASE:
            return 0.0
        p0_p1, _, p2_p3 = CURRICULUM_PHASE_WIN_RATE_THRESHOLDS
        span = max(1e-6, p2_p3 - p0_p1)
        return float(np.clip((wr - p0_p1) / span, 0.0, 1.0))

    def hp_weight(self, encounter_id: str, tier: str) -> float:
        """Current tier-aware HP-loss shaping multiplier for this encounter.

        Returns the lerped weight that should be MULTIPLIED with
        PLAYER_HP_LOSS_TIER_SCALE[tier] to get the effective scale.
        """
        prog = self.progress(encounter_id)
        tier_key = tier if tier in CURRICULUM_HP_WEIGHT_LERP_MIN_TIER else "unknown"
        lo = float(CURRICULUM_HP_WEIGHT_LERP_MIN_TIER[tier_key])
        hi = float(CURRICULUM_HP_WEIGHT_LERP_MAX_TIER[tier_key])
        return lo + prog * (hi - lo)

    def check_phase_switch(self, encounter_id: str) -> tuple[int, int] | None:
        """Return (old_phase, new_phase) if a phase boundary was just crossed."""
        key = str(encounter_id or "unknown").strip().lower() or "unknown"
        new_phase = self.phase(key)
        old_phase = self._phase_cache.get(key, -1)
        if new_phase != old_phase:
            self._phase_cache[key] = new_phase
            if old_phase >= 0:
                return (old_phase, new_phase)
        return None

    def dump_all_state(self) -> str:
        """Render every tracked encounter's current (tier, n, wr, phase) as
        one line per encounter, sorted by tier-then-name.

        Used for the periodic `[curriculum/state]` snapshot so the operator
        can see encounters that haven't yet crossed a phase boundary (which
        would normally never appear in the per-transition log).
        """
        def _tier_for(enc: str) -> str:
            e = enc.lower()
            if "boss" in e:
                return "boss"
            if "elite" in e:
                return "elite"
            if "weak" in e:
                return "weak"
            return "normal"

        # Order: boss > elite > normal > weak so heaviest tiers print first.
        tier_order = {"boss": 0, "elite": 1, "normal": 2, "weak": 3, "unknown": 4}
        rows: list[tuple[int, str, str, int, float, int]] = []
        for encounter, bucket in self._outcomes.items():
            tier = _tier_for(encounter)
            n = len(bucket)
            wr = float(sum(bucket)) / float(n) if n > 0 else 0.0
            phase = self._phase_for_win_rate(wr) if n >= CURRICULUM_MIN_EPISODES_FOR_PHASE else 0
            rows.append((tier_order.get(tier, 4), encounter, tier, n, wr, phase))
        rows.sort()
        lines = [
            f"  {tier:>6} {enc} n={n} wr={wr:.3f} phase=P{phase}"
            for _, enc, tier, n, wr, phase in rows
        ]
        return "\n".join(lines) if lines else "  (no encounters tracked)"


_CURRICULUM_TRACKER_SINGLETON: CurriculumTracker | None = None


def get_curriculum_tracker() -> CurriculumTracker:
    """Process-wide shared CurriculumTracker so all envs contribute to the same window."""
    global _CURRICULUM_TRACKER_SINGLETON
    if _CURRICULUM_TRACKER_SINGLETON is None:
        _CURRICULUM_TRACKER_SINGLETON = CurriculumTracker()
    return _CURRICULUM_TRACKER_SINGLETON
