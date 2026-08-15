"""Absolute revival-price controller for hybrid Candidate-Q reward v8."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

DEATH_PRICE_CONTROLLER_VERSION: Final = "sts2-absolute-death-price-controller-v1"
DEATH_PRICE_COURSE_VERSION: Final = "sts2-absolute-death-price-courses-v1"


@dataclass(frozen=True, slots=True)
class DeathPriceCourse:
    revival_budget: int
    act1_target: float
    run_target: float
    bootstrap_lambda: float
    lambda_max: float
    eta: float
    support_window: int = 50
    minimum_successes: int = 8
    heldout_floor_fraction: float = 0.5
    ema_alpha: float = 0.25
    discounted_margin_safety: float = 1.25
    version: str = DEATH_PRICE_COURSE_VERSION

    def __post_init__(self) -> None:
        if self.revival_budget not in {0, 2, 4, 8, 16, 32, 64}:
            raise ValueError("unsupported absolute-death-price revival budget")
        if min(self.act1_target, self.run_target, self.bootstrap_lambda) < 0.0:
            raise ValueError("death-price targets and bootstrap lambda cannot be negative")
        if self.lambda_max < self.bootstrap_lambda:
            raise ValueError("death-price lambda maximum is below its bootstrap")
        if self.revival_budget and self.eta <= 0.0:
            raise ValueError("active death-price course eta must be positive")
        if self.support_window <= 0 or not 0 < self.minimum_successes <= self.support_window:
            raise ValueError("death-price support window is invalid")
        if not 0.0 < self.heldout_floor_fraction <= 1.0:
            raise ValueError("held-out floor fraction must be in (0, 1]")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("death-price EMA alpha must be in (0, 1]")
        if self.discounted_margin_safety <= 1.0:
            raise ValueError("discounted margin safety must exceed one")


_COURSES: Final[dict[int, DeathPriceCourse]] = {
    64: DeathPriceCourse(64, 6.0, 32.0, 0.00625, 0.030, 0.00025),
    32: DeathPriceCourse(32, 4.0, 16.0, 0.030, 0.040, 0.00025),
    16: DeathPriceCourse(16, 2.0, 8.0, 0.040, 0.055, 0.00025),
    8: DeathPriceCourse(8, 1.0, 4.0, 0.055, 0.075, 0.00020),
    4: DeathPriceCourse(4, 1.0, 2.0, 0.075, 0.100, 0.00015),
    2: DeathPriceCourse(2, 0.0, 1.0, 0.100, 0.140, 0.00010),
    0: DeathPriceCourse(0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


def death_price_course(revival_budget: int) -> DeathPriceCourse:
    if isinstance(revival_budget, bool) or not isinstance(revival_budget, int):
        raise TypeError("revival budget must be an integer")
    try:
        return _COURSES[revival_budget]
    except KeyError as error:
        raise ValueError(f"no versioned absolute-death-price course for budget {revival_budget}") from error


def validate_discounted_outcome_margin(
    course: DeathPriceCourse,
    *,
    durable_floor_discount: float = 0.997,
    durable_floor_span: int = 59,
    terminal_margin: float = 4.0,
    hp_loss_span: float = 0.55,
    pace_span: float = 0.04,
) -> tuple[float, float]:
    """Return (discounted margin, required margin) or reject the course."""

    discounted = terminal_margin * durable_floor_discount**durable_floor_span
    secondary = course.revival_budget * course.lambda_max + hp_loss_span + pace_span
    required = course.discounted_margin_safety * secondary
    if discounted < required:
        raise ValueError(
            "death-price course violates discounted outcome-first ordering: " f"{discounted:.6f} < {required:.6f}"
        )
    return float(discounted), float(required)


@dataclass(frozen=True, slots=True)
class DeathPriceEpisodeOutcome:
    act1_cleared: bool
    run_success: bool
    act1_revivals: int | None
    run_revivals: int

    def __post_init__(self) -> None:
        if self.act1_revivals is not None and (
            isinstance(self.act1_revivals, bool) or not isinstance(self.act1_revivals, int) or self.act1_revivals < 0
        ):
            raise ValueError("act1 revivals must be non-negative or absent")
        if isinstance(self.run_revivals, bool) or not isinstance(self.run_revivals, int) or self.run_revivals < 0:
            raise ValueError("run revivals must be non-negative")
        if self.act1_cleared and self.act1_revivals is None:
            raise ValueError("cleared Act 1 requires its factual revival count")
        if self.run_success and not self.act1_cleared:
            raise ValueError("run success without Act-1 clear is contradictory")


@dataclass(slots=True)
class DeathPriceMetrics:
    lambda_value: float
    active_statistic: Literal["act1", "run"] = "act1"
    act1_ema: float | None = None
    run_ema: float | None = None
    completed_blocks: int = 0
    act1_target_streak: int = 0
    last_support_successes: int = 0
    last_run_successes: int = 0
    last_increment: float = 0.0
    freeze_reason: str = "awaiting_support_window"
    baseline_act1_rate: float | None = None
    baseline_run_rate: float | None = None
    latest_act1_rate: float | None = None
    latest_run_rate: float | None = None
    heldout_policy_version: int | None = None
    last_control_gate_policy_version: int | None = None

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)


class AbsoluteDeathPriceController:
    """Monotone lambda ratchet driven by successful factual outcomes."""

    def __init__(
        self,
        course: DeathPriceCourse,
        *,
        initial_lambda: float | None = None,
    ) -> None:
        self.course = course
        value = course.bootstrap_lambda if initial_lambda is None else float(initial_lambda)
        if not math.isfinite(value) or not 0.0 <= value <= course.lambda_max:
            raise ValueError("initial death price is outside its course")
        self.metrics = DeathPriceMetrics(lambda_value=value)
        self._window: list[DeathPriceEpisodeOutcome] = []

    @staticmethod
    def _ema(previous: float | None, observed: float, alpha: float) -> float:
        return observed if previous is None else alpha * observed + (1.0 - alpha) * previous

    @property
    def lambda_value(self) -> float:
        return self.metrics.lambda_value

    def observe_heldout(
        self,
        *,
        act1_rate: float,
        run_rate: float,
        policy_version: int,
    ) -> None:
        for label, value in (("act1", act1_rate), ("run", run_rate)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"held-out {label} rate must be in [0, 1]")
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise ValueError("held-out policy version must be non-negative")
        if self.metrics.baseline_act1_rate is None and policy_version == 0:
            self.metrics.baseline_act1_rate = float(act1_rate)
            self.metrics.baseline_run_rate = float(run_rate)
        self.metrics.latest_act1_rate = float(act1_rate)
        self.metrics.latest_run_rate = float(run_rate)
        self.metrics.heldout_policy_version = int(policy_version)

    def observe_episode(self, outcome: DeathPriceEpisodeOutcome) -> bool:
        """Consume one run; return True when a complete control block closes."""

        self._window.append(outcome)
        if len(self._window) < self.course.support_window:
            return False
        block = tuple(self._window[: self.course.support_window])
        del self._window[: self.course.support_window]
        self._finish_block(block)
        return True

    def _finish_block(self, block: tuple[DeathPriceEpisodeOutcome, ...]) -> None:
        act1_values = [
            float(item.act1_revivals) for item in block if item.act1_cleared and item.act1_revivals is not None
        ]
        run_values = [float(item.run_revivals) for item in block if item.run_success]
        if act1_values:
            self.metrics.act1_ema = self._ema(
                self.metrics.act1_ema,
                sum(act1_values) / len(act1_values),
                self.course.ema_alpha,
            )
        if run_values:
            self.metrics.run_ema = self._ema(
                self.metrics.run_ema,
                sum(run_values) / len(run_values),
                self.course.ema_alpha,
            )
        self.metrics.completed_blocks += 1
        self.metrics.last_run_successes = len(run_values)

        if self.metrics.active_statistic == "act1":
            supported = len(act1_values)
            active_ema = self.metrics.act1_ema
            target = self.course.act1_target
            latest_rate = self.metrics.latest_act1_rate
            baseline_rate = self.metrics.baseline_act1_rate
        else:
            supported = len(run_values)
            active_ema = self.metrics.run_ema
            target = self.course.run_target
            latest_rate = self.metrics.latest_run_rate
            baseline_rate = self.metrics.baseline_run_rate
        self.metrics.last_support_successes = supported
        self.metrics.last_increment = 0.0

        if self.course.revival_budget == 0:
            self.metrics.freeze_reason = "zero_revival_course"
            return
        if supported < self.course.minimum_successes or active_ema is None:
            self.metrics.freeze_reason = "insufficient_success_support"
            return
        if latest_rate is None or baseline_rate is None:
            self.metrics.freeze_reason = "missing_heldout_gate"
            return
        heldout_policy_version = self.metrics.heldout_policy_version
        if heldout_policy_version is None:
            self.metrics.freeze_reason = "missing_heldout_gate"
            return
        if (
            self.metrics.last_control_gate_policy_version is not None
            and heldout_policy_version <= self.metrics.last_control_gate_policy_version
        ):
            self.metrics.freeze_reason = "stale_heldout_gate"
            return
        self.metrics.last_control_gate_policy_version = heldout_policy_version
        if latest_rate < self.course.heldout_floor_fraction * baseline_rate:
            self.metrics.freeze_reason = "heldout_below_course_floor"
            return

        excess = max(0.0, active_ema - target)
        increment = self.course.eta * excess
        previous = self.metrics.lambda_value
        self.metrics.lambda_value = min(self.course.lambda_max, previous + increment)
        self.metrics.last_increment = self.metrics.lambda_value - previous
        self.metrics.freeze_reason = "at_or_below_target" if excess == 0.0 else "none"

        if self.metrics.active_statistic == "act1":
            if active_ema <= self.course.act1_target:
                self.metrics.act1_target_streak += 1
            else:
                self.metrics.act1_target_streak = 0
            if self.metrics.act1_target_streak >= 2 and len(run_values) >= self.course.minimum_successes:
                self.metrics.active_statistic = "run"
                self.metrics.freeze_reason = "switched_to_run_statistic"

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": DEATH_PRICE_CONTROLLER_VERSION,
            "course": asdict(self.course),
            "metrics": self.metrics.as_mapping(),
            "window": tuple(asdict(item) for item in self._window),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != DEATH_PRICE_CONTROLLER_VERSION:
            raise ValueError("death-price controller version differs")
        if state.get("course") != asdict(self.course):
            raise ValueError("death-price course differs")
        self.metrics = DeathPriceMetrics(**dict(state["metrics"]))
        if not 0.0 <= self.metrics.lambda_value <= self.course.lambda_max:
            raise ValueError("restored death price is outside its course")
        self._window = [DeathPriceEpisodeOutcome(**dict(item)) for item in state.get("window", ())]
        if len(self._window) >= self.course.support_window:
            raise ValueError("restored death-price window should have been reduced")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "version": DEATH_PRICE_CONTROLLER_VERSION,
            "course": asdict(self.course),
            **self.metrics.as_mapping(),
            "pending_window_runs": len(self._window),
        }


__all__ = [
    "DEATH_PRICE_CONTROLLER_VERSION",
    "DEATH_PRICE_COURSE_VERSION",
    "AbsoluteDeathPriceController",
    "DeathPriceCourse",
    "DeathPriceEpisodeOutcome",
    "DeathPriceMetrics",
    "death_price_course",
    "validate_discounted_outcome_margin",
]
