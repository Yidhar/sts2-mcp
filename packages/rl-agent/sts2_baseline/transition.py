"""Small immutable transition and scalar-target contracts for the baseline.

The baseline intentionally has no objective-reward vector, settlement bonus,
guard-retarget flag, or backend reward escape hatch.  A reward is derived from
the transition by :mod:`sts2_baseline.reward`; replay stores that scalar and an
optional policy/value target only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

BASELINE_TRANSITION_VERSION = "sts2-baseline-transition-v1"
BASELINE_TARGET_VERSION = "sts2-baseline-target-v1"

TerminalResult = Literal["none", "win", "loss"]


def _finite(value: float, *, label: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _unit_interval(value: float, *, label: str) -> float:
    normalized = _finite(value, label=label)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return normalized


@dataclass(frozen=True, slots=True)
class PotentialState:
    """The normalized facts used by the fixed potential-based reward.

    Raw HP values are retained so the ratio calculation is auditable.  A
    ``0 / 0`` player pair represents an explicitly terminal observation whose
    transport no longer exposes a player snapshot; its HP ratio is zero.
    Enemy HP may be zero outside combat.  ``run_progress`` is a caller-defined,
    versioned fraction in ``[0, 1]`` (for example cleared floors / run cap); it
    must never be an unbounded floor counter.
    """

    player_hp: float
    player_max_hp: float
    enemy_hp: float = 0.0
    enemy_max_hp: float = 0.0
    run_progress: float = 0.0
    in_combat: bool = False

    def __post_init__(self) -> None:
        player_hp = _finite(self.player_hp, label="player_hp")
        player_max_hp = _finite(self.player_max_hp, label="player_max_hp")
        enemy_hp = _finite(self.enemy_hp, label="enemy_hp")
        enemy_max_hp = _finite(self.enemy_max_hp, label="enemy_max_hp")
        if player_max_hp < 0.0:
            raise ValueError("player_max_hp must be non-negative")
        if player_max_hp == 0.0 and player_hp != 0.0:
            raise ValueError("player_hp must be zero when player_max_hp is zero")
        if player_hp < 0.0 or enemy_hp < 0.0 or enemy_max_hp < 0.0:
            raise ValueError("HP values must be non-negative")
        if enemy_max_hp == 0.0 and enemy_hp > 0.0:
            raise ValueError("enemy_hp requires a positive enemy_max_hp")
        _unit_interval(self.run_progress, label="run_progress")

    @property
    def player_hp_ratio(self) -> float:
        if self.player_max_hp == 0.0:
            return 0.0
        return min(max(float(self.player_hp) / float(self.player_max_hp), 0.0), 1.0)

    @property
    def enemy_remaining_ratio(self) -> float:
        if not self.in_combat or self.enemy_max_hp <= 0.0:
            return 0.0
        return min(max(float(self.enemy_hp) / float(self.enemy_max_hp), 0.0), 1.0)

    @property
    def enemy_progress_ratio(self) -> float:
        if not self.in_combat or self.enemy_max_hp <= 0.0:
            return 0.0
        return 1.0 - self.enemy_remaining_ratio


@dataclass(frozen=True, slots=True)
class BaselineTransition:
    """One canonical environment transition for baseline learning."""

    episode_id: str
    step_index: int
    action_handle: str
    before: PotentialState
    after: PotentialState
    combat_result: TerminalResult = "none"
    run_result: TerminalResult = "none"
    truncated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    version: str = BASELINE_TRANSITION_VERSION

    def __post_init__(self) -> None:
        if not str(self.episode_id).strip():
            raise ValueError("episode_id must be non-empty")
        if int(self.step_index) < 0:
            raise ValueError("step_index must be non-negative")
        if not str(self.action_handle).strip():
            raise ValueError("action_handle must be non-empty")
        if self.combat_result not in {"none", "win", "loss"}:
            raise ValueError(f"invalid combat_result: {self.combat_result!r}")
        if self.run_result not in {"none", "win", "loss"}:
            raise ValueError(f"invalid run_result: {self.run_result!r}")
        if self.truncated and (self.combat_result == "win" or self.run_result == "win"):
            raise ValueError("a truncated transition cannot be marked as a win")
        if self.version != BASELINE_TRANSITION_VERSION:
            raise ValueError(f"unsupported baseline transition version: {self.version!r}")
        # Copy caller-owned metadata so later mutations of the input mapping do
        # not silently rewrite an already-recorded transition.  Keep the copy a
        # regular dict: replay/checkpoint payloads must remain pickleable.
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class BaselineTargets:
    """Only the scalar reward and return target stored beside a transition.

    The restarted baseline has no search policy, demonstration target, or
    legacy objective vector.  Its behavior action lives in the typed replay
    payload and policy gradients are computed directly by the learner.
    """

    reward: float
    value: float | None = None
    version: str = BASELINE_TARGET_VERSION

    def __post_init__(self) -> None:
        _finite(self.reward, label="reward")
        if self.value is not None:
            _finite(self.value, label="value")
        if self.version != BASELINE_TARGET_VERSION:
            raise ValueError(f"unsupported baseline target version: {self.version!r}")
