"""Macro collection authority: isolated stage-two action ownership.

During stage-two collection the frozen combat champion drives every decision
the authority declines.  At recognized macro surfaces the authority owns the
choice: branch-balanced epsilon-greedy over the macro candidate-Q values,
with composite flows decomposed into two native decisions — the branch entry
at the parent surface and the target at the picker — linked by the durable
floor clock (Gamma = 1 within a floor), while confirm traffic is dispatched
mechanically and deselect/cancel are never emitted (monotone contract).

The authority records macro transitions as it acts; the legacy learner never
sees them and the macro learner never sees anything else.  Every override it
returns is a legal native candidate index — legality authority stays with
the engine.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor

from sts2_rl.encoding import EncodedDecisionSnapshot
from sts2_rl.semantics.clock import decision_discount
from sts2_rl.semantics.forward import ForwardDecision, forward_decision

from .transitions import MacroEpisode, MacroStep

MACRO_AUTHORITY_VERSION: Final = "sts2-macro-authority-v1"

_PICKER_SELECT_KINDS: Final[frozenset[str]] = frozenset(
    {"select_card", "select_card_option"}
)
_PICKER_CONFIRM_KINDS: Final[frozenset[str]] = frozenset(
    {"confirm_selection"}
)


def _kind(action: Mapping[str, Any]) -> str:
    return str(action.get("kind") or action.get("action") or "").strip().lower()


def _card_matches(action: Mapping[str, Any], target: Mapping[str, Any] | None) -> bool:
    if target is None:
        return True
    card = action.get("card")
    if not isinstance(card, Mapping):
        return False
    for key in ("id", "instance_id"):
        expected = target.get(key)
        if expected is not None and card.get(key) != expected:
            return False
    return True


@dataclass(slots=True)
class _OpenTransition:
    snapshot: EncodedDecisionSnapshot
    action_index: int
    surface: str
    branch: str
    target_key: str | None
    floor: int
    behavior_epsilon: float
    reward_accumulator: float = 0.0


class MacroCollectionAuthority:
    """Owns macro decisions during isolated stage-two collection."""

    def __init__(
        self,
        *,
        forward_q: Callable[[EncodedDecisionSnapshot, Any], tuple[Tensor, Any]],
        initial_state: Callable[[], Any],
        epsilon: float = 0.1,
        seed: int = 0,
        evaluation_ownership: bool = False,
    ) -> None:
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("authority epsilon must be in [0, 1]")
        self.forward_q = forward_q
        self.initial_state = initial_state
        self.epsilon = epsilon
        # Stage-3 joined evaluation: macro ownership extends into
        # deterministic held-out collection (whole-segment ownership, §10).
        self.evaluation_ownership = bool(evaluation_ownership)
        self._rng = np.random.default_rng(seed)
        self._hidden: Any = None
        self._episode_id: str | None = None
        self._steps: list[MacroStep] = []
        self._open: _OpenTransition | None = None
        self._pending_target: Mapping[str, Any] | None = None
        self._await_picker = False
        self._await_confirm = False
        self._last_floor = 0
        self.overrides = 0
        self.mechanical_dispatches = 0
        self.declined = 0
        self._auto_counter = 0

    # ------------------------------------------------------------------ episode
    def begin_episode(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self._hidden = self.initial_state()
        self._steps = []
        self._open = None
        self._pending_target = None
        self._await_picker = False
        self._await_confirm = False
        self._last_floor = 0
        self.overrides = 0
        self.mechanical_dispatches = 0
        self.declined = 0

    def _ensure_episode(self) -> None:
        if self._episode_id is None:
            self._auto_counter += 1
            self.begin_episode(f"macro-auto-{self._auto_counter}")

    def observe_step(self, *, reward: float, floor: int | None, terminal: bool) -> None:
        """Accumulate factual reward between macro decisions."""

        self._ensure_episode()

        if self._open is not None and math.isfinite(reward):
            self._open.reward_accumulator += float(reward)
        if floor is not None and floor >= 0:
            self._last_floor = int(floor)
        if terminal:
            self._close_open(terminal=True)

    def finish_episode(self, episode_id: str | None = None) -> MacroEpisode | None:
        self._close_open(terminal=True)
        if not self._steps:
            self._episode_id = None
            return None
        episode = MacroEpisode(
            episode_id=episode_id or self._episode_id or "macro-episode",
            steps=tuple(self._steps),
        )
        self._episode_id = None
        self._steps = []
        return episode

    # ------------------------------------------------------------------ choice
    def choose(
        self,
        *,
        observation: Mapping[str, Any],
        semantic_actions: Sequence[Mapping[str, Any]],
        snapshot: EncodedDecisionSnapshot,
        valid: np.typing.NDArray[np.bool_],
    ) -> int | None:
        """Return a native candidate index to execute, or None to decline."""

        self._ensure_episode()
        # Collector candidates arrive as strict-group wrappers; the native
        # action facts live under their ``prototype`` key.
        semantic_actions = [
            action["prototype"]
            if isinstance(action.get("prototype"), Mapping)
            else action
            for action in semantic_actions
        ]
        floor = self._observed_floor(observation)
        if self._await_confirm:
            index = self._match_index(
                semantic_actions, valid, kinds=_PICKER_CONFIRM_KINDS, target=None
            )
            if index is not None:
                self._await_confirm = False
                self.mechanical_dispatches += 1
                return index
            self._await_confirm = False  # surface changed: fail closed to champion
            self.declined += 1
            return None
        if self._await_picker:
            self._await_picker = False
            picker = self._picker_indices(semantic_actions, valid)
            if picker:
                self._close_open(terminal=False, floor=floor)
                chosen = self._q_choice(snapshot, picker)
                self._open_transition(
                    snapshot=snapshot,
                    action_index=chosen,
                    surface="picker",
                    branch="target",
                    target_key=None,
                    floor=floor,
                )
                self._await_confirm = True
                self.overrides += 1
                return chosen
            self.declined += 1
            return None

        decision = forward_decision(observation, semantic_actions)
        if decision is None:
            self.declined += 1
            return None
        chosen_candidate = self._epsilon_greedy(decision, snapshot, valid)
        if chosen_candidate is None:
            self.declined += 1
            return None
        native_index, candidate = chosen_candidate
        self._close_open(terminal=False, floor=floor)
        self._open_transition(
            snapshot=snapshot,
            action_index=native_index,
            surface=decision.surface,
            branch=candidate.branch,
            target_key=candidate.target_key,
            floor=floor,
        )
        if len(candidate.plan) > 1:
            self._pending_target = candidate.target
            self._await_picker = True
        self.overrides += 1
        return native_index

    # ------------------------------------------------------------------ internals
    def _observed_floor(self, observation: Mapping[str, Any]) -> int:
        run = observation.get("run")
        floor = run.get("floor") if isinstance(run, Mapping) else None
        if isinstance(floor, int) and floor >= 0:
            self._last_floor = floor
        return self._last_floor

    def _match_index(
        self,
        semantic_actions: Sequence[Mapping[str, Any]],
        valid: np.typing.NDArray[np.bool_],
        *,
        kinds: frozenset[str],
        target: Mapping[str, Any] | None,
    ) -> int | None:
        for index, action in enumerate(semantic_actions):
            if (
                index < len(valid)
                and bool(valid[index])
                and _kind(action) in kinds
                and _card_matches(action, target)
            ):
                return index
        return None

    def _picker_indices(
        self,
        semantic_actions: Sequence[Mapping[str, Any]],
        valid: np.typing.NDArray[np.bool_],
    ) -> list[int]:
        indices = [
            index
            for index, action in enumerate(semantic_actions)
            if index < len(valid)
            and bool(valid[index])
            and _kind(action) in _PICKER_SELECT_KINDS
            and _card_matches(action, self._pending_target)
        ]
        if not indices and self._pending_target is not None:
            # Target vanished (fail closed): fall back to any legal select.
            indices = [
                index
                for index, action in enumerate(semantic_actions)
                if index < len(valid)
                and bool(valid[index])
                and _kind(action) in _PICKER_SELECT_KINDS
            ]
        self._pending_target = None
        return indices

    def _q_values(self, snapshot: EncodedDecisionSnapshot) -> Tensor:
        with torch.no_grad():
            values, self._hidden = self.forward_q(snapshot, self._hidden)
        return values

    def _q_choice(self, snapshot: EncodedDecisionSnapshot, indices: list[int]) -> int:
        if len(indices) == 1 or self._rng.random() < self.epsilon:
            return int(self._rng.choice(indices))
        values = self._q_values(snapshot)
        best = max(indices, key=lambda index: float(values[index].item()))
        return best

    def _epsilon_greedy(
        self,
        decision: ForwardDecision,
        snapshot: EncodedDecisionSnapshot,
        valid: np.typing.NDArray[np.bool_],
    ) -> tuple[int, Any] | None:
        executable = [
            candidate
            for candidate in decision.candidates
            if candidate.native_index is not None
            and candidate.native_index < len(valid)
            and bool(valid[candidate.native_index])
        ]
        if not executable:
            return None
        if self._rng.random() < self.epsilon:
            # Branch-balanced: uniform branch, then uniform candidate inside.
            branches = sorted({candidate.branch for candidate in executable})
            branch = branches[int(self._rng.integers(len(branches)))]
            pool = [c for c in executable if c.branch == branch]
            candidate = pool[int(self._rng.integers(len(pool)))]
            return int(candidate.native_index), candidate  # type: ignore[arg-type]
        values = self._q_values(snapshot)
        candidate = max(
            executable,
            key=lambda item: float(values[int(item.native_index)].item()),  # type: ignore[arg-type]
        )
        return int(candidate.native_index), candidate  # type: ignore[arg-type]

    def _open_transition(
        self,
        *,
        snapshot: EncodedDecisionSnapshot,
        action_index: int,
        surface: str,
        branch: str,
        target_key: str | None,
        floor: int,
    ) -> None:
        self._open = _OpenTransition(
            snapshot=snapshot,
            action_index=action_index,
            surface=surface,
            branch=branch,
            target_key=target_key,
            floor=floor,
            behavior_epsilon=self.epsilon,
        )

    def _close_open(self, *, terminal: bool, floor: int | None = None) -> None:
        if self._open is None:
            return
        closing_floor = self._last_floor if floor is None else floor
        discount = decision_discount(
            floor_before=self._open.floor,
            floor_after=max(closing_floor, self._open.floor),
            terminal=terminal,
        )
        self._steps.append(
            MacroStep(
                snapshot=self._open.snapshot,
                action_index=self._open.action_index,
                reward=self._open.reward_accumulator,
                discount=discount,
                terminal=terminal,
                surface=self._open.surface,
                branch=self._open.branch,
                target_key=self._open.target_key,
                behavior_epsilon=self._open.behavior_epsilon,
            )
        )
        self._open = None

    def metrics(self) -> dict[str, Any]:
        return {
            "version": MACRO_AUTHORITY_VERSION,
            "overrides": self.overrides,
            "mechanical_dispatches": self.mechanical_dispatches,
            "declined": self.declined,
            "recorded_steps": len(self._steps),
            "epsilon": self.epsilon,
        }


__all__ = ["MACRO_AUTHORITY_VERSION", "MacroCollectionAuthority"]
