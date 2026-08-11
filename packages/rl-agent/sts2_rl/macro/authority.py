"""Semantic collection authority with one recurrent state per domain.

During stage-two collection the frozen combat champion drives every decision
the authority declines.  At recognized macro surfaces the authority owns the
choice: branch-balanced epsilon-greedy over atomic semantic candidates.  A
composite candidate such as ``Smith(card)`` is learned once at its parent
surface and then executed as a native entry/select/confirm plan.  Picker and
confirm traffic are mechanical executor suffixes, never second learning
transitions.

The authority records macro transitions as it acts; the legacy learner never
sees them and the macro learner never sees anything else.  Every override it
returns is a legal native candidate index — legality authority stays with
the engine.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import torch
from torch import Tensor

from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedObservationEncoder
from sts2_rl.semantics.clock import decision_discount
from sts2_rl.semantics.forward import ForwardDecision, NativeStep, SemanticCandidate, forward_decision
from sts2_rl.semantics.grouping import semantic_card_projection, strict_action_groups
from sts2_rl.semantics.identity import SemanticContractError

from .transitions import MacroEpisode, MacroStep

MACRO_AUTHORITY_VERSION: Final = "sts2-macro-authority-v4"

_PICKER_SELECT_KINDS: Final[frozenset[str]] = frozenset(
    {"select_card", "select_card_option", "select_hand_card", "combat_select_card"}
)
_PICKER_CONFIRM_KINDS: Final[frozenset[str]] = frozenset(
    {"confirm_selection", "combat_confirm_selection"}
)
_PICKER_REVERSE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "cancel_selection",
        "deselect_card",
        "deselect_card_option",
        "deselect_hand_card",
        "combat_deselect_card",
    }
)


def _kind(action: Mapping[str, Any]) -> str:
    return (
        str(action.get("kind") or action.get("action") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _kind_for_step(step: NativeStep) -> str:
    return _kind({"kind": step.kind})


def _card_matches(action: Mapping[str, Any], target: Mapping[str, Any] | None) -> bool:
    if target is None:
        return True
    card = action.get("card")
    if not isinstance(card, Mapping):
        return False
    try:
        return semantic_card_projection(card) == semantic_card_projection(target)
    except SemanticContractError:
        return False


@dataclass(slots=True)
class _OpenTransition:
    snapshot: EncodedDecisionSnapshot
    action_index: int
    surface: str
    branch: str
    target_key: str | None
    floor: int
    behavior_epsilon: float
    recurrent_reset: bool
    reward_accumulator: float = 0.0


@dataclass(slots=True)
class _PendingPlan:
    steps: tuple[NativeStep, ...]


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
        control_domain: Literal["macro", "combat"] = "macro",
    ) -> None:
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("authority epsilon must be in [0, 1]")
        self.forward_q = forward_q
        self.initial_state = initial_state
        self.epsilon = epsilon
        # Stage-3 joined evaluation: macro ownership extends into
        # deterministic held-out collection (whole-segment ownership, §10).
        self.evaluation_ownership = bool(evaluation_ownership)
        if control_domain not in {"macro", "combat"}:
            raise ValueError("control_domain must be 'macro' or 'combat'")
        # One recurrent authority owns exactly one domain.  Joined control is
        # routed through two instances so combat history cannot leak into the
        # macro recurrent state (or vice versa).
        self.control_domain = control_domain
        # Diagnostic-only decision log (never a training input): one row per
        # owned decision with the observation facts needed to inspect
        # state-conditioning behaviorally (reset doc §11 items 2/4).
        self.record_decisions = False
        self.decision_log: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(seed)
        self._hidden: Any = None
        self._episode_id: str | None = None
        self._steps: list[MacroStep] = []
        self._open: _OpenTransition | None = None
        self._pending_plan: _PendingPlan | None = None
        self._episode_replay_invalid = False
        self._executor_failures = 0
        self._last_floor = 0
        self._combat_active = False
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
        self._pending_plan = None
        self._episode_replay_invalid = False
        self._executor_failures = 0
        self._last_floor = 0
        self._combat_active = False
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
            self._pending_plan = None

    def observe_observation(self, observation: Mapping[str, Any]) -> None:
        """Track domain boundaries even when another authority owns the turn.

        An isolated collector calls :meth:`choose` at every decision, so a
        combat authority naturally observes the intervening macro surface.
        The joined router delegates directly to one authority; this explicit
        observation hook keeps the combat recurrent reset identical in both
        execution modes without invoking a second policy on the same turn.
        """

        if self.control_domain != "combat":
            return
        combat = observation.get("combat")
        if not (isinstance(combat, Mapping) and combat.get("in_progress") is True):
            self._combat_active = False

    def finish_episode(self, episode_id: str | None = None) -> MacroEpisode | None:
        if self._episode_replay_invalid:
            self._episode_id = None
            self._steps = []
            self._open = None
            self._pending_plan = None
            return None
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
        if self._episode_replay_invalid:
            # A composite action changed recurrent state before its native
            # suffix failed.  No later transition in this episode can be
            # replayed against the behavior recurrent chain, so the authority
            # cedes the remainder of the episode.
            self.declined += 1
            return None
        # Collector candidates arrive as strict-group wrappers; the native
        # action facts live under their ``prototype`` key.
        semantic_actions = [
            action["prototype"]
            if isinstance(action.get("prototype"), Mapping)
            else action
            for action in semantic_actions
        ]
        floor = self._observed_floor(observation)
        self.observe_observation(observation)

        # Complete the native suffix of a previously learned semantic action.
        # Auto-confirm may move directly to a new meaningful decision; in that
        # case we consume the absent confirm and keep processing this same
        # observation rather than ceding it to another policy.
        while self._pending_plan is not None:
            step = self._pending_plan.steps[0]
            matches = self._step_indices(step, semantic_actions, valid)
            dispatch_index = self._equivalent_dispatch_index(
                matches,
                semantic_actions,
            )
            if dispatch_index is not None:
                remaining = self._pending_plan.steps[1:]
                self._pending_plan = _PendingPlan(remaining) if remaining else None
                self.mechanical_dispatches += 1
                return dispatch_index
            if (
                _kind_for_step(step) in _PICKER_CONFIRM_KINDS
                and not self._is_selection_surface(semantic_actions, valid)
            ):
                remaining = self._pending_plan.steps[1:]
                self._pending_plan = _PendingPlan(remaining) if remaining else None
                continue

            # The executor cannot realize the learned semantic action.  The Q
            # recurrent state already consumed that action, so dropping only
            # its transition would also corrupt every later replay state.  The
            # whole domain episode is therefore ineligible for replay.
            self._pending_plan = None
            self._open = None
            self._steps = []
            self._episode_replay_invalid = True
            self._executor_failures += 1
            self.declined += 1
            return None

        decision = forward_decision(
            observation,
            semantic_actions,
            control_domain=self.control_domain,
        )
        if decision is None:
            self.declined += 1
            return None
        executable = self._executable_candidates(decision, valid)
        if not executable:
            self.declined += 1
            return None
        recurrent_reset = False
        if self.control_domain == "combat" and not self._combat_active:
            # Tactical memory begins at the encounter boundary.  The replay
            # records this reset so behavior and learner recurrence agree.
            self._hidden = self.initial_state()
            self._combat_active = True
            recurrent_reset = True
        semantic_snapshot = GroundedObservationEncoder(snapshot.config).encode(
            observation,
            [candidate.semantic_action for candidate in executable],
        ).snapshot
        if semantic_snapshot.candidate_count != len(executable):
            raise RuntimeError("semantic candidate encoding changed the candidate set")
        semantic_index = self._semantic_choice(
            candidates=executable,
            snapshot=semantic_snapshot,
        )
        candidate = executable[semantic_index]
        if candidate.native_index is None:  # excluded above; keeps typing exact
            raise RuntimeError("executable semantic candidate has no native index")
        native_index = int(candidate.native_index)
        if self.record_decisions:
            player = observation.get("player")
            player = player if isinstance(player, Mapping) else {}
            self.decision_log.append(
                {
                    "surface": decision.surface,
                    "branch": candidate.branch,
                    "floor": floor,
                    "hp": player.get("hp"),
                    "max_hp": player.get("max_hp"),
                    "gold": player.get("gold"),
                    "branches_offered": list(decision.branches),
                }
            )
        self._close_open(terminal=False, floor=floor)
        self._open_transition(
            snapshot=semantic_snapshot,
            action_index=semantic_index,
            surface=decision.surface,
            branch=candidate.branch,
            target_key=candidate.target_key,
            floor=floor,
            recurrent_reset=recurrent_reset,
        )
        if len(candidate.plan) > 1:
            self._pending_plan = _PendingPlan(candidate.plan[1:])
        self.overrides += 1
        return native_index

    # ------------------------------------------------------------------ internals
    def _observed_floor(self, observation: Mapping[str, Any]) -> int:
        run = observation.get("run")
        floor = run.get("floor") if isinstance(run, Mapping) else None
        if isinstance(floor, int) and floor >= 0:
            self._last_floor = floor
        return self._last_floor

    def _step_indices(
        self,
        step: NativeStep,
        semantic_actions: Sequence[Mapping[str, Any]],
        valid: np.typing.NDArray[np.bool_],
    ) -> list[int]:
        expected_kind = _kind_for_step(step)
        accepted_kinds = (
            _PICKER_SELECT_KINDS
            if expected_kind == "select_card"
            else _PICKER_CONFIRM_KINDS
            if expected_kind == "confirm_selection"
            else frozenset({expected_kind})
        )
        return [
            index
            for index, action in enumerate(semantic_actions)
            if index < len(valid)
            and bool(valid[index])
            and _kind(action) in accepted_kinds
            and _card_matches(action, step.target)
        ]

    @staticmethod
    def _equivalent_dispatch_index(
        matches: Sequence[int],
        semantic_actions: Sequence[Mapping[str, Any]],
    ) -> int | None:
        """Return a stable representative only for one strict action class.

        An exact duplicate card may appear through more than one native
        picker row.  Dispatching the first member is sound only when the
        shared strict grouping projection proves those rows equivalent.  A
        set containing genuinely different copies remains ambiguous and is
        declined rather than silently targeting the wrong card.
        """

        if not matches:
            return None
        if len(matches) == 1:
            return int(matches[0])
        projected = [semantic_actions[index] for index in matches]
        groups = strict_action_groups(projected)
        if len(groups) != 1 or groups[0].multiplicity != len(matches):
            return None
        return int(matches[groups[0].member_positions[0]])

    @staticmethod
    def _is_selection_surface(
        semantic_actions: Sequence[Mapping[str, Any]],
        valid: np.typing.NDArray[np.bool_],
    ) -> bool:
        selection_kinds = (
            _PICKER_SELECT_KINDS | _PICKER_CONFIRM_KINDS | _PICKER_REVERSE_KINDS
        )
        return any(
            index < len(valid)
            and bool(valid[index])
            and _kind(action) in selection_kinds
            for index, action in enumerate(semantic_actions)
        )

    def _q_values(self, snapshot: EncodedDecisionSnapshot) -> Tensor:
        with torch.no_grad():
            values, self._hidden = self.forward_q(snapshot, self._hidden)
        return values

    @staticmethod
    def _executable_candidates(
        decision: ForwardDecision,
        valid: np.typing.NDArray[np.bool_],
    ) -> list[SemanticCandidate]:
        return [
            candidate
            for candidate in decision.candidates
            if candidate.native_index is not None
            and candidate.native_index < len(valid)
            and bool(valid[candidate.native_index])
        ]

    def _semantic_choice(
        self,
        *,
        candidates: Sequence[SemanticCandidate],
        snapshot: EncodedDecisionSnapshot,
    ) -> int:
        # Advance recurrent state on every meaningful decision, including a
        # singleton or epsilon-explored one.  Replay sees the same sequence.
        values = self._q_values(snapshot)
        if self._rng.random() < self.epsilon:
            # Branch-balanced: uniform branch, then uniform candidate inside.
            branches = sorted({candidate.branch for candidate in candidates})
            branch = branches[int(self._rng.integers(len(branches)))]
            pool = [index for index, item in enumerate(candidates) if item.branch == branch]
            return int(pool[int(self._rng.integers(len(pool)))])
        return max(
            range(len(candidates)),
            key=lambda index: float(values[index].item()),
        )

    def _open_transition(
        self,
        *,
        snapshot: EncodedDecisionSnapshot,
        action_index: int,
        surface: str,
        branch: str,
        target_key: str | None,
        floor: int,
        recurrent_reset: bool,
    ) -> None:
        self._open = _OpenTransition(
            snapshot=snapshot,
            action_index=action_index,
            surface=surface,
            branch=branch,
            target_key=target_key,
            floor=floor,
            behavior_epsilon=self.epsilon,
            recurrent_reset=recurrent_reset,
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
                control_domain=self.control_domain,
                recurrent_reset=self._open.recurrent_reset,
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
            "episode_replay_invalid": self._episode_replay_invalid,
            "executor_failures": self._executor_failures,
            "epsilon": self.epsilon,
            "control_domain": self.control_domain,
        }


__all__ = ["MACRO_AUTHORITY_VERSION", "MacroCollectionAuthority"]
