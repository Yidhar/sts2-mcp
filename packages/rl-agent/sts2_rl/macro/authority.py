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
    """Container-invariant card identity for executor binding.

    Different containers render different views of the SAME card: deck
    aggregates carry quantity/description/rarity, selection pickers carry
    upgrade previews instead (measured on real journals — full-projection
    equality can never bind a deck-compiled target to a picker action).
    A field a container does not render cannot disagree, so identity is
    exact equality over the keys BOTH projections present; id and upgrade
    state are always present on both sides, keeping distinct cards apart.
    """

    if target is None:
        return True
    card = action.get("card")
    if not isinstance(card, Mapping):
        return False
    try:
        action_view = {
            key: value
            for key, value in semantic_card_projection(card).items()
            if value is not None
        }
        target_view = {
            key: value
            for key, value in semantic_card_projection(target).items()
            if value is not None
        }
    except SemanticContractError:
        return False
    shared = action_view.keys() & target_view.keys()
    if "id" not in shared:
        return False
    return all(action_view[key] == target_view[key] for key in shared)


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
    control_domain: Literal["macro", "combat"]
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
        control_domain: Literal["macro", "combat", "joint"] = "macro",
    ) -> None:
        if not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
            raise ValueError("authority epsilon must be in [0, 1]")
        self.forward_q = forward_q
        self.initial_state = initial_state
        self.epsilon = epsilon
        # Stage-3 joined evaluation: macro ownership extends into
        # deterministic held-out collection (whole-segment ownership, §10).
        self.evaluation_ownership = bool(evaluation_ownership)
        if control_domain not in {"macro", "combat", "joint"}:
            raise ValueError("control_domain must be 'macro', 'combat', or 'joint'")
        # Isolated stage-2 authorities own one domain.  The hybrid lineage
        # selects ``joint`` and supplies split run/combat recurrence in one
        # model; the semantic executor below remains shared by both modes.
        self.control_domain = control_domain
        # Diagnostic-only decision log (never a training input): one row per
        # owned decision with the observation facts needed to inspect
        # state-conditioning behaviorally (reset doc §11 items 2/4).
        self.record_decisions = False
        self.decision_log: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(seed)
        # Greedy legal value of the most recent OWNED decision (max over the
        # executable candidates' Q, regardless of the epsilon draw actually
        # executed), evaluated under this authority's REAL recurrent state.
        # Freshness contract: invalidated at the start of every ``choose``
        # call and by ``begin_episode``, published only when this very call
        # owned a semantic decision; declines and mechanical executor
        # dispatches leave it ``None``.  The joined router reads-and-clears
        # it to close a pending combat bridge at collection time.
        self.last_decision_value: float | None = None
        self.last_q_values: np.typing.NDArray[np.float32] | None = None
        # The semantic compiler can collapse/reorder native candidates.  This
        # companion view maps the most recent semantic Q values back onto the
        # collector's native candidate indexes for diagnostics only.
        self.last_native_q_values: np.typing.NDArray[np.float32] | None = None
        self._hidden: Any = None
        self._episode_id: str | None = None
        self._steps: list[MacroStep] = []
        self._open: _OpenTransition | None = None
        self._pending_plan: _PendingPlan | None = None
        self._episode_replay_invalid = False
        self._executor_failures = 0
        self.semantic_encode_collisions = 0
        self.last_executor_failure: dict[str, Any] | None = None
        self._last_floor = 0
        self._combat_active = False
        self._awaiting_bridge = False
        self._bridge_misses = 0
        self.overrides = 0
        self.mechanical_dispatches = 0
        self.declined = 0
        self._auto_counter = 0

    # ------------------------------------------------------------------ episode
    def begin_episode(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self.last_decision_value = None
        self.last_q_values = None
        self.last_native_q_values = None
        self._hidden = self.initial_state()
        self._steps = []
        self._open = None
        self._pending_plan = None
        self._episode_replay_invalid = False
        self._executor_failures = 0
        self.semantic_encode_collisions = 0
        self._last_floor = 0
        self._combat_active = False
        self._awaiting_bridge = False
        self._bridge_misses = 0
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
            if self._open is not None:
                # The encounter ended with its closing transition still open.
                # It stays open across the boundary until the next macro
                # decision snapshot arrives (:meth:`close_encounter`); a run
                # terminal instead closes it terminally with bootstrap 0.
                self._awaiting_bridge = True

    @property
    def has_pending_bridge(self) -> bool:
        """True while an encounter-closing transition awaits its bridge."""

        return self._awaiting_bridge and self._open is not None

    def close_encounter(
        self,
        bridge_snapshot: EncodedDecisionSnapshot,
        bridge_value: float,
    ) -> None:
        """Close the open encounter transition on the post-combat macro surface.

        The closing combat transition receives the boundary rewards accrued
        since the last combat decision, the clock discount to the current
        floor, the macro decision snapshot, and the macro authority's greedy
        legal value of that exact surface — captured at COLLECTION time under
        the partner's real run-scale recurrent state (bridge doc §2).  The
        partner is frozen all segment, so the recorded value keeps the
        pinned-partner stationarity contract.
        """

        if self.control_domain != "combat":
            raise RuntimeError("only the combat authority closes encounters over a bridge")
        if not isinstance(bridge_snapshot, EncodedDecisionSnapshot):
            raise TypeError("bridge snapshot has the wrong type")
        value = float(bridge_value)
        if not math.isfinite(value):
            raise ValueError("bridge value must be finite")
        if self._open is None or not self._awaiting_bridge:
            raise RuntimeError("no open encounter transition awaits a bridge snapshot")
        self._close_open(
            terminal=False,
            bridge_snapshot=bridge_snapshot,
            bridge_value=value,
        )

    def finish_episode(self, episode_id: str | None = None) -> MacroEpisode | None:
        if self._episode_replay_invalid:
            self._episode_id = None
            self._steps = []
            self._open = None
            self._pending_plan = None
            self._awaiting_bridge = False
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
        # Freshness: the published greedy value belongs to at most one call.
        # Every path that does not OWN a semantic decision below (declines,
        # mechanical executor dispatches, failures) leaves it ``None``.
        self.last_decision_value = None
        self.last_q_values = None
        self.last_native_q_values = None
        self._activate_domain_epsilon(observation)
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
                # This native action is the mechanical suffix of the prior
                # semantic decision, not a second learned preference.  Give
                # the generic collector a deterministic native diagnostic
                # view without running either control network again.
                native_q: np.typing.NDArray[np.float32] = np.full(
                    len(valid), -np.inf, dtype=np.float32
                )
                native_q[dispatch_index] = 0.0
                self.last_native_q_values = native_q
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
            self._awaiting_bridge = False
            self._episode_replay_invalid = True
            self._executor_failures += 1
            self.last_executor_failure = {
                "step_kind": _kind_for_step(step),
                "step_target": dict(step.target) if step.target else None,
                "surface_kinds": sorted({_kind(action) for action in semantic_actions})[:12],
                "floor": floor,
            }
            self.declined += 1
            return None

        decision_domain: Literal["macro", "combat"] = (
            "combat"
            if self.control_domain == "joint"
            and isinstance(observation.get("combat"), Mapping)
            and observation["combat"].get("in_progress") is True
            else "macro"
            if self.control_domain == "joint"
            else self.control_domain
        )
        decision = forward_decision(
            observation,
            semantic_actions,
            control_domain=decision_domain,
        )
        if decision is None:
            if self.control_domain == "joint" and int(np.count_nonzero(valid)) > 1:
                # A joint controller may pass through forced/mechanical
                # singletons, but it must not hide an uncompiled strategic
                # choice inside the previous transition.
                self._pending_plan = None
                self._open = None
                self._steps = []
                self._episode_replay_invalid = True
                self.last_executor_failure = {
                    "reason": "uncompiled_strategic_surface",
                    "control_domain": decision_domain,
                    "phase": str(observation.get("phase") or ""),
                    "surface_kinds": sorted(
                        {_kind(action) for action in semantic_actions}
                    )[:12],
                    "valid_candidates": int(np.count_nonzero(valid)),
                    "floor": floor,
                }
            self.declined += 1
            return None
        executable = self._executable_candidates(decision, valid)
        if not executable:
            self.declined += 1
            return None
        if self.control_domain == "combat" and self._awaiting_bridge:
            # Fail-closed (bridge doc §6): a new encounter is beginning but
            # the previous encounter's closing transition never captured its
            # macro bridge snapshot.  Folding the boundary silently would
            # bake macro behavior into combat values, so the whole episode
            # leaves combat replay — the existing replay-invalid mechanism.
            self._pending_plan = None
            self._open = None
            self._steps = []
            self._awaiting_bridge = False
            self._episode_replay_invalid = True
            self._bridge_misses += 1
            self.declined += 1
            return None
        recurrent_reset = False
        if self.control_domain == "combat" and not self._combat_active:
            # Tactical memory begins at the encounter boundary.  The replay
            # records this reset so behavior and learner recurrence agree.
            self._hidden = self.initial_state()
            self._combat_active = True
            recurrent_reset = True
        try:
            semantic_snapshot = GroundedObservationEncoder(snapshot.config).encode(
                observation,
                [candidate.semantic_action for candidate in executable],
            ).snapshot
        except ValueError:
            # The configured role vocabulary cannot represent this candidate
            # set distinctly (e.g. co-occurring reward-claim branches hashing
            # to one role). The vocabulary is encoding ABI shared with the
            # frozen champion, so the authority fails closed instead: this
            # decision returns to the champion and is counted, never learned.
            if self.control_domain == "combat" and recurrent_reset:
                self._combat_active = False
                self._hidden = self.initial_state()
            self.semantic_encode_collisions += 1
            self.declined += 1
            return None
        if semantic_snapshot.candidate_count != len(executable):
            raise RuntimeError("semantic candidate encoding changed the candidate set")
        semantic_index = self._semantic_choice(
            candidates=executable,
            snapshot=semantic_snapshot,
        )
        semantic_q = self.last_q_values
        if semantic_q is None or semantic_q.shape != (len(executable),):
            raise RuntimeError("semantic Q publication does not match candidates")
        mapped_native_q: np.typing.NDArray[np.float32] = np.full(
            len(valid), -np.inf, dtype=np.float32
        )
        for index, executable_candidate in enumerate(executable):
            executable_native = executable_candidate.native_index
            if executable_native is None:
                continue
            mapped_native_q[executable_native] = max(
                float(mapped_native_q[executable_native]),
                float(semantic_q[index]),
            )
        self.last_native_q_values = mapped_native_q
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
            control_domain=decision_domain,
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

    def _activate_domain_epsilon(self, observation: Mapping[str, Any]) -> None:
        """Hook for a joint authority with independent domain clocks."""

        del observation

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
        self.last_q_values = values.detach().float().cpu().numpy().copy()
        # Publish the greedy legal value of this OWNED decision — the max over
        # the executable candidates' Q under the live recurrent state — no
        # matter which action the epsilon draw below actually executes.
        self.last_decision_value = float(
            max(float(values[index].item()) for index in range(len(candidates)))
        )
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
        control_domain: Literal["macro", "combat"] | None = None,
    ) -> None:
        resolved_domain: Literal["macro", "combat"]
        if control_domain is not None:
            resolved_domain = control_domain
        elif self.control_domain == "joint":
            raise RuntimeError("joint authority must provide the active control domain")
        else:
            resolved_domain = self.control_domain
        self._open = _OpenTransition(
            snapshot=snapshot,
            action_index=action_index,
            surface=surface,
            branch=branch,
            target_key=target_key,
            floor=floor,
            behavior_epsilon=self.epsilon,
            recurrent_reset=recurrent_reset,
            control_domain=resolved_domain,
        )

    def _close_open(
        self,
        *,
        terminal: bool,
        floor: int | None = None,
        bridge_snapshot: EncodedDecisionSnapshot | None = None,
        bridge_value: float | None = None,
    ) -> None:
        if self._open is None:
            self._awaiting_bridge = False
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
                control_domain=self._open.control_domain,
                recurrent_reset=self._open.recurrent_reset,
                target_key=self._open.target_key,
                behavior_epsilon=self._open.behavior_epsilon,
                bridge_snapshot=bridge_snapshot,
                bridge_value=bridge_value,
            )
        )
        self._open = None
        self._awaiting_bridge = False

    def metrics(self) -> dict[str, Any]:
        return {
            "version": MACRO_AUTHORITY_VERSION,
            "overrides": self.overrides,
            "mechanical_dispatches": self.mechanical_dispatches,
            "declined": self.declined,
            "recorded_steps": len(self._steps),
            "episode_replay_invalid": self._episode_replay_invalid,
            "executor_failures": self._executor_failures,
            "semantic_encode_collisions": self.semantic_encode_collisions,
            "bridge_misses": self._bridge_misses,
            "pending_bridge": self.has_pending_bridge,
            "epsilon": self.epsilon,
            "control_domain": self.control_domain,
        }


__all__ = ["MACRO_AUTHORITY_VERSION", "MacroCollectionAuthority"]
