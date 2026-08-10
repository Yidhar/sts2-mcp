"""Semantic action compiler: a stateful episode pass above the stateless kernel.

The kernel identifies one decision at a time.  The compiler owns what the
kernel deliberately does not: sequence state.  It threads the progress-scope
stack across steps so a card picker keeps its rest/shop/event parent identity
(the kernel's ``parent_scopes`` inheritance), and it folds mechanical
selection traffic — select/deselect/confirm/cancel — into composite semantic
decisions per the reset contract:

- a single-target selection compiles into ONE decision whose chosen candidate
  is ``(parent operation, selected target)``; the native toggle/confirm
  sequence is a mechanical suffix, never a set of decisions;
- a multi-target selection compiles into monotone ``Add(item)`` decisions and
  one ``CommitSet``; deselect churn is folded away and counted;
- a cancel that merely returns to the already-known parent state compiles into
  the parent-level refusal candidate, not a decision of its own;
- every step under an unknown (opaque) root passes through untouched —
  fail-closed means raw exposure, never guessing.

The compiler emits facts and counters only.  It assigns no preference, no
reward and no label; training layers consume its output the same way they
consume the kernel's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from .clock import DecisionClockTick, clock_tick
from .contracts import DecisionSemantics
from .grouping import card_selection_operation
from .identity import canonical_payload_bytes
from .kernel import DecisionSemanticsKernel
from .scopes import ProgressScopeStack

SEMANTIC_COMPILER_CONTRACT_VERSION: Final = "sts2-semantic-compiler-v1"

_SELECT_OPERATIONS: Final[frozenset[str]] = frozenset({"select"})
_DESELECT_OPERATIONS: Final[frozenset[str]] = frozenset({"deselect"})
_CONFIRM_TOKENS: Final[frozenset[str]] = frozenset(
    {"confirm", "confirm_selection", "combat_confirm_selection"}
)
_CANCEL_TOKENS: Final[frozenset[str]] = frozenset(
    {"cancel", "cancel_prompt", "cancel_selection", "combat_cancel_selection"}
)


def _normalized_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _selection_traffic_operation(action: Mapping[str, Any]) -> str | None:
    """Classify overlay traffic: select/deselect via the reviewed grouping
    contract, confirm/cancel via their stable action tokens (the grouping
    contract deliberately scopes itself to target toggles)."""

    reviewed = card_selection_operation(action)
    if reviewed is not None:
        return reviewed
    for key in ("selection_operation", "model_action_variant", "kind", "action"):
        token = _normalized_token(action.get(key))
        if token in _CONFIRM_TOKENS:
            return "confirm"
        if token in _CANCEL_TOKENS:
            return "cancel"
    return None


class CompiledKind(StrEnum):
    """What one native step became after compilation."""

    SEMANTIC_DECISION = "semantic_decision"
    COMPOSITE_TARGET = "composite_target"
    COMPOSITE_COMMIT = "composite_commit"
    COMPOSITE_REFUSAL = "composite_refusal"
    MECHANICAL_FOLDED = "mechanical_folded"
    FORCED_SINGLETON = "forced_singleton"
    OPAQUE_PASSTHROUGH = "opaque_passthrough"


@dataclass(frozen=True, slots=True)
class CompiledEvent:
    """One compiled fact for one native step."""

    step_index: int
    kind: CompiledKind
    root_spec: str
    overlay_active: bool
    clock: DecisionClockTick | None = None
    operation: str | None = None
    target_identity: str | None = None
    folded_steps: int = 0
    deselect_churn: int = 0
    reveal_boundary: bool = False


@dataclass(slots=True)
class _PendingComposite:
    root_spec: str
    opened_step: int
    max_select: int
    selected_targets: list[str] = field(default_factory=list)
    folded_steps: int = 0
    deselect_churn: int = 0
    selectable_digest: str | None = None


def _selection_contract(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("selection", "card_selection"):
        value = observation.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _max_select(observation: Mapping[str, Any]) -> int:
    contract = _selection_contract(observation)
    raw = contract.get("max_select")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 1
    return max(raw, 1)


def _selectable_digest(legal_actions: Sequence[Mapping[str, Any]]) -> str:
    payload = sorted(
        canonical_payload_bytes(
            {
                "card": action.get("card"),
                "item": action.get("item"),
                "kind": action.get("kind") or action.get("action"),
            }
        ).decode("utf-8")
        for action in legal_actions
        if card_selection_operation(action) in _SELECT_OPERATIONS
    )
    return canonical_payload_bytes(payload).decode("utf-8")


def _target_identity(action: Mapping[str, Any]) -> str:
    return canonical_payload_bytes(
        {
            "card": action.get("card"),
            "item": action.get("item"),
        }
    ).decode("utf-8")


class SemanticActionCompiler:
    """Compile one episode's native decision stream into semantic events.

    Feed steps in order; read ``events`` afterwards.  The compiler is
    deliberately forgiving about sparse streams (journal shadow corpora only
    carry a subset of steps): a composite that never sees its confirm closes
    as censored mechanics, and scope threading survives gaps because the
    kernel re-anchors whenever a specific root matches directly.
    """

    def __init__(self, kernel: DecisionSemanticsKernel | None = None) -> None:
        self.kernel = kernel or DecisionSemanticsKernel()
        self.events: list[CompiledEvent] = []
        self._scopes: ProgressScopeStack | None = None
        self._pending: _PendingComposite | None = None
        self._last_floor: int = 0

    def feed(
        self,
        *,
        step_index: int,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
        selected_action: Mapping[str, Any] | None,
        floor: int | None = None,
        terminal: bool = False,
    ) -> CompiledEvent:
        semantics = self.kernel.identify(
            observation=observation,
            legal_actions=legal_actions,
            parent_scopes=self._scopes,
        )
        self._scopes = semantics.scopes
        floor_now = self._last_floor if floor is None else max(int(floor), 0)
        root_spec = semantics.scopes.root.spec_id
        overlay_active = bool(semantics.scopes.overlays)

        event = self._compile_step(
            step_index=step_index,
            semantics=semantics,
            observation=observation,
            legal_actions=legal_actions,
            selected_action=selected_action,
            root_spec=root_spec,
            overlay_active=overlay_active,
            floor_now=floor_now,
            terminal=terminal,
        )
        self._last_floor = floor_now
        self.events.append(event)
        return event

    def _compile_step(
        self,
        *,
        step_index: int,
        semantics: DecisionSemantics,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
        selected_action: Mapping[str, Any] | None,
        root_spec: str,
        overlay_active: bool,
        floor_now: int,
        terminal: bool,
    ) -> CompiledEvent:
        tick = clock_tick(
            floor_before=self._last_floor,
            floor_after=floor_now,
            terminal=terminal,
        )
        if root_spec == "opaque":
            # Fail-closed: unknown surfaces are exposed raw, never compiled.
            self._pending = None
            return CompiledEvent(
                step_index=step_index,
                kind=CompiledKind.OPAQUE_PASSTHROUGH,
                root_spec=root_spec,
                overlay_active=overlay_active,
                clock=tick,
            )
        if not overlay_active:
            self._pending = None
            if not semantics.is_policy_choice:
                # Forced singletons are mechanical suffix material.
                return CompiledEvent(
                    step_index=step_index,
                    kind=CompiledKind.FORCED_SINGLETON,
                    root_spec=root_spec,
                    overlay_active=False,
                    clock=tick,
                )
            return CompiledEvent(
                step_index=step_index,
                kind=CompiledKind.SEMANTIC_DECISION,
                root_spec=root_spec,
                overlay_active=False,
                clock=tick,
            )

        # Selection overlay is active under a known parent root.
        pending = self._pending
        digest = _selectable_digest(legal_actions)
        reveal = False
        if pending is None or pending.root_spec != root_spec:
            pending = _PendingComposite(
                root_spec=root_spec,
                opened_step=step_index,
                max_select=_max_select(observation),
                selectable_digest=digest,
            )
            self._pending = pending
        elif pending.selectable_digest is not None and digest != pending.selectable_digest:
            # The selectable set changed while the overlay stayed open: a
            # random reveal introduced new information, so a NEW meaningful
            # decision begins here (reset contract, reveal boundary).
            reveal = True
            pending.selectable_digest = digest

        operation = (
            _selection_traffic_operation(selected_action)
            if isinstance(selected_action, Mapping)
            else None
        )
        if operation in _SELECT_OPERATIONS:
            assert isinstance(selected_action, Mapping)
            target = _target_identity(selected_action)
            pending.folded_steps += 1
            if pending.max_select <= 1:
                pending.selected_targets = [target]
                return CompiledEvent(
                    step_index=step_index,
                    kind=CompiledKind.MECHANICAL_FOLDED,
                    root_spec=root_spec,
                    overlay_active=True,
                    operation="select",
                    target_identity=target,
                    reveal_boundary=reveal,
                )
            # Monotone set controller: a NEW target is an irreversible
            # Add(item) commitment in the compiled graph.
            if target not in pending.selected_targets:
                pending.selected_targets.append(target)
                return CompiledEvent(
                    step_index=step_index,
                    kind=CompiledKind.COMPOSITE_TARGET,
                    root_spec=root_spec,
                    overlay_active=True,
                    operation="add",
                    target_identity=target,
                    clock=tick,
                    reveal_boundary=reveal,
                )
            return CompiledEvent(
                step_index=step_index,
                kind=CompiledKind.MECHANICAL_FOLDED,
                root_spec=root_spec,
                overlay_active=True,
                operation="select",
                target_identity=target,
                reveal_boundary=reveal,
            )
        if operation in _DESELECT_OPERATIONS:
            assert isinstance(selected_action, Mapping)
            pending.folded_steps += 1
            pending.deselect_churn += 1
            target = _target_identity(selected_action)
            if target in pending.selected_targets:
                pending.selected_targets.remove(target)
            return CompiledEvent(
                step_index=step_index,
                kind=CompiledKind.MECHANICAL_FOLDED,
                root_spec=root_spec,
                overlay_active=True,
                operation="deselect",
                target_identity=target,
                deselect_churn=1,
                reveal_boundary=reveal,
            )
        if operation == "confirm":
            chosen = tuple(pending.selected_targets)
            folded = pending.folded_steps + 1
            churn = pending.deselect_churn
            self._pending = None
            return CompiledEvent(
                step_index=step_index,
                kind=CompiledKind.COMPOSITE_COMMIT,
                root_spec=root_spec,
                overlay_active=True,
                operation="commit_set" if pending.max_select > 1 else "commit",
                target_identity=canonical_payload_bytes(sorted(chosen)).decode("utf-8"),
                clock=tick,
                folded_steps=folded,
                deselect_churn=churn,
                reveal_boundary=reveal,
            )
        if operation == "cancel":
            folded = pending.folded_steps + 1
            churn = pending.deselect_churn
            self._pending = None
            return CompiledEvent(
                step_index=step_index,
                kind=CompiledKind.COMPOSITE_REFUSAL,
                root_spec=root_spec,
                overlay_active=True,
                operation="refuse",
                clock=tick,
                folded_steps=folded,
                deselect_churn=churn,
                reveal_boundary=reveal,
            )
        # Unrecognized traffic inside a known overlay is folded, not guessed.
        pending.folded_steps += 1
        return CompiledEvent(
            step_index=step_index,
            kind=CompiledKind.MECHANICAL_FOLDED,
            root_spec=root_spec,
            overlay_active=True,
            reveal_boundary=reveal,
        )

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.kind.value] = counts.get(event.kind.value, 0) + 1
        return {
            "version": SEMANTIC_COMPILER_CONTRACT_VERSION,
            "events": len(self.events),
            "kinds": counts,
            "deselect_churn": sum(
                event.deselect_churn
                for event in self.events
                if event.kind
                in (CompiledKind.COMPOSITE_COMMIT, CompiledKind.COMPOSITE_REFUSAL)
            ),
            "reveal_boundaries": sum(
                1 for event in self.events if event.reveal_boundary
            ),
        }


__all__ = [
    "SEMANTIC_COMPILER_CONTRACT_VERSION",
    "CompiledEvent",
    "CompiledKind",
    "SemanticActionCompiler",
]
