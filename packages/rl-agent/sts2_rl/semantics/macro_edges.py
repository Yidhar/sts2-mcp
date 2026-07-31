"""Macro-edge contract joining one policy choice to its forced suffix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from .identity import IdentityTriple, SemanticKey
from .progress import ProgressReceipt

MACRO_EDGE_CONTRACT_VERSION: Final = "sts2-policy-macro-edge-v1"


class MacroEdgeOutcome(StrEnum):
    NEXT_POLICY = "next_policy"
    ANCHOR_ADVANCE = "anchor_advance"
    TERMINAL = "terminal"
    CENSORED = "censored"


@dataclass(frozen=True, slots=True)
class ForcedTransition:
    step_id: int
    node: IdentityTriple
    action: IdentityTriple
    receipt: ProgressReceipt
    candidate_count: int = 1

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("forced transition step_id must be non-negative")
        if self.candidate_count != 1:
            raise ValueError("forced suffix transitions must have exactly one candidate")


@dataclass(frozen=True, slots=True)
class MacroEdge:
    anchor: SemanticKey
    source_node: IdentityTriple
    chosen_action: IdentityTriple
    policy_step_id: int
    legal_candidate_count: int
    forced_suffix: tuple[ForcedTransition, ...]
    outcome: MacroEdgeOutcome
    closing_step_id: int
    destination_node: IdentityTriple | None = None
    destination_anchor: SemanticKey | None = None

    def __post_init__(self) -> None:
        if self.legal_candidate_count <= 1:
            raise ValueError("a macro edge must start from a genuine policy choice")
        if self.policy_step_id < 0:
            raise ValueError("policy_step_id must be non-negative")
        if self.closing_step_id < self.policy_step_id:
            raise ValueError("macro edge cannot close before its policy choice")
        previous = self.policy_step_id
        for transition in self.forced_suffix:
            if transition.step_id <= previous:
                raise ValueError("forced suffix step ids must be strictly increasing")
            previous = transition.step_id
        if previous > self.closing_step_id:
            raise ValueError("forced suffix extends beyond closing_step_id")

    @property
    def actor_credit_step_id(self) -> int:
        """The only step eligible for direct actor credit."""

        return self.policy_step_id


class MacroEdgeBuilder:
    """Mutable assembly helper; built :class:`MacroEdge` remains immutable."""

    def __init__(
        self,
        *,
        anchor: SemanticKey,
        source_node: IdentityTriple,
        chosen_action: IdentityTriple,
        policy_step_id: int,
        legal_candidate_count: int,
    ) -> None:
        if legal_candidate_count <= 1:
            raise ValueError("forced or singleton actions cannot begin a policy macro edge")
        self._anchor = anchor
        self._source_node = source_node
        self._chosen_action = chosen_action
        self._policy_step_id = policy_step_id
        self._legal_candidate_count = legal_candidate_count
        self._forced: list[ForcedTransition] = []
        self._closed = False

    def append_forced(self, transition: ForcedTransition) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed macro edge")
        previous_step = self._forced[-1].step_id if self._forced else self._policy_step_id
        if transition.step_id <= previous_step:
            raise ValueError("forced suffix step ids must be strictly increasing")
        self._forced.append(transition)

    def close(
        self,
        *,
        outcome: MacroEdgeOutcome,
        closing_step_id: int,
        destination_node: IdentityTriple | None = None,
        destination_anchor: SemanticKey | None = None,
    ) -> MacroEdge:
        if self._closed:
            raise RuntimeError("macro edge builder may close only once")
        self._closed = True
        return MacroEdge(
            anchor=self._anchor,
            source_node=self._source_node,
            chosen_action=self._chosen_action,
            policy_step_id=self._policy_step_id,
            legal_candidate_count=self._legal_candidate_count,
            forced_suffix=tuple(self._forced),
            outcome=outcome,
            closing_step_id=closing_step_id,
            destination_node=destination_node,
            destination_anchor=destination_anchor,
        )
