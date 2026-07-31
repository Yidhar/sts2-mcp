"""Public, versioned contracts for the Decision Semantics Kernel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .grouping import StrictActionGroup
from .identity import IdentityTriple, SemanticKey
from .progress import ProgressReceipt
from .scopes import ProgressScopeStack, SurfaceRole


@dataclass(frozen=True, slots=True)
class SurfaceSpec:
    spec_id: str
    version: str
    role: SurfaceRole
    priority: int
    claimed_paths: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    allows_direct_policy_credit: bool = True
    authoritative_phases: tuple[str, ...] = ()
    authoritative_state_types: tuple[str, ...] = ()
    authoritative_action_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.spec_id.strip() or not self.version.strip():
            raise ValueError("surface spec identity and version must be non-empty")
        if self.priority < 0:
            raise ValueError("surface spec priority must be non-negative")
        if not self.claimed_paths:
            raise ValueError("surface spec must claim at least one DTO path")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "version": self.version,
            "role": self.role.value,
            "priority": self.priority,
            "claimed_paths": list(self.claimed_paths),
            "aliases": list(self.aliases),
            "allows_direct_policy_credit": self.allows_direct_policy_credit,
            "authoritative_phases": list(self.authoritative_phases),
            "authoritative_state_types": list(self.authoritative_state_types),
            "authoritative_action_kinds": list(self.authoritative_action_kinds),
        }


@runtime_checkable
class SurfaceAdapter(Protocol):
    spec: SurfaceSpec

    def matches(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> bool: ...

    def anchor_payload(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def exact_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...

    def loop_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...

    def comparison_node_payload(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...

    def exact_action_payload(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def loop_action_payload(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def comparison_action_payload(
        self,
        action: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def classify_progress(
        self,
        *,
        before_observation: Mapping[str, Any],
        after_observation: Mapping[str, Any],
        before_anchor: SemanticKey,
        after_anchor: SemanticKey,
    ) -> ProgressReceipt: ...


@dataclass(frozen=True, slots=True)
class SemanticAction:
    identities: IdentityTriple
    group: StrictActionGroup


@dataclass(frozen=True, slots=True)
class DecisionSemantics:
    manifest: SemanticKey
    scopes: ProgressScopeStack
    node: IdentityTriple
    actions: tuple[SemanticAction, ...]

    @property
    def anchor(self) -> SemanticKey:
        return self.scopes.root.identity

    @property
    def is_policy_choice(self) -> bool:
        return len(self.actions) > 1
