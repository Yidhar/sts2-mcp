"""Nested progress scopes preserving root ownership across UI overlays."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from .identity import SemanticKey

PROGRESS_SCOPE_CONTRACT_VERSION: Final = "sts2-progress-scope-v1"


class SurfaceRole(StrEnum):
    ROOT = "root"
    OVERLAY = "overlay"


@dataclass(frozen=True, slots=True)
class ProgressScope:
    spec_id: str
    spec_version: str
    role: SurfaceRole
    identity: SemanticKey

    def __post_init__(self) -> None:
        if not self.spec_id.strip() or not self.spec_version.strip():
            raise ValueError("progress scope spec identity must be non-empty")


@dataclass(frozen=True, slots=True)
class ProgressScopeStack:
    """A root surface followed by zero or more nested overlays."""

    scopes: tuple[ProgressScope, ...]

    def __post_init__(self) -> None:
        if not self.scopes:
            raise ValueError("progress scope stack requires one root scope")
        if self.scopes[0].role is not SurfaceRole.ROOT:
            raise ValueError("the first progress scope must be a root surface")
        if any(scope.role is not SurfaceRole.OVERLAY for scope in self.scopes[1:]):
            raise ValueError("all nested progress scopes must be overlays")

    @property
    def root(self) -> ProgressScope:
        return self.scopes[0]

    @property
    def active(self) -> ProgressScope:
        return self.scopes[-1]

    @property
    def overlays(self) -> tuple[ProgressScope, ...]:
        return self.scopes[1:]

    def push(self, overlay: ProgressScope) -> ProgressScopeStack:
        if overlay.role is not SurfaceRole.OVERLAY:
            raise ValueError("only overlay scopes may be pushed")
        return ProgressScopeStack((*self.scopes, overlay))

    def pop(self) -> ProgressScopeStack:
        if len(self.scopes) == 1:
            raise ValueError("the root progress scope cannot be popped")
        return ProgressScopeStack(self.scopes[:-1])

    def root_only(self) -> ProgressScopeStack:
        return ProgressScopeStack((self.root,))
