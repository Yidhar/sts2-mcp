"""Versioned, deterministic registry for semantic surface adapters."""

from __future__ import annotations

import json
import weakref
from collections.abc import Mapping, Sequence
from typing import Any, Final

from .contracts import SurfaceAdapter
from .identity import SemanticContractError, SemanticKey, SemanticKeyIndex
from .scopes import SurfaceRole
from .surfaces import (
    CombatSurfaceAdapter,
    EventSurfaceAdapter,
    MapSurfaceAdapter,
    OpaqueSurfaceAdapter,
    RestSurfaceAdapter,
    RewardSurfaceAdapter,
    SelectionSurfaceAdapter,
    ShopSurfaceAdapter,
)

SURFACE_REGISTRY_CONTRACT_VERSION: Final = "sts2-surface-registry-v1"


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _action_kind(action: Mapping[str, Any]) -> str:
    for key in ("model_action_kind", "kind", "action"):
        value = action.get(key)
        if value is not None and str(value).strip():
            return _normalize_token(value)
    return ""


class SurfaceRegistry:
    def __init__(self, adapters: Sequence[SurfaceAdapter] = ()) -> None:
        self._adapters: dict[str, SurfaceAdapter] = {}
        self._phase_roots: dict[str, SurfaceAdapter] = {}
        self._state_type_roots: dict[str, SurfaceAdapter] = {}
        self._action_kind_roots: dict[str, SurfaceAdapter] = {}
        # Manifest caches.  The adapter set only changes through ``register``,
        # which invalidates all three, so cached values always describe the
        # current registry.  The JSON string is immutable and re-decoded per
        # call; the interned-index set records which ``SemanticKeyIndex``
        # instances already hold this manifest payload for collision auditing.
        self._manifest_json: str | None = None
        self._manifest_key: SemanticKey | None = None
        self._manifest_interned: weakref.WeakSet[SemanticKeyIndex] = weakref.WeakSet()
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: SurfaceAdapter) -> None:
        existing = self._adapters.get(adapter.spec.spec_id)
        if existing is not None:
            raise ValueError(f"surface adapter {adapter.spec.spec_id!r} is already registered")
        if adapter.spec.role is SurfaceRole.ROOT:
            self._validate_authoritative_claims(adapter)
        self._adapters[adapter.spec.spec_id] = adapter
        if adapter.spec.role is SurfaceRole.ROOT:
            self._index_authoritative_claims(adapter)
        self._manifest_json = None
        self._manifest_key = None
        self._manifest_interned.clear()

    def _validate_authoritative_claims(self, adapter: SurfaceAdapter) -> None:
        claims = (
            ("phase", self._phase_roots, adapter.spec.authoritative_phases),
            (
                "state_type",
                self._state_type_roots,
                adapter.spec.authoritative_state_types,
            ),
            (
                "action kind",
                self._action_kind_roots,
                adapter.spec.authoritative_action_kinds,
            ),
        )
        for label, index, values in claims:
            for raw_value in values:
                value = _normalize_token(raw_value)
                existing = index.get(value)
                if existing is not None:
                    raise ValueError(
                        f"{label} {value!r} is authoritatively claimed by both "
                        f"{existing.spec.spec_id!r} and {adapter.spec.spec_id!r}"
                    )

    def _index_authoritative_claims(self, adapter: SurfaceAdapter) -> None:
        for value in adapter.spec.authoritative_phases:
            self._phase_roots[_normalize_token(value)] = adapter
        for value in adapter.spec.authoritative_state_types:
            self._state_type_roots[_normalize_token(value)] = adapter
        for value in adapter.spec.authoritative_action_kinds:
            self._action_kind_roots[_normalize_token(value)] = adapter

    @property
    def adapters(self) -> tuple[SurfaceAdapter, ...]:
        return tuple(
            sorted(
                self._adapters.values(),
                key=lambda adapter: (
                    -adapter.spec.priority,
                    adapter.spec.spec_id,
                ),
            )
        )

    def by_id(self, spec_id: str) -> SurfaceAdapter:
        try:
            return self._adapters[spec_id]
        except KeyError as error:
            raise KeyError(f"unknown surface adapter {spec_id!r}") from error

    def resolve_root(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> SurfaceAdapter:
        action_kinds = {_action_kind(action) for action in legal_actions}
        action_kinds.discard("")
        authoritative_actions = {
            adapter.spec.spec_id: adapter
            for kind in action_kinds
            if (adapter := self._action_kind_roots.get(kind)) is not None
        }
        if len(authoritative_actions) > 1:
            raise SemanticContractError(
                f"legal actions claim conflicting root surfaces: {tuple(sorted(authoritative_actions))!r}"
            )
        if authoritative_actions:
            return next(iter(authoritative_actions.values()))

        state_type = _normalize_token(observation.get("state_type"))
        authoritative_state = self._state_type_roots.get(state_type)
        if authoritative_state is not None:
            return authoritative_state

        phase = _normalize_token(observation.get("phase"))
        authoritative_phase = self._phase_roots.get(phase)
        if authoritative_phase is not None:
            return authoritative_phase

        for adapter in self.adapters:
            if adapter.spec.role is SurfaceRole.ROOT and adapter.matches(
                observation,
                legal_actions,
            ):
                return adapter
        raise RuntimeError("surface registry requires an opaque root fallback")

    def resolve_overlays(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
    ) -> tuple[SurfaceAdapter, ...]:
        return tuple(
            adapter
            for adapter in self.adapters
            if adapter.spec.role is SurfaceRole.OVERLAY and adapter.matches(observation, legal_actions)
        )

    def manifest_payload(self) -> dict[str, Any]:
        if self._manifest_json is None:
            self._manifest_json = json.dumps(
                {
                    "contract_version": SURFACE_REGISTRY_CONTRACT_VERSION,
                    "adapters": [
                        adapter.spec.to_manifest()
                        for adapter in sorted(
                            self._adapters.values(),
                            key=lambda item: item.spec.spec_id,
                        )
                    ],
                }
            )
        payload: dict[str, Any] = json.loads(self._manifest_json)
        return payload

    def manifest_key(self, key_index: SemanticKeyIndex) -> SemanticKey:
        cached = self._manifest_key
        if cached is not None and key_index in self._manifest_interned:
            # ``key_index`` already interned exactly this canonical payload,
            # so repeating ``intern`` would re-store identical bytes and
            # return an equal key.  Skipping it loses no collision audit.
            return cached
        key = key_index.intern(
            namespace="surface_registry_manifest",
            schema_version=SURFACE_REGISTRY_CONTRACT_VERSION,
            payload=self.manifest_payload(),
        )
        self._manifest_key = key
        self._manifest_interned.add(key_index)
        return key


def default_surface_registry() -> SurfaceRegistry:
    return SurfaceRegistry(
        (
            CombatSurfaceAdapter(),
            EventSurfaceAdapter(),
            RestSurfaceAdapter(),
            ShopSurfaceAdapter(),
            RewardSurfaceAdapter(),
            MapSurfaceAdapter(),
            SelectionSurfaceAdapter(),
            OpaqueSurfaceAdapter(),
        )
    )
