"""Pure Decision Semantics Kernel, independent of collector and learner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .contracts import (
    DecisionSemantics,
    SemanticAction,
    SurfaceAdapter,
)
from .grouping import strict_action_groups
from .identity import (
    IdentityTriple,
    SemanticKey,
    SemanticKeyIndex,
    verify_coarse_action_injectivity,
)
from .progress import ProgressKind, ProgressReceipt
from .registry import SurfaceRegistry, default_surface_registry
from .scopes import (
    PROGRESS_SCOPE_CONTRACT_VERSION,
    ProgressScope,
    ProgressScopeStack,
)

DECISION_IDENTITY_CONTRACT_VERSION: Final = "sts2-decision-identity-v1"


class DecisionSemanticsKernel:
    """Build exact/loop/comparison identities from one factual decision."""

    def __init__(
        self,
        *,
        registry: SurfaceRegistry | None = None,
        key_index: SemanticKeyIndex | None = None,
    ) -> None:
        self.registry = registry or default_surface_registry()
        self.key_index = key_index or SemanticKeyIndex()

    def _key(self, namespace: str, payload: Any) -> SemanticKey:
        return self.key_index.intern(
            namespace=namespace,
            schema_version=DECISION_IDENTITY_CONTRACT_VERSION,
            payload=payload,
        )

    def _root_adapter(
        self,
        *,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
        parent_scopes: ProgressScopeStack | None,
        overlays: Sequence[SurfaceAdapter],
    ) -> tuple[SurfaceAdapter, SemanticKey | None]:
        resolved = self.registry.resolve_root(observation, legal_actions)
        if resolved.spec.spec_id == "opaque" and overlays and parent_scopes is not None:
            inherited = self.registry.by_id(parent_scopes.root.spec_id)
            return inherited, parent_scopes.root.identity
        return resolved, None

    def identify(
        self,
        *,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
        parent_scopes: ProgressScopeStack | None = None,
    ) -> DecisionSemantics:
        overlays = self.registry.resolve_overlays(observation, legal_actions)
        root_adapter, inherited_anchor = self._root_adapter(
            observation=observation,
            legal_actions=legal_actions,
            parent_scopes=parent_scopes,
            overlays=overlays,
        )
        anchor = inherited_anchor or self._key(
            f"anchor:{root_adapter.spec.spec_id}",
            root_adapter.anchor_payload(observation),
        )
        root_scope = ProgressScope(
            spec_id=root_adapter.spec.spec_id,
            spec_version=root_adapter.spec.version,
            role=root_adapter.spec.role,
            identity=anchor,
        )
        scopes = ProgressScopeStack((root_scope,))
        overlay_payloads: list[dict[str, Any]] = []
        for overlay in overlays:
            payload = overlay.loop_node_payload(observation, legal_actions)
            scope_key = self.key_index.intern(
                namespace=f"scope:{overlay.spec.spec_id}",
                schema_version=PROGRESS_SCOPE_CONTRACT_VERSION,
                payload=payload,
            )
            scope = ProgressScope(
                spec_id=overlay.spec.spec_id,
                spec_version=overlay.spec.version,
                role=overlay.spec.role,
                identity=scope_key,
            )
            scopes = scopes.push(scope)
            overlay_payloads.append(
                {
                    "spec_id": overlay.spec.spec_id,
                    "version": overlay.spec.version,
                    "node": payload,
                }
            )

        exact_payload = {
            "root_spec": root_adapter.spec.spec_id,
            "root_node": root_adapter.exact_node_payload(
                observation,
                legal_actions,
            ),
            "overlays": [
                {
                    "spec_id": overlay.spec.spec_id,
                    "node": overlay.exact_node_payload(
                        observation,
                        legal_actions,
                    ),
                }
                for overlay in overlays
            ],
        }
        loop_payload = {
            "anchor": anchor.payload,
            "root_spec": root_adapter.spec.spec_id,
            "root_node": root_adapter.loop_node_payload(
                observation,
                legal_actions,
            ),
            "overlays": overlay_payloads,
        }
        comparison_payload = {
            "root_spec": root_adapter.spec.spec_id,
            "root_node": root_adapter.comparison_node_payload(
                observation,
                legal_actions,
            ),
            "overlays": [
                {
                    "spec_id": overlay.spec.spec_id,
                    "node": overlay.comparison_node_payload(
                        observation,
                        legal_actions,
                    ),
                }
                for overlay in overlays
            ],
        }
        node = IdentityTriple(
            exact=self._key("decision_node:exact", exact_payload),
            loop=self._key("decision_node:loop", loop_payload),
            comparison=self._key(
                "decision_node:comparison",
                comparison_payload,
            ),
        )

        action_adapter = overlays[-1] if overlays else root_adapter
        groups = strict_action_groups(legal_actions)
        semantic_actions: list[SemanticAction] = []
        for group in groups:
            action = group.prototype
            identities = IdentityTriple(
                exact=self._key(
                    "decision_action:exact",
                    {
                        "surface": action_adapter.spec.spec_id,
                        "action": action_adapter.exact_action_payload(action),
                    },
                ),
                loop=self._key(
                    "decision_action:loop",
                    {
                        "surface": action_adapter.spec.spec_id,
                        "action": action_adapter.loop_action_payload(action),
                    },
                ),
                comparison=self._key(
                    "decision_action:comparison",
                    {
                        "surface": action_adapter.spec.spec_id,
                        "action": action_adapter.comparison_action_payload(action),
                    },
                ),
            )
            semantic_actions.append(
                SemanticAction(
                    identities=identities,
                    group=group,
                )
            )
        verify_coarse_action_injectivity(
            [action.identities for action in semantic_actions],
            strict_equivalence_fingerprints=[action.group.equivalence_fingerprint for action in semantic_actions],
        )
        return DecisionSemantics(
            manifest=self.registry.manifest_key(self.key_index),
            scopes=scopes,
            node=node,
            actions=tuple(semantic_actions),
        )

    def classify_transition(
        self,
        *,
        before: DecisionSemantics,
        after: DecisionSemantics,
        before_observation: Mapping[str, Any],
        after_observation: Mapping[str, Any],
    ) -> ProgressReceipt:
        if before.scopes.root.spec_id != after.scopes.root.spec_id:
            return ProgressReceipt(
                kind=ProgressKind.FLOW_ADVANCE,
                source="kernel:root_surface_changed",
            )
        adapter = self.registry.by_id(before.scopes.root.spec_id)
        receipt = adapter.classify_progress(
            before_observation=before_observation,
            after_observation=after_observation,
            before_anchor=before.anchor,
            after_anchor=after.anchor,
        )
        if receipt.kind in {
            ProgressKind.FLOW_ADVANCE,
            ProgressKind.DURABLE_COMMIT,
            ProgressKind.CONTROL_MOVE,
            ProgressKind.COST_ONLY,
        }:
            return receipt
        scope_changed = tuple((scope.spec_id, scope.identity.digest) for scope in before.scopes.scopes) != tuple(
            (scope.spec_id, scope.identity.digest) for scope in after.scopes.scopes
        )
        if scope_changed or before.node.loop != after.node.loop:
            return ProgressReceipt(
                kind=ProgressKind.CONTROL_MOVE,
                source="kernel:reviewed_control_identity_changed",
                changed_paths=receipt.changed_paths,
                details=(
                    ("before_active_scope", before.scopes.active.spec_id),
                    ("after_active_scope", after.scopes.active.spec_id),
                    ("before_loop", before.node.loop.digest),
                    ("after_loop", after.node.loop.digest),
                ),
            )
        return receipt
