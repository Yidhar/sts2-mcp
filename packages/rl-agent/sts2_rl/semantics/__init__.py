"""Versioned Decision Semantics Kernel.

This package is deliberately independent from collector, learner and model
code.  It defines factual identities and transition evidence; training policy
is layered on top by later integration.
"""

from .clock import (
    DECISION_CLOCK_BASE,
    DECISION_CLOCK_CONTRACT_VERSION,
    DecisionClockTick,
    clock_tick,
    decision_discount,
)
from .compiler import (
    SEMANTIC_COMPILER_CONTRACT_VERSION,
    CompiledEvent,
    CompiledKind,
    SemanticActionCompiler,
)
from .contracts import (
    DecisionSemantics,
    SemanticAction,
    SurfaceAdapter,
    SurfaceSpec,
)
from .grouping import (
    STRICT_ACTION_GROUPING_CONTRACT_VERSION,
    StrictActionGroup,
    card_selection_operation,
    semantic_card_projection,
    strict_action_grouping_contract,
    strict_action_groups,
)
from .identity import (
    CoarseActionCollisionError,
    IdentityTriple,
    SemanticCollisionError,
    SemanticContractError,
    SemanticKey,
    SemanticKeyIndex,
    canonical_payload_bytes,
)
from .kernel import (
    DECISION_IDENTITY_CONTRACT_VERSION,
    DecisionSemanticsKernel,
)
from .macro_edges import (
    MACRO_EDGE_CONTRACT_VERSION,
    ForcedTransition,
    MacroEdge,
    MacroEdgeBuilder,
    MacroEdgeOutcome,
)
from .progress import (
    PROGRESS_RECEIPT_CONTRACT_VERSION,
    ProgressKind,
    ProgressReceipt,
)
from .registry import (
    SURFACE_REGISTRY_CONTRACT_VERSION,
    SurfaceRegistry,
    default_surface_registry,
)
from .scopes import (
    PROGRESS_SCOPE_CONTRACT_VERSION,
    ProgressScope,
    ProgressScopeStack,
    SurfaceRole,
)

__all__ = [
    "DECISION_CLOCK_BASE",
    "DECISION_CLOCK_CONTRACT_VERSION",
    "DECISION_IDENTITY_CONTRACT_VERSION",
    "MACRO_EDGE_CONTRACT_VERSION",
    "PROGRESS_RECEIPT_CONTRACT_VERSION",
    "PROGRESS_SCOPE_CONTRACT_VERSION",
    "SEMANTIC_COMPILER_CONTRACT_VERSION",
    "STRICT_ACTION_GROUPING_CONTRACT_VERSION",
    "SURFACE_REGISTRY_CONTRACT_VERSION",
    "CoarseActionCollisionError",
    "CompiledEvent",
    "CompiledKind",
    "DecisionClockTick",
    "DecisionSemantics",
    "DecisionSemanticsKernel",
    "ForcedTransition",
    "IdentityTriple",
    "MacroEdge",
    "MacroEdgeBuilder",
    "MacroEdgeOutcome",
    "ProgressKind",
    "ProgressReceipt",
    "ProgressScope",
    "ProgressScopeStack",
    "SemanticAction",
    "SemanticActionCompiler",
    "SemanticCollisionError",
    "SemanticContractError",
    "SemanticKey",
    "SemanticKeyIndex",
    "StrictActionGroup",
    "SurfaceAdapter",
    "SurfaceRegistry",
    "SurfaceRole",
    "SurfaceSpec",
    "canonical_payload_bytes",
    "card_selection_operation",
    "clock_tick",
    "decision_discount",
    "default_surface_registry",
    "semantic_card_projection",
    "strict_action_grouping_contract",
    "strict_action_groups",
]
