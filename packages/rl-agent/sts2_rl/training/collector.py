"""Typed recurrent actor collecting fixed-length v2 sequence unrolls."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Literal, Protocol
from uuid import uuid4

import numpy as np
import torch
from numpy.typing import NDArray

from sts2_baseline import (
    RolloutStep,
    SequenceUnroll,
    TaskReward,
    TaskRewardCalculator,
)
from sts2_rl.contracts import (
    ENVIRONMENT_SCHEMA_VERSION,
    CombatResetRequest,
    EnvironmentBackend,
    EnvironmentResult,
    ResetRequest,
    StepRequest,
)
from sts2_rl.encoding import (
    ActionReference,
    EncodedDecisionSnapshot,
    GroundedObservationEncoder,
    SemanticActionGroup,
)
from sts2_rl.models import RecurrentCandidateModel

from .episode_replay import (
    ActSegmentHealth,
    BoundaryOutcome,
    CompletedEpisode,
    EpisodeCompletion,
    EpisodeDecisionStep,
    backfill_completed_episode,
)
from .failure_credit import (
    EvidenceRecord,
    FailureCreditEpisodePipeline,
    FailureCreditPipelineConfig,
    FailureCreditShadowMetrics,
)
from .runaway_combat import RunawayCombatGuard
from .seeding import (
    EVALUATION_SEED_PARITY,
    SIGNED_INT32_MAX,
    training_seed_start,
)
from .trajectory import (
    SemanticDeadlockDetector,
    TrajectoryJournal,
    canonical_json,
    semantic_action_fingerprint,
    semantic_decision_fingerprint,
    semantic_fingerprint,
    semantic_projection,
)
from .transaction import (
    TransactionEffect,
    TransactionLifecycleEvidence,
    TransactionLifecycleOutcome,
    TransactionOutcome,
    TransactionStep,
    TransactionTrace,
    backfill_factual_monte_carlo_returns,
    factual_transaction_policy_targets,
)
from .transaction_operations import (
    TRANSACTION_EXPLORATION_OPERATIONS,
    TRANSACTION_GUIDANCE_OPERATIONS,
    canonical_transaction_operation,
)


class CollectionProtocolError(RuntimeError):
    """The environment exposed no dispatchable legal candidate."""


class RewardCalculator(Protocol):
    def evaluate(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
        *,
        deadlock: bool = False,
        horizon_exhausted: bool = False,
    ) -> TaskReward: ...


@dataclass(frozen=True, slots=True)
class MacroReturnDiagnostic:
    """Non-learning factual return aggregate for one macro decision slice."""

    decision_surface: str
    action_key: str
    hp_band: str
    act: int
    count: int
    mean_return: float

    def __post_init__(self) -> None:
        if not self.decision_surface or not self.action_key or not self.hp_band:
            raise ValueError("macro return diagnostic keys must be non-empty")
        if self.act < 0 or self.count <= 0:
            raise ValueError("macro return diagnostic counters are invalid")
        if not math.isfinite(self.mean_return):
            raise ValueError("macro return diagnostic mean must be finite")


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    episode_id: str
    reset_seed: int
    steps: int
    reward_total: float
    terminal_reason: str | None
    truncated: bool
    run_won: bool
    combat_won: bool
    act1_cleared: bool
    max_act: int
    max_floor: int
    policy_decisions: int
    forced_decisions: int
    maximum_observed_candidates: int
    deadlocked: bool
    combat_progress_stalled: bool
    maximum_combat_no_net_progress_steps: int
    noncombat_progress_stalled: bool
    maximum_noncombat_no_durable_progress_steps: int
    revivals_used: int
    revival_free_combat_win: bool
    revival_free_act1_clear: bool
    revival_free_run_win: bool
    player_hp_lost: float
    stall_evidence: dict[str, object] | None
    # A collector-observed combat liveness failure is a trustworthy policy
    # outcome even though the simulator has not ended the native run. Keep it
    # separate from run_defeat (environment outcome) and transport incidents
    # (which never produce EpisodeMetrics).
    combat_policy_failed: bool = False
    # Generic confirmed policy-liveness failures include exact semantic cycles
    # and recurrent non-combat event-page actions.  They are authoritative task
    # losses even when the engine bailout keeps the simulator run alive.
    trusted_policy_failure: bool = False
    noncombat_event_cycle: bool = False
    selection_action_cycle: bool = False
    maximum_observed_semantic_candidates: int = 0
    maximum_equivalence_class_size: int = 0
    # Cumulative efficiency counters at successful, real Act boundaries.  The
    # tuples are ordered by completed Act and deliberately exclude the
    # bootstrap act-0 -> act-1 transition.  Defaults keep archived metrics and
    # focused tests source-compatible.
    act_revival_counts: tuple[int, ...] = ()
    act_hp_loss_counts: tuple[float, ...] = ()
    maximum_definition_hash_collisions_per_decision: int = 0
    maximum_relation_hash_collisions_per_decision: int = 0
    definition_hash_collisions_total: int = 0
    relation_hash_collisions_total: int = 0
    targeted_selection_exploration_decisions: int = 0
    targeted_transaction_entry_exploration_decisions: int = 0
    transaction_completion_guidance_decisions: int = 0
    transaction_completion_forward_decisions: int = 0
    transaction_completion_guidance_fallbacks: int = 0
    maximum_effective_collection_epsilon: float = 0.0
    policy_top1_top2_logit_margin_mean: float = 0.0
    policy_top1_top2_logit_margin_max: float = 0.0
    selection_transactions_started: int = 0
    selection_transactions_closed: int = 0
    # ``completed`` is the compatibility name for a verified commit, not a
    # generic page exit.  Cancelled and unresolved teardowns are reported
    # separately so monitoring cannot call a rollback successful.
    selection_transactions_completed: int = 0
    selection_transactions_cancelled: int = 0
    selection_transactions_unresolved: int = 0
    rest_site_selection_transactions_started: int = 0
    rest_site_selection_transactions_closed: int = 0
    rest_site_selection_transactions_completed: int = 0
    rest_site_selection_transactions_cancelled: int = 0
    rest_site_selection_transactions_unresolved: int = 0
    forge_selection_transactions_started: int = 0
    forge_selection_transactions_closed: int = 0
    forge_selection_transactions_completed: int = 0
    forge_selection_transactions_cancelled: int = 0
    forge_selection_transactions_unresolved: int = 0
    shop_card_removal_transactions_started: int = 0
    shop_card_removal_transactions_closed: int = 0
    shop_card_removal_transactions_completed: int = 0
    shop_card_removal_transactions_cancelled: int = 0
    shop_card_removal_transactions_unresolved: int = 0
    shaping_reward_total: float = 0.0
    shaping_reward_per_max_floor: float = 0.0
    boss_victory_acts: tuple[int, ...] = ()
    macro_return_diagnostics: tuple[MacroReturnDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class CollectedEpisode:
    unrolls: tuple[SequenceUnroll, ...]
    metrics: EpisodeMetrics
    actor_policy_version: int
    behavior_policy_version: int
    timings: CollectorTimings | None = None
    transaction_traces: tuple[TransactionTrace, ...] = ()
    completed_episode: CompletedEpisode | None = None
    liveness_probe: bool = False
    # Formal v4 evidence is kept separate from legacy transaction-v3 traces.
    # In shadow mode these immutable records are audited but never sampled by
    # the learner; learning mode may insert exactly this tuple into replay-v5.
    failure_credit_records: tuple[EvidenceRecord, ...] = ()
    failure_credit_shadow_metrics: FailureCreditShadowMetrics | None = None


@dataclass(frozen=True, slots=True)
class EpisodeProgress:
    """Compact actor progress published once per completed recurrent unroll."""

    episode_id: str
    reset_seed: int
    steps: int
    reward_total: float
    max_act: int
    max_floor: int
    policy_decisions: int
    forced_decisions: int
    maximum_observed_candidates: int
    revivals_used: int
    player_hp_lost: float
    combat_in_progress: bool
    phase: str
    decision_domain: str
    combat_no_net_progress_steps: int
    noncombat_no_durable_progress_steps: int
    combat_anchor_enemy_hp_total: float
    combat_required_net_hp_progress: float
    enemy_hp_total: float
    enemy_max_hp_total: float
    hand_cards: int
    draw_cards: int
    discard_cards: int
    exhaust_cards: int
    legal_action_kinds: dict[str, int]
    selected_action_kinds: dict[str, int]
    last_selected_action_kind: str
    behavior_policy_version: int
    maximum_observed_semantic_candidates: int = 0
    maximum_equivalence_class_size: int = 0
    combat_net_progress_window: int = 0
    combat_progress_window_source: str = "default"
    combat_progress_window_match_id: str = ""
    maximum_definition_hash_collisions_per_decision: int = 0
    maximum_relation_hash_collisions_per_decision: int = 0
    definition_hash_collisions_total: int = 0
    relation_hash_collisions_total: int = 0


@dataclass(frozen=True, slots=True)
class CollectorStageTiming:
    count: int
    total_ms: float
    min_ms: float
    max_ms: float

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.total_ms / self.count,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True, slots=True)
class CollectorTimings:
    """Episode-aggregated collector timings; never written per environment step."""

    stages: dict[str, CollectorStageTiming]

    def to_mapping(self) -> dict[str, dict[str, float | int]]:
        return {name: timing.to_mapping() for name, timing in sorted(self.stages.items())}


@dataclass(frozen=True, slots=True)
class _ActionChoice:
    """One model-space choice and its non-learned dispatch projection.

    ``candidate_index`` is the index whose probability was sampled and is the
    only index that may enter rollout/replay learning.  ``dispatch_position``
    and ``dispatch_handle`` identify the representative raw backend action for
    that semantic candidate.  They deliberately remain outside model tensors.
    """

    candidate_index: int
    dispatch_position: int
    dispatch_handle: str | None
    equivalence_fingerprint: str | None
    multiplicity: int
    action_references: tuple[ActionReference, ...]
    semantic_actions: tuple[Mapping[str, object], ...]
    behavior_log_probability: float
    model_log_probability: float
    valid_count: int
    snapshot: EncodedDecisionSnapshot
    recurrent_state: torch.Tensor
    policy: NDArray[np.float32]
    value: float
    encoding_ms: float
    policy_forward_ms: float
    definition_hash_collisions: int
    relation_hash_collisions: int
    effective_epsilon: float
    targeted_selection_exploration: bool
    targeted_transaction_entry_exploration: bool
    transaction_completion_guidance: bool
    transaction_completion_forward_selected: bool
    transaction_completion_guidance_fallback: bool
    transaction_operation: str


@dataclass(slots=True)
class _MutableStageTiming:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def add(self, duration_ms: float) -> None:
        duration_ms = max(0.0, float(duration_ms))
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)


class _CollectorTimingAccumulator:
    def __init__(self) -> None:
        self._stages: dict[str, _MutableStageTiming] = {}

    def add(self, stage: str, duration_ms: float) -> None:
        self._stages.setdefault(stage, _MutableStageTiming()).add(duration_ms)

    def record(self, stage: str, started_ns: int) -> None:
        self.add(stage, (time.perf_counter_ns() - started_ns) / 1_000_000.0)

    def snapshot(self) -> CollectorTimings:
        return CollectorTimings(
            stages={
                name: CollectorStageTiming(
                    count=timing.count,
                    total_ms=timing.total_ms,
                    min_ms=timing.min_ms,
                    max_ms=timing.max_ms,
                )
                for name, timing in self._stages.items()
            }
        )


def _branch_balanced_epsilon_behavior(
    *,
    policy: NDArray[np.float32],
    valid: NDArray[np.bool_],
    policy_branch_ids: NDArray[np.int64],
    epsilon: float,
) -> NDArray[np.float64]:
    """Mix the target policy with a count-balanced hierarchical explorer.

    The learned policy first chooses one semantic action branch and then one
    concrete action inside that branch. Epsilon exploration must use the same
    factorization. A flat ``epsilon / valid_candidate_count`` explorer would
    make a five-card ``play_card`` branch receive five times the exploratory
    mass of singleton ``end_turn`` and would reintroduce the exact candidate
    cardinality bias that the hierarchical target policy removes.

    Strictly grouped duplicate backend actions are already represented by one
    candidate here. Their raw multiplicity therefore cannot silently increase
    exploratory probability.
    """

    if policy.ndim != 1 or valid.ndim != 1 or policy_branch_ids.ndim != 1:
        raise CollectionProtocolError("policy, valid mask, and policy branch IDs must be one-dimensional")
    if policy.shape != valid.shape or policy.shape != policy_branch_ids.shape:
        raise CollectionProtocolError("policy, valid mask, and policy branch IDs have different shapes")
    normalized_epsilon = float(epsilon)
    if not math.isfinite(normalized_epsilon) or not 0.0 <= normalized_epsilon <= 1.0:
        raise ValueError("exploration epsilon must be finite and in [0, 1]")

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise CollectionProtocolError("cannot build behavior policy without a legal action")
    valid_policy = policy[valid_indices].astype(np.float64, copy=False)
    if not np.all(np.isfinite(valid_policy)) or np.any(valid_policy < 0.0):
        raise CollectionProtocolError("model produced a non-finite policy distribution")
    policy_mass = float(valid_policy.sum())
    if not math.isfinite(policy_mass) or policy_mass <= 0.0:
        raise CollectionProtocolError("model produced zero/non-finite legal policy mass")

    valid_branch_ids = policy_branch_ids[valid_indices]
    if np.any(valid_branch_ids < 0):
        raise CollectionProtocolError("model produced a negative legal policy branch ID")
    _, inverse, branch_candidate_counts = np.unique(
        valid_branch_ids,
        return_inverse=True,
        return_counts=True,
    )
    branch_count = int(branch_candidate_counts.size)
    if branch_count <= 0:  # pragma: no cover - valid_indices makes this impossible
        raise CollectionProtocolError("model produced no legal policy branch")

    exploration = np.zeros(policy.shape, dtype=np.float64)
    exploration[valid_indices] = 1.0 / (float(branch_count) * branch_candidate_counts[inverse].astype(np.float64))
    behavior = np.zeros(policy.shape, dtype=np.float64)
    behavior[valid_indices] = (
        1.0 - normalized_epsilon
    ) * valid_policy / policy_mass + normalized_epsilon * exploration[valid_indices]
    behavior_mass = float(behavior.sum())
    if not math.isfinite(behavior_mass) or behavior_mass <= 0.0:
        raise CollectionProtocolError("collector produced zero/non-finite behavior policy mass")
    behavior /= behavior_mass
    if not np.all(np.isfinite(behavior)) or np.any(behavior < 0.0) or np.any(behavior[~valid] != 0.0):
        raise CollectionProtocolError("collector produced an invalid behavior policy")
    return behavior


def _number(value: object, default: float = 0.0) -> float:
    if not isinstance(value, str | int | float):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _transaction_action_fingerprint(
    selected_action: Mapping[str, object],
    *,
    equivalence_fingerprint: str | None,
) -> str:
    """Fingerprint the learned semantic action, never a chosen group member.

    Strict action grouping supplies a fingerprint of the complete normalized
    equivalence payload.  Falling back to the representative raw action keeps
    pre-grouping/non-groupable decisions byte-for-byte compatible.
    """

    if equivalence_fingerprint is not None:
        normalized = str(equivalence_fingerprint).strip()
        if not normalized:
            raise CollectionProtocolError("grouped candidate exposed an empty equivalence fingerprint")
        return normalized
    return semantic_action_fingerprint(selected_action)


_SEMANTIC_ACTION_SURFACE_KIND = "strict_action_group"
# Transaction batches are replayed recurrently and sampled several traces at a
# time. Keep enough tail decisions to expose short cycles without turning one
# no-progress episode into hundreds of sequential learner forwards.
_LIVENESS_LEARN_TAIL_STEPS = 32


def _semantic_action_surface(
    groups: tuple[SemanticActionGroup, ...],
) -> tuple[Mapping[str, object], ...]:
    """Return an order-stable, counted surface for diagnostics/transactions."""

    return tuple(
        {
            "surface_kind": _SEMANTIC_ACTION_SURFACE_KIND,
            "prototype": group.prototype,
            "multiplicity": group.reference.multiplicity,
            "equivalence_fingerprint": (group.reference.equivalence_fingerprint),
            "enabled": group.reference.enabled,
        }
        for group in groups
    )


def _semantic_action_prototype(
    action: Mapping[str, object],
) -> Mapping[str, object]:
    if action.get("surface_kind") != _SEMANTIC_ACTION_SURFACE_KIND:
        return action
    prototype = action.get("prototype")
    if not isinstance(prototype, Mapping):
        raise CollectionProtocolError("semantic action group lost its prototype")
    return prototype


def _semantic_action_multiplicity(action: Mapping[str, object]) -> int:
    if action.get("surface_kind") != _SEMANTIC_ACTION_SURFACE_KIND:
        return 1
    raw = action.get("multiplicity")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise CollectionProtocolError("semantic action group has invalid multiplicity")
    return raw


def _semantic_surface_action_fingerprint(action: Mapping[str, object]) -> str:
    """Return the exact learned action identity for one semantic candidate."""

    prototype = _semantic_action_prototype(action)
    raw_equivalence = action.get("equivalence_fingerprint")
    equivalence_fingerprint = str(raw_equivalence) if raw_equivalence is not None else None
    return _transaction_action_fingerprint(
        prototype,
        equivalence_fingerprint=equivalence_fingerprint,
    )


def _semantic_action_enabled(action: Mapping[str, object]) -> bool:
    prototype = _semantic_action_prototype(action)
    return bool(action.get("enabled", prototype.get("is_enabled", prototype.get("enabled", True))))


def _semantic_surface_candidate_identity(
    action: Mapping[str, object],
) -> tuple[str, int, bool]:
    """Return all candidate facts that can change the actor's scored input.

    Strictly equal physical actions share one learned candidate, but their
    multiplicity is encoded into that candidate. Liveness recurrence must
    therefore distinguish a group of N equal choices from a group of N-1;
    dropping the count could attach an AVOID label to a different actor node.
    """

    return (
        _semantic_surface_action_fingerprint(action),
        _semantic_action_multiplicity(action),
        _semantic_action_enabled(action),
    )


_TRANSACTION_SURFACE_FIELDS = (
    "mode",
    "prompt_id",
    "operation_type",
    "source_zone",
    "destination_zone",
    "min_select",
    "max_select",
    "requires_manual_confirmation",
    "can_skip",
    "cancelable",
)


_TRANSACTION_DEFINITION_KEYS = (
    "model_id",
    "entity_id",
    "id",
    "card_id",
    "definition_id",
)
_TRANSACTION_INSTANCE_KEYS = (
    "instance_uuid",
    "card_instance_id",
    "instance_id",
    "card_ref",
    "ref",
    "uuid",
    "uid",
)
_TRANSACTION_PHYSICAL_SOURCE_KEYS = (
    "source_pile",
    "source_zone",
    "physical_pile",
    "physical_zone",
)
_TRANSACTION_CARD_IDENTITY_IGNORED_KEYS = frozenset(
    {
        *_TRANSACTION_DEFINITION_KEYS,
        *_TRANSACTION_INSTANCE_KEYS,
        *_TRANSACTION_PHYSICAL_SOURCE_KEYS,
        "index",
        "idx",
        "card_index",
        "choice_index",
        "option_index",
        # These describe the prompt membership of the same card, not its
        # gameplay semantics. The transaction node records membership
        # separately through the selected multiset.
        "selection_membership",
        "is_selected",
        # Some surfaces replace the physical pile with a Selected/Selectable
        # pseudo-zone after a toggle. The stable physical source above is the
        # only zone identity retained here.
        "pile",
    }
)


def _first_nonempty(value: Mapping[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        item = value.get(key)
        if item is not None and str(item).strip():
            return item
    return None


def _transaction_option_identity(
    value: object,
    *,
    include_instance: bool = True,
) -> object:
    """Return a stable selectable-card identity.

    Physical instance IDs, UI ordinals and selection membership may change
    after a toggle and therefore cannot define the semantic node. All other
    visible card facts are retained fail-closed. Consequently two cards with
    the same definition but different upgrades, enchantments, modifiers,
    costs, lifecycle flags or future schema fields never collapse merely
    because their base ID matches. This mirrors the encoder invariant that
    duplicate-action merging is allowed only for strictly equal cards.
    """

    if not isinstance(value, Mapping):
        return {"opaque_identity": semantic_projection(value)}
    nested = value.get("card")
    card = nested if isinstance(nested, Mapping) else value
    definition = _first_nonempty(card, _TRANSACTION_DEFINITION_KEYS)
    instance = _first_nonempty(card, _TRANSACTION_INSTANCE_KEYS)
    physical_source: object | None = None
    for key in _TRANSACTION_PHYSICAL_SOURCE_KEYS:
        item = card.get(key)
        if item is None or not str(item).strip():
            continue
        physical_source = item
        break
    identity: dict[str, object] = {
        "definition": str(definition).strip() if definition is not None else "<unknown>",
    }
    if include_instance and instance is not None:
        identity["instance"] = str(instance).strip()
    if physical_source is not None:
        identity["physical_source"] = str(physical_source).strip()
    projected = semantic_projection(card)
    if isinstance(projected, Mapping):
        facts = {
            str(key): child
            for key, child in projected.items()
            if str(key).lower() not in _TRANSACTION_CARD_IDENTITY_IGNORED_KEYS
        }
        if facts:
            identity["facts"] = facts
    return identity


def _selection_actions(
    legal_actions: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        prototype
        for action in legal_actions
        if str((prototype := _semantic_action_prototype(action)).get("model_action_kind") or "") == "card_selection"
    )


def _transaction_selection_context(
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> Mapping[str, object] | None:
    actions = _selection_actions(legal_actions)
    # The v2 observation contract exposes ``card_selection: {}`` on ordinary
    # combat, map, reward, event, shop and rest decisions.  A mapping's mere
    # presence is therefore not an active transaction.  Requiring a grounded
    # card-selection candidate prevents the replay from incorrectly wrapping
    # whole runs as completed selection transactions.
    if not actions:
        return None
    raw_selection = observation.get("card_selection")
    if isinstance(raw_selection, Mapping):
        return raw_selection
    # Legacy/live surfaces may expose selection semantics only on candidates.
    # Merge the stable prompt contract without depending on candidate order.
    synthesized: dict[str, object] = {"mode": "action-derived"}
    for field in _TRANSACTION_SURFACE_FIELDS:
        values: list[object] = []
        for action in actions:
            nested = action.get("selection")
            item = nested.get(field) if isinstance(nested, Mapping) else None
            if item is None:
                item = action.get(field)
            if item is not None:
                values.append(item)
        if values:
            canonical_values = sorted({canonical_json(semantic_projection(item)) for item in values})
            synthesized[field] = canonical_values[0] if len(canonical_values) == 1 else canonical_values
    return synthesized


def _transaction_option_universe(
    selection: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> tuple[str, ...]:
    """Return an order-independent, compact counted option universe."""

    counted: dict[str, tuple[object, int]] = {}

    def add(identity: object, count: int = 1) -> None:
        key = canonical_json(identity)
        previous = counted.get(key)
        counted[key] = (identity, count + (previous[1] if previous is not None else 0))

    # Grouped legal actions are authoritative when they expose card-bearing
    # select/deselect operations.  This keeps a 2,040-Wound prompt as one
    # counted semantic option rather than embedding 2,040 physical instances
    # into every transaction surface key.
    for grouped_action in legal_actions:
        action = _semantic_action_prototype(grouped_action)
        if str(action.get("model_action_kind") or "") != "card_selection":
            continue
        variant = str(
            action.get("selection_operation") or action.get("model_action_variant") or action.get("kind") or ""
        ).lower()
        if "select" not in variant:
            continue
        card = action.get("card")
        if isinstance(card, Mapping):
            add(
                _transaction_option_identity(card, include_instance=False),
                _semantic_action_multiplicity(grouped_action),
            )

    if not counted:
        options = selection.get("options")
        if isinstance(options, list | tuple) and options:
            for option in options:
                add(_transaction_option_identity(option, include_instance=False))
        else:
            # Simulator translation marks ``cards`` as the *currently
            # selectable* membership. A toggle moves a card between that
            # collection and ``selected_cards``, so their union is stable.
            selectable_field = "selectable_cards" if "selectable_cards" in selection else "cards"
            for field in (selectable_field, "selected_cards"):
                items = selection.get(field)
                if isinstance(items, list | tuple):
                    for item in items:
                        add(
                            _transaction_option_identity(
                                item,
                                include_instance=False,
                            )
                        )
    if not counted:
        # Last-resort projection for legacy DTOs: use only card-bearing
        # selection candidates and erase the select/deselect membership role.
        for grouped_action in legal_actions:
            action = _semantic_action_prototype(grouped_action)
            if str(action.get("model_action_kind") or "") != "card_selection":
                continue
            card = action.get("card")
            if isinstance(card, Mapping):
                add(
                    _transaction_option_identity(card, include_instance=False),
                    _semantic_action_multiplicity(grouped_action),
                )
    return tuple(
        canonical_json({"identity": identity, "multiplicity": count})
        for _, (identity, count) in sorted(counted.items())
    )


def _transaction_surface_key(
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> str | None:
    """Return a stable card-selection transaction kind, without membership.

    Selection counts, selected cards, the currently exposed option subset and
    transport handles are intentionally excluded. Those values identify nodes
    *inside* one transaction and belong in ``_transaction_node_key`` instead.

    In particular, a max-one grid commonly exposes all selectable cards before
    a choice but only ``deselect/confirm/cancel`` afterwards. Hashing that
    membership-dependent option universe split one real transaction into two
    independently "completed" traces and taught both select and deselect as
    preferred actions.
    """

    raw_selection = _transaction_selection_context(observation, legal_actions)
    if raw_selection is None:
        return None
    selection = {key: raw_selection[key] for key in _TRANSACTION_SURFACE_FIELDS if raw_selection.get(key) is not None}
    raw_run = observation.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    return semantic_fingerprint(
        {
            "kind": "card_selection_transaction",
            "locus": {
                "act": run.get("act"),
                "floor": run.get("floor"),
                "room_type": run.get("room_type"),
                "room_model_id": run.get("room_model_id"),
            },
            "selection": selection,
        }
    )


def _transaction_selected_identities(
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> tuple[str, ...]:
    raw_selection = observation.get("card_selection")
    identities: list[object] = []
    if isinstance(raw_selection, Mapping):
        options = raw_selection.get("options")
        if isinstance(options, list | tuple):
            identities.extend(
                _transaction_option_identity(option, include_instance=False)
                for option in options
                if isinstance(option, Mapping)
                and (
                    option.get("is_selected") is True
                    or option.get("selected") is True
                    or str(option.get("selection_membership") or "").lower() == "selected"
                )
            )
        if not identities:
            selected = raw_selection.get("selected_cards")
            if isinstance(selected, list | tuple):
                identities.extend(_transaction_option_identity(item, include_instance=False) for item in selected)
    if not identities:
        for action in _selection_actions(legal_actions):
            variant = str(action.get("model_action_variant") or action.get("kind") or "").lower()
            card = action.get("card")
            if isinstance(card, Mapping) and (
                "deselect" in variant
                or card.get("is_selected") is True
                or str(card.get("selection_membership") or "").lower() == "selected"
            ):
                identities.append(_transaction_option_identity(card, include_instance=False))
    return tuple(sorted(canonical_json(identity) for identity in identities))


def _transaction_selected_count(
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...] = (),
) -> int:
    raw_selection = observation.get("card_selection")
    if not isinstance(raw_selection, Mapping):
        return len(_transaction_selected_identities(observation, legal_actions))
    raw_count = raw_selection.get("selected_count")
    if isinstance(raw_count, int) and not isinstance(raw_count, bool):
        return max(0, raw_count)
    selected = raw_selection.get("selected_cards")
    if isinstance(selected, list | tuple):
        return len(selected)
    options = raw_selection.get("options")
    if isinstance(options, list | tuple):
        return sum(int(isinstance(option, Mapping) and option.get("is_selected") is True) for option in options)
    return len(_transaction_selected_identities(observation, legal_actions))


def _deadlock_semantic_observation(
    observation: Mapping[str, object],
    semantic_actions: tuple[Mapping[str, object], ...],
) -> Mapping[str, object]:
    """Remove duplicated physical selection surfaces from deadlock identity.

    The complete strict action prototypes and their multiplicities are already
    supplied separately to ``SemanticDeadlockDetector``.  Re-hashing raw
    ``cards/options/available_actions`` would both retain arbitrary instance
    ordering and turn a 2,040-copy prompt back into a 2,040-item diagnostic
    surface.  Scalar prompt state and every non-selection world fact remain.
    """

    result = {key: value for key, value in observation.items() if key not in {"available_actions", "legal_actions"}}
    raw_selection = observation.get("card_selection")
    if not isinstance(raw_selection, Mapping):
        return result
    selection = {
        key: value
        for key, value in raw_selection.items()
        if key
        not in {
            "cards",
            "options",
            "selectable_cards",
            "selected_cards",
        }
    }
    selection["semantic_option_universe"] = _transaction_option_universe(
        raw_selection,
        semantic_actions,
    )
    selection["semantic_selected_identities"] = _transaction_selected_identities(observation, semantic_actions)
    result["card_selection"] = selection
    return result


def _transaction_node_key(
    surface_key: str | None,
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> str:
    """Identify a selection node independent of DTO/candidate ordering."""

    if surface_key is None:
        return semantic_decision_fingerprint(observation, legal_actions)
    selection = _transaction_selection_context(observation, legal_actions) or {}
    remaining: object | None = None
    for key in ("remaining_select", "remaining_picks", "remaining", "required_remaining"):
        if selection.get(key) is not None:
            remaining = selection[key]
            break
    can_confirm = selection.get("can_confirm")
    if can_confirm is None:
        can_confirm = any(
            str(action.get("model_action_variant") or action.get("kind") or "").lower()
            in {"confirm", "confirm_selection"}
            for action in _selection_actions(legal_actions)
        )
    # Pairwise factual outcomes may only meet at the same reward-relevant
    # world state.  A selection prompt/membership alone is insufficient: the
    # same discard/transform UI at different HP, decks, relics or cumulative
    # revival cost is a different decision.  Remove only the transaction DTO
    # and duplicated candidate surfaces whose membership/order is represented
    # below; preserve the complete remaining semantic observation.  Reward
    # counters live under an underscore-prefixed transport namespace, so copy
    # that exact mapping under a semantic key before canonicalization drops
    # private transport fields.
    world_context = {
        key: value
        for key, value in observation.items()
        if key
        not in {
            "card_selection",
            "available_actions",
            "legal_actions",
            # These bridge hashes already include UI membership and transport
            # projection details represented explicitly below. Retaining them
            # makes a semantic select/deselect return look like a novel node.
            "semantic_state_hash",
            "state_hash",
        }
        and not str(key).startswith("_")
    }
    training_reward_state = observation.get("_training")
    if isinstance(training_reward_state, Mapping):
        world_context["training_reward_state"] = training_reward_state
    return semantic_fingerprint(
        {
            "kind": "card_selection_node",
            "surface_key": surface_key,
            "world_context": world_context,
            "selected": _transaction_selected_identities(observation, legal_actions),
            "selected_count": _transaction_selected_count(observation, legal_actions),
            "remaining": remaining,
            "can_confirm": bool(can_confirm),
            "legal_action_fingerprints": tuple(sorted(semantic_action_fingerprint(action) for action in legal_actions)),
        }
    )


def _signed_selection_delta(before: int, after: int) -> int:
    return 1 if after > before else -1 if after < before else 0


def _classify_transaction_transition(
    *,
    current_surface: str | None,
    next_surface: str | None,
    current_node: str,
    next_node: str,
    seen_nodes: set[str],
    before_selected_count: int,
    after_selected_count: int,
) -> tuple[TransactionEffect, int]:
    if current_surface is None:
        return (
            TransactionEffect.STAY if next_node == current_node else TransactionEffect.MOVE,
            0,
        )
    if next_surface != current_surface:
        # Exit tears down the selection DTO.  That is not a factual deselect.
        return TransactionEffect.EXIT, 0
    effect = (
        TransactionEffect.STAY
        if next_node == current_node
        else TransactionEffect.REVISIT
        if next_node in seen_nodes
        else TransactionEffect.MOVE
    )
    return effect, _signed_selection_delta(
        before_selected_count,
        after_selected_count,
    )


def _run_position(observation: Mapping[str, object]) -> tuple[int, int]:
    raw_run = observation.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    act = int(max(0.0, _number(run.get("act", observation.get("act")))))
    floor = int(max(0.0, _number(run.get("floor", observation.get("floor")))))
    return act, floor


def _player_hp_ratio(observation: Mapping[str, object]) -> float:
    raw_player = observation.get("player")
    player = raw_player if isinstance(raw_player, Mapping) else {}
    hp = max(0.0, _number(player.get("hp", player.get("current_hp"))))
    maximum = max(
        0.0,
        _number(player.get("max_hp", player.get("maximum_hp"))),
    )
    if maximum <= 0.0:
        raise CollectionProtocolError(
            "authoritative Act/macro health receipt has no positive player max HP"
        )
    return min(1.0, hp / maximum)


def _player_hp(observation: Mapping[str, object]) -> float:
    """Return the authoritative visible player HP for factual effect checks."""

    raw_player = observation.get("player")
    player = raw_player if isinstance(raw_player, Mapping) else {}
    for key in ("hp", "current_hp"):
        value = player.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            normalized = float(value)
            if math.isfinite(normalized) and normalized >= 0.0:
                return normalized
    raise CollectionProtocolError(
        "authoritative transaction effect receipt has no finite player HP"
    )


def _act_exit_hp_ratio(
    *,
    before_observation: Mapping[str, object],
    after_observation: Mapping[str, object],
    authoritative_run_result: str | None,
) -> float:
    """Read the factual HP ratio at an authoritative Act-success boundary.

    Ordinary Act transitions retain a player receipt in the post-action
    observation, so that receipt remains authoritative.  A typed run victory
    is different: the simulator is allowed to return a sparse terminal
    observation after the final proceed action.  In that one case the last
    pre-terminal observation is the final player receipt for the completed
    Act.  Do not turn this into a general fallback; a missing or malformed
    player receipt on a non-terminal Act transition remains a protocol error.
    """

    raw_after_player = after_observation.get("player")
    after_player = raw_after_player if isinstance(raw_after_player, Mapping) else {}
    has_after_player_receipt = bool(
        {"hp", "current_hp", "max_hp", "maximum_hp"}.intersection(after_player)
    )
    if has_after_player_receipt:
        return _player_hp_ratio(after_observation)
    if authoritative_run_result == "victory":
        return _player_hp_ratio(before_observation)
    return _player_hp_ratio(after_observation)


def _hp_band(observation: Mapping[str, object]) -> str:
    try:
        ratio = _player_hp_ratio(observation)
    except CollectionProtocolError:
        return "unknown"
    if ratio < 0.25:
        return "critical_0_25"
    if ratio < 0.50:
        return "low_25_50"
    if ratio < 0.75:
        return "mid_50_75"
    return "high_75_100"


def _diagnostic_action_key(action: Mapping[str, object]) -> str:
    prototype = _semantic_action_prototype(action)
    kind = str(
        prototype.get("model_action_kind")
        or prototype.get("action")
        or prototype.get("kind")
        or "unknown"
    ).strip().lower()
    variant = str(
        prototype.get("model_action_variant")
        or prototype.get("operation")
        or prototype.get("option_id")
        or ""
    ).strip().lower()
    return f"{kind}:{variant}" if variant else kind


def _macro_return_diagnostics(
    *,
    rewards: tuple[float, ...],
    discounts: tuple[float, ...],
    decisions: tuple[tuple[int, str, str, str, int], ...],
    authoritative: bool,
) -> tuple[MacroReturnDiagnostic, ...]:
    """Aggregate exact observed reward-to-terminal returns; never train on it."""

    if not authoritative or not rewards:
        return ()
    if len(rewards) != len(discounts):
        raise RuntimeError("macro diagnostic reward/discount streams are misaligned")
    returns = [0.0] * len(rewards)
    suffix = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        suffix = float(rewards[index]) + float(discounts[index]) * suffix
        returns[index] = suffix
    grouped: dict[tuple[str, str, str, int], list[float]] = {}
    for step_index, surface, action_key, hp_band, act in decisions:
        if not 0 <= step_index < len(returns):
            raise RuntimeError("macro diagnostic decision lies outside its reward stream")
        grouped.setdefault((surface, action_key, hp_band, act), []).append(
            returns[step_index]
        )
    return tuple(
        MacroReturnDiagnostic(
            decision_surface=surface,
            action_key=action_key,
            hp_band=hp_band,
            act=act,
            count=len(values),
            mean_return=sum(values) / len(values),
        )
        for (surface, action_key, hp_band, act), values in sorted(grouped.items())
    )


def _extended_transaction_option_endpoint(
    trace: TransactionTrace,
    *,
    episodic_steps: tuple[EpisodeDecisionStep, ...],
    authoritative_outcome: bool,
) -> tuple[int, str] | None:
    """Locate the first fully observed next-rest/Act/run option boundary."""

    lifecycle = trace.lifecycle
    if lifecycle is None or not lifecycle.support_eligible:
        return None
    entry = trace.start_step + lifecycle.entry_step_index
    exit_step = trace.start_step + lifecycle.exit_step_index
    if not 0 <= entry <= exit_step < len(episodic_steps):
        raise RuntimeError("transaction lifecycle is not aligned to episodic facts")
    entry_floor = episodic_steps[entry].floor
    for index in range(exit_step + 1, len(episodic_steps)):
        step = episodic_steps[index]
        if step.act_boundary in {
            BoundaryOutcome.SUCCEEDED,
            BoundaryOutcome.FAILED,
        }:
            return index, "act_boundary"
        if (
            step.decision_surface in {"rest_site", "restsite", "rest"}
            and step.floor != entry_floor
        ):
            # Arrival at the next rest site ends the option before executing
            # the next rest-vs-forge choice itself.
            return index - 1, "next_rest_site"
    if authoritative_outcome:
        return len(episodic_steps) - 1, "run_terminal"
    return None


def _combat_in_progress(observation: Mapping[str, object]) -> bool:
    combat = observation.get("combat")
    return bool(isinstance(combat, Mapping) and combat.get("in_progress") is True)


def _episodic_decision_surface(
    observation: Mapping[str, object],
    *,
    combat_in_progress: bool,
) -> str:
    """Return a factual, strategy-free surface label for replay sampling.

    The label never enters the model.  It preserves the authoritative runtime
    screen/state identity long enough for detached complete-episode replay to
    keep sparse build, route and resource decisions visible beside the much
    larger combat stream.  Combat is deliberately collapsed to one surface so
    monster/boss identities cannot become replay priorities.
    """

    if combat_in_progress:
        return "combat"
    raw_run = observation.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    for value in (
        observation.get("state_type"),
        observation.get("screen"),
        observation.get("phase"),
        observation.get("decision_domain"),
        run.get("room_type"),
    ):
        if value is None:
            continue
        normalized = "_".join(str(value).strip().lower().replace("-", " ").split())
        if normalized:
            return normalized
    return "noncombat"


def _is_rest_site_decision_surface(
    observation: Mapping[str, object],
    semantic_actions: tuple[Mapping[str, object], ...],
) -> bool:
    """Recognize the generic rest-site decision surface, without option IDs."""

    if _combat_in_progress(observation):
        return False
    surface = _episodic_decision_surface(
        observation,
        combat_in_progress=False,
    )
    if surface in {"rest_site", "restsite", "rest"}:
        return True
    for action in semantic_actions:
        prototype = _semantic_action_prototype(action)
        action_kind = str(prototype.get("action") or prototype.get("kind") or "").strip().lower()
        model_kind = str(prototype.get("model_action_kind") or "").strip().lower()
        if model_kind in {"rest_site", "restsite"} or action_kind == "choose_rest_option":
            return True
    return False


def _uses_targeted_selection_exploration(
    observation: Mapping[str, object],
    semantic_actions: tuple[Mapping[str, object], ...],
) -> bool:
    """Return whether this generic decision surface owns the v33 ε floor."""

    return bool(
        _transaction_selection_context(observation, semantic_actions) is not None
        or _is_rest_site_decision_surface(observation, semantic_actions)
    )


def _canonical_transaction_operation(value: object) -> str:
    """Compatibility wrapper around the shared reviewed operation registry."""

    return canonical_transaction_operation(value)


def _selection_transaction_operation(
    observation: Mapping[str, object],
    semantic_actions: tuple[Mapping[str, object], ...],
) -> str:
    selection = _transaction_selection_context(observation, semantic_actions)
    if selection is None:
        return ""
    return _canonical_transaction_operation(selection.get("operation_type", selection.get("operation")))


def _transaction_entry_operation(action: Mapping[str, object]) -> str:
    """Return the operation opened by one macro action, if reviewed.

    This classification is restricted to generic protocol fields.  It never
    inspects a card definition, event, encounter, relic, or character ID.
    """

    prototype = _semantic_action_prototype(action)
    nested_transaction = prototype.get("transaction")
    if isinstance(nested_transaction, Mapping):
        operation = _canonical_transaction_operation(
            nested_transaction.get("operation_type", nested_transaction.get("operation"))
        )
        if operation:
            return operation

    for key in ("model_action_kind", "action", "kind"):
        normalized_kind = "_".join(
            str(prototype.get(key) or "").strip().lower().replace("-", " ").split()
        )
        if normalized_kind in {"skip_card_reward", "reward_skip", "skip_reward"}:
            # Declining a combat card reward is a reviewed single-decision
            # entrance: the deck-growth alternative to taking a card. The
            # classification reads only the generic action kind.
            return "reward_skip"

    model_kind = "_".join(str(prototype.get("model_action_kind") or "").strip().lower().replace("-", " ").split())
    if model_kind == "shop":
        item = prototype.get("item")
        if isinstance(item, Mapping):
            operation = _canonical_transaction_operation(item.get("category", item.get("type")))
            if operation == "remove":
                return operation
            category = "_".join(
                str(item.get("category") or item.get("type") or "")
                .strip()
                .lower()
                .replace("-", " ")
                .split()
            )
            if category == "relic":
                # Purchasing a relic is a reviewed single-decision entrance.
                # Only the generic item category is read, never a relic ID.
                return "relic_purchase"
    if model_kind in {"rest_site", "restsite"}:
        option = prototype.get("option")
        if isinstance(option, Mapping):
            for key in ("type", "option_type", "id", "option_id"):
                operation = _canonical_transaction_operation(option.get(key))
                if operation:
                    return operation
        for key in (
            "model_action_variant",
            "option_type",
            "option_id",
            "kind",
            "action",
        ):
            operation = _canonical_transaction_operation(prototype.get(key))
            if operation:
                return operation
    return ""


def _is_shop_card_removal_entry(action: Mapping[str, object]) -> bool:
    prototype = _semantic_action_prototype(action)
    model_kind = "_".join(str(prototype.get("model_action_kind") or "").strip().lower().replace("-", " ").split())
    return bool(model_kind == "shop" and _transaction_entry_operation(prototype) == "remove")


def _transaction_entry_exploration_branch_ids(
    *,
    policy_branch_ids: NDArray[np.int64],
    semantic_actions: tuple[Mapping[str, object], ...],
    enabled_operations: frozenset[str],
) -> tuple[NDArray[np.int64], bool]:
    """Give each reviewed entrance a cardinality-independent explorer branch."""

    if policy_branch_ids.ndim != 1 or len(semantic_actions) != len(policy_branch_ids):
        raise CollectionProtocolError("transaction entry branch IDs and semantic actions have different shapes")
    if not enabled_operations:
        return policy_branch_ids, False
    unknown = enabled_operations - TRANSACTION_EXPLORATION_OPERATIONS
    if unknown:
        raise ValueError("unsupported transaction exploration operations: " + ", ".join(sorted(unknown)))
    operations = tuple(_transaction_entry_operation(action) for action in semantic_actions)
    present = tuple(sorted({item for item in operations if item in enabled_operations}))
    if not present:
        return policy_branch_ids, False
    remapped = policy_branch_ids.astype(np.int64, copy=True)
    next_branch_id = int(remapped.max(initial=-1)) + 1
    for offset, operation in enumerate(present):
        for index, candidate_operation in enumerate(operations):
            if candidate_operation == operation:
                remapped[index] = next_branch_id + offset
    return remapped, True


def _is_forge_selection_surface(
    observation: Mapping[str, object],
    semantic_actions: tuple[Mapping[str, object], ...],
) -> bool:
    """Recognize an authoritative upgrade/forge selection transaction.

    This is a decision-surface classification used only for observability.  It
    reads the generic selection operation contract and never names a card,
    character, relic, room model, or encounter.
    """

    selection = _transaction_selection_context(observation, semantic_actions)
    if selection is None:
        return False
    operation = "_".join(str(selection.get("operation_type") or "").strip().lower().replace("-", " ").split())
    return operation in {"upgrade", "forge"}


_SELECTION_CANCEL_OPERATIONS = frozenset(
    {
        "cancel",
        "cancel_prompt",
        "cancel_selection",
    }
)
_SELECTION_COMMIT_OPERATIONS = frozenset(
    {
        "confirm",
        "confirm_selection",
        # Max-one/automatic selection grids commit directly on Select and do
        # not expose a separate confirmation action.
        "select",
        "select_card",
    }
)


def _normalized_selection_action_operation(action: Mapping[str, object]) -> str:
    """Return the reviewed selection operation, independent of dispatch IDs."""

    prototype = _semantic_action_prototype(action)
    nested = prototype.get("selection")
    nested_operation = nested.get("operation_type") if isinstance(nested, Mapping) else None
    raw = next(
        (
            value
            for value in (
                prototype.get("selection_operation"),
                prototype.get("model_action_variant"),
                prototype.get("operation"),
                prototype.get("operation_type"),
                nested_operation,
                prototype.get("kind"),
                prototype.get("action"),
            )
            if value is not None and str(value).strip()
        ),
        "",
    )
    return "_".join(str(raw).strip().lower().replace("-", " ").split())


def _transaction_completion_guided_behavior(
    *,
    policy: NDArray[np.float32],
    valid: NDArray[np.bool_],
    semantic_actions: tuple[Mapping[str, object], ...],
    base_behavior: NDArray[np.float64],
    operation: str,
    guidance_probability: float,
) -> tuple[NDArray[np.float64], frozenset[int], bool]:
    """Mix exact behavior with a forward Select/Confirm transaction proposal.

    Confirm dominates Select once it is legal.  Before that point, all legal
    Select candidates retain their learned relative policy mass.  Cancel and
    Deselect keep non-zero support through ``base_behavior``; consequently the
    returned distribution is a valid, fully auditable behavior policy rather
    than a hidden action override.
    """

    if operation not in TRANSACTION_GUIDANCE_OPERATIONS:
        return base_behavior, frozenset(), False
    if policy.shape != valid.shape or base_behavior.shape != valid.shape:
        raise CollectionProtocolError("transaction guidance inputs have different shapes")
    if len(semantic_actions) != len(valid):
        raise CollectionProtocolError("transaction guidance lost semantic candidate alignment")
    probability = float(guidance_probability)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("transaction completion guidance probability must be in [0, 1)")
    if probability == 0.0:
        return base_behavior, frozenset(), False

    confirm_indices: list[int] = []
    select_indices: list[int] = []
    for index, action in enumerate(semantic_actions):
        if not bool(valid[index]):
            continue
        normalized = _normalized_selection_action_operation(action)
        if normalized in {"confirm", "confirm_selection"}:
            confirm_indices.append(index)
        elif normalized in {"select", "select_card"}:
            select_indices.append(index)
    forward_indices = confirm_indices or select_indices
    if not forward_indices:
        return base_behavior, frozenset(), True

    guided = np.zeros(policy.shape, dtype=np.float64)
    forward_policy = policy[forward_indices].astype(np.float64, copy=False)
    if not np.all(np.isfinite(forward_policy)) or np.any(forward_policy < 0.0):
        raise CollectionProtocolError("model produced invalid transaction forward policy mass")
    forward_mass = float(forward_policy.sum())
    if math.isfinite(forward_mass) and forward_mass > 0.0:
        guided[forward_indices] = forward_policy / forward_mass
    else:  # pragma: no cover - the full legal policy is validated by caller
        guided[forward_indices] = 1.0 / len(forward_indices)
    behavior = (1.0 - probability) * base_behavior + probability * guided
    behavior_mass = float(behavior.sum())
    if not math.isfinite(behavior_mass) or behavior_mass <= 0.0:
        raise CollectionProtocolError("transaction guidance produced invalid behavior mass")
    behavior /= behavior_mass
    if not np.all(np.isfinite(behavior)) or np.any(behavior < 0.0) or np.any(behavior[~valid] != 0.0):
        raise CollectionProtocolError("transaction guidance produced an invalid behavior policy")
    return behavior, frozenset(forward_indices), False


def _card_upgrade_level(card: Mapping[str, object]) -> int:
    nested = card.get("card")
    value = nested if isinstance(nested, Mapping) else card
    levels: list[int] = []
    for key in (
        "upgrade_level",
        "current_upgrade_level",
        "upgrades",
        "upgrade_count",
    ):
        raw = value.get(key)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int | float) and math.isfinite(float(raw)):
            levels.append(max(0, int(raw)))
    if bool(value.get("is_upgraded")):
        levels.append(1)
    return max(levels, default=0)


def _deck_upgrade_signature(
    observation: Mapping[str, object],
) -> tuple[tuple[str, int, int], ...] | None:
    """Return definition counts and total upgrade levels for the real deck.

    The signature deliberately ignores the screen-shaped ``_sim_raw`` mirror.
    A forge commit is proved only when the same definition multiset remains and
    at least one aggregate upgrade level increases.
    """

    raw_player = observation.get("player")
    player = raw_player if isinstance(raw_player, Mapping) else {}
    raw_deck = player.get("deck_cards", player.get("deck"))
    if isinstance(raw_deck, Mapping):
        raw_cards = raw_deck.get("cards")
    else:
        raw_cards = raw_deck
    if not isinstance(raw_cards, list | tuple):
        return None
    aggregate: dict[str, tuple[int, int]] = {}
    for item in raw_cards:
        if not isinstance(item, Mapping):
            return None
        nested = item.get("card")
        card = nested if isinstance(nested, Mapping) else item
        raw_definition = _first_nonempty(card, _TRANSACTION_DEFINITION_KEYS)
        if raw_definition is None:
            return None
        definition = str(raw_definition).strip()
        raw_quantity = card.get("quantity", card.get("count", card.get("copies", 1)))
        quantity = (
            int(raw_quantity)
            if not isinstance(raw_quantity, bool)
            and isinstance(raw_quantity, int | float)
            and math.isfinite(float(raw_quantity))
            and int(raw_quantity) > 0
            else 1
        )
        previous_count, previous_levels = aggregate.get(definition, (0, 0))
        aggregate[definition] = (
            previous_count + quantity,
            previous_levels + quantity * _card_upgrade_level(card),
        )
    return tuple((definition, count, levels) for definition, (count, levels) in sorted(aggregate.items()))


def _forge_upgrade_committed(
    before: tuple[tuple[str, int, int], ...] | None,
    after_observation: Mapping[str, object],
) -> bool:
    after = _deck_upgrade_signature(after_observation)
    if before is None or after is None:
        return False
    before_counts = tuple((definition, count) for definition, count, _ in before)
    after_counts = tuple((definition, count) for definition, count, _ in after)
    if before_counts != after_counts:
        return False
    before_levels = {definition: levels for definition, _, levels in before}
    return any(levels > before_levels.get(definition, levels) for definition, _, levels in after)


def _deck_card_removal_committed(
    before: tuple[tuple[str, int, int], ...] | None,
    after_observation: Mapping[str, object],
) -> bool:
    """Prove that exactly one existing deck card was removed.

    Gold spend, page exit, and a Confirm click are insufficient.  The real
    player deck must contain exactly one fewer card, no definition may gain a
    copy, and no previously unseen definition may appear.
    """

    after = _deck_upgrade_signature(after_observation)
    if before is None or after is None:
        return False
    before_counts = {definition: count for definition, count, _ in before}
    after_counts = {definition: count for definition, count, _ in after}
    if sum(after_counts.values()) != sum(before_counts.values()) - 1:
        return False
    if any(definition not in before_counts for definition in after_counts):
        return False
    return all(after_counts.get(definition, 0) <= before_count for definition, before_count in before_counts.items())


def _selection_transaction_exit_outcome(
    *,
    selected_action: Mapping[str, object],
    clean_exit: bool,
    operation: str,
    opening_deck_signature: tuple[tuple[str, int, int], ...] | None,
    after_observation: Mapping[str, object],
) -> Literal["committed", "cancelled", "unresolved"]:
    """Classify transaction teardown without equating exit with success."""

    if not clean_exit:
        return "unresolved"
    transaction_operation = _canonical_transaction_operation(operation)
    selected_operation = _normalized_selection_action_operation(selected_action)
    if transaction_operation == "upgrade":
        if _forge_upgrade_committed(opening_deck_signature, after_observation):
            return "committed"
        if selected_operation in _SELECTION_CANCEL_OPERATIONS:
            return "cancelled"
        return "unresolved"
    if transaction_operation == "remove":
        if _deck_card_removal_committed(opening_deck_signature, after_observation):
            return "committed"
        if selected_operation in _SELECTION_CANCEL_OPERATIONS:
            return "cancelled"
        return "unresolved"
    if selected_operation in _SELECTION_CANCEL_OPERATIONS:
        return "cancelled"
    if selected_operation in _SELECTION_COMMIT_OPERATIONS:
        return "committed"
    return "unresolved"


def _room_type(observation: Mapping[str, object]) -> str:
    raw_run = observation.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    raw = run.get("room_type", observation.get("room_type", ""))
    return "_".join(str(raw or "").strip().lower().replace("-", " ").split())


def _episodic_combat_boundary(
    *,
    was_active: bool,
    is_active: bool,
    transition_facts: Mapping[str, object],
    authoritative_run_result: str | None,
    trusted_policy_failure: bool,
    episode_censored: bool,
) -> BoundaryOutcome:
    """Classify a factual combat horizon without treating revival as exit.

    Engine bailout keeps ``combat.in_progress`` true.  That observable state
    has priority over any incidental result string and therefore never closes
    a combat horizon.  In a full run the bridge normally emits
    ``combat_result=victory`` on a true -> false transition.  The explicit
    state transition remains a documented success fallback because older
    bridge revisions did not emit that mid-run fact.
    """

    if not was_active:
        return BoundaryOutcome.NONE
    if authoritative_run_result == "defeat":
        return BoundaryOutcome.FAILED
    if trusted_policy_failure:
        return BoundaryOutcome.FAILED
    if is_active:
        return BoundaryOutcome.CENSORED if episode_censored else BoundaryOutcome.NONE

    combat_result = transition_facts.get("combat_result", "none")
    if combat_result in {"defeat", "escaped"}:
        return BoundaryOutcome.FAILED
    if combat_result not in {None, "none", "victory"}:
        raise CollectionProtocolError(f"unsupported typed combat_result for episodic replay: {combat_result!r}")
    # A mid-run active -> inactive transition is a factual completed combat,
    # even when an older bridge omitted its explicit result.  A terminal run
    # victory is the same final-combat success fallback.
    return BoundaryOutcome.SUCCEEDED


def _episodic_act_boundary(
    *,
    before_act: int,
    after_act: int,
    authoritative_run_result: str | None,
    trusted_policy_failure: bool,
    episode_censored: bool,
) -> BoundaryOutcome:
    """Classify the Act containing the pre-action decision."""

    # Terminal DTOs may omit the run block and consequently decode as act 0;
    # typed full-run outcome remains authoritative in that case.
    if after_act > before_act:
        return BoundaryOutcome.SUCCEEDED
    if authoritative_run_result == "victory":
        return BoundaryOutcome.SUCCEEDED
    if authoritative_run_result == "defeat":
        return BoundaryOutcome.FAILED
    if trusted_policy_failure:
        return BoundaryOutcome.FAILED
    if after_act < before_act:
        raise CollectionProtocolError("non-terminal full-run transition regressed its Act index")
    return BoundaryOutcome.CENSORED if episode_censored else BoundaryOutcome.NONE


def _enemy_hp_totals(observation: Mapping[str, object]) -> tuple[float, float]:
    combat = observation.get("combat")
    enemies = combat.get("enemies") if isinstance(combat, Mapping) else None
    if not isinstance(enemies, list | tuple):
        return 0.0, 0.0
    current = 0.0
    maximum = 0.0
    for enemy in enemies:
        if not isinstance(enemy, Mapping):
            continue
        current += max(0.0, _number(enemy.get("hp", enemy.get("current_hp"))))
        maximum += max(
            0.0,
            _number(enemy.get("max_hp", enemy.get("maximum_hp"))),
        )
    return current, maximum


def _enemy_roster_signature(observation: Mapping[str, object]) -> tuple[str, ...]:
    """Return a stable multiset identity without volatile HP/status fields."""

    combat = observation.get("combat")
    enemies = combat.get("enemies") if isinstance(combat, Mapping) else None
    if not isinstance(enemies, list | tuple):
        return ()
    identities: list[str] = []
    for index, enemy in enumerate(enemies):
        if not isinstance(enemy, Mapping):
            continue
        definition = next(
            (
                str(enemy[key])
                for key in (
                    "instance_uuid",
                    "instance_id",
                    "entity_uuid",
                    "entity_id",
                    "monster_id",
                    "id",
                )
                if enemy.get(key) not in (None, "")
            ),
            f"anonymous:{index}",
        )
        identities.append(definition.strip().lower())
    return tuple(sorted(identities))


def _combat_phase_signature(observation: Mapping[str, object]) -> tuple[str, ...]:
    """Extract explicit phase/wave identities, deliberately excluding turns."""

    combat = observation.get("combat")
    if not isinstance(combat, Mapping):
        return ()
    markers: list[str] = []
    for key in (
        "encounter_id",
        "combat_id",
        "wave",
        "wave_id",
        "wave_index",
        "stage",
        "stage_id",
    ):
        value = combat.get(key)
        if isinstance(value, str | int | float) and not isinstance(value, bool):
            markers.append(f"combat.{key}={value}")
    enemies = combat.get("enemies")
    if isinstance(enemies, list | tuple):
        for index, enemy in enumerate(enemies):
            if not isinstance(enemy, Mapping):
                continue
            for key in ("stage", "stage_id"):
                value = enemy.get(key)
                if isinstance(value, str | int | float) and not isinstance(value, bool):
                    markers.append(f"enemy[{index}].{key}={value}")
    return tuple(markers)


@dataclass(frozen=True, slots=True)
class _CombatProgressWindowSelection:
    default_window: int
    effective_window: int
    source: Literal["default", "room_model_id", "encounter_id"]
    match_id: str
    room_model_id: str
    encounter_id: str


def _canonical_combat_locus_id(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def _combat_locus_id(
    owners: tuple[Mapping[str, object], ...],
    *keys: str,
) -> str:
    for owner in owners:
        for key in keys:
            identifier = _canonical_combat_locus_id(owner.get(key))
            if identifier:
                return identifier
    return ""


def _combat_progress_window_selection(
    observation: Mapping[str, object],
    *,
    default_window: int,
    room_windows: Mapping[str, int],
    encounter_windows: Mapping[str, int],
) -> _CombatProgressWindowSelection:
    """Resolve one deterministic liveness window from public combat locus IDs.

    A room match wins over an encounter match. The translated headless full-run
    state always exposes ``run.room_model_id`` while some combat-only backends
    expose only ``combat.encounter_id``; supporting both keeps the rule backend
    neutral without reaching into ``_sim_raw``.
    """

    raw_run = observation.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    raw_combat = observation.get("combat")
    combat = raw_combat if isinstance(raw_combat, Mapping) else {}
    room_model_id = _combat_locus_id(
        (run, observation),
        "room_model_id",
        "room_model",
    )
    encounter_id = _combat_locus_id(
        (combat, run, observation),
        "encounter_id",
        "canonical_encounter_id",
    )
    if room_model_id in room_windows:
        return _CombatProgressWindowSelection(
            default_window=default_window,
            effective_window=room_windows[room_model_id],
            source="room_model_id",
            match_id=room_model_id,
            room_model_id=room_model_id,
            encounter_id=encounter_id,
        )
    if encounter_id in encounter_windows:
        return _CombatProgressWindowSelection(
            default_window=default_window,
            effective_window=encounter_windows[encounter_id],
            source="encounter_id",
            match_id=encounter_id,
            room_model_id=room_model_id,
            encounter_id=encounter_id,
        )
    return _CombatProgressWindowSelection(
        default_window=default_window,
        effective_window=default_window,
        source="default",
        match_id="",
        room_model_id=room_model_id,
        encounter_id=encounter_id,
    )


def _normalize_combat_progress_windows(
    value: Mapping[str, int] | None,
    *,
    label: str,
) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    normalized: dict[str, int] = {}
    for raw_identifier, raw_window in value.items():
        if not isinstance(raw_identifier, str) or not raw_identifier.strip():
            raise TypeError(f"{label} identifiers must be non-empty strings")
        identifier = raw_identifier.strip().upper()
        if identifier in normalized:
            raise ValueError(f"{label} contains duplicate normalized identifier {identifier!r}")
        if isinstance(raw_window, bool) or not isinstance(raw_window, int):
            raise TypeError(f"{label}[{identifier!r}] must be an integer")
        if raw_window <= 0:
            raise ValueError(f"{label}[{identifier!r}] must be positive")
        normalized[identifier] = raw_window
    return dict(sorted(normalized.items()))


@dataclass(frozen=True, slots=True)
class _CombatNetProgressStatus:
    age_steps: int
    maximum_age_steps: int
    stalled: bool
    current_hp: float
    maximum_hp: float
    anchor_hp: float
    required_hp_progress: float
    net_hp_progress: float
    progress_kind: str


class _CombatNetProgressTracker:
    """Detect combat loops by monotonic net progress, not transient damage.

    A hit followed by healing no longer resets the window.  The anchor advances
    only after a meaningful *net* reduction in the current enemy health burden,
    or after an explicit phase/wave transition.  Adding summons is not progress;
    defeating them without reducing the pre-summon burden is intentionally
    neutral.  This closes the old loophole where tiny recurring damage kept a
    hopeless combat alive until the 30,000-step transport ceiling.
    """

    def __init__(self, *, window: int, minimum_hp_fraction: float) -> None:
        self.window = int(window)
        self.minimum_hp_fraction = float(minimum_hp_fraction)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._anchor_step = 0
        self._anchor_hp = 0.0
        self._anchor_max_hp = 0.0
        self._last_hp = 0.0
        self._last_max_hp = 0.0
        self._roster: tuple[str, ...] = ()
        self._phase: tuple[str, ...] = ()
        self._maximum_age = 0

    def _start(
        self,
        *,
        step: int,
        current_hp: float,
        maximum_hp: float,
        roster: tuple[str, ...],
        phase: tuple[str, ...],
    ) -> None:
        self._active = True
        self._anchor_step = step
        self._anchor_hp = current_hp
        self._anchor_max_hp = maximum_hp
        self._last_hp = current_hp
        self._last_max_hp = maximum_hp
        self._roster = roster
        self._phase = phase

    def observe(
        self,
        *,
        step: int,
        observation: Mapping[str, object],
    ) -> _CombatNetProgressStatus:
        current_hp, maximum_hp = _enemy_hp_totals(observation)
        if not _combat_in_progress(observation):
            maximum_age = self._maximum_age
            self.reset()
            self._maximum_age = maximum_age
            return _CombatNetProgressStatus(
                age_steps=0,
                maximum_age_steps=self._maximum_age,
                stalled=False,
                current_hp=current_hp,
                maximum_hp=maximum_hp,
                anchor_hp=current_hp,
                required_hp_progress=0.0,
                net_hp_progress=0.0,
                progress_kind="outside_combat",
            )

        roster = _enemy_roster_signature(observation)
        phase = _combat_phase_signature(observation)
        if not self._active:
            self._start(
                step=step,
                current_hp=current_hp,
                maximum_hp=maximum_hp,
                roster=roster,
                phase=phase,
            )
            return _CombatNetProgressStatus(
                age_steps=0,
                maximum_age_steps=self._maximum_age,
                stalled=False,
                current_hp=current_hp,
                maximum_hp=maximum_hp,
                anchor_hp=current_hp,
                required_hp_progress=max(1.0, maximum_hp * self.minimum_hp_fraction),
                net_hp_progress=0.0,
                progress_kind="combat_started",
            )

        prior_roster = self._roster
        prior_phase = self._phase
        required = max(
            1.0,
            max(self._anchor_max_hp, maximum_hp) * self.minimum_hp_fraction,
        )
        net_progress = self._anchor_hp - current_hp
        roster_replaced = bool(
            roster != prior_roster and prior_roster and roster and not set(prior_roster).intersection(roster)
        )
        advanced_from_defeated_wave = bool(
            roster != prior_roster
            and self._last_hp <= max(1.0, self._last_max_hp * self.minimum_hp_fraction)
            and current_hp > self._last_hp
        )
        explicit_phase_advance = bool(phase != prior_phase and prior_phase and phase)
        progress_kind = "waiting_for_net_progress"
        if explicit_phase_advance or roster_replaced or advanced_from_defeated_wave:
            self._start(
                step=step,
                current_hp=current_hp,
                maximum_hp=maximum_hp,
                roster=roster,
                phase=phase,
            )
            progress_kind = "phase_or_wave_advanced"
            net_progress = 0.0
            required = max(1.0, maximum_hp * self.minimum_hp_fraction)
        elif net_progress >= required:
            self._start(
                step=step,
                current_hp=current_hp,
                maximum_hp=maximum_hp,
                roster=roster,
                phase=phase,
            )
            progress_kind = "meaningful_net_hp_reduction"
            net_progress = 0.0
            required = max(1.0, maximum_hp * self.minimum_hp_fraction)
        else:
            # A summon/addition changes the burden but must not reset the age.
            # Retain the old anchor while tracking the latest structural facts.
            self._roster = roster
            self._phase = phase
            self._last_hp = current_hp
            self._last_max_hp = maximum_hp

        age = step - self._anchor_step
        self._maximum_age = max(self._maximum_age, age)
        return _CombatNetProgressStatus(
            age_steps=age,
            maximum_age_steps=self._maximum_age,
            stalled=age >= self.window,
            current_hp=current_hp,
            maximum_hp=maximum_hp,
            anchor_hp=self._anchor_hp,
            required_hp_progress=required,
            net_hp_progress=self._anchor_hp - current_hp,
            progress_kind=progress_kind,
        )


def _first_durable_scalar(
    value: Mapping[str, object],
    *keys: str,
) -> object | None:
    """Return the first scalar fact from a positive durable-state allowlist."""

    for key in keys:
        item = value.get(key)
        if item is None or isinstance(item, Mapping | list | tuple):
            continue
        if isinstance(item, str | int | float | bool):
            return item
    return None


def _durable_entity_id(
    value: Mapping[str, object],
    *keys: str,
) -> str:
    item = _first_durable_scalar(value, *keys)
    return str(item).strip() if item not in (None, "") else "<unknown>"


def _nested_modifier_signature(value: object) -> tuple[tuple[object, ...], ...]:
    if not isinstance(value, list | tuple):
        return ()
    result: list[tuple[object, ...]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        result.append(
            (
                _durable_entity_id(item, "id", "model_id", "name", "type"),
                _first_durable_scalar(item, "level", "amount", "stacks"),
                _first_durable_scalar(item, "is_active", "is_enabled"),
            )
        )
    return tuple(sorted(result, key=repr))


def _durable_card_signature(value: Mapping[str, object]) -> tuple[object, ...]:
    """Describe permanent card composition without preview/dynamic values."""

    return (
        _durable_entity_id(value, "id", "card_id", "model_id", "name", "title"),
        _first_durable_scalar(
            value,
            "upgrade_level",
            "upgrade_count",
            "times_upgraded",
            "is_upgraded",
            "upgraded",
        ),
        _nested_modifier_signature(value.get("enchantments")),
        _nested_modifier_signature(value.get("afflictions")),
    )


def _durable_relic_signature(value: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _durable_entity_id(value, "id", "relic_id", "model_id", "name", "title"),
        _first_durable_scalar(value, "stack_count", "quantity"),
        _first_durable_scalar(value, "is_used_up"),
        _first_durable_scalar(value, "is_melted"),
        _first_durable_scalar(value, "status"),
    )


def _durable_potion_signature(value: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _durable_entity_id(value, "id", "potion_id", "model_id", "name", "title"),
        _first_durable_scalar(value, "slot_index", "slot"),
    )


def _durable_collection(
    value: object,
    projector: Callable[[Mapping[str, object]], tuple[object, ...]],
    *,
    counted_multiset: bool = False,
) -> tuple[tuple[object, ...], ...]:
    if isinstance(value, Mapping):
        nested = value.get("cards", value.get("items"))
        value = nested if isinstance(nested, list | tuple) else ()
    if not isinstance(value, list | tuple):
        return ()
    if not counted_multiset:
        projected = [projector(item) for item in value if isinstance(item, Mapping)]
        return tuple(sorted(projected, key=repr))

    counts: dict[tuple[object, ...], int] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        quantity = item.get("quantity", 1)
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise CollectionProtocolError("durable card quantity must be a positive integer")
        signature = projector(item)
        counts[signature] = counts.get(signature, 0) + quantity
    return tuple(
        (*signature, quantity) for signature, quantity in sorted(counts.items(), key=lambda item: repr(item[0]))
    )


def _noncombat_durable_projections(
    observation: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Split forward run locus from same-locus persistent resources.

    This deliberately excludes event/card/relic ``dynamic_vars``, descriptions,
    pages, screens, selection membership, HP/max-HP vitality, training-revival
    telemetry, damage/heal previews, counters and legal-option text.  Those
    values are costs or reversible state and may change forever without moving
    the run.  The positive allowlist keeps the detector generic across events
    while preventing a newly exposed preview counter from silently disabling
    it.
    """

    raw_run = observation.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    raw_player = observation.get("player")
    player = raw_player if isinstance(raw_player, Mapping) else {}
    raw_event = observation.get("event")
    event = raw_event if isinstance(raw_event, Mapping) else {}
    raw_map = observation.get("map")
    map_state = raw_map if isinstance(raw_map, Mapping) else {}

    run_projection: dict[str, object] = {}
    for key, aliases in {
        "act": ("act", "act_index"),
        "floor": ("floor", "total_floor"),
        "room_type": ("room_type",),
        "room_model_id": ("room_model_id", "room_model"),
    }.items():
        item = _first_durable_scalar(run, *aliases)
        if item is None:
            item = _first_durable_scalar(observation, *aliases)
        if item is not None:
            run_projection[key] = item
    coord = run.get("coord", map_state.get("current_coord"))
    if isinstance(coord, Mapping):
        run_projection["coord"] = tuple(
            _first_durable_scalar(coord, *aliases) for aliases in (("x", "col", "column"), ("y", "row"))
        )

    player_projection: dict[str, object] = {}
    for key, aliases in {
        "character": ("character_id", "character", "id"),
        "gold": ("gold",),
        "open_potion_slots": ("open_potion_slots",),
    }.items():
        item = _first_durable_scalar(player, *aliases)
        if item is not None:
            player_projection[key] = item
    # The live bridge exposes ``deck`` as a count and the inspectable card
    # collection as ``deck_cards``; the headless bridge exposes ``deck`` as the
    # collection itself.  Prefer the explicit collection and only fall back to
    # ``deck`` when it actually has collection shape.
    deck_value = player.get("deck_cards")
    if not isinstance(deck_value, Mapping | list | tuple):
        deck_value = player.get("deck")
    player_projection["deck"] = _durable_collection(
        deck_value,
        _durable_card_signature,
        counted_multiset=True,
    )
    player_projection["relics"] = _durable_collection(
        player.get("relics"),
        _durable_relic_signature,
    )
    player_projection["potions"] = _durable_collection(
        player.get("potions"),
        _durable_potion_signature,
    )

    event_projection: dict[str, object] = {}
    for key, aliases in {
        "event_id": ("event_id", "id", "model_id"),
        "encounter_id": ("encounter_id", "canonical_encounter_id"),
        "is_finished": ("is_finished",),
    }.items():
        item = _first_durable_scalar(event, *aliases)
        if item is not None:
            event_projection[key] = item

    completion_projection: dict[str, object] = {}
    for key in ("terminated", "truncated"):
        item = _first_durable_scalar(observation, key)
        if item is not None:
            completion_projection[key] = item

    locus = {
        "run": run_projection,
        "event": event_projection,
        "completion": completion_projection,
    }
    resources = {"player": player_projection}
    return locus, resources


def _durable_action_projection(action: Mapping[str, object]) -> Mapping[str, object]:
    """Identify a chosen operation without transport handles or preview text."""

    result: dict[str, object] = {}
    for key in (
        "action",
        "kind",
        "model_action_kind",
        "model_action_variant",
        "transport_kind",
        "index",
        "idx",
        "action_index",
        "card_index",
        "target_id",
        "target_index",
        "slot",
        "slot_index",
        "selection_operation",
        "is_selected",
        "option_id",
        "choice_id",
        "text_key",
        "is_proceed",
    ):
        item = _first_durable_scalar(action, key)
        if item is not None:
            result[key] = item
    for key, projector in (
        ("card", _durable_card_signature),
        ("relic", _durable_relic_signature),
        ("potion", _durable_potion_signature),
    ):
        item = action.get(key)
        if isinstance(item, Mapping):
            result[key] = projector(item)
    option = action.get("option")
    if isinstance(option, Mapping):
        result["option_index"] = _first_durable_scalar(
            option,
            "option_index",
            "index",
        )
        for key, aliases in (
            ("option_id", ("option_id", "choice_id", "id")),
            ("option_text_key", ("text_key", "label_key")),
            ("option_is_proceed", ("is_proceed",)),
        ):
            value = _first_durable_scalar(option, *aliases)
            if value is not None:
                result[key] = value
    return result


def _noncombat_context_summary(
    observation: Mapping[str, object],
) -> Mapping[str, object]:
    run = observation.get("run")
    run = run if isinstance(run, Mapping) else {}
    event = observation.get("event")
    event = event if isinstance(event, Mapping) else {}
    return {
        "phase": str(observation.get("phase") or ""),
        "decision_domain": str(observation.get("decision_domain") or ""),
        "state_type": str(observation.get("state_type") or ""),
        "act": int(max(0.0, _number(run.get("act", observation.get("act"))))),
        "floor": int(max(0.0, _number(run.get("floor", observation.get("floor"))))),
        "room_type": str(run.get("room_type") or ""),
        "room_model_id": str(run.get("room_model_id", run.get("room_model")) or ""),
        "event_id": str(event.get("event_id", event.get("id")) or ""),
        "event_finished": event.get("is_finished") is True,
    }


_EVENT_PAGE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("description_key", ("description_key",)),
    ("page_id", ("page_id", "page_key", "current_page_id")),
    ("page", ("page", "current_page")),
    ("state_id", ("state_id", "event_state_id")),
    ("stage_id", ("stage_id", "dialogue_id")),
)


def _noncombat_event_page_projection(
    observation: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return a positive-allowlist identity for one visible event page.

    Event pages are *not* durable progress: a finite page cycle must not reset
    the room-level clock.  They are nevertheless useful factual cycle nodes.
    This projection intentionally excludes HP, max HP, previews, dynamic vars,
    option text, damage counters and training-revival telemetry.
    """

    raw_event = observation.get("event")
    if not isinstance(raw_event, Mapping):
        return None
    event_id = _first_durable_scalar(
        raw_event,
        "event_id",
        "id",
        "model_id",
    )
    if event_id is None:
        return None
    page: dict[str, object] = {}
    for label, aliases in _EVENT_PAGE_FIELDS:
        value = _first_durable_scalar(raw_event, *aliases)
        if value is not None:
            page[label] = value
    if not page:
        return None
    return {
        "event_id": event_id,
        "page": page,
    }


def _noncombat_liveness_selection_projection(
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> Mapping[str, object] | None:
    """Project visible prompt/membership facts without vitality/world costs."""

    surface_key = _transaction_surface_key(observation, legal_actions)
    if surface_key is None:
        return None
    selection = _transaction_selection_context(observation, legal_actions) or {}
    remaining: object | None = None
    for key in ("remaining_select", "remaining_picks", "remaining", "required_remaining"):
        if selection.get(key) is not None:
            remaining = selection[key]
            break
    can_confirm = selection.get("can_confirm")
    if can_confirm is None:
        can_confirm = any(
            str(action.get("model_action_variant") or action.get("kind") or "").lower()
            in {"confirm", "confirm_selection"}
            for action in _selection_actions(legal_actions)
        )
    return {
        "surface_key": surface_key,
        "selected": _transaction_selected_identities(observation, legal_actions),
        "selected_count": _transaction_selected_count(observation, legal_actions),
        "remaining": remaining,
        "can_confirm": bool(can_confirm),
    }


def _noncombat_liveness_policy_node_key(
    observation: Mapping[str, object],
    legal_actions: tuple[Mapping[str, object], ...],
) -> str | None:
    """Build a stable event-cycle policy node without weakening Q identity.

    The exact transaction ``node_key`` still owns complete reward-relevant
    state.  This second key exists only to decide whether the *same visible
    event-page action* has factually recurred while vitality and revival costs
    changed.  Durable resources remain in the key so an action that is really
    gaining gold, cards, relics or potions is not mislabeled as a loop.
    """

    if _combat_in_progress(observation) or sum(_semantic_action_enabled(action) for action in legal_actions) <= 1:
        return None
    page = _noncombat_event_page_projection(observation)
    if page is None:
        return None
    locus, resources = _noncombat_durable_projections(observation)
    selection = _noncombat_liveness_selection_projection(
        observation,
        legal_actions,
    )
    return semantic_fingerprint(
        {
            "kind": "noncombat_event_liveness_policy_node",
            "locus": locus,
            "resources": resources,
            "page": page,
            "decision_domain": observation.get("decision_domain"),
            "state_type": observation.get("state_type"),
            "selection": selection,
            "legal_actions": tuple(sorted(_semantic_surface_candidate_identity(action) for action in legal_actions)),
        }
    )


@dataclass(frozen=True, slots=True)
class _NonCombatEventCycleEvidence:
    window: int
    repeat_threshold: int
    occurrences: int
    cycle_span: int
    transition_fingerprint: str
    from_page_fingerprint: str
    to_page_fingerprint: str
    action_fingerprint: str
    policy_node_key: str
    detected_step: int
    context: Mapping[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": "noncombat_event_action_cycle",
            "window": self.window,
            "repeat_threshold": self.repeat_threshold,
            "occurrences": self.occurrences,
            "cycle_span": self.cycle_span,
            "transition_fingerprint": self.transition_fingerprint,
            "from_page_fingerprint": self.from_page_fingerprint,
            "to_page_fingerprint": self.to_page_fingerprint,
            "action_fingerprint": self.action_fingerprint,
            "policy_node_key": self.policy_node_key,
            "detected_step": self.detected_step,
            "context": dict(self.context),
        }


class _NonCombatEventCycleTracker:
    """Detect repeated factual event-page transitions under volatile vitality.

    A transition includes its source page, complete semantic action surface,
    chosen semantic action and factual successor page.  Therefore a novel exit
    or genuinely different choice surface at the threshold clears the
    recurrence naturally, while A<->B and same-page cycles are detected without
    naming an event or guessing which unexecuted option is the exit.
    """

    def __init__(self, *, window: int, repeat_threshold: int) -> None:
        if window < 2 or repeat_threshold < 2 or repeat_threshold > window:
            raise ValueError("event cycle window/threshold are inconsistent")
        self.window = int(window)
        self.repeat_threshold = int(repeat_threshold)
        self._history: deque[tuple[int, str]] = deque()
        self._counts: dict[str, int] = {}
        self._last_steps: dict[str, int] = {}
        self._locus_fingerprint = ""

    def reset(self) -> None:
        self._history.clear()
        self._counts.clear()
        self._last_steps.clear()
        self._locus_fingerprint = ""

    def _append(self, *, step: int, fingerprint: str) -> tuple[int, int]:
        if self._history and step <= self._history[-1][0]:
            raise RuntimeError("event-cycle observations must have increasing environment steps")

        # The emitted learner trace is bounded by environment decisions, not by
        # the number of policy choices. Forced pages may separate two choices
        # by hundreds of steps, so evict by absolute step age before counting a
        # recurrence. Otherwise an old choice absent from the learnable tail
        # could still manufacture a trusted AVOID target.
        while self._history and step - self._history[0][0] >= self.window:
            old_step, old = self._history.popleft()
            remaining = self._counts[old] - 1
            if remaining:
                self._counts[old] = remaining
                if self._last_steps.get(old) == old_step:
                    self._last_steps[old] = max(
                        historical_step for historical_step, historical in self._history if historical == old
                    )
            else:
                del self._counts[old]
                self._last_steps.pop(old, None)

        previous_step = self._last_steps.get(fingerprint)
        self._history.append((step, fingerprint))
        self._counts[fingerprint] = self._counts.get(fingerprint, 0) + 1
        self._last_steps[fingerprint] = step
        return self._counts[fingerprint], (0 if previous_step is None else step - previous_step)

    def observe(
        self,
        *,
        step: int,
        before_observation: Mapping[str, object],
        after_observation: Mapping[str, object],
        action_fingerprint: str,
        policy_node_key: str | None,
        durable_progress_kind: str,
        terminated: bool = False,
        truncated: bool = False,
    ) -> _NonCombatEventCycleEvidence | None:
        if terminated or truncated or _combat_in_progress(before_observation) or _combat_in_progress(after_observation):
            self.reset()
            return None
        before_page = _noncombat_event_page_projection(before_observation)
        after_page = _noncombat_event_page_projection(after_observation)
        if before_page is None or after_page is None:
            self.reset()
            return None
        before_locus, _ = _noncombat_durable_projections(before_observation)
        after_locus, _ = _noncombat_durable_projections(after_observation)
        before_locus_fingerprint = semantic_fingerprint(before_locus)
        after_locus_fingerprint = semantic_fingerprint(after_locus)
        if before_locus_fingerprint != after_locus_fingerprint:
            self.reset()
            return None
        if self._locus_fingerprint and self._locus_fingerprint != after_locus_fingerprint:
            self.reset()
        self._locus_fingerprint = after_locus_fingerprint
        # A genuinely novel structural resource state resets the actionable
        # cycle history.  HP/max HP/revival churn cannot produce this kind.
        if durable_progress_kind in {
            "forward_locus_changed",
            "new_durable_resource_state",
            "noncombat_started",
        }:
            self._history.clear()
            self._counts.clear()
            self._last_steps.clear()
            return None
        # The exact coarse node used by actor liveness credit is the authority
        # for recurrence too. This makes page/domain/state/legal-action/durable-
        # resource identity congruent with the eventual AVOID grouping. A
        # forced event transition has no such node; keep prior actionable
        # history across it but omit the forced action from recurrence counts.
        if policy_node_key is None:
            return None

        from_page_fingerprint = semantic_fingerprint(before_page)
        to_page_fingerprint = semantic_fingerprint(after_page)
        transition_fingerprint = semantic_fingerprint(
            {
                "policy_node_key": policy_node_key,
                "action": action_fingerprint,
                "to_page": to_page_fingerprint,
            }
        )
        occurrences, cycle_span = self._append(
            step=step,
            fingerprint=transition_fingerprint,
        )
        if occurrences < self.repeat_threshold:
            return None
        return _NonCombatEventCycleEvidence(
            window=self.window,
            repeat_threshold=self.repeat_threshold,
            occurrences=occurrences,
            cycle_span=cycle_span,
            transition_fingerprint=transition_fingerprint,
            from_page_fingerprint=from_page_fingerprint,
            to_page_fingerprint=to_page_fingerprint,
            action_fingerprint=action_fingerprint,
            policy_node_key=policy_node_key,
            detected_step=step,
            context=_noncombat_context_summary(after_observation),
        )


@dataclass(frozen=True, slots=True)
class _NonCombatDurableProgressStatus:
    age_steps: int
    maximum_age_steps: int
    stalled: bool
    durable_state_fingerprint: str
    locus_fingerprint: str
    resource_fingerprint: str
    action_fingerprint: str
    progress_kind: str
    context: Mapping[str, object]


class _NonCombatDurableProgressTracker:
    """Bound a repeated non-combat action that makes no durable run progress."""

    def __init__(self, *, window: int) -> None:
        self.window = int(window)
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._anchor_step = 0
        self._locus_fingerprint = ""
        self._resource_fingerprint = ""
        self._durable_state_fingerprint = ""
        self._action_fingerprint = ""
        self._seen_resource_fingerprints: set[str] = set()
        self._maximum_age = 0

    @staticmethod
    def _outside_noncombat(
        observation: Mapping[str, object],
        *,
        terminated: bool = False,
        truncated: bool = False,
    ) -> bool:
        return bool(
            _combat_in_progress(observation)
            or terminated
            or truncated
            or observation.get("terminated") is True
            or observation.get("truncated") is True
        )

    @staticmethod
    def _fingerprints(
        observation: Mapping[str, object],
    ) -> tuple[str, str, str]:
        locus, resources = _noncombat_durable_projections(observation)
        locus_fingerprint = semantic_fingerprint(locus)
        resource_fingerprint = semantic_fingerprint(resources)
        durable_fingerprint = semantic_fingerprint(
            {
                "locus": locus_fingerprint,
                "resources": resource_fingerprint,
            }
        )
        return locus_fingerprint, resource_fingerprint, durable_fingerprint

    def _start_locus(
        self,
        *,
        step: int,
        locus_fingerprint: str,
        resource_fingerprint: str,
        durable_fingerprint: str,
        action_fingerprint: str,
    ) -> None:
        self._active = True
        self._anchor_step = step
        self._locus_fingerprint = locus_fingerprint
        self._resource_fingerprint = resource_fingerprint
        self._durable_state_fingerprint = durable_fingerprint
        self._action_fingerprint = action_fingerprint
        self._seen_resource_fingerprints = {resource_fingerprint}

    def _status(
        self,
        *,
        age: int,
        progress_kind: str,
        context: Mapping[str, object],
    ) -> _NonCombatDurableProgressStatus:
        self._maximum_age = max(self._maximum_age, age)
        return _NonCombatDurableProgressStatus(
            age_steps=age,
            maximum_age_steps=self._maximum_age,
            stalled=age >= self.window,
            durable_state_fingerprint=self._durable_state_fingerprint,
            locus_fingerprint=self._locus_fingerprint,
            resource_fingerprint=self._resource_fingerprint,
            action_fingerprint=self._action_fingerprint,
            progress_kind=progress_kind,
            context=context,
        )

    def seed(
        self,
        *,
        step: int,
        observation: Mapping[str, object],
        terminated: bool = False,
        truncated: bool = False,
    ) -> _NonCombatDurableProgressStatus:
        """Seed the reset state so the configured window starts at step zero."""

        context = _noncombat_context_summary(observation)
        if self._outside_noncombat(
            observation,
            terminated=terminated,
            truncated=truncated,
        ):
            return self._status(
                age=0,
                progress_kind="outside_noncombat_decision",
                context=context,
            )
        locus, resources, durable = self._fingerprints(observation)
        self._start_locus(
            step=step,
            locus_fingerprint=locus,
            resource_fingerprint=resources,
            durable_fingerprint=durable,
            action_fingerprint="",
        )
        return self._status(
            age=0,
            progress_kind="noncombat_seeded",
            context=context,
        )

    def observe(
        self,
        *,
        step: int,
        observation: Mapping[str, object],
        selected_action: Mapping[str, object],
        terminated: bool = False,
        truncated: bool = False,
    ) -> _NonCombatDurableProgressStatus:
        context = _noncombat_context_summary(observation)
        if self._outside_noncombat(
            observation,
            terminated=terminated,
            truncated=truncated,
        ):
            maximum_age = self._maximum_age
            self.reset()
            self._maximum_age = maximum_age
            return self._status(
                age=0,
                progress_kind="outside_noncombat_decision",
                context=context,
            )

        locus_fingerprint, resource_fingerprint, durable_fingerprint = self._fingerprints(observation)
        action_fingerprint = semantic_fingerprint(_durable_action_projection(selected_action))
        if not self._active:
            progress_kind = "noncombat_started"
            self._start_locus(
                step=step,
                locus_fingerprint=locus_fingerprint,
                resource_fingerprint=resource_fingerprint,
                durable_fingerprint=durable_fingerprint,
                action_fingerprint=action_fingerprint,
            )
            return self._status(age=0, progress_kind=progress_kind, context=context)
        if locus_fingerprint != self._locus_fingerprint:
            self._start_locus(
                step=step,
                locus_fingerprint=locus_fingerprint,
                resource_fingerprint=resource_fingerprint,
                durable_fingerprint=durable_fingerprint,
                action_fingerprint=action_fingerprint,
            )
            return self._status(
                age=0,
                progress_kind="forward_locus_changed",
                context=context,
            )
        if resource_fingerprint not in self._seen_resource_fingerprints:
            self._seen_resource_fingerprints.add(resource_fingerprint)
            self._anchor_step = step
            self._resource_fingerprint = resource_fingerprint
            self._durable_state_fingerprint = durable_fingerprint
            self._action_fingerprint = action_fingerprint
            return self._status(
                age=0,
                progress_kind="new_durable_resource_state",
                context=context,
            )
        # Recurring A<->B (or larger finite) resource cycles are not forward
        # progress. Only the first sighting of a state at this locus earns a
        # reset; subsequent visits age from the last genuinely novel state.
        self._resource_fingerprint = resource_fingerprint
        self._durable_state_fingerprint = durable_fingerprint
        self._action_fingerprint = action_fingerprint
        return self._status(
            age=step - self._anchor_step,
            progress_kind="recurring_durable_resource_state",
            context=context,
        )


def _diagnostic_nonnegative_count(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CollectionProtocolError(f"{label} must be a non-negative integer")
    return value


def _diagnostic_card_quantity_total(value: object) -> int | None:
    if isinstance(value, Mapping):
        explicit = value.get("count")
        if explicit is not None:
            if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 0:
                raise CollectionProtocolError("diagnostic card collection count must be a non-negative integer")
            return int(explicit)
        value = value.get("cards", value.get("items"))
    if not isinstance(value, list | tuple):
        return None
    total = 0
    for card in value:
        if not isinstance(card, Mapping) or "quantity" not in card:
            total += 1
            continue
        raw = card["quantity"]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise CollectionProtocolError("diagnostic card quantity must be a positive integer")
        total += raw
    return total


def _zone_count(observation: Mapping[str, object], key: str) -> int:
    player = observation.get("player")
    combat = observation.get("combat")
    for owner in (player, combat):
        if not isinstance(owner, Mapping):
            continue
        explicit = _diagnostic_nonnegative_count(
            owner.get(f"{key}_count"),
            label=f"diagnostic {key}_count",
        )
        if explicit is not None:
            return explicit
        total = _diagnostic_card_quantity_total(owner.get(key))
        if total is not None:
            return total
    return 0


def _legal_action_kind_counts(
    legal_actions: tuple[dict[str, object], ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in legal_actions:
        kind = str(action.get("model_action_kind", action.get("kind", "unknown")) or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


_COMBAT_STALL_HAND_PREVIEW_LIMIT = 8


def _visible_zone_cards(
    observation: Mapping[str, object],
    key: str,
) -> tuple[Mapping[str, object], ...] | None:
    """Return visible cards for a diagnostic, without treating redaction as empty."""

    player = observation.get("player")
    combat = observation.get("combat")
    value = player.get(key) if isinstance(player, Mapping) else None
    if value is None and isinstance(combat, Mapping):
        value = combat.get(key)
    if isinstance(value, Mapping):
        value = value.get("cards", value.get("items"))
    if not isinstance(value, list | tuple):
        return None
    return tuple(card for card in value if isinstance(card, Mapping))


def _diagnostic_zone_count(
    observation: Mapping[str, object],
    key: str,
) -> int | None:
    """Read a visible pile count while preserving unknown/redacted as None."""

    player = observation.get("player")
    combat = observation.get("combat")
    owners = tuple(owner for owner in (player, combat) if isinstance(owner, Mapping))
    for owner in owners:
        count = _diagnostic_nonnegative_count(
            owner.get(f"{key}_count"),
            label=f"diagnostic {key}_count",
        )
        if count is not None:
            return count
        value = owner.get(key)
        total = _diagnostic_card_quantity_total(value)
        if total is not None:
            return total
        scalar = _diagnostic_nonnegative_count(
            value,
            label=f"diagnostic {key}",
        )
        if scalar is not None:
            return scalar
    return None


def _bounded_stable_id(value: object) -> str:
    """Bound an externally supplied identifier while retaining stable identity."""

    text = str(value).strip()
    encoded = text.encode("utf-8")
    if len(encoded) <= 128:
        return text
    prefix = encoded[:64].decode("utf-8", errors="ignore")
    return f"{prefix}#sha256:{semantic_fingerprint({'stable_id': text})}"


def _combat_stall_hand_preview(
    observation: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Return at most a few stable hand identities and explicit playability facts."""

    cards = _visible_zone_cards(observation, "hand")
    if cards is None:
        return ()
    preview: list[dict[str, object]] = []
    for card in cards:
        identity = next(
            (
                card[key]
                for key in (
                    "id",
                    "card_id",
                    "model_id",
                    "definition_id",
                    "card_definition_id",
                )
                if card.get(key) not in (None, "")
            ),
            None,
        )
        if identity is None:
            continue
        item: dict[str, object] = {"id": _bounded_stable_id(identity)}
        for key in ("is_playable", "playable", "can_play"):
            playable = card.get(key)
            if isinstance(playable, bool):
                item["is_playable"] = playable
                break
        preview.append(item)
    preview.sort(key=lambda item: (str(item["id"]), repr(item.get("is_playable"))))
    return tuple(preview[:_COMBAT_STALL_HAND_PREVIEW_LIMIT])


def _combat_stall_evidence(
    *,
    status: _CombatNetProgressStatus,
    observation: Mapping[str, object],
    legal_actions: tuple[dict[str, object], ...],
    window_selection: _CombatProgressWindowSelection,
    detected_step: int,
) -> dict[str, object]:
    """Build one bounded, mechanics-agnostic diagnostic at stall termination."""

    hand_preview = _combat_stall_hand_preview(observation)
    hand_count = _diagnostic_zone_count(observation, "hand")
    return {
        "kind": "combat_no_net_progress",
        "window": window_selection.effective_window,
        "default_window": window_selection.default_window,
        "window_source": window_selection.source,
        "window_match_id": window_selection.match_id or None,
        "room_model_id": window_selection.room_model_id or None,
        "encounter_id": window_selection.encounter_id or None,
        "steps_without_net_progress": status.age_steps,
        "anchor_enemy_hp_total": status.anchor_hp,
        "current_enemy_hp_total": status.current_hp,
        "current_enemy_max_hp_total": status.maximum_hp,
        "net_enemy_hp_progress": status.net_hp_progress,
        "required_net_enemy_hp_progress": status.required_hp_progress,
        "progress_kind": status.progress_kind,
        "detected_step": detected_step,
        "hand_cards": hand_count,
        "draw_cards": _diagnostic_zone_count(observation, "draw_pile"),
        "discard_cards": _diagnostic_zone_count(observation, "discard_pile"),
        "exhaust_cards": _diagnostic_zone_count(observation, "exhaust_pile"),
        "legal_action_kinds": _legal_action_kind_counts(legal_actions),
        "hand_card_preview": hand_preview,
        "hand_card_preview_truncated": bool(hand_count is not None and hand_count > len(hand_preview)),
    }


def _failure_credit_game_version(state: EnvironmentResult) -> str:
    """Return an auditable game build identity without guessing from schemas."""

    sources: tuple[Mapping[str, object], ...] = (
        state.observation,
        state.info,
        state.raw,
    )
    for source in sources:
        for key in ("game_version", "game_build", "build_id"):
            raw = source.get(key)
            if raw is not None and str(raw).strip():
                return str(raw).strip()
        nested = source.get("game")
        if isinstance(nested, Mapping):
            for key in ("version", "build", "build_id"):
                raw = nested.get(key)
                if raw is not None and str(raw).strip():
                    return str(raw).strip()
    # Absence is itself useful provenance.  Never substitute a protocol or
    # mod schema version and pretend it identifies the game build.
    return "game-version:not-exposed"


class GroundedCollector:
    """Collect policy trajectories with no MCTS, guard, or action rewrite."""

    def __init__(
        self,
        *,
        model: RecurrentCandidateModel,
        encoder: GroundedObservationEncoder,
        backend: EnvironmentBackend,
        scenario: Literal["full-run", "combat"],
        objective: Literal["combat", "act1", "run"],
        discount: float,
        max_episode_steps: int,
        character: str | None = None,
        encounter_id: str | None = None,
        seed: int = 0,
        unroll_length: int = 64,
        deadlock_window: int = 128,
        deadlock_repeat_threshold: int = 8,
        combat_net_progress_window: int = 256,
        combat_net_progress_room_windows: Mapping[str, int] | None = None,
        combat_net_progress_encounter_windows: Mapping[str, int] | None = None,
        noncombat_durable_progress_window: int = 256,
        combat_min_net_hp_fraction: float = 0.05,
        journal_policy_topk: int = 5,
        reward_calculator: RewardCalculator | None = None,
        additional_relics: tuple[str, ...] = (),
        training_revival_budget: int | None = None,
        horizon_as_failure: bool = False,
        transaction_burn_in_steps: int | None = None,
        transaction_smdp_horizon: Literal[
            "transaction_exit", "next_rest_or_act"
        ] = "transaction_exit",
        episodic_learning_enabled: bool = False,
        failure_credit_shadow_enabled: bool = False,
        failure_credit_learning_enabled: bool = False,
        failure_credit_run_id: str | None = None,
        failure_credit_pipeline_config: FailureCreditPipelineConfig | None = None,
        selection_surface_epsilon_floor: float = 0.0,
        transaction_exploration_operations: tuple[str, ...] = (),
        transaction_entry_epsilon_floor: float = 0.0,
        transaction_completion_guidance_probability: float = 0.0,
    ) -> None:
        if scenario not in {"full-run", "combat"}:
            raise ValueError("scenario must be full-run or combat")
        if objective not in {"combat", "act1", "run"}:
            raise ValueError("objective must be combat, act1, or run")
        if scenario == "combat" and objective != "combat":
            raise ValueError("combat scenario requires the combat objective")
        if scenario == "full-run" and objective == "combat":
            raise ValueError("full-run scenario requires the act1 or run objective")
        if (
            isinstance(discount, bool)
            or not isinstance(discount, int | float)
            or not math.isfinite(float(discount))
            or not 0.0 < float(discount) <= 1.0
        ):
            raise ValueError("discount must be in (0, 1]")
        if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int) or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("collector seed must be a non-negative integer")
        if isinstance(unroll_length, bool) or not isinstance(unroll_length, int):
            raise TypeError("unroll_length must be an integer")
        if unroll_length <= 0:
            raise ValueError("unroll_length must be positive")
        if transaction_burn_in_steps is not None and (
            isinstance(transaction_burn_in_steps, bool)
            or not isinstance(transaction_burn_in_steps, int)
            or transaction_burn_in_steps < 0
        ):
            raise ValueError("transaction_burn_in_steps must be non-negative or null")
        if transaction_smdp_horizon not in {
            "transaction_exit",
            "next_rest_or_act",
        }:
            raise ValueError("unsupported transaction SMDP horizon")
        if not isinstance(episodic_learning_enabled, bool):
            raise TypeError("episodic_learning_enabled must be a boolean")
        if episodic_learning_enabled and (scenario != "full-run" or objective != "run"):
            raise ValueError("episodic learning requires the full-run scenario and run objective")
        if transaction_smdp_horizon == "next_rest_or_act" and not episodic_learning_enabled:
            raise ValueError(
                "extended transaction SMDP horizons require episodic factual boundaries"
            )
        if not isinstance(failure_credit_shadow_enabled, bool):
            raise TypeError("failure_credit_shadow_enabled must be a boolean")
        if not isinstance(failure_credit_learning_enabled, bool):
            raise TypeError("failure_credit_learning_enabled must be a boolean")
        if failure_credit_shadow_enabled and failure_credit_learning_enabled:
            raise ValueError("failure-credit shadow and learning modes are mutually exclusive")
        if failure_credit_run_id is not None and (
            not isinstance(failure_credit_run_id, str) or not failure_credit_run_id.strip()
        ):
            raise ValueError("failure_credit_run_id must be a non-empty string or null")
        if failure_credit_pipeline_config is not None and not isinstance(
            failure_credit_pipeline_config,
            FailureCreditPipelineConfig,
        ):
            raise TypeError("failure_credit_pipeline_config has the wrong type")
        if isinstance(journal_policy_topk, bool) or not isinstance(journal_policy_topk, int):
            raise TypeError("journal_policy_topk must be an integer")
        if journal_policy_topk <= 0:
            raise ValueError("journal_policy_topk must be positive")
        if (
            isinstance(selection_surface_epsilon_floor, bool)
            or not isinstance(selection_surface_epsilon_floor, int | float)
            or not math.isfinite(float(selection_surface_epsilon_floor))
            or not 0.0 <= float(selection_surface_epsilon_floor) <= 1.0
        ):
            raise ValueError("selection_surface_epsilon_floor must be finite and in [0, 1]")
        if not isinstance(transaction_exploration_operations, tuple):
            raise TypeError("transaction_exploration_operations must be a tuple")
        normalized_transaction_operations = frozenset(
            _canonical_transaction_operation(operation) for operation in transaction_exploration_operations
        )
        if "" in normalized_transaction_operations or len(normalized_transaction_operations) != len(
            transaction_exploration_operations
        ):
            raise ValueError("transaction_exploration_operations must contain unique reviewed operations")
        if normalized_transaction_operations - TRANSACTION_EXPLORATION_OPERATIONS:
            raise ValueError("transaction_exploration_operations contains an unsupported operation")
        for name, value in (
            ("transaction_entry_epsilon_floor", transaction_entry_epsilon_floor),
            (
                "transaction_completion_guidance_probability",
                transaction_completion_guidance_probability,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if transaction_completion_guidance_probability >= 1.0:
            raise ValueError("transaction_completion_guidance_probability must be less than 1")
        if normalized_transaction_operations:
            if transaction_entry_epsilon_floor <= 0.0:
                raise ValueError("transaction exploration operations require a positive entry epsilon floor")
            guidance_operations = (
                normalized_transaction_operations
                & TRANSACTION_GUIDANCE_OPERATIONS
            )
            if guidance_operations and transaction_completion_guidance_probability <= 0.0:
                raise ValueError(
                    "selection transaction exploration operations require positive completion guidance"
                )
            if (
                not guidance_operations
                and transaction_completion_guidance_probability != 0.0
            ):
                raise ValueError(
                    "single-decision transaction exploration requires zero completion guidance"
                )
        elif transaction_entry_epsilon_floor != 0.0 or transaction_completion_guidance_probability != 0.0:
            raise ValueError("transaction exploration probabilities require reviewed operations")
        if isinstance(combat_net_progress_window, bool) or not isinstance(combat_net_progress_window, int):
            raise TypeError("combat_net_progress_window must be an integer")
        if combat_net_progress_window <= 0:
            raise ValueError("combat_net_progress_window must be positive")
        if isinstance(noncombat_durable_progress_window, bool) or not isinstance(
            noncombat_durable_progress_window, int
        ):
            raise TypeError("noncombat_durable_progress_window must be an integer")
        if noncombat_durable_progress_window <= 0:
            raise ValueError("noncombat_durable_progress_window must be positive")
        if (
            isinstance(combat_min_net_hp_fraction, bool)
            or not isinstance(combat_min_net_hp_fraction, int | float)
            or not math.isfinite(float(combat_min_net_hp_fraction))
            or not 0.0 < float(combat_min_net_hp_fraction) <= 1.0
        ):
            raise ValueError("combat_min_net_hp_fraction must be in (0, 1]")
        self.model = model
        self.encoder = encoder
        self.backend = backend
        self.scenario = scenario
        self.objective = objective
        self.discount = float(discount)
        self.max_episode_steps = int(max_episode_steps)
        self.character = character
        self.encounter_id = encounter_id
        self.unroll_length = unroll_length
        self.journal_policy_topk = journal_policy_topk
        self.combat_net_progress_window = combat_net_progress_window
        self.combat_net_progress_room_windows = _normalize_combat_progress_windows(
            combat_net_progress_room_windows,
            label="combat_net_progress_room_windows",
        )
        self.combat_net_progress_encounter_windows = _normalize_combat_progress_windows(
            combat_net_progress_encounter_windows,
            label="combat_net_progress_encounter_windows",
        )
        self.noncombat_durable_progress_window = noncombat_durable_progress_window
        self.deadlock_window = deadlock_window
        self.deadlock_repeat_threshold = deadlock_repeat_threshold
        self.combat_min_net_hp_fraction = float(combat_min_net_hp_fraction)
        self.additional_relics = tuple(str(item) for item in additional_relics)
        self.training_revival_budget = training_revival_budget
        self.horizon_as_failure = bool(horizon_as_failure)
        self.transaction_burn_in_steps = transaction_burn_in_steps
        self.transaction_smdp_horizon = transaction_smdp_horizon
        self.episodic_learning_enabled = episodic_learning_enabled
        self.failure_credit_shadow_enabled = failure_credit_shadow_enabled
        self.failure_credit_learning_enabled = failure_credit_learning_enabled
        self.selection_surface_epsilon_floor = float(selection_surface_epsilon_floor)
        self.transaction_exploration_operations = normalized_transaction_operations
        self.transaction_entry_epsilon_floor = float(transaction_entry_epsilon_floor)
        self.transaction_completion_guidance_probability = float(transaction_completion_guidance_probability)
        self.failure_credit_pipeline_config = failure_credit_pipeline_config or FailureCreditPipelineConfig(
            detector_window_steps=deadlock_window,
            context_burn_in_steps=transaction_burn_in_steps or 0,
        )
        self._failure_credit_run_id = failure_credit_run_id
        if self.training_revival_budget is not None:
            if isinstance(self.training_revival_budget, bool) or not isinstance(self.training_revival_budget, int):
                raise TypeError("training_revival_budget must be an integer or null")
            if self.training_revival_budget < -1:
                raise ValueError("training_revival_budget must be -1 or non-negative")
        self._rng = np.random.default_rng(int(seed))
        self._episode_seed = training_seed_start(int(seed))
        self._active_state_version: int | None = None
        self.reward_calculator = reward_calculator or TaskRewardCalculator(
            objective,
            discount=discount,
        )
        self.deadlock_detector = SemanticDeadlockDetector(
            window_size=deadlock_window,
            repeat_threshold=deadlock_repeat_threshold,
        )

    @property
    def failure_credit_enabled(self) -> bool:
        return bool(self.failure_credit_shadow_enabled or self.failure_credit_learning_enabled)

    def bind_failure_credit_run_id(self, run_id: str) -> None:
        """Bind runtime provenance exactly once before recorded collection."""

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("failure-credit run_id must be a non-empty string")
        normalized = run_id.strip()
        if self._failure_credit_run_id is not None and self._failure_credit_run_id != normalized:
            raise RuntimeError("failure-credit collector is already bound to a different run_id")
        self._failure_credit_run_id = normalized

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def state_dict(self) -> dict[str, object]:
        """Return every collector-owned stochastic continuation input.

        Checkpoints are only published between episodes, so no live environment
        state belongs here.  The next reset seed and the epsilon-action RNG are
        nevertheless part of an exact continuation and must not silently reset.
        """

        return {
            "version": "sts2-recurrent-collector-state-v3",
            "episode_seed": self._episode_seed,
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        """Restore a fail-closed collector continuation state."""

        expected_keys = {"version", "episode_seed", "rng_state"}
        actual_keys = set(payload)
        if actual_keys != expected_keys:
            raise ValueError(
                "collector state keys mismatch: "
                f"missing={sorted(expected_keys - actual_keys)} "
                f"unknown={sorted(actual_keys - expected_keys)}"
            )
        if payload["version"] != "sts2-recurrent-collector-state-v3":
            raise ValueError("unsupported collector checkpoint state")
        episode_seed = payload["episode_seed"]
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("collector episode_seed must be an integer")
        if episode_seed < 0:
            raise ValueError("collector episode_seed must be non-negative")
        if episode_seed > SIGNED_INT32_MAX or episode_seed % 2 != 0:
            raise ValueError("collector episode_seed must be an even signed 32-bit seed")
        rng_state = payload["rng_state"]
        if not isinstance(rng_state, Mapping):
            raise TypeError("collector rng_state must be a mapping")
        candidate_rng = np.random.default_rng()
        try:
            candidate_rng.bit_generator.state = dict(deepcopy(rng_state))
        except (TypeError, ValueError) as exc:
            raise ValueError("collector rng_state is invalid") from exc
        self._episode_seed = episode_seed
        self._rng = candidate_rng

    def replace_backend(self, backend: EnvironmentBackend) -> None:
        """Adopt a fresh backend at a quiescent incident boundary.

        A poisoned simulator may already have committed the mutation whose
        result was rejected.  Its active revision must never leak into a new
        process/session, so the collector deliberately forgets that revision.
        """

        self.backend = backend
        self._active_state_version = None

    def _state_version(self) -> int:
        state = self.backend.get_state()
        if not isinstance(state, Mapping) or state.get("ok") is not True:
            raise CollectionProtocolError("backend state read was not explicitly successful")
        revision = state.get("state_version")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise CollectionProtocolError("backend state_version must be an exact non-negative integer")
        return revision

    @staticmethod
    def _validate_common_result(result: EnvironmentResult, *, surface: str) -> None:
        if result.ok is not True:
            raise CollectionProtocolError(f"{surface} result was not explicitly successful")
        if not isinstance(result.episode_id, str) or not result.episode_id.strip():
            raise CollectionProtocolError(f"{surface} result has no episode identity")
        if isinstance(result.step_index, bool) or not isinstance(result.step_index, int) or result.step_index < 0:
            raise CollectionProtocolError(f"{surface} result step_index must be an exact non-negative integer")
        if not isinstance(result.terminated, bool) or not isinstance(result.truncated, bool):
            raise CollectionProtocolError(f"{surface} terminated/truncated flags must be booleans")
        if result.terminated and result.truncated:
            raise CollectionProtocolError(f"{surface} result cannot be both terminated and truncated")
        if result.terminal_reason is not None and not isinstance(result.terminal_reason, str):
            raise CollectionProtocolError(f"{surface} terminal_reason must be a string or null")
        if not isinstance(result.info, Mapping):
            raise CollectionProtocolError(f"{surface} info must be an object")

    def _validate_reset_result(
        self,
        result: EnvironmentResult,
        *,
        before_state_version: int,
        after_state_version: int,
    ) -> None:
        self._validate_common_result(result, surface="reset")
        if result.step_index != 0:
            raise CollectionProtocolError("fresh reset result must start at step_index=0")
        if result.terminated or result.truncated:
            raise CollectionProtocolError("fresh reset returned a terminal/truncated episode")
        if not result.legal_actions:
            raise CollectionProtocolError("fresh reset returned zero legal actions")
        if result.info.get("reward_authority") != "external-rl":
            raise CollectionProtocolError("reset result must delegate reward authority to external-rl")
        transition = result.transition
        if transition is None:
            raise CollectionProtocolError("reset result has no typed transition")
        if transition.episode_id != result.episode_id or transition.step_index != 0:
            raise CollectionProtocolError("reset transition identity does not match result")
        if (
            transition.before_state_version != before_state_version
            or transition.after_state_version != after_state_version
        ):
            raise CollectionProtocolError("reset transition revision identity is inconsistent")
        if not isinstance(transition.facts, Mapping):
            raise CollectionProtocolError("reset transition facts must be an object")

    def _validate_step_result(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
    ) -> None:
        self._validate_common_result(after, surface="step")
        if after.episode_id != before.episode_id:
            raise CollectionProtocolError("step result episode_id differs from the active episode")
        if after.step_index != before.step_index + 1:
            raise CollectionProtocolError("step result must advance step_index by exactly one")
        transition = after.transition
        if transition is None:
            raise CollectionProtocolError("step result has no typed transition")
        if transition.episode_id != after.episode_id or transition.step_index != after.step_index:
            raise CollectionProtocolError("step transition episode/step identity differs from result")
        if not isinstance(transition.facts, Mapping):
            raise CollectionProtocolError("step transition facts must be an object")
        if after.terminated:
            fact_name = "combat_result" if self.objective == "combat" else "run_result"
            terminal_result = transition.facts.get(fact_name)
            supported_results = (
                {"victory", "defeat", "escaped"} if self.objective == "combat" else {"victory", "defeat"}
            )
            if terminal_result not in supported_results:
                raise CollectionProtocolError(
                    f"terminated {self.objective} step has no authoritative typed {fact_name}"
                )
            expected_reason = f"{'combat' if self.objective == 'combat' else 'run'}_{terminal_result}"
            if after.terminal_reason != expected_reason or transition.facts.get("terminal_reason") != expected_reason:
                raise CollectionProtocolError(
                    "terminated step facts/result reason disagrees with its objective-scoped result"
                )
        active_revision = self._active_state_version
        if active_revision is None:
            raise CollectionProtocolError("collector has no active state revision")
        before_revision = transition.before_state_version
        after_revision = transition.after_state_version
        if (
            isinstance(before_revision, bool)
            or not isinstance(before_revision, int)
            or isinstance(after_revision, bool)
            or not isinstance(after_revision, int)
            or before_revision < 0
            or after_revision < 0
        ):
            raise CollectionProtocolError("step transition revisions must be exact non-negative integers")
        if before_revision != active_revision or after_revision != before_revision + 1:
            raise CollectionProtocolError("step transition revision chain is stale or non-contiguous")
        if after.info.get("reward_authority") != "external-rl":
            raise CollectionProtocolError("step result must delegate reward authority to external-rl")
        if not after.terminated and not after.truncated and not after.legal_actions:
            raise CollectionProtocolError("non-terminal step result returned zero legal actions")
        if (after.terminated or after.truncated) and after.legal_actions:
            raise CollectionProtocolError("terminal/truncated step result returned legal actions")
        self._active_state_version = after_revision

    def reset(self, *, evaluation_seed: int | None = None) -> EnvironmentResult:
        request_id = str(uuid4())
        expected_state_version = self._state_version()
        if evaluation_seed is None:
            seed = self._episode_seed
        else:
            if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
                raise TypeError("evaluation seed must be an integer")
            if (
                evaluation_seed < 0
                or evaluation_seed > SIGNED_INT32_MAX
                or evaluation_seed % 2 != EVALUATION_SEED_PARITY
            ):
                raise ValueError("evaluation seed must be an odd signed 32-bit seed")
            seed = evaluation_seed
        if self.scenario == "combat":
            result = self.backend.combat_reset(
                CombatResetRequest(
                    request_id=request_id,
                    session_id=self.backend.session_id,
                    expected_state_version=expected_state_version,
                    character=self.character,
                    encounter_id=self.encounter_id,
                    seed=seed,
                    additional_relics=self.additional_relics or None,
                    training_revival_budget=self.training_revival_budget,
                )
            )
        else:
            result = self.backend.reset(
                ResetRequest(
                    request_id=request_id,
                    session_id=self.backend.session_id,
                    scenario="full-run",
                    expected_state_version=expected_state_version,
                    character=self.character,
                    seed=seed,
                    force_fresh=True,
                    additional_relics=self.additional_relics or None,
                    training_revival_budget=self.training_revival_budget,
                )
            )
        committed_state_version = self._state_version()
        if committed_state_version != expected_state_version + 1:
            raise CollectionProtocolError("reset did not advance state_version by exactly one")
        self._validate_reset_result(
            result,
            before_state_version=expected_state_version,
            after_state_version=committed_state_version,
        )
        self._active_state_version = committed_state_version
        if evaluation_seed is None:
            self._episode_seed += 2
        return result

    def _choose_action(
        self,
        state: EnvironmentResult,
        recurrent_state: torch.Tensor,
        *,
        epsilon: float,
        deterministic: bool,
    ) -> _ActionChoice:
        normalized_epsilon = float(epsilon)
        if not math.isfinite(normalized_epsilon) or not 0.0 <= normalized_epsilon <= 1.0:
            raise ValueError("exploration epsilon must be finite and in [0, 1]")
        encoding_started_ns = time.perf_counter_ns()
        encoded = self.encoder.encode(
            state.observation,
            state.legal_actions,
            device=self.device,
        )
        semantic_actions = _semantic_action_surface(encoded.semantic_groups)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        encoding_ms = (time.perf_counter_ns() - encoding_started_ns) / 1_000_000.0
        policy_started_ns = time.perf_counter_ns()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(
                    encoded.batch,
                    recurrent_state,
                    validate=False,
                )
                policy = output.policy_probabilities()[0].float().cpu().numpy()
                model_log_probabilities = (
                    output.policy_log_probabilities()[0].float().cpu().numpy()
                )
                policy_branch_ids = output.policy_branch_ids[0].long().cpu().numpy()
                greedy_selected = int(output.greedy_action_indices()[0].item())
                valid = output.action_mask[0].cpu().numpy().astype(bool)
                next_recurrent_state = output.recurrent_state.detach()
                value = float(output.value[0].item())
        finally:
            self.model.train(was_training)
        policy_forward_ms = (time.perf_counter_ns() - policy_started_ns) / 1_000_000.0
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            raise CollectionProtocolError(
                f"episode={state.episode_id!r} step={state.step_index} has no enabled legal action"
            )
        valid_count = int(valid_indices.size)
        valid_policy = policy[valid_indices]
        if not np.all(np.isfinite(valid_policy)) or np.any(valid_policy < 0.0):
            raise CollectionProtocolError("model produced a non-finite policy distribution")
        policy_mass = float(valid_policy.sum())
        if not math.isfinite(policy_mass) or policy_mass <= 0.0:
            raise CollectionProtocolError("model produced zero/non-finite legal policy mass")
        if deterministic:
            selected = greedy_selected
            if selected < 0 or selected >= len(valid) or not bool(valid[selected]):
                raise CollectionProtocolError("model produced an invalid hierarchical greedy action")
            reference = encoded.action(selected)
            return _ActionChoice(
                candidate_index=selected,
                dispatch_position=reference.position,
                dispatch_handle=reference.handle,
                equivalence_fingerprint=reference.equivalence_fingerprint,
                multiplicity=reference.multiplicity,
                action_references=encoded.actions,
                semantic_actions=semantic_actions,
                # Deterministic collection samples from a delta behavior
                # policy.  This is irrelevant to held-out evaluation, but a
                # recorded training liveness probe must expose log(1)=0 to
                # V-trace/episodic importance weighting instead of pretending
                # that it sampled from the model distribution.
                behavior_log_probability=0.0,
                model_log_probability=float(model_log_probabilities[selected]),
                valid_count=valid_count,
                snapshot=encoded.snapshot,
                recurrent_state=next_recurrent_state,
                policy=policy,
                value=value,
                encoding_ms=encoding_ms,
                policy_forward_ms=policy_forward_ms,
                definition_hash_collisions=(encoded.definition_hash_collisions),
                relation_hash_collisions=encoded.relation_hash_collisions,
                effective_epsilon=0.0,
                targeted_selection_exploration=False,
                targeted_transaction_entry_exploration=False,
                transaction_completion_guidance=False,
                transaction_completion_forward_selected=False,
                transaction_completion_guidance_fallback=False,
                transaction_operation=_selection_transaction_operation(
                    state.observation,
                    semantic_actions,
                ),
            )

        targeted_surface = _uses_targeted_selection_exploration(
            state.observation,
            semantic_actions,
        )
        exploration_branch_ids, targeted_transaction_entry_surface = _transaction_entry_exploration_branch_ids(
            policy_branch_ids=policy_branch_ids,
            semantic_actions=semantic_actions,
            enabled_operations=self.transaction_exploration_operations,
        )
        effective_epsilon = normalized_epsilon
        if targeted_surface:
            effective_epsilon = max(
                effective_epsilon,
                self.selection_surface_epsilon_floor,
            )
        if targeted_transaction_entry_surface:
            effective_epsilon = max(
                effective_epsilon,
                self.transaction_entry_epsilon_floor,
            )
        base_behavior = _branch_balanced_epsilon_behavior(
            policy=policy,
            valid=valid,
            policy_branch_ids=exploration_branch_ids,
            epsilon=effective_epsilon,
        )
        transaction_operation = _selection_transaction_operation(
            state.observation,
            semantic_actions,
        )
        guidance_eligible = bool(transaction_operation in self.transaction_exploration_operations)
        if guidance_eligible:
            behavior, forward_indices, guidance_fallback = _transaction_completion_guided_behavior(
                policy=policy,
                valid=valid,
                semantic_actions=semantic_actions,
                base_behavior=base_behavior,
                operation=transaction_operation,
                guidance_probability=(self.transaction_completion_guidance_probability),
            )
        else:
            behavior = base_behavior
            forward_indices = frozenset()
            guidance_fallback = False
        selected = int(self._rng.choice(len(behavior), p=behavior))
        reference = encoded.action(selected)
        return _ActionChoice(
            candidate_index=selected,
            dispatch_position=reference.position,
            dispatch_handle=reference.handle,
            equivalence_fingerprint=reference.equivalence_fingerprint,
            multiplicity=reference.multiplicity,
            action_references=encoded.actions,
            semantic_actions=semantic_actions,
            behavior_log_probability=float(math.log(max(float(behavior[selected]), 1e-30))),
            # Keep the model log-probability directly. Converting through a
            # float32 probability destroys exact provenance precisely in the
            # saturated-zero regime that the option actor bridge recovers.
            model_log_probability=float(model_log_probabilities[selected]),
            valid_count=valid_count,
            snapshot=encoded.snapshot,
            recurrent_state=next_recurrent_state,
            policy=policy,
            value=value,
            encoding_ms=encoding_ms,
            policy_forward_ms=policy_forward_ms,
            definition_hash_collisions=encoded.definition_hash_collisions,
            relation_hash_collisions=encoded.relation_hash_collisions,
            effective_epsilon=effective_epsilon,
            targeted_selection_exploration=bool(targeted_surface and effective_epsilon > normalized_epsilon),
            targeted_transaction_entry_exploration=bool(targeted_transaction_entry_surface and effective_epsilon > 0.0),
            transaction_completion_guidance=bool(guidance_eligible and not guidance_fallback),
            transaction_completion_forward_selected=bool(guidance_eligible and selected in forward_indices),
            transaction_completion_guidance_fallback=bool(guidance_fallback),
            transaction_operation=transaction_operation,
        )

    def _step(
        self,
        state: EnvironmentResult,
        *,
        dispatch_position: int,
        dispatch_handle: str | None,
    ) -> tuple[EnvironmentResult, str]:
        if not 0 <= dispatch_position < len(state.legal_actions):
            raise CollectionProtocolError(
                "encoder dispatch position is outside the raw legal-action range: "
                f"position={dispatch_position} count={len(state.legal_actions)}"
            )
        action = state.legal_actions[dispatch_position]
        handle_value = action.get("action_handle", action.get("action_id"))
        raw_handle = str(handle_value) if handle_value is not None and str(handle_value) else ""
        encoded_handle = str(dispatch_handle) if dispatch_handle is not None else ""
        if encoded_handle and raw_handle and encoded_handle != raw_handle:
            raise CollectionProtocolError("encoder dispatch handle disagrees with the representative raw action")
        handle = encoded_handle or raw_handle
        request = StepRequest(
            request_id=str(uuid4()),
            session_id=self.backend.session_id,
            episode_id=state.episode_id,
            expected_step_index=state.step_index,
            action_id=handle or None,
            action_index=None if handle else dispatch_position,
        )
        result = self.backend.step(request)
        self._validate_step_result(state, result)
        return result, handle or f"index:{dispatch_position}"

    def collect_episode(
        self,
        *,
        epsilon: float = 0.0,
        deterministic: bool = False,
        record: bool = True,
        evaluation_seed: int | None = None,
        policy_version: int = 0,
        trajectory_journal: TrajectoryJournal | None = None,
        maximum_steps: int | None = None,
        unroll_sink: Callable[[SequenceUnroll], int | None] | None = None,
        failure_credit_sink: Callable[[tuple[EvidenceRecord, ...]], None] | None = None,
        progress_sink: Callable[[EpisodeProgress], None] | None = None,
        accepted_step_sink: Callable[[int, int], None] | None = None,
        journal_episode_id_prefix: str = "",
        liveness_probe: bool = False,
    ) -> CollectedEpisode:
        if isinstance(policy_version, bool) or not isinstance(policy_version, int):
            raise TypeError("policy_version must be an integer")
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        if not isinstance(journal_episode_id_prefix, str):
            raise TypeError("journal_episode_id_prefix must be a string")
        if not isinstance(liveness_probe, bool):
            raise TypeError("liveness_probe must be a boolean")
        if liveness_probe:
            if not record:
                raise ValueError("liveness probes must be recorded training data")
            if evaluation_seed is not None:
                raise ValueError("liveness probes cannot use a held-out evaluation seed")
            # A probe executes the current greedy policy on the normal even
            # training-seed partition.  Unlike held-out evaluation it remains
            # recorded, so factual deadlock prefixes can enter transaction
            # replay and teach the policy to escape its own argmax failures.
            epsilon = 0.0
            deterministic = True
        if record and evaluation_seed is not None:
            raise ValueError("recorded training collection cannot use a held-out evaluation seed")
        episode_limit = self.max_episode_steps
        if maximum_steps is not None:
            if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
                raise TypeError("maximum_steps must be an integer or null")
            if maximum_steps <= 0:
                raise ValueError("maximum_steps must be positive")
            episode_limit = min(episode_limit, maximum_steps)
        timings = _CollectorTimingAccumulator()
        reset_seed = self._episode_seed if evaluation_seed is None else evaluation_seed
        reset_started_ns = time.perf_counter_ns()
        state = self.reset(evaluation_seed=evaluation_seed)
        timings.record("reset", reset_started_ns)
        failure_credit_pipeline: FailureCreditEpisodePipeline | None = None
        if record and self.failure_credit_enabled:
            run_id = self._failure_credit_run_id
            if run_id is None:
                raise RuntimeError(
                    "failure-credit collection requires bind_failure_credit_run_id() "
                    "before the first recorded episode"
                )
            provenance = FailureCreditEpisodePipeline.build_provenance(
                run_id=run_id,
                game_version=_failure_credit_game_version(state),
                environment_schema_version=ENVIRONMENT_SCHEMA_VERSION,
                policy_version=policy_version,
            )
            failure_credit_pipeline = FailureCreditEpisodePipeline(
                episode_id=f"seed-{reset_seed}:{state.episode_id}",
                provenance=provenance,
                config=self.failure_credit_pipeline_config,
            )
        self.deadlock_detector.reset()
        recurrent_state = self.model.initial_state(1, device=self.device)
        segment_initial_state = recurrent_state[0].detach().float().cpu().numpy().copy()
        segment_start_step = state.step_index
        segment_policy_version = policy_version
        segment_steps: list[RolloutStep] = []
        unrolls: list[SequenceUnroll] = []
        reward_total = 0.0
        max_act, max_floor = _run_position(state.observation)
        policy_decisions = 0
        forced_decisions = 0
        # Historical/checkpoint consumers interpret this field as the raw
        # backend surface.  Keep that meaning explicit; semantic model width
        # and strict-equivalence multiplicity are tracked separately below.
        maximum_observed_candidates = 0
        maximum_observed_semantic_candidates = 0
        maximum_equivalence_class_size = 0
        maximum_definition_hash_collisions_per_decision = 0
        maximum_relation_hash_collisions_per_decision = 0
        definition_hash_collisions_total = 0
        relation_hash_collisions_total = 0
        targeted_selection_exploration_decisions = 0
        targeted_transaction_entry_exploration_decisions = 0
        transaction_completion_guidance_decisions = 0
        transaction_completion_forward_decisions = 0
        transaction_completion_guidance_fallbacks = 0
        maximum_effective_collection_epsilon = 0.0
        policy_logit_margin_total = 0.0
        policy_logit_margin_count = 0
        policy_top1_top2_logit_margin_max = 0.0
        selection_transactions_started = 0
        selection_transactions_closed = 0
        selection_transactions_completed = 0
        selection_transactions_cancelled = 0
        selection_transactions_unresolved = 0
        rest_site_selection_transactions_started = 0
        rest_site_selection_transactions_closed = 0
        rest_site_selection_transactions_completed = 0
        rest_site_selection_transactions_cancelled = 0
        rest_site_selection_transactions_unresolved = 0
        forge_selection_transactions_started = 0
        forge_selection_transactions_closed = 0
        forge_selection_transactions_completed = 0
        forge_selection_transactions_cancelled = 0
        forge_selection_transactions_unresolved = 0
        shop_card_removal_transactions_started = 0
        shop_card_removal_transactions_closed = 0
        shop_card_removal_transactions_completed = 0
        shop_card_removal_transactions_cancelled = 0
        shop_card_removal_transactions_unresolved = 0
        shaping_reward_total = 0.0
        boss_victory_acts: set[int] = set()
        steps_taken = 0
        final_outcome = "ongoing"
        deadlocked = False
        combat_progress_stalled = False
        noncombat_progress_stalled = False
        noncombat_durable_stalled = False
        noncombat_event_cycle = False
        selection_action_cycle = False
        trusted_policy_failure = False
        stall_evidence: dict[str, object] | None = None
        event_cycle_evidence: _NonCombatEventCycleEvidence | None = None
        combat_window_selection = _combat_progress_window_selection(
            state.observation,
            default_window=self.combat_net_progress_window,
            room_windows=self.combat_net_progress_room_windows,
            encounter_windows=self.combat_net_progress_encounter_windows,
        )
        combat_progress = _CombatNetProgressTracker(
            window=combat_window_selection.effective_window,
            minimum_hp_fraction=self.combat_min_net_hp_fraction,
        )
        runaway_combat_guard = RunawayCombatGuard()
        noncombat_progress = _NonCombatDurableProgressTracker(
            window=self.noncombat_durable_progress_window,
        )
        # A trusted event recurrence must fit wholly inside the learnable tail
        # of its emitted global trace; otherwise the detector could prove an
        # old sparse repeat that no longer owns a factual actor AVOID label.
        liveness_learn_tail_steps = max(
            _LIVENESS_LEARN_TAIL_STEPS,
            self.deadlock_repeat_threshold,
        )
        noncombat_event_cycles = _NonCombatEventCycleTracker(
            window=min(self.deadlock_window, liveness_learn_tail_steps),
            repeat_threshold=self.deadlock_repeat_threshold,
        )
        noncombat_progress_status = noncombat_progress.seed(
            step=0,
            observation=state.observation,
            terminated=state.terminated,
            truncated=state.truncated,
        )
        combat_progress_status = combat_progress.observe(
            step=0,
            observation=state.observation,
        )
        runaway_combat_guard.observe(
            observation=state.observation,
            legal_actions=state.legal_actions,
            no_net_progress_steps=combat_progress_status.age_steps,
        )
        combat_in_progress = _combat_in_progress(state.observation)
        combat_no_net_progress_steps = combat_progress_status.age_steps
        maximum_combat_no_net_progress_steps = combat_progress_status.maximum_age_steps
        noncombat_no_durable_progress_steps = noncombat_progress_status.age_steps
        maximum_noncombat_no_durable_progress_steps = noncombat_progress_status.maximum_age_steps
        forced_horizon = False
        curriculum_horizon = False
        revivals_used = 0
        player_hp_lost = 0.0
        final_behavior_policy_version = policy_version
        selected_action_kind_counts: dict[str, int] = {}
        last_selected_action_kind = ""
        transaction_enabled = bool(record and self.transaction_burn_in_steps is not None)
        transaction_context: deque[tuple[int, TransactionStep]] = deque(
            maxlen=(self.transaction_burn_in_steps or 0) + liveness_learn_tail_steps
        )
        # A liveness window can end with a long forced-action suffix. Keep a
        # tiny factual prefix around the most recent actual policy decision so
        # a trusted deadlock still yields an actionable AVOID target instead
        # of a trace containing only singleton choices.
        last_liveness_policy_prefix: (
            tuple[
                int,
                tuple[TransactionStep, ...],
                int,
            ]
            | None
        ) = None
        transaction_traces_pending: list[TransactionTrace] = []
        active_transaction_surface: str | None = None
        active_transaction_steps: list[TransactionStep] = []
        active_transaction_start_step = 0
        active_transaction_burn_in = 0
        active_transaction_policy_version = policy_version
        active_transaction_seen_nodes: set[str] = set()
        active_transaction_entry_step_index: int | None = None
        active_transaction_entry_behavior_log_probability = 0.0
        active_transaction_entry_model_probability = 0.0
        active_transaction_entry_action_fingerprint = ""
        active_transaction_operation = ""
        observed_transaction_surface: str | None = None
        observed_transaction_entry_episodic_index: int | None = None
        observed_transaction_opened_from_rest_site = False
        observed_transaction_is_forge = False
        observed_transaction_operation = ""
        observed_transaction_is_shop_card_removal = False
        observed_transaction_has_policy_choice = False
        observed_transaction_opening_deck_signature: tuple[tuple[str, int, int], ...] | None = None
        observed_transaction_entry_behavior_log_probability = 0.0
        observed_transaction_entry_model_probability = 0.0
        observed_transaction_entry_action_fingerprint = ""
        observed_transaction_entry_policy_version = policy_version
        pending_observed_transaction_entry_episodic_index: int | None = None
        pending_observed_transaction_opened_from_rest_site = False
        pending_observed_transaction_entry_operation = ""
        pending_observed_transaction_is_shop_card_removal = False
        pending_transaction_entry_behavior_log_probability = 0.0
        pending_transaction_entry_model_probability = 0.0
        pending_transaction_entry_action_fingerprint = ""
        pending_transaction_entry_policy_version = policy_version
        episode_rewards: list[float] = []
        episode_discounts: list[float] = []
        episodic_diagnostic_rewards: list[float] = []
        episodic_diagnostic_discounts: list[float] = []
        episodic_macro_decisions: list[tuple[int, str, str, str, int]] = []
        # Complete-run replay is collected independently of fixed unroll
        # streaming.  These are immutable CPU snapshots, never recurrent
        # hidden tensors or autograd graphs.
        episodic_enabled = bool(record and self.episodic_learning_enabled and self.objective == "run")
        episodic_steps: list[EpisodeDecisionStep] = []
        next_combat_identity = 0
        active_combat_id: str | None = None
        if episodic_enabled and combat_in_progress:
            active_combat_id = f"combat:{next_combat_identity}"
            next_combat_identity += 1
        act_boundary_efficiency: dict[int, tuple[int, float]] = {}
        act_segment_health: dict[int, ActSegmentHealth] = {}

        for step_offset in range(episode_limit):
            if state.terminated or state.truncated:
                break
            if not state.legal_actions:
                raise CollectionProtocolError(
                    f"episode={state.episode_id!r} step={state.step_index} returned zero legal actions"
                )
            pre_action_act, pre_action_floor = _run_position(state.observation)
            pre_action_combat = _combat_in_progress(state.observation)
            if episodic_enabled and pre_action_combat != (active_combat_id is not None):
                raise RuntimeError("episodic combat identity drifted from the factual state")
            step_combat_id = active_combat_id
            revivals_before_step = revivals_used
            hp_loss_before_step = player_hp_lost
            failure_credit_pre_recurrent_state = (
                recurrent_state[0].detach().float().cpu().numpy().copy()
                if failure_credit_pipeline is not None
                else None
            )
            choice = self._choose_action(
                state,
                recurrent_state,
                epsilon=epsilon,
                deterministic=deterministic,
            )
            timings.add("observation_encoding", choice.encoding_ms)
            timings.add("policy_forward", choice.policy_forward_ms)
            maximum_observed_candidates = max(
                maximum_observed_candidates,
                len(state.legal_actions),
            )
            maximum_observed_semantic_candidates = max(
                maximum_observed_semantic_candidates,
                choice.snapshot.candidate_count,
            )
            maximum_equivalence_class_size = max(
                maximum_equivalence_class_size,
                *(reference.multiplicity for reference in choice.action_references),
            )
            maximum_definition_hash_collisions_per_decision = max(
                maximum_definition_hash_collisions_per_decision,
                choice.definition_hash_collisions,
            )
            maximum_relation_hash_collisions_per_decision = max(
                maximum_relation_hash_collisions_per_decision,
                choice.relation_hash_collisions,
            )
            definition_hash_collisions_total += choice.definition_hash_collisions
            relation_hash_collisions_total += choice.relation_hash_collisions
            targeted_selection_exploration_decisions += int(choice.targeted_selection_exploration)
            targeted_transaction_entry_exploration_decisions += int(choice.targeted_transaction_entry_exploration)
            transaction_completion_guidance_decisions += int(choice.transaction_completion_guidance)
            transaction_completion_forward_decisions += int(choice.transaction_completion_forward_selected)
            transaction_completion_guidance_fallbacks += int(choice.transaction_completion_guidance_fallback)
            maximum_effective_collection_epsilon = max(
                maximum_effective_collection_epsilon,
                choice.effective_epsilon,
            )
            valid_policy_indices = np.flatnonzero(choice.snapshot.action_mask)
            if valid_policy_indices.size > 1:
                ranked_probabilities = np.sort(choice.policy[valid_policy_indices].astype(np.float64, copy=False))
                logit_margin = float(
                    math.log(max(float(ranked_probabilities[-1]), 1e-30))
                    - math.log(max(float(ranked_probabilities[-2]), 1e-30))
                )
                policy_logit_margin_total += logit_margin
                policy_logit_margin_count += 1
                policy_top1_top2_logit_margin_max = max(
                    policy_top1_top2_logit_margin_max,
                    logit_margin,
                )
            policy_decisions += int(choice.valid_count > 1)
            forced_decisions += int(choice.valid_count == 1)
            selected_action = state.legal_actions[choice.dispatch_position]
            transaction_action_fingerprint = _transaction_action_fingerprint(
                selected_action,
                equivalence_fingerprint=choice.equivalence_fingerprint,
            )
            last_selected_action_kind = str(
                selected_action.get(
                    "model_action_kind",
                    selected_action.get("kind", "unknown"),
                )
                or "unknown"
            )
            selected_action_kind_counts[last_selected_action_kind] = (
                selected_action_kind_counts.get(last_selected_action_kind, 0) + 1
            )
            current_observed_transaction_surface = _transaction_surface_key(
                state.observation,
                choice.semantic_actions,
            )
            current_transaction_surface = current_observed_transaction_surface if transaction_enabled else None
            if current_observed_transaction_surface is not None and observed_transaction_surface is None:
                observed_transaction_surface = current_observed_transaction_surface
                observed_transaction_entry_episodic_index = pending_observed_transaction_entry_episodic_index
                observed_transaction_opened_from_rest_site = (
                    pending_observed_transaction_opened_from_rest_site
                    or _is_rest_site_decision_surface(
                        state.observation,
                        choice.semantic_actions,
                    )
                )
                observed_transaction_operation = (
                    _selection_transaction_operation(
                        state.observation,
                        choice.semantic_actions,
                    )
                    or pending_observed_transaction_entry_operation
                )
                observed_transaction_is_forge = bool(
                    observed_transaction_opened_from_rest_site and observed_transaction_operation == "upgrade"
                )
                observed_transaction_is_shop_card_removal = bool(
                    pending_observed_transaction_is_shop_card_removal and observed_transaction_operation == "remove"
                )
                observed_transaction_opening_deck_signature = (
                    _deck_upgrade_signature(state.observation)
                    if observed_transaction_operation in {"upgrade", "remove"}
                    else None
                )
                observed_transaction_entry_behavior_log_probability = (
                    pending_transaction_entry_behavior_log_probability
                )
                observed_transaction_entry_model_probability = (
                    pending_transaction_entry_model_probability
                )
                observed_transaction_entry_action_fingerprint = (
                    pending_transaction_entry_action_fingerprint
                )
                observed_transaction_entry_policy_version = (
                    pending_transaction_entry_policy_version
                )
                observed_transaction_has_policy_choice = choice.valid_count > 1
                pending_observed_transaction_entry_episodic_index = None
                pending_observed_transaction_opened_from_rest_site = False
                pending_observed_transaction_entry_operation = ""
                pending_observed_transaction_is_shop_card_removal = False
                pending_transaction_entry_behavior_log_probability = 0.0
                pending_transaction_entry_model_probability = 0.0
                pending_transaction_entry_action_fingerprint = ""
                pending_transaction_entry_policy_version = policy_version
                selection_transactions_started += 1
                rest_site_selection_transactions_started += int(observed_transaction_opened_from_rest_site)
                forge_selection_transactions_started += int(observed_transaction_is_forge)
                shop_card_removal_transactions_started += int(observed_transaction_is_shop_card_removal)
            if (
                current_observed_transaction_surface is not None
                and current_observed_transaction_surface == observed_transaction_surface
            ):
                observed_transaction_has_policy_choice = bool(
                    observed_transaction_has_policy_choice or choice.valid_count > 1
                )
            current_transaction_node = (
                _transaction_node_key(
                    current_transaction_surface,
                    state.observation,
                    choice.semantic_actions,
                )
                if transaction_enabled
                else ""
            )
            current_transaction_policy_node = _noncombat_liveness_policy_node_key(
                state.observation,
                choice.semantic_actions,
            )
            if current_transaction_surface is not None:
                if active_transaction_surface is None:
                    burn_in_context = self.transaction_burn_in_steps or 0
                    lifecycle_candidate = bool(
                        observed_transaction_operation in {"upgrade", "remove"}
                        and observed_transaction_entry_action_fingerprint
                    )
                    context_count = burn_in_context + int(lifecycle_candidate)
                    context = (
                        tuple(transaction_context)[-context_count:]
                        if context_count
                        else ()
                    )
                    active_transaction_surface = current_transaction_surface
                    active_transaction_steps = [item[1] for item in context]
                    active_transaction_start_step = context[0][0] if context else step_offset
                    active_transaction_burn_in = (
                        min(burn_in_context, max(0, len(context) - 1))
                        if lifecycle_candidate
                        else len(context)
                    )
                    active_transaction_entry_step_index = (
                        active_transaction_burn_in
                        if lifecycle_candidate
                        else None
                    )
                    active_transaction_entry_behavior_log_probability = (
                        observed_transaction_entry_behavior_log_probability
                    )
                    active_transaction_entry_model_probability = (
                        observed_transaction_entry_model_probability
                    )
                    active_transaction_entry_action_fingerprint = (
                        observed_transaction_entry_action_fingerprint
                    )
                    active_transaction_operation = observed_transaction_operation
                    active_transaction_policy_version = (
                        observed_transaction_entry_policy_version
                        if lifecycle_candidate
                        else segment_policy_version
                    )
                    if lifecycle_candidate:
                        if not context:
                            raise RuntimeError(
                                "transaction lifecycle lost its factual entry context"
                            )
                        entry_step_index = active_transaction_entry_step_index
                        if entry_step_index is None:  # pragma: no cover - construction invariant
                            raise RuntimeError(
                                "transaction lifecycle lost its entry step index"
                            )
                        entry_step = active_transaction_steps[entry_step_index]
                        if (
                            entry_step.action_fingerprint
                            != active_transaction_entry_action_fingerprint
                        ):
                            raise RuntimeError(
                                "transaction lifecycle entry fingerprint differs from context"
                            )
                    active_transaction_seen_nodes = {current_transaction_node}
                elif active_transaction_surface != current_transaction_surface:
                    raise RuntimeError("transaction surface changed without an observed exit transition")
            deadlock_evidence = self.deadlock_detector.observe(
                step_index=state.step_index,
                observation=_deadlock_semantic_observation(
                    state.observation,
                    choice.semantic_actions,
                ),
                legal_actions=choice.semantic_actions,
                # The detector must follow the semantic action sampled by the
                # policy, not the arbitrary physical representative used for
                # backend dispatch.  In particular, a selection grid may
                # reorder equal card instances after every toggle; hashing the
                # raw representative would reintroduce card_index/instance
                # churn and hide an otherwise exact select/deselect cycle.
                selected_action=choice.semantic_actions[choice.candidate_index],
            )
            sim_step_started_ns = time.perf_counter_ns()
            next_state, _ = self._step(
                state,
                dispatch_position=choice.dispatch_position,
                dispatch_handle=choice.dispatch_handle,
            )
            timings.record("sim_step", sim_step_started_ns)
            steps_taken += 1
            if next_state.truncated:
                raise CollectionProtocolError("transport/outcome-unknown truncation discarded before rollout")
            next_action_groups = self.encoder.semantic_action_groups(next_state.legal_actions)
            next_semantic_actions = _semantic_action_surface(next_action_groups)
            next_observed_transaction_surface = _transaction_surface_key(
                next_state.observation,
                next_semantic_actions,
            )
            next_transaction_surface = next_observed_transaction_surface if transaction_enabled else None
            opens_selection_transaction = bool(
                current_observed_transaction_surface is None and next_observed_transaction_surface is not None
            )
            result_terminal = next_state.terminated or next_state.truncated
            forced_horizon = step_offset + 1 >= episode_limit and not next_state.terminated and not next_state.truncated
            deadlock_evidence = self.deadlock_detector.confirm_after_step(
                deadlock_evidence,
                observation=_deadlock_semantic_observation(
                    next_state.observation,
                    next_semantic_actions,
                ),
                legal_actions=next_semantic_actions,
            )
            if failure_credit_pipeline is not None:
                if failure_credit_pre_recurrent_state is None:  # pragma: no cover - construction invariant
                    raise RuntimeError("failure-credit transition lost its pre-step recurrent state")
                failure_credit_started_ns = time.perf_counter_ns()
                failure_credit_pipeline.observe_transition(
                    episode_step=state.step_index,
                    before_observation=state.observation,
                    before_legal_actions=state.legal_actions,
                    after_observation=next_state.observation,
                    after_legal_actions=next_state.legal_actions,
                    snapshot=choice.snapshot,
                    action_index=choice.candidate_index,
                    behavior_log_probability=choice.behavior_log_probability,
                    policy_version=segment_policy_version,
                    pre_recurrent_state=failure_credit_pre_recurrent_state,
                    terminal=result_terminal,
                )
                if failure_credit_sink is not None:
                    ready_failure_credit = failure_credit_pipeline.drain_ready_records()
                    if ready_failure_credit:
                        failure_credit_sink(ready_failure_credit)
                timings.record(
                    "failure_credit_shadow",
                    failure_credit_started_ns,
                )
            if next_state.transition is None:  # pragma: no cover - validated above
                raise CollectionProtocolError("step result lost its typed transition")
            # The validated transition may itself expose a much wider next
            # decision than the pre-action state.  Record that accepted state
            # now, before a later mutation from it has a chance to fault.
            maximum_observed_candidates = max(
                maximum_observed_candidates,
                len(next_state.legal_actions),
            )
            maximum_observed_semantic_candidates = max(
                maximum_observed_semantic_candidates,
                len(next_action_groups),
            )
            if next_action_groups:
                maximum_equivalence_class_size = max(
                    maximum_equivalence_class_size,
                    max(group.multiplicity for group in next_action_groups),
                )
            # Publish only fully validated environment steps.  The asynchronous
            # supervisor uses this monotonic count to distinguish valid but
            # unflushed tail steps from recurrent unrolls already emitted to
            # the learner when an infrastructure incident aborts an episode.
            if accepted_step_sink is not None:
                accepted_step_sink(steps_taken, maximum_observed_candidates)
            next_combat_in_progress = _combat_in_progress(next_state.observation)
            combat_window_selection = _combat_progress_window_selection(
                next_state.observation,
                default_window=self.combat_net_progress_window,
                room_windows=self.combat_net_progress_room_windows,
                encounter_windows=self.combat_net_progress_encounter_windows,
            )
            combat_progress.window = combat_window_selection.effective_window
            combat_progress_status = combat_progress.observe(
                step=steps_taken,
                observation=next_state.observation,
            )
            combat_no_net_progress_steps = combat_progress_status.age_steps
            maximum_combat_no_net_progress_steps = max(
                maximum_combat_no_net_progress_steps,
                combat_progress_status.maximum_age_steps,
            )
            runaway_combat_status = runaway_combat_guard.observe(
                observation=next_state.observation,
                legal_actions=next_state.legal_actions,
                no_net_progress_steps=combat_progress_status.age_steps,
            )
            runaway_combat_stalled = bool(
                runaway_combat_status.triggered and not result_terminal and not forced_horizon
            )
            semantic_liveness_eligible = bool(not result_terminal and not forced_horizon)
            combat_progress_stalled = bool(
                (combat_progress_status.stalled or runaway_combat_stalled) and semantic_liveness_eligible
            )
            if runaway_combat_stalled:
                # The accepted transition is the factual terminal learning
                # prefix. Do not dispatch another forced end-turn into a
                # simulator state whose Status-card work is already runaway.
                stall_evidence = runaway_combat_status.evidence(
                    detected_step=next_state.step_index,
                )
                stall_evidence.update(
                    {
                        "window": combat_window_selection.effective_window,
                        "default_window": combat_window_selection.default_window,
                        "window_source": combat_window_selection.source,
                        "window_match_id": combat_window_selection.match_id or None,
                        "room_model_id": combat_window_selection.room_model_id or None,
                        "encounter_id": combat_window_selection.encounter_id or None,
                        "anchor_enemy_hp_total": combat_progress_status.anchor_hp,
                        "current_enemy_hp_total": combat_progress_status.current_hp,
                        "current_enemy_max_hp_total": combat_progress_status.maximum_hp,
                        "net_enemy_hp_progress": combat_progress_status.net_hp_progress,
                        "required_net_enemy_hp_progress": (combat_progress_status.required_hp_progress),
                        "progress_kind": combat_progress_status.progress_kind,
                        "legal_action_kinds": _legal_action_kind_counts(next_state.legal_actions),
                    }
                )
            elif combat_progress_stalled:
                # Construct this bounded snapshot only for the terminating
                # transition. Ordinary decisions retain no additional state.
                stall_evidence = _combat_stall_evidence(
                    status=combat_progress_status,
                    observation=next_state.observation,
                    legal_actions=next_state.legal_actions,
                    window_selection=combat_window_selection,
                    detected_step=next_state.step_index,
                )
            combat_in_progress = bool(next_combat_in_progress and not result_terminal)
            noncombat_progress_status = noncombat_progress.observe(
                step=steps_taken,
                observation=next_state.observation,
                selected_action=selected_action,
                terminated=next_state.terminated,
                truncated=next_state.truncated,
            )
            noncombat_no_durable_progress_steps = noncombat_progress_status.age_steps
            maximum_noncombat_no_durable_progress_steps = max(
                maximum_noncombat_no_durable_progress_steps,
                noncombat_progress_status.maximum_age_steps,
            )
            noncombat_durable_stalled = bool(noncombat_progress_status.stalled and semantic_liveness_eligible)
            event_cycle_evidence = noncombat_event_cycles.observe(
                step=steps_taken,
                before_observation=state.observation,
                after_observation=next_state.observation,
                action_fingerprint=transaction_action_fingerprint,
                policy_node_key=current_transaction_policy_node,
                durable_progress_kind=noncombat_progress_status.progress_kind,
                terminated=next_state.terminated,
                truncated=next_state.truncated,
            )
            noncombat_event_cycle = bool(event_cycle_evidence is not None and semantic_liveness_eligible)
            noncombat_progress_stalled = bool(noncombat_durable_stalled or noncombat_event_cycle)
            # ``maximum_steps`` may be a runtime's remaining global budget,
            # which can cut an otherwise healthy episode after only one or a
            # few decisions. Only the configured task horizon is a semantic
            # failure. A shorter collection-budget cut keeps a positive
            # discount and a bootstrap snapshot instead of fabricating a loss.
            curriculum_horizon = bool(forced_horizon and episode_limit >= self.max_episode_steps)
            reward_started_ns = time.perf_counter_ns()
            # EnvironmentResult is the terminal authority. The pre-action
            # recurrence has now also been confirmed against its factual
            # successor, so a novel semantic exit is not a synthetic deadlock.
            effective_deadlock_evidence = deadlock_evidence if semantic_liveness_eligible else None
            exact_cycle_has_policy_choice = bool(
                choice.valid_count > 1
                or (
                    current_transaction_surface is not None
                    and any(
                        np.count_nonzero(step.snapshot.action_mask) > 1
                        for step in active_transaction_steps[active_transaction_burn_in:]
                    )
                )
            )
            selection_action_cycle = bool(
                effective_deadlock_evidence is not None
                and current_observed_transaction_surface is not None
                and (observed_transaction_has_policy_choice or exact_cycle_has_policy_choice)
                and not forced_horizon
            )
            # Combat no-progress, a repeated factual event transition and a
            # confirmed exact semantic cycle with a factual choice are policy
            # outcomes.  Native
            # revival may keep the avatar alive, but cannot turn these into a
            # censored infrastructure timeout.  A generic unique-moving
            # noncombat durable-window stop remains censored because it cannot
            # factually name a responsible action.
            trusted_policy_failure = bool(
                combat_progress_stalled
                or noncombat_event_cycle
                or (effective_deadlock_evidence is not None and exact_cycle_has_policy_choice and not forced_horizon)
            )
            if noncombat_event_cycle:
                if event_cycle_evidence is None:  # pragma: no cover - invariant
                    raise RuntimeError("event cycle lost its terminal evidence")
                stall_evidence = event_cycle_evidence.to_mapping()
            elif selection_action_cycle and effective_deadlock_evidence is not None:
                stall_evidence = dict(effective_deadlock_evidence.to_mapping())
            elif (
                effective_deadlock_evidence is not None
                and not combat_progress_stalled
                and not noncombat_durable_stalled
            ):
                stall_evidence = dict(effective_deadlock_evidence.to_mapping())
            breakdown = self.reward_calculator.evaluate(
                state,
                next_state,
                deadlock=(
                    not result_terminal
                    and (
                        effective_deadlock_evidence is not None or combat_progress_stalled or noncombat_progress_stalled
                    )
                ),
                horizon_exhausted=bool(curriculum_horizon and self.horizon_as_failure),
            )
            reward_total += breakdown.reward
            shaping_reward_total += float(breakdown.reward - breakdown.terminal_reward - breakdown.progress_reward)
            revivals_used += breakdown.revivals_used_delta
            player_hp_lost += breakdown.player_hp_lost_delta
            final_outcome = breakdown.outcome
            deadlocked = breakdown.outcome == "deadlock"
            if opens_selection_transaction:
                pending_observed_transaction_opened_from_rest_site = _is_rest_site_decision_surface(
                    state.observation,
                    choice.semantic_actions,
                )
                pending_observed_transaction_entry_operation = _transaction_entry_operation(selected_action)
                pending_observed_transaction_is_shop_card_removal = _is_shop_card_removal_entry(selected_action)
                pending_transaction_entry_behavior_log_probability = (
                    choice.behavior_log_probability
                )
                pending_transaction_entry_model_probability = float(
                    choice.policy[choice.candidate_index]
                )
                pending_transaction_entry_action_fingerprint = (
                    transaction_action_fingerprint
                )
                pending_transaction_entry_policy_version = segment_policy_version
            selection_exit_outcome: Literal["committed", "cancelled", "unresolved"] | None = None
            if observed_transaction_surface is not None and (
                next_observed_transaction_surface != observed_transaction_surface
                or result_terminal
                or breakdown.task_terminal
                or forced_horizon
            ):
                selection_transaction_closed = bool(next_observed_transaction_surface != observed_transaction_surface)
                clean_transaction_exit = bool(
                    next_observed_transaction_surface is None
                    and (
                        (not result_terminal and not breakdown.task_terminal and not forced_horizon)
                        or (result_terminal and breakdown.outcome == "success")
                    )
                )
                selection_exit_outcome = _selection_transaction_exit_outcome(
                    selected_action=selected_action,
                    clean_exit=clean_transaction_exit,
                    operation=observed_transaction_operation,
                    opening_deck_signature=(observed_transaction_opening_deck_signature),
                    after_observation=next_state.observation,
                )
                selection_transactions_closed += int(selection_transaction_closed)
                rest_site_selection_transactions_closed += int(
                    selection_transaction_closed and observed_transaction_opened_from_rest_site
                )
                forge_selection_transactions_closed += int(
                    selection_transaction_closed and observed_transaction_is_forge
                )
                shop_card_removal_transactions_closed += int(
                    selection_transaction_closed and observed_transaction_is_shop_card_removal
                )
                if selection_exit_outcome == "committed":
                    selection_transactions_completed += 1
                    rest_site_selection_transactions_completed += int(observed_transaction_opened_from_rest_site)
                    forge_selection_transactions_completed += int(observed_transaction_is_forge)
                    shop_card_removal_transactions_completed += int(observed_transaction_is_shop_card_removal)
                elif selection_exit_outcome == "cancelled":
                    selection_transactions_cancelled += 1
                    rest_site_selection_transactions_cancelled += int(observed_transaction_opened_from_rest_site)
                    forge_selection_transactions_cancelled += int(observed_transaction_is_forge)
                    shop_card_removal_transactions_cancelled += int(observed_transaction_is_shop_card_removal)
                else:
                    selection_transactions_unresolved += 1
                    rest_site_selection_transactions_unresolved += int(observed_transaction_opened_from_rest_site)
                    forge_selection_transactions_unresolved += int(observed_transaction_is_forge)
                    shop_card_removal_transactions_unresolved += int(observed_transaction_is_shop_card_removal)
                if (
                    clean_transaction_exit
                    and selection_exit_outcome in {"cancelled", "unresolved"}
                    and observed_transaction_entry_episodic_index is not None
                ):
                    entry_index = observed_transaction_entry_episodic_index
                    if not 0 <= entry_index < len(episodic_steps):
                        raise RuntimeError("selection transaction entrance episodic index escaped its episode")
                    # An aborted transaction is policy-neutral in the complete-
                    # episode actor objective.  Keep every task/value target,
                    # but do not let a later run victory behavior-clone the
                    # Smith -> Select -> Cancel rollback path (or let a later
                    # defeat blame merely entering a recoverable transaction).
                    for episodic_index in range(entry_index, len(episodic_steps)):
                        episodic_steps[episodic_index] = replace(
                            episodic_steps[episodic_index],
                            policy_decision=False,
                        )
                if (
                    breakdown.outcome == "deadlock"
                    and trusted_policy_failure
                    and observed_transaction_entry_episodic_index is not None
                ):
                    entry_index = observed_transaction_entry_episodic_index
                    if not 0 <= entry_index < len(episodic_steps):
                        raise RuntimeError("selection transaction entrance episodic index escaped its episode")
                    # A delayed selection-cycle terminal is not a factual
                    # one-step consequence of the action that opened the
                    # surface.  Preserve its value target but do not turn
                    # trying the multi-step transaction into an anti-policy
                    # label.  This state machine is deliberately independent
                    # of the retired transaction replay plane.
                    episodic_steps[entry_index] = replace(
                        episodic_steps[entry_index],
                        policy_decision=False,
                    )
                observed_transaction_surface = None
                observed_transaction_entry_episodic_index = None
                observed_transaction_opened_from_rest_site = False
                observed_transaction_is_forge = False
                observed_transaction_operation = ""
                observed_transaction_is_shop_card_removal = False
                observed_transaction_has_policy_choice = False
                observed_transaction_opening_deck_signature = None
                observed_transaction_entry_behavior_log_probability = 0.0
                observed_transaction_entry_model_probability = 0.0
                observed_transaction_entry_action_fingerprint = ""
                observed_transaction_entry_policy_version = policy_version
            after_action_act, _ = _run_position(next_state.observation)
            transition_facts = next_state.transition.facts
            typed_run_result = transition_facts.get("run_result")
            authoritative_run_result = (
                str(typed_run_result)
                if (self.scenario == "full-run" and next_state.terminated and typed_run_result in {"victory", "defeat"})
                else None
            )
            episode_censored = bool(
                not trusted_policy_failure
                and (forced_horizon or (breakdown.task_terminal and authoritative_run_result is None))
            )
            combat_boundary = _episodic_combat_boundary(
                was_active=pre_action_combat,
                is_active=next_combat_in_progress,
                transition_facts=transition_facts,
                authoritative_run_result=authoritative_run_result,
                trusted_policy_failure=trusted_policy_failure,
                episode_censored=episode_censored,
            )
            act_boundary = _episodic_act_boundary(
                before_act=pre_action_act,
                after_act=after_action_act,
                authoritative_run_result=authoritative_run_result,
                trusted_policy_failure=trusted_policy_failure,
                episode_censored=episode_censored,
            )
            if act_boundary is BoundaryOutcome.SUCCEEDED:
                completed_acts: tuple[int, ...]
                if after_action_act > pre_action_act:
                    completed_acts = tuple(range(pre_action_act, after_action_act))
                else:
                    # A typed run victory closes the final Act without
                    # requiring a synthetic act+1 terminal observation.
                    completed_acts = (pre_action_act,)
                for completed_act in completed_acts:
                    if completed_act >= 1:
                        act_boundary_efficiency.setdefault(
                            completed_act,
                            (revivals_used, player_hp_lost),
                        )
                        if completed_act == pre_action_act:
                            configured_budget = self.training_revival_budget
                            effective_budget = (
                                configured_budget
                                if configured_budget is not None
                                and configured_budget > 0
                                else 64
                            )
                            act_segment_health.setdefault(
                                completed_act,
                                ActSegmentHealth(
                                    act=completed_act,
                                    boundary_step_index=len(episodic_steps),
                                    exit_hp_ratio=_act_exit_hp_ratio(
                                        before_observation=state.observation,
                                        after_observation=next_state.observation,
                                        authoritative_run_result=authoritative_run_result,
                                    ),
                                    cumulative_revivals=revivals_used,
                                    revival_budget=effective_budget,
                                ),
                            )
            if (
                combat_boundary is BoundaryOutcome.SUCCEEDED
                and pre_action_act >= 1
                and "boss" in _room_type(state.observation)
            ):
                boss_victory_acts.add(pre_action_act)
            # Synthetic liveness terminals summarize a delayed window, not a
            # one-step causal consequence of the action that happened to cross
            # its threshold. Keep the terminal reward/value target and the
            # bounded transaction trace, but do not assign that -1 directly to
            # the final selected action through FIFO or episodic policy loss.
            one_step_policy_decision = bool(
                choice.valid_count > 1
                and breakdown.outcome != "deadlock"
                and not opens_selection_transaction
                and selection_exit_outcome not in {"cancelled", "unresolved"}
            )
            episodic_policy_decision = bool(
                choice.valid_count > 1
                and breakdown.outcome != "deadlock"
                and selection_exit_outcome not in {"cancelled", "unresolved"}
            )
            if episodic_enabled:
                episodic_step_index = len(episodic_steps)
                episodic_steps.append(
                    EpisodeDecisionStep(
                        snapshot=choice.snapshot,
                        step_index=len(episodic_steps),
                        action_index=choice.candidate_index,
                        behavior_log_probability=choice.behavior_log_probability,
                        policy_decision=episodic_policy_decision,
                        policy_version=segment_policy_version,
                        act=pre_action_act,
                        floor=pre_action_floor,
                        combat_id=step_combat_id,
                        # Long-horizon primary credit excludes every
                        # efficiency/pace shaping term by construction.
                        task_reward=float(breakdown.terminal_reward + breakdown.progress_reward),
                        discount=breakdown.discount,
                        revivals_before=revivals_before_step,
                        revivals_after=revivals_used,
                        hp_loss_before=hp_loss_before_step,
                        hp_loss_after=player_hp_lost,
                        combat_boundary=combat_boundary,
                        act_boundary=act_boundary,
                        decision_surface=_episodic_decision_surface(
                            state.observation,
                            combat_in_progress=pre_action_combat,
                        ),
                    )
                )
                episodic_diagnostic_rewards.append(float(breakdown.reward))
                episodic_diagnostic_discounts.append(float(breakdown.discount))
                if episodic_policy_decision and not pre_action_combat:
                    episodic_macro_decisions.append(
                        (
                            episodic_step_index,
                            episodic_steps[-1].decision_surface,
                            _diagnostic_action_key(
                                choice.semantic_actions[choice.candidate_index]
                            ),
                            _hp_band(state.observation),
                            pre_action_act,
                        )
                    )
                if opens_selection_transaction:
                    pending_observed_transaction_entry_episodic_index = episodic_step_index
                if pre_action_combat and not next_combat_in_progress:
                    active_combat_id = None
                elif not pre_action_combat and next_combat_in_progress:
                    active_combat_id = f"combat:{next_combat_identity}"
                    next_combat_identity += 1
            if record:
                segment_steps.append(
                    RolloutStep(
                        snapshot=choice.snapshot,
                        action_index=choice.candidate_index,
                        behavior_log_probability=choice.behavior_log_probability,
                        reward=breakdown.reward,
                        discount=breakdown.discount,
                        policy_decision=one_step_policy_decision,
                    )
                )
            if transaction_enabled:
                episode_rewards.append(float(breakdown.reward))
                episode_discounts.append(float(breakdown.discount))
                next_transaction_node = _transaction_node_key(
                    next_transaction_surface,
                    next_state.observation,
                    next_semantic_actions,
                )
                effect, selected_count_delta = _classify_transaction_transition(
                    current_surface=current_transaction_surface,
                    next_surface=next_transaction_surface,
                    current_node=current_transaction_node,
                    next_node=next_transaction_node,
                    seen_nodes=active_transaction_seen_nodes,
                    before_selected_count=_transaction_selected_count(
                        state.observation,
                        choice.semantic_actions,
                    ),
                    after_selected_count=_transaction_selected_count(
                        next_state.observation,
                        next_semantic_actions,
                    ),
                )
                factual_transaction_step = TransactionStep(
                    snapshot=choice.snapshot,
                    action_index=choice.candidate_index,
                    node_key=current_transaction_node,
                    next_node_key=next_transaction_node,
                    action_fingerprint=transaction_action_fingerprint,
                    effect=effect,
                    selected_count_delta=selected_count_delta,
                    transaction_return=None,
                    return_steps=None,
                    behavior_log_probability=choice.behavior_log_probability,
                    model_log_probability=choice.model_log_probability,
                    model_probability=float(
                        choice.policy[choice.candidate_index]
                    ),
                    policy_node_key=current_transaction_policy_node,
                    policy_action_fingerprint=(
                        transaction_action_fingerprint if current_transaction_policy_node is not None else None
                    ),
                )
                # Rest is a one-step resource transaction rather than a card
                # selection surface.  Give a *factual successful heal* the
                # same next-rest/Act option horizon as forge so the learner can
                # compare rest and forge symmetrically from their shared legal
                # candidate state.  Merely clicking a rest-labelled option is
                # insufficient: HP must actually increase in the authoritative
                # post-state receipt.
                selected_transaction_operation = _transaction_entry_operation(
                    selected_action
                )
                if (
                    selected_transaction_operation == "rest"
                    and current_transaction_surface is None
                    and _is_rest_site_decision_surface(
                        state.observation,
                        choice.semantic_actions,
                    )
                    and _player_hp(next_state.observation)
                    > _player_hp(state.observation)
                ):
                    rest_post_terminal = bool(
                        result_terminal
                        or breakdown.task_terminal
                        or breakdown.discount == 0.0
                    )
                    rest_post_snapshot: EncodedDecisionSnapshot | None = None
                    if not rest_post_terminal:
                        if not next_state.legal_actions:
                            raise RuntimeError(
                                "successful rest lifecycle has no post-state legal actions"
                            )
                        rest_encoding_started_ns = time.perf_counter_ns()
                        rest_post_snapshot = self.encoder.encode(
                            next_state.observation,
                            next_state.legal_actions,
                            device=self.device,
                        ).snapshot
                        timings.record(
                            "transaction_post_encoding",
                            rest_encoding_started_ns,
                        )
                    rest_burn_in = self.transaction_burn_in_steps or 0
                    rest_context = (
                        tuple(transaction_context)[-rest_burn_in:]
                        if rest_burn_in
                        else ()
                    )
                    rest_steps = (
                        *(item[1] for item in rest_context),
                        factual_transaction_step,
                    )
                    rest_entry_index = len(rest_context)
                    rest_start_step = (
                        rest_context[0][0]
                        if rest_context
                        else step_offset
                    )
                    transaction_traces_pending.append(
                        TransactionTrace(
                            trace_id=(
                                f"seed-{reset_seed}:{state.episode_id}:"
                                f"{rest_start_step}:{step_offset}:rest-resource"
                            ),
                            episode_id=f"seed-{reset_seed}:{state.episode_id}",
                            surface_key=semantic_fingerprint(
                                {
                                    "kind": "single_step_resource_transaction",
                                    "operation": "rest",
                                    "entry_node": current_transaction_node,
                                }
                            ),
                            start_step=rest_start_step,
                            policy_version=segment_policy_version,
                            initial_recurrent_state=np.zeros(
                                self.model.config.recurrent_hidden_dim,
                                dtype=np.float32,
                            ),
                            steps=rest_steps,
                            burn_in_steps=rest_entry_index,
                            outcome=TransactionOutcome.COMPLETED,
                            lifecycle=TransactionLifecycleEvidence(
                                operation="rest",
                                entry_step_index=rest_entry_index,
                                exit_step_index=rest_entry_index,
                                entry_behavior_log_probability=(
                                    choice.behavior_log_probability
                                ),
                                entry_model_probability=float(
                                    choice.policy[choice.candidate_index]
                                ),
                                entry_policy_version=segment_policy_version,
                                entry_action_fingerprint=(
                                    transaction_action_fingerprint
                                ),
                                outcome=(
                                    TransactionLifecycleOutcome.COMMITTED
                                ),
                                effect_verified=True,
                                post_snapshot=rest_post_snapshot,
                                post_terminal=rest_post_terminal,
                            ),
                        )
                    )
                if choice.valid_count > 1:
                    prefix_burn_in = self.transaction_burn_in_steps or 0
                    prefix_context = tuple(transaction_context)[-prefix_burn_in:] if prefix_burn_in else ()
                    last_liveness_policy_prefix = (
                        prefix_context[0][0] if prefix_context else step_offset,
                        (
                            *(item[1] for item in prefix_context),
                            factual_transaction_step,
                        ),
                        len(prefix_context),
                    )
                trusted_deadlock_trace_emitted = False
                if current_transaction_surface is not None:
                    active_transaction_steps.append(factual_transaction_step)
                    maximum_transaction_steps = (
                        (self.transaction_burn_in_steps or 0)
                        + liveness_learn_tail_steps
                        + int(active_transaction_entry_step_index is not None)
                    )
                    if len(active_transaction_steps) > maximum_transaction_steps:
                        overflow = len(active_transaction_steps) - maximum_transaction_steps
                        del active_transaction_steps[:overflow]
                        active_transaction_start_step += overflow
                        if active_transaction_entry_step_index is not None:
                            # A pathological unresolved transaction outlived
                            # the exact entry-context window. It remains valid
                            # liveness/value evidence, but can no longer claim
                            # exact entry support or SMDP credit.
                            active_transaction_entry_step_index = None
                        active_transaction_burn_in = min(
                            self.transaction_burn_in_steps or 0,
                            max(0, len(active_transaction_steps) - 1),
                        )
                    active_transaction_seen_nodes.add(next_transaction_node)
                    transaction_ended = bool(
                        next_transaction_surface != current_transaction_surface
                        or result_terminal
                        or breakdown.task_terminal
                        or forced_horizon
                    )
                    if transaction_ended:
                        if active_transaction_surface is None:  # pragma: no cover - start invariant
                            raise RuntimeError("active transaction lost its surface")
                        if breakdown.outcome == "deadlock" and trusted_policy_failure:
                            # A same-surface recurrence is itself the grounded
                            # failed transaction.  If the final action locally
                            # exits the selection surface but closes a larger
                            # event/combat liveness cycle, censor this local
                            # completion and emit a separate global DEADLOCK
                            # trace below.  Otherwise local EXIT would create a
                            # PREFER label for the very action the run must
                            # learn to avoid.
                            active_transaction_outcome = (
                                TransactionOutcome.DEADLOCK
                                if next_transaction_surface == current_transaction_surface
                                else TransactionOutcome.CENSORED
                            )
                        elif next_transaction_surface != current_transaction_surface:
                            active_transaction_outcome = TransactionOutcome.COMPLETED
                        else:
                            active_transaction_outcome = TransactionOutcome.CENSORED
                        lifecycle: TransactionLifecycleEvidence | None = None
                        if active_transaction_entry_step_index is not None:
                            if breakdown.outcome == "deadlock" and trusted_policy_failure:
                                lifecycle_outcome = TransactionLifecycleOutcome.DEADLOCK
                            elif selection_exit_outcome == "committed":
                                lifecycle_outcome = TransactionLifecycleOutcome.COMMITTED
                            elif selection_exit_outcome == "cancelled":
                                lifecycle_outcome = TransactionLifecycleOutcome.CANCELLED
                            else:
                                lifecycle_outcome = TransactionLifecycleOutcome.UNRESOLVED
                            post_snapshot: EncodedDecisionSnapshot | None = None
                            post_terminal = bool(
                                lifecycle_outcome
                                is TransactionLifecycleOutcome.COMMITTED
                                and (
                                    result_terminal
                                    or breakdown.task_terminal
                                    or breakdown.discount == 0.0
                                )
                            )
                            if lifecycle_outcome is TransactionLifecycleOutcome.COMMITTED:
                                if not next_state.legal_actions and not post_terminal:
                                    raise RuntimeError(
                                        "committed transaction lifecycle has no post-state legal actions"
                                    )
                                if not post_terminal:
                                    lifecycle_encoding_started_ns = time.perf_counter_ns()
                                    post_snapshot = self.encoder.encode(
                                        next_state.observation,
                                        next_state.legal_actions,
                                        device=self.device,
                                    ).snapshot
                                    timings.record(
                                        "transaction_post_encoding",
                                        lifecycle_encoding_started_ns,
                                    )
                            lifecycle = TransactionLifecycleEvidence(
                                operation=active_transaction_operation,
                                entry_step_index=active_transaction_entry_step_index,
                                exit_step_index=len(active_transaction_steps) - 1,
                                entry_behavior_log_probability=(
                                    active_transaction_entry_behavior_log_probability
                                ),
                                entry_model_probability=(
                                    active_transaction_entry_model_probability
                                ),
                                entry_policy_version=(
                                    active_transaction_policy_version
                                ),
                                entry_action_fingerprint=(
                                    active_transaction_entry_action_fingerprint
                                ),
                                outcome=lifecycle_outcome,
                                effect_verified=(
                                    lifecycle_outcome
                                    is TransactionLifecycleOutcome.COMMITTED
                                ),
                                post_snapshot=post_snapshot,
                                post_terminal=post_terminal,
                            )
                        transaction_traces_pending.append(
                            TransactionTrace(
                                trace_id=(
                                    f"seed-{reset_seed}:{state.episode_id}:"
                                    f"{active_transaction_start_step}:"
                                    f"{step_offset}:{active_transaction_surface}"
                                ),
                                episode_id=f"seed-{reset_seed}:{state.episode_id}",
                                surface_key=active_transaction_surface,
                                start_step=active_transaction_start_step,
                                policy_version=active_transaction_policy_version,
                                initial_recurrent_state=np.zeros(
                                    self.model.config.recurrent_hidden_dim,
                                    dtype=np.float32,
                                ),
                                steps=tuple(active_transaction_steps),
                                burn_in_steps=active_transaction_burn_in,
                                outcome=active_transaction_outcome,
                                lifecycle=lifecycle,
                            )
                        )
                        trusted_deadlock_trace_emitted = active_transaction_outcome is TransactionOutcome.DEADLOCK
                        active_transaction_surface = None
                        active_transaction_steps = []
                        active_transaction_seen_nodes = set()
                        active_transaction_entry_step_index = None
                        active_transaction_entry_behavior_log_probability = 0.0
                        active_transaction_entry_model_probability = 0.0
                        active_transaction_entry_action_fingerprint = ""
                        active_transaction_operation = ""
                if breakdown.outcome == "deadlock" and (
                    current_transaction_surface is None
                    or (trusted_policy_failure and not trusted_deadlock_trace_emitted)
                ):
                    # Same-surface selection cycles are emitted by the active
                    # transaction above. Global liveness failures whose final
                    # action locally exits/reopens a transaction, plus failures
                    # outside a transaction, still need this factual trace.
                    # Retain a bounded prefix: configured recurrent burn-in is
                    # detached by the learner and the tail supplies labels.
                    maximum_liveness_steps = (self.transaction_burn_in_steps or 0) + liveness_learn_tail_steps
                    context = (
                        *tuple(transaction_context),
                        (step_offset, factual_transaction_step),
                    )[-maximum_liveness_steps:]
                    liveness_steps = tuple(item[1] for item in context)
                    liveness_burn_in = min(
                        self.transaction_burn_in_steps or 0,
                        max(0, len(liveness_steps) - 1),
                    )
                    liveness_start_step = context[0][0]
                    if not any(
                        np.count_nonzero(step.snapshot.action_mask) > 1 for step in liveness_steps[liveness_burn_in:]
                    ):
                        # The bounded tail may be entirely forced even though
                        # an earlier policy choice caused the failure. Prefer
                        # the most recent bounded decision prefix so factual
                        # liveness supervision can update a policy parameter.
                        if last_liveness_policy_prefix is not None:
                            (
                                liveness_start_step,
                                liveness_steps,
                                liveness_burn_in,
                            ) = last_liveness_policy_prefix
                    failure_kind = (
                        "combat_no_net_progress"
                        if combat_progress_stalled
                        else "noncombat_event_action_cycle"
                        if noncombat_event_cycle
                        else "selection_action_cycle"
                        if selection_action_cycle
                        else "noncombat_no_durable_progress"
                        if noncombat_durable_stalled
                        else "semantic_action_cycle"
                    )
                    failure_surface = semantic_fingerprint(
                        {
                            "kind": "training_liveness_failure",
                            "failure_kind": failure_kind,
                            "locus": {
                                "act": pre_action_act,
                                "floor": _run_position(state.observation)[1],
                                "decision_domain": state.observation.get("decision_domain"),
                            },
                        }
                    )
                    liveness_trace = TransactionTrace(
                        trace_id=(
                            f"seed-{reset_seed}:{state.episode_id}:"
                            f"{liveness_start_step}:"
                            f"{step_offset}:{failure_surface}"
                        ),
                        episode_id=f"seed-{reset_seed}:{state.episode_id}",
                        surface_key=failure_surface,
                        start_step=liveness_start_step,
                        policy_version=segment_policy_version,
                        initial_recurrent_state=np.zeros(
                            self.model.config.recurrent_hidden_dim,
                            dtype=np.float32,
                        ),
                        steps=liveness_steps,
                        burn_in_steps=liveness_burn_in,
                        outcome=(
                            TransactionOutcome.DEADLOCK if trusted_policy_failure else TransactionOutcome.CENSORED
                        ),
                    )
                    if trusted_policy_failure:
                        failed_policy_pairs = {
                            (
                                liveness_trace.steps[target.step_index].effective_policy_node_key,
                                liveness_trace.steps[target.step_index].effective_policy_action_fingerprint,
                            )
                            for target in factual_transaction_policy_targets(liveness_trace)
                        }
                        if failed_policy_pairs:
                            # Before the global loop became provable, its action
                            # may have locally completed/reopened one or more
                            # selection prompts. Censor only COMPLETED traces
                            # containing a now-grounded AVOID pair so the same
                            # actor action cannot receive contradictory PREFER
                            # and AVOID labels. Q/effect facts remain intact.
                            for trace_index, pending_trace in enumerate(transaction_traces_pending):
                                if pending_trace.outcome is not TransactionOutcome.COMPLETED:
                                    continue
                                if any(
                                    (
                                        step.effective_policy_node_key,
                                        step.effective_policy_action_fingerprint,
                                    )
                                    in failed_policy_pairs
                                    for step in pending_trace.learn_steps
                                ):
                                    transaction_traces_pending[trace_index] = replace(
                                        pending_trace,
                                        outcome=TransactionOutcome.CENSORED,
                                    )
                    transaction_traces_pending.append(liveness_trace)
                transaction_context.append((step_offset, factual_transaction_step))
            if trajectory_journal is not None:
                journal_deadlock: Mapping[str, object] | None = None
                if combat_progress_stalled:
                    if stall_evidence is None:  # pragma: no cover - construction invariant
                        raise RuntimeError("combat stall lost its terminal evidence")
                    journal_deadlock = stall_evidence
                elif noncombat_event_cycle:
                    if stall_evidence is None:  # pragma: no cover - construction invariant
                        raise RuntimeError("event cycle lost its terminal evidence")
                    journal_deadlock = stall_evidence
                elif selection_action_cycle:
                    if effective_deadlock_evidence is None:  # pragma: no cover - invariant
                        raise RuntimeError("selection cycle lost its terminal evidence")
                    journal_deadlock = effective_deadlock_evidence.to_mapping()
                elif noncombat_durable_stalled:
                    journal_deadlock = {
                        "kind": "noncombat_no_durable_progress",
                        "window": self.noncombat_durable_progress_window,
                        "steps_without_durable_progress": (noncombat_no_durable_progress_steps),
                        "durable_state_fingerprint": (noncombat_progress_status.durable_state_fingerprint),
                        "locus_fingerprint": (noncombat_progress_status.locus_fingerprint),
                        "resource_fingerprint": (noncombat_progress_status.resource_fingerprint),
                        "action_fingerprint": (noncombat_progress_status.action_fingerprint),
                        "progress_kind": (noncombat_progress_status.progress_kind),
                        "context": dict(noncombat_progress_status.context),
                        # Durable progress is evaluated on the transition
                        # result, not the pre-action decision state below.
                        "detected_step": next_state.step_index,
                    }
                elif effective_deadlock_evidence is not None:
                    journal_deadlock = effective_deadlock_evidence.to_mapping()
                valid_indices = np.flatnonzero(choice.snapshot.action_mask)
                ranked = sorted(
                    valid_indices.tolist(),
                    key=lambda index: float(choice.policy[index]),
                    reverse=True,
                )[: self.journal_policy_topk]
                journal_event: dict[str, object] = {
                    "event": "decision",
                    "episode_id": f"{journal_episode_id_prefix}{state.episode_id}",
                    "reset_seed": reset_seed,
                    "step_index": state.step_index,
                    # The journal performs compact per-step projection and
                    # only canonicalizes complete DTOs for bounded rich
                    # snapshots. Avoid copying the entire game state on
                    # every held-out decision.
                    "observation": state.observation,
                    "legal_actions": state.legal_actions,
                    # ``selected_index`` remains the learned/model index for
                    # backward-compatible journal readers.  The explicit
                    # fields below remove all ambiguity once strict grouping
                    # makes that differ from the raw dispatch position.
                    "selected_index": choice.candidate_index,
                    "selected_candidate_index": choice.candidate_index,
                    "selected_dispatch_index": choice.dispatch_position,
                    "selected_action_multiplicity": choice.multiplicity,
                    "selected_action_equivalence_fingerprint": (choice.equivalence_fingerprint),
                    "semantic_candidate_count": choice.snapshot.candidate_count,
                    "raw_legal_action_count": len(state.legal_actions),
                    "maximum_candidate_multiplicity": max(
                        reference.multiplicity for reference in choice.action_references
                    ),
                    "selected_action": selected_action,
                    "policy_topk": [
                        {
                            "index": index,
                            "candidate_index": index,
                            "dispatch_index": choice.action_references[index].position,
                            "multiplicity": choice.action_references[index].multiplicity,
                            "equivalence_fingerprint": choice.action_references[index].equivalence_fingerprint,
                            "probability": float(choice.policy[index]),
                        }
                        for index in ranked
                    ],
                    "value": choice.value,
                    "behavior_log_probability": choice.behavior_log_probability,
                    "effective_collection_epsilon": choice.effective_epsilon,
                    "targeted_selection_exploration": (choice.targeted_selection_exploration),
                    "targeted_transaction_entry_exploration": (choice.targeted_transaction_entry_exploration),
                    "transaction_completion_guidance": (choice.transaction_completion_guidance),
                    "transaction_completion_forward_selected": (choice.transaction_completion_forward_selected),
                    "transaction_completion_guidance_fallback": (choice.transaction_completion_guidance_fallback),
                    "transaction_operation": choice.transaction_operation or None,
                    "reward": breakdown.reward,
                    "terminal_reward": breakdown.terminal_reward,
                    "potential_reward": breakdown.potential_reward,
                    "revival_penalty": breakdown.revival_penalty,
                    "pace_penalty": breakdown.pace_penalty,
                    "hp_loss_penalty": breakdown.hp_loss_penalty,
                    "player_hp_lost": player_hp_lost,
                    "revivals_used": revivals_used,
                    "outcome": breakdown.outcome,
                    "deadlock": journal_deadlock,
                }
                if journal_deadlock is not None:
                    # The decision record is anchored at the pre-action state,
                    # while progress stalls are evaluated on the transition
                    # result.  Preserve the exact result DTO only for the
                    # bounded rich anomaly snapshot so the trigger is auditable
                    # without restoring full-state logging on ordinary steps.
                    journal_event.update(
                        {
                            "result_step_index": next_state.step_index,
                            "result_observation": next_state.observation,
                            "result_legal_actions": next_state.legal_actions,
                            "result_terminated": next_state.terminated,
                            "result_truncated": next_state.truncated,
                            "result_terminal_reason": next_state.terminal_reason,
                        }
                    )
                trajectory_journal.write(journal_event)
            timings.record("reward_and_diagnostics", reward_started_ns)
            recurrent_state = choice.recurrent_state
            state = next_state
            act, floor = _run_position(state.observation)
            max_act = max(max_act, act)
            max_floor = max(max_floor, floor)

            flush_segment = bool(
                record
                and segment_steps
                and (len(segment_steps) >= self.unroll_length or breakdown.task_terminal or forced_horizon)
            )
            if flush_segment:
                bootstrap_snapshot: EncodedDecisionSnapshot | None = None
                if segment_steps[-1].discount > 0.0:
                    bootstrap_started_ns = time.perf_counter_ns()
                    bootstrap_snapshot = self.encoder.encode(
                        state.observation,
                        state.legal_actions,
                        device="cpu",
                    ).snapshot
                    maximum_observed_semantic_candidates = max(
                        maximum_observed_semantic_candidates,
                        bootstrap_snapshot.candidate_count,
                    )
                    timings.record("bootstrap_encoding", bootstrap_started_ns)
                completed_unroll = SequenceUnroll(
                    episode_id=state.episode_id,
                    start_step=segment_start_step,
                    policy_version=segment_policy_version,
                    initial_recurrent_state=segment_initial_state,
                    steps=tuple(segment_steps),
                    bootstrap_snapshot=bootstrap_snapshot,
                )
                final_behavior_policy_version = completed_unroll.policy_version
                if unroll_sink is None:
                    unrolls.append(completed_unroll)
                else:
                    adopted_policy_version = unroll_sink(completed_unroll)
                    if adopted_policy_version is not None:
                        if (
                            isinstance(adopted_policy_version, bool)
                            or not isinstance(adopted_policy_version, int)
                            or adopted_policy_version < segment_policy_version
                        ):
                            raise ValueError("unroll sink returned an invalid actor policy version")
                        segment_policy_version = adopted_policy_version
                if progress_sink is not None:
                    progress_sink(
                        EpisodeProgress(
                            episode_id=state.episode_id,
                            reset_seed=reset_seed,
                            steps=steps_taken,
                            reward_total=reward_total,
                            max_act=max_act,
                            max_floor=max_floor,
                            policy_decisions=policy_decisions,
                            forced_decisions=forced_decisions,
                            maximum_observed_candidates=(maximum_observed_candidates),
                            revivals_used=revivals_used,
                            player_hp_lost=player_hp_lost,
                            combat_in_progress=combat_in_progress,
                            phase=str(state.observation.get("phase") or ""),
                            decision_domain=str(state.observation.get("decision_domain") or ""),
                            combat_no_net_progress_steps=(combat_no_net_progress_steps),
                            noncombat_no_durable_progress_steps=(noncombat_no_durable_progress_steps),
                            combat_anchor_enemy_hp_total=(combat_progress_status.anchor_hp),
                            combat_required_net_hp_progress=(combat_progress_status.required_hp_progress),
                            enemy_hp_total=_enemy_hp_totals(state.observation)[0],
                            enemy_max_hp_total=_enemy_hp_totals(state.observation)[1],
                            hand_cards=_zone_count(state.observation, "hand"),
                            draw_cards=_zone_count(state.observation, "draw_pile"),
                            discard_cards=_zone_count(
                                state.observation,
                                "discard_pile",
                            ),
                            exhaust_cards=_zone_count(
                                state.observation,
                                "exhaust_pile",
                            ),
                            legal_action_kinds=_legal_action_kind_counts(state.legal_actions),
                            selected_action_kinds=dict(sorted(selected_action_kind_counts.items())),
                            last_selected_action_kind=last_selected_action_kind,
                            behavior_policy_version=completed_unroll.policy_version,
                            maximum_observed_semantic_candidates=(maximum_observed_semantic_candidates),
                            maximum_equivalence_class_size=(maximum_equivalence_class_size),
                            combat_net_progress_window=(combat_window_selection.effective_window),
                            combat_progress_window_source=(combat_window_selection.source),
                            combat_progress_window_match_id=(combat_window_selection.match_id),
                            maximum_definition_hash_collisions_per_decision=(
                                maximum_definition_hash_collisions_per_decision
                            ),
                            maximum_relation_hash_collisions_per_decision=(
                                maximum_relation_hash_collisions_per_decision
                            ),
                            definition_hash_collisions_total=(definition_hash_collisions_total),
                            relation_hash_collisions_total=(relation_hash_collisions_total),
                        )
                    )
                segment_steps = []
                segment_initial_state = recurrent_state[0].detach().float().cpu().numpy().copy()
                segment_start_step = state.step_index

            if state.terminated or breakdown.task_terminal or forced_horizon:
                break

        if segment_steps:
            raise RuntimeError("collector exited with an unflushed rollout segment")
        if active_transaction_surface is not None:
            raise RuntimeError("collector exited with an unclosed transaction trace")
        terminal_facts = state.transition.facts if state.transition is not None else {}
        typed_run_result = terminal_facts.get("run_result")
        typed_combat_result = terminal_facts.get("combat_result")
        authoritative_native_outcome = bool(
            state.terminated
            and (
                (self.objective == "run" and typed_run_result in {"victory", "defeat"})
                or (self.objective == "combat" and typed_combat_result in {"victory", "defeat"})
            )
        )
        # Only typed simulator terminals and failures with grounded policy
        # evidence may create Monte-Carlo Q labels.  A generic liveness stop is
        # deliberately censored: ``last_task_terminal`` merely says that the
        # collector stopped, not that the selected action factually failed.
        authoritative_outcome = bool(authoritative_native_outcome or trusted_policy_failure)
        transaction_traces: tuple[TransactionTrace, ...] = ()
        if transaction_traces_pending:
            backfilled_traces: list[TransactionTrace] = []
            for trace in transaction_traces_pending:
                option_endpoint = (
                    _extended_transaction_option_endpoint(
                        trace,
                        episodic_steps=tuple(episodic_steps),
                        authoritative_outcome=authoritative_outcome,
                    )
                    if self.transaction_smdp_horizon == "next_rest_or_act"
                    else None
                )
                backfilled_traces.append(
                    backfill_factual_monte_carlo_returns(
                        trace,
                        episode_rewards=tuple(episode_rewards),
                        episode_discounts=tuple(episode_discounts),
                        authoritative_outcome=authoritative_outcome,
                        option_horizon_end=(
                            option_endpoint[0]
                            if option_endpoint is not None
                            else None
                        ),
                        option_horizon_boundary=(
                            option_endpoint[1]
                            if option_endpoint is not None
                            else None
                        ),
                        require_extended_option_horizon=(
                            self.transaction_smdp_horizon == "next_rest_or_act"
                        ),
                    )
                )
            transaction_traces = tuple(backfilled_traces)
        run_won = bool(self.objective == "run" and state.terminated and terminal_facts.get("run_result") == "victory")
        combat_won = bool(
            self.objective == "combat" and state.terminated and terminal_facts.get("combat_result") == "victory"
        )
        if state.terminated and self.objective in {"run", "combat"}:
            typed_success = run_won if self.objective == "run" else combat_won
            if typed_success != (final_outcome == "success"):
                raise CollectionProtocolError("typed terminal result disagrees with the reward outcome")
        resolved_terminal_reason = (
            "combat_progress_stall"
            if combat_progress_stalled
            else "noncombat_event_action_cycle"
            if noncombat_event_cycle
            else "selection_action_cycle"
            if selection_action_cycle
            else "noncombat_progress_stall"
            if noncombat_durable_stalled
            else "semantic_deadlock"
            if effective_deadlock_evidence is not None
            else "curriculum_horizon"
            if curriculum_horizon and self.horizon_as_failure
            else "collection_budget"
            if forced_horizon
            else state.terminal_reason
        )
        completed_episode: CompletedEpisode | None = None
        if episodic_enabled:
            if not episodic_steps:  # reset validation and a positive horizon make this unreachable
                raise RuntimeError("run episode ended without an accepted episodic decision")
            authoritative_run_outcome = bool(state.terminated and typed_run_result in {"victory", "defeat"})
            observed_task_outcome = bool(authoritative_run_outcome or trusted_policy_failure)
            completion = EpisodeCompletion(
                authoritative=observed_task_outcome,
                won=(
                    typed_run_result == "victory"
                    if authoritative_run_outcome
                    else False
                    if trusted_policy_failure
                    else None
                ),
                final_revivals=revivals_used,
                final_hp_loss=player_hp_lost,
                terminal_reason=str(resolved_terminal_reason or "censored_run"),
            )
            completed_episode = backfill_completed_episode(
                episode_id=f"seed-{reset_seed}:{state.episode_id}",
                steps=tuple(episodic_steps),
                completion=completion,
                act_segment_health=tuple(
                    act_segment_health[act] for act in sorted(act_segment_health)
                ),
                data_partition="training",
            )
        failure_credit_records: tuple[EvidenceRecord, ...] = ()
        failure_credit_shadow_metrics: FailureCreditShadowMetrics | None = None
        if failure_credit_pipeline is not None:
            formal_local_failure = bool(
                combat_progress_stalled
                or noncombat_event_cycle
                or selection_action_cycle
                or noncombat_durable_stalled
                or effective_deadlock_evidence is not None
            )
            failure_credit_result = failure_credit_pipeline.finalize(
                failure_kind=(
                    str(resolved_terminal_reason)
                    if formal_local_failure and resolved_terminal_reason is not None
                    else None
                ),
                local_failure=formal_local_failure,
                terminal_succeeded=bool(final_outcome == "success"),
                censored_reason=str(resolved_terminal_reason or "episode_boundary_censored"),
            )
            failure_credit_records = failure_credit_result.records
            failure_credit_shadow_metrics = failure_credit_result.metrics
        ordered_act_efficiency = tuple(act_boundary_efficiency[act] for act in sorted(act_boundary_efficiency))
        macro_return_diagnostics = _macro_return_diagnostics(
            rewards=tuple(episodic_diagnostic_rewards),
            discounts=tuple(episodic_diagnostic_discounts),
            decisions=tuple(episodic_macro_decisions),
            authoritative=bool(completed_episode is not None and completed_episode.completion.authoritative),
        )
        return CollectedEpisode(
            unrolls=tuple(unrolls),
            metrics=EpisodeMetrics(
                episode_id=state.episode_id,
                reset_seed=reset_seed,
                steps=steps_taken,
                reward_total=reward_total,
                terminal_reason=resolved_terminal_reason,
                truncated=forced_horizon,
                run_won=run_won,
                combat_won=combat_won,
                act1_cleared=bool(max_act >= 2),
                max_act=max_act,
                max_floor=max_floor,
                policy_decisions=policy_decisions,
                forced_decisions=forced_decisions,
                maximum_observed_candidates=maximum_observed_candidates,
                deadlocked=deadlocked,
                combat_progress_stalled=combat_progress_stalled,
                maximum_combat_no_net_progress_steps=(maximum_combat_no_net_progress_steps),
                noncombat_progress_stalled=noncombat_progress_stalled,
                maximum_noncombat_no_durable_progress_steps=(maximum_noncombat_no_durable_progress_steps),
                revivals_used=revivals_used,
                revival_free_combat_win=bool(combat_won and revivals_used == 0),
                revival_free_act1_clear=bool(1 in act_boundary_efficiency and act_boundary_efficiency[1][0] == 0),
                revival_free_run_win=bool(run_won and revivals_used == 0),
                player_hp_lost=player_hp_lost,
                stall_evidence=stall_evidence,
                combat_policy_failed=combat_progress_stalled,
                trusted_policy_failure=trusted_policy_failure,
                noncombat_event_cycle=noncombat_event_cycle,
                selection_action_cycle=selection_action_cycle,
                maximum_observed_semantic_candidates=(maximum_observed_semantic_candidates),
                maximum_equivalence_class_size=(maximum_equivalence_class_size),
                act_revival_counts=tuple(item[0] for item in ordered_act_efficiency),
                act_hp_loss_counts=tuple(item[1] for item in ordered_act_efficiency),
                maximum_definition_hash_collisions_per_decision=(maximum_definition_hash_collisions_per_decision),
                maximum_relation_hash_collisions_per_decision=(maximum_relation_hash_collisions_per_decision),
                definition_hash_collisions_total=(definition_hash_collisions_total),
                relation_hash_collisions_total=(relation_hash_collisions_total),
                targeted_selection_exploration_decisions=(targeted_selection_exploration_decisions),
                targeted_transaction_entry_exploration_decisions=(targeted_transaction_entry_exploration_decisions),
                transaction_completion_guidance_decisions=(transaction_completion_guidance_decisions),
                transaction_completion_forward_decisions=(transaction_completion_forward_decisions),
                transaction_completion_guidance_fallbacks=(transaction_completion_guidance_fallbacks),
                maximum_effective_collection_epsilon=(maximum_effective_collection_epsilon),
                policy_top1_top2_logit_margin_mean=(
                    policy_logit_margin_total / policy_logit_margin_count if policy_logit_margin_count else 0.0
                ),
                policy_top1_top2_logit_margin_max=(policy_top1_top2_logit_margin_max),
                selection_transactions_started=selection_transactions_started,
                selection_transactions_closed=selection_transactions_closed,
                selection_transactions_completed=selection_transactions_completed,
                selection_transactions_cancelled=selection_transactions_cancelled,
                selection_transactions_unresolved=selection_transactions_unresolved,
                rest_site_selection_transactions_started=(rest_site_selection_transactions_started),
                rest_site_selection_transactions_closed=(rest_site_selection_transactions_closed),
                rest_site_selection_transactions_completed=(rest_site_selection_transactions_completed),
                rest_site_selection_transactions_cancelled=(rest_site_selection_transactions_cancelled),
                rest_site_selection_transactions_unresolved=(rest_site_selection_transactions_unresolved),
                forge_selection_transactions_started=(forge_selection_transactions_started),
                forge_selection_transactions_closed=(forge_selection_transactions_closed),
                forge_selection_transactions_completed=(forge_selection_transactions_completed),
                forge_selection_transactions_cancelled=(forge_selection_transactions_cancelled),
                forge_selection_transactions_unresolved=(forge_selection_transactions_unresolved),
                shop_card_removal_transactions_started=(shop_card_removal_transactions_started),
                shop_card_removal_transactions_closed=(shop_card_removal_transactions_closed),
                shop_card_removal_transactions_completed=(shop_card_removal_transactions_completed),
                shop_card_removal_transactions_cancelled=(shop_card_removal_transactions_cancelled),
                shop_card_removal_transactions_unresolved=(shop_card_removal_transactions_unresolved),
                shaping_reward_total=shaping_reward_total,
                shaping_reward_per_max_floor=(shaping_reward_total / max(1, max_floor)),
                boss_victory_acts=tuple(sorted(boss_victory_acts)),
                macro_return_diagnostics=macro_return_diagnostics,
            ),
            actor_policy_version=segment_policy_version,
            behavior_policy_version=final_behavior_policy_version,
            timings=timings.snapshot(),
            transaction_traces=transaction_traces,
            completed_episode=completed_episode,
            liveness_probe=liveness_probe,
            failure_credit_records=failure_credit_records,
            failure_credit_shadow_metrics=failure_credit_shadow_metrics,
        )


__all__ = [
    "CollectedEpisode",
    "CollectionProtocolError",
    "EpisodeMetrics",
    "EpisodeProgress",
    "GroundedCollector",
    "MacroReturnDiagnostic",
]
