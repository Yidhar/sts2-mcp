"""Recurrent sequence actor-critic learner using IMPALA V-trace targets."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor

from sts2_baseline import SequenceUnroll
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.encoding.grounded import grounding_encoding_identity
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.models import RecurrentCandidateModel, RecurrentCandidateOutput

from .config import (
    EpisodicLearningConfig,
    FailureCreditConfig,
    OptimizationConfig,
    TransactionLearningConfig,
)
from .episode_replay import (
    HorizonTargets,
    ReplaySequence,
    healthy_act_segment_for_step,
)
from .failure_credit.actor_eligibility import (
    actor_step_is_fresh,
    contrast_actor_unit_effective,
    cycle_actor_unit_effective,
    direct_actor_unit_effective,
    risk_actor_row_effective,
)
from .failure_credit.contracts import (
    CreditPlan,
    DirectPolicyTarget,
    EvidenceStratum,
    LearningContext,
)
from .transaction import (
    TransactionLifecycleOutcome,
    TransactionPolicyTarget,
    TransactionTrace,
    factual_transaction_policy_targets,
    observed_outcome_pairs,
    selection_delta_index,
)

_LEARNER_DYNAMICS_STATE_VERSION = "sts2-vtrace-learner-dynamics-v2"
_ONE_HOT_BREAKER_CONSECUTIVE_BATCHES_V1 = 8
_ONE_HOT_BREAKER_DURATION_UPDATES_V1 = 8
_ONE_HOT_BREAKER_ENTROPY_WEIGHT_V1 = 0.012
_ONE_HOT_BREAKER_RATIO_TOLERANCE_V1 = 1.0e-6
_POLICY_COLLAPSE_BREAKER_CONSECUTIVE_BATCHES_V2 = 8
_POLICY_COLLAPSE_BREAKER_DURATION_UPDATES_V2 = 16
_POLICY_COLLAPSE_BREAKER_ENTROPY_WEIGHT_V2 = 0.012
_POLICY_COLLAPSE_BREAKER_NORMALIZED_ENTROPY_V2 = 0.10
_POLICY_COLLAPSE_BREAKER_LOW_ENTROPY_FRACTION_V2 = 0.75
_POLICY_COLLAPSE_BREAKER_MINIMUM_DECISIONS_V2 = 16

# Execution-only liveness graph bounds. They do not change sampled records,
# labels, loss weights or the equal-record objective, so exact resume may adopt
# this safer execution plan without changing training lineage. A single record
# is never truncated: an over-budget record becomes an auditable singleton.
_LIVENESS_AUTOGRAD_PACKING_VERSION = 1
_LIVENESS_AUTOGRAD_MAX_STEPS = 256
_LIVENESS_AUTOGRAD_MAX_CANDIDATES = 1_024
_LIVENESS_AUTOGRAD_MAX_SEGMENTS = 16


@dataclass(frozen=True, slots=True)
class LearnerTimings:
    validation_ms: float
    recurrent_forward_ms: float
    target_and_loss_ms: float
    backward_ms: float
    episodic_replay_ms: float
    optimizer_step_ms: float
    total_ms: float

    def to_mapping(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LearnerMetrics:
    loss: float
    online_objective_loss: float
    transaction_objective_loss: float
    liveness_objective_loss: float
    episodic_objective_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    normalized_entropy: float
    entropy_weight: float
    entropy_breaker_active: int
    entropy_breaker_one_hot_condition: int
    entropy_breaker_soft_collapse_condition: int
    entropy_breaker_consecutive_batches: int
    entropy_breaker_remaining_updates: int
    entropy_breaker_triggers: int
    advantage_mean: float
    value_target_mean: float
    importance_ratio_mean: float
    importance_ratio_max: float
    importance_clip_fraction: float
    gradient_norm: float
    unrolls: int
    environment_steps: int
    policy_decisions: int
    maximum_policy_lag: int
    transaction_effect_loss: float
    transaction_delta_loss: float
    transaction_q_loss: float
    transaction_pairwise_ranking_loss: float
    transaction_completion_policy_loss: float
    transaction_entry_support_loss: float
    transaction_smdp_q_loss: float
    transaction_traces: int
    transaction_effect_labels: int
    transaction_q_labels: int
    transaction_pairs: int
    transaction_policy_labels: int
    transaction_policy_preferred_labels: int
    transaction_policy_avoided_labels: int
    transaction_entry_support_labels: int
    transaction_entry_support_satisfied_labels: int
    transaction_entry_support_singleton_suppressed_labels: int
    transaction_smdp_q_labels: int
    transaction_lifecycle_committed: int
    transaction_lifecycle_cancelled: int
    transaction_lifecycle_unresolved: int
    transaction_lifecycle_deadlock: int
    transaction_entry_model_probability_mean: float
    transaction_entry_collection_model_probability_mean: float
    transaction_entry_behavior_probability_mean: float
    transaction_entry_log_support_gap_mean: float
    transaction_upgrade_entry_support_labels: int
    transaction_remove_entry_support_labels: int
    transaction_upgrade_entry_model_probability_mean: float
    transaction_remove_entry_model_probability_mean: float
    transaction_upgrade_entry_behavior_probability_mean: float
    transaction_remove_entry_behavior_probability_mean: float
    liveness_credit_loss: float
    liveness_cost_critic_loss: float
    liveness_value_critic_loss: float
    liveness_q_critic_loss: float
    liveness_cost_actor_loss: float
    liveness_direct_policy_loss: float
    liveness_cycle_policy_loss: float
    liveness_contrast_policy_loss: float
    liveness_completion_policy_loss: float
    liveness_credit_plans: int
    liveness_cost_labels: int
    liveness_value_labels: int
    liveness_q_labels: int
    liveness_cost_actor_labels: int
    liveness_direct_policy_labels: int
    liveness_cycle_policy_labels: int
    liveness_contrast_policy_labels: int
    liveness_completion_policy_labels: int
    liveness_forced_actor_suppressed_labels: int
    liveness_censored_suppressed_labels: int
    liveness_policy_lag_suppressed_labels: int
    liveness_risk_actor_phase_suppressed_labels: int
    liveness_risk_actor_saturation_suppressed_labels: int
    liveness_centered_risk_mean: float
    liveness_centered_risk_max_abs: float
    liveness_policy_gradient_norm: float
    liveness_critic_gradient_norm: float
    liveness_gradient_norm_before_clip: float
    liveness_gradient_norm_after_clip: float
    liveness_gradient_clip_scale: float
    liveness_head_calibration_active: int
    liveness_risk_actor_enabled: int
    liveness_replayed_contexts: int
    liveness_replayed_steps: int
    liveness_replayed_candidates: int
    liveness_autograd_microbatches: int
    liveness_autograd_segments: int
    episodic_loss: float
    episodic_primary_policy_loss: float
    episodic_task_value_loss: float
    episodic_revival_value_loss: float
    episodic_revival_policy_loss: float
    episodic_act_segment_policy_loss: float
    episodic_combat_hp_loss_value_loss: float
    episodic_sequences: int
    episodic_burn_in_steps: int
    episodic_learn_steps: int
    episodic_success_policy_candidate_labels: int
    episodic_policy_labels: int
    episodic_policy_active_sequences: int
    episodic_failure_policy_suppressed_labels: int
    episodic_policy_lag_suppressed_labels: int
    episodic_success_trust_region_suppressed_labels: int
    episodic_success_surface_exempted_labels: int
    episodic_act_segment_policy_labels: int
    episodic_act_segment_health_gate_suppressed_labels: int
    episodic_act_segment_healthy_acts: int
    episodic_task_value_labels: int
    episodic_combat_hp_loss_value_labels: int
    episodic_revival_value_labels: int
    episodic_efficiency_policy_labels: int
    episodic_importance_ratio_mean: float
    episodic_importance_ratio_max: float
    episodic_importance_clip_fraction: float
    episodic_maximum_policy_lag: int
    timings: LearnerTimings

    def to_mapping(self) -> dict[str, float | int | dict[str, float]]:
        payload: dict[str, float | int | dict[str, float]] = asdict(self)
        payload["batch_environment_steps"] = payload.pop("environment_steps")
        payload["timings"] = self.timings.to_mapping()
        return payload


@dataclass(frozen=True, slots=True)
class _TransactionLossBatch:
    effect_loss: Tensor
    delta_loss: Tensor
    q_loss: Tensor
    pairwise_loss: Tensor
    completion_policy_loss: Tensor
    entry_support_loss: Tensor
    smdp_q_loss: Tensor
    effect_labels: int
    q_labels: int
    pair_count: int
    policy_labels: int
    policy_preferred_labels: int
    policy_avoided_labels: int
    entry_support_labels: int
    entry_support_satisfied_labels: int
    entry_support_singleton_suppressed_labels: int
    smdp_q_labels: int
    lifecycle_committed: int
    lifecycle_cancelled: int
    lifecycle_unresolved: int
    lifecycle_deadlock: int
    entry_model_probability_mean: float
    entry_collection_model_probability_mean: float
    entry_behavior_probability_mean: float
    entry_log_support_gap_mean: float
    upgrade_entry_support_labels: int
    remove_entry_support_labels: int
    upgrade_entry_model_probability_mean: float
    remove_entry_model_probability_mean: float
    upgrade_entry_behavior_probability_mean: float
    remove_entry_behavior_probability_mean: float


@dataclass(frozen=True, slots=True)
class LivenessCreditLosses:
    """Independent bounded-risk and factual policy-credit losses.

    The helper that produces this bundle deliberately receives model outputs,
    not game-specific failure records.  A versioned credit-plan/replay adapter
    can therefore resolve factual ``decision_id`` references into active-shape
    snapshots without coupling this learner to collector internals.
    """

    value_critic_loss: Tensor
    critic_loss: Tensor
    risk_actor_loss: Tensor
    direct_avoid_loss: Tensor
    cycle_likelihood_loss: Tensor
    contrast_loss: Tensor
    completion_loss: Tensor
    value_labels: int
    critic_labels: int
    risk_actor_labels: int
    direct_avoid_labels: int
    cycle_labels: int
    contrast_labels: int
    completion_labels: int
    forced_actor_suppressed_labels: int
    censored_suppressed_labels: int
    policy_lag_suppressed_labels: int
    risk_actor_phase_suppressed_labels: int
    risk_actor_saturation_suppressed_labels: int
    centered_risk_mean: float
    centered_risk_max_abs: float
    replayed_contexts: int = 0
    replayed_steps: int = 0
    replayed_candidates: int = 0
    autograd_segments: int = 0


@dataclass(frozen=True, slots=True)
class LivenessReplayWork:
    """Exact active-shape work admitted by the failure-credit learner."""

    contexts: int
    steps: int
    candidates: int
    autograd_segments: int


@dataclass(frozen=True, slots=True)
class LivenessAutogradPack:
    """One contiguous, equal-record-preserving liveness backward pack."""

    start: int
    end: int
    work: LivenessReplayWork
    oversized_singleton: bool


@dataclass(frozen=True, slots=True)
class LivenessLabelRow:
    """Pure DTO actor/critic-mask decision for one recurrent replay row."""

    context_id: str
    step_index: int
    decision_id: str
    forced: bool
    censored: bool
    legal_candidates: int
    fresh: bool
    value_critic_requested: bool
    q_critic_requested: bool
    risk_actor_requested: bool
    risk_actor_mask: bool
    direct_target: str | None
    direct_actor_mask: bool

    @property
    def key(self) -> tuple[str, int]:
        return (self.context_id, self.step_index)

    @property
    def actor_eligible(self) -> bool:
        return not self.forced and not self.censored and self.legal_candidates > 1

    @property
    def effective_risk_actor(self) -> bool:
        return self.risk_actor_mask and self.actor_eligible

    @property
    def effective_direct_actor(self) -> bool:
        return self.direct_actor_mask and self.actor_eligible


@dataclass(frozen=True, slots=True)
class LivenessGroupLabel:
    """Pure DTO decision for one cycle or matched-outcome actor group."""

    kind: str
    row_keys: tuple[tuple[str, int], ...]
    fresh: bool
    effective: bool


@dataclass(frozen=True, slots=True)
class LivenessLabelManifest:
    """Auditable phase, masks and work budget compiled without a model."""

    learner_update: int
    calibration_active: bool
    risk_actor_enabled: bool
    rows: tuple[LivenessLabelRow, ...]
    cycle_groups: tuple[LivenessGroupLabel, ...]
    contrast_groups: tuple[LivenessGroupLabel, ...]
    policy_lag_suppressed_labels: int
    risk_actor_phase_suppressed_labels: int
    work: LivenessReplayWork

    def row_map(self) -> dict[tuple[str, int], LivenessLabelRow]:
        return {row.key: row for row in self.rows}


def _liveness_autograd_packs(
    manifests: Sequence[LivenessLabelManifest],
    *,
    maximum_records: int,
) -> tuple[LivenessAutogradPack, ...]:
    """Partition replay records by actual active-shape graph work.

    Packing only by record count allowed one 256-step failure context plus
    three short controls to form a graph far larger than the benchmark
    fixture. This preserves order and the exact global equal-record objective
    while bounding steps, candidates and TBPTT graph segments. Evidence is
    never truncated or rejected.
    """

    if isinstance(maximum_records, bool) or not isinstance(maximum_records, int):
        raise TypeError("maximum_records must be an integer")
    if maximum_records <= 0:
        raise ValueError("maximum_records must be positive")
    if not manifests:
        return ()

    packs: list[LivenessAutogradPack] = []
    pack_start = 0
    contexts = 0
    steps = 0
    candidates = 0
    segments = 0

    def publish(end: int) -> None:
        nonlocal pack_start, contexts, steps, candidates, segments
        if end <= pack_start:
            return
        record_count = end - pack_start
        oversized = record_count == 1 and (
            steps > _LIVENESS_AUTOGRAD_MAX_STEPS
            or candidates > _LIVENESS_AUTOGRAD_MAX_CANDIDATES
            or segments > _LIVENESS_AUTOGRAD_MAX_SEGMENTS
        )
        packs.append(
            LivenessAutogradPack(
                start=pack_start,
                end=end,
                work=LivenessReplayWork(
                    contexts=contexts,
                    steps=steps,
                    candidates=candidates,
                    autograd_segments=segments,
                ),
                oversized_singleton=oversized,
            )
        )
        pack_start = end
        contexts = 0
        steps = 0
        candidates = 0
        segments = 0

    for index, manifest in enumerate(manifests):
        work = manifest.work
        current_records = index - pack_start
        would_exceed = current_records > 0 and (
            current_records + 1 > maximum_records
            or steps + work.steps > _LIVENESS_AUTOGRAD_MAX_STEPS
            or candidates + work.candidates > _LIVENESS_AUTOGRAD_MAX_CANDIDATES
            or segments + work.autograd_segments > _LIVENESS_AUTOGRAD_MAX_SEGMENTS
        )
        if would_exceed:
            publish(index)

        contexts += work.contexts
        steps += work.steps
        candidates += work.candidates
        segments += work.autograd_segments

        # An individually over-budget record stays intact but cannot pull
        # another record into the same graph.
        if (
            work.steps > _LIVENESS_AUTOGRAD_MAX_STEPS
            or work.candidates > _LIVENESS_AUTOGRAD_MAX_CANDIDATES
            or work.autograd_segments > _LIVENESS_AUTOGRAD_MAX_SEGMENTS
        ):
            publish(index + 1)

    publish(len(manifests))
    return tuple(packs)


@dataclass(slots=True)
class _LivenessLabelRowBuilder:
    context_id: str
    step_index: int
    decision_id: str
    forced: bool
    censored: bool
    legal_candidates: int
    fresh: bool
    value_critic_requested: bool = False
    q_critic_requested: bool = False
    risk_actor_requested: bool = False
    risk_actor_mask: bool = False
    direct_target: str | None = None
    direct_actor_mask: bool = False

    @property
    def actor_eligible(self) -> bool:
        return not self.forced and not self.censored and self.legal_candidates > 1

    def freeze(self) -> LivenessLabelRow:
        return LivenessLabelRow(
            context_id=self.context_id,
            step_index=self.step_index,
            decision_id=self.decision_id,
            forced=self.forced,
            censored=self.censored,
            legal_candidates=self.legal_candidates,
            fresh=self.fresh,
            value_critic_requested=self.value_critic_requested,
            q_critic_requested=self.q_critic_requested,
            risk_actor_requested=self.risk_actor_requested,
            risk_actor_mask=self.risk_actor_mask,
            direct_target=self.direct_target,
            direct_actor_mask=self.direct_actor_mask,
        )


def compile_liveness_label_manifest(
    credit_plans: tuple[CreditPlan, ...],
    *,
    config: FailureCreditConfig,
    current_policy_version: int,
    current_learner_update: int,
) -> LivenessLabelManifest:
    """Compile failure-credit masks and hard work bounds from immutable DTOs.

    This function deliberately performs no model forward and owns no optimizer.
    The production learner consumes this same manifest when materializing
    tensors, so shadow validation observes the actual freshness, phase,
    forced/censored and work-budget contract rather than a second
    approximation of it.
    """

    if not isinstance(config, FailureCreditConfig):
        raise TypeError("config must be FailureCreditConfig")
    if not isinstance(credit_plans, tuple) or not all(isinstance(plan, CreditPlan) for plan in credit_plans):
        raise TypeError("credit_plans must be a CreditPlan tuple")
    for label, value in (
        ("current_policy_version", current_policy_version),
        ("current_learner_update", current_learner_update),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    if credit_plans and not config.learning_enabled:
        raise ValueError("credit plans require formal liveness learning to be enabled")
    if len(credit_plans) > config.sample_records:
        raise ValueError("credit plan batch exceeds failure_credit.sample_records")

    calibration_active = current_learner_update < config.liveness_head_calibration_updates
    risk_actor_enabled = current_learner_update >= config.liveness_risk_actor_start_update
    contexts: dict[str, LearningContext] = {}
    context_order: list[str] = []
    needed_steps: dict[str, set[int]] = {}

    def register_context_step(
        context: LearningContext,
        step_index: int,
    ) -> tuple[str, int]:
        if not context.burn_in_steps <= step_index < len(context.steps):
            raise ValueError("liveness target must reference a post-burn-in learning step")
        previous = contexts.get(context.context_id)
        if previous is not None and previous is not context:
            raise ValueError("different LearningContext objects reuse one context_id")
        if previous is None:
            contexts[context.context_id] = context
            context_order.append(context.context_id)
        needed_steps.setdefault(context.context_id, set()).add(step_index)
        return (context.context_id, step_index)

    for plan in credit_plans:
        if plan.provenance.policy_version > current_policy_version:
            raise ValueError("credit-plan policy version is newer than the learner")
        if EvidenceStratum.CENSORED in plan.strata and plan.strata != (EvidenceStratum.CENSORED,):
            raise ValueError("CENSORED cannot be mixed with authoritative credit strata")
        plan_contexts = {plan.context.context_id: plan.context}
        for value_target in plan.liveness_value_targets:
            register_context_step(plan.context, value_target.step_index)
        for q_target in plan.liveness_q_targets:
            register_context_step(plan.context, q_target.step_index)
        for direct_target in plan.direct_policy_targets:
            register_context_step(plan.context, direct_target.step_index)
        for cycle_target in plan.cycle_policy_targets:
            for step_index in cycle_target.step_indices:
                register_context_step(plan.context, step_index)
        for risk_sequence in plan.risk_sequences:
            for step_index in risk_sequence.step_indices:
                register_context_step(plan.context, step_index)
        for contrast_target in plan.contrast_policy_targets:
            for arm in (
                contrast_target.pair.better,
                contrast_target.pair.worse,
            ):
                previous = plan_contexts.get(arm.context.context_id)
                if previous is not None and previous is not arm.context:
                    raise ValueError("different LearningContext objects reuse one context_id")
                plan_contexts[arm.context.context_id] = arm.context
                register_context_step(arm.context, arm.step_index)
        if len(plan_contexts) > config.liveness_maximum_contexts_per_record:
            raise ValueError("credit plan exceeds " "failure_credit.liveness_maximum_contexts_per_record")

    builders: dict[tuple[str, int], _LivenessLabelRowBuilder] = {}
    for context_id in context_order:
        context = contexts[context_id]
        for step_index in sorted(needed_steps.get(context_id, ())):
            step = context.steps[step_index]
            if step.policy_version > current_policy_version:
                raise ValueError("liveness learning step is newer than the learner")
            builders[(context_id, step_index)] = _LivenessLabelRowBuilder(
                context_id=context_id,
                step_index=step_index,
                decision_id=step.decision_id,
                forced=step.forced,
                censored=False,
                legal_candidates=int(step.snapshot.action_mask.sum()),
                fresh=actor_step_is_fresh(
                    step,
                    current_policy_version=current_policy_version,
                    policy_gradient_max_lag=config.policy_gradient_max_lag,
                ),
            )

    cycle_groups: list[LivenessGroupLabel] = []
    contrast_groups: list[LivenessGroupLabel] = []
    policy_lag_suppressed_labels = 0
    risk_actor_phase_suppressed_labels = 0

    def builder(
        context: LearningContext,
        step_index: int,
    ) -> _LivenessLabelRowBuilder:
        return builders[(context.context_id, step_index)]

    # Mark censoring before actor-group eligibility is derived.
    for plan in credit_plans:
        if EvidenceStratum.CENSORED in plan.strata:
            for item in builders.values():
                # A censored plan normally has no target rows.  If a malformed
                # future compiler references one, suppress only that plan's
                # owned context rather than contaminating unrelated records.
                if item.context_id == plan.context.context_id:
                    item.censored = True

    for plan in credit_plans:
        for value_target in plan.liveness_value_targets:
            builder(
                plan.context,
                value_target.step_index,
            ).value_critic_requested = True
        for q_target in plan.liveness_q_targets:
            builder(
                plan.context,
                q_target.step_index,
            ).q_critic_requested = True
        for sequence in plan.risk_sequences:
            for step_index in sequence.step_indices:
                item = builder(plan.context, step_index)
                item.risk_actor_requested = True
                if not item.fresh:
                    policy_lag_suppressed_labels += 1
                elif not item.actor_eligible:
                    continue
                elif not risk_actor_enabled:
                    risk_actor_phase_suppressed_labels += 1
                else:
                    item.risk_actor_mask = risk_actor_row_effective(
                        plan,
                        step_index,
                        current_policy_version=current_policy_version,
                        policy_gradient_max_lag=config.policy_gradient_max_lag,
                        risk_actor_enabled=risk_actor_enabled,
                    )
        for direct_target in plan.direct_policy_targets:
            item = builder(plan.context, direct_target.step_index)
            target_name = direct_target.target.value
            previous_direct_target = item.direct_target
            if previous_direct_target is not None and previous_direct_target != target_name:
                raise ValueError("conflicting direct policy targets reference one decision")
            item.direct_target = target_name
            if not item.fresh:
                policy_lag_suppressed_labels += 1
            else:
                item.direct_actor_mask = direct_actor_unit_effective(
                    plan,
                    direct_target,
                    current_policy_version=current_policy_version,
                    policy_gradient_max_lag=config.policy_gradient_max_lag,
                )
        for cycle_target in plan.cycle_policy_targets:
            keys = tuple((plan.context.context_id, step_index) for step_index in cycle_target.step_indices)
            fresh = all(builders[key].fresh for key in keys)
            if not fresh:
                policy_lag_suppressed_labels += 1
            cycle_groups.append(
                LivenessGroupLabel(
                    kind="cycle",
                    row_keys=keys,
                    fresh=fresh,
                    effective=cycle_actor_unit_effective(
                        plan,
                        cycle_target,
                        current_policy_version=current_policy_version,
                        policy_gradient_max_lag=config.policy_gradient_max_lag,
                    ),
                )
            )
        for contrast_target in plan.contrast_policy_targets:
            keys = (
                (
                    contrast_target.pair.better.context.context_id,
                    contrast_target.pair.better.step_index,
                ),
                (
                    contrast_target.pair.worse.context.context_id,
                    contrast_target.pair.worse.step_index,
                ),
            )
            fresh = all(builders[key].fresh for key in keys)
            if not fresh:
                policy_lag_suppressed_labels += 1
            contrast_groups.append(
                LivenessGroupLabel(
                    kind="contrast",
                    row_keys=keys,
                    fresh=fresh,
                    effective=contrast_actor_unit_effective(
                        plan,
                        contrast_target,
                        current_policy_version=current_policy_version,
                        policy_gradient_max_lag=config.policy_gradient_max_lag,
                    ),
                )
            )

    rows = tuple(
        builders[(context_id, step_index)].freeze()
        for context_id in context_order
        for step_index in sorted(needed_steps.get(context_id, ()))
    )
    replayed_steps = 0
    replayed_candidates = 0
    autograd_segments = 0
    for context_id in context_order:
        step_indices = needed_steps.get(context_id)
        if not step_indices:
            continue
        context = contexts[context_id]
        prefix_steps = max(step_indices) + 1
        replayed_steps += prefix_steps
        replayed_candidates += sum(context.steps[index].snapshot.candidate_count for index in range(prefix_steps))
        trainable_steps = prefix_steps - context.burn_in_steps
        autograd_segments += math.ceil(trainable_steps / config.liveness_tbptt_window_steps)
    if replayed_steps > config.liveness_maximum_replayed_steps_per_update:
        raise ValueError("failure-credit replay exceeds " "liveness_maximum_replayed_steps_per_update")
    if replayed_candidates > config.liveness_maximum_replayed_candidates_per_update:
        raise ValueError("failure-credit replay exceeds " "liveness_maximum_replayed_candidates_per_update")
    return LivenessLabelManifest(
        learner_update=current_learner_update,
        calibration_active=calibration_active,
        risk_actor_enabled=risk_actor_enabled,
        rows=rows,
        cycle_groups=tuple(cycle_groups),
        contrast_groups=tuple(contrast_groups),
        policy_lag_suppressed_labels=policy_lag_suppressed_labels,
        risk_actor_phase_suppressed_labels=(risk_actor_phase_suppressed_labels),
        work=LivenessReplayWork(
            contexts=len(context_order),
            steps=replayed_steps,
            candidates=replayed_candidates,
            autograd_segments=autograd_segments,
        ),
    )


def _optional_row_mask(
    value: Tensor | None,
    *,
    rows: int,
    device: torch.device,
    default: bool = False,
    label: str,
) -> Tensor:
    if value is None:
        return torch.full((rows,), default, device=device, dtype=torch.bool)
    if value.shape != (rows,) or value.dtype != torch.bool:
        raise ValueError(f"{label} must be a rank-1 bool tensor with {rows} rows")
    if value.device != device:
        raise ValueError(f"{label} must be on {device}")
    return value


def liveness_credit_losses(
    *,
    policy_log_probabilities: Tensor,
    candidate_liveness_cost_values: Tensor,
    liveness_cost_values: Tensor | None = None,
    action_mask: Tensor,
    selected_action_indices: Tensor,
    value_targets: Tensor | None = None,
    value_critic_mask: Tensor | None = None,
    risk_targets: Tensor | None = None,
    risk_critic_mask: Tensor | None = None,
    risk_actor_mask: Tensor | None = None,
    forced_mask: Tensor | None = None,
    censored_mask: Tensor | None = None,
    direct_avoid_mask: Tensor | None = None,
    completion_mask: Tensor | None = None,
    cycle_groups: tuple[tuple[int, ...], ...] = (),
    cycle_behavior_mean_log_probabilities: tuple[float, ...] | None = None,
    cycle_margins: tuple[float, ...] | None = None,
    contrast_pairs: tuple[tuple[int, int], ...] = (),
    contrast_margins: tuple[float, ...] | None = None,
    risk_advantage_clip: float = 0.25,
    risk_actor_min_selected_probability: float = 0.0,
    contrast_margin: float = 0.10,
) -> LivenessCreditLosses:
    """Build factual liveness losses over one active-shape decision batch.

    ``risk_critic_mask`` controls factual cost supervision.  Forced decisions
    may still train that critic, but they can never train an actor because no
    alternative action exists.  ``censored_mask`` suppresses every liveness
    target.  Direct AVOID, cycle likelihood, contrast, completion, and the
    centered risk actor all operate only on non-forced rows with at least two
    legal candidates.

    The centered actor objective is independent of the primary task value:

    ``log pi(a|s) * clip(C(s,a) - E_pi[C(s,.)])``.

    Consequently a task baseline fixed at ``V_task=-1`` cannot erase this
    gradient.  Candidate costs are detached from the actor term so policy
    optimization cannot lower the objective by corrupting its own critic.
    """

    if policy_log_probabilities.ndim != 2:
        raise ValueError("policy_log_probabilities must have shape [N, A]")
    rows, candidates = policy_log_probabilities.shape
    if rows <= 0 or candidates <= 0:
        raise ValueError("liveness decision batch must be non-empty")
    expected = (rows, candidates)
    if candidate_liveness_cost_values.shape != expected:
        raise ValueError("candidate_liveness_cost_values must match policy shape")
    if action_mask.shape != expected or action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be a bool tensor matching policy shape")
    device = policy_log_probabilities.device
    for label, value in (
        ("candidate_liveness_cost_values", candidate_liveness_cost_values),
        ("action_mask", action_mask),
        ("selected_action_indices", selected_action_indices),
    ):
        if value.device != device:
            raise ValueError(f"{label} must be on {device}")
    if liveness_cost_values is not None:
        if liveness_cost_values.shape != (rows,) or not torch.is_floating_point(liveness_cost_values):
            raise ValueError("liveness_cost_values must be a rank-1 floating tensor")
        if liveness_cost_values.device != device:
            raise ValueError(f"liveness_cost_values must be on {device}")
        if not bool(torch.isfinite(liveness_cost_values).all().item()):
            raise ValueError("liveness_cost_values contains NaN or infinity")
        if bool(((liveness_cost_values < 0.0) | (liveness_cost_values > 1.0)).any().item()):
            raise ValueError("liveness_cost_values must be in [0, 1]")
    if not torch.is_floating_point(policy_log_probabilities):
        raise TypeError("policy_log_probabilities must be floating point")
    if not torch.is_floating_point(candidate_liveness_cost_values):
        raise TypeError("candidate_liveness_cost_values must be floating point")
    if selected_action_indices.shape != (rows,) or selected_action_indices.dtype != torch.long:
        raise ValueError("selected_action_indices must be a rank-1 torch.long tensor")
    if bool(((selected_action_indices < 0) | (selected_action_indices >= candidates)).any().item()):
        raise ValueError("selected_action_indices contains an out-of-range action")
    selected_is_legal = action_mask.gather(1, selected_action_indices[:, None]).squeeze(1)
    if not bool(selected_is_legal.all().item()):
        raise ValueError("liveness batch selected an invalid candidate")
    legal_costs = candidate_liveness_cost_values.masked_select(action_mask)
    if not bool(torch.isfinite(legal_costs).all().item()):
        raise ValueError("candidate liveness costs contain NaN or infinity")
    if bool(((legal_costs < 0.0) | (legal_costs > 1.0)).any().item()):
        raise ValueError("candidate liveness costs must be in [0, 1]")
    selected_log_probabilities = policy_log_probabilities.gather(1, selected_action_indices[:, None]).squeeze(1)
    if not bool(torch.isfinite(selected_log_probabilities).all().item()):
        raise ValueError("selected policy log probabilities must be finite")
    if (
        isinstance(risk_advantage_clip, bool)
        or not isinstance(risk_advantage_clip, int | float)
        or not math.isfinite(float(risk_advantage_clip))
        or not 0.0 <= float(risk_advantage_clip) <= 1.0
    ):
        raise ValueError("risk_advantage_clip must be finite and in [0, 1]")
    if (
        isinstance(contrast_margin, bool)
        or not isinstance(contrast_margin, int | float)
        or not math.isfinite(float(contrast_margin))
        or float(contrast_margin) < 0.0
    ):
        raise ValueError("contrast_margin must be finite and non-negative")
    if (
        isinstance(risk_actor_min_selected_probability, bool)
        or not isinstance(risk_actor_min_selected_probability, int | float)
        or not math.isfinite(float(risk_actor_min_selected_probability))
        or not 0.0 <= float(risk_actor_min_selected_probability) < 1.0
    ):
        raise ValueError(
            "risk_actor_min_selected_probability must be finite and in [0, 1)"
        )

    forced = _optional_row_mask(
        forced_mask,
        rows=rows,
        device=device,
        label="forced_mask",
    )
    censored = _optional_row_mask(
        censored_mask,
        rows=rows,
        device=device,
        label="censored_mask",
    )
    critic_requested = _optional_row_mask(
        risk_critic_mask,
        rows=rows,
        device=device,
        label="risk_critic_mask",
    )
    value_requested = _optional_row_mask(
        value_critic_mask,
        rows=rows,
        device=device,
        label="value_critic_mask",
    )
    actor_requested = _optional_row_mask(
        risk_actor_mask,
        rows=rows,
        device=device,
        label="risk_actor_mask",
    )
    direct_requested = _optional_row_mask(
        direct_avoid_mask,
        rows=rows,
        device=device,
        label="direct_avoid_mask",
    )
    completion_requested = _optional_row_mask(
        completion_mask,
        rows=rows,
        device=device,
        label="completion_mask",
    )
    legal_counts = action_mask.sum(dim=1)
    actor_eligible = (~forced) & (~censored) & (legal_counts > 1)
    critic_eligible = critic_requested & (~censored)
    value_eligible = value_requested & (~censored)
    direct_eligible = direct_requested & actor_eligible
    completion_eligible = completion_requested & actor_eligible

    zero = (
        selected_log_probabilities.sum() * 0.0
        + candidate_liveness_cost_values.sum() * 0.0
        + (
            liveness_cost_values.sum() * 0.0
            if liveness_cost_values is not None
            else selected_log_probabilities.sum() * 0.0
        )
    )
    selected_costs = candidate_liveness_cost_values.gather(1, selected_action_indices[:, None]).squeeze(1)
    if risk_targets is None:
        targets = selected_costs.new_zeros(rows)
    else:
        if risk_targets.shape != (rows,) or not torch.is_floating_point(risk_targets):
            raise ValueError("risk_targets must be a rank-1 floating tensor")
        if risk_targets.device != device:
            raise ValueError(f"risk_targets must be on {device}")
        if not bool(torch.isfinite(risk_targets).all().item()):
            raise ValueError("risk_targets contains NaN or infinity")
        if bool(((risk_targets < 0.0) | (risk_targets > 1.0)).any().item()):
            raise ValueError("risk_targets must be in [0, 1]")
        targets = risk_targets
    if value_targets is None:
        state_targets = selected_costs.new_zeros(rows)
    else:
        if value_targets.shape != (rows,) or not torch.is_floating_point(value_targets):
            raise ValueError("value_targets must be a rank-1 floating tensor")
        if value_targets.device != device:
            raise ValueError(f"value_targets must be on {device}")
        if not bool(torch.isfinite(value_targets).all().item()):
            raise ValueError("value_targets contains NaN or infinity")
        if bool(((value_targets < 0.0) | (value_targets > 1.0)).any().item()):
            raise ValueError("value_targets must be in [0, 1]")
        state_targets = value_targets
    if bool(value_eligible.any().item()):
        if liveness_cost_values is None:
            raise ValueError("value critic labels require liveness_cost_values")
        value_critic_loss = F.binary_cross_entropy(
            liveness_cost_values[value_eligible].float().clamp(1e-6, 1.0 - 1e-6),
            state_targets[value_eligible].float(),
        )
    else:
        value_critic_loss = zero
    if bool(critic_eligible.any().item()):
        critic_loss = F.binary_cross_entropy(
            selected_costs[critic_eligible].float().clamp(1e-6, 1.0 - 1e-6),
            targets[critic_eligible].float(),
        )
    else:
        critic_loss = zero

    legal_probabilities = torch.where(
        action_mask,
        # The cost actor must use the current policy only as a detached
        # baseline distribution.  Its policy gradient is owned exclusively by
        # ``selected_log_probabilities`` below; otherwise the expectation term
        # leaks a second, critic-shaped gradient through every legal action.
        policy_log_probabilities.exp().detach(),
        torch.zeros_like(policy_log_probabilities),
    )
    centered_risk = (
        selected_costs.detach() - (legal_probabilities * candidate_liveness_cost_values.detach()).sum(dim=1)
    ).clamp(
        min=-float(risk_advantage_clip),
        max=float(risk_advantage_clip),
    )
    selected_probabilities = selected_log_probabilities.detach().exp()
    # A positive centered risk asks gradient descent to lower the factual
    # selected action.  Once that action is already below the reviewed floor,
    # repeating the same direction changes shared representations far more
    # than behaviour.  Keep critic labels and all negative-risk recovery
    # labels; suppress only the already-satisfied actor direction.
    saturation_suppressed = (
        actor_requested
        & actor_eligible
        & (centered_risk >= 0.0)
        & (
            selected_probabilities
            < float(risk_actor_min_selected_probability)
        )
    )
    risk_actor_eligible = actor_requested & actor_eligible & (~saturation_suppressed)
    if bool(risk_actor_eligible.any().item()):
        risk_actor_loss = (selected_log_probabilities[risk_actor_eligible] * centered_risk[risk_actor_eligible]).mean()
        active_centered = centered_risk[risk_actor_eligible]
        centered_risk_mean = float(active_centered.mean().item())
        centered_risk_max_abs = float(active_centered.abs().max().item())
    else:
        risk_actor_loss = zero
        centered_risk_mean = 0.0
        centered_risk_max_abs = 0.0

    direct_losses: list[Tensor] = []
    for row in torch.nonzero(direct_eligible, as_tuple=False).flatten().tolist():
        alternatives = action_mask[row].clone()
        alternatives[int(selected_action_indices[row].item())] = False
        alternative_log_mass = torch.logsumexp(
            policy_log_probabilities[row].masked_select(alternatives),
            dim=0,
        )
        # Raw -log(1-p_selected) has an unbounded derivative as the selected
        # action approaches probability one.  Weighting by the detached
        # factual alternative mass preserves the AVOID direction while making
        # a single sparse witness incapable of hijacking an update.
        alternative_mass = alternative_log_mass.detach().exp()
        direct_losses.append(-alternative_mass * alternative_log_mass)
    direct_avoid_loss = torch.stack(direct_losses).mean() if direct_losses else zero
    completion_loss = (
        -selected_log_probabilities[completion_eligible].mean() if bool(completion_eligible.any().item()) else zero
    )

    if cycle_behavior_mean_log_probabilities is not None and len(cycle_behavior_mean_log_probabilities) != len(
        cycle_groups
    ):
        raise ValueError("cycle behavior log-probabilities must align with cycle groups")
    if cycle_margins is not None and len(cycle_margins) != len(cycle_groups):
        raise ValueError("cycle margins must align with cycle groups")
    cycle_losses: list[Tensor] = []
    for cycle_index, group in enumerate(cycle_groups):
        if not isinstance(group, tuple) or not group:
            raise ValueError("cycle groups must be non-empty index tuples")
        if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < rows for index in group):
            raise ValueError("cycle group contains an invalid decision index")
        eligible_indices = [index for index in group if bool(actor_eligible[index].item())]
        if not eligible_indices:
            continue
        mean_log_probability = selected_log_probabilities[
            torch.tensor(eligible_indices, device=device, dtype=torch.long)
        ].mean()
        if cycle_behavior_mean_log_probabilities is None:
            cycle_probability = mean_log_probability.exp().clamp(max=1.0 - 1e-6)
            cycle_losses.append(-torch.log1p(-cycle_probability))
        else:
            behavior_mean = cycle_behavior_mean_log_probabilities[cycle_index]
            margin = cycle_margins[cycle_index] if cycle_margins is not None else float(contrast_margin)
            if not math.isfinite(behavior_mean) or behavior_mean > 0.0:
                raise ValueError("cycle behavior mean log-probabilities must be finite and non-positive")
            if not math.isfinite(margin) or margin < 0.0:
                raise ValueError("cycle margins must be finite and non-negative")
            # Reduce the complete cycle's geometric-mean likelihood below the
            # factual behavior policy by the compiled margin.  This objective
            # remains well scaled for long cycles and never singles out the
            # arbitrary action that happened to cross a stall threshold.
            cycle_losses.append(
                F.softplus(
                    mean_log_probability
                    - mean_log_probability.new_tensor(behavior_mean)
                    + mean_log_probability.new_tensor(margin)
                )
            )
    cycle_likelihood_loss = torch.stack(cycle_losses).mean() if cycle_losses else zero

    contrast_losses: list[Tensor] = []
    if contrast_margins is not None and len(contrast_margins) != len(contrast_pairs):
        raise ValueError("contrast margins must align with contrast pairs")
    for pair_index, pair in enumerate(contrast_pairs):
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < rows for index in pair)
        ):
            raise ValueError("contrast pairs must contain two valid decision indexes")
        better_index, worse_index = pair
        if not bool((actor_eligible[better_index] & actor_eligible[worse_index]).item()):
            continue
        pair_margin = contrast_margins[pair_index] if contrast_margins is not None else float(contrast_margin)
        if not math.isfinite(pair_margin) or pair_margin < 0.0:
            raise ValueError("contrast margins must be finite and non-negative")
        contrast_losses.append(
            F.softplus(
                selected_log_probabilities.new_tensor(float(pair_margin))
                - (selected_log_probabilities[better_index] - selected_log_probabilities[worse_index])
            )
        )
    contrast_loss = torch.stack(contrast_losses).mean() if contrast_losses else zero

    forced_suppressed = int(
        (
            (actor_requested & forced).sum() + (direct_requested & forced).sum() + (completion_requested & forced).sum()
        ).item()
    )
    censored_suppressed = int(
        (
            (value_requested & censored).sum()
            + (critic_requested & censored).sum()
            + (actor_requested & censored).sum()
            + (direct_requested & censored).sum()
            + (completion_requested & censored).sum()
        ).item()
    )
    return LivenessCreditLosses(
        value_critic_loss=value_critic_loss,
        critic_loss=critic_loss,
        risk_actor_loss=risk_actor_loss,
        direct_avoid_loss=direct_avoid_loss,
        cycle_likelihood_loss=cycle_likelihood_loss,
        contrast_loss=contrast_loss,
        completion_loss=completion_loss,
        value_labels=int(value_eligible.sum().item()),
        critic_labels=int(critic_eligible.sum().item()),
        risk_actor_labels=int(risk_actor_eligible.sum().item()),
        direct_avoid_labels=len(direct_losses),
        cycle_labels=len(cycle_losses),
        contrast_labels=len(contrast_losses),
        completion_labels=int(completion_eligible.sum().item()),
        forced_actor_suppressed_labels=forced_suppressed,
        censored_suppressed_labels=censored_suppressed,
        policy_lag_suppressed_labels=0,
        risk_actor_phase_suppressed_labels=0,
        risk_actor_saturation_suppressed_labels=int(
            saturation_suppressed.sum().item()
        ),
        centered_risk_mean=centered_risk_mean,
        centered_risk_max_abs=centered_risk_max_abs,
    )


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _require_finite(stage: str, values: tuple[tuple[str, Tensor], ...]) -> None:
    invalid = [name for name, value in values if not bool(torch.isfinite(value).all().item())]
    if invalid:
        raise FloatingPointError(f"non-finite learner {stage}: {', '.join(invalid)}")


def one_sided_policy_support_loss(
    selected_log_probability: Tensor,
    *,
    probability_floor: float,
) -> tuple[Tensor, Tensor]:
    """Recover finite policy support without creating permanent imitation.

    Working in log-probability space preserves the ordinary ``1 - pi(a|s)``
    logit gradient even after the selected softmax probability underflows.
    The loss becomes identically zero once the configured support floor is
    reached; factual return learning remains responsible for every preference
    above that floor.  The second result is the unreduced log-support gap used
    by telemetry and threshold tests.
    """

    if not isinstance(selected_log_probability, Tensor):
        raise TypeError("selected_log_probability must be a tensor")
    if not math.isfinite(probability_floor) or not 0.0 < probability_floor < 1.0:
        raise ValueError("probability_floor must be finite and strictly between 0 and 1")
    log_probability = selected_log_probability.float()
    if not bool(torch.isfinite(log_probability).all().item()):
        raise FloatingPointError("selected policy log probability is non-finite")
    log_floor = log_probability.new_tensor(math.log(probability_floor))
    log_gap = torch.relu(log_floor - log_probability)
    loss = F.smooth_l1_loss(
        log_gap,
        torch.zeros_like(log_gap),
        reduction="mean",
    )
    return loss, log_gap


def two_sided_policy_support_loss(
    selected_log_probability: Tensor,
    alternative_log_probability: Tensor,
    *,
    probability_floor: float,
) -> tuple[Tensor, Tensor]:
    """Keep both an action and its legal complement out of absorption.

    This loss is preference-free: it is zero whenever the selected action and
    the aggregate probability of every other legal action are both at least
    ``probability_floor``. Outside that interval it restores only missing
    numerical support; factual return learning decides which side should be
    larger in a given state.
    """

    if not math.isfinite(probability_floor) or not 0.0 < probability_floor < 0.5:
        raise ValueError(
            "two-sided probability_floor must be finite and strictly between 0 and 0.5"
        )
    selected_loss, selected_gap = one_sided_policy_support_loss(
        selected_log_probability,
        probability_floor=probability_floor,
    )
    alternative_loss, alternative_gap = one_sided_policy_support_loss(
        alternative_log_probability,
        probability_floor=probability_floor,
    )
    return selected_loss + alternative_loss, torch.maximum(
        selected_gap,
        alternative_gap,
    )


def _annealed_entropy_weight(
    config: OptimizationConfig,
    *,
    policy_version: int,
) -> float:
    """Return the explicit non-zero entropy schedule for this update."""

    start = float(config.entropy_weight)
    end = float(config.entropy_weight_end)
    decay_updates = int(config.entropy_decay_updates)
    progress = min(1.0, max(0.0, float(policy_version) / float(decay_updates)))
    return start + progress * (end - start)


@dataclass(frozen=True, slots=True)
class _EpisodicLossBatch:
    total_loss: Tensor
    primary_policy_loss: Tensor
    task_value_loss: Tensor
    revival_value_loss: Tensor
    revival_policy_loss: Tensor
    act_segment_policy_loss: Tensor
    combat_hp_loss_value_loss: Tensor
    burn_in_steps: int
    learn_steps: int
    success_policy_candidate_labels: int
    policy_labels: int
    policy_active_sequences: int
    failure_policy_suppressed_labels: int
    policy_lag_suppressed_labels: int
    success_trust_region_suppressed_labels: int
    success_surface_exempted_labels: int
    act_segment_policy_labels: int
    act_segment_health_gate_suppressed_labels: int
    act_segment_healthy_acts: int
    task_value_labels: int
    combat_hp_loss_value_labels: int
    revival_value_labels: int
    efficiency_policy_labels: int
    importance_ratios: Tensor
    maximum_policy_lag: int


@dataclass(frozen=True, slots=True)
class _LivenessReplayDecision:
    policy_log_probabilities: Tensor
    candidate_cost_values: Tensor
    state_cost_value: Tensor
    action_mask: Tensor
    selected_action_index: int
    forced: bool


def _empty_liveness_credit_losses(reference: Tensor) -> LivenessCreditLosses:
    zero = reference.sum() * 0.0
    return LivenessCreditLosses(
        value_critic_loss=zero,
        critic_loss=zero,
        risk_actor_loss=zero,
        direct_avoid_loss=zero,
        cycle_likelihood_loss=zero,
        contrast_loss=zero,
        completion_loss=zero,
        value_labels=0,
        critic_labels=0,
        risk_actor_labels=0,
        direct_avoid_labels=0,
        cycle_labels=0,
        contrast_labels=0,
        completion_labels=0,
        forced_actor_suppressed_labels=0,
        censored_suppressed_labels=0,
        policy_lag_suppressed_labels=0,
        risk_actor_phase_suppressed_labels=0,
        risk_actor_saturation_suppressed_labels=0,
        centered_risk_mean=0.0,
        centered_risk_max_abs=0.0,
    )


def _mean_liveness_credit_losses(
    batches: tuple[LivenessCreditLosses, ...],
    *,
    reference: Tensor,
    weights: tuple[int, ...] | None = None,
    retain_graph: bool = False,
) -> LivenessCreditLosses:
    """Aggregate formal record means and additive telemetry.

    ``weights`` describes how many equally weighted evidence records each
    input already represents.  Loss tensors are therefore weighted means,
    while label/work counters remain additive.  Production metric aggregation
    detaches tensors by default; the recurrent packer opts into
    ``retain_graph`` so one packed replay can backpropagate the exact same
    equal-record objective as the original record-at-a-time executor.
    """

    if not batches:
        return _empty_liveness_credit_losses(reference)
    if weights is None:
        weights = (1,) * len(batches)
    if len(weights) != len(batches):
        raise ValueError("liveness aggregation weights must align with batches")
    if any(isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0 for weight in weights):
        raise ValueError("liveness aggregation weights must be positive integers")
    total_weight = float(sum(weights))

    def mean_tensor(name: str) -> Tensor:
        values = tuple(getattr(batch, name) for batch in batches)
        if not retain_graph:
            values = tuple(value.detach() for value in values)
        return (
            torch.stack(tuple(value * float(weight) for value, weight in zip(values, weights, strict=True))).sum()
            / total_weight
        )

    risk_labels = sum(batch.risk_actor_labels for batch in batches)
    centered_risk_mean = (
        sum(batch.centered_risk_mean * batch.risk_actor_labels for batch in batches) / risk_labels
        if risk_labels
        else 0.0
    )
    return LivenessCreditLosses(
        value_critic_loss=mean_tensor("value_critic_loss"),
        critic_loss=mean_tensor("critic_loss"),
        risk_actor_loss=mean_tensor("risk_actor_loss"),
        direct_avoid_loss=mean_tensor("direct_avoid_loss"),
        cycle_likelihood_loss=mean_tensor("cycle_likelihood_loss"),
        contrast_loss=mean_tensor("contrast_loss"),
        completion_loss=mean_tensor("completion_loss"),
        value_labels=sum(batch.value_labels for batch in batches),
        critic_labels=sum(batch.critic_labels for batch in batches),
        risk_actor_labels=risk_labels,
        direct_avoid_labels=sum(batch.direct_avoid_labels for batch in batches),
        cycle_labels=sum(batch.cycle_labels for batch in batches),
        contrast_labels=sum(batch.contrast_labels for batch in batches),
        completion_labels=sum(batch.completion_labels for batch in batches),
        forced_actor_suppressed_labels=sum(batch.forced_actor_suppressed_labels for batch in batches),
        censored_suppressed_labels=sum(batch.censored_suppressed_labels for batch in batches),
        policy_lag_suppressed_labels=sum(batch.policy_lag_suppressed_labels for batch in batches),
        risk_actor_phase_suppressed_labels=sum(batch.risk_actor_phase_suppressed_labels for batch in batches),
        risk_actor_saturation_suppressed_labels=sum(
            batch.risk_actor_saturation_suppressed_labels for batch in batches
        ),
        centered_risk_mean=centered_risk_mean,
        centered_risk_max_abs=max(batch.centered_risk_max_abs for batch in batches),
        replayed_contexts=sum(batch.replayed_contexts for batch in batches),
        replayed_steps=sum(batch.replayed_steps for batch in batches),
        replayed_candidates=sum(batch.replayed_candidates for batch in batches),
        autograd_segments=sum(batch.autograd_segments for batch in batches),
    )


def _parameter_gradient_snapshot(
    parameters: tuple[Tensor, ...],
) -> tuple[Tensor | None, ...]:
    return tuple(parameter.grad.detach().clone() if parameter.grad is not None else None for parameter in parameters)


def _parameter_gradient_delta_norm(
    parameters: tuple[Tensor, ...],
    before: tuple[Tensor | None, ...],
) -> float:
    if len(parameters) != len(before):
        raise ValueError("gradient snapshot differs from parameter group")
    squared_norm = torch.zeros((), dtype=torch.float64)
    for parameter, previous in zip(parameters, before, strict=True):
        current = parameter.grad
        if current is None:
            continue
        delta = current.detach().float()
        if previous is not None:
            delta = delta - previous.to(device=delta.device, dtype=delta.dtype)
        squared_norm += delta.square().sum().cpu().double()
    if not bool(torch.isfinite(squared_norm).item()):
        raise FloatingPointError("non-finite liveness objective gradient norm")
    return float(squared_norm.sqrt().item())


def _clip_parameter_gradient_delta_(
    parameters: tuple[Tensor, ...],
    before: tuple[Tensor | None, ...],
    *,
    maximum_norm: float,
) -> tuple[float, float, float]:
    """Clip only the gradient added since ``before`` and preserve base work.

    Failure-credit replay is an auxiliary, sparse objective.  Applying only the
    final global clip lets one unusually sharp witness scale down the unrelated
    FIFO V-trace gradient.  This helper gives the complete liveness family its
    own budget while leaving the already accumulated base gradient unchanged.
    Callers must remove this already-clipped delta before clipping the base
    objective, then add it back.  That keeps the two budgets independent.
    """

    if len(parameters) != len(before):
        raise ValueError("gradient snapshot differs from parameter group")
    if not math.isfinite(maximum_norm) or maximum_norm <= 0.0:
        raise ValueError("maximum gradient-delta norm must be positive and finite")
    raw_norm = _parameter_gradient_delta_norm(parameters, before)
    scale = min(1.0, maximum_norm / max(raw_norm, 1.0e-12))
    if scale < 1.0:
        with torch.no_grad():
            for parameter, previous in zip(parameters, before, strict=True):
                current = parameter.grad
                if current is None:
                    if previous is not None:  # pragma: no cover - autograd cannot remove a gradient
                        raise RuntimeError("gradient disappeared after auxiliary backward")
                    continue
                if previous is None:
                    current.mul_(scale)
                    continue
                base = previous.to(device=current.device, dtype=current.dtype)
                current.copy_(base + (current - base) * scale)
    applied_norm = _parameter_gradient_delta_norm(parameters, before)
    return raw_norm, applied_norm, scale


def _parameter_gradient_delta_snapshot(
    parameters: tuple[Tensor, ...],
    before: tuple[Tensor | None, ...],
) -> tuple[Tensor | None, ...]:
    if len(parameters) != len(before):
        raise ValueError("gradient snapshot differs from parameter group")
    deltas: list[Tensor | None] = []
    for parameter, previous in zip(parameters, before, strict=True):
        current = parameter.grad
        if current is None:
            deltas.append(None)
            continue
        delta = current.detach().clone()
        if previous is not None:
            delta.sub_(previous.to(device=delta.device, dtype=delta.dtype))
        deltas.append(delta)
    return tuple(deltas)


def _apply_parameter_gradient_delta_(
    parameters: tuple[Tensor, ...],
    deltas: tuple[Tensor | None, ...],
    *,
    sign: float,
) -> None:
    if len(parameters) != len(deltas):
        raise ValueError("gradient delta differs from parameter group")
    if sign not in {-1.0, 1.0}:
        raise ValueError("gradient delta sign must be -1 or +1")
    with torch.no_grad():
        for parameter, delta in zip(parameters, deltas, strict=True):
            if delta is None:
                continue
            if parameter.grad is None:
                if sign < 0.0:  # pragma: no cover - autograd cannot remove it
                    raise RuntimeError("gradient disappeared before delta removal")
                parameter.grad = delta.clone()
            else:
                parameter.grad.add_(delta, alpha=sign)


class VTraceLearner:
    """Consume main-policy unrolls FIFO with an optional factual transaction sidecar."""

    def __init__(
        self,
        *,
        model: RecurrentCandidateModel,
        encoder: GroundedObservationEncoder,
        optimizer: torch.optim.Optimizer,
        config: OptimizationConfig,
        maximum_unroll_length: int,
        maximum_policy_lag: int,
        transaction_config: TransactionLearningConfig | None = None,
        failure_credit_config: FailureCreditConfig | None = None,
        episodic_config: EpisodicLearningConfig | None = None,
    ) -> None:
        if maximum_unroll_length <= 0 or maximum_policy_lag <= 0:
            raise ValueError("unroll length and policy lag limits must be positive")
        self.model = model
        self.encoder = encoder
        self.optimizer = optimizer
        self.config = config
        self.maximum_unroll_length = maximum_unroll_length
        self.maximum_policy_lag = maximum_policy_lag
        self.transaction_config = transaction_config or TransactionLearningConfig()
        self.failure_credit_config = failure_credit_config or FailureCreditConfig()
        self.episodic_config = episodic_config or EpisodicLearningConfig()
        if self.transaction_config.enabled != self.model.transaction_heads_enabled:
            raise ValueError("transaction learner config and model-head configuration differ")
        if self.failure_credit_config.learning_enabled != self.model.liveness_head_enabled:
            raise ValueError("liveness-credit config and model-head configuration differ")
        self._collapse_batch_streak = 0
        self._entropy_breaker_remaining_updates = 0
        self._entropy_breaker_triggers = 0

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def dynamics_state_dict(self) -> dict[str, int | str]:
        """Return non-parameter learner dynamics required for exact resume."""

        return {
            "version": _LEARNER_DYNAMICS_STATE_VERSION,
            "collapse_batch_streak": self._collapse_batch_streak,
            "entropy_breaker_remaining_updates": self._entropy_breaker_remaining_updates,
            "entropy_breaker_triggers": self._entropy_breaker_triggers,
        }

    @staticmethod
    def validate_dynamics_state_dict(payload: object) -> dict[str, int | str]:
        if not isinstance(payload, dict):
            raise TypeError("learner dynamics state must be an object")
        expected = {
            "version",
            "collapse_batch_streak",
            "entropy_breaker_remaining_updates",
            "entropy_breaker_triggers",
        }
        if set(payload) != expected:
            raise ValueError("learner dynamics state keys mismatch")
        if payload["version"] != _LEARNER_DYNAMICS_STATE_VERSION:
            raise ValueError("unsupported learner dynamics state")
        values: dict[str, int] = {}
        for key in expected - {"version"}:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"learner dynamics {key} must be a non-negative integer")
            values[key] = value
        if values["collapse_batch_streak"] >= max(
            _ONE_HOT_BREAKER_CONSECUTIVE_BATCHES_V1,
            _POLICY_COLLAPSE_BREAKER_CONSECUTIVE_BATCHES_V2,
        ):
            raise ValueError("learner collapse streak must be below its trigger threshold")
        if values["entropy_breaker_remaining_updates"] > max(
            _ONE_HOT_BREAKER_DURATION_UPDATES_V1,
            _POLICY_COLLAPSE_BREAKER_DURATION_UPDATES_V2,
        ):
            raise ValueError("learner entropy-breaker duration exceeds the source contract")
        return {
            "version": _LEARNER_DYNAMICS_STATE_VERSION,
            **values,
        }

    def load_dynamics_state_dict(self, payload: object) -> None:
        validated = self.validate_dynamics_state_dict(payload)
        self._collapse_batch_streak = int(validated["collapse_batch_streak"])
        self._entropy_breaker_remaining_updates = int(validated["entropy_breaker_remaining_updates"])
        self._entropy_breaker_triggers = int(validated["entropy_breaker_triggers"])

    def _entropy_weight_for_batch(
        self,
        *,
        base_weight: float,
        active_ratios: Tensor,
        clipped: Tensor,
        active_normalized_entropies: Tensor | None = None,
    ) -> tuple[float, bool, bool, bool]:
        """Apply the selected source-versioned policy-collapse breaker."""

        if self.config.entropy_breaker == "disabled":
            self._collapse_batch_streak = 0
            self._entropy_breaker_remaining_updates = 0
            return base_weight, False, False, False
        if active_ratios.ndim != 1 or clipped.shape != active_ratios.shape:
            raise ValueError("entropy breaker ratios and clip mask must be aligned vectors")
        if active_normalized_entropies is not None:
            if (
                active_normalized_entropies.ndim != 1
                or active_normalized_entropies.shape != active_ratios.shape
                or not bool(torch.isfinite(active_normalized_entropies).all().item())
                or bool(
                    (
                        (active_normalized_entropies < 0.0)
                        | (active_normalized_entropies > 1.0 + 1.0e-6)
                    ).any().item()
                )
            ):
                raise ValueError(
                    "entropy breaker normalized entropies must be finite aligned values in [0, 1]"
                )
        maximum_ratio = float(active_ratios.detach().max().item()) if active_ratios.numel() else 0.0
        clip_fraction = float(clipped.detach().mean().item()) if clipped.numel() else 0.0
        one_hot_condition = bool(
            active_ratios.numel()
            and math.isclose(
                maximum_ratio,
                1.0,
                rel_tol=0.0,
                abs_tol=_ONE_HOT_BREAKER_RATIO_TOLERANCE_V1,
            )
            and clip_fraction == 0.0
        )
        soft_collapse_condition = False
        if self.config.entropy_breaker == "policy-collapse-v2":
            if active_normalized_entropies is None:
                raise ValueError(
                    "policy-collapse-v2 requires normalized policy entropy"
                )
            normalized = active_normalized_entropies.detach()
            soft_collapse_condition = bool(
                normalized.numel()
                >= _POLICY_COLLAPSE_BREAKER_MINIMUM_DECISIONS_V2
                and float(normalized.mean().item())
                <= _POLICY_COLLAPSE_BREAKER_NORMALIZED_ENTROPY_V2
                and float(
                    (
                        normalized
                        <= _POLICY_COLLAPSE_BREAKER_NORMALIZED_ENTROPY_V2
                    )
                    .float()
                    .mean()
                    .item()
                )
                >= _POLICY_COLLAPSE_BREAKER_LOW_ENTROPY_FRACTION_V2
            )
            # V2 intentionally keys the breaker to policy entropy rather than
            # the legacy importance-ratio heuristic.  Fresh on-policy batches
            # commonly have rho == 1 with no clipping even when their action
            # distribution is healthy, so OR-ing that condition into V2 would
            # turn the circuit breaker on during ordinary learning.  Keep the
            # legacy observation as telemetry, but only entropy collapse may
            # trigger the V2 intervention.
            condition = soft_collapse_condition
            consecutive_batches = _POLICY_COLLAPSE_BREAKER_CONSECUTIVE_BATCHES_V2
            duration_updates = _POLICY_COLLAPSE_BREAKER_DURATION_UPDATES_V2
            breaker_weight = _POLICY_COLLAPSE_BREAKER_ENTROPY_WEIGHT_V2
        elif self.config.entropy_breaker == "one-hot-v1":
            condition = one_hot_condition
            consecutive_batches = _ONE_HOT_BREAKER_CONSECUTIVE_BATCHES_V1
            duration_updates = _ONE_HOT_BREAKER_DURATION_UPDATES_V1
            breaker_weight = _ONE_HOT_BREAKER_ENTROPY_WEIGHT_V1
        else:  # pragma: no cover - config validates
            raise RuntimeError("unsupported entropy breaker")
        self._collapse_batch_streak = self._collapse_batch_streak + 1 if condition else 0
        if self._collapse_batch_streak >= consecutive_batches:
            self._collapse_batch_streak = 0
            self._entropy_breaker_remaining_updates = duration_updates
            self._entropy_breaker_triggers += 1
        active = self._entropy_breaker_remaining_updates > 0
        effective = max(base_weight, breaker_weight) if active else base_weight
        if active:
            self._entropy_breaker_remaining_updates -= 1
        return effective, active, one_hot_condition, soft_collapse_condition

    def credit_plan_liveness_losses(
        self,
        credit_plans: tuple[CreditPlan, ...],
        *,
        current_policy_version: int,
        current_learner_update: int,
    ) -> LivenessCreditLosses:
        """Replay typed v4 credit plans into the independent liveness plane.

        The adapter consumes only immutable :class:`CreditPlan` facts.  It
        neither reconstructs loops from a truncated tail nor reaches into
        collector/runtime state.  Each recurrent context is replayed from its
        owned initial state, with the declared prefix under ``no_grad``.
        Contexts are batched by local timestep so recurrent replay costs scale
        with the longest referenced prefix rather than the sum of all context
        lengths.  Burn-in and trainable rows use separate active-shape model
        invocations, preserving both the recurrent graph boundary and
        candidate-shape efficiency.

        The current model exposes separate bounded state-value and
        action-conditioned Q heads.  Their formally separate target tuples are
        consumed independently even when today's compiler happens to emit
        numerically identical targets.
        """

        if not isinstance(credit_plans, tuple) or not all(isinstance(plan, CreditPlan) for plan in credit_plans):
            raise TypeError("credit_plans must be a CreditPlan tuple")
        if (
            isinstance(current_policy_version, bool)
            or not isinstance(current_policy_version, int)
            or current_policy_version < 0
        ):
            raise ValueError("current_policy_version must be a non-negative integer")
        if (
            isinstance(current_learner_update, bool)
            or not isinstance(current_learner_update, int)
            or current_learner_update < 0
        ):
            raise ValueError("current_learner_update must be a non-negative integer")
        if credit_plans and not self.failure_credit_config.learning_enabled:
            raise ValueError("credit plans require formal liveness learning to be enabled")
        if len(credit_plans) > self.failure_credit_config.sample_records:
            raise ValueError("credit plan batch exceeds failure_credit.sample_records")
        reference = next(self.model.parameters())
        if not credit_plans:
            return _empty_liveness_credit_losses(reference)
        if self.model.candidate_liveness_cost_head is None:
            raise RuntimeError("enabled liveness learning has no model cost head")
        # Compile one manifest per evidence record. Model execution may be
        # shared across records, but target reduction remains local to each
        # record before the formal equal-record mean is taken.
        label_manifests = tuple(
            compile_liveness_label_manifest(
                (credit_plan,),
                config=self.failure_credit_config,
                current_policy_version=current_policy_version,
                current_learner_update=current_learner_update,
            )
            for credit_plan in credit_plans
        )
        admitted_steps = sum(manifest.work.steps for manifest in label_manifests)
        admitted_candidates = sum(manifest.work.candidates for manifest in label_manifests)
        if admitted_steps > self.failure_credit_config.liveness_maximum_replayed_steps_per_update:
            raise ValueError("failure-credit replay exceeds " "liveness_maximum_replayed_steps_per_update")
        if admitted_candidates > self.failure_credit_config.liveness_maximum_replayed_candidates_per_update:
            raise ValueError("failure-credit replay exceeds " "liveness_maximum_replayed_candidates_per_update")
        calibration_active = label_manifests[0].calibration_active
        if any(manifest.calibration_active is not calibration_active for manifest in label_manifests):
            raise RuntimeError("one liveness autograd batch crossed a calibration phase boundary")

        contexts: dict[int, LearningContext] = {}
        context_ids: dict[str, int] = {}
        needed_steps: dict[int, set[int]] = {}

        def register_context_step(
            context: LearningContext,
            step_index: int,
        ) -> tuple[int, int]:
            if not context.burn_in_steps <= step_index < len(context.steps):
                raise ValueError("liveness target must reference a post-burn-in learning step")
            identity = id(context)
            previous_identity = context_ids.get(context.context_id)
            if previous_identity is not None and previous_identity != identity:
                raise ValueError("different LearningContext objects reuse one context_id")
            context_ids[context.context_id] = identity
            contexts[identity] = context
            needed_steps.setdefault(identity, set()).add(step_index)
            return identity, step_index

        for plan in credit_plans:
            if plan.provenance.policy_version > current_policy_version:
                raise ValueError("credit-plan policy version is newer than the learner")
            if EvidenceStratum.CENSORED in plan.strata and plan.strata != (EvidenceStratum.CENSORED,):
                raise ValueError("CENSORED cannot be mixed with authoritative credit strata")
            for value_target in plan.liveness_value_targets:
                register_context_step(plan.context, value_target.step_index)
            for q_target in plan.liveness_q_targets:
                register_context_step(plan.context, q_target.step_index)
            for direct_target in plan.direct_policy_targets:
                register_context_step(plan.context, direct_target.step_index)
            for cycle_target in plan.cycle_policy_targets:
                for step_index in cycle_target.step_indices:
                    register_context_step(plan.context, step_index)
            for risk_sequence in plan.risk_sequences:
                for step_index in risk_sequence.step_indices:
                    register_context_step(plan.context, step_index)
            for contrast_target in plan.contrast_policy_targets:
                register_context_step(
                    contrast_target.pair.better.context,
                    contrast_target.pair.better.step_index,
                )
                register_context_step(
                    contrast_target.pair.worse.context,
                    contrast_target.pair.worse.step_index,
                )

        if not needed_steps:
            return _empty_liveness_credit_losses(reference)

        encoding_fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        replayed: dict[tuple[int, int], _LivenessReplayDecision] = {}
        maximum_replayed_step = {identity: max(step_indices) for identity, step_indices in needed_steps.items()}
        ordered_context_identities = tuple(identity for identity in contexts if identity in needed_steps)
        hidden_by_context: dict[int, Tensor] = {}
        for identity in ordered_context_identities:
            context = contexts[identity]
            if context.initial_recurrent_state.shape != (self.model.config.recurrent_hidden_dim,):
                raise ValueError("liveness context recurrent state differs from model hidden size")
            hidden_by_context[identity] = torch.from_numpy(context.initial_recurrent_state.copy()).to(
                device=self.device,
                dtype=reference.dtype,
            )[None, :]

        was_training = self.model.training
        self.model.eval()
        try:

            def replay_timestep_rows(
                identities: tuple[int, ...],
                *,
                step_index: int,
                track_gradients: bool,
            ) -> None:
                if not identities:
                    return
                steps = tuple(contexts[identity].steps[step_index] for identity in identities)
                for step in steps:
                    if step.policy_version > current_policy_version:
                        raise ValueError("liveness learning step is newer than the learner")
                encoded = collate_encoded_snapshots(
                    tuple(step.snapshot for step in steps),
                    expected_config=self.encoder.config,
                    expected_fingerprint=encoding_fingerprint,
                    device=self.device,
                )
                hidden_rows = tuple(
                    (
                        hidden_by_context[identity].detach()
                        if track_gradients and step_index == contexts[identity].burn_in_steps
                        else hidden_by_context[identity]
                    )
                    for identity in identities
                )
                hidden = torch.cat(hidden_rows, dim=0)
                if track_gradients:
                    output = self.model(
                        encoded,
                        hidden,
                        validate=False,
                        detach_liveness_shared_features=calibration_active,
                    )
                else:
                    with torch.no_grad():
                        output = self.model(
                            encoded,
                            hidden,
                            validate=False,
                            detach_liveness_shared_features=calibration_active,
                        )
                if output.recurrent_state.shape != (
                    len(identities),
                    self.model.config.recurrent_hidden_dim,
                ):
                    raise RuntimeError("liveness recurrent replay returned an invalid hidden shape")
                policy_log_probabilities: Tensor | None = None
                costs = output.candidate_liveness_cost_values
                state_cost = output.liveness_cost_value
                for row, (identity, step) in enumerate(zip(identities, steps, strict=True)):
                    hidden_by_context[identity] = output.recurrent_state[row : row + 1]
                    if step_index not in needed_steps[identity]:
                        continue
                    if costs is None or state_cost is None:
                        raise RuntimeError("liveness model output omitted value/Q costs")
                    if policy_log_probabilities is None:
                        policy_log_probabilities = output.policy_log_probabilities()
                    candidate_count = step.snapshot.candidate_count
                    replayed[(identity, step_index)] = _LivenessReplayDecision(
                        policy_log_probabilities=(policy_log_probabilities[row, :candidate_count]),
                        candidate_cost_values=costs[row, :candidate_count],
                        state_cost_value=state_cost[row],
                        action_mask=encoded.candidates.action_mask[row, :candidate_count],
                        selected_action_index=step.action_index,
                        forced=step.forced,
                    )

            longest_replayed_prefix = max(maximum_replayed_step.values())
            for step_index in range(longest_replayed_prefix + 1):
                active = tuple(
                    identity for identity in ordered_context_identities if step_index <= maximum_replayed_step[identity]
                )
                burn_in = tuple(identity for identity in active if step_index < contexts[identity].burn_in_steps)
                trainable = tuple(identity for identity in active if step_index >= contexts[identity].burn_in_steps)
                for identity in trainable:
                    context = contexts[identity]
                    trainable_offset = step_index - context.burn_in_steps
                    if trainable_offset % self.failure_credit_config.liveness_tbptt_window_steps == 0:
                        hidden_by_context[identity] = hidden_by_context[identity].detach()
                replay_timestep_rows(
                    burn_in,
                    step_index=step_index,
                    track_gradients=False,
                )
                replay_timestep_rows(
                    trainable,
                    step_index=step_index,
                    track_gradients=True,
                )
        finally:
            self.model.train(was_training)

        expected_keys = {
            (identity, step_index) for identity in contexts for step_index in needed_steps.get(identity, ())
        }
        if expected_keys != set(replayed):
            raise RuntimeError("liveness recurrent replay lost a referenced decision")

        def record_losses(
            plan: CreditPlan,
            manifest: LivenessLabelManifest,
        ) -> LivenessCreditLosses:
            """Reduce one record over outputs produced by the shared replay."""

            if not manifest.rows:
                return replace(
                    _empty_liveness_credit_losses(reference),
                    policy_lag_suppressed_labels=manifest.policy_lag_suppressed_labels,
                    risk_actor_phase_suppressed_labels=(manifest.risk_actor_phase_suppressed_labels),
                    replayed_contexts=manifest.work.contexts,
                    replayed_steps=manifest.work.steps,
                    replayed_candidates=manifest.work.candidates,
                    autograd_segments=manifest.work.autograd_segments,
                )
            record_keys: list[tuple[int, int]] = []
            for manifest_row in manifest.rows:
                identity = context_ids.get(manifest_row.context_id)
                if identity is None:  # pragma: no cover - compiler/replay invariant
                    raise RuntimeError("compiled liveness context was not registered for replay")
                record_keys.append((identity, manifest_row.step_index))
            decisions = tuple(replayed[key] for key in record_keys)
            maximum_candidates = max(decision.policy_log_probabilities.shape[0] for decision in decisions)

            def padded(value: Tensor, *, fill: float) -> Tensor:
                missing = maximum_candidates - value.shape[0]
                return F.pad(value, (0, missing), value=fill) if missing else value

            policy_log_probabilities = torch.stack(
                tuple(padded(decision.policy_log_probabilities, fill=-torch.inf) for decision in decisions)
            )
            candidate_cost_values = torch.stack(
                tuple(padded(decision.candidate_cost_values, fill=0.0) for decision in decisions)
            )
            state_cost_values = torch.stack(tuple(decision.state_cost_value for decision in decisions))
            action_mask = torch.stack(
                tuple(
                    F.pad(
                        decision.action_mask,
                        (0, maximum_candidates - decision.action_mask.shape[0]),
                        value=False,
                    )
                    for decision in decisions
                )
            )
            selected_action_indices = torch.tensor(
                tuple(decision.selected_action_index for decision in decisions),
                device=self.device,
                dtype=torch.long,
            )
            forced_mask = torch.tensor(
                tuple(decision.forced for decision in decisions),
                device=self.device,
                dtype=torch.bool,
            )
            rows = len(decisions)
            # Targets and masks are immutable compiler facts. Assemble them on
            # the host and transfer each vector once; probing GPU tensors with
            # ``.item()`` inside a 256-step Python loop serialized ROCm replay.
            censored_values = [False] * rows
            risk_target_values = [0.0] * rows
            value_target_values = [0.0] * rows
            value_critic_values = [False] * rows
            risk_critic_values = [False] * rows
            risk_actor_values = [False] * rows
            direct_avoid_values = [False] * rows
            completion_values = [False] * rows
            direct_targets: dict[int, DirectPolicyTarget] = {}
            cycle_groups: list[tuple[int, ...]] = []
            cycle_behavior: list[float] = []
            cycle_margins: list[float] = []
            contrast_pairs: list[tuple[int, int]] = []
            contrast_margins: list[float] = []
            manifest_rows = manifest.row_map()
            row_index = {row.key: index for index, row in enumerate(manifest.rows)}

            def decision_row(context: LearningContext, step_index: int) -> int:
                try:
                    return row_index[(context.context_id, step_index)]
                except KeyError as error:  # pragma: no cover - internal invariant
                    raise RuntimeError("compiled liveness target was not recurrently replayed") from error

            def merge_risk_target(row: int, target: float) -> None:
                value = float(target)
                if not 0.0 <= value <= 1.0 or not math.isfinite(value):
                    raise ValueError("compiled liveness target must be in [0, 1]")
                if risk_critic_values[row]:
                    previous = risk_target_values[row]
                    if not math.isclose(previous, value, rel_tol=1e-6, abs_tol=1e-6):
                        raise ValueError("conflicting liveness targets reference one decision")
                    return
                risk_target_values[row] = value
                risk_critic_values[row] = True

            def merge_value_target(row: int, target: float) -> None:
                value = float(target)
                if not 0.0 <= value <= 1.0 or not math.isfinite(value):
                    raise ValueError("compiled liveness value target must be in [0, 1]")
                if value_critic_values[row]:
                    previous = value_target_values[row]
                    if not math.isclose(previous, value, rel_tol=1e-6, abs_tol=1e-6):
                        raise ValueError("conflicting liveness value targets reference one decision")
                    return
                value_target_values[row] = value
                value_critic_values[row] = True

            plan_censored = EvidenceStratum.CENSORED in plan.strata
            for value_target in plan.liveness_value_targets:
                row = decision_row(plan.context, value_target.step_index)
                merge_value_target(row, value_target.target)
                if plan_censored:
                    censored_values[row] = True
            for q_target in plan.liveness_q_targets:
                row = decision_row(plan.context, q_target.step_index)
                merge_risk_target(row, q_target.target)
                if plan_censored:
                    censored_values[row] = True
            for sequence in plan.risk_sequences:
                sequence_length = len(sequence.step_indices)
                for ordinal, step_index in enumerate(sequence.step_indices):
                    row = decision_row(plan.context, step_index)
                    horizon = sequence_length - ordinal
                    merge_risk_target(
                        row,
                        sequence.terminal_cost * sequence.discount ** (horizon - 1),
                    )
                    manifest_row = manifest_rows[(plan.context.context_id, step_index)]
                    if manifest_row.risk_actor_mask:
                        risk_actor_values[row] = True
                    if plan_censored:
                        censored_values[row] = True
            for direct_target in plan.direct_policy_targets:
                row = decision_row(plan.context, direct_target.step_index)
                manifest_row = manifest_rows[(plan.context.context_id, direct_target.step_index)]
                if not manifest_row.direct_actor_mask:
                    continue
                previous = direct_targets.get(row)
                if previous is not None and previous is not direct_target.target:
                    raise ValueError("conflicting direct policy targets reference one decision")
                direct_targets[row] = direct_target.target
                if direct_target.target is DirectPolicyTarget.AVOID:
                    direct_avoid_values[row] = True
                elif direct_target.target is DirectPolicyTarget.PREFER:
                    completion_values[row] = True
                else:  # pragma: no cover - enum exhaustiveness
                    raise RuntimeError("unsupported direct liveness target")
                if plan_censored:
                    censored_values[row] = True
            for group_index, cycle_target in enumerate(plan.cycle_policy_targets):
                if not manifest.cycle_groups[group_index].effective:
                    continue
                cycle_groups.append(
                    tuple(decision_row(plan.context, step_index) for step_index in cycle_target.step_indices)
                )
                cycle_behavior.append(cycle_target.behavior_mean_log_probability)
                cycle_margins.append(cycle_target.margin)
            for group_index, contrast_target in enumerate(plan.contrast_policy_targets):
                if not manifest.contrast_groups[group_index].effective:
                    continue
                contrast_pairs.append(
                    (
                        decision_row(
                            contrast_target.pair.better.context,
                            contrast_target.pair.better.step_index,
                        ),
                        decision_row(
                            contrast_target.pair.worse.context,
                            contrast_target.pair.worse.step_index,
                        ),
                    )
                )
                contrast_margins.append(contrast_target.margin)

            def bool_tensor(values: list[bool]) -> Tensor:
                return torch.tensor(values, device=self.device, dtype=torch.bool)

            losses = liveness_credit_losses(
                policy_log_probabilities=policy_log_probabilities,
                candidate_liveness_cost_values=candidate_cost_values,
                liveness_cost_values=state_cost_values,
                action_mask=action_mask,
                selected_action_indices=selected_action_indices,
                value_targets=torch.tensor(
                    value_target_values,
                    device=self.device,
                    dtype=candidate_cost_values.dtype,
                ),
                value_critic_mask=bool_tensor(value_critic_values),
                risk_targets=torch.tensor(
                    risk_target_values,
                    device=self.device,
                    dtype=candidate_cost_values.dtype,
                ),
                risk_critic_mask=bool_tensor(risk_critic_values),
                risk_actor_mask=bool_tensor(risk_actor_values),
                forced_mask=forced_mask,
                censored_mask=bool_tensor(censored_values),
                direct_avoid_mask=bool_tensor(direct_avoid_values),
                completion_mask=bool_tensor(completion_values),
                cycle_groups=tuple(cycle_groups),
                cycle_behavior_mean_log_probabilities=tuple(cycle_behavior),
                cycle_margins=tuple(cycle_margins),
                contrast_pairs=tuple(contrast_pairs),
                contrast_margins=tuple(contrast_margins),
                risk_advantage_clip=(self.failure_credit_config.liveness_risk_advantage_clip),
                risk_actor_min_selected_probability=(
                    self.failure_credit_config.liveness_risk_actor_min_selected_probability
                ),
                contrast_margin=self.failure_credit_config.liveness_contrast_margin,
            )
            return replace(
                losses,
                policy_lag_suppressed_labels=manifest.policy_lag_suppressed_labels,
                risk_actor_phase_suppressed_labels=(manifest.risk_actor_phase_suppressed_labels),
                replayed_contexts=manifest.work.contexts,
                replayed_steps=manifest.work.steps,
                replayed_candidates=manifest.work.candidates,
                autograd_segments=manifest.work.autograd_segments,
            )

        per_record_losses = tuple(
            record_losses(plan, manifest) for plan, manifest in zip(credit_plans, label_manifests, strict=True)
        )
        return _mean_liveness_credit_losses(
            per_record_losses,
            reference=reference,
            retain_graph=True,
        )

    def update(
        self,
        unrolls: tuple[SequenceUnroll, ...],
        *,
        current_policy_version: int,
        current_learner_update: int,
        schedule_policy_version: int | None = None,
        schedule_learner_update: int | None = None,
        transaction_traces: tuple[TransactionTrace, ...] = (),
        credit_plans: tuple[CreditPlan, ...] = (),
        episodic_sequences: tuple[ReplaySequence, ...] = (),
        progress: Callable[[str, dict[str, int | float]], None] | None = None,
    ) -> LearnerMetrics:
        total_started_ns = time.perf_counter_ns()

        def report(stage: str, **fields: int | float) -> None:
            if progress is not None:
                progress(
                    stage,
                    {"elapsed_ms": _elapsed_ms(total_started_ns), **fields},
                )

        if not unrolls:
            raise ValueError("learner unroll batch cannot be empty")
        if isinstance(current_policy_version, bool) or not isinstance(current_policy_version, int):
            raise TypeError("current_policy_version must be an integer")
        if current_policy_version < 0:
            raise ValueError("current_policy_version must be non-negative")
        if isinstance(current_learner_update, bool) or not isinstance(current_learner_update, int):
            raise TypeError("current_learner_update must be an integer")
        if current_learner_update < 0:
            raise ValueError("current_learner_update must be non-negative")
        if schedule_policy_version is None:
            schedule_policy_version = current_policy_version
        if schedule_learner_update is None:
            schedule_learner_update = current_learner_update
        for label, value in (
            ("schedule_policy_version", schedule_policy_version),
            ("schedule_learner_update", schedule_learner_update),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not isinstance(transaction_traces, tuple) or not all(
            isinstance(trace, TransactionTrace) for trace in transaction_traces
        ):
            raise TypeError("transaction_traces must be a TransactionTrace tuple")
        if transaction_traces and not self.transaction_config.enabled:
            raise ValueError("transaction traces require transaction learning to be enabled")
        if not isinstance(credit_plans, tuple) or not all(isinstance(plan, CreditPlan) for plan in credit_plans):
            raise TypeError("credit_plans must be a CreditPlan tuple")
        if credit_plans and not self.failure_credit_config.learning_enabled:
            raise ValueError("credit plans require formal liveness learning to be enabled")
        if len(credit_plans) > self.failure_credit_config.sample_records:
            raise ValueError("credit plan batch exceeds failure_credit.sample_records")
        admitted_manifests: tuple[LivenessLabelManifest, ...] = ()
        if credit_plans:
            # Admit the complete sampled set before any model forward.  The
            # production gradient path below may pack several records into one
            # graph, but aggregate hard budgets apply to the update as a whole
            # and may never be bypassed by execution packing.
            admitted_manifests = tuple(
                compile_liveness_label_manifest(
                    (credit_plan,),
                    config=self.failure_credit_config,
                    current_policy_version=current_policy_version,
                    current_learner_update=schedule_learner_update,
                )
                for credit_plan in credit_plans
            )
            admitted_steps = sum(manifest.work.steps for manifest in admitted_manifests)
            admitted_candidates = sum(manifest.work.candidates for manifest in admitted_manifests)
            if admitted_steps > self.failure_credit_config.liveness_maximum_replayed_steps_per_update:
                raise ValueError("failure-credit replay exceeds " "liveness_maximum_replayed_steps_per_update")
            if admitted_candidates > self.failure_credit_config.liveness_maximum_replayed_candidates_per_update:
                raise ValueError("failure-credit replay exceeds " "liveness_maximum_replayed_candidates_per_update")
        if not isinstance(episodic_sequences, tuple) or not all(
            isinstance(sequence, ReplaySequence) for sequence in episodic_sequences
        ):
            raise TypeError("episodic_sequences must be a ReplaySequence tuple")
        if episodic_sequences and not self.episodic_config.enabled:
            raise ValueError("episodic sequences require episodic learning to be enabled")

        validation_started_ns = time.perf_counter_ns()
        encoding_fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        lags: list[int] = []
        for unroll in unrolls:
            unroll.validate(
                expected_config=self.encoder.config,
                expected_fingerprint=encoding_fingerprint,
                recurrent_hidden_dim=self.model.config.recurrent_hidden_dim,
                maximum_length=self.maximum_unroll_length,
            )
            lag = current_policy_version - unroll.policy_version
            if lag < 0:
                raise ValueError("rollout policy version is newer than the learner")
            if lag > self.maximum_policy_lag:
                raise ValueError(f"rollout policy lag {lag} exceeds limit {self.maximum_policy_lag}")
            lags.append(lag)
        validation_ms = _elapsed_ms(validation_started_ns)
        report("validation_complete", validation_ms=validation_ms)

        recurrent_started_ns = time.perf_counter_ns()
        batch_size = len(unrolls)
        maximum_time = max(len(unroll.steps) for unroll in unrolls)
        report(
            "recurrent_batch_setup_start",
            recurrent_batch_size=batch_size,
            recurrent_maximum_time_steps=maximum_time,
        )
        hidden = torch.stack(
            [torch.from_numpy(unroll.initial_recurrent_state.copy()) for unroll in unrolls],
            dim=0,
        ).to(device=self.device, dtype=next(self.model.parameters()).dtype)
        report(
            "recurrent_batch_setup_complete",
            recurrent_batch_size=batch_size,
            recurrent_maximum_time_steps=maximum_time,
        )

        log_prob_rows: list[Tensor] = []
        value_rows: list[Tensor] = []
        entropy_rows: list[Tensor] = []
        normalized_entropy_rows: list[Tensor] = []
        valid_rows: list[Tensor] = []
        policy_rows: list[Tensor] = []
        behavior_rows: list[Tensor] = []
        reward_rows: list[Tensor] = []
        discount_rows: list[Tensor] = []

        self.model.train()
        for time_index in range(maximum_time):
            completed_steps = time_index + 1
            detailed_progress = (
                completed_steps == 1
                or completed_steps == maximum_time
                or completed_steps % 4 == 0
            )
            active = [index for index, unroll in enumerate(unrolls) if time_index < len(unroll.steps)]
            active_tensor = torch.tensor(active, device=self.device, dtype=torch.long)
            snapshots = tuple(unrolls[index].steps[time_index].snapshot for index in active)
            if detailed_progress:
                report(
                    "recurrent_step_collate_start",
                    recurrent_time_step=completed_steps,
                    recurrent_maximum_time_steps=maximum_time,
                    recurrent_active_unrolls=len(active),
                )
            encoded = collate_encoded_snapshots(
                snapshots,
                expected_config=self.encoder.config,
                expected_fingerprint=encoding_fingerprint,
                device=self.device,
            )
            if detailed_progress:
                report(
                    "recurrent_step_collate_complete",
                    recurrent_time_step=completed_steps,
                    recurrent_maximum_time_steps=maximum_time,
                    recurrent_active_unrolls=len(active),
                )
                report(
                    "recurrent_step_forward_start",
                    recurrent_time_step=completed_steps,
                    recurrent_maximum_time_steps=maximum_time,
                    recurrent_active_unrolls=len(active),
                )
            output = self.model(
                encoded,
                hidden.index_select(0, active_tensor),
                validate=False,
            )
            if detailed_progress:
                report(
                    "recurrent_step_forward_complete",
                    recurrent_time_step=completed_steps,
                    recurrent_maximum_time_steps=maximum_time,
                    recurrent_active_unrolls=len(active),
                )
            hidden = hidden.index_copy(0, active_tensor, output.recurrent_state)
            log_policy = output.policy_log_probabilities()
            selected = torch.tensor(
                [unrolls[index].steps[time_index].action_index for index in active],
                device=self.device,
                dtype=torch.long,
            )
            selected_log_prob = log_policy.gather(1, selected[:, None]).squeeze(1)
            entropy, normalized_entropy = output.policy_entropy_and_normalized()

            floating_zero = output.value.new_zeros(batch_size)
            bool_zero = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            log_prob_rows.append(floating_zero.index_copy(0, active_tensor, selected_log_prob))
            value_rows.append(floating_zero.index_copy(0, active_tensor, output.value))
            entropy_rows.append(floating_zero.index_copy(0, active_tensor, entropy))
            normalized_entropy_rows.append(
                floating_zero.index_copy(0, active_tensor, normalized_entropy)
            )
            valid_rows.append(bool_zero.index_fill(0, active_tensor, True))
            policy_rows.append(
                bool_zero.index_copy(
                    0,
                    active_tensor,
                    torch.tensor(
                        [unrolls[index].steps[time_index].policy_decision for index in active],
                        device=self.device,
                        dtype=torch.bool,
                    ),
                )
            )
            behavior_rows.append(
                floating_zero.index_copy(
                    0,
                    active_tensor,
                    torch.tensor(
                        [unrolls[index].steps[time_index].behavior_log_probability for index in active],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            )
            reward_rows.append(
                floating_zero.index_copy(
                    0,
                    active_tensor,
                    torch.tensor(
                        [unrolls[index].steps[time_index].reward for index in active],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            )
            discount_rows.append(
                floating_zero.index_copy(
                    0,
                    active_tensor,
                    torch.tensor(
                        [unrolls[index].steps[time_index].discount for index in active],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            )
            if detailed_progress:
                report(
                    "recurrent_forward_progress",
                    completed_time_steps=completed_steps,
                    maximum_time_steps=maximum_time,
                )

        bootstrap_values = hidden.new_zeros(batch_size)
        bootstrapped = [index for index, unroll in enumerate(unrolls) if unroll.bootstrap_snapshot is not None]
        if bootstrapped:
            bootstrap_indices = torch.tensor(bootstrapped, device=self.device, dtype=torch.long)
            bootstrap_snapshots = []
            for index in bootstrapped:
                snapshot = unrolls[index].bootstrap_snapshot
                if snapshot is None:  # pragma: no cover - filtered above
                    raise RuntimeError("bootstrap index has no snapshot")
                bootstrap_snapshots.append(snapshot)
            bootstrap_batch = collate_encoded_snapshots(
                tuple(bootstrap_snapshots),
                expected_config=self.encoder.config,
                expected_fingerprint=encoding_fingerprint,
                device=self.device,
            )
            bootstrap_output = self.model(
                bootstrap_batch,
                hidden.index_select(0, bootstrap_indices),
                validate=False,
            )
            bootstrap_values = bootstrap_values.index_copy(
                0,
                bootstrap_indices,
                bootstrap_output.value,
            )
        recurrent_forward_ms = _elapsed_ms(recurrent_started_ns)
        report("recurrent_forward_complete", recurrent_forward_ms=recurrent_forward_ms)

        target_started_ns = time.perf_counter_ns()
        log_probs = torch.stack(log_prob_rows)
        values = torch.stack(value_rows)
        entropies = torch.stack(entropy_rows)
        normalized_entropies = torch.stack(normalized_entropy_rows)
        valid = torch.stack(valid_rows)
        policy_decisions = torch.stack(policy_rows) & valid
        behavior_log_probs = torch.stack(behavior_rows)
        rewards = torch.stack(reward_rows)
        discounts = torch.stack(discount_rows)

        ratios = torch.exp((log_probs - behavior_log_probs).clamp(-20.0, 20.0))
        rho = ratios.clamp(max=self.config.vtrace_rho_clip)
        trace_c = ratios.clamp(max=self.config.vtrace_c_clip)
        policy_rho = ratios.clamp(max=self.config.policy_rho_clip)

        value_targets = torch.zeros_like(values)
        next_targets = torch.zeros_like(values)
        next_baseline = bootstrap_values
        next_vtrace = bootstrap_values
        for time_index in range(maximum_time - 1, -1, -1):
            active_mask = valid[time_index]
            next_targets[time_index] = torch.where(active_mask, next_vtrace, torch.zeros_like(next_vtrace))
            delta = rho[time_index] * (rewards[time_index] + discounts[time_index] * next_baseline - values[time_index])
            candidate_target = (
                values[time_index] + delta + discounts[time_index] * trace_c[time_index] * (next_vtrace - next_baseline)
            )
            value_targets[time_index] = torch.where(active_mask, candidate_target, torch.zeros_like(candidate_target))
            next_baseline = torch.where(active_mask, values[time_index], next_baseline)
            next_vtrace = torch.where(active_mask, candidate_target, next_vtrace)

        advantages = policy_rho * (rewards + discounts * next_targets - values)
        valid_float = valid.to(dtype=values.dtype)
        policy_float = policy_decisions.to(dtype=values.dtype)
        policy_denominator = policy_float.sum().clamp_min(1.0)
        value_denominator = valid_float.sum().clamp_min(1.0)
        policy_loss = -(log_probs * advantages.detach() * policy_float).sum() / policy_denominator
        value_loss = (
            F.smooth_l1_loss(
                values,
                value_targets.detach(),
                reduction="none",
            )
            * valid_float
        ).sum() / value_denominator
        entropy = (entropies * policy_float).sum() / policy_denominator
        # Policy-health/importance telemetry excludes singleton forced steps.
        # Their ratio is mechanically one and would otherwise manufacture the
        # exact pattern used by the one-hot breaker.
        active_ratios = ratios[policy_decisions]
        clipped = (active_ratios > self.config.vtrace_rho_clip).float()
        active_normalized_entropies = normalized_entropies[policy_decisions]
        scheduled_entropy_weight = _annealed_entropy_weight(
            self.config,
            policy_version=schedule_policy_version,
        )
        (
            entropy_weight,
            entropy_breaker_active,
            entropy_breaker_one_hot_condition,
            entropy_breaker_soft_collapse_condition,
        ) = self._entropy_weight_for_batch(
            base_weight=scheduled_entropy_weight,
            active_ratios=active_ratios,
            clipped=clipped,
            active_normalized_entropies=active_normalized_entropies,
        )
        online_objective_loss = (
            self.config.policy_weight * policy_loss + self.config.value_weight * value_loss - entropy_weight * entropy
        )
        transaction_losses = self._transaction_losses(transaction_traces)
        transaction_effect_loss = transaction_losses.effect_loss
        transaction_delta_loss = transaction_losses.delta_loss
        transaction_q_loss = transaction_losses.q_loss
        transaction_pairwise_loss = transaction_losses.pairwise_loss
        transaction_completion_policy_loss = (
            transaction_losses.completion_policy_loss
        )
        transaction_objective_loss = (
            self.transaction_config.effect_weight * (transaction_effect_loss + transaction_delta_loss)
            + self.transaction_config.transaction_q_weight * transaction_q_loss
            + self.transaction_config.pairwise_ranking_weight * transaction_pairwise_loss
            + self.transaction_config.completion_policy_weight * transaction_completion_policy_loss
            + self.transaction_config.lifecycle_entry_support_weight
            * transaction_losses.entry_support_loss
            + self.transaction_config.lifecycle_smdp_q_weight
            * transaction_losses.smdp_q_loss
        )
        total_loss = online_objective_loss + transaction_objective_loss
        _require_finite(
            "targets/loss",
            (
                ("value_targets", value_targets),
                ("advantages", advantages),
                ("loss", total_loss),
                ("transaction_effect_loss", transaction_effect_loss),
                ("transaction_delta_loss", transaction_delta_loss),
                ("transaction_q_loss", transaction_q_loss),
                ("transaction_pairwise_loss", transaction_pairwise_loss),
                (
                    "transaction_completion_policy_loss",
                    transaction_completion_policy_loss,
                ),
                (
                    "transaction_entry_support_loss",
                    transaction_losses.entry_support_loss,
                ),
                ("transaction_smdp_q_loss", transaction_losses.smdp_q_loss),
            ),
        )
        target_and_loss_ms = _elapsed_ms(target_started_ns)
        report("targets_complete", target_and_loss_ms=target_and_loss_ms)

        backward_started_ns = time.perf_counter_ns()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        all_model_parameters = tuple(self.model.parameters())
        gradients_before_liveness = _parameter_gradient_snapshot(all_model_parameters)
        policy_head_parameters = tuple(self.model.policy_head.parameters())
        liveness_head_parameters = (
            tuple(self.model.candidate_liveness_cost_head.parameters())
            + tuple(self.model.liveness_cost_value_head.parameters())
            if (self.model.candidate_liveness_cost_head is not None and self.model.liveness_cost_value_head is not None)
            else ()
        )
        policy_gradients_before_liveness = _parameter_gradient_snapshot(policy_head_parameters)
        critic_gradients_before_liveness = _parameter_gradient_snapshot(liveness_head_parameters)
        liveness_started_ns = time.perf_counter_ns()
        packed_liveness_losses: list[LivenessCreditLosses] = []
        packed_record_counts: list[int] = []
        liveness_pack_size = self.failure_credit_config.liveness_records_per_autograd_batch
        liveness_packs = _liveness_autograd_packs(
            admitted_manifests,
            maximum_records=liveness_pack_size,
        )
        liveness_autograd_microbatches = len(liveness_packs)
        for pack_index, pack in enumerate(liveness_packs):
            pack_start = pack.start
            pack_end = pack.end
            plan_pack = credit_plans[pack_start:pack_end]
            manifest_pack = admitted_manifests[pack_start:pack_end]
            pack_record_count = len(plan_pack)
            pack_contexts = pack.work.contexts
            pack_steps = pack.work.steps
            pack_candidates = pack.work.candidates
            pack_segments = pack.work.autograd_segments
            # Report the complete active-shape pack before model forward. A
            # native GPU stall can then be attributed to one bounded graph and
            # its workload. Per-record events remain for monitor compatibility.
            report(
                "liveness_autograd_batch_start",
                liveness_autograd_batch_index=pack_index,
                liveness_autograd_batches=liveness_autograd_microbatches,
                liveness_batch_first_record=pack_start,
                liveness_batch_records=pack_record_count,
                liveness_records=len(credit_plans),
                liveness_batch_contexts=pack_contexts,
                liveness_batch_steps=pack_steps,
                liveness_batch_candidates=pack_candidates,
                liveness_batch_autograd_segments=pack_segments,
                liveness_batch_oversized_singleton=int(pack.oversized_singleton),
                liveness_autograd_packing_version=_LIVENESS_AUTOGRAD_PACKING_VERSION,
                liveness_autograd_max_steps=_LIVENESS_AUTOGRAD_MAX_STEPS,
                liveness_autograd_max_candidates=_LIVENESS_AUTOGRAD_MAX_CANDIDATES,
                liveness_autograd_max_segments=_LIVENESS_AUTOGRAD_MAX_SEGMENTS,
            )
            for offset, admitted_manifest in enumerate(manifest_pack):
                report(
                    "liveness_record_start",
                    liveness_record_index=pack_start + offset,
                    liveness_records=len(credit_plans),
                    liveness_autograd_batch_index=pack_index,
                    liveness_record_contexts=admitted_manifest.work.contexts,
                    liveness_record_steps=admitted_manifest.work.steps,
                    liveness_record_candidates=admitted_manifest.work.candidates,
                    liveness_record_autograd_segments=(admitted_manifest.work.autograd_segments),
                )
            pack_losses = self.credit_plan_liveness_losses(
                plan_pack,
                current_policy_version=current_policy_version,
                current_learner_update=schedule_learner_update,
            )
            pack_critic_objective = (
                self.failure_credit_config.liveness_value_critic_weight * pack_losses.value_critic_loss
                + self.failure_credit_config.liveness_cost_critic_weight * pack_losses.critic_loss
            )
            pack_policy_objective = (
                self.failure_credit_config.liveness_cost_actor_weight * pack_losses.risk_actor_loss
                + self.failure_credit_config.liveness_direct_policy_weight * pack_losses.direct_avoid_loss
                + self.failure_credit_config.liveness_cycle_policy_weight * pack_losses.cycle_likelihood_loss
                + self.failure_credit_config.liveness_contrast_policy_weight * pack_losses.contrast_loss
                + self.failure_credit_config.liveness_completion_policy_weight * pack_losses.completion_loss
            )
            pack_objective = pack_critic_objective + pack_policy_objective
            _require_finite(
                "liveness targets/loss",
                (
                    (
                        "liveness_value_critic_loss",
                        pack_losses.value_critic_loss,
                    ),
                    ("liveness_q_critic_loss", pack_losses.critic_loss),
                    (
                        "liveness_cost_actor_loss",
                        pack_losses.risk_actor_loss,
                    ),
                    (
                        "liveness_direct_policy_loss",
                        pack_losses.direct_avoid_loss,
                    ),
                    (
                        "liveness_cycle_policy_loss",
                        pack_losses.cycle_likelihood_loss,
                    ),
                    (
                        "liveness_contrast_policy_loss",
                        pack_losses.contrast_loss,
                    ),
                    (
                        "liveness_completion_policy_loss",
                        pack_losses.completion_loss,
                    ),
                    ("liveness_autograd_batch_objective", pack_objective),
                ),
            )
            # ``pack_losses`` is the equal-record mean inside this pack. Scale
            # by its fraction of the full sample so arbitrary final-pack sizes
            # preserve the exact global equal-record objective.
            (pack_objective * (pack_record_count / float(len(credit_plans)))).backward()  # type: ignore[no-untyped-call]
            packed_liveness_losses.append(
                _mean_liveness_credit_losses(
                    (pack_losses,),
                    reference=next(self.model.parameters()),
                )
            )
            packed_record_counts.append(pack_record_count)
        liveness_losses = _mean_liveness_credit_losses(
            tuple(packed_liveness_losses),
            reference=next(self.model.parameters()),
            weights=tuple(packed_record_counts),
        )
        liveness_critic_objective = (
            self.failure_credit_config.liveness_value_critic_weight * liveness_losses.value_critic_loss
            + self.failure_credit_config.liveness_cost_critic_weight * liveness_losses.critic_loss
        )
        liveness_policy_objective = (
            self.failure_credit_config.liveness_cost_actor_weight * liveness_losses.risk_actor_loss
            + self.failure_credit_config.liveness_direct_policy_weight * liveness_losses.direct_avoid_loss
            + self.failure_credit_config.liveness_cycle_policy_weight * liveness_losses.cycle_likelihood_loss
            + self.failure_credit_config.liveness_contrast_policy_weight * liveness_losses.contrast_loss
            + self.failure_credit_config.liveness_completion_policy_weight * liveness_losses.completion_loss
        )
        liveness_credit_loss = liveness_critic_objective + liveness_policy_objective
        (
            liveness_gradient_norm_before_clip,
            liveness_gradient_norm_after_clip,
            liveness_gradient_clip_scale,
        ) = _clip_parameter_gradient_delta_(
            all_model_parameters,
            gradients_before_liveness,
            maximum_norm=self.failure_credit_config.liveness_gradient_clip_norm,
        )
        liveness_gradient_delta = _parameter_gradient_delta_snapshot(
            all_model_parameters,
            gradients_before_liveness,
        )
        liveness_policy_gradient_norm = _parameter_gradient_delta_norm(
            policy_head_parameters,
            policy_gradients_before_liveness,
        )
        liveness_critic_gradient_norm = _parameter_gradient_delta_norm(
            liveness_head_parameters,
            critic_gradients_before_liveness,
        )
        liveness_replay_ms = _elapsed_ms(liveness_started_ns)
        liveness_critic_labels = liveness_losses.value_labels + liveness_losses.critic_labels
        report(
            "liveness_replay_complete",
            liveness_replay_ms=liveness_replay_ms,
            liveness_records=len(credit_plans),
            liveness_autograd_batches=liveness_autograd_microbatches,
            liveness_replayed_contexts=liveness_losses.replayed_contexts,
            liveness_replayed_steps=liveness_losses.replayed_steps,
            liveness_replayed_candidates=liveness_losses.replayed_candidates,
            liveness_autograd_segments=liveness_losses.autograd_segments,
            liveness_head_calibration_active=int(
                schedule_learner_update < self.failure_credit_config.liveness_head_calibration_updates
            ),
            liveness_risk_actor_enabled=int(
                schedule_learner_update >= self.failure_credit_config.liveness_risk_actor_start_update
            ),
        )
        episodic_started_ns = time.perf_counter_ns()
        episodic_losses = self._episodic_losses(
            episodic_sequences,
            current_policy_version=current_policy_version,
        )
        if episodic_sequences:
            _require_finite(
                "episodic targets/loss",
                (
                    ("episodic_loss", episodic_losses.total_loss),
                    (
                        "episodic_primary_policy_loss",
                        episodic_losses.primary_policy_loss,
                    ),
                    ("episodic_task_value_loss", episodic_losses.task_value_loss),
                    (
                        "episodic_revival_value_loss",
                        episodic_losses.revival_value_loss,
                    ),
                    (
                        "episodic_revival_policy_loss",
                        episodic_losses.revival_policy_loss,
                    ),
                    (
                        "episodic_act_segment_policy_loss",
                        episodic_losses.act_segment_policy_loss,
                    ),
                    (
                        "episodic_combat_hp_loss_value_loss",
                        episodic_losses.combat_hp_loss_value_loss,
                    ),
                ),
            )
            episodic_losses.total_loss.backward()  # type: ignore[no-untyped-call]
        episodic_replay_ms = _elapsed_ms(episodic_started_ns)
        report(
            "episodic_replay_complete",
            episodic_replay_ms=episodic_replay_ms,
            episodic_sequences=len(episodic_sequences),
            episodic_burn_in_steps=episodic_losses.burn_in_steps,
            episodic_learn_steps=episodic_losses.learn_steps,
        )
        gradients = tuple(
            (name, parameter.grad) for name, parameter in self.model.named_parameters() if parameter.grad is not None
        )
        _require_finite("gradients", gradients)
        # The primary/episodic objective and liveness auxiliary own independent
        # norm budgets. Temporarily remove the already-clipped liveness delta,
        # clip the remaining gradient, then restore that auxiliary delta. This
        # prevents one plane from consuming the other's clipping allowance.
        _apply_parameter_gradient_delta_(
            all_model_parameters,
            liveness_gradient_delta,
            sign=-1.0,
        )
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        _apply_parameter_gradient_delta_(
            all_model_parameters,
            liveness_gradient_delta,
            sign=1.0,
        )
        _require_finite(
            "independently clipped gradients",
            tuple(
                (name, parameter.grad)
                for name, parameter in self.model.named_parameters()
                if parameter.grad is not None
            ),
        )
        gradient_norm = float(gradient_norm_tensor.item())
        backward_ms = _elapsed_ms(backward_started_ns)
        report("backward_complete", backward_ms=backward_ms)

        optimizer_started_ns = time.perf_counter_ns()
        self.optimizer.step()
        _require_finite(
            "parameters",
            tuple((name, parameter) for name, parameter in self.model.named_parameters()),
        )
        optimizer_step_ms = _elapsed_ms(optimizer_started_ns)
        total_ms = _elapsed_ms(total_started_ns)
        report(
            "optimizer_complete",
            optimizer_step_ms=optimizer_step_ms,
            total_ms=total_ms,
        )

        active_advantages = advantages[policy_decisions]
        active_targets = value_targets[valid]
        timings = LearnerTimings(
            validation_ms=validation_ms,
            recurrent_forward_ms=recurrent_forward_ms,
            target_and_loss_ms=target_and_loss_ms,
            backward_ms=backward_ms,
            episodic_replay_ms=episodic_replay_ms,
            optimizer_step_ms=optimizer_step_ms,
            total_ms=total_ms,
        )
        return LearnerMetrics(
            loss=float(
                total_loss.detach().item()
                + liveness_credit_loss.detach().item()
                + episodic_losses.total_loss.detach().item()
            ),
            online_objective_loss=float(online_objective_loss.detach().item()),
            transaction_objective_loss=float(
                transaction_objective_loss.detach().item()
            ),
            liveness_objective_loss=float(liveness_credit_loss.detach().item()),
            episodic_objective_loss=float(
                episodic_losses.total_loss.detach().item()
            ),
            policy_loss=float(policy_loss.detach().item()),
            value_loss=float(value_loss.detach().item()),
            entropy=float(entropy.detach().item()),
            normalized_entropy=(
                float(active_normalized_entropies.detach().mean().item())
                if active_normalized_entropies.numel()
                else 0.0
            ),
            entropy_weight=entropy_weight,
            entropy_breaker_active=int(entropy_breaker_active),
            entropy_breaker_one_hot_condition=int(entropy_breaker_one_hot_condition),
            entropy_breaker_soft_collapse_condition=int(
                entropy_breaker_soft_collapse_condition
            ),
            entropy_breaker_consecutive_batches=self._collapse_batch_streak,
            entropy_breaker_remaining_updates=(self._entropy_breaker_remaining_updates),
            entropy_breaker_triggers=self._entropy_breaker_triggers,
            advantage_mean=(float(active_advantages.detach().mean().item()) if active_advantages.numel() else 0.0),
            value_target_mean=float(active_targets.detach().mean().item()),
            importance_ratio_mean=(float(active_ratios.detach().mean().item()) if active_ratios.numel() else 0.0),
            importance_ratio_max=(float(active_ratios.detach().max().item()) if active_ratios.numel() else 0.0),
            importance_clip_fraction=(float(clipped.detach().mean().item()) if clipped.numel() else 0.0),
            gradient_norm=gradient_norm,
            unrolls=len(unrolls),
            environment_steps=int(valid.sum().item()),
            policy_decisions=int(policy_decisions.sum().item()),
            maximum_policy_lag=max(lags),
            transaction_effect_loss=float(transaction_effect_loss.detach().item()),
            transaction_delta_loss=float(transaction_delta_loss.detach().item()),
            transaction_q_loss=float(transaction_q_loss.detach().item()),
            transaction_pairwise_ranking_loss=float(transaction_pairwise_loss.detach().item()),
            transaction_completion_policy_loss=float(transaction_completion_policy_loss.detach().item()),
            transaction_entry_support_loss=float(
                transaction_losses.entry_support_loss.detach().item()
            ),
            transaction_smdp_q_loss=float(
                transaction_losses.smdp_q_loss.detach().item()
            ),
            transaction_traces=len(transaction_traces),
            transaction_effect_labels=transaction_losses.effect_labels,
            transaction_q_labels=transaction_losses.q_labels,
            transaction_pairs=transaction_losses.pair_count,
            transaction_policy_labels=transaction_losses.policy_labels,
            transaction_policy_preferred_labels=(
                transaction_losses.policy_preferred_labels
            ),
            transaction_policy_avoided_labels=(
                transaction_losses.policy_avoided_labels
            ),
            transaction_entry_support_labels=(
                transaction_losses.entry_support_labels
            ),
            transaction_entry_support_satisfied_labels=(
                transaction_losses.entry_support_satisfied_labels
            ),
            transaction_entry_support_singleton_suppressed_labels=(
                transaction_losses.entry_support_singleton_suppressed_labels
            ),
            transaction_smdp_q_labels=transaction_losses.smdp_q_labels,
            transaction_lifecycle_committed=(
                transaction_losses.lifecycle_committed
            ),
            transaction_lifecycle_cancelled=(
                transaction_losses.lifecycle_cancelled
            ),
            transaction_lifecycle_unresolved=(
                transaction_losses.lifecycle_unresolved
            ),
            transaction_lifecycle_deadlock=(
                transaction_losses.lifecycle_deadlock
            ),
            transaction_entry_model_probability_mean=(
                transaction_losses.entry_model_probability_mean
            ),
            transaction_entry_collection_model_probability_mean=(
                transaction_losses.entry_collection_model_probability_mean
            ),
            transaction_entry_behavior_probability_mean=(
                transaction_losses.entry_behavior_probability_mean
            ),
            transaction_entry_log_support_gap_mean=(
                transaction_losses.entry_log_support_gap_mean
            ),
            transaction_upgrade_entry_support_labels=(
                transaction_losses.upgrade_entry_support_labels
            ),
            transaction_remove_entry_support_labels=(
                transaction_losses.remove_entry_support_labels
            ),
            transaction_upgrade_entry_model_probability_mean=(
                transaction_losses.upgrade_entry_model_probability_mean
            ),
            transaction_remove_entry_model_probability_mean=(
                transaction_losses.remove_entry_model_probability_mean
            ),
            transaction_upgrade_entry_behavior_probability_mean=(
                transaction_losses.upgrade_entry_behavior_probability_mean
            ),
            transaction_remove_entry_behavior_probability_mean=(
                transaction_losses.remove_entry_behavior_probability_mean
            ),
            liveness_credit_loss=float(liveness_credit_loss.detach().item()),
            liveness_cost_critic_loss=float(
                (liveness_losses.value_critic_loss + liveness_losses.critic_loss).detach().item()
            ),
            liveness_value_critic_loss=float(liveness_losses.value_critic_loss.detach().item()),
            liveness_q_critic_loss=float(liveness_losses.critic_loss.detach().item()),
            liveness_cost_actor_loss=float(liveness_losses.risk_actor_loss.detach().item()),
            liveness_direct_policy_loss=float(liveness_losses.direct_avoid_loss.detach().item()),
            liveness_cycle_policy_loss=float(liveness_losses.cycle_likelihood_loss.detach().item()),
            liveness_contrast_policy_loss=float(liveness_losses.contrast_loss.detach().item()),
            liveness_completion_policy_loss=float(liveness_losses.completion_loss.detach().item()),
            liveness_credit_plans=len(credit_plans),
            liveness_cost_labels=liveness_critic_labels,
            liveness_value_labels=liveness_losses.value_labels,
            liveness_q_labels=liveness_losses.critic_labels,
            liveness_cost_actor_labels=liveness_losses.risk_actor_labels,
            liveness_direct_policy_labels=liveness_losses.direct_avoid_labels,
            liveness_cycle_policy_labels=liveness_losses.cycle_labels,
            liveness_contrast_policy_labels=liveness_losses.contrast_labels,
            liveness_completion_policy_labels=(liveness_losses.completion_labels),
            liveness_forced_actor_suppressed_labels=(liveness_losses.forced_actor_suppressed_labels),
            liveness_censored_suppressed_labels=(liveness_losses.censored_suppressed_labels),
            liveness_policy_lag_suppressed_labels=(liveness_losses.policy_lag_suppressed_labels),
            liveness_risk_actor_phase_suppressed_labels=(liveness_losses.risk_actor_phase_suppressed_labels),
            liveness_risk_actor_saturation_suppressed_labels=(
                liveness_losses.risk_actor_saturation_suppressed_labels
            ),
            liveness_centered_risk_mean=liveness_losses.centered_risk_mean,
            liveness_centered_risk_max_abs=(liveness_losses.centered_risk_max_abs),
            liveness_policy_gradient_norm=liveness_policy_gradient_norm,
            liveness_critic_gradient_norm=liveness_critic_gradient_norm,
            liveness_gradient_norm_before_clip=(liveness_gradient_norm_before_clip),
            liveness_gradient_norm_after_clip=(liveness_gradient_norm_after_clip),
            liveness_gradient_clip_scale=liveness_gradient_clip_scale,
            liveness_head_calibration_active=int(
                schedule_learner_update < self.failure_credit_config.liveness_head_calibration_updates
            ),
            liveness_risk_actor_enabled=int(
                schedule_learner_update >= self.failure_credit_config.liveness_risk_actor_start_update
            ),
            liveness_replayed_contexts=liveness_losses.replayed_contexts,
            liveness_replayed_steps=liveness_losses.replayed_steps,
            liveness_replayed_candidates=(liveness_losses.replayed_candidates),
            liveness_autograd_microbatches=liveness_autograd_microbatches,
            liveness_autograd_segments=liveness_losses.autograd_segments,
            episodic_loss=float(episodic_losses.total_loss.detach().item()),
            episodic_primary_policy_loss=float(episodic_losses.primary_policy_loss.detach().item()),
            episodic_task_value_loss=float(episodic_losses.task_value_loss.detach().item()),
            episodic_revival_value_loss=float(episodic_losses.revival_value_loss.detach().item()),
            episodic_revival_policy_loss=float(episodic_losses.revival_policy_loss.detach().item()),
            episodic_act_segment_policy_loss=float(
                episodic_losses.act_segment_policy_loss.detach().item()
            ),
            episodic_combat_hp_loss_value_loss=float(
                episodic_losses.combat_hp_loss_value_loss.detach().item()
            ),
            episodic_sequences=len(episodic_sequences),
            episodic_burn_in_steps=episodic_losses.burn_in_steps,
            episodic_learn_steps=episodic_losses.learn_steps,
            episodic_success_policy_candidate_labels=(episodic_losses.success_policy_candidate_labels),
            episodic_policy_labels=episodic_losses.policy_labels,
            episodic_policy_active_sequences=(episodic_losses.policy_active_sequences),
            episodic_failure_policy_suppressed_labels=(episodic_losses.failure_policy_suppressed_labels),
            episodic_policy_lag_suppressed_labels=(episodic_losses.policy_lag_suppressed_labels),
            episodic_success_trust_region_suppressed_labels=(episodic_losses.success_trust_region_suppressed_labels),
            episodic_success_surface_exempted_labels=(
                episodic_losses.success_surface_exempted_labels
            ),
            episodic_act_segment_policy_labels=(
                episodic_losses.act_segment_policy_labels
            ),
            episodic_act_segment_health_gate_suppressed_labels=(
                episodic_losses.act_segment_health_gate_suppressed_labels
            ),
            episodic_act_segment_healthy_acts=(
                episodic_losses.act_segment_healthy_acts
            ),
            episodic_task_value_labels=episodic_losses.task_value_labels,
            episodic_combat_hp_loss_value_labels=(
                episodic_losses.combat_hp_loss_value_labels
            ),
            episodic_revival_value_labels=(episodic_losses.revival_value_labels),
            episodic_efficiency_policy_labels=(episodic_losses.efficiency_policy_labels),
            episodic_importance_ratio_mean=(
                float(episodic_losses.importance_ratios.detach().mean().item())
                if episodic_losses.importance_ratios.numel()
                else 0.0
            ),
            episodic_importance_ratio_max=(
                float(episodic_losses.importance_ratios.detach().max().item())
                if episodic_losses.importance_ratios.numel()
                else 0.0
            ),
            episodic_importance_clip_fraction=(
                float(
                    (episodic_losses.importance_ratios > self.episodic_config.importance_ratio_clip)
                    .float()
                    .mean()
                    .item()
                )
                if episodic_losses.importance_ratios.numel()
                else 0.0
            ),
            episodic_maximum_policy_lag=episodic_losses.maximum_policy_lag,
            timings=timings,
        )

    @staticmethod
    def _episodic_task_target(
        horizon: str,
        target: HorizonTargets,
        *,
        run_target: HorizonTargets,
    ) -> float:
        """Return one-outcome-unit, cost-free primary supervision.

        ``task_return`` contains the base terminal/progress reward captured by
        the collector; revival, HP-loss, and pace shaping are deliberately
        absent.  An authoritative run terminal already contributes its factual
        ``+1``/``-1`` terminal reward, so the run head must consume that return
        directly.  A combat or Act boundary normally has no task terminal of its
        own and therefore receives one explicit local outcome unit.  When that
        local boundary coincides with the authoritative run terminal, its return
        already contains the same terminal unit and must not add it again.
        """

        if horizon not in {"combat", "act", "run"}:
            raise ValueError(f"unsupported episodic horizon {horizon!r}")
        if not target.observed or target.success is None or target.task_return is None:
            raise ValueError("episodic task target is not observed")
        if horizon == "run":
            return float(target.task_return)

        shares_run_terminal = bool(run_target.observed and target.return_steps == run_target.return_steps)
        if shares_run_terminal:
            if target.success is not run_target.success:
                raise ValueError("a local boundary sharing the run terminal has a conflicting outcome")
            return float(target.task_return)
        return float(target.task_return + (1.0 if target.success else -1.0))

    @staticmethod
    def _horizon_outputs(
        output: RecurrentCandidateOutput,
        row: int,
        *,
        combat: HorizonTargets,
        act: HorizonTargets,
        run: HorizonTargets,
    ) -> tuple[tuple[str, HorizonTargets, Tensor, Tensor], ...]:
        return (
            (
                "combat",
                combat,
                output.combat_task_value[row],
                output.combat_revival_cost_value[row],
            ),
            (
                "act",
                act,
                output.act_task_value[row],
                output.act_revival_cost_value[row],
            ),
            (
                "run",
                run,
                output.run_task_value[row],
                output.run_revival_cost_value[row],
            ),
        )

    def _episodic_losses(
        self,
        sequences: tuple[ReplaySequence, ...],
        *,
        current_policy_version: int,
    ) -> _EpisodicLossBatch:
        """Compute episodic losses with deterministic recurrent reconstruction.

        Episodic sequences deliberately omit recurrently irrelevant combat
        prefixes.  Dropout would nevertheless consume a different RNG stream
        for the sparse and full histories and would make the supposedly exact
        reconstruction depend on which irrelevant steps were skipped.  Run
        both the no-grad prefix and differentiable suffix in evaluation mode so
        the sparse replay has deterministic semantics, while preserving the
        caller's model mode even when validation or collation raises.

        Evaluation mode does not disable autograd.  The learning suffix still
        owns the complete bounded graph; only dropout is disabled (the active
        model contains no batch-normalization state).
        """

        was_training = self.model.training
        self.model.eval()
        try:
            return self._episodic_losses_deterministic(
                sequences,
                current_policy_version=current_policy_version,
            )
        finally:
            self.model.train(was_training)

    def _episodic_losses_deterministic(
        self,
        sequences: tuple[ReplaySequence, ...],
        *,
        current_policy_version: int,
    ) -> _EpisodicLossBatch:
        """Replay complete-episode labels through one bounded GPU graph.

        The caller holds the model in evaluation mode.  Every sequence first
        reconstructs the current model's split recurrent
        state from its exact sparse prefix under ``torch.no_grad()``.  Only the
        configured contiguous suffix is batched with autograd.  Thus a 30,000
        step source episode can increase CPU storage and no-grad compute, but it
        cannot increase activation memory beyond
        ``len(sequences) * episodic_config.learn_steps``.

        Successful-horizon selected-action replay is clipped by the factual
        behavior probability. Failed horizons remain authoritative value
        targets but never become blanket anti-imitation policy labels. Revival
        cost can affect policy only for a successfully completed horizon.
        Outside a configured, success-classified primary
        tie band, the already-weighted cost signal is capped below the absolute
        primary residual.  Inside that narrow band, a small nominal-primary
        floor keeps the cost tie-break alive even when the task advantage is
        calibrated to zero.  This implements the staged ordering ``complete
        first, then reduce revivals`` instead of silently deleting the
        secondary objective at primary convergence.
        """

        zero = next(self.model.parameters()).sum() * 0.0
        empty_ratios = torch.empty(0, device=self.device, dtype=torch.float32)
        if not sequences:
            return _EpisodicLossBatch(
                total_loss=zero,
                primary_policy_loss=zero,
                task_value_loss=zero,
                revival_value_loss=zero,
                revival_policy_loss=zero,
                act_segment_policy_loss=zero,
                combat_hp_loss_value_loss=zero,
                burn_in_steps=0,
                learn_steps=0,
                success_policy_candidate_labels=0,
                policy_labels=0,
                policy_active_sequences=0,
                failure_policy_suppressed_labels=0,
                policy_lag_suppressed_labels=0,
                success_trust_region_suppressed_labels=0,
                success_surface_exempted_labels=0,
                act_segment_policy_labels=0,
                act_segment_health_gate_suppressed_labels=0,
                act_segment_healthy_acts=0,
                task_value_labels=0,
                combat_hp_loss_value_labels=0,
                revival_value_labels=0,
                efficiency_policy_labels=0,
                importance_ratios=empty_ratios,
                maximum_policy_lag=0,
            )

        fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        hidden_rows: list[Tensor] = []
        total_burn_in_steps = 0
        total_learn_steps = 0
        maximum_policy_lag = 0
        for sequence in sequences:
            if not sequence.exact_recurrent_reconstruction:
                raise ValueError("episodic replay sequence is not an exact reconstruction")
            if sequence.configured_burn_in_steps != self.episodic_config.burn_in_steps:
                raise ValueError("episodic replay burn-in configuration differs from learner")
            if not 0 < len(sequence.learn_steps) <= self.episodic_config.learn_steps:
                raise ValueError("episodic replay learning suffix exceeds configured length")
            hidden = self.model.initial_state(1, device=self.device)
            with torch.no_grad():
                for step in sequence.burn_in:
                    step.snapshot.validate(
                        expected_config=self.encoder.config,
                        expected_fingerprint=fingerprint,
                    )
                    encoded = collate_encoded_snapshots(
                        (step.snapshot,),
                        expected_config=self.encoder.config,
                        expected_fingerprint=fingerprint,
                        device=self.device,
                    )
                    hidden = self.model(encoded, hidden, validate=False).recurrent_state
            hidden_rows.append(hidden[0].detach())
            total_burn_in_steps += len(sequence.burn_in)
            total_learn_steps += len(sequence.learn_steps)
            for step in sequence.learn_steps:
                lag = current_policy_version - step.decision.policy_version
                if lag < 0:
                    raise ValueError("episodic behavior policy version is newer than the learner")
                maximum_policy_lag = max(maximum_policy_lag, lag)

        hidden = torch.stack(hidden_rows, dim=0)
        maximum_time = max(len(sequence.learn_steps) for sequence in sequences)
        task_predictions: list[Tensor] = []
        task_targets: list[float] = []
        revival_predictions: list[Tensor] = []
        revival_targets: list[float] = []
        combat_hp_loss_predictions: list[Tensor] = []
        combat_hp_loss_targets: list[float] = []
        primary_policy_terms: list[Tensor] = []
        act_segment_policy_terms: list[Tensor] = []
        revival_policy_terms: list[Tensor] = []
        combined_policy_terms: list[Tensor] = []
        importance_ratios: list[Tensor] = []
        success_policy_candidate_labels = 0
        policy_active_sequence_indexes: set[int] = set()
        efficiency_policy_labels = 0
        failure_policy_suppressed_labels = 0
        policy_lag_suppressed_labels = 0
        success_trust_region_suppressed_labels = 0
        success_surface_exempted_labels = 0
        act_segment_policy_labels = 0
        act_segment_health_gate_suppressed_labels = 0
        healthy_act_keys: set[tuple[str, int]] = set()

        for time_index in range(maximum_time):
            active = [index for index, sequence in enumerate(sequences) if time_index < len(sequence.learn_steps)]
            active_tensor = torch.tensor(active, device=self.device, dtype=torch.long)
            active_steps = tuple(sequences[index].learn_steps[time_index] for index in active)
            snapshots = tuple(step.snapshot for step in active_steps)
            encoded = collate_encoded_snapshots(
                snapshots,
                expected_config=self.encoder.config,
                expected_fingerprint=fingerprint,
                device=self.device,
            )
            output = self.model(
                encoded,
                hidden.index_select(0, active_tensor),
                validate=False,
            )
            hidden = hidden.index_copy(0, active_tensor, output.recurrent_state)
            log_policy = output.policy_log_probabilities()

            for row, step in enumerate(active_steps):
                decision = step.decision
                action_index = decision.action_index
                selected_log_probability = log_policy[row, action_index]
                horizons = self._horizon_outputs(
                    output,
                    row,
                    combat=step.combat,
                    act=step.act,
                    run=step.run,
                )

                for horizon, target, task_prediction, revival_prediction in horizons:
                    if not target.observed:
                        continue
                    task_predictions.append(task_prediction)
                    task_targets.append(
                        self._episodic_task_target(
                            horizon,
                            target,
                            run_target=step.run,
                        )
                    )
                    if target.efficiency_eligible:
                        if target.future_revivals is None:  # pragma: no cover - invariant
                            raise RuntimeError("successful horizon has no revival target")
                        revival_predictions.append(revival_prediction)
                        revival_targets.append(float(target.future_revivals))
                    if (
                        horizon == "combat"
                        and self.episodic_config.combat_hp_loss_value_weight > 0.0
                    ):
                        if target.future_hp_loss is None:  # pragma: no cover
                            raise RuntimeError("observed combat has no HP-loss target")
                        combat_hp_loss_predictions.append(
                            output.combat_hp_loss_value[row]
                        )
                        combat_hp_loss_targets.append(
                            1.0
                            - math.exp(
                                -float(target.future_hp_loss)
                                / self.episodic_config.combat_hp_loss_reference
                            )
                        )

                if not decision.policy_decision:
                    continue
                primary = next(
                    (item for item in reversed(horizons) if item[1].observed),
                    None,
                )
                if primary is None:
                    continue
                primary_horizon, primary_target, primary_prediction, _ = primary
                act_segment_imitation = False
                policy_weight = self.episodic_config.primary_policy_weight
                if (
                    primary_target.success is not True
                    and self.episodic_config.act_segment_imitation_enabled
                    and step.run.observed
                    and step.run.success is False
                    and step.act.observed
                    and step.act.success is True
                ):
                    act_health = healthy_act_segment_for_step(
                        step,
                        sequences[active[row]].act_segment_health,
                        minimum_exit_hp_ratio=(
                            self.episodic_config.act_segment_min_exit_hp_ratio
                        ),
                        maximum_revival_fraction=(
                            self.episodic_config.act_segment_max_revival_fraction
                        ),
                    )
                    if act_health is True:
                        act_item = next(
                            item for item in horizons if item[0] == "act"
                        )
                        primary_horizon, primary_target, primary_prediction, _ = (
                            act_item
                        )
                        act_segment_imitation = True
                        policy_weight = (
                            self.episodic_config.primary_policy_weight
                            * self.episodic_config.act_segment_policy_weight
                        )
                        healthy_act_keys.add(
                            (sequences[active[row]].episode_id, decision.act)
                        )
                    else:
                        act_segment_health_gate_suppressed_labels += 1
                if primary_target.success is not True:
                    # A failed long horizon is authoritative evidence for the
                    # task-value heads, but it is not a counterfactual action
                    # label. Applying a negative selected-action likelihood
                    # update to every decision in a long failed run performs
                    # anti-imitation of the entire factual trajectory. With a
                    # hierarchical policy this systematically suppresses the
                    # frequently sampled, many-candidate branches and leaks
                    # probability into unsampled singleton branches (for
                    # example END_TURN). FIFO V-trace and bounded factual
                    # transaction failures still provide local policy
                    # gradients; complete-episode replay reinforces policy
                    # only after an observed successful horizon.
                    failure_policy_suppressed_labels += 1
                    continue
                success_policy_candidate_labels += 1
                if (
                    decision.decision_surface
                    in self.episodic_config.success_imitation_exempt_surfaces
                ):
                    # The complete successful trajectory still supervises all
                    # horizon value heads above.  Only repeated selected-action
                    # imitation is disabled on explicitly reviewed sparse
                    # strategic surfaces; online V-trace and factual SMDP-Q
                    # retain contextual preference learning there.
                    success_surface_exempted_labels += 1
                    continue
                policy_lag = current_policy_version - decision.policy_version
                if policy_lag > self.episodic_config.policy_gradient_max_lag:
                    # Complete episodes remain authoritative long-horizon value
                    # supervision after their successful behavior policy becomes
                    # stale. They must not, however, keep applying selected-action
                    # likelihood gradients to a policy hundreds of updates newer
                    # than the one that generated those decisions. Failed primary
                    # horizons are classified above and never pollute this counter.
                    policy_lag_suppressed_labels += 1
                    continue
                raw_ratio = torch.exp(
                    (selected_log_probability - float(decision.behavior_log_probability)).clamp(-20.0, 20.0)
                )
                # Importance sampling corrects the factual behavior/current
                # distribution mismatch; it is not itself a differentiable
                # policy objective.  Letting gradients flow through rho would
                # add a ``rho * log(pi)`` product-rule term and can reverse a
                # positive-advantage update whenever ``log(pi) < -1``.
                detached_ratio = raw_ratio.detach()
                importance_ratios.append(detached_ratio)
                clipped_ratio = detached_ratio.clamp(max=self.episodic_config.importance_ratio_clip)
                primary_advantage = (
                    primary_prediction.new_tensor(
                        self._episodic_task_target(
                            primary_horizon,
                            primary_target,
                            run_target=step.run,
                        )
                    )
                    - primary_prediction
                )
                if bool(
                    (
                        (primary_advantage.detach() > 0.0)
                        & (detached_ratio > 1.0 + self.episodic_config.success_policy_trust_region_epsilon)
                    ).item()
                ):
                    # The selected successful action is already materially
                    # more likely than under its factual behavior policy.
                    # Preserve value supervision and importance telemetry, but
                    # stop the positive imitation gradient instead of applying
                    # an indefinitely repeated sharpening pressure.
                    success_trust_region_suppressed_labels += 1
                    continue
                policy_active_sequence_indexes.add(active[row])
                primary_signal = policy_weight * primary_advantage.detach()
                raw_policy_term = (
                    -clipped_ratio
                    * selected_log_probability
                    * primary_advantage.detach()
                )
                primary_policy_terms.append(raw_policy_term)
                if act_segment_imitation:
                    act_segment_policy_terms.append(
                        policy_weight * raw_policy_term
                    )
                    act_segment_policy_labels += 1

                secondary_signal = primary_signal.new_zeros(())
                efficiency = (
                    None
                    if act_segment_imitation
                    else next(
                        (
                            item
                            for item in reversed(horizons)
                            if item[1].efficiency_eligible
                        ),
                        None,
                    )
                )
                if efficiency is not None:
                    _, efficiency_target, _, cost_prediction = efficiency
                    if efficiency_target.future_revivals is None:  # pragma: no cover
                        raise RuntimeError("successful horizon has no revival target")
                    cost_advantage = (
                        cost_prediction.new_tensor(float(efficiency_target.future_revivals)) - cost_prediction
                    )
                    raw_secondary_signal = self.episodic_config.revival_policy_weight * cost_advantage.detach()
                    # A pure ``fraction * abs(primary_advantage)`` cap makes
                    # the revival objective exactly zero once the task value
                    # is calibrated.  That prevents successful 0-revival and
                    # 100-revival paths from ever becoming distinguishable at
                    # the point where primary outcomes tie.
                    #
                    # Before the state value is on the successful side of the
                    # signed task target, or while its residual lies outside
                    # the explicit tie tolerance, retain the strict
                    # residual-relative cap.  Only inside that narrow success
                    # tie stratum admit a fraction of one nominal primary
                    # policy unit. Failure/censored horizons never enter this
                    # branch at all.
                    primary_tie = (
                        (primary_prediction.detach() > 0.0)
                        & (primary_advantage.detach().abs() <= self.episodic_config.primary_success_tie_tolerance)
                    ).to(dtype=primary_signal.dtype)
                    success_tie_floor = (
                        primary_signal.new_tensor(self.episodic_config.primary_policy_weight) * primary_tie
                    )
                    protected_primary_scale = torch.maximum(
                        primary_signal.abs(),
                        success_tie_floor,
                    )
                    secondary_limit = self.episodic_config.secondary_advantage_fraction * protected_primary_scale
                    secondary_signal = torch.maximum(
                        torch.minimum(raw_secondary_signal, secondary_limit),
                        -secondary_limit,
                    )
                    revival_policy_terms.append(clipped_ratio * selected_log_probability * secondary_signal)
                    efficiency_policy_labels += 1

                combined_policy_terms.append(
                    -clipped_ratio * selected_log_probability * (primary_signal - secondary_signal)
                )

        task_value_loss = (
            F.smooth_l1_loss(
                torch.stack(task_predictions).float(),
                torch.tensor(task_targets, device=self.device, dtype=torch.float32),
            )
            if task_predictions
            else zero
        )
        # Revival counts have an extremely long tail under unlimited native
        # revival.  Regressing raw counts makes this auxiliary value head
        # numerically dominate the complete-episode objective even though its
        # policy advantage is already secondary.  A log1p observation model
        # keeps zero exact, remains monotone over factual counts, and prevents
        # a 100-revival path from contributing roughly 100x the representation
        # gradient of a one-revival path.
        revival_value_loss = (
            F.smooth_l1_loss(
                torch.log1p(torch.stack(revival_predictions).float()),
                torch.log1p(
                    torch.tensor(
                        revival_targets,
                        device=self.device,
                        dtype=torch.float32,
                    )
                ),
            )
            if revival_predictions
            else zero
        )
        combat_hp_loss_value_loss = (
            F.smooth_l1_loss(
                torch.stack(combat_hp_loss_predictions).float(),
                torch.tensor(
                    combat_hp_loss_targets,
                    device=self.device,
                    dtype=torch.float32,
                ),
            )
            if combat_hp_loss_predictions
            else zero
        )
        primary_policy_loss = torch.stack(primary_policy_terms).mean() if primary_policy_terms else zero
        revival_policy_loss = torch.stack(revival_policy_terms).mean() if revival_policy_terms else zero
        act_segment_policy_loss = (
            torch.stack(act_segment_policy_terms).mean()
            if act_segment_policy_terms
            else zero
        )
        combined_policy_loss = torch.stack(combined_policy_terms).mean() if combined_policy_terms else zero
        total_loss = (
            combined_policy_loss
            + self.episodic_config.task_value_weight * task_value_loss
            + self.episodic_config.revival_value_weight * revival_value_loss
            + self.episodic_config.combat_hp_loss_value_weight
            * combat_hp_loss_value_loss
        )
        ratio_tensor = torch.stack(importance_ratios).float() if importance_ratios else empty_ratios
        return _EpisodicLossBatch(
            total_loss=total_loss,
            primary_policy_loss=primary_policy_loss,
            task_value_loss=task_value_loss,
            revival_value_loss=revival_value_loss,
            revival_policy_loss=revival_policy_loss,
            act_segment_policy_loss=act_segment_policy_loss,
            combat_hp_loss_value_loss=combat_hp_loss_value_loss,
            burn_in_steps=total_burn_in_steps,
            learn_steps=total_learn_steps,
            success_policy_candidate_labels=success_policy_candidate_labels,
            policy_labels=len(primary_policy_terms),
            policy_active_sequences=len(policy_active_sequence_indexes),
            failure_policy_suppressed_labels=failure_policy_suppressed_labels,
            policy_lag_suppressed_labels=policy_lag_suppressed_labels,
            success_trust_region_suppressed_labels=(success_trust_region_suppressed_labels),
            success_surface_exempted_labels=success_surface_exempted_labels,
            act_segment_policy_labels=act_segment_policy_labels,
            act_segment_health_gate_suppressed_labels=(
                act_segment_health_gate_suppressed_labels
            ),
            act_segment_healthy_acts=len(healthy_act_keys),
            task_value_labels=len(task_predictions),
            combat_hp_loss_value_labels=len(combat_hp_loss_predictions),
            revival_value_labels=len(revival_predictions),
            efficiency_policy_labels=efficiency_policy_labels,
            importance_ratios=ratio_tensor,
            maximum_policy_lag=maximum_policy_lag,
        )

    def _transaction_losses(
        self,
        traces: tuple[TransactionTrace, ...],
    ) -> _TransactionLossBatch:
        """Compute factual transaction, entry-support and SMDP losses."""

        zero = next(self.model.parameters()).sum() * 0.0
        if not traces:
            return _TransactionLossBatch(
                effect_loss=zero,
                delta_loss=zero,
                q_loss=zero,
                pairwise_loss=zero,
                completion_policy_loss=zero,
                entry_support_loss=zero,
                smdp_q_loss=zero,
                effect_labels=0,
                q_labels=0,
                pair_count=0,
                policy_labels=0,
                policy_preferred_labels=0,
                policy_avoided_labels=0,
                entry_support_labels=0,
                entry_support_satisfied_labels=0,
                entry_support_singleton_suppressed_labels=0,
                smdp_q_labels=0,
                lifecycle_committed=0,
                lifecycle_cancelled=0,
                lifecycle_unresolved=0,
                lifecycle_deadlock=0,
                entry_model_probability_mean=0.0,
                entry_collection_model_probability_mean=0.0,
                entry_behavior_probability_mean=0.0,
                entry_log_support_gap_mean=0.0,
                upgrade_entry_support_labels=0,
                remove_entry_support_labels=0,
                upgrade_entry_model_probability_mean=0.0,
                remove_entry_model_probability_mean=0.0,
                upgrade_entry_behavior_probability_mean=0.0,
                remove_entry_behavior_probability_mean=0.0,
            )
        fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        effect_logits: list[Tensor] = []
        effect_targets: list[int] = []
        delta_logits: list[Tensor] = []
        delta_targets: list[int] = []
        q_values: list[Tensor] = []
        q_targets: list[float] = []
        selected_policy_log_probabilities: dict[tuple[int, int], Tensor] = {}
        completion_policy_trace_losses: list[Tensor] = []
        completion_policy_preferred_labels = 0
        completion_policy_avoided_labels = 0
        entry_support_terms: list[Tensor] = []
        smdp_q_predictions: list[Tensor] = []
        smdp_q_targets: list[Tensor] = []
        entry_model_probabilities: list[float] = []
        entry_collection_model_probabilities: list[float] = []
        entry_behavior_probabilities: list[float] = []
        entry_log_support_gaps: list[float] = []
        operation_entry_model_probabilities: dict[str, list[float]] = {
            "upgrade": [],
            "remove": [],
        }
        operation_entry_behavior_probabilities: dict[str, list[float]] = {
            "upgrade": [],
            "remove": [],
        }
        entry_support_satisfied_labels = 0
        entry_support_singleton_suppressed_labels = 0
        lifecycle_counts = {
            TransactionLifecycleOutcome.COMMITTED: 0,
            TransactionLifecycleOutcome.CANCELLED: 0,
            TransactionLifecycleOutcome.UNRESOLVED: 0,
            TransactionLifecycleOutcome.DEADLOCK: 0,
        }
        for trace_index, trace in enumerate(traces):
            configured_burn_in = self.transaction_config.burn_in_steps
            if trace.burn_in_steps > configured_burn_in:
                raise ValueError("transaction trace exceeds configured burn-in length")
            if trace.start_step > 0 and trace.burn_in_steps != configured_burn_in:
                raise ValueError("non-initial transaction context must provide the configured burn-in")
            if trace.initial_recurrent_state.shape != (self.model.config.recurrent_hidden_dim,):
                raise ValueError("transaction recurrent state differs from model hidden size")
            hidden = torch.from_numpy(trace.initial_recurrent_state.copy()).to(
                device=self.device,
                dtype=next(self.model.parameters()).dtype,
            )[None, :]
            policy_targets = {item.step_index: item.target for item in factual_transaction_policy_targets(trace)}
            trace_policy_losses: list[Tensor] = []
            entry_output: RecurrentCandidateOutput | None = None
            for step_index, step in enumerate(trace.steps):
                step.snapshot.validate(
                    expected_config=self.encoder.config,
                    expected_fingerprint=fingerprint,
                )
                encoded = collate_encoded_snapshots(
                    (step.snapshot,),
                    expected_config=self.encoder.config,
                    expected_fingerprint=fingerprint,
                    device=self.device,
                )
                output = self.model(encoded, hidden, validate=False)
                hidden = output.recurrent_state
                if (
                    trace.lifecycle is not None
                    and step_index == trace.lifecycle.entry_step_index
                ):
                    entry_output = output
                if step_index + 1 == trace.burn_in_steps:
                    hidden = hidden.detach()
                if step_index < trace.burn_in_steps:
                    continue
                if (
                    output.candidate_effect_logits is None
                    or output.selection_delta_logits is None
                    or output.transaction_q_values is None
                ):
                    raise RuntimeError("transaction-enabled learner received a headless model")
                action_index = step.action_index
                effect_logits.append(output.candidate_effect_logits[0, action_index])
                effect_targets.append(int(step.effect))
                delta_logits.append(output.selection_delta_logits[0, action_index])
                delta_targets.append(selection_delta_index(step.selected_count_delta))
                # Both pairwise outcome ranking and factual liveness supervise
                # normalized legal-candidate preference, never a free logit
                # offset or a fabricated unexecuted action target.
                policy_log_probabilities = output.policy_log_probabilities()
                selected_policy_log_probability = policy_log_probabilities[0, action_index]
                selected_policy_log_probabilities[(trace_index, step_index)] = selected_policy_log_probability
                policy_target = policy_targets.get(step_index)
                if policy_target is TransactionPolicyTarget.PREFER:
                    trace_policy_losses.append(-selected_policy_log_probability)
                    completion_policy_preferred_labels += 1
                elif policy_target is TransactionPolicyTarget.AVOID:
                    other_action_indices = [
                        index
                        for index, enabled in enumerate(step.snapshot.action_mask)
                        if bool(enabled) and index != action_index
                    ]
                    if not other_action_indices:  # pragma: no cover - target invariant
                        raise RuntimeError("transaction avoid target has no legal alternative")
                    other_indices = torch.tensor(
                        other_action_indices,
                        device=self.device,
                        dtype=torch.long,
                    )
                    trace_policy_losses.append(
                        -torch.logsumexp(
                            policy_log_probabilities[0, other_indices],
                            dim=0,
                        )
                    )
                    completion_policy_avoided_labels += 1
                lifecycle_owns_entry_q = bool(
                    trace.lifecycle is not None
                    and trace.lifecycle.support_eligible
                    and step_index == trace.lifecycle.entry_step_index
                )
                if step.q_observed and not lifecycle_owns_entry_q:
                    if step.transaction_return is None:  # pragma: no cover - property invariant
                        raise RuntimeError("q_observed transaction has no return")
                    lifecycle_target = trace.lifecycle
                    if (
                        lifecycle_target is not None
                        and lifecycle_target.support_eligible
                        and lifecycle_target.option_target_observed
                        and lifecycle_target.option_return is not None
                        and lifecycle_target.entry_step_index
                        < step_index
                        <= lifecycle_target.exit_step_index
                        and step.selected_count_delta > 0
                    ):
                        # Card-target selection steps inside a verified
                        # lifecycle learn Q against the SAME factual SMDP
                        # option return as the entry (through the next
                        # rest-site/Act/terminal boundary) instead of the
                        # whole-episode transaction return.  This is the
                        # target-level short-horizon credit: Q(select A) and
                        # Q(select B) now differ on a 10-20 floor factual
                        # horizon.  Non-lifecycle q_observed steps keep their
                        # original whole-horizon labels.
                        q_values.append(output.transaction_q_values[0, action_index])
                        q_targets.append(float(lifecycle_target.option_return))
                    else:
                        q_values.append(output.transaction_q_values[0, action_index])
                        q_targets.append(step.transaction_return)
            if trace_policy_losses:
                # Equal trace weight prevents a long repeated cycle from
                # overwhelming many short, factual completion paths.
                completion_policy_trace_losses.append(torch.stack(trace_policy_losses).mean())

            lifecycle = trace.lifecycle
            if lifecycle is not None:
                lifecycle_counts[lifecycle.outcome] += 1
                if lifecycle.support_eligible:
                    if entry_output is None:
                        raise RuntimeError(
                            "committed transaction lifecycle entry was not replayed"
                        )
                    if entry_output.transaction_q_values is None:
                        raise RuntimeError(
                            "transaction lifecycle learner received a headless model"
                        )
                    entry_action_index = trace.steps[
                        lifecycle.entry_step_index
                    ].action_index
                    entry_policy_log_probabilities = (
                        entry_output.policy_log_probabilities()[0].float()
                    )
                    entry_log_probability = entry_policy_log_probabilities[
                        entry_action_index
                    ]
                    entry_snapshot = trace.steps[
                        lifecycle.entry_step_index
                    ].snapshot
                    alternative_action_indices = [
                        index
                        for index, enabled in enumerate(entry_snapshot.action_mask)
                        if bool(enabled) and index != entry_action_index
                    ]
                    if not alternative_action_indices:
                        # A committed lifecycle may factually begin at a forced
                        # singleton decision (for example an event that opens
                        # the upgrade selection with no other legal action).
                        # The two-sided corridor is undefined without a legal
                        # alternative, and forced singleton actions never
                        # generate a policy target.  Emit no support-corridor
                        # label; the factual SMDP entry value labels below
                        # remain valid value supervision.
                        entry_support_singleton_suppressed_labels += 1
                    else:
                        alternative_indices = torch.tensor(
                            alternative_action_indices,
                            device=self.device,
                            dtype=torch.long,
                        )
                        alternative_log_probability = torch.logsumexp(
                            entry_policy_log_probabilities[alternative_indices],
                            dim=0,
                        )
                        support_loss, log_gap = two_sided_policy_support_loss(
                            entry_log_probability,
                            alternative_log_probability,
                            probability_floor=(
                                self.transaction_config.lifecycle_entry_support_probability_floor
                            ),
                        )
                        entry_support_terms.append(support_loss)
                        detached_gap = float(log_gap.detach().item())
                        entry_support_satisfied_labels += int(detached_gap <= 1.0e-7)
                        entry_log_support_gaps.append(detached_gap)
                        entry_model_probabilities.append(
                            float(torch.exp(entry_log_probability.detach()).item())
                        )
                        entry_collection_model_probabilities.append(
                            lifecycle.entry_model_probability
                        )
                        entry_behavior_probabilities.append(
                            math.exp(lifecycle.entry_behavior_log_probability)
                        )
                        operation_entry_model_probabilities[lifecycle.operation].append(
                            entry_model_probabilities[-1]
                        )
                        operation_entry_behavior_probabilities[lifecycle.operation].append(
                            entry_behavior_probabilities[-1]
                        )

                    if not lifecycle.option_target_observed:
                        if (
                            self.transaction_config.lifecycle_smdp_horizon
                            == "next_rest_or_act"
                        ):
                            # A committed transaction near a censored episode
                            # end may never observe the requested next macro
                            # boundary.  Keep its factual support-corridor
                            # label, but do not fabricate or bootstrap an
                            # option-Q target.
                            continue
                        raise RuntimeError(
                            "committed transaction lifecycle has no SMDP option target"
                        )
                    option_return = lifecycle.option_return
                    option_discount = lifecycle.option_discount
                    if option_return is None or option_discount is None:  # pragma: no cover - DTO invariant
                        raise RuntimeError(
                            "committed transaction lifecycle has an incomplete SMDP target"
                        )
                    if option_discount == 0.0:
                        post_value = torch.zeros(
                            (),
                            device=self.device,
                            dtype=torch.float32,
                        )
                    else:
                        if lifecycle.post_snapshot is None:  # pragma: no cover - DTO invariant
                            raise RuntimeError(
                                "non-terminal transaction lifecycle lost its post snapshot"
                            )
                        lifecycle.post_snapshot.validate(
                            expected_config=self.encoder.config,
                            expected_fingerprint=fingerprint,
                        )
                        post_encoded = collate_encoded_snapshots(
                            (lifecycle.post_snapshot,),
                            expected_config=self.encoder.config,
                            expected_fingerprint=fingerprint,
                            device=self.device,
                        )
                        with torch.no_grad():
                            post_output = self.model(
                                post_encoded,
                                hidden.detach(),
                                validate=False,
                            )
                            post_value = post_output.value[0].float()
                    smdp_target = torch.as_tensor(
                        option_return,
                        device=self.device,
                        dtype=torch.float32,
                    ) + option_discount * post_value
                    smdp_q_predictions.append(
                        entry_output.transaction_q_values[
                            0, entry_action_index
                        ].float()
                    )
                    smdp_q_targets.append(smdp_target.detach())

        effect_loss = (
            F.cross_entropy(
                torch.stack(effect_logits),
                torch.tensor(effect_targets, device=self.device, dtype=torch.long),
            )
            if effect_logits
            else zero
        )
        delta_loss = (
            F.cross_entropy(
                torch.stack(delta_logits),
                torch.tensor(delta_targets, device=self.device, dtype=torch.long),
            )
            if delta_logits
            else zero
        )
        if q_values:
            q_loss = F.smooth_l1_loss(
                torch.stack(q_values).float(),
                torch.tensor(q_targets, device=self.device, dtype=torch.float32),
            )
        else:
            q_loss = zero

        pairs = (
            observed_outcome_pairs(
                traces,
                minimum_return_gap=self.transaction_config.minimum_return_gap,
            )[: self.transaction_config.maximum_pairs]
            if self.transaction_config.pairwise_ranking_weight > 0.0
            else ()
        )
        if pairs:
            better = torch.stack(
                [selected_policy_log_probabilities[(pair.better_trace, pair.better_step)] for pair in pairs]
            ).float()
            worse = torch.stack(
                [selected_policy_log_probabilities[(pair.worse_trace, pair.worse_step)] for pair in pairs]
            ).float()
            pairwise_loss = F.softplus(self.transaction_config.pairwise_margin - (better - worse)).mean()
        else:
            pairwise_loss = zero
        completion_policy_loss = (
            torch.stack(completion_policy_trace_losses).mean() if completion_policy_trace_losses else zero
        )
        completion_policy_labels = completion_policy_preferred_labels + completion_policy_avoided_labels
        entry_support_loss = (
            torch.stack(entry_support_terms).mean()
            if entry_support_terms
            else zero
        )
        smdp_q_loss = (
            F.smooth_l1_loss(
                torch.stack(smdp_q_predictions),
                torch.stack(smdp_q_targets),
            )
            if smdp_q_predictions
            else zero
        )

        def mean_or_zero(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        return _TransactionLossBatch(
            effect_loss=effect_loss,
            delta_loss=delta_loss,
            q_loss=q_loss,
            pairwise_loss=pairwise_loss,
            completion_policy_loss=completion_policy_loss,
            entry_support_loss=entry_support_loss,
            smdp_q_loss=smdp_q_loss,
            effect_labels=len(effect_targets),
            q_labels=len(q_targets),
            pair_count=len(pairs),
            policy_labels=completion_policy_labels,
            policy_preferred_labels=completion_policy_preferred_labels,
            policy_avoided_labels=completion_policy_avoided_labels,
            entry_support_labels=len(entry_support_terms),
            entry_support_satisfied_labels=entry_support_satisfied_labels,
            entry_support_singleton_suppressed_labels=(
                entry_support_singleton_suppressed_labels
            ),
            smdp_q_labels=len(smdp_q_predictions),
            lifecycle_committed=lifecycle_counts[
                TransactionLifecycleOutcome.COMMITTED
            ],
            lifecycle_cancelled=lifecycle_counts[
                TransactionLifecycleOutcome.CANCELLED
            ],
            lifecycle_unresolved=lifecycle_counts[
                TransactionLifecycleOutcome.UNRESOLVED
            ],
            lifecycle_deadlock=lifecycle_counts[
                TransactionLifecycleOutcome.DEADLOCK
            ],
            entry_model_probability_mean=mean_or_zero(
                entry_model_probabilities
            ),
            entry_collection_model_probability_mean=mean_or_zero(
                entry_collection_model_probabilities
            ),
            entry_behavior_probability_mean=mean_or_zero(
                entry_behavior_probabilities
            ),
            entry_log_support_gap_mean=mean_or_zero(
                entry_log_support_gaps
            ),
            upgrade_entry_support_labels=len(
                operation_entry_model_probabilities["upgrade"]
            ),
            remove_entry_support_labels=len(
                operation_entry_model_probabilities["remove"]
            ),
            upgrade_entry_model_probability_mean=mean_or_zero(
                operation_entry_model_probabilities["upgrade"]
            ),
            remove_entry_model_probability_mean=mean_or_zero(
                operation_entry_model_probabilities["remove"]
            ),
            upgrade_entry_behavior_probability_mean=mean_or_zero(
                operation_entry_behavior_probabilities["upgrade"]
            ),
            remove_entry_behavior_probability_mean=mean_or_zero(
                operation_entry_behavior_probabilities["remove"]
            ),
        )


__all__ = [
    "LearnerMetrics",
    "LearnerTimings",
    "LivenessCreditLosses",
    "LivenessGroupLabel",
    "LivenessLabelManifest",
    "LivenessLabelRow",
    "LivenessReplayWork",
    "VTraceLearner",
    "compile_liveness_label_manifest",
    "liveness_credit_losses",
    "one_sided_policy_support_loss",
    "two_sided_policy_support_loss",
]
