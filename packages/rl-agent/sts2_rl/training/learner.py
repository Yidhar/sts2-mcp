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
from sts2_rl.encoding.snapshot import (
    EncodedDecisionSnapshot,
    collate_encoded_snapshots,
)
from sts2_rl.models import (
    MACRO_ECONOMIC_SURFACE_CARD_REWARD,
    MACRO_ECONOMIC_SURFACE_REMOVAL_SELECTION,
    MACRO_ECONOMIC_SURFACE_REST,
    MACRO_ECONOMIC_SURFACE_SHOP,
    MACRO_ECONOMIC_SURFACE_UPGRADE_SELECTION,
    RecurrentCandidateModel,
    RecurrentCandidateOutput,
)

from .config import (
    EpisodicLearningConfig,
    FailureCreditConfig,
    OptimizationConfig,
    TransactionLearningConfig,
)
from .episode_replay import (
    HorizonTargets,
    ReplaySequence,
)
from .failure_credit.contracts import (
    CreditPlan,
    EvidenceStratum,
    LearningContext,
)
from .transaction import (
    TransactionLifecycleOutcome,
    TransactionTrace,
    observed_outcome_pairs,
    selection_delta_index,
)

# The v3 payload is intentionally empty apart from its version: the v33 macro
# entropy collapse breaker and its counters were retired.  The versioned
# container remains so exact resume keeps failing closed on any older payload
# instead of silently reinterpreting retired breaker state.
_LEARNER_DYNAMICS_STATE_VERSION = "sts2-vtrace-learner-dynamics-v3"

# Execution-only liveness graph bounds. They do not change sampled records,
# labels, loss weights or the equal-record objective, so exact resume may adopt
# this safer execution plan without changing training lineage. A single record
# is never truncated: an over-budget record becomes an auditable singleton.
_LIVENESS_AUTOGRAD_PACKING_VERSION = 1
_LIVENESS_AUTOGRAD_MAX_STEPS = 256
_LIVENESS_AUTOGRAD_MAX_CANDIDATES = 1_024
_LIVENESS_AUTOGRAD_MAX_SEGMENTS = 16

_MACRO_SURFACE_NAME_BY_ID = {
    MACRO_ECONOMIC_SURFACE_REST: "rest_site",
    MACRO_ECONOMIC_SURFACE_SHOP: "shop",
    MACRO_ECONOMIC_SURFACE_CARD_REWARD: "card_reward",
    MACRO_ECONOMIC_SURFACE_UPGRADE_SELECTION: "upgrade_selection",
    MACRO_ECONOMIC_SURFACE_REMOVAL_SELECTION: "removal_selection",
}


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
    transaction_smdp_q_loss: float
    transaction_advantage_policy_loss: float
    transaction_macro_option_value_loss: float
    transaction_macro_option_actor_loss: float
    transaction_traces: int
    transaction_effect_labels: int
    transaction_q_labels: int
    transaction_pairs: int
    transaction_smdp_q_labels: int
    transaction_advantage_policy_labels: int
    transaction_advantage_policy_positive_labels: int
    transaction_advantage_policy_negative_labels: int
    transaction_advantage_policy_phase_suppressed_labels: int
    transaction_advantage_policy_lag_suppressed_labels: int
    transaction_advantage_policy_q_error_suppressed_labels: int
    transaction_advantage_policy_drift_suppressed_labels: int
    transaction_advantage_policy_singleton_suppressed_labels: int
    transaction_advantage_mean: float
    transaction_macro_option_value_labels: int
    transaction_macro_option_actor_labels: int
    transaction_macro_option_actor_lag_suppressed_labels: int
    transaction_macro_option_actor_drift_suppressed_labels: int
    transaction_macro_option_actor_singleton_suppressed_labels: int
    transaction_macro_option_advantage_mean: float
    transaction_macro_option_weight_mean: float
    transaction_macro_option_weight_max: float
    transaction_macro_option_value_labels_by_surface: dict[str, int]
    transaction_macro_option_actor_labels_by_surface: dict[str, int]
    transaction_lifecycle_committed: int
    transaction_lifecycle_cancelled: int
    transaction_lifecycle_unresolved: int
    transaction_lifecycle_deadlock: int
    liveness_credit_loss: float
    liveness_cost_critic_loss: float
    liveness_value_critic_loss: float
    liveness_q_critic_loss: float
    liveness_credit_plans: int
    liveness_cost_labels: int
    liveness_value_labels: int
    liveness_q_labels: int
    liveness_censored_suppressed_labels: int
    liveness_critic_gradient_norm: float
    liveness_gradient_norm_before_clip: float
    liveness_gradient_norm_after_clip: float
    liveness_gradient_clip_scale: float
    liveness_head_calibration_active: int
    liveness_replayed_contexts: int
    liveness_replayed_steps: int
    liveness_replayed_candidates: int
    liveness_autograd_microbatches: int
    liveness_autograd_segments: int
    episodic_loss: float
    episodic_task_value_loss: float
    episodic_revival_value_loss: float
    episodic_combat_hp_loss_value_loss: float
    episodic_sequences: int
    episodic_burn_in_steps: int
    episodic_learn_steps: int
    episodic_task_value_labels: int
    episodic_combat_hp_loss_value_labels: int
    episodic_revival_value_labels: int
    timings: LearnerTimings

    def to_mapping(
        self,
    ) -> dict[str, float | int | dict[str, float | int]]:
        payload: dict[str, float | int | dict[str, float | int]] = asdict(self)
        payload["batch_environment_steps"] = payload.pop("environment_steps")
        payload["timings"] = self.timings.to_mapping()
        return payload


@dataclass(frozen=True, slots=True)
class _TransactionLossBatch:
    effect_loss: Tensor
    delta_loss: Tensor
    q_loss: Tensor
    pairwise_loss: Tensor
    smdp_q_loss: Tensor
    advantage_policy_loss: Tensor
    macro_option_value_loss: Tensor
    macro_option_actor_loss: Tensor
    effect_labels: int
    q_labels: int
    pair_count: int
    smdp_q_labels: int
    advantage_policy_labels: int
    advantage_policy_positive_labels: int
    advantage_policy_negative_labels: int
    advantage_policy_phase_suppressed_labels: int
    advantage_policy_lag_suppressed_labels: int
    advantage_policy_q_error_suppressed_labels: int
    advantage_policy_drift_suppressed_labels: int
    advantage_policy_singleton_suppressed_labels: int
    advantage_mean: float
    macro_option_value_labels: int
    macro_option_actor_labels: int
    macro_option_actor_lag_suppressed_labels: int
    macro_option_actor_drift_suppressed_labels: int
    macro_option_actor_singleton_suppressed_labels: int
    macro_option_advantage_mean: float
    macro_option_weight_mean: float
    macro_option_weight_max: float
    macro_option_value_labels_by_surface: dict[str, int]
    macro_option_actor_labels_by_surface: dict[str, int]
    lifecycle_committed: int
    lifecycle_cancelled: int
    lifecycle_unresolved: int
    lifecycle_deadlock: int


@dataclass(frozen=True, slots=True)
class LivenessCreditLosses:
    """Independent factual liveness value/cost critic losses.

    The helper that produces this bundle deliberately receives model outputs,
    not game-specific failure records.  A versioned credit-plan/replay adapter
    can therefore resolve factual ``decision_id`` references into active-shape
    snapshots without coupling this learner to collector internals.  Since v20
    the plane trains only the state value and candidate cost critics; every
    liveness policy-actor channel was retired.
    """

    value_critic_loss: Tensor
    critic_loss: Tensor
    value_labels: int
    critic_labels: int
    censored_suppressed_labels: int
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
    """Pure DTO critic-mask decision for one recurrent replay row."""

    context_id: str
    step_index: int
    decision_id: str
    censored: bool
    legal_candidates: int
    value_critic_requested: bool
    q_critic_requested: bool

    @property
    def key(self) -> tuple[str, int]:
        return (self.context_id, self.step_index)


@dataclass(frozen=True, slots=True)
class LivenessLabelManifest:
    """Auditable phase, masks and work budget compiled without a model."""

    learner_update: int
    calibration_active: bool
    rows: tuple[LivenessLabelRow, ...]
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
    censored: bool
    legal_candidates: int
    value_critic_requested: bool = False
    q_critic_requested: bool = False

    def freeze(self) -> LivenessLabelRow:
        return LivenessLabelRow(
            context_id=self.context_id,
            step_index=self.step_index,
            decision_id=self.decision_id,
            censored=self.censored,
            legal_candidates=self.legal_candidates,
            value_critic_requested=self.value_critic_requested,
            q_critic_requested=self.q_critic_requested,
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
                censored=False,
                legal_candidates=int(step.snapshot.action_mask.sum()),
            )

    def builder(
        context: LearningContext,
        step_index: int,
    ) -> _LivenessLabelRowBuilder:
        return builders[(context.context_id, step_index)]

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
        rows=rows,
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
    candidate_liveness_cost_values: Tensor,
    liveness_cost_values: Tensor | None = None,
    action_mask: Tensor,
    selected_action_indices: Tensor,
    value_targets: Tensor | None = None,
    value_critic_mask: Tensor | None = None,
    risk_targets: Tensor | None = None,
    risk_critic_mask: Tensor | None = None,
    censored_mask: Tensor | None = None,
) -> LivenessCreditLosses:
    """Build factual liveness critic losses over one active-shape batch.

    ``risk_critic_mask`` controls factual selected-action cost supervision and
    ``value_critic_mask`` controls the candidate-independent state cost head.
    Forced decisions may still train both critics because their factual
    outcome remains authoritative.  ``censored_mask`` suppresses every
    liveness target.  Since v20 this function owns no policy-actor objective:
    the risk/direct/cycle/contrast/completion channels were retired.
    """

    if candidate_liveness_cost_values.ndim != 2:
        raise ValueError("candidate_liveness_cost_values must have shape [N, A]")
    rows, candidates = candidate_liveness_cost_values.shape
    if rows <= 0 or candidates <= 0:
        raise ValueError("liveness decision batch must be non-empty")
    expected = (rows, candidates)
    if action_mask.shape != expected or action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be a bool tensor matching the candidate cost shape")
    device = candidate_liveness_cost_values.device
    for label, value in (
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
    critic_eligible = critic_requested & (~censored)
    value_eligible = value_requested & (~censored)

    zero = (
        candidate_liveness_cost_values.sum() * 0.0
        + (
            liveness_cost_values.sum() * 0.0
            if liveness_cost_values is not None
            else candidate_liveness_cost_values.sum() * 0.0
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

    censored_suppressed = int(
        ((value_requested & censored).sum() + (critic_requested & censored).sum()).item()
    )
    return LivenessCreditLosses(
        value_critic_loss=value_critic_loss,
        critic_loss=critic_loss,
        value_labels=int(value_eligible.sum().item()),
        critic_labels=int(critic_eligible.sum().item()),
        censored_suppressed_labels=censored_suppressed,
    )


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _require_finite(stage: str, values: tuple[tuple[str, Tensor], ...]) -> None:
    invalid = [name for name, value in values if not bool(torch.isfinite(value).all().item())]
    if invalid:
        raise FloatingPointError(f"non-finite learner {stage}: {', '.join(invalid)}")


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
    task_value_loss: Tensor
    revival_value_loss: Tensor
    combat_hp_loss_value_loss: Tensor
    burn_in_steps: int
    learn_steps: int
    task_value_labels: int
    combat_hp_loss_value_labels: int
    revival_value_labels: int


@dataclass(frozen=True, slots=True)
class _LivenessReplayDecision:
    candidate_cost_values: Tensor
    state_cost_value: Tensor
    action_mask: Tensor
    selected_action_index: int


def _empty_liveness_credit_losses(reference: Tensor) -> LivenessCreditLosses:
    zero = reference.sum() * 0.0
    return LivenessCreditLosses(
        value_critic_loss=zero,
        critic_loss=zero,
        value_labels=0,
        critic_labels=0,
        censored_suppressed_labels=0,
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

    return LivenessCreditLosses(
        value_critic_loss=mean_tensor("value_critic_loss"),
        critic_loss=mean_tensor("critic_loss"),
        value_labels=sum(batch.value_labels for batch in batches),
        critic_labels=sum(batch.critic_labels for batch in batches),
        censored_suppressed_labels=sum(batch.censored_suppressed_labels for batch in batches),
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

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def dynamics_state_dict(self) -> dict[str, int | str]:
        """Return non-parameter learner dynamics required for exact resume."""

        return {
            "version": _LEARNER_DYNAMICS_STATE_VERSION,
        }

    @staticmethod
    def validate_dynamics_state_dict(payload: object) -> dict[str, int | str]:
        if not isinstance(payload, dict):
            raise TypeError("learner dynamics state must be an object")
        if set(payload) != {"version"}:
            raise ValueError("learner dynamics state keys mismatch")
        if payload["version"] != _LEARNER_DYNAMICS_STATE_VERSION:
            raise ValueError("unsupported learner dynamics state")
        return {
            "version": _LEARNER_DYNAMICS_STATE_VERSION,
        }

    def load_dynamics_state_dict(self, payload: object) -> None:
        self.validate_dynamics_state_dict(payload)

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
                costs = output.candidate_liveness_cost_values
                state_cost = output.liveness_cost_value
                for row, (identity, step) in enumerate(zip(identities, steps, strict=True)):
                    hidden_by_context[identity] = output.recurrent_state[row : row + 1]
                    if step_index not in needed_steps[identity]:
                        continue
                    if costs is None or state_cost is None:
                        raise RuntimeError("liveness model output omitted value/Q costs")
                    candidate_count = step.snapshot.candidate_count
                    replayed[(identity, step_index)] = _LivenessReplayDecision(
                        candidate_cost_values=costs[row, :candidate_count],
                        state_cost_value=state_cost[row],
                        action_mask=encoded.candidates.action_mask[row, :candidate_count],
                        selected_action_index=step.action_index,
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
            maximum_candidates = max(decision.candidate_cost_values.shape[0] for decision in decisions)

            def padded(value: Tensor, *, fill: float) -> Tensor:
                missing = maximum_candidates - value.shape[0]
                return F.pad(value, (0, missing), value=fill) if missing else value

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
            rows = len(decisions)
            # Targets and masks are immutable compiler facts. Assemble them on
            # the host and transfer each vector once; probing GPU tensors with
            # ``.item()`` inside a 256-step Python loop serialized ROCm replay.
            censored_values = [False] * rows
            risk_target_values = [0.0] * rows
            value_target_values = [0.0] * rows
            value_critic_values = [False] * rows
            risk_critic_values = [False] * rows
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

            def bool_tensor(values: list[bool]) -> Tensor:
                return torch.tensor(values, device=self.device, dtype=torch.bool)

            losses = liveness_credit_losses(
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
                censored_mask=bool_tensor(censored_values),
            )
            return replace(
                losses,
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
        # Policy-health/importance telemetry excludes singleton forced steps;
        # their importance ratio is mechanically one.
        active_ratios = ratios[policy_decisions]
        clipped = (active_ratios > self.config.vtrace_rho_clip).float()
        active_normalized_entropies = normalized_entropies[policy_decisions]
        # The explicit annealing schedule is the only entropy control; the v33
        # macro entropy collapse breaker was retired.
        entropy_weight = _annealed_entropy_weight(
            self.config,
            policy_version=schedule_policy_version,
        )
        online_objective_loss = (
            self.config.policy_weight * policy_loss + self.config.value_weight * value_loss - entropy_weight * entropy
        )
        transaction_losses = self._transaction_losses(
            transaction_traces,
            current_policy_version=current_policy_version,
            schedule_learner_update=schedule_learner_update,
        )
        transaction_effect_loss = transaction_losses.effect_loss
        transaction_delta_loss = transaction_losses.delta_loss
        transaction_q_loss = transaction_losses.q_loss
        transaction_pairwise_loss = transaction_losses.pairwise_loss
        transaction_objective_loss = (
            self.transaction_config.effect_weight * (transaction_effect_loss + transaction_delta_loss)
            + self.transaction_config.transaction_q_weight * transaction_q_loss
            + self.transaction_config.pairwise_ranking_weight * transaction_pairwise_loss
            + self.transaction_config.lifecycle_smdp_q_weight
            * transaction_losses.smdp_q_loss
            + self.transaction_config.lifecycle_advantage_policy_weight
            * transaction_losses.advantage_policy_loss
            + self.transaction_config.macro_option_value_weight
            * transaction_losses.macro_option_value_loss
            + self.transaction_config.macro_option_actor_weight
            * transaction_losses.macro_option_actor_loss
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
                ("transaction_smdp_q_loss", transaction_losses.smdp_q_loss),
                (
                    "transaction_advantage_policy_loss",
                    transaction_losses.advantage_policy_loss,
                ),
                (
                    "transaction_macro_option_value_loss",
                    transaction_losses.macro_option_value_loss,
                ),
                (
                    "transaction_macro_option_actor_loss",
                    transaction_losses.macro_option_actor_loss,
                ),
            ),
        )
        target_and_loss_ms = _elapsed_ms(target_started_ns)
        report("targets_complete", target_and_loss_ms=target_and_loss_ms)

        backward_started_ns = time.perf_counter_ns()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        all_model_parameters = tuple(self.model.parameters())
        gradients_before_liveness = _parameter_gradient_snapshot(all_model_parameters)
        liveness_head_parameters = (
            tuple(self.model.candidate_liveness_cost_head.parameters())
            + tuple(self.model.liveness_cost_value_head.parameters())
            if (self.model.candidate_liveness_cost_head is not None and self.model.liveness_cost_value_head is not None)
            else ()
        )
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
            pack_objective = (
                self.failure_credit_config.liveness_value_critic_weight * pack_losses.value_critic_loss
                + self.failure_credit_config.liveness_cost_critic_weight * pack_losses.critic_loss
            )
            _require_finite(
                "liveness targets/loss",
                (
                    (
                        "liveness_value_critic_loss",
                        pack_losses.value_critic_loss,
                    ),
                    ("liveness_q_critic_loss", pack_losses.critic_loss),
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
        liveness_credit_loss = (
            self.failure_credit_config.liveness_value_critic_weight * liveness_losses.value_critic_loss
            + self.failure_credit_config.liveness_cost_critic_weight * liveness_losses.critic_loss
        )
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
        )
        episodic_started_ns = time.perf_counter_ns()
        episodic_losses = self._episodic_losses(episodic_sequences)
        if episodic_sequences:
            _require_finite(
                "episodic targets/loss",
                (
                    ("episodic_loss", episodic_losses.total_loss),
                    ("episodic_task_value_loss", episodic_losses.task_value_loss),
                    (
                        "episodic_revival_value_loss",
                        episodic_losses.revival_value_loss,
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
            transaction_smdp_q_loss=float(
                transaction_losses.smdp_q_loss.detach().item()
            ),
            transaction_advantage_policy_loss=float(
                transaction_losses.advantage_policy_loss.detach().item()
            ),
            transaction_macro_option_value_loss=float(
                transaction_losses.macro_option_value_loss.detach().item()
            ),
            transaction_macro_option_actor_loss=float(
                transaction_losses.macro_option_actor_loss.detach().item()
            ),
            transaction_traces=len(transaction_traces),
            transaction_effect_labels=transaction_losses.effect_labels,
            transaction_q_labels=transaction_losses.q_labels,
            transaction_pairs=transaction_losses.pair_count,
            transaction_smdp_q_labels=transaction_losses.smdp_q_labels,
            transaction_advantage_policy_labels=(
                transaction_losses.advantage_policy_labels
            ),
            transaction_advantage_policy_positive_labels=(
                transaction_losses.advantage_policy_positive_labels
            ),
            transaction_advantage_policy_negative_labels=(
                transaction_losses.advantage_policy_negative_labels
            ),
            transaction_advantage_policy_phase_suppressed_labels=(
                transaction_losses.advantage_policy_phase_suppressed_labels
            ),
            transaction_advantage_policy_lag_suppressed_labels=(
                transaction_losses.advantage_policy_lag_suppressed_labels
            ),
            transaction_advantage_policy_q_error_suppressed_labels=(
                transaction_losses.advantage_policy_q_error_suppressed_labels
            ),
            transaction_advantage_policy_drift_suppressed_labels=(
                transaction_losses.advantage_policy_drift_suppressed_labels
            ),
            transaction_advantage_policy_singleton_suppressed_labels=(
                transaction_losses.advantage_policy_singleton_suppressed_labels
            ),
            transaction_advantage_mean=transaction_losses.advantage_mean,
            transaction_macro_option_value_labels=(
                transaction_losses.macro_option_value_labels
            ),
            transaction_macro_option_actor_labels=(
                transaction_losses.macro_option_actor_labels
            ),
            transaction_macro_option_actor_lag_suppressed_labels=(
                transaction_losses.macro_option_actor_lag_suppressed_labels
            ),
            transaction_macro_option_actor_drift_suppressed_labels=(
                transaction_losses.macro_option_actor_drift_suppressed_labels
            ),
            transaction_macro_option_actor_singleton_suppressed_labels=(
                transaction_losses.macro_option_actor_singleton_suppressed_labels
            ),
            transaction_macro_option_advantage_mean=(
                transaction_losses.macro_option_advantage_mean
            ),
            transaction_macro_option_weight_mean=(
                transaction_losses.macro_option_weight_mean
            ),
            transaction_macro_option_weight_max=(
                transaction_losses.macro_option_weight_max
            ),
            transaction_macro_option_value_labels_by_surface=(
                transaction_losses.macro_option_value_labels_by_surface
            ),
            transaction_macro_option_actor_labels_by_surface=(
                transaction_losses.macro_option_actor_labels_by_surface
            ),
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
            liveness_credit_loss=float(liveness_credit_loss.detach().item()),
            liveness_cost_critic_loss=float(
                (liveness_losses.value_critic_loss + liveness_losses.critic_loss).detach().item()
            ),
            liveness_value_critic_loss=float(liveness_losses.value_critic_loss.detach().item()),
            liveness_q_critic_loss=float(liveness_losses.critic_loss.detach().item()),
            liveness_credit_plans=len(credit_plans),
            liveness_cost_labels=liveness_critic_labels,
            liveness_value_labels=liveness_losses.value_labels,
            liveness_q_labels=liveness_losses.critic_labels,
            liveness_censored_suppressed_labels=(liveness_losses.censored_suppressed_labels),
            liveness_critic_gradient_norm=liveness_critic_gradient_norm,
            liveness_gradient_norm_before_clip=(liveness_gradient_norm_before_clip),
            liveness_gradient_norm_after_clip=(liveness_gradient_norm_after_clip),
            liveness_gradient_clip_scale=liveness_gradient_clip_scale,
            liveness_head_calibration_active=int(
                schedule_learner_update < self.failure_credit_config.liveness_head_calibration_updates
            ),
            liveness_replayed_contexts=liveness_losses.replayed_contexts,
            liveness_replayed_steps=liveness_losses.replayed_steps,
            liveness_replayed_candidates=(liveness_losses.replayed_candidates),
            liveness_autograd_microbatches=liveness_autograd_microbatches,
            liveness_autograd_segments=liveness_losses.autograd_segments,
            episodic_loss=float(episodic_losses.total_loss.detach().item()),
            episodic_task_value_loss=float(episodic_losses.task_value_loss.detach().item()),
            episodic_revival_value_loss=float(episodic_losses.revival_value_loss.detach().item()),
            episodic_combat_hp_loss_value_loss=float(
                episodic_losses.combat_hp_loss_value_loss.detach().item()
            ),
            episodic_sequences=len(episodic_sequences),
            episodic_burn_in_steps=episodic_losses.burn_in_steps,
            episodic_learn_steps=episodic_losses.learn_steps,
            episodic_task_value_labels=episodic_losses.task_value_labels,
            episodic_combat_hp_loss_value_labels=(
                episodic_losses.combat_hp_loss_value_labels
            ),
            episodic_revival_value_labels=(episodic_losses.revival_value_labels),
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
            return self._episodic_losses_deterministic(sequences)
        finally:
            self.model.train(was_training)

    def _episodic_losses_deterministic(
        self,
        sequences: tuple[ReplaySequence, ...],
    ) -> _EpisodicLossBatch:
        """Replay complete-episode value labels through one bounded GPU graph.

        The caller holds the model in evaluation mode.  Every sequence first
        reconstructs the current model's split recurrent
        state from its exact sparse prefix under ``torch.no_grad()``.  Only the
        configured contiguous suffix is batched with autograd.  Thus a 30,000
        step source episode can increase CPU storage and no-grad compute, but it
        cannot increase activation memory beyond
        ``len(sequences) * episodic_config.learn_steps``.

        Since v20 this plane owns no policy objective: every observed horizon
        supervises only the candidate-independent task/revival-cost value
        heads (and the optional bounded combat HP-loss head).  Failed and
        stale trajectories therefore remain full-strength factual value
        targets without ever becoming imitation labels.
        """

        zero = next(self.model.parameters()).sum() * 0.0
        if not sequences:
            return _EpisodicLossBatch(
                total_loss=zero,
                task_value_loss=zero,
                revival_value_loss=zero,
                combat_hp_loss_value_loss=zero,
                burn_in_steps=0,
                learn_steps=0,
                task_value_labels=0,
                combat_hp_loss_value_labels=0,
                revival_value_labels=0,
            )

        fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        hidden_rows: list[Tensor] = []
        total_burn_in_steps = 0
        total_learn_steps = 0
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

        hidden = torch.stack(hidden_rows, dim=0)
        maximum_time = max(len(sequence.learn_steps) for sequence in sequences)
        task_predictions: list[Tensor] = []
        task_targets: list[float] = []
        revival_predictions: list[Tensor] = []
        revival_targets: list[float] = []
        combat_hp_loss_predictions: list[Tensor] = []
        combat_hp_loss_targets: list[float] = []

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

            for row, step in enumerate(active_steps):
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
        # cost signal is secondary.  A log1p observation model
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
        total_loss = (
            self.episodic_config.task_value_weight * task_value_loss
            + self.episodic_config.revival_value_weight * revival_value_loss
            + self.episodic_config.combat_hp_loss_value_weight
            * combat_hp_loss_value_loss
        )
        return _EpisodicLossBatch(
            total_loss=total_loss,
            task_value_loss=task_value_loss,
            revival_value_loss=revival_value_loss,
            combat_hp_loss_value_loss=combat_hp_loss_value_loss,
            burn_in_steps=total_burn_in_steps,
            learn_steps=total_learn_steps,
            task_value_labels=len(task_predictions),
            combat_hp_loss_value_labels=len(combat_hp_loss_predictions),
            revival_value_labels=len(revival_predictions),
        )

    def _transaction_losses(
        self,
        traces: tuple[TransactionTrace, ...],
        *,
        current_policy_version: int = 0,
        schedule_learner_update: int = 0,
    ) -> _TransactionLossBatch:
        """Compute factual transaction effect/Q and SMDP option losses."""

        for label, value in (
            ("current_policy_version", current_policy_version),
            ("schedule_learner_update", schedule_learner_update),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"transaction {label} must be a non-negative integer")
        zero = next(self.model.parameters()).sum() * 0.0
        if not traces:
            return _TransactionLossBatch(
                effect_loss=zero,
                delta_loss=zero,
                q_loss=zero,
                pairwise_loss=zero,
                smdp_q_loss=zero,
                advantage_policy_loss=zero,
                macro_option_value_loss=zero,
                macro_option_actor_loss=zero,
                effect_labels=0,
                q_labels=0,
                pair_count=0,
                smdp_q_labels=0,
                advantage_policy_labels=0,
                advantage_policy_positive_labels=0,
                advantage_policy_negative_labels=0,
                advantage_policy_phase_suppressed_labels=0,
                advantage_policy_lag_suppressed_labels=0,
                advantage_policy_q_error_suppressed_labels=0,
                advantage_policy_drift_suppressed_labels=0,
                advantage_policy_singleton_suppressed_labels=0,
                advantage_mean=0.0,
                macro_option_value_labels=0,
                macro_option_actor_labels=0,
                macro_option_actor_lag_suppressed_labels=0,
                macro_option_actor_drift_suppressed_labels=0,
                macro_option_actor_singleton_suppressed_labels=0,
                macro_option_advantage_mean=0.0,
                macro_option_weight_mean=0.0,
                macro_option_weight_max=0.0,
                macro_option_value_labels_by_surface={},
                macro_option_actor_labels_by_surface={},
                lifecycle_committed=0,
                lifecycle_cancelled=0,
                lifecycle_unresolved=0,
                lifecycle_deadlock=0,
            )
        fingerprint = grounding_encoding_identity()["fingerprint_sha256"]
        effect_logits: list[Tensor] = []
        effect_targets: list[int] = []
        delta_logits: list[Tensor] = []
        delta_targets: list[int] = []
        q_values: list[Tensor] = []
        q_targets: list[float] = []
        selected_policy_log_probabilities: dict[tuple[int, int], Tensor] = {}
        smdp_q_predictions: list[Tensor] = []
        smdp_q_targets: list[Tensor] = []
        advantage_policy_terms: list[Tensor] = []
        advantage_values: list[float] = []
        advantage_policy_positive_labels = 0
        advantage_policy_negative_labels = 0
        advantage_policy_phase_suppressed_labels = 0
        advantage_policy_lag_suppressed_labels = 0
        advantage_policy_q_error_suppressed_labels = 0
        advantage_policy_drift_suppressed_labels = 0
        advantage_policy_singleton_suppressed_labels = 0
        macro_option_value_predictions: list[Tensor] = []
        macro_option_value_targets: list[Tensor] = []
        # ``surface -> [(selected macro log-probability, factual advantage)]``.
        # Normalizing within the resource-opportunity surface prevents the much
        # more frequent card-reward rows from setting the scale for scarce shop
        # and rest-site decisions.
        macro_option_actor_rows: dict[str, list[tuple[Tensor, Tensor]]] = {}
        macro_option_actor_lag_suppressed_labels = 0
        macro_option_actor_drift_suppressed_labels = 0
        macro_option_actor_singleton_suppressed_labels = 0
        macro_option_advantages: list[float] = []
        macro_option_weights: list[float] = []
        macro_option_value_labels_by_surface: dict[str, int] = {}
        macro_option_actor_labels_by_surface: dict[str, int] = {}
        lifecycle_counts = {
            TransactionLifecycleOutcome.COMMITTED: 0,
            TransactionLifecycleOutcome.CANCELLED: 0,
            TransactionLifecycleOutcome.UNRESOLVED: 0,
            TransactionLifecycleOutcome.DEADLOCK: 0,
        }

        def add_advantage_policy_target(
            *,
            output: RecurrentCandidateOutput,
            snapshot: EncodedDecisionSnapshot,
            action_index: int,
            target: Tensor,
            collection_model_log_probability: float,
            provenance_policy_version: int,
        ) -> None:
            """Admit one bounded, factual option-advantage actor target.

            Unlike V-trace/importance-weighted imitation, the CE derivative is
            non-zero when the selected policy probability is numerically zero.
            Admission is nevertheless fail-closed: only a calibrated selected
            Q prediction from a recent, collection-compatible policy may move
            the actor.  The target and all gating/strength terms are detached;
            gradients flow only through normalized policy log-probabilities.
            """

            nonlocal advantage_policy_positive_labels
            nonlocal advantage_policy_negative_labels
            nonlocal advantage_policy_phase_suppressed_labels
            nonlocal advantage_policy_lag_suppressed_labels
            nonlocal advantage_policy_q_error_suppressed_labels
            nonlocal advantage_policy_drift_suppressed_labels
            nonlocal advantage_policy_singleton_suppressed_labels

            if self.transaction_config.lifecycle_advantage_policy_weight <= 0.0:
                return
            legal_indices = [
                index
                for index, enabled in enumerate(snapshot.action_mask)
                if bool(enabled)
            ]
            if len(legal_indices) <= 1:
                advantage_policy_singleton_suppressed_labels += 1
                return
            if (
                schedule_learner_update
                < self.transaction_config.lifecycle_advantage_start_update
            ):
                advantage_policy_phase_suppressed_labels += 1
                return
            lag = current_policy_version - provenance_policy_version
            if lag < 0:
                raise ValueError(
                    "transaction option target comes from a future policy version"
                )
            if lag > self.transaction_config.lifecycle_advantage_max_policy_lag:
                advantage_policy_lag_suppressed_labels += 1
                return
            if output.transaction_q_values is None:  # pragma: no cover - caller invariant
                raise RuntimeError("option actor target requires transaction Q values")
            log_policy = output.policy_log_probabilities()[0].float()
            current_log_probability = log_policy[action_index].detach()
            if (
                abs(
                    float(current_log_probability.item())
                    - float(collection_model_log_probability)
                )
                > self.transaction_config.lifecycle_advantage_max_log_probability_shift
            ):
                advantage_policy_drift_suppressed_labels += 1
                return
            selected_q = output.transaction_q_values[0, action_index].float()
            detached_target = target.detach().float()
            if (
                abs(float((selected_q.detach() - detached_target).item()))
                > self.transaction_config.lifecycle_advantage_q_error_gate
            ):
                advantage_policy_q_error_suppressed_labels += 1
                return
            legal = torch.tensor(
                legal_indices,
                device=self.device,
                dtype=torch.long,
            )
            alternatives = legal[legal != action_index]
            if alternatives.numel() == 0:  # pragma: no cover - singleton gate
                raise RuntimeError("option target lost its legal alternative")
            # The baseline must be counterfactual.  Including the selected
            # action makes a probability-one incumbent cancel its own
            # advantage exactly, recreating the absorbing state this bridge
            # exists to remove.  Renormalizing only the other legal actions is
            # stable even when every alternative has exponentially small
            # current probability.
            alternative_weights = torch.softmax(
                log_policy[alternatives].detach(),
                dim=0,
            )
            alternative_q_values = output.transaction_q_values[
                0, alternatives
            ].float().detach()
            baseline = (alternative_weights * alternative_q_values).sum()
            advantage = float((detached_target - baseline).item())
            if abs(advantage) <= 1.0e-8:
                return
            strength = min(
                abs(advantage)
                / self.transaction_config.lifecycle_advantage_temperature,
                self.transaction_config.lifecycle_advantage_clip,
            )
            if advantage > 0.0:
                advantage_policy_terms.append(-strength * log_policy[action_index])
                advantage_policy_positive_labels += 1
            else:
                advantage_policy_terms.append(
                    -strength * torch.logsumexp(log_policy[alternatives], dim=0)
                )
                advantage_policy_negative_labels += 1
            advantage_values.append(advantage)

        def add_macro_option_target(
            *,
            output: RecurrentCandidateOutput,
            snapshot: EncodedDecisionSnapshot,
            action_index: int,
            target: Tensor,
            collection_model_log_probability: float,
            provenance_policy_version: int,
        ) -> None:
            """Add one factual macro-value/AWR row without touching the trunk.

            The baseline is a candidate-independent factual value at the same
            resource-opportunity surface.  It is *not* an estimate for an
            unexecuted alternative.  ``macro_policy_logits`` and
            ``macro_option_value`` are structurally detached from the shared
            state/policy features in the model, so this objective can recover a
            saturated reject branch without displacing combat or target-choice
            competence.
            """

            nonlocal macro_option_actor_lag_suppressed_labels
            nonlocal macro_option_actor_drift_suppressed_labels
            nonlocal macro_option_actor_singleton_suppressed_labels

            if (
                self.transaction_config.macro_option_value_weight <= 0.0
                and self.transaction_config.macro_option_actor_weight <= 0.0
            ):
                return
            if snapshot.macro_economic_surface_id <= 0:
                # Old/non-economic rows must not be silently aliased onto a
                # reviewed macro surface.
                return
            if output.macro_option_value is None or output.macro_policy_logits is None:
                raise RuntimeError(
                    "macro-option learner received a model without v47 heads"
                )
            detached_target = target.detach().float()
            value_prediction = output.macro_option_value[0].float()
            macro_option_value_predictions.append(value_prediction)
            macro_option_value_targets.append(detached_target)
            surface_name = _MACRO_SURFACE_NAME_BY_ID.get(
                snapshot.macro_economic_surface_id
            )
            if surface_name is None:  # pragma: no cover - reviewed-ID invariant
                raise RuntimeError("macro option target has an unknown surface ID")
            macro_option_value_labels_by_surface[surface_name] = (
                macro_option_value_labels_by_surface.get(surface_name, 0) + 1
            )

            if self.transaction_config.macro_option_actor_weight <= 0.0:
                return
            legal_indices = [
                index
                for index, enabled in enumerate(snapshot.action_mask)
                if bool(enabled)
            ]
            if len(legal_indices) <= 1:
                macro_option_actor_singleton_suppressed_labels += 1
                return
            lag = current_policy_version - provenance_policy_version
            if lag < 0:
                raise ValueError(
                    "macro option target comes from a future policy version"
                )
            if lag > self.transaction_config.macro_option_max_policy_lag:
                macro_option_actor_lag_suppressed_labels += 1
                return
            macro_log_policy = output.macro_policy_log_probabilities()[0].float()
            current_log_probability = macro_log_policy[action_index].detach()
            if (
                abs(
                    float(current_log_probability.item())
                    - float(collection_model_log_probability)
                )
                > self.transaction_config.macro_option_max_log_probability_shift
            ):
                macro_option_actor_drift_suppressed_labels += 1
                return
            advantage = detached_target - value_prediction.detach()
            macro_option_actor_rows.setdefault(
                surface_name, []
            ).append((macro_log_policy[action_index], advantage))
            macro_option_actor_labels_by_surface[surface_name] = (
                macro_option_actor_labels_by_surface.get(surface_name, 0) + 1
            )
            macro_option_advantages.append(float(advantage.item()))

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
            entry_output: RecurrentCandidateOutput | None = None
            step_outputs: dict[int, RecurrentCandidateOutput] = {}
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
                step_outputs[step_index] = output
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
                lifecycle_owns_entry_q = bool(
                    trace.lifecycle is not None
                    and trace.lifecycle.support_eligible
                    and step_index == trace.lifecycle.entry_step_index
                )
                if step.q_observed and not lifecycle_owns_entry_q:
                    if step.transaction_return is None:  # pragma: no cover - property invariant
                        raise RuntimeError("q_observed transaction has no return")
                    # Selection-origin option targets need the lifecycle post
                    # bootstrap calculated below. Defer those exact rows;
                    # ordinary factual transaction returns remain immediate.
                    if not step.option_target_observed:
                        q_values.append(output.transaction_q_values[0, action_index])
                        q_targets.append(step.transaction_return)

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
                    entry_snapshot = trace.steps[
                        lifecycle.entry_step_index
                    ].snapshot

                    if not lifecycle.option_target_observed:
                        if (
                            self.transaction_config.lifecycle_smdp_horizon
                            in {
                                "next_rest_or_act",
                                "next_resource_opportunity",
                            }
                        ):
                            # A committed transaction near a censored episode
                            # end may never observe the requested next macro
                            # boundary.  Do not fabricate or bootstrap an
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
                    add_macro_option_target(
                        output=entry_output,
                        snapshot=entry_snapshot,
                        action_index=entry_action_index,
                        target=smdp_target,
                        collection_model_log_probability=trace.steps[
                            lifecycle.entry_step_index
                        ].model_log_probability,
                        provenance_policy_version=(
                            lifecycle.entry_policy_version
                        ),
                    )
                    add_advantage_policy_target(
                        output=entry_output,
                        snapshot=entry_snapshot,
                        action_index=entry_action_index,
                        target=smdp_target,
                        collection_model_log_probability=trace.steps[
                            lifecycle.entry_step_index
                        ].model_log_probability,
                        provenance_policy_version=(
                            lifecycle.entry_policy_version
                        ),
                    )

                    # Target selection rows use their own action-origin return;
                    # the entry-prefix return is never copied onto them.  A
                    # transaction-exit horizon shares only the detached post
                    # value bootstrap, while the production next-rest/Act
                    # horizon is fully observed and therefore has discount 0.
                    for step_index in range(
                        lifecycle.entry_step_index + 1,
                        lifecycle.exit_step_index + 1,
                    ):
                        step = trace.steps[step_index]
                        if not step.option_target_observed:
                            continue
                        if step.selected_count_delta <= 0:
                            raise RuntimeError(
                                "selection-origin option target is not a positive selection"
                            )
                        step_output = step_outputs.get(step_index)
                        if step_output is None or step_output.transaction_q_values is None:
                            raise RuntimeError(
                                "selection-origin option target was not replayed"
                            )
                        if step.option_return is None or step.option_discount is None:
                            raise RuntimeError(
                                "selection-origin option target is incomplete"
                            )
                        step_target = torch.as_tensor(
                            step.option_return,
                            device=self.device,
                            dtype=torch.float32,
                        ) + step.option_discount * post_value
                        q_values.append(
                            step_output.transaction_q_values[
                                0, step.action_index
                            ]
                        )
                        q_targets.append(float(step_target.detach().item()))
                        add_macro_option_target(
                            output=step_output,
                            snapshot=step.snapshot,
                            action_index=step.action_index,
                            target=step_target,
                            collection_model_log_probability=(
                                step.model_log_probability
                            ),
                            provenance_policy_version=trace.policy_version,
                        )
                        add_advantage_policy_target(
                            output=step_output,
                            snapshot=step.snapshot,
                            action_index=step.action_index,
                            target=step_target,
                            collection_model_log_probability=(
                                step.model_log_probability
                            ),
                            provenance_policy_version=trace.policy_version,
                        )

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
        smdp_q_loss = (
            F.smooth_l1_loss(
                torch.stack(smdp_q_predictions),
                torch.stack(smdp_q_targets),
            )
            if smdp_q_predictions
            else zero
        )
        advantage_policy_loss = (
            torch.stack(advantage_policy_terms).mean()
            if advantage_policy_terms
            else zero
        )
        macro_option_value_loss = (
            F.smooth_l1_loss(
                torch.stack(macro_option_value_predictions).float(),
                torch.stack(macro_option_value_targets).float(),
            )
            if macro_option_value_predictions
            else zero
        )
        macro_option_actor_surface_terms: list[Tensor] = []
        macro_option_actor_label_count = 0
        for rows in macro_option_actor_rows.values():
            surface_log_probabilities = torch.stack(
                [log_probability for log_probability, _ in rows]
            ).float()
            surface_advantages = torch.stack(
                [advantage for _, advantage in rows]
            ).float()
            log_weights = (
                surface_advantages
                / self.transaction_config.macro_option_actor_temperature
            ).clamp(
                min=-self.transaction_config.macro_option_actor_log_weight_clip,
                max=self.transaction_config.macro_option_actor_log_weight_clip,
            )
            weights = torch.exp(log_weights)
            # Preserve the absolute factual-advantage scale, including for a
            # surface represented by only one row.  Dividing by the observed
            # mean would turn every singleton into weight 1 and erase exactly
            # the state-level signal this AWR bridge exists to provide.
            # Instead, average rows *within* each reviewed surface and then
            # average surfaces below.  A busy card-reward stream therefore
            # cannot overwhelm a scarce rest or shop opportunity by count.
            macro_option_actor_surface_terms.append(
                (-weights.detach() * surface_log_probabilities).mean()
            )
            macro_option_actor_label_count += len(rows)
            macro_option_weights.extend(
                float(item)
                for item in weights.detach().cpu().tolist()
            )
        macro_option_actor_loss = (
            torch.stack(macro_option_actor_surface_terms).mean()
            if macro_option_actor_surface_terms
            else zero
        )

        def mean_or_zero(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        return _TransactionLossBatch(
            effect_loss=effect_loss,
            delta_loss=delta_loss,
            q_loss=q_loss,
            pairwise_loss=pairwise_loss,
            smdp_q_loss=smdp_q_loss,
            advantage_policy_loss=advantage_policy_loss,
            macro_option_value_loss=macro_option_value_loss,
            macro_option_actor_loss=macro_option_actor_loss,
            effect_labels=len(effect_targets),
            q_labels=len(q_targets),
            pair_count=len(pairs),
            smdp_q_labels=len(smdp_q_predictions),
            advantage_policy_labels=len(advantage_policy_terms),
            advantage_policy_positive_labels=(
                advantage_policy_positive_labels
            ),
            advantage_policy_negative_labels=(
                advantage_policy_negative_labels
            ),
            advantage_policy_phase_suppressed_labels=(
                advantage_policy_phase_suppressed_labels
            ),
            advantage_policy_lag_suppressed_labels=(
                advantage_policy_lag_suppressed_labels
            ),
            advantage_policy_q_error_suppressed_labels=(
                advantage_policy_q_error_suppressed_labels
            ),
            advantage_policy_drift_suppressed_labels=(
                advantage_policy_drift_suppressed_labels
            ),
            advantage_policy_singleton_suppressed_labels=(
                advantage_policy_singleton_suppressed_labels
            ),
            advantage_mean=mean_or_zero(advantage_values),
            macro_option_value_labels=len(macro_option_value_predictions),
            macro_option_actor_labels=macro_option_actor_label_count,
            macro_option_actor_lag_suppressed_labels=(
                macro_option_actor_lag_suppressed_labels
            ),
            macro_option_actor_drift_suppressed_labels=(
                macro_option_actor_drift_suppressed_labels
            ),
            macro_option_actor_singleton_suppressed_labels=(
                macro_option_actor_singleton_suppressed_labels
            ),
            macro_option_advantage_mean=mean_or_zero(macro_option_advantages),
            macro_option_weight_mean=mean_or_zero(macro_option_weights),
            macro_option_weight_max=(
                max(macro_option_weights) if macro_option_weights else 0.0
            ),
            macro_option_value_labels_by_surface=dict(
                sorted(macro_option_value_labels_by_surface.items())
            ),
            macro_option_actor_labels_by_surface=dict(
                sorted(macro_option_actor_labels_by_surface.items())
            ),
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
        )


__all__ = [
    "LearnerMetrics",
    "LearnerTimings",
    "LivenessCreditLosses",
    "LivenessLabelManifest",
    "LivenessLabelRow",
    "LivenessReplayWork",
    "VTraceLearner",
    "compile_liveness_label_manifest",
    "liveness_credit_losses",
]
