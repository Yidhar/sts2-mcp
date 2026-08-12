"""Strict configuration ABI for the relational recurrent V-trace v3 baseline."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

from sts2_baseline import TASK_REWARD_SPEC
from sts2_rl.encoding import GroundedEncodingConfig
from sts2_rl.models import GroundedCandidateConfig

from .seeding import validate_seed_budget

CONFIG_VERSION = "sts2-relational-curriculum-config-v20"
_MODEL_INITIALIZATION_SOURCE_CONFIG_V10 = "sts2-relational-curriculum-config-v10"
_MODEL_INITIALIZATION_SOURCE_CONFIG_V11 = "sts2-relational-curriculum-config-v11"
_MODEL_INITIALIZATION_SOURCE_CONFIG_V12 = "sts2-relational-curriculum-config-v12"
_MODEL_INITIALIZATION_SOURCE_CONFIG_V13 = "sts2-relational-curriculum-config-v13"
_MODEL_INITIALIZATION_SOURCE_CONFIG_V14 = "sts2-relational-curriculum-config-v14"
_MODEL_INITIALIZATION_SOURCE_CONFIG_V15 = "sts2-relational-curriculum-config-v15"
_MODEL_INITIALIZATION_SOURCE_CONFIG_V16 = "sts2-relational-curriculum-config-v16"
_MODEL_INITIALIZATION_SOURCE_CONFIG_V17 = "sts2-relational-curriculum-config-v17"
_MODEL_INITIALIZATION_SOURCE_CONFIG_V18 = "sts2-relational-curriculum-config-v18"
_MODEL_INITIALIZATION_SOURCE_CONFIG_V19 = "sts2-relational-curriculum-config-v19"
ENGINE_REVIVAL_MECHANISM = "engine-bailout-v1"
PROFILE_DIR = Path(__file__).resolve().parents[2] / "config" / "profiles"
T = TypeVar("T")


def engine_revival_identity() -> dict[str, Any]:
    """Return the immutable contract for privileged engine revival."""

    payload: dict[str, Any] = {
        "version": ENGINE_REVIVAL_MECHANISM,
        "activation": "reset.training_revival_budget",
        "scope": "engine-owned run state",
        "model_visible_game_entity": None,
        "native_death_prevention_order": "native hooks before training bailout",
        "forced_kill_policy": "not intercepted",
        "nonpositive_max_hp_policy": "not intercepted",
        "telemetry": ("observation._training.revival_budget+revivals_used+player_hp_lost"),
        "model_input_policy": "underscore training telemetry excluded",
    }
    return payload


def _require_int(value: object, *, label: str, minimum: int | None = None) -> int:
    """Validate a config integer without accepting ``bool`` or lossy coercion."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _require_finite_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate a finite TOML number without treating booleans as numbers."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return normalized


def _require_optional_text(value: object, *, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string or null")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture: str = "relational_candidate_v3"
    token_feature_dim: int = 224
    d_model: int = 128
    n_heads: int = 4
    ffn_dim: int = 384
    world_layers: int = 3
    latent_slots: int = 12
    latent_layers: int = 2
    local_layers: int = 1
    candidate_layers: int = 1
    recurrent_hidden_dim: int = 256
    dropout: float = 0.05
    type_vocab_size: int = 128
    role_vocab_size: int = 64
    owner_vocab_size: int = 128
    entity_vocab_size: int = 8192
    zone_vocab_size: int = 32
    order_vocab_size: int = 128
    domain_count: int = 8
    max_world_tokens: int = 2048
    max_candidates: int = 256
    max_candidate_local_tokens: int = 64

    def __post_init__(self) -> None:
        for name in (
            "token_feature_dim",
            "d_model",
            "n_heads",
            "ffn_dim",
            "world_layers",
            "latent_slots",
            "latent_layers",
            "local_layers",
            "candidate_layers",
            "recurrent_hidden_dim",
            "type_vocab_size",
            "role_vocab_size",
            "owner_vocab_size",
            "entity_vocab_size",
            "zone_vocab_size",
            "order_vocab_size",
            "domain_count",
            "max_world_tokens",
            "max_candidates",
            "max_candidate_local_tokens",
        ):
            _require_int(getattr(self, name), label=f"model.{name}", minimum=1)
        _require_finite_number(
            self.dropout,
            label="model.dropout",
            minimum=0.0,
        )
        if self.architecture != "relational_candidate_v3":
            raise ValueError("only architecture='relational_candidate_v3' is supported")
        # Reuse the model's own shape validation as the single source of truth.
        self.to_model_config()
        if (
            min(
                self.max_world_tokens,
                self.max_candidates,
                self.max_candidate_local_tokens,
            )
            <= 0
        ):
            raise ValueError("model token capacities must be positive")

    def to_model_config(self) -> GroundedCandidateConfig:
        return GroundedCandidateConfig(
            token_feature_dim=self.token_feature_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            ffn_dim=self.ffn_dim,
            world_layers=self.world_layers,
            latent_slots=self.latent_slots,
            latent_layers=self.latent_layers,
            local_layers=self.local_layers,
            candidate_layers=self.candidate_layers,
            recurrent_hidden_dim=self.recurrent_hidden_dim,
            dropout=self.dropout,
            domain_count=self.domain_count,
            type_vocab_size=self.type_vocab_size,
            role_vocab_size=self.role_vocab_size,
            owner_vocab_size=self.owner_vocab_size,
            entity_vocab_size=self.entity_vocab_size,
            zone_vocab_size=self.zone_vocab_size,
            order_vocab_size=self.order_vocab_size,
        )

    def to_encoding_config(self) -> GroundedEncodingConfig:
        return GroundedEncodingConfig.from_model_config(
            self.to_model_config(),
            max_world_tokens=self.max_world_tokens,
            max_candidates=self.max_candidates,
            max_candidate_local_tokens=self.max_candidate_local_tokens,
        )


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_unrolls: int = 8
    discount: float = 0.997
    gradient_clip_norm: float = 1.0
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0
    policy_rho_clip: float = 1.0
    policy_weight: float = 1.0
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    # Entropy is annealed by learner update rather than dropped abruptly.
    # A non-zero floor preserves exploration while allowing deterministic
    # policy margins to emerge as factual liveness supervision accumulates.
    entropy_weight_end: float = 0.01
    entropy_decay_updates: int = 2_000
    # Source-versioned v33 circuit breaker.  Its thresholds and temporary
    # weight are code constants rather than launch-time knobs, so an operator
    # cannot silently change the learning recipe with a CLI override.
    entropy_breaker: Literal[
        "disabled",
        "one-hot-v1",
        "policy-collapse-v2",
    ] = "disabled"

    def __post_init__(self) -> None:
        learning_rate = _require_finite_number(
            self.learning_rate,
            label="optimization.learning_rate",
        )
        weight_decay = _require_finite_number(
            self.weight_decay,
            label="optimization.weight_decay",
        )
        _require_int(
            self.batch_unrolls,
            label="optimization.batch_unrolls",
            minimum=1,
        )
        _require_int(
            self.entropy_decay_updates,
            label="optimization.entropy_decay_updates",
            minimum=1,
        )
        discount = _require_finite_number(
            self.discount,
            label="optimization.discount",
        )
        gradient_clip_norm = _require_finite_number(
            self.gradient_clip_norm,
            label="optimization.gradient_clip_norm",
        )
        if learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        for name in ("vtrace_rho_clip", "vtrace_c_clip", "policy_rho_clip"):
            value = _require_finite_number(
                getattr(self, name),
                label=f"optimization.{name}",
            )
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "policy_weight",
            "value_weight",
            "entropy_weight",
            "entropy_weight_end",
        ):
            value = _require_finite_number(
                getattr(self, name),
                label=f"optimization.{name}",
            )
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.entropy_weight_end > self.entropy_weight:
            raise ValueError("entropy_weight_end cannot exceed entropy_weight")
        if self.entropy_breaker not in {
            "disabled",
            "one-hot-v1",
            "policy-collapse-v2",
        }:
            raise ValueError(
                "optimization.entropy_breaker must be 'disabled', "
                "'one-hot-v1', or 'policy-collapse-v2'"
            )


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Short-lived FIFO data plane; every unroll is consumed at most once."""

    unroll_length: int = 64
    queue_capacity: int = 64
    minimum_unrolls: int = 8
    collector_workers: int = 1
    policy_sync_interval_unrolls: int = 8
    max_policy_lag: int = 64
    # Every Nth training episode is collected greedily on the ordinary even
    # training seed stream.  It remains training data and is never allowed to
    # consume the odd validation or final-audit seed namespaces.
    deterministic_probe_interval_episodes: int = 0
    # Full-run episodes can span thousands of decisions, so an episode-count
    # interval alone may never expose a collapsing greedy policy before the
    # first held-out gate. Each positive environment-step milestone schedules
    # one training-only greedy episode at the first subsequent episode
    # boundary. Milestones crossed by the same long episode are coalesced into
    # one probe; exact resume treats milestones at or below the restored
    # environment step as already observed instead of replaying stale probes.
    deterministic_probe_environment_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "unroll_length",
            "queue_capacity",
            "minimum_unrolls",
            "collector_workers",
            "policy_sync_interval_unrolls",
            "max_policy_lag",
        ):
            _require_int(
                getattr(self, name),
                label=f"rollout.{name}",
                minimum=1,
            )
        _require_int(
            self.deterministic_probe_interval_episodes,
            label="rollout.deterministic_probe_interval_episodes",
            minimum=0,
        )
        milestones = self.deterministic_probe_environment_steps
        if not isinstance(milestones, tuple):
            milestones = tuple(milestones)
            object.__setattr__(
                self,
                "deterministic_probe_environment_steps",
                milestones,
            )
        previous = 0
        for index, step in enumerate(milestones):
            _require_int(
                step,
                label=f"rollout.deterministic_probe_environment_steps[{index}]",
                minimum=1,
            )
            if step <= previous:
                raise ValueError("rollout.deterministic_probe_environment_steps must be " "strictly increasing")
            previous = step
        if self.minimum_unrolls > self.queue_capacity:
            raise ValueError("rollout minimum_unrolls cannot exceed queue_capacity")
        if self.collector_workers != 1:
            raise ValueError("v2 currently requires one collector worker per typed backend session")


@dataclass(frozen=True, slots=True)
class TransactionLearningConfig:
    """Optional factual-outcome replay sidecar for multi-step transactions.

    Disabled remains parameter-compatible with the base model. Enabling this
    section adds model heads and replay state. Changing its factual policy
    objective also changes immutable training semantics. Both require a fresh
    lineage through ``model_parameter_initialization`` rather than exact resume;
    compatible network tensors are inherited, while optimizer/replay/RNG state
    is intentionally reset.
    """

    enabled: bool = False
    replay_capacity: int = 4_096
    replay_byte_capacity: int = 536_870_912
    sample_traces: int = 8
    burn_in_steps: int = 24
    effect_weight: float = 0.10
    transaction_q_weight: float = 0.25
    # Factual completed paths and exact repeated node/action cycles supervise the
    # shared legal-candidate policy directly. This is not a reward, action mask,
    # action rewrite, or card/prompt-specific rule.
    completion_policy_weight: float = 0.25
    # A verified upgrade/removal lifecycle receives a two-sided support
    # corridor at the action that opened the transaction: both the entry and
    # the aggregate of its legal alternatives retain a minimum probability.
    # The loss is exactly zero inside the corridor, so it cannot decide when to
    # upgrade/remove; factual long-horizon return learning owns that preference.
    lifecycle_entry_support_weight: float = 0.0
    lifecycle_entry_support_probability_floor: float = 0.05
    # Semi-Markov entry-Q target: factual reward over the whole transaction,
    # followed by a detached post-transaction value bootstrap. This is not a
    # hand-authored forge/removal bonus.
    lifecycle_smdp_q_weight: float = 0.0
    # The legacy target stops at the transaction exit and bootstraps from the
    # immediately following state. ``next_rest_or_act`` is the legacy
    # rest-only extension. ``next_resource_opportunity`` instead uses only the
    # factual reward sequence through the next opportunity of the same
    # reviewed resource family, Act boundary, or authoritative terminal and
    # deliberately sets the bootstrap discount to zero there.
    lifecycle_smdp_horizon: Literal[
        "transaction_exit",
        "next_rest_or_act",
        "next_resource_opportunity",
    ] = (
        "transaction_exit"
    )
    # A calibrated factual option-Q target can directly move the shared actor
    # even when the selected action has reached a softmax absorbing state.
    # This bounded CE bridge is deliberately opt-in and separately gated by
    # learner age, policy provenance, Q residual and collection/current policy
    # drift.  It is neither an action bonus nor an expert label.
    lifecycle_advantage_policy_weight: float = 0.0
    lifecycle_advantage_start_update: int = 0
    lifecycle_advantage_temperature: float = 0.25
    lifecycle_advantage_clip: float = 1.0
    lifecycle_advantage_q_error_gate: float = 0.25
    lifecycle_advantage_max_policy_lag: int = 128
    lifecycle_advantage_max_log_probability_shift: float = 1.0
    # V47 macro economy path.  The factual state baseline replaces v46's
    # unexecuted-alternative Q baseline.  AWR uses the observed option return
    # and a detached candidate-independent value; both actor and value heads
    # are isolated from the shared tactical trunk by model construction.
    macro_option_value_weight: float = 0.0
    macro_option_actor_weight: float = 0.0
    macro_option_actor_temperature: float = 0.25
    macro_option_actor_log_weight_clip: float = 2.0
    macro_option_max_policy_lag: int = 128
    macro_option_max_log_probability_shift: float = 2.0
    # Completed select/confirm paths teach only the forward semantic branch as
    # a group.  Target-card preference remains owned by factual AWR/Q.
    macro_option_group_completion_weight: float = 0.0
    # Cross-trajectory outcome ranking remains explicit opt-in.  Factual Q,
    # effect and selection-delta heads are safe by default; pairwise policy
    # supervision requires context-equivalent repeated states and must not be
    # inferred merely from superficially similar selection panes.
    pairwise_ranking_weight: float = 0.0
    pairwise_margin: float = 0.10
    minimum_return_gap: float = 0.0
    maximum_pairs: int = 256

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("transaction_learning.enabled must be a boolean")
        for name in (
            "replay_capacity",
            "replay_byte_capacity",
            "sample_traces",
            "maximum_pairs",
        ):
            _require_int(
                getattr(self, name),
                label=f"transaction_learning.{name}",
                minimum=1,
            )
        _require_int(
            self.burn_in_steps,
            label="transaction_learning.burn_in_steps",
            minimum=0,
        )
        for name in (
            "lifecycle_advantage_start_update",
            "lifecycle_advantage_max_policy_lag",
            "macro_option_max_policy_lag",
        ):
            _require_int(
                getattr(self, name),
                label=f"transaction_learning.{name}",
                minimum=0,
            )
        for name in (
            "effect_weight",
            "transaction_q_weight",
            "completion_policy_weight",
            "lifecycle_entry_support_weight",
            "lifecycle_entry_support_probability_floor",
            "lifecycle_smdp_q_weight",
            "lifecycle_advantage_policy_weight",
            "lifecycle_advantage_temperature",
            "lifecycle_advantage_clip",
            "lifecycle_advantage_q_error_gate",
            "lifecycle_advantage_max_log_probability_shift",
            "macro_option_value_weight",
            "macro_option_actor_weight",
            "macro_option_actor_temperature",
            "macro_option_actor_log_weight_clip",
            "macro_option_max_log_probability_shift",
            "macro_option_group_completion_weight",
            "pairwise_ranking_weight",
            "pairwise_margin",
            "minimum_return_gap",
        ):
            _require_finite_number(
                getattr(self, name),
                label=f"transaction_learning.{name}",
                minimum=0.0,
            )
        if not 0.0 < self.lifecycle_entry_support_probability_floor < 0.5:
            raise ValueError(
                "transaction_learning.lifecycle_entry_support_probability_floor "
                "must be strictly between 0 and 0.5"
            )
        if self.lifecycle_smdp_horizon not in {
            "transaction_exit",
            "next_rest_or_act",
            "next_resource_opportunity",
        }:
            raise ValueError(
                "transaction_learning.lifecycle_smdp_horizon must be "
                "transaction_exit, next_rest_or_act or next_resource_opportunity"
            )
        if self.lifecycle_advantage_temperature <= 0.0:
            raise ValueError(
                "transaction_learning.lifecycle_advantage_temperature must be positive"
            )
        if self.lifecycle_advantage_clip <= 0.0:
            raise ValueError(
                "transaction_learning.lifecycle_advantage_clip must be positive"
            )
        if self.macro_option_actor_temperature <= 0.0:
            raise ValueError(
                "transaction_learning.macro_option_actor_temperature must be positive"
            )
        if self.macro_option_actor_log_weight_clip <= 0.0:
            raise ValueError(
                "transaction_learning.macro_option_actor_log_weight_clip must be positive"
            )
        if not self.enabled and (
            self.lifecycle_entry_support_weight > 0.0
            or self.lifecycle_smdp_q_weight > 0.0
            or self.lifecycle_advantage_policy_weight > 0.0
            or self.macro_option_value_weight > 0.0
            or self.macro_option_actor_weight > 0.0
            or self.macro_option_group_completion_weight > 0.0
        ):
            raise ValueError(
                "transaction lifecycle losses require transaction_learning.enabled"
            )


@dataclass(frozen=True, slots=True)
class FailureCreditConfig:
    """Formal detector-evidence, replay-v5 and liveness-learning contract.

    This plane is deliberately independent from the legacy transaction-v3
    sidecar.  ``shadow`` performs the complete semantic/evidence compilation
    and emits funnel metrics but cannot alter learner gradients.  ``learning``
    additionally owns a bounded replay-v5 corpus and enables the independent
    state/candidate liveness heads.  Changing mode is therefore a lineage
    change; it is never an exact-resume-compatible runtime toggle.
    """

    mode: Literal["disabled", "shadow", "learning"] = "disabled"
    replay_capacity: int = 4_096
    # Failure records retain recurrent snapshots plus collision-auditable
    # semantic identities. Production traces have demonstrated records near
    # 90 MiB, so the old 512 MiB default could hold only a handful of worst-case
    # records and continuously evicted completion controls. Match the episodic
    # replay's reviewed host-memory budget instead of silently starving the
    # evidence strata.
    replay_byte_capacity: int = 2_147_483_648
    sample_records: int = 8
    burn_in_steps: int = 32
    maximum_context_steps: int = 256
    # Collector-side completion controls are staged until the authoritative
    # episode boundary.  These two independent limits prevent ordinary
    # progress transitions from retaining an unbounded number of encoded DTOs.
    maximum_episode_completion_controls: int = 32
    maximum_episode_completion_bytes: int = 134_217_728
    # Minimum evidence representation per learner sample. Deficits are
    # explicit metrics, never silently back-filled by unrelated evidence.
    direct_witness_quota: int = 1
    multi_edge_cycle_quota: int = 1
    risk_sequence_quota: int = 1
    # DIRECT/MULTI records may also carry RISK_SEQUENCE.  This independent
    # quota guarantees that unique unresolved stalls are represented rather
    # than letting those overlap records satisfy the whole risk policy.
    unresolved_stall_quota: int = 1
    completion_control_quota: int = 1
    # Matched pairs are optional until the cross-episode matcher has produced
    # a non-empty stratum, but remain an explicit sampling-policy field.
    matched_outcome_pair_quota: int = 0
    # The matcher is stateless, but one terminal publication may expose many
    # retained completion controls. Bound how many failed incidents it may
    # enrich at once so a publication cannot create unbounded compile work.
    maximum_matched_pairs_per_publication: int = 8
    # Critics may use all authoritative evidence. Actor labels older than this
    # policy distance are retained for audit/value learning but suppressed.
    policy_gradient_max_lag: int = 128
    # Newly initialized liveness heads first learn against detached world,
    # recurrent and candidate features.  This keeps their initially
    # uncalibrated gradients out of the inherited policy trunk while still
    # allowing the heads themselves to fit factual zero/one controls.  The
    # checkpointed learner-update counter is the sole phase clock.
    liveness_head_calibration_updates: int = 256
    # The centered-risk actor consumes predictions from the new candidate
    # cost head, so it starts only after the detached-head calibration phase.
    # Direct/cycle/contrast witnesses do not depend on that head and remain
    # independently eligible throughout calibration.
    liveness_risk_actor_start_update: int = 512
    # Failure-credit records may share one recurrent replay/autograd graph.
    # The learner still reduces losses per record before averaging, so this is
    # an execution-only packing limit rather than a change to evidence weight.
    # A value of one preserves the original bounded-memory execution ABI.
    liveness_records_per_autograd_batch: int = 1
    # Detach recurrent state at each trainable window boundary. Together with
    # the bounded record pack this limits graph depth without discarding any
    # factual target or shortening the no-grad recurrent reconstruction.
    liveness_tbptt_window_steps: int = 16
    # Matched outcome evidence may add one comparison context to the incident
    # context.  More contexts in one record are rejected until a reviewed
    # learner ABI explicitly raises this bound.
    liveness_maximum_contexts_per_record: int = 2
    # Aggregate hard work budgets cover the sampled record set.  They are
    # checked before the first model forward and fail closed rather than
    # partially training or silently dropping replay evidence.
    liveness_maximum_replayed_steps_per_update: int = 4_096
    liveness_maximum_replayed_candidates_per_update: int = 1_048_576
    liveness_value_critic_weight: float = 0.10
    liveness_cost_critic_weight: float = 0.25
    liveness_cost_actor_weight: float = 0.10
    liveness_direct_policy_weight: float = 0.25
    liveness_cycle_policy_weight: float = 0.10
    liveness_contrast_policy_weight: float = 0.10
    # Generic durable/FLOW completion is a zero-cost critic control, not
    # evidence that the final action should be imitated.  Keep the standalone
    # completion actor channel disabled unless a future evidence contract adds
    # an explicit, causal PREFER witness.
    liveness_completion_policy_weight: float = 0.0
    liveness_risk_advantage_clip: float = 0.25
    # A positive centered-risk label suppresses its factual selected action.
    # Once the current policy has already pushed that action below this floor,
    # repeating the same suppression has negligible behavioural value but can
    # coherently translate the shared policy trunk.  Critics and negative-risk
    # (recovery) labels remain active.  Zero preserves the pre-v17 behaviour.
    liveness_risk_actor_min_selected_probability: float = 0.0
    liveness_contrast_margin: float = 0.10
    # The complete formal liveness objective receives this independent
    # gradient-delta budget before episodic gradients and the global optimizer
    # clip are applied.  This prevents one sparse witness from consuming the
    # ordinary V-trace gradient budget.
    liveness_gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "shadow", "learning"}:
            raise ValueError("failure_credit.mode must be 'disabled', 'shadow', or 'learning'")
        for name in (
            "replay_capacity",
            "replay_byte_capacity",
            "sample_records",
            "maximum_context_steps",
            "maximum_episode_completion_controls",
            "maximum_episode_completion_bytes",
            "policy_gradient_max_lag",
            "liveness_records_per_autograd_batch",
            "liveness_tbptt_window_steps",
            "liveness_maximum_contexts_per_record",
            "liveness_maximum_replayed_steps_per_update",
            "liveness_maximum_replayed_candidates_per_update",
        ):
            _require_int(
                getattr(self, name),
                label=f"failure_credit.{name}",
                minimum=1,
            )
        _require_int(
            self.burn_in_steps,
            label="failure_credit.burn_in_steps",
            minimum=0,
        )
        _require_int(
            self.liveness_head_calibration_updates,
            label="failure_credit.liveness_head_calibration_updates",
            minimum=0,
        )
        _require_int(
            self.maximum_matched_pairs_per_publication,
            label="failure_credit.maximum_matched_pairs_per_publication",
            minimum=0,
        )
        _require_int(
            self.liveness_risk_actor_start_update,
            label="failure_credit.liveness_risk_actor_start_update",
            minimum=0,
        )
        if self.burn_in_steps >= self.maximum_context_steps:
            raise ValueError("failure_credit.burn_in_steps must be smaller than " "maximum_context_steps")
        if self.maximum_episode_completion_bytes > self.replay_byte_capacity:
            raise ValueError(
                "failure_credit.maximum_episode_completion_bytes cannot exceed " "failure_credit.replay_byte_capacity"
            )
        if self.liveness_risk_actor_start_update < self.liveness_head_calibration_updates:
            raise ValueError(
                "failure_credit.liveness_risk_actor_start_update cannot precede "
                "failure_credit.liveness_head_calibration_updates"
            )
        if self.liveness_records_per_autograd_batch > self.sample_records:
            raise ValueError(
                "failure_credit.liveness_records_per_autograd_batch cannot exceed " "failure_credit.sample_records"
            )
        quota_total = 0
        for name in (
            "direct_witness_quota",
            "multi_edge_cycle_quota",
            "risk_sequence_quota",
            "unresolved_stall_quota",
            "completion_control_quota",
            "matched_outcome_pair_quota",
        ):
            quota_total += _require_int(
                getattr(self, name),
                label=f"failure_credit.{name}",
                minimum=0,
            )
        if quota_total > self.sample_records:
            raise ValueError("failure_credit evidence quotas cannot exceed sample_records")
        for name in (
            "liveness_value_critic_weight",
            "liveness_cost_critic_weight",
            "liveness_cost_actor_weight",
            "liveness_direct_policy_weight",
            "liveness_cycle_policy_weight",
            "liveness_contrast_policy_weight",
            "liveness_completion_policy_weight",
            "liveness_risk_advantage_clip",
            "liveness_risk_actor_min_selected_probability",
            "liveness_contrast_margin",
            "liveness_gradient_clip_norm",
        ):
            _require_finite_number(
                getattr(self, name),
                label=f"failure_credit.{name}",
                minimum=0.0,
            )
        if self.liveness_risk_advantage_clip > 1.0:
            raise ValueError("failure_credit.liveness_risk_advantage_clip must be in [0, 1]")
        if self.liveness_risk_actor_min_selected_probability >= 1.0:
            raise ValueError(
                "failure_credit.liveness_risk_actor_min_selected_probability "
                "must be in [0, 1)"
            )
        if self.liveness_gradient_clip_norm <= 0.0:
            raise ValueError("failure_credit.liveness_gradient_clip_norm must be positive")

    @property
    def shadow_enabled(self) -> bool:
        return self.mode in {"shadow", "learning"}

    @property
    def learning_enabled(self) -> bool:
        return self.mode == "learning"


@dataclass(frozen=True, slots=True)
class EpisodicLearningConfig:
    """Complete-episode labels replayed through short recurrent graphs.

    Completed training episodes remain detached CPU data. ``burn_in_steps``
    reconstructs recurrent state under ``no_grad`` and only ``learn_steps``
    participates in autograd, so episode length does not determine accelerator
    activation memory. The two byte limits make the host-memory contract
    explicit and prevent one pathological episode from evicting the corpus.

    Revival policy supervision is a secondary objective. Its implementation
    must gate cost advantages on factual horizon success. Before the task value
    classifies the horizon as successful, ``secondary_advantage_fraction``
    caps cost against the primary residual. Only after success classification
    *and* inside ``primary_success_tie_tolerance`` may it cap against one
    nominal primary unit, so equal-primary successful paths can still be
    ordered by revival cost without overriding a material completion residual.
    """

    enabled: bool = False
    replay_capacity_episodes: int = 64
    replay_capacity_bytes: int = 2_147_483_648
    per_episode_capacity_bytes: int = 536_870_912
    max_segments_per_episode: int = 8
    sample_sequences: int = 2
    # Reserve this many existing sampling slots for suffixes that are known to
    # contain at least one successful policy label inside the strict policy-lag
    # gate. Remaining slots keep the all-age value/outcome replay plane.
    fresh_policy_sequences: int = 0
    burn_in_steps: int = 32
    learn_steps: int = 32
    # Complete runs contain far more combat actions than build/route/resource
    # decisions.  This fraction reserves replay sequences for factual
    # non-combat policy decisions, stratified by their observed decision
    # surface.  It changes replay sampling only: no action is fabricated, no
    # reward is rewritten, and the one-pass FIFO V-trace plane is untouched.
    macro_sample_fraction: float = 0.0
    primary_policy_weight: float = 0.25
    task_value_weight: float = 0.25
    revival_value_weight: float = 0.10
    revival_policy_weight: float = 0.05
    secondary_advantage_fraction: float = 0.25
    primary_success_tie_tolerance: float = 0.05
    importance_ratio_clip: float = 1.0
    # Positive-advantage success imitation stops once the current policy is
    # already more than this multiplicative trust region above the factual
    # behavior policy.  Value labels remain active.
    success_policy_trust_region_epsilon: float = 0.20
    # Successful complete episodes remain factual value supervision on every
    # surface, but selected-action imitation can become a one-way ratchet on
    # sparse strategic entry surfaces.  Listed surface identities therefore
    # stay available to value/SMDP learning while being excluded from repeated
    # success imitation and from the reserved fresh-policy sampler.
    success_imitation_exempt_surfaces: tuple[str, ...] = ()
    # A failed run can still contain a factually completed, healthy Act.  This
    # optional lower imitation rung reuses the exact observed actions from that
    # Act instead of treating the entire failed run as value-only.  Run wins
    # remain the dominant rung; no failed final Act is imitated.
    act_segment_imitation_enabled: bool = False
    # Multiplicative fraction of ``primary_policy_weight``.  Keeping this at
    # or below one makes a complete run win structurally dominant even when
    # failed-prefix examples are much more numerous.
    act_segment_policy_weight: float = 0.30
    act_segment_min_exit_hp_ratio: float = 0.35
    act_segment_max_revival_fraction: float = 0.34
    # Candidate-independent, bounded factual HP-loss regression for completed
    # combats.  A zero weight keeps the auxiliary head dormant while retaining
    # an explicit ABI for model-initialization migrations.
    combat_hp_loss_value_weight: float = 0.0
    combat_hp_loss_reference: float = 80.0
    # Old complete episodes remain useful factual value targets, but their
    # selected-action likelihood must not continue moving a much newer policy.
    # The learner therefore keeps value supervision and suppresses only policy
    # gradients whose behavior version exceeds this strict lag.
    policy_gradient_max_lag: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("episodic_learning.enabled must be a boolean")
        if not isinstance(self.act_segment_imitation_enabled, bool):
            raise TypeError(
                "episodic_learning.act_segment_imitation_enabled must be a boolean"
            )
        for name in (
            "replay_capacity_episodes",
            "replay_capacity_bytes",
            "per_episode_capacity_bytes",
            "max_segments_per_episode",
            "sample_sequences",
            "learn_steps",
            "policy_gradient_max_lag",
        ):
            _require_int(
                getattr(self, name),
                label=f"episodic_learning.{name}",
                minimum=1,
            )
        _require_int(
            self.fresh_policy_sequences,
            label="episodic_learning.fresh_policy_sequences",
            minimum=0,
        )
        if self.fresh_policy_sequences > self.sample_sequences:
            raise ValueError("episodic_learning.fresh_policy_sequences cannot exceed " "sample_sequences")
        surfaces = self.success_imitation_exempt_surfaces
        if not isinstance(surfaces, tuple):
            surfaces = tuple(surfaces)
            object.__setattr__(self, "success_imitation_exempt_surfaces", surfaces)
        normalized_surfaces: list[str] = []
        for index, surface in enumerate(surfaces):
            if not isinstance(surface, str):
                raise TypeError(
                    "episodic_learning.success_imitation_exempt_surfaces"
                    f"[{index}] must be text"
                )
            normalized = surface.strip().lower()
            if (
                not normalized
                or normalized != surface
                or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in normalized)
            ):
                raise ValueError(
                    "episodic_learning.success_imitation_exempt_surfaces "
                    "must contain canonical lowercase surface keys"
                )
            if normalized in normalized_surfaces:
                raise ValueError(
                    "episodic_learning.success_imitation_exempt_surfaces "
                    "contains a duplicate surface"
                )
            normalized_surfaces.append(normalized)
        _require_int(
            self.burn_in_steps,
            label="episodic_learning.burn_in_steps",
            minimum=0,
        )
        _require_finite_number(
            self.macro_sample_fraction,
            label="episodic_learning.macro_sample_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if self.per_episode_capacity_bytes > self.replay_capacity_bytes:
            raise ValueError("episodic_learning.per_episode_capacity_bytes cannot exceed " "replay_capacity_bytes")
        for name in (
            "primary_policy_weight",
            "task_value_weight",
            "revival_value_weight",
            "revival_policy_weight",
            "combat_hp_loss_value_weight",
        ):
            _require_finite_number(
                getattr(self, name),
                label=f"episodic_learning.{name}",
                minimum=0.0,
            )
        _require_finite_number(
            self.act_segment_policy_weight,
            label="episodic_learning.act_segment_policy_weight",
            minimum=0.0,
            maximum=1.0,
        )
        _require_finite_number(
            self.act_segment_min_exit_hp_ratio,
            label="episodic_learning.act_segment_min_exit_hp_ratio",
            minimum=0.0,
            maximum=1.0,
        )
        _require_finite_number(
            self.act_segment_max_revival_fraction,
            label="episodic_learning.act_segment_max_revival_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        hp_reference = _require_finite_number(
            self.combat_hp_loss_reference,
            label="episodic_learning.combat_hp_loss_reference",
        )
        if hp_reference <= 0.0:
            raise ValueError(
                "episodic_learning.combat_hp_loss_reference must be positive"
            )
        _require_finite_number(
            self.secondary_advantage_fraction,
            label="episodic_learning.secondary_advantage_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        _require_finite_number(
            self.primary_success_tie_tolerance,
            label="episodic_learning.primary_success_tie_tolerance",
            minimum=0.0,
            maximum=0.5,
        )
        importance_ratio_clip = _require_finite_number(
            self.importance_ratio_clip,
            label="episodic_learning.importance_ratio_clip",
        )
        if importance_ratio_clip <= 0.0:
            raise ValueError("episodic_learning.importance_ratio_clip must be positive")
        _require_finite_number(
            self.success_policy_trust_region_epsilon,
            label="episodic_learning.success_policy_trust_region_epsilon",
            minimum=0.0,
            maximum=1.0,
        )


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    backend: Literal["live", "headless"] = "headless"
    scenario: Literal["full-run", "combat"] = "full-run"
    session_path: str | None = None
    sim_exe_path: str | None = None
    character: str | None = None
    encounter_id: str | None = None
    max_episode_steps: int = 10_000

    def __post_init__(self) -> None:
        if self.backend not in {"live", "headless"}:
            raise ValueError("environment backend must be live or headless")
        if self.scenario not in {"full-run", "combat"}:
            raise ValueError("environment scenario must be full-run or combat")
        _require_int(
            self.max_episode_steps,
            label="environment.max_episode_steps",
            minimum=1,
        )
        for name in ("session_path", "sim_exe_path", "character", "encounter_id"):
            _require_optional_text(
                getattr(self, name),
                label=f"environment.{name}",
            )
        if self.backend == "headless" and self.session_path is not None:
            raise ValueError("environment.session_path is only valid for the live backend")
        if self.backend == "live" and self.sim_exe_path is not None:
            raise ValueError("environment.sim_exe_path is only valid for the headless backend")
        if self.scenario == "full-run" and self.encounter_id is not None:
            raise ValueError("environment.encounter_id is only valid for combat scenarios")


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Task horizon and exploration schedule without mechanics rules."""

    mode: Literal["standard", "native-revival-preheat"] = "standard"
    reward_objective: Literal["combat", "act1", "run"] = "run"
    revival_mechanism: Literal["engine-bailout-v1"] | None = None
    revival_budget: int | None = None
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 250_000

    def __post_init__(self) -> None:
        if self.mode not in {"standard", "native-revival-preheat"}:
            raise ValueError("unsupported curriculum mode")
        if self.reward_objective not in {"combat", "act1", "run"}:
            raise ValueError("reward_objective must be combat, act1, or run")
        _require_optional_text(
            self.revival_mechanism,
            label="curriculum.revival_mechanism",
        )
        if self.revival_mechanism not in {None, ENGINE_REVIVAL_MECHANISM}:
            raise ValueError("curriculum.revival_mechanism must be " f"{ENGINE_REVIVAL_MECHANISM!r} or null")
        if self.mode == "standard" and self.revival_mechanism is not None:
            raise ValueError("standard curriculum cannot enable an engine revival mechanism")
        if self.mode == "standard" and self.revival_budget is not None:
            raise ValueError("standard curriculum cannot set a revival budget")
        if self.mode == "native-revival-preheat" and self.revival_mechanism is None:
            raise ValueError("engine-bailout preheat requires revival_mechanism")
        if self.mode == "native-revival-preheat" and self.revival_budget is None:
            raise ValueError("engine-bailout preheat requires revival_budget")
        if self.revival_budget is not None:
            _require_int(
                self.revival_budget,
                label="curriculum.revival_budget",
                minimum=-1,
            )
        epsilon_start = _require_finite_number(
            self.epsilon_start,
            label="curriculum.epsilon_start",
        )
        epsilon_end = _require_finite_number(
            self.epsilon_end,
            label="curriculum.epsilon_end",
        )
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError("exploration epsilon must satisfy 0 <= end <= start <= 1")
        _require_int(
            self.epsilon_decay_steps,
            label="curriculum.epsilon_decay_steps",
            minimum=1,
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = "auto"
    collector_device: str = "cpu"
    # Execution-only kernel selection.  This is recorded in run/checkpoint
    # provenance but deliberately excluded from immutable learning lineage:
    # an exact resume still restores model/optimizer/replay/RNG/counters while
    # allowing a reviewed ROCm kernel hardening transition.
    rocm_sdpa_backend: Literal["auto", "math"] = "auto"
    # Model initialization always starts a new lineage. This execution-time
    # migration choice controls only whether mature schedule clocks are carried
    # across that explicit boundary; optimizer/replay/RNG/counters still reset.
    model_initialization_schedule_mode: Literal["reset", "inherit"] = "reset"
    # Entropy/epsilon clocks may inherit mature source progress while the new
    # liveness heads deliberately replay their calibration and risk-actor
    # phases from lineage-local learner update zero.
    model_initialization_liveness_schedule_mode: Literal["reset", "inherit"] = "inherit"
    total_environment_steps: int = 1_000_000
    seed: int = 0
    log_dir: str = "runs/recurrent-vtrace"
    checkpoint_dir: str = "checkpoints/recurrent-vtrace"
    checkpoint_interval_steps: int = 25_000
    evaluation_steps: tuple[int, ...] = (0, 10_000, 25_000, 50_000)
    evaluation_episodes: int = 20
    # Lightweight repeated validation gates are explicitly separate from the
    # larger evaluation schedule and from the never-reused final-audit seeds.
    early_evaluation_steps: tuple[int, ...] = ()
    early_evaluation_episodes: int = 0
    final_audit_steps: tuple[int, ...] = ()
    final_audit_episodes: int = 0
    evaluation_liveness_guard_enabled: bool = False
    evaluation_guard_min_confirm_ready: int = 8
    evaluation_guard_min_multi_action_end_turn: int = 32
    evaluation_guard_min_liveness_episodes: int = 8
    evaluation_guard_max_confirm_failure_rate: float = 0.95
    evaluation_guard_max_multi_action_end_turn_rate: float = 0.75
    evaluation_guard_max_selection_cycle_episode_rate: float = 0.75
    evaluation_guard_max_liveness_failure_episode_rate: float = 0.50
    # A weak model-initialization policy may already exceed the absolute
    # liveness ceiling.  Repeated validation must then detect regression from
    # that frozen Gate-0 control instead of stopping the first time the same
    # baseline failure rate is sampled again.  A zero baseline keeps the
    # historical absolute-only contract.
    evaluation_guard_liveness_baseline_failures: int = 0
    evaluation_guard_liveness_baseline_episodes: int = 0
    evaluation_guard_min_liveness_regression_rate: float = 0.0
    training_deadlock_streak_alert_episodes: int = 0
    # Failed gates below this scheduled step are checkpointed/reported and
    # consumed, but cannot stop or roll back the lineage. This is an explicitly
    # bounded recovery phase, not a retry loop; at/above the boundary the
    # configured guard action is enforced.
    evaluation_guard_enforcement_start_steps: int = 0
    evaluation_guard_failure_action: Literal["stop", "rollback_continue"] = "stop"
    evaluation_guard_max_rollbacks: int = 0

    def __post_init__(self) -> None:
        for name in (
            "total_environment_steps",
            "checkpoint_interval_steps",
        ):
            _require_int(
                getattr(self, name),
                label=f"runtime.{name}",
                minimum=1,
            )
        _require_int(
            self.seed,
            label="runtime.seed",
            minimum=0,
        )
        _require_int(
            self.evaluation_episodes,
            label="runtime.evaluation_episodes",
            minimum=0,
        )
        for name in (
            "early_evaluation_episodes",
            "final_audit_episodes",
            "evaluation_guard_min_confirm_ready",
            "evaluation_guard_min_multi_action_end_turn",
            "evaluation_guard_min_liveness_episodes",
            "evaluation_guard_liveness_baseline_failures",
            "evaluation_guard_liveness_baseline_episodes",
            "training_deadlock_streak_alert_episodes",
            "evaluation_guard_enforcement_start_steps",
            "evaluation_guard_max_rollbacks",
        ):
            _require_int(
                getattr(self, name),
                label=f"runtime.{name}",
                minimum=0,
            )
        if not isinstance(self.evaluation_liveness_guard_enabled, bool):
            raise TypeError("runtime.evaluation_liveness_guard_enabled must be a boolean")
        if (
            self.evaluation_guard_liveness_baseline_episodes == 0
            and self.evaluation_guard_liveness_baseline_failures != 0
        ):
            raise ValueError("runtime evaluation liveness baseline failures require baseline episodes")
        if self.evaluation_guard_liveness_baseline_failures > self.evaluation_guard_liveness_baseline_episodes:
            raise ValueError("runtime evaluation liveness baseline failures cannot exceed episodes")
        for name in (
            "evaluation_steps",
            "early_evaluation_steps",
            "final_audit_steps",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                values = tuple(values)
                object.__setattr__(self, name, values)
            previous = -1
            for index, step in enumerate(values):
                _require_int(
                    step,
                    label=f"runtime.{name}[{index}]",
                    minimum=0,
                )
                if step <= previous:
                    raise ValueError(f"runtime.{name} must be strictly increasing")
                previous = step
        if set(self.early_evaluation_steps) & set(self.evaluation_steps):
            raise ValueError("runtime early_evaluation_steps and evaluation_steps must be disjoint")
        if set(self.final_audit_steps) & (set(self.early_evaluation_steps) | set(self.evaluation_steps)):
            raise ValueError("runtime final_audit_steps must be disjoint from repeated validation")
        if self.early_evaluation_steps and self.early_evaluation_episodes <= 0:
            raise ValueError("runtime early_evaluation_steps require early_evaluation_episodes")
        if self.final_audit_steps and self.final_audit_episodes <= 0:
            raise ValueError("runtime final_audit_steps require final_audit_episodes")
        if self.evaluation_liveness_guard_enabled and not self.early_evaluation_steps and not self.evaluation_steps:
            raise ValueError(
                "runtime evaluation liveness guard requires early evaluation or " "scheduled validation gates"
            )
        for name in (
            "evaluation_guard_max_confirm_failure_rate",
            "evaluation_guard_max_multi_action_end_turn_rate",
            "evaluation_guard_max_selection_cycle_episode_rate",
            "evaluation_guard_max_liveness_failure_episode_rate",
            "evaluation_guard_min_liveness_regression_rate",
        ):
            _require_finite_number(
                getattr(self, name),
                label=f"runtime.{name}",
                minimum=0.0,
                maximum=1.0,
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise TypeError("runtime.device must be a non-empty string")
        if not isinstance(self.collector_device, str) or not self.collector_device.strip():
            raise TypeError("runtime.collector_device must be a non-empty string")
        if self.rocm_sdpa_backend not in ("auto", "math"):
            raise ValueError("runtime.rocm_sdpa_backend must be 'auto' or 'math'")
        if self.model_initialization_schedule_mode not in ("reset", "inherit"):
            raise ValueError("runtime.model_initialization_schedule_mode must be 'reset' or 'inherit'")
        if self.model_initialization_liveness_schedule_mode not in ("reset", "inherit"):
            raise ValueError("runtime.model_initialization_liveness_schedule_mode must be 'reset' or 'inherit'")
        if self.evaluation_guard_failure_action not in {"stop", "rollback_continue"}:
            raise ValueError("runtime.evaluation_guard_failure_action must be 'stop' or 'rollback_continue'")
        if self.evaluation_guard_failure_action == "rollback_continue":
            if not self.evaluation_liveness_guard_enabled:
                raise ValueError("rollback_continue requires the evaluation liveness guard")
            if self.evaluation_guard_max_rollbacks <= 0:
                raise ValueError("rollback_continue requires evaluation_guard_max_rollbacks > 0")
        if (
            self.evaluation_guard_enforcement_start_steps > 0
            and not self.evaluation_liveness_guard_enabled
        ):
            raise ValueError(
                "runtime evaluation_guard_enforcement_start_steps requires "
                "the evaluation liveness guard"
            )
        if (
            not isinstance(self.log_dir, str)
            or not isinstance(self.checkpoint_dir, str)
            or not self.log_dir.strip()
            or not self.checkpoint_dir.strip()
        ):
            raise ValueError("log_dir and checkpoint_dir must be non-empty")
        validate_seed_budget(
            self.seed,
            maximum_training_episodes=self.total_environment_steps,
            evaluation_episodes=max(
                self.evaluation_episodes,
                self.early_evaluation_episodes,
            ),
            final_audit_episodes=self.final_audit_episodes,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    deadlock_window: int = 128
    deadlock_repeat_threshold: int = 8
    combat_net_progress_window: int = 256
    # An engine-bailout curriculum can keep a strategically lost combat alive
    # long enough for encounter mechanics to make one simulator step
    # pathologically expensive. These maps shorten the same generic net-HP
    # liveness rule at a known room/encounter locus; they do not add a
    # mechanics-specific reward or action rewrite. Because the effective
    # window changes terminal labels, both maps remain immutable lineage
    # semantics rather than an observation-only runtime knob.
    combat_net_progress_room_windows: dict[str, int] = field(default_factory=dict)
    combat_net_progress_encounter_windows: dict[str, int] = field(default_factory=dict)
    noncombat_durable_progress_window: int = 256
    combat_min_net_hp_fraction: float = 0.05
    journal_policy_topk: int = 5

    def __post_init__(self) -> None:
        for name in (
            "deadlock_window",
            "deadlock_repeat_threshold",
            "combat_net_progress_window",
            "noncombat_durable_progress_window",
            "journal_policy_topk",
        ):
            _require_int(
                getattr(self, name),
                label=f"diagnostics.{name}",
                minimum=1,
            )
        for name in (
            "combat_net_progress_room_windows",
            "combat_net_progress_encounter_windows",
        ):
            raw = getattr(self, name)
            if not isinstance(raw, Mapping):
                raise TypeError(f"diagnostics.{name} must be a table")
            normalized: dict[str, int] = {}
            for raw_identifier, raw_window in raw.items():
                if not isinstance(raw_identifier, str) or not raw_identifier.strip():
                    raise TypeError(f"diagnostics.{name} identifiers must be non-empty strings")
                identifier = raw_identifier.strip().upper()
                if identifier in normalized:
                    raise ValueError(f"diagnostics.{name} contains duplicate normalized identifier " f"{identifier!r}")
                normalized[identifier] = _require_int(
                    raw_window,
                    label=f"diagnostics.{name}[{identifier!r}]",
                    minimum=1,
                )
            object.__setattr__(self, name, dict(sorted(normalized.items())))
        if self.deadlock_window < 2:
            raise ValueError("diagnostics.deadlock_window must be at least 2")
        if self.deadlock_repeat_threshold < 2:
            raise ValueError("diagnostics.deadlock_repeat_threshold must be at least 2")
        if self.deadlock_repeat_threshold > self.deadlock_window:
            raise ValueError("diagnostics.deadlock_repeat_threshold cannot exceed deadlock_window")
        minimum_fraction = _require_finite_number(
            self.combat_min_net_hp_fraction,
            label="diagnostics.combat_min_net_hp_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        if minimum_fraction <= 0.0:
            raise ValueError("diagnostics.combat_min_net_hp_fraction must be greater than zero")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    version: str = CONFIG_VERSION
    profile: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    transaction_learning: TransactionLearningConfig = field(default_factory=TransactionLearningConfig)
    failure_credit: FailureCreditConfig = field(default_factory=FailureCreditConfig)
    episodic_learning: EpisodicLearningConfig = field(default_factory=EpisodicLearningConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str):
            raise TypeError("training config version must be a string")
        if self.version != CONFIG_VERSION:
            raise ValueError(f"unsupported training config version {self.version!r}; expected {CONFIG_VERSION!r}")
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("training profile must be non-empty")
        for name, expected_type in (
            ("model", ModelConfig),
            ("optimization", OptimizationConfig),
            ("rollout", RolloutConfig),
            ("transaction_learning", TransactionLearningConfig),
            ("failure_credit", FailureCreditConfig),
            ("episodic_learning", EpisodicLearningConfig),
            ("environment", EnvironmentConfig),
            ("curriculum", CurriculumConfig),
            ("runtime", RuntimeConfig),
            ("diagnostics", DiagnosticsConfig),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"training config {name} must be {expected_type.__name__}")
        if self.environment.scenario == "combat" and self.curriculum.reward_objective != "combat":
            raise ValueError(
                "reward objective must match the environment horizon: " "combat scenarios require objective='combat'"
            )
        if self.environment.scenario == "full-run" and self.curriculum.reward_objective == "combat":
            raise ValueError("full-run scenarios require objective='act1' or objective='run'")
        if self.episodic_learning.enabled:
            if self.environment.scenario != "full-run":
                raise ValueError("episodic learning requires environment.scenario='full-run'")
            if self.curriculum.reward_objective != "run":
                raise ValueError("episodic learning requires curriculum.reward_objective='run'")
        if self.failure_credit.learning_enabled:
            maximum_replay_contexts = (
                self.failure_credit.sample_records * self.failure_credit.liveness_maximum_contexts_per_record
            )
            required_step_budget = maximum_replay_contexts * self.failure_credit.maximum_context_steps
            if self.failure_credit.liveness_maximum_replayed_steps_per_update < required_step_budget:
                raise ValueError(
                    "failure_credit.liveness_maximum_replayed_steps_per_update "
                    "cannot cover sample_records * maximum contexts * maximum_context_steps"
                )
            required_candidate_budget = required_step_budget * self.model.max_candidates
            if self.failure_credit.liveness_maximum_replayed_candidates_per_update < required_candidate_budget:
                raise ValueError(
                    "failure_credit.liveness_maximum_replayed_candidates_per_update "
                    "cannot cover the configured active candidate capacity"
                )
        if (
            (
                self.rollout.deterministic_probe_interval_episodes > 0
                or bool(self.rollout.deterministic_probe_environment_steps)
            )
            and not self.transaction_learning.enabled
            and not self.failure_credit.shadow_enabled
        ):
            raise ValueError(
                "deterministic liveness probes require transaction-v3 or " "formal failure-credit collection"
            )
        if self.curriculum.mode == "native-revival-preheat":
            if self.environment.backend != "headless":
                raise ValueError("engine-bailout preheat requires the headless backend")
        expected_discount = 1.0 if self.curriculum.mode == "native-revival-preheat" else TASK_REWARD_SPEC.discount
        if self.optimization.discount != expected_discount:
            raise ValueError(
                "optimization.discount must equal the reward contract discount "
                f"{expected_discount} for curriculum mode {self.curriculum.mode!r}"
            )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint_sha256(self) -> str:
        """Return a canonical identity for the complete effective config.

        Unlike ``lineage_mapping()``, this includes observation-only schedules
        and output paths.  Evaluation records use it to prove that two gates
        were produced by the same fully resolved runtime configuration.
        """

        serialized = json.dumps(
            self.to_mapping(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def lineage_mapping(self) -> dict[str, Any]:
        """Return only immutable training semantics for exact-resume identity.

        Execution horizon, output locations, and observation-only schedules may
        change when continuing a checkpoint.  Model, task, rollout, optimizer,
        seed, and environment semantics remain part of exact lineage identity.
        """

        payload = self.to_mapping()
        rollout = payload["rollout"]
        if not isinstance(rollout, dict):  # pragma: no cover - asdict invariant
            raise TypeError("serialized rollout config must be an object")
        # Metadata JSON represents immutable tuples as arrays. Canonicalize the
        # new lineage schedule before comparison so a checkpoint written from
        # this exact config can resume without a tuple/list false mismatch.
        rollout["deterministic_probe_environment_steps"] = list(self.rollout.deterministic_probe_environment_steps)
        episodic_learning = payload["episodic_learning"]
        if not isinstance(episodic_learning, dict):  # pragma: no cover - asdict invariant
            raise TypeError("serialized episodic learning config must be an object")
        episodic_learning["success_imitation_exempt_surfaces"] = list(
            self.episodic_learning.success_imitation_exempt_surfaces
        )
        runtime = payload["runtime"]
        if not isinstance(runtime, dict):  # pragma: no cover - asdict invariant
            raise TypeError("serialized runtime config must be an object")
        for key in (
            # Kernel backend selection is execution provenance like the
            # resolved device/driver, not optimizer or task semantics.
            "rocm_sdpa_backend",
            "model_initialization_schedule_mode",
            "model_initialization_liveness_schedule_mode",
            "total_environment_steps",
            "log_dir",
            "checkpoint_dir",
            "checkpoint_interval_steps",
            "evaluation_steps",
            "evaluation_episodes",
            "early_evaluation_steps",
            "early_evaluation_episodes",
            "final_audit_steps",
            "final_audit_episodes",
            "evaluation_liveness_guard_enabled",
            "evaluation_guard_min_confirm_ready",
            "evaluation_guard_min_multi_action_end_turn",
            "evaluation_guard_min_liveness_episodes",
            "evaluation_guard_max_confirm_failure_rate",
            "evaluation_guard_max_multi_action_end_turn_rate",
            "evaluation_guard_max_selection_cycle_episode_rate",
            "evaluation_guard_max_liveness_failure_episode_rate",
            "evaluation_guard_liveness_baseline_failures",
            "evaluation_guard_liveness_baseline_episodes",
            "evaluation_guard_min_liveness_regression_rate",
            "training_deadlock_streak_alert_episodes",
            "evaluation_guard_enforcement_start_steps",
            "evaluation_guard_failure_action",
            "evaluation_guard_max_rollbacks",
        ):
            runtime.pop(key)
        return payload


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"training config not found: {path}")
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"training config must be a TOML table: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> None:
    for raw_key, value in override.items():
        key = str(raw_key).replace("-", "_")
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = dict(value) if isinstance(value, Mapping) else value


def _override_value(raw: str) -> Any:
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw


def _apply_dotted_override(payload: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ValueError(f"config override must be KEY=VALUE, got {item!r}")
    raw_path, raw_value = item.split("=", 1)
    parts = [part.strip().replace("-", "_") for part in raw_path.split(".") if part.strip()]
    if not parts:
        raise ValueError(f"config override has no key: {item!r}")
    cursor = payload
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"unknown/non-table config override path: {raw_path!r}")
        cursor = child
    if parts[-1] not in cursor:
        raise ValueError(f"unknown config override key: {raw_path!r}")
    cursor[parts[-1]] = _override_value(raw_value.strip())


def _construct(cls: type[T], payload: Mapping[str, Any], *, label: str) -> T:
    raw_fields = getattr(cls, "__dataclass_fields__", None)
    if not isinstance(raw_fields, dict):
        raise TypeError(f"{cls!r} is not a dataclass type")
    allowed = set(raw_fields)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} config keys: {', '.join(unknown)}")
    return cls(**dict(payload))


def training_config_from_mapping(payload: Mapping[str, Any]) -> TrainingConfig:
    allowed = {field.name for field in fields(TrainingConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown training config sections: {', '.join(unknown)}")
    version = payload.get("version", CONFIG_VERSION)
    profile = payload.get("profile", "default")
    if not isinstance(version, str):
        raise TypeError("training config version must be a string")
    if not isinstance(profile, str):
        raise TypeError("training config profile must be a string")
    return TrainingConfig(
        version=version,
        profile=profile,
        model=_construct(ModelConfig, _table(payload, "model"), label="model"),
        optimization=_construct(OptimizationConfig, _table(payload, "optimization"), label="optimization"),
        rollout=_construct(RolloutConfig, _table(payload, "rollout"), label="rollout"),
        transaction_learning=_construct(
            TransactionLearningConfig,
            _table(payload, "transaction_learning"),
            label="transaction_learning",
        ),
        failure_credit=_construct(
            FailureCreditConfig,
            _table(payload, "failure_credit"),
            label="failure_credit",
        ),
        episodic_learning=_construct(
            EpisodicLearningConfig,
            _table(payload, "episodic_learning"),
            label="episodic_learning",
        ),
        environment=_construct(EnvironmentConfig, _table(payload, "environment"), label="environment"),
        curriculum=_construct(CurriculumConfig, _table(payload, "curriculum"), label="curriculum"),
        runtime=_construct(RuntimeConfig, _table(payload, "runtime"), label="runtime"),
        diagnostics=_construct(DiagnosticsConfig, _table(payload, "diagnostics"), label="diagnostics"),
    )


def model_initialization_config_from_mapping(
    payload: Mapping[str, Any],
) -> TrainingConfig:
    """Interpret one reviewed source-config change for model-only use.

    V12 adds the independent liveness-credit model/loss ABI. V13 hardens its
    optimization/runtime semantics. V14 adds an explicit, training-only
    transaction exploration contract. V15 adds authoritative transaction
    lifecycle replay and entry-support/SMDP loss controls. V16 replaces the
    one-sided entry floor with a two-sided support corridor and adds a bounded,
    non-destructive evaluation-guard recovery phase. The lifecycle heads use
    the already reviewed optional transaction-head migration gate.
    V17 adds a convex revival-efficiency reward contract, a policy-saturation
    eligibility floor for the liveness risk actor, an explicit success-
    imitation surface exemption, and the policy-collapse-v2 entropy breaker.
    V18 adds healthy Act-prefix imitation, a bounded factual combat HP-loss
    head, and the extended factual transaction option horizon. V19 adds a
    guarded factual option-advantage actor bridge; all of its new controls
    migrate disabled so an older policy is never reinterpreted as supervised
    actor data. V20 retires the operation-specific entry/selection exploration
    floors and completion guidance entirely: the ``transaction_exploration``
    table and the curriculum ``selection_surface_epsilon_floor`` key are
    stripped from every older payload, and a V20 payload that still contains
    them fails the strict parser. These are training/data/model semantics, so V19 checkpoints are accepted only for
    model-parameter initialization; optimizer/replay/RNG state cannot cross
    the boundary.
    V11 checkpoints
    can initialize compatible shared parameters only; their missing liveness
    head is freshly initialized by the reviewed checkpoint overlay.  V10 also
    predates ``fresh_policy_sequences``, so that field receives its
    behavior-preserving disabled value before applying the same V12 migration.

    This helper is deliberately separate from :func:`training_config_from_mapping`.
    Exact resume continues to call that strict parser, so a pre-V12 checkpoint
    can never acquire V12 liveness semantics while retaining optimizer, queue,
    replay, recurrent, RNG, or counter state.
    """

    source_version = payload.get("version", CONFIG_VERSION)
    if source_version == CONFIG_VERSION:
        return training_config_from_mapping(payload)
    if source_version not in {
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V10,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V11,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V12,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V13,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V14,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V15,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V16,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V17,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V18,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V19,
    }:
        raise ValueError(
            "model-parameter initialization has no reviewed config migration "
            f"from {source_version!r} to {CONFIG_VERSION!r}"
        )
    migrated = dict(payload)
    if source_version == _MODEL_INITIALIZATION_SOURCE_CONFIG_V10:
        raw_episodic = payload.get("episodic_learning")
        if not isinstance(raw_episodic, Mapping):
            raise ValueError(
                "reviewed V10 model-initialization config migration requires " "an episodic_learning table"
            )
        if "fresh_policy_sequences" in raw_episodic:
            raise ValueError("V10 model-initialization config unexpectedly contains " "fresh_policy_sequences")
        migrated["episodic_learning"] = {
            **dict(raw_episodic),
            "fresh_policy_sequences": 0,
        }
    if source_version in {
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V10,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V11,
    }:
        if "failure_credit" in payload:
            raise ValueError("pre-V12 model-initialization config unexpectedly contains a " "failure_credit table")
        # A legacy checkpoint can provide compatible shared model parameters,
        # but it cannot claim the new semantic/evidence/replay contract.
        migrated["failure_credit"] = {"mode": "disabled"}
    if source_version not in {
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V14,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V15,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V16,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V17,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V18,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V19,
    }:
        if "transaction_exploration" in payload:
            raise ValueError(
                "pre-V14 model-initialization config unexpectedly contains a "
                "transaction_exploration table"
            )
    # V20 retired the operation-specific entry/selection exploration floors
    # and completion guidance. The reviewed migration strips the retired
    # table/key from every older payload before construction; a V20 payload
    # that still contains them fails the strict parser instead.
    migrated.pop("transaction_exploration", None)
    raw_curriculum = migrated.get("curriculum")
    if isinstance(raw_curriculum, Mapping):
        curriculum_table = dict(raw_curriculum)
        curriculum_table.pop("selection_surface_epsilon_floor", None)
        migrated["curriculum"] = curriculum_table
    transaction_learning = payload.get("transaction_learning")
    if not isinstance(transaction_learning, Mapping):
        raise ValueError(
            f"{source_version} model-initialization config has no "
            "transaction_learning table"
        )
    lifecycle_fields = {
        "lifecycle_entry_support_weight",
        "lifecycle_entry_support_probability_floor",
        "lifecycle_smdp_q_weight",
    }
    if source_version not in {
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V15,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V16,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V17,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V18,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V19,
    }:
        unexpected_lifecycle_fields = lifecycle_fields.intersection(
            transaction_learning
        )
        if unexpected_lifecycle_fields:
            raise ValueError(
                f"{source_version} model-initialization config unexpectedly "
                "contains V15 transaction lifecycle fields: "
                + ", ".join(sorted(unexpected_lifecycle_fields))
            )
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError(
            f"{source_version} model-initialization config has no runtime table"
        )
    if source_version in {
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V16,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V17,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V18,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V19,
    }:
        if "evaluation_guard_enforcement_start_steps" not in runtime:
            raise ValueError(
                "V16 model-initialization config is missing its evaluation "
                "guard enforcement boundary"
            )
    else:
        if "evaluation_guard_enforcement_start_steps" in runtime:
            raise ValueError(
                f"{source_version} model-initialization config unexpectedly "
                "contains the V16 evaluation guard enforcement boundary"
            )
        # Pre-V16 configs enforced every guard exactly as before. A successor
        # recipe may opt into a non-zero recovery phase only after migration.
        migrated["runtime"] = {
            **dict(runtime),
            "evaluation_guard_enforcement_start_steps": 0,
        }

    raw_failure = migrated.get("failure_credit")
    if not isinstance(raw_failure, Mapping):
        raise ValueError(
            f"{source_version} model-initialization config has no "
            "failure_credit table after reviewed migration"
        )
    if source_version in {
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V17,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V18,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V19,
    }:
        if "liveness_risk_actor_min_selected_probability" not in raw_failure:
            raise ValueError(
                "V17 model-initialization config is missing its risk-actor "
                "saturation floor"
            )
    else:
        if "liveness_risk_actor_min_selected_probability" in raw_failure:
            raise ValueError(
                f"{source_version} model-initialization config unexpectedly "
                "contains the V17 risk-actor saturation floor"
            )
        migrated["failure_credit"] = {
            **dict(raw_failure),
            "liveness_risk_actor_min_selected_probability": 0.0,
        }

    raw_episodic = migrated.get("episodic_learning")
    if not isinstance(raw_episodic, Mapping):
        raise ValueError(
            f"{source_version} model-initialization config has no "
            "episodic_learning table after reviewed migration"
        )
    if source_version in {
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V17,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V18,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V19,
    }:
        if "success_imitation_exempt_surfaces" not in raw_episodic:
            raise ValueError(
                "V17 model-initialization config is missing its success-"
                "imitation surface exemption"
            )
    else:
        if "success_imitation_exempt_surfaces" in raw_episodic:
            raise ValueError(
                f"{source_version} model-initialization config unexpectedly "
                "contains the V17 success-imitation surface exemption"
            )
        raw_episodic = {
            **dict(raw_episodic),
            "success_imitation_exempt_surfaces": (),
        }
    v18_fields = {
        "act_segment_imitation_enabled": False,
        "act_segment_policy_weight": 0.30,
        "act_segment_min_exit_hp_ratio": 0.35,
        "act_segment_max_revival_fraction": 0.34,
        "combat_hp_loss_value_weight": 0.0,
        "combat_hp_loss_reference": 80.0,
    }
    if source_version in {
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V18,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V19,
    }:
        missing_v18 = set(v18_fields) - set(raw_episodic)
        if missing_v18:
            raise ValueError(
                "V18 model-initialization config is missing episodic fields: "
                + ", ".join(sorted(missing_v18))
            )
        migrated["episodic_learning"] = dict(raw_episodic)
    else:
        unexpected_v18 = set(v18_fields).intersection(raw_episodic)
        if unexpected_v18:
            raise ValueError(
                f"{source_version} model-initialization config unexpectedly "
                "contains V18 episodic fields: "
                + ", ".join(sorted(unexpected_v18))
            )
        migrated["episodic_learning"] = {
            **dict(raw_episodic),
            **v18_fields,
        }

    raw_transaction = migrated.get("transaction_learning")
    if not isinstance(raw_transaction, Mapping):  # pragma: no cover - checked above
        raise ValueError("model-initialization config lost transaction_learning")
    if source_version in {
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V18,
        _MODEL_INITIALIZATION_SOURCE_CONFIG_V19,
    }:
        if "lifecycle_smdp_horizon" not in raw_transaction:
            raise ValueError(
                "V18 model-initialization config is missing its transaction option horizon"
            )
    elif "lifecycle_smdp_horizon" in raw_transaction:
        raise ValueError(
            f"{source_version} model-initialization config unexpectedly "
            "contains the V18 transaction option horizon"
        )
    v19_transaction_fields = {
        "lifecycle_advantage_policy_weight": 0.0,
        "lifecycle_advantage_start_update": 0,
        "lifecycle_advantage_temperature": 0.25,
        "lifecycle_advantage_clip": 1.0,
        "lifecycle_advantage_q_error_gate": 0.25,
        "lifecycle_advantage_max_policy_lag": 128,
        "lifecycle_advantage_max_log_probability_shift": 1.0,
    }
    if source_version == _MODEL_INITIALIZATION_SOURCE_CONFIG_V19:
        missing_v19 = set(v19_transaction_fields) - set(raw_transaction)
        if missing_v19:
            raise ValueError(
                "V19 model-initialization config is missing transaction "
                "actor fields: " + ", ".join(sorted(missing_v19))
            )
    else:
        unexpected_v19 = set(v19_transaction_fields).intersection(raw_transaction)
        if unexpected_v19:
            raise ValueError(
                f"{source_version} model-initialization config unexpectedly "
                "contains V19 transaction actor fields: "
                + ", ".join(sorted(unexpected_v19))
            )
    migrated["transaction_learning"] = {
        **dict(raw_transaction),
        **(
            {}
            if source_version
            in {
                _MODEL_INITIALIZATION_SOURCE_CONFIG_V18,
                _MODEL_INITIALIZATION_SOURCE_CONFIG_V19,
            }
            else {"lifecycle_smdp_horizon": "transaction_exit"}
        ),
        **(
            {}
            if source_version == _MODEL_INITIALIZATION_SOURCE_CONFIG_V19
            else v19_transaction_fields
        ),
    }
    migrated["version"] = CONFIG_VERSION
    return training_config_from_mapping(migrated)


def _table(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"training config section {name!r} must be a table")
    return value


def load_training_config(
    *,
    profile: str = "default",
    config_path: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> TrainingConfig:
    profile_name = str(profile).strip()
    if not profile_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid profile name: {profile_name!r}")
    payload = _load_toml(PROFILE_DIR / f"{profile_name}.toml")
    if config_path is not None:
        _deep_merge(payload, _load_toml(Path(config_path).expanduser()))
    for item in overrides:
        _apply_dotted_override(payload, item)
    payload["profile"] = profile_name
    return training_config_from_mapping(payload)


def replace_runtime(config: TrainingConfig, **changes: Any) -> TrainingConfig:
    """Typed CLI convenience without reintroducing a flat option bag."""

    return replace(config, runtime=replace(config.runtime, **changes))


__all__ = [
    "CONFIG_VERSION",
    "ENGINE_REVIVAL_MECHANISM",
    "PROFILE_DIR",
    "CurriculumConfig",
    "DiagnosticsConfig",
    "EnvironmentConfig",
    "EpisodicLearningConfig",
    "FailureCreditConfig",
    "ModelConfig",
    "OptimizationConfig",
    "RolloutConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "TransactionLearningConfig",
    "engine_revival_identity",
    "load_training_config",
    "model_initialization_config_from_mapping",
    "replace_runtime",
    "training_config_from_mapping",
]
