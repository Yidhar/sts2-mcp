"""Single authority for failure-credit actor-unit eligibility.

Replay quotas and the production learner must agree on what can actually
produce a policy gradient.  A record-level approximation such as "any target
row is fresh" is insufficient: direct and risk labels are row-wise, while a
cycle or matched-outcome comparison is an indivisible multi-row unit.

The pure helpers in this module operate only on immutable failure-credit DTOs.
They perform no model forward and own no mutable state, so both corpus sampling
and learner manifest compilation can consume the exact same rules.
"""

from __future__ import annotations

from typing import Final

from .contracts import (
    ContrastPolicyCredit,
    CreditPlan,
    CyclePolicyCredit,
    DirectPolicyCredit,
    EvidenceStratum,
    LearningStep,
)

ACTOR_CREDIT_STRATA: Final = frozenset(
    {
        EvidenceStratum.DIRECT_WITNESS,
        EvidenceStratum.MULTI_EDGE_CYCLE,
        EvidenceStratum.RISK_SEQUENCE,
        EvidenceStratum.MATCHED_OUTCOME_PAIR,
    }
)


def _actor_clock(
    *,
    current_policy_version: int,
    policy_gradient_max_lag: int,
) -> tuple[int, int]:
    for label, value in (
        ("current_policy_version", current_policy_version),
        ("policy_gradient_max_lag", policy_gradient_max_lag),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value < 0:
            raise ValueError(f"{label} must be non-negative")
    return current_policy_version, policy_gradient_max_lag


def actor_step_is_fresh(
    step: LearningStep,
    *,
    current_policy_version: int,
    policy_gradient_max_lag: int,
) -> bool:
    """Return freshness for one factual behavior step, rejecting the future."""

    if not isinstance(step, LearningStep):
        raise TypeError("step must be LearningStep")
    current, maximum_lag = _actor_clock(
        current_policy_version=current_policy_version,
        policy_gradient_max_lag=policy_gradient_max_lag,
    )
    if step.policy_version > current:
        raise ValueError("failure evidence actor target is newer than the learner")
    return current - step.policy_version <= maximum_lag


def _validate_plan_authority(
    plan: CreditPlan,
    *,
    current_policy_version: int,
) -> None:
    if not isinstance(plan, CreditPlan):
        raise TypeError("plan must be CreditPlan")
    if plan.provenance.policy_version > current_policy_version:
        raise ValueError("failure evidence provenance is newer than the learner")
    if EvidenceStratum.CENSORED in plan.strata and plan.strata != (EvidenceStratum.CENSORED,):
        raise ValueError("CENSORED cannot be mixed with authoritative credit strata")


def direct_actor_unit_effective(
    plan: CreditPlan,
    target: DirectPolicyCredit,
    *,
    current_policy_version: int,
    policy_gradient_max_lag: int,
) -> bool:
    """Whether one independent direct target can train the policy."""

    _validate_plan_authority(plan, current_policy_version=current_policy_version)
    if not isinstance(target, DirectPolicyCredit) or not any(
        candidate is target for candidate in plan.direct_policy_targets
    ):
        raise ValueError("direct target does not belong to the credit plan")
    step = plan.context.steps[target.step_index]
    fresh = actor_step_is_fresh(
        step,
        current_policy_version=current_policy_version,
        policy_gradient_max_lag=policy_gradient_max_lag,
    )
    return EvidenceStratum.CENSORED not in plan.strata and step.actor_eligible and fresh


def cycle_actor_unit_effective(
    plan: CreditPlan,
    target: CyclePolicyCredit,
    *,
    current_policy_version: int,
    policy_gradient_max_lag: int,
) -> bool:
    """Whether one indivisible cycle group can train the policy."""

    _validate_plan_authority(plan, current_policy_version=current_policy_version)
    if not isinstance(target, CyclePolicyCredit) or not any(
        candidate is target for candidate in plan.cycle_policy_targets
    ):
        raise ValueError("cycle target does not belong to the credit plan")
    steps = tuple(plan.context.steps[index] for index in target.step_indices)
    freshness = tuple(
        actor_step_is_fresh(
            step,
            current_policy_version=current_policy_version,
            policy_gradient_max_lag=policy_gradient_max_lag,
        )
        for step in steps
    )
    all_fresh = all(freshness)
    return (
        EvidenceStratum.CENSORED not in plan.strata
        and all_fresh
        and any(step.actor_eligible for step in steps)
    )


def risk_actor_row_effective(
    plan: CreditPlan,
    step_index: int,
    *,
    current_policy_version: int,
    policy_gradient_max_lag: int,
    risk_actor_enabled: bool,
) -> bool:
    """Whether one row-wise risk target can train the policy in this phase."""

    _validate_plan_authority(plan, current_policy_version=current_policy_version)
    if not isinstance(risk_actor_enabled, bool):
        raise TypeError("risk_actor_enabled must be a boolean")
    if not any(step_index in sequence.step_indices for sequence in plan.risk_sequences):
        raise ValueError("risk step does not belong to the credit plan")
    step = plan.context.steps[step_index]
    fresh = actor_step_is_fresh(
        step,
        current_policy_version=current_policy_version,
        policy_gradient_max_lag=policy_gradient_max_lag,
    )
    return (
        EvidenceStratum.CENSORED not in plan.strata
        and risk_actor_enabled
        and step.actor_eligible
        and fresh
    )


def contrast_actor_unit_effective(
    plan: CreditPlan,
    target: ContrastPolicyCredit,
    *,
    current_policy_version: int,
    policy_gradient_max_lag: int,
) -> bool:
    """Whether both arms of one atomic comparison can train the policy."""

    _validate_plan_authority(plan, current_policy_version=current_policy_version)
    if not isinstance(target, ContrastPolicyCredit) or not any(
        candidate is target for candidate in plan.contrast_policy_targets
    ):
        raise ValueError("contrast target does not belong to the credit plan")
    steps = (
        target.pair.better.step,
        target.pair.worse.step,
    )
    freshness = tuple(
        actor_step_is_fresh(
            step,
            current_policy_version=current_policy_version,
            policy_gradient_max_lag=policy_gradient_max_lag,
        )
        for step in steps
    )
    all_fresh = all(freshness)
    return (
        EvidenceStratum.CENSORED not in plan.strata
        and all_fresh
        and all(step.actor_eligible for step in steps)
    )


def actor_policy_versions(
    plan: CreditPlan,
    stratum: EvidenceStratum,
) -> tuple[int, ...]:
    """Return all behavior versions governing a stratum's actor units.

    Cycle and contrast units include every row whose joint freshness is
    required, including a future malformed forced row.  Direct and risk units
    are independent and therefore expose only rows that can individually train
    the actor.
    """

    if not isinstance(plan, CreditPlan):
        raise TypeError("plan must be CreditPlan")
    if not isinstance(stratum, EvidenceStratum):
        raise TypeError("stratum must be EvidenceStratum")
    if stratum is EvidenceStratum.DIRECT_WITNESS:
        return tuple(
            plan.context.steps[target.step_index].policy_version
            for target in plan.direct_policy_targets
            if plan.context.steps[target.step_index].actor_eligible
        )
    if stratum is EvidenceStratum.MULTI_EDGE_CYCLE:
        return tuple(
            plan.context.steps[index].policy_version
            for target in plan.cycle_policy_targets
            if any(plan.context.steps[item].actor_eligible for item in target.step_indices)
            for index in target.step_indices
        )
    if stratum is EvidenceStratum.RISK_SEQUENCE:
        return tuple(
            plan.context.steps[index].policy_version
            for sequence in plan.risk_sequences
            for index in sequence.step_indices
            if plan.context.steps[index].actor_eligible
        )
    if stratum is EvidenceStratum.MATCHED_OUTCOME_PAIR:
        return tuple(
            step.policy_version
            for target in plan.contrast_policy_targets
            for step in (target.pair.better.step, target.pair.worse.step)
        )
    return ()


def plan_has_effective_actor_unit(
    plan: CreditPlan,
    stratum: EvidenceStratum,
    *,
    current_policy_version: int,
    policy_gradient_max_lag: int,
    risk_actor_enabled: bool,
) -> bool:
    """Return whether a record can emit a policy gradient for ``stratum``."""

    if not isinstance(stratum, EvidenceStratum):
        raise TypeError("stratum must be EvidenceStratum")
    if not isinstance(risk_actor_enabled, bool):
        raise TypeError("risk_actor_enabled must be a boolean")
    _actor_clock(
        current_policy_version=current_policy_version,
        policy_gradient_max_lag=policy_gradient_max_lag,
    )
    _validate_plan_authority(plan, current_policy_version=current_policy_version)
    if stratum not in plan.strata:
        return False
    if stratum is EvidenceStratum.DIRECT_WITNESS:
        # Materialize every unit before reducing.  ``any(generator)`` would
        # stop after an early fresh row and could hide a malformed future row
        # later in the same record, after which replay RNG would advance.
        effective_units = tuple(
            direct_actor_unit_effective(
                plan,
                target,
                current_policy_version=current_policy_version,
                policy_gradient_max_lag=policy_gradient_max_lag,
            )
            for target in plan.direct_policy_targets
        )
        return any(effective_units)
    if stratum is EvidenceStratum.MULTI_EDGE_CYCLE:
        effective_units = tuple(
            cycle_actor_unit_effective(
                plan,
                target,
                current_policy_version=current_policy_version,
                policy_gradient_max_lag=policy_gradient_max_lag,
            )
            for target in plan.cycle_policy_targets
        )
        return any(effective_units)
    if stratum is EvidenceStratum.RISK_SEQUENCE:
        # Validate every referenced row exactly as the learner manifest does,
        # even when the current phase suppresses all risk actor gradients.
        effective = False
        for sequence in plan.risk_sequences:
            for step_index in sequence.step_indices:
                effective = (
                    risk_actor_row_effective(
                        plan,
                        step_index,
                        current_policy_version=current_policy_version,
                        policy_gradient_max_lag=policy_gradient_max_lag,
                        risk_actor_enabled=risk_actor_enabled,
                    )
                    or effective
                )
        return effective
    if stratum is EvidenceStratum.MATCHED_OUTCOME_PAIR:
        effective_units = tuple(
            contrast_actor_unit_effective(
                plan,
                target,
                current_policy_version=current_policy_version,
                policy_gradient_max_lag=policy_gradient_max_lag,
            )
            for target in plan.contrast_policy_targets
        )
        return any(effective_units)
    return False


__all__ = [
    "ACTOR_CREDIT_STRATA",
    "actor_policy_versions",
    "actor_step_is_fresh",
    "contrast_actor_unit_effective",
    "cycle_actor_unit_effective",
    "direct_actor_unit_effective",
    "plan_has_effective_actor_unit",
    "risk_actor_row_effective",
]
