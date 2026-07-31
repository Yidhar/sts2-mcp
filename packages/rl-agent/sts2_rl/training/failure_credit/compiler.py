"""Pure evidence-to-credit compiler for failure-credit v4."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .contracts import (
    FAILURE_CREDIT_COMPILER_VERSION,
    ContrastPolicyCredit,
    CreditPlan,
    CyclePolicyCredit,
    DirectPolicyCredit,
    DirectPolicyTarget,
    EvidenceStratum,
    FailureIncident,
    FailureOutcome,
    PolicyWitness,
    RiskSequenceCredit,
    ScalarCredit,
    TargetAuthority,
    WitnessKind,
)


class CreditCompilationError(ValueError):
    """The immutable evidence is internally valid but semantically conflicting."""


@dataclass(frozen=True, slots=True)
class CreditCompilerConfig:
    task_discount: float = 1.0
    liveness_discount: float = 0.99
    cycle_margin: float = 0.10
    contrast_margin: float = 0.10

    def __post_init__(self) -> None:
        for label, value in (
            ("task_discount", self.task_discount),
            ("liveness_discount", self.liveness_discount),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise TypeError(f"{label} must be finite")
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{label} must be in (0, 1]")
        for label, value in (
            ("cycle_margin", self.cycle_margin),
            ("contrast_margin", self.contrast_margin),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise TypeError(f"{label} must be finite")
            if float(value) <= 0.0:
                raise ValueError(f"{label} must be positive")


class CreditCompiler:
    """Compile typed targets without consulting a detector tail or mutable replay."""

    def __init__(self, config: CreditCompilerConfig | None = None) -> None:
        self.config = config or CreditCompilerConfig()

    def compile(self, incident: FailureIncident) -> CreditPlan:
        if not isinstance(incident, FailureIncident):
            raise TypeError("incident must be FailureIncident")
        context = incident.context
        learn_indices = context.learn_step_indices

        if incident.outcome is FailureOutcome.CENSORED:
            return CreditPlan(
                plan_id=self._plan_id(incident),
                incident_id=incident.incident_id,
                context=context,
                task_value_targets=(),
                task_q_targets=(),
                liveness_value_targets=(),
                liveness_q_targets=(),
                direct_policy_targets=(),
                cycle_policy_targets=(),
                contrast_policy_targets=(),
                risk_sequences=(),
                strata=(EvidenceStratum.CENSORED,),
                provenance=incident.provenance,
            )

        task_targets: list[ScalarCredit] = []
        if incident.task_authority is not TargetAuthority.CENSORED:
            if incident.task_return is None:  # pragma: no cover - contract invariant
                raise CreditCompilationError("authoritative task incident lost its return")
            for ordinal, step_index in enumerate(learn_indices):
                horizon = len(learn_indices) - ordinal
                target = float(incident.task_return) * self.config.task_discount ** (horizon - 1)
                task_targets.append(
                    ScalarCredit(
                        step_index=step_index,
                        target=target,
                        horizon=horizon,
                    )
                )

        liveness_targets: list[ScalarCredit] = []
        risk_sequences: list[RiskSequenceCredit] = []
        if incident.local_authority is not TargetAuthority.CENSORED:
            if incident.local_failure_cost is None:  # pragma: no cover - contract invariant
                raise CreditCompilationError("authoritative local incident lost its failure cost")
            for ordinal, step_index in enumerate(learn_indices):
                horizon = len(learn_indices) - ordinal
                target = float(incident.local_failure_cost) * self.config.liveness_discount ** (horizon - 1)
                liveness_targets.append(
                    ScalarCredit(
                        step_index=step_index,
                        target=target,
                        horizon=horizon,
                    )
                )
            if incident.outcome.failed:
                risk_witnesses = tuple(
                    witness.witness_id for witness in incident.witnesses if witness.kind is WitnessKind.RISK_SEQUENCE
                )
                risk_sequences.append(
                    RiskSequenceCredit(
                        step_indices=learn_indices,
                        terminal_cost=float(incident.local_failure_cost),
                        discount=self.config.liveness_discount,
                        witness_id=risk_witnesses[0] if len(risk_witnesses) == 1 else None,
                    )
                )

        direct_by_step: dict[int, DirectPolicyCredit] = {}
        cycle_targets: list[CyclePolicyCredit] = []
        contrast_targets: list[ContrastPolicyCredit] = []
        emitted_strata: set[EvidenceStratum] = set()
        if risk_sequences:
            emitted_strata.add(EvidenceStratum.RISK_SEQUENCE)

        for witness in incident.witnesses:
            if witness.kind is WitnessKind.DIRECT_WITNESS:
                if not incident.outcome.failed:
                    raise CreditCompilationError("direct failure witness belongs to a non-failed incident")
                witnessed_steps = self._validated_loop_steps(incident, witness)
                actor_steps = self._actor_steps(incident, witnessed_steps)
                for step_index in actor_steps:
                    self._insert_direct(
                        direct_by_step,
                        DirectPolicyCredit(
                            step_index=step_index,
                            target=DirectPolicyTarget.AVOID,
                            witness_id=witness.witness_id,
                        ),
                    )
                if actor_steps:
                    emitted_strata.add(EvidenceStratum.DIRECT_WITNESS)
            elif witness.kind is WitnessKind.MULTI_EDGE_CYCLE:
                if not incident.outcome.failed:
                    raise CreditCompilationError("cycle witness belongs to a non-failed incident")
                witnessed_steps = self._validated_loop_steps(incident, witness)
                actor_steps = self._actor_steps(incident, witnessed_steps)
                if actor_steps:
                    behavior = witness.behavior_mean_log_probability
                    if behavior is None:
                        behavior = sum(context.steps[index].behavior_log_probability for index in actor_steps) / len(
                            actor_steps
                        )
                    cycle_targets.append(
                        CyclePolicyCredit(
                            step_indices=actor_steps,
                            behavior_mean_log_probability=behavior,
                            margin=self.config.cycle_margin,
                            witness_id=witness.witness_id,
                        )
                    )
                    emitted_strata.add(EvidenceStratum.MULTI_EDGE_CYCLE)
            elif witness.kind is WitnessKind.MATCHED_OUTCOME_PAIR:
                if witness.outcome_pair is None:  # pragma: no cover - contract invariant
                    raise CreditCompilationError("matched witness lost its atomic pair")
                contrast_targets.append(
                    ContrastPolicyCredit(
                        pair=witness.outcome_pair,
                        margin=self.config.contrast_margin,
                        witness_id=witness.witness_id,
                    )
                )
                emitted_strata.add(EvidenceStratum.MATCHED_OUTCOME_PAIR)
            elif witness.kind is WitnessKind.COMPLETION_CONTROL:
                if incident.outcome is not FailureOutcome.COMPLETED:
                    raise CreditCompilationError("completion witness belongs to a non-completed incident")
                actor_steps = self._actor_steps(incident, witness.attributed_step_indices)
                for step_index in actor_steps:
                    self._insert_direct(
                        direct_by_step,
                        DirectPolicyCredit(
                            step_index=step_index,
                            target=DirectPolicyTarget.PREFER,
                            witness_id=witness.witness_id,
                        ),
                    )
                if actor_steps:
                    emitted_strata.add(EvidenceStratum.COMPLETION_CONTROL)
            elif witness.kind is WitnessKind.RISK_SEQUENCE:
                # The local outcome above is the authoritative target.  The
                # witness supplies provenance only and never fabricates actor blame.
                continue

        if incident.outcome is FailureOutcome.DEADLOCK_STALL and not (
            direct_by_step or cycle_targets or contrast_targets
        ):
            emitted_strata.add(EvidenceStratum.UNRESOLVED_STALL)
        if not emitted_strata:
            # A verified completion remains useful value/Q control evidence
            # even when no individual choice is causally attributable.
            emitted_strata.add(EvidenceStratum.COMPLETION_CONTROL)

        ordered_strata = tuple(stratum for stratum in EvidenceStratum if stratum in emitted_strata)
        return CreditPlan(
            plan_id=self._plan_id(incident),
            incident_id=incident.incident_id,
            context=context,
            task_value_targets=tuple(task_targets),
            task_q_targets=tuple(task_targets),
            liveness_value_targets=tuple(liveness_targets),
            liveness_q_targets=tuple(liveness_targets),
            direct_policy_targets=tuple(direct_by_step[index] for index in sorted(direct_by_step)),
            cycle_policy_targets=tuple(cycle_targets),
            contrast_policy_targets=tuple(contrast_targets),
            risk_sequences=tuple(risk_sequences),
            strata=ordered_strata,
            provenance=incident.provenance,
        )

    @staticmethod
    def _actor_steps(
        incident: FailureIncident,
        step_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        return tuple(
            index
            for index in step_indices
            if index >= incident.context.burn_in_steps and incident.context.steps[index].actor_eligible
        )

    @staticmethod
    def _validated_loop_steps(
        incident: FailureIncident,
        witness: PolicyWitness,
    ) -> tuple[int, ...]:
        """Bind detector-time edges to factual retained steps.

        This is deliberately a membership check against immutable evidence.
        It never counts pairs, searches a tail, or infers a cycle.
        """

        detector_edges = {(edge.node, edge.action): edge for edge in witness.loop_edges}
        for step_index in witness.attributed_step_indices:
            step = incident.context.steps[step_index]
            edge = detector_edges.get((step.node.loop, step.selected_action.loop))
            if edge is None:
                raise CreditCompilationError("cycle witness does not name the attributed node/action loop edge")
            if step.episode_step not in edge.supporting_episode_steps:
                raise CreditCompilationError("cycle witness edge lacks the attributed detector episode step")
        return witness.attributed_step_indices

    @staticmethod
    def _insert_direct(
        targets: dict[int, DirectPolicyCredit],
        target: DirectPolicyCredit,
    ) -> None:
        prior = targets.get(target.step_index)
        if prior is not None and prior.target is not target.target:
            raise CreditCompilationError("one exact decision received contradictory direct policy targets")
        if prior is None:
            targets[target.step_index] = target

    def _plan_id(self, incident: FailureIncident) -> str:
        payload = (
            f"{FAILURE_CREDIT_COMPILER_VERSION}\0{incident.incident_id}\0"
            f"{self.config.task_discount:.17g}\0{self.config.liveness_discount:.17g}\0"
            f"{self.config.cycle_margin:.17g}\0{self.config.contrast_margin:.17g}"
        )
        return f"credit-plan:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


__all__ = [
    "CreditCompilationError",
    "CreditCompiler",
    "CreditCompilerConfig",
]
