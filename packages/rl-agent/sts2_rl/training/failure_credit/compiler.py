"""Pure evidence-to-credit compiler for failure-credit v4.

Since v20 the compiler emits only scalar task/liveness value/Q critic targets
plus the record's detector-stratum classification.  The direct/cycle/contrast
policy credits and the risk actor sequence were retired; detector witnesses
remain validated evidence and continue to drive stratum classification.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .contracts import (
    FAILURE_CREDIT_COMPILER_VERSION,
    CreditPlan,
    EvidenceStratum,
    FailureIncident,
    FailureOutcome,
    PolicyWitness,
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

    def __post_init__(self) -> None:
        for label, value in (
            ("task_discount", self.task_discount),
            ("liveness_discount", self.liveness_discount),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise TypeError(f"{label} must be finite")
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{label} must be in (0, 1]")


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
        emitted_strata: set[EvidenceStratum] = set()
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
                emitted_strata.add(EvidenceStratum.RISK_SEQUENCE)

        for witness in incident.witnesses:
            if witness.kind is WitnessKind.DIRECT_WITNESS:
                if not incident.outcome.failed:
                    raise CreditCompilationError("direct failure witness belongs to a non-failed incident")
                self._validated_loop_steps(incident, witness)
                emitted_strata.add(EvidenceStratum.DIRECT_WITNESS)
            elif witness.kind is WitnessKind.MULTI_EDGE_CYCLE:
                if not incident.outcome.failed:
                    raise CreditCompilationError("cycle witness belongs to a non-failed incident")
                self._validated_loop_steps(incident, witness)
                emitted_strata.add(EvidenceStratum.MULTI_EDGE_CYCLE)
            elif witness.kind is WitnessKind.MATCHED_OUTCOME_PAIR:
                if witness.outcome_pair is None:  # pragma: no cover - contract invariant
                    raise CreditCompilationError("matched witness lost its atomic pair")
                emitted_strata.add(EvidenceStratum.MATCHED_OUTCOME_PAIR)
            elif witness.kind is WitnessKind.COMPLETION_CONTROL:
                if incident.outcome is not FailureOutcome.COMPLETED:
                    raise CreditCompilationError("completion witness belongs to a non-completed incident")
                emitted_strata.add(EvidenceStratum.COMPLETION_CONTROL)
            elif witness.kind is WitnessKind.RISK_SEQUENCE:
                # The local outcome above is the authoritative target.  The
                # witness supplies provenance only and never fabricates actor blame.
                continue

        if incident.outcome is FailureOutcome.DEADLOCK_STALL and not (
            emitted_strata
            & {
                EvidenceStratum.DIRECT_WITNESS,
                EvidenceStratum.MULTI_EDGE_CYCLE,
                EvidenceStratum.MATCHED_OUTCOME_PAIR,
            }
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
            strata=ordered_strata,
            provenance=incident.provenance,
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

    def _plan_id(self, incident: FailureIncident) -> str:
        payload = (
            f"{FAILURE_CREDIT_COMPILER_VERSION}\0{incident.incident_id}\0"
            f"{self.config.task_discount:.17g}\0{self.config.liveness_discount:.17g}"
        )
        return f"credit-plan:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


__all__ = [
    "CreditCompilationError",
    "CreditCompiler",
    "CreditCompilerConfig",
]
