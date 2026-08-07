"""Strict cross-episode outcome matching for failure-credit replay v5.

The detector can prove that an action participates in a semantic cycle, but it
cannot invent the action that would have escaped it.  This module joins that
authoritative failure evidence with an independently observed, verified
completion at the *same* comparison decision.  The resulting contrast target
therefore says only what the data demonstrated: under the same reviewed node
and candidate set, one action completed the transaction while another entered
an attributed failure.

Matching is deliberately stateless.  The retained immutable replay corpus is
the catalog, so exact checkpoint/resume needs no second mutable index or RNG.
Every digest comparison includes its canonical payload and every pair keeps
both recurrent contexts atomically in the enriched failure record.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Final

from .compiler import CreditCompiler, CreditCompilerConfig
from .contracts import (
    CreditProvenance,
    EvidenceStratum,
    FailureIncident,
    FailureOutcome,
    LearningStep,
    MatchedOutcomePair,
    OutcomeArm,
    PolicyWitness,
    SemanticKey,
    TargetAuthority,
    WitnessKind,
)
from .corpus import EvidenceRecord

FAILURE_CREDIT_MATCHER_VERSION: Final = "sts2-outcome-pair-matcher-v3"


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\0".join((prefix, *(str(part) for part in parts)))
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _semantic_identity(key: SemanticKey) -> tuple[str, str, str, bytes]:
    # SemanticKey validates its digest/payload relationship at construction
    # and again inside every LearningStep.  Keep the payload in the join key:
    # a digest is an index, never collision authority.
    return (
        key.namespace,
        key.schema_version,
        key.digest,
        key.canonical_payload,
    )


def _candidate_comparison_set(
    step: LearningStep,
) -> tuple[tuple[str, str, str, bytes], ...]:
    # Candidate order is presentation detail, but multiplicity is semantic:
    # two strictly equal selectable instances are not the same decision as one.
    return tuple(sorted(_semantic_identity(candidate.comparison) for candidate in step.candidate_actions))


def _comparison_key(
    record: EvidenceRecord,
    step_index: int,
) -> tuple[
    str,
    tuple[str, str, str, bytes],
    tuple[tuple[str, str, str, bytes], ...],
]:
    step = record.incident.context.steps[step_index]
    return (
        record.incident.scope_key,
        _semantic_identity(step.node.comparison),
        _candidate_comparison_set(step),
    )


def _provenance_abi(provenance: CreditProvenance) -> tuple[str, ...]:
    """Return semantic provenance shared across exact continuation segments.

    ``run_id`` is a storage/launch lineage coordinate, not an environment or
    evidence semantic.  Keeping it here made otherwise identical records on
    opposite sides of a resume boundary impossible to pair.  Every actual
    semantic version and the training partition remain fail-closed below.
    """

    return (
        provenance.game_version,
        provenance.environment_schema_version,
        provenance.identity_version,
        provenance.detector_version,
        provenance.adapter_version,
        provenance.collector_version,
        provenance.data_partition,
        provenance.schema_version,
    )


def _completion_step_indices(record: EvidenceRecord) -> tuple[int, ...]:
    incident = record.incident
    if (
        incident.outcome is not FailureOutcome.COMPLETED
        or incident.local_authority is not TargetAuthority.VERIFIED_TRANSITION
        or incident.local_failure_cost != 0.0
        or EvidenceStratum.CENSORED in record.plan.strata
    ):
        return ()
    return tuple(index for index in incident.context.learn_step_indices if incident.context.steps[index].actor_eligible)


def _attributed_failure_step_indices(record: EvidenceRecord) -> tuple[int, ...]:
    """Return failure-arm decisions eligible for an exact outcome contrast.

    Direct and cycle witnesses remain the strongest local attribution. A
    detector-confirmed unresolved stall has no defensible single-step AVOID
    target, but its immutable risk sequence may still be used as the *worse*
    arm when another episode demonstrates a different action from the exact
    same comparison node and candidate multiset completing the scope. The
    matcher therefore adds information rather than inventing escape credit.
    """

    incident = record.incident
    if (
        not incident.outcome.failed
        or incident.local_authority is TargetAuthority.CENSORED
        or incident.local_failure_cost is None
        or incident.local_failure_cost <= 0.0
        or EvidenceStratum.CENSORED in record.plan.strata
        or EvidenceStratum.MATCHED_OUTCOME_PAIR in record.plan.strata
    ):
        return ()
    candidates = {target.step_index for target in record.plan.direct_policy_targets}
    candidates.update(
        step_index
        for target in record.plan.cycle_policy_targets
        for step_index in target.step_indices
    )
    if not candidates and EvidenceStratum.UNRESOLVED_STALL in record.plan.strata:
        candidates.update(
            step_index
            for sequence in record.plan.risk_sequences
            for step_index in sequence.step_indices
        )
    return tuple(
        index
        for index in sorted(candidates)
        if incident.context.steps[index].actor_eligible
    )


@dataclass(frozen=True, slots=True)
class OutcomeMatchPublication:
    """One atomic publication decision produced from a replay snapshot."""

    records: tuple[EvidenceRecord, ...]
    replacements: tuple[EvidenceRecord, ...]
    matched_pair_count: int

    def __post_init__(self) -> None:
        if not all(isinstance(record, EvidenceRecord) for record in self.records):
            raise TypeError("matched publication records have the wrong type")
        if not all(isinstance(record, EvidenceRecord) for record in self.replacements):
            raise TypeError("matched publication replacements have the wrong type")
        if isinstance(self.matched_pair_count, bool) or not isinstance(
            self.matched_pair_count,
            int,
        ):
            raise TypeError("matched_pair_count must be an integer")
        if self.matched_pair_count < 0:
            raise ValueError("matched_pair_count must be non-negative")
        record_ids = tuple(record.incident.incident_id for record in self.records)
        replacement_ids = tuple(record.incident.incident_id for record in self.replacements)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("matched publication contains duplicate incoming incidents")
        if len(set(replacement_ids)) != len(replacement_ids):
            raise ValueError("matched publication contains duplicate replacements")
        if set(record_ids) & set(replacement_ids):
            raise ValueError("one incident cannot be both inserted and replaced")


class OutcomePairMatcher:
    """Derive strict completion-vs-attributed-failure comparisons.

    A failure receives at most one matched pair.  This bounds each learner
    record to the reviewed two-context ABI and avoids turning repeated ordinary
    completions into an unbounded evidence fan-out.
    """

    def __init__(
        self,
        *,
        maximum_pairs_per_publication: int,
        contrast_margin: float = 0.10,
    ) -> None:
        if isinstance(maximum_pairs_per_publication, bool) or not isinstance(maximum_pairs_per_publication, int):
            raise TypeError("maximum_pairs_per_publication must be an integer")
        if maximum_pairs_per_publication < 0:
            raise ValueError("maximum_pairs_per_publication must be non-negative")
        self.maximum_pairs_per_publication = maximum_pairs_per_publication
        self.compiler = CreditCompiler(CreditCompilerConfig(contrast_margin=contrast_margin))

    def match(
        self,
        records: tuple[EvidenceRecord, ...],
        *,
        retained_records: tuple[EvidenceRecord, ...],
    ) -> OutcomeMatchPublication:
        if not isinstance(records, tuple) or not all(isinstance(record, EvidenceRecord) for record in records):
            raise TypeError("outcome matcher records must be an EvidenceRecord tuple")
        if not isinstance(retained_records, tuple) or not all(
            isinstance(record, EvidenceRecord) for record in retained_records
        ):
            raise TypeError("outcome matcher retained_records must be an EvidenceRecord tuple")
        if not records or self.maximum_pairs_per_publication == 0:
            return OutcomeMatchPublication(records, (), 0)

        retained_ids = {record.incident.incident_id for record in retained_records}
        incoming_by_id = {record.incident.incident_id: record for record in records}
        if len(incoming_by_id) != len(records):
            raise ValueError("outcome matcher incoming incidents must be unique")
        overlap = retained_ids & set(incoming_by_id)
        if overlap:
            raise ValueError("outcome matcher received incidents already present in replay")

        # Deterministic ordering makes a publication independent of hash-map
        # layout and wall-clock interleaving.  The retained corpus order remains
        # FIFO evidence order; incident/step tie-breakers make it auditable.
        all_records = (*retained_records, *records)
        completions = sorted(
            ((record, step_index) for record in all_records for step_index in _completion_step_indices(record)),
            key=lambda item: (
                item[0].incident.incident_id,
                item[1],
            ),
        )
        failures = sorted(
            ((record, step_index) for record in all_records for step_index in _attributed_failure_step_indices(record)),
            key=lambda item: (
                item[0].incident.incident_id,
                item[1],
            ),
        )

        enriched: dict[str, EvidenceRecord] = {}
        pair_count = 0
        for worse_record, worse_index in failures:
            if pair_count >= self.maximum_pairs_per_publication:
                break
            if worse_record.incident.incident_id in enriched:
                continue
            # Only derive a pair when at least one arm arrived in this
            # publication.  Otherwise every later call would rescan and emit
            # the same retained-vs-retained join.
            for better_record, better_index in completions:
                if (
                    worse_record.incident.incident_id not in incoming_by_id
                    and better_record.incident.incident_id not in incoming_by_id
                ):
                    continue
                if not self._compatible_sources(worse_record, better_record):
                    continue
                if _comparison_key(worse_record, worse_index) != _comparison_key(
                    better_record,
                    better_index,
                ):
                    continue
                worse_step = worse_record.incident.context.steps[worse_index]
                better_step = better_record.incident.context.steps[better_index]
                if _semantic_identity(worse_step.selected_action.comparison) == _semantic_identity(
                    better_step.selected_action.comparison
                ):
                    continue
                enriched_record = self._enrich_failure(
                    worse_record,
                    worse_index=worse_index,
                    better_record=better_record,
                    better_index=better_index,
                )
                enriched[worse_record.incident.incident_id] = enriched_record
                pair_count += 1
                break

        transformed = tuple(enriched.get(record.incident.incident_id, record) for record in records)
        replacements = tuple(
            enriched[record.incident.incident_id]
            for record in retained_records
            if record.incident.incident_id in enriched
        )
        return OutcomeMatchPublication(
            records=transformed,
            replacements=replacements,
            matched_pair_count=pair_count,
        )

    @staticmethod
    def _compatible_sources(
        worse_record: EvidenceRecord,
        better_record: EvidenceRecord,
    ) -> bool:
        return (
            worse_record.incident.incident_id != better_record.incident.incident_id
            and worse_record.incident.scope_key == better_record.incident.scope_key
            and _provenance_abi(worse_record.incident.provenance) == _provenance_abi(better_record.incident.provenance)
        )

    def _enrich_failure(
        self,
        worse_record: EvidenceRecord,
        *,
        worse_index: int,
        better_record: EvidenceRecord,
        better_index: int,
    ) -> EvidenceRecord:
        worse = worse_record.incident
        better = better_record.incident
        pair_id = _stable_id(
            "matched-outcome-pair",
            better.incident_id,
            better_index,
            worse.incident_id,
            worse_index,
            _semantic_identity(better.context.steps[better_index].selected_action.comparison),
            _semantic_identity(worse.context.steps[worse_index].selected_action.comparison),
        )
        pair = MatchedOutcomePair(
            pair_id=pair_id,
            better=OutcomeArm(
                incident_id=better.incident_id,
                context=better.context,
                step_index=better_index,
                outcome=better.outcome,
            ),
            worse=OutcomeArm(
                incident_id=worse.incident_id,
                context=worse.context,
                step_index=worse_index,
                outcome=worse.outcome,
            ),
        )
        witness = PolicyWitness(
            witness_id=_stable_id("matched-outcome-witness", pair_id),
            kind=WitnessKind.MATCHED_OUTCOME_PAIR,
            attributed_step_indices=(),
            supporting_episode_steps=(worse.context.steps[worse_index].episode_step,),
            occurrences=2,
            cycle_span=None,
            successor_confirmed=True,
            outcome_pair=pair,
        )
        provenance = replace(
            worse.provenance,
            policy_version=max(
                worse.provenance.policy_version,
                better.provenance.policy_version,
                worse.context.steps[worse_index].policy_version,
                better.context.steps[better_index].policy_version,
            ),
        )
        incident: FailureIncident = replace(
            worse,
            witnesses=(*worse.witnesses, witness),
            provenance=provenance,
        )
        return EvidenceRecord(
            incident=incident,
            plan=self.compiler.compile(incident),
        )


__all__ = [
    "FAILURE_CREDIT_MATCHER_VERSION",
    "OutcomeMatchPublication",
    "OutcomePairMatcher",
]
