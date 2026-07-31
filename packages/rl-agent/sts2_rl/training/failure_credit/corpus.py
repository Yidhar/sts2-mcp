"""Immutable, pre-indexed evidence corpus for failure-credit v4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np

from .actor_eligibility import (
    ACTOR_CREDIT_STRATA,
    actor_policy_versions,
    plan_has_effective_actor_unit,
)
from .contracts import (
    FAILURE_CREDIT_SCHEMA_VERSION,
    CreditPlan,
    EvidenceStratum,
    FailureIncident,
    IdentityTriple,
    LearningContext,
    MatchedOutcomePair,
    SemanticKey,
)

FAILURE_EVIDENCE_REPLAY_VERSION: Final = "sts2-failure-evidence-replay-v4"
ACTOR_QUOTA_STRATA: Final = ACTOR_CREDIT_STRATA


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    incident: FailureIncident
    plan: CreditPlan

    def __post_init__(self) -> None:
        if not isinstance(self.incident, FailureIncident):
            raise TypeError("evidence record incident has the wrong type")
        if not isinstance(self.plan, CreditPlan):
            raise TypeError("evidence record plan has the wrong type")
        if self.plan.incident_id != self.incident.incident_id:
            raise ValueError("evidence record incident/plan IDs differ")
        # The compiler and pipeline intentionally reuse one immutable context
        # object.  Accepting a distinct object with the same logical ID would
        # let incident provenance and learner inputs silently diverge, while
        # also making byte accounting undercount the second retained graph.
        if self.plan.context is not self.incident.context:
            raise ValueError(
                "evidence record incident/plan must share one immutable context"
            )
        if self.plan.provenance != self.incident.provenance:
            raise ValueError("evidence record incident/plan provenance differs")


@dataclass(frozen=True, slots=True)
class StratumQuota:
    stratum: EvidenceStratum
    minimum: int

    def __post_init__(self) -> None:
        if not isinstance(self.stratum, EvidenceStratum):
            raise TypeError("quota stratum has the wrong type")
        if isinstance(self.minimum, bool) or not isinstance(self.minimum, int):
            raise TypeError("quota minimum must be an integer")
        if self.minimum < 0:
            raise ValueError("quota minimum must be non-negative")


@dataclass(frozen=True, slots=True)
class StratumQuotaStatus:
    """One quota's total, actor-fresh, and critic-only selection evidence.

    For actor-credit strata, only ``selected_actor_fresh`` satisfies the
    requested minimum.  ``selected_stale_critic`` remains legal batch content
    for factual value/Q learning but cannot erase an actor-credit deficit.
    Non-actor strata mirror their ordinary eligible totals into the
    ``*_actor_fresh`` fields so the same deficit equation remains explicit.
    """

    stratum: EvidenceStratum
    requested: int
    available_total: int
    available_actor_fresh: int
    selected_actor_fresh: int
    selected_stale_critic: int
    deficit: int

    def __post_init__(self) -> None:
        if not isinstance(self.stratum, EvidenceStratum):
            raise TypeError("quota status stratum has the wrong type")
        for label, value in (
            ("requested", self.requested),
            ("available_total", self.available_total),
            ("available_actor_fresh", self.available_actor_fresh),
            ("selected_actor_fresh", self.selected_actor_fresh),
            ("selected_stale_critic", self.selected_stale_critic),
            ("deficit", self.deficit),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"quota status {label} must be an integer")
            if value < 0:
                raise ValueError(f"quota status {label} must be non-negative")
        if self.available_actor_fresh > self.available_total:
            raise ValueError("fresh quota availability exceeds total availability")
        if self.selected_actor_fresh > self.available_actor_fresh:
            raise ValueError("fresh quota selection exceeds fresh availability")
        if self.selected_actor_fresh + self.selected_stale_critic > self.available_total:
            raise ValueError("quota selected count exceeds total availability")
        if self.deficit != max(
            0,
            self.requested - self.selected_actor_fresh,
        ):
            raise ValueError("quota deficit is inconsistent")

    @property
    def available(self) -> int:
        """Compatibility spelling for total critic-eligible availability."""

        return self.available_total

    @property
    def selected(self) -> int:
        """Compatibility spelling for every selected record in the stratum."""

        return self.selected_actor_fresh + self.selected_stale_critic


@dataclass(frozen=True, slots=True)
class QuotaDiagnostics:
    statuses: tuple[StratumQuotaStatus, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.statuses, tuple) or not all(
            isinstance(status, StratumQuotaStatus) for status in self.statuses
        ):
            raise TypeError("quota diagnostics statuses have the wrong type")
        strata = tuple(status.stratum for status in self.statuses)
        if len(set(strata)) != len(strata):
            raise ValueError("quota diagnostics contain duplicate strata")

    @property
    def satisfied(self) -> bool:
        return all(status.deficit == 0 for status in self.statuses)

    @property
    def total_deficit(self) -> int:
        return sum(status.deficit for status in self.statuses)


@dataclass(frozen=True, slots=True)
class StratumCount:
    stratum: EvidenceStratum
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.stratum, EvidenceStratum):
            raise TypeError("stratum count has the wrong type")
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("stratum count must be an integer")
        if self.count < 0:
            raise ValueError("stratum count must be non-negative")


@dataclass(frozen=True, slots=True)
class EvidenceCorpusMetrics:
    record_count: int
    storage_nbytes: int
    actor_actionable_records: int
    direct_actor_eligible_records: int
    multi_edge_actor_eligible_records: int
    risk_actor_eligible_records: int
    outcome_pair_count: int
    stratum_counts: tuple[StratumCount, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("record_count", self.record_count),
            ("storage_nbytes", self.storage_nbytes),
            ("actor_actionable_records", self.actor_actionable_records),
            (
                "direct_actor_eligible_records",
                self.direct_actor_eligible_records,
            ),
            (
                "multi_edge_actor_eligible_records",
                self.multi_edge_actor_eligible_records,
            ),
            (
                "risk_actor_eligible_records",
                self.risk_actor_eligible_records,
            ),
            ("outcome_pair_count", self.outcome_pair_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"corpus metric {label} must be an integer")
            if value < 0:
                raise ValueError(f"corpus metric {label} must be non-negative")
        if not isinstance(self.stratum_counts, tuple) or not all(
            isinstance(item, StratumCount) for item in self.stratum_counts
        ):
            raise TypeError("corpus stratum counts have the wrong type")


@dataclass(frozen=True, slots=True)
class EvidenceSample:
    records: tuple[EvidenceRecord, ...]
    quota_diagnostics: QuotaDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.records, tuple) or not all(
            isinstance(record, EvidenceRecord) for record in self.records
        ):
            raise TypeError("evidence sample records have the wrong type")
        if not isinstance(self.quota_diagnostics, QuotaDiagnostics):
            raise TypeError("evidence sample diagnostics have the wrong type")


def _semantic_key_storage_nbytes(key: SemanticKey) -> int:
    return (
        len(key.namespace.encode("utf-8"))
        + len(key.schema_version.encode("utf-8"))
        + len(key.canonical_payload)
        + len(key.digest.encode("utf-8"))
        + 96
    )


def _identity_storage_nbytes(identity: IdentityTriple) -> int:
    return (
        _semantic_key_storage_nbytes(identity.exact)
        + _semantic_key_storage_nbytes(identity.loop)
        + _semantic_key_storage_nbytes(identity.comparison)
        + 64
    )


def _context_storage_nbytes(context: LearningContext) -> int:
    """Conservatively account one complete recurrent learning context."""

    total = int(context.initial_recurrent_state.nbytes) + 256
    total += len(context.context_id.encode("utf-8"))
    total += len(context.episode_id.encode("utf-8"))
    for step in context.steps:
        total += step.snapshot.storage_nbytes() + 160
        total += len(step.decision_id.encode("utf-8"))
        total += _identity_storage_nbytes(step.node)
        total += _semantic_key_storage_nbytes(step.anchor)
        total += sum(_identity_storage_nbytes(candidate) for candidate in step.candidate_actions)
    return total


def evidence_record_storage_nbytes(record: EvidenceRecord) -> int:
    """Return one record's logical payload size without corpus container bytes.

    Matched-outcome credit embeds both factual recurrent contexts.  The worse
    arm normally aliases ``incident.context`` while the better arm is an
    additional complete context; account every distinct retained context
    object exactly once so byte-bounded replay cannot silently omit the
    comparison arm.  Object identity is intentional here: two separately
    retained contexts that accidentally reuse one logical ID must not make
    byte accounting smaller.
    """

    if not isinstance(record, EvidenceRecord):
        raise TypeError("record must be EvidenceRecord")
    incident = record.incident
    contexts: list[LearningContext] = [incident.context]
    seen_contexts = {id(incident.context)}

    def retain_context(context: LearningContext) -> None:
        if id(context) not in seen_contexts:
            seen_contexts.add(id(context))
            contexts.append(context)

    # Defense in depth for accounting: EvidenceRecord currently requires this
    # to alias ``incident.context``, but keeping the ownership edge explicit
    # prevents a future relaxation from silently violating the byte bound.
    retain_context(record.plan.context)

    pair_metadata_nbytes = 0
    seen_pair_ids: set[str] = set()
    for target in record.plan.contrast_policy_targets:
        pair = target.pair
        retain_context(pair.better.context)
        retain_context(pair.worse.context)
        if pair.pair_id not in seen_pair_ids:
            seen_pair_ids.add(pair.pair_id)
            pair_metadata_nbytes += 256
            pair_metadata_nbytes += sum(
                len(value.encode("utf-8"))
                for value in (
                    pair.pair_id,
                    pair.better.incident_id,
                    pair.worse.incident_id,
                )
            )

    total = 512 + sum(_context_storage_nbytes(context) for context in contexts)
    total += pair_metadata_nbytes
    total += sum(
        _semantic_key_storage_nbytes(edge.node)
        + _semantic_key_storage_nbytes(edge.action)
        + 8 * len(edge.supporting_episode_steps)
        + 64
        for witness in incident.witnesses
        for edge in witness.loop_edges
    )
    total += 128 * (
        len(record.plan.task_value_targets)
        + len(record.plan.task_q_targets)
        + len(record.plan.liveness_value_targets)
        + len(record.plan.liveness_q_targets)
        + len(record.plan.direct_policy_targets)
        + len(record.plan.cycle_policy_targets)
        + len(record.plan.contrast_policy_targets)
        + len(record.plan.risk_sequences)
        + len(incident.witnesses)
    )
    total += sum(
        len(value.encode("utf-8"))
        for value in (
            incident.incident_id,
            incident.scope_key,
            incident.failure_kind,
            record.plan.plan_id,
            incident.provenance.run_id,
            incident.provenance.game_version,
            incident.provenance.environment_schema_version,
            incident.provenance.identity_version,
            incident.provenance.detector_version,
            incident.provenance.adapter_version,
            incident.provenance.collector_version,
        )
    )
    return total


def evidence_actor_policy_versions(
    record: EvidenceRecord,
    stratum: EvidenceStratum,
) -> tuple[int, ...]:
    """Return factual behavior versions for one actor-credit channel.

    This function intentionally follows compiled target indices instead of
    incident-level provenance.  Contexts may cross an actor publication
    boundary, and the learner suppresses stale policy labels per factual
    decision.  Forced steps never make a record actor-eligible.
    """

    if not isinstance(record, EvidenceRecord):
        raise TypeError("record must be EvidenceRecord")
    if not isinstance(stratum, EvidenceStratum):
        raise TypeError("stratum must be EvidenceStratum")
    return actor_policy_versions(record.plan, stratum)


def evidence_actor_is_fresh(
    record: EvidenceRecord,
    stratum: EvidenceStratum,
    *,
    current_policy_version: int,
    policy_gradient_max_lag: int,
    risk_actor_enabled: bool,
) -> bool:
    """Return whether a stratum offers an effective learner-fresh actor unit."""

    return plan_has_effective_actor_unit(
        record.plan,
        stratum,
        current_policy_version=current_policy_version,
        policy_gradient_max_lag=policy_gradient_max_lag,
        risk_actor_enabled=risk_actor_enabled,
    )


def _pair_signature(pair: MatchedOutcomePair) -> tuple[object, ...]:
    return (
        pair.better.incident_id,
        pair.better.context.context_id,
        pair.better.step.decision_id,
        pair.better.step.node.comparison,
        pair.better.step.selected_action.comparison,
        pair.worse.incident_id,
        pair.worse.context.context_id,
        pair.worse.step.decision_id,
        pair.worse.step.node.comparison,
        pair.worse.step.selected_action.comparison,
    )


@dataclass(frozen=True, slots=True)
class ImmutableEvidenceCorpus:
    """Persistent-value corpus; mutation returns a new fully indexed instance."""

    records: tuple[EvidenceRecord, ...] = ()
    version: str = FAILURE_EVIDENCE_REPLAY_VERSION
    _record_by_incident: Mapping[str, EvidenceRecord] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _stratum_index: Mapping[EvidenceStratum, tuple[str, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _outcome_pairs: Mapping[str, MatchedOutcomePair] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.version != FAILURE_EVIDENCE_REPLAY_VERSION:
            raise ValueError(f"unsupported evidence corpus version: {self.version!r}")
        if not isinstance(self.records, tuple) or not all(
            isinstance(record, EvidenceRecord) for record in self.records
        ):
            raise TypeError("evidence corpus records have the wrong type")
        by_incident: dict[str, EvidenceRecord] = {}
        strata: dict[EvidenceStratum, list[str]] = {stratum: [] for stratum in EvidenceStratum}
        outcome_pairs: dict[str, MatchedOutcomePair] = {}
        pair_signatures: dict[str, tuple[object, ...]] = {}
        for record in self.records:
            incident_id = record.incident.incident_id
            if incident_id in by_incident:
                raise ValueError(f"duplicate evidence incident ID: {incident_id}")
            by_incident[incident_id] = record
            for stratum in record.plan.strata:
                strata[stratum].append(incident_id)
            for target in record.plan.contrast_policy_targets:
                pair = target.pair
                signature = _pair_signature(pair)
                prior_signature = pair_signatures.get(pair.pair_id)
                if prior_signature is not None and prior_signature != signature:
                    raise ValueError("one outcome-pair ID names different atomic arms")
                pair_signatures[pair.pair_id] = signature
                outcome_pairs[pair.pair_id] = pair
        object.__setattr__(
            self,
            "_record_by_incident",
            MappingProxyType(by_incident),
        )
        object.__setattr__(
            self,
            "_stratum_index",
            MappingProxyType({stratum: tuple(incident_ids) for stratum, incident_ids in strata.items()}),
        )
        object.__setattr__(
            self,
            "_outcome_pairs",
            MappingProxyType(outcome_pairs),
        )

    def with_record(self, record: EvidenceRecord) -> ImmutableEvidenceCorpus:
        if not isinstance(record, EvidenceRecord):
            raise TypeError("record must be EvidenceRecord")
        if record.incident.incident_id in self._record_by_incident:
            raise ValueError("evidence corpus already contains this incident ID")
        return ImmutableEvidenceCorpus(
            records=(*self.records, record),
            version=self.version,
        )

    def record(self, incident_id: str) -> EvidenceRecord:
        try:
            return self._record_by_incident[incident_id]
        except KeyError as exc:
            raise KeyError(f"unknown evidence incident: {incident_id}") from exc

    def incident_ids(self, stratum: EvidenceStratum) -> tuple[str, ...]:
        if not isinstance(stratum, EvidenceStratum):
            raise TypeError("stratum has the wrong type")
        return self._stratum_index[stratum]

    @property
    def stratum_index(self) -> Mapping[EvidenceStratum, tuple[str, ...]]:
        return self._stratum_index

    @property
    def outcome_pairs(self) -> Mapping[str, MatchedOutcomePair]:
        return self._outcome_pairs

    def quota_diagnostics(
        self,
        quotas: tuple[StratumQuota, ...],
        *,
        risk_actor_enabled: bool,
        selected_incident_ids: tuple[str, ...] = (),
        current_policy_version: int | None = None,
        policy_gradient_max_lag: int | None = None,
    ) -> QuotaDiagnostics:
        if not isinstance(quotas, tuple) or not all(isinstance(quota, StratumQuota) for quota in quotas):
            raise TypeError("quotas must be a tuple of StratumQuota")
        if not isinstance(risk_actor_enabled, bool):
            raise TypeError("risk_actor_enabled must be a boolean")
        if len({quota.stratum for quota in quotas}) != len(quotas):
            raise ValueError("quotas contain duplicate strata")
        if current_policy_version is None and policy_gradient_max_lag is None:
            # Stand-alone diagnostics remain useful before a learner version
            # is known, but must not claim any actor quota is satisfiable.
            # ``sample`` performs the stricter check before selection.
            actor_fresh: Mapping[EvidenceStratum, tuple[str, ...]] = MappingProxyType(
                {stratum: () for stratum in ACTOR_QUOTA_STRATA}
            )
        else:
            actor_fresh = self._actor_fresh_incident_ids(
                quotas=quotas,
                current_policy_version=current_policy_version,
                policy_gradient_max_lag=policy_gradient_max_lag,
                risk_actor_enabled=risk_actor_enabled,
            )
        if not isinstance(selected_incident_ids, tuple) or not all(
            isinstance(incident_id, str) for incident_id in selected_incident_ids
        ):
            raise TypeError("selected_incident_ids must be a tuple of strings")
        unknown = set(selected_incident_ids).difference(self._record_by_incident)
        if unknown:
            raise ValueError(f"selected incident IDs are not in the corpus: {sorted(unknown)!r}")
        selected_set = set(selected_incident_ids)
        statuses: list[StratumQuotaStatus] = []
        for quota in quotas:
            available_ids = set(self._stratum_index[quota.stratum])
            selected_ids = available_ids.intersection(selected_set)
            if quota.stratum in ACTOR_QUOTA_STRATA:
                fresh_ids = set(actor_fresh[quota.stratum])
                selected_fresh = len(selected_ids.intersection(fresh_ids))
                selected_stale = len(selected_ids.difference(fresh_ids))
                available_fresh = len(fresh_ids)
            else:
                available_fresh = len(available_ids)
                selected_fresh = len(selected_ids)
                selected_stale = 0
            statuses.append(
                StratumQuotaStatus(
                    stratum=quota.stratum,
                    requested=quota.minimum,
                    available_total=len(available_ids),
                    available_actor_fresh=available_fresh,
                    selected_actor_fresh=selected_fresh,
                    selected_stale_critic=selected_stale,
                    deficit=max(0, quota.minimum - selected_fresh),
                )
            )
        return QuotaDiagnostics(statuses=tuple(statuses))

    def _actor_fresh_incident_ids(
        self,
        *,
        quotas: tuple[StratumQuota, ...],
        current_policy_version: int | None,
        policy_gradient_max_lag: int | None,
        risk_actor_enabled: bool,
    ) -> Mapping[EvidenceStratum, tuple[str, ...]]:
        """Build actor-fresh indexes for one immutable sampling operation."""

        if not isinstance(risk_actor_enabled, bool):
            raise TypeError("risk_actor_enabled must be a boolean")
        if (current_policy_version is None) != (policy_gradient_max_lag is None):
            raise ValueError("current_policy_version and policy_gradient_max_lag must be supplied together")
        requires_actor_freshness = any(quota.minimum > 0 and quota.stratum in ACTOR_QUOTA_STRATA for quota in quotas)
        if current_policy_version is None:
            if requires_actor_freshness:
                raise ValueError("actor evidence quotas require policy freshness parameters")
            return MappingProxyType({stratum: () for stratum in ACTOR_QUOTA_STRATA})
        if (
            isinstance(current_policy_version, bool)
            or not isinstance(current_policy_version, int)
            or current_policy_version < 0
        ):
            raise ValueError("current_policy_version must be a non-negative integer")
        if (
            isinstance(policy_gradient_max_lag, bool)
            or not isinstance(policy_gradient_max_lag, int)
            or policy_gradient_max_lag < 0
        ):
            raise ValueError("policy_gradient_max_lag must be a non-negative integer")
        for record in self.records:
            if record.plan.provenance.policy_version > current_policy_version:
                raise ValueError("failure evidence provenance is newer than the learner")
        return MappingProxyType(
            {
                stratum: tuple(
                    incident_id
                    for incident_id in self._stratum_index[stratum]
                    if evidence_actor_is_fresh(
                        self._record_by_incident[incident_id],
                        stratum,
                        current_policy_version=current_policy_version,
                        policy_gradient_max_lag=policy_gradient_max_lag,
                        risk_actor_enabled=risk_actor_enabled,
                    )
                )
                for stratum in ACTOR_QUOTA_STRATA
            }
        )

    @property
    def storage_nbytes(self) -> int:
        """Conservative accounting without serializing or duplicating contexts."""

        return 256 + sum(evidence_record_storage_nbytes(record) for record in self.records)

    def metrics(self) -> EvidenceCorpusMetrics:
        direct_actor_eligible = sum(
            bool(
                evidence_actor_policy_versions(
                    record,
                    EvidenceStratum.DIRECT_WITNESS,
                )
            )
            for record in self.records
        )
        multi_edge_actor_eligible = sum(
            bool(
                evidence_actor_policy_versions(
                    record,
                    EvidenceStratum.MULTI_EDGE_CYCLE,
                )
            )
            for record in self.records
        )
        risk_actor_eligible = sum(
            bool(
                evidence_actor_policy_versions(
                    record,
                    EvidenceStratum.RISK_SEQUENCE,
                )
            )
            for record in self.records
        )
        return EvidenceCorpusMetrics(
            record_count=len(self.records),
            storage_nbytes=self.storage_nbytes,
            actor_actionable_records=sum(
                record.plan.actor_label_count > 0
                or bool(
                    evidence_actor_policy_versions(
                        record,
                        EvidenceStratum.RISK_SEQUENCE,
                    )
                )
                for record in self.records
            ),
            direct_actor_eligible_records=direct_actor_eligible,
            multi_edge_actor_eligible_records=multi_edge_actor_eligible,
            risk_actor_eligible_records=risk_actor_eligible,
            outcome_pair_count=len(self._outcome_pairs),
            stratum_counts=tuple(
                StratumCount(
                    stratum=stratum,
                    count=len(self._stratum_index[stratum]),
                )
                for stratum in EvidenceStratum
            ),
        )

    def sample(
        self,
        *,
        batch_size: int,
        rng: np.random.Generator,
        risk_actor_enabled: bool,
        quotas: tuple[StratumQuota, ...] = (),
        current_policy_version: int | None = None,
        policy_gradient_max_lag: int | None = None,
    ) -> EvidenceSample:
        """Sample without locks or corpus mutation; caller checkpoints ``rng``."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        if not isinstance(quotas, tuple) or not all(isinstance(quota, StratumQuota) for quota in quotas):
            raise TypeError("quotas must be a tuple of StratumQuota")
        if len({quota.stratum for quota in quotas}) != len(quotas):
            raise ValueError("quotas contain duplicate strata")
        if sum(quota.minimum for quota in quotas) > batch_size:
            raise ValueError("quota minima exceed the requested batch size")
        actor_fresh = self._actor_fresh_incident_ids(
            quotas=quotas,
            current_policy_version=current_policy_version,
            policy_gradient_max_lag=policy_gradient_max_lag,
            risk_actor_enabled=risk_actor_enabled,
        )
        target_size = min(batch_size, len(self.records))
        selected: list[str] = []
        selected_set: set[str] = set()
        for quota in quotas:
            eligible_ids = (
                actor_fresh[quota.stratum]
                if quota.stratum in ACTOR_QUOTA_STRATA
                else self._stratum_index[quota.stratum]
            )
            already_selected = sum(incident_id in selected_set for incident_id in eligible_ids)
            needed = max(0, quota.minimum - already_selected)
            candidates = [incident_id for incident_id in eligible_ids if incident_id not in selected_set]
            if candidates:
                for position in rng.permutation(len(candidates))[:needed]:
                    incident_id = candidates[int(position)]
                    selected.append(incident_id)
                    selected_set.add(incident_id)
        remaining = [
            record.incident.incident_id for record in self.records if record.incident.incident_id not in selected_set
        ]
        if remaining and len(selected) < target_size:
            needed = target_size - len(selected)
            for position in rng.permutation(len(remaining))[:needed]:
                incident_id = remaining[int(position)]
                selected.append(incident_id)
                selected_set.add(incident_id)
        selected_ids = tuple(selected)
        return EvidenceSample(
            records=tuple(self._record_by_incident[item] for item in selected_ids),
            quota_diagnostics=self.quota_diagnostics(
                quotas,
                selected_incident_ids=selected_ids,
                current_policy_version=current_policy_version,
                policy_gradient_max_lag=policy_gradient_max_lag,
                risk_actor_enabled=risk_actor_enabled,
            ),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "version": FAILURE_EVIDENCE_REPLAY_VERSION,
            "schema_version": FAILURE_CREDIT_SCHEMA_VERSION,
            "records": self.records,
        }

    @classmethod
    def load_state_dict(cls, payload: object) -> ImmutableEvidenceCorpus:
        if not isinstance(payload, dict) or set(payload) != {
            "version",
            "schema_version",
            "records",
        }:
            raise ValueError("evidence corpus state has the wrong schema")
        if payload.get("version") != FAILURE_EVIDENCE_REPLAY_VERSION:
            raise ValueError("unsupported evidence corpus state version")
        if payload.get("schema_version") != FAILURE_CREDIT_SCHEMA_VERSION:
            raise ValueError("unsupported evidence contract schema version")
        records = payload.get("records")
        if not isinstance(records, tuple) or not all(isinstance(record, EvidenceRecord) for record in records):
            raise TypeError("evidence corpus state records have the wrong type")
        return cls(records=records)

    @classmethod
    def from_state_dict(cls, payload: object) -> ImmutableEvidenceCorpus:
        """Backward-compatible spelling for callers that construct a value."""

        return cls.load_state_dict(payload)


__all__ = [
    "ACTOR_QUOTA_STRATA",
    "FAILURE_EVIDENCE_REPLAY_VERSION",
    "EvidenceCorpusMetrics",
    "EvidenceRecord",
    "EvidenceSample",
    "ImmutableEvidenceCorpus",
    "QuotaDiagnostics",
    "StratumCount",
    "StratumQuota",
    "StratumQuotaStatus",
    "evidence_actor_is_fresh",
    "evidence_actor_policy_versions",
    "evidence_record_storage_nbytes",
]
