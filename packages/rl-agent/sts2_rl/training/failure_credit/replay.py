"""Bounded, thread-safe owner for immutable failure-credit v4 evidence.

The replay has two deliberately separate synchronization domains:

* ``_lock`` protects the currently published immutable corpus and accounting;
* ``_sample_lock`` serializes the owned NumPy RNG and exact checkpoint state.

A sampler acquires ``_lock`` only long enough to copy one
:class:`ImmutableEvidenceCorpus` reference.  All quota work and random
selection then happens against that immutable value outside ``_lock``.  An
actor can therefore publish evidence while a learner is sampling a large
corpus.  Checkpointing takes ``_sample_lock`` before ``_lock`` so the corpus,
RNG, and sampling counters form one exact-resume state.
"""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Final

import numpy as np

from .contracts import (
    FAILURE_CREDIT_SCHEMA_VERSION,
    EvidenceStratum,
)
from .corpus import (
    ACTOR_QUOTA_STRATA,
    FAILURE_EVIDENCE_REPLAY_VERSION,
    EvidenceRecord,
    EvidenceSample,
    ImmutableEvidenceCorpus,
    StratumQuota,
    evidence_record_storage_nbytes,
)

_REPLAY_STATE_FIELDS: Final = frozenset(
    {
        "version",
        "schema_version",
        "capacity",
        "byte_capacity",
        "corpus",
        "rng_state",
        "put_count",
        "put_batch_count",
        "sample_request_count",
        "sample_count",
        "eviction_count",
        "duplicate_count",
        "oversize_count",
        "quota_deficit_count",
        "maximum_observed_record_nbytes",
        "sampled_direct_actor_fresh_count",
        "sampled_direct_lag_suppressed_critic_count",
        "sampled_multi_edge_actor_fresh_count",
        "sampled_multi_edge_lag_suppressed_critic_count",
        "sampled_risk_actor_fresh_count",
        "sampled_risk_lag_suppressed_critic_count",
        "sampled_matched_pair_actor_fresh_count",
        "sampled_matched_pair_lag_suppressed_critic_count",
    }
)
_ACTOR_METRIC_PREFIX: Final = {
    EvidenceStratum.DIRECT_WITNESS: "direct",
    EvidenceStratum.MULTI_EDGE_CYCLE: "multi_edge",
    EvidenceStratum.RISK_SEQUENCE: "risk",
    EvidenceStratum.MATCHED_OUTCOME_PAIR: "matched_pair",
}


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _record_storage_nbytes(record: EvidenceRecord) -> int:
    """Return the record's conservative corpus contribution exactly once."""

    if not isinstance(record, EvidenceRecord):
        raise TypeError("failure evidence replay accepts only EvidenceRecord")
    size = evidence_record_storage_nbytes(record)
    if size <= 0:  # pragma: no cover - the corpus contract always has payload
        raise RuntimeError("failure evidence record has invalid storage accounting")
    return size


class BoundedFailureCreditReplay:
    """FIFO-bounded persistent replay with exact v4 checkpoint semantics.

    Records are immutable values.  Every accepted ``put`` publishes a newly
    indexed corpus with a compare-and-swap style generation check; the
    potentially expensive index construction is outside the publication lock.
    Eviction is deterministic FIFO in committed insertion order.  A matched
    outcome pair is never flattened into separate arms: both arms remain
    embedded in the owning :class:`EvidenceRecord` and are inserted, evicted,
    snapshotted, and checkpointed as one atomic value.
    """

    def __init__(self, *, capacity: int, byte_capacity: int, seed: int) -> None:
        self.capacity = _integer(
            capacity,
            label="failure evidence replay capacity",
            minimum=1,
        )
        self.byte_capacity = _integer(
            byte_capacity,
            label="failure evidence replay byte_capacity",
            minimum=1,
        )
        seed = _integer(
            seed,
            label="failure evidence replay seed",
        )
        self._rng = np.random.default_rng(seed)
        self._corpus = ImmutableEvidenceCorpus()
        self._record_sizes: tuple[int, ...] = ()
        self._incident_ids: frozenset[str] = frozenset()
        self._storage_nbytes = 0
        self._actor_actionable_records = 0
        self._direct_actor_eligible_records = 0
        self._multi_edge_actor_eligible_records = 0
        self._risk_actor_eligible_records = 0
        self._outcome_pair_count = 0
        self._stratum_counts: dict[EvidenceStratum, int] = {stratum: 0 for stratum in EvidenceStratum}
        self._put_count = 0
        self._put_batch_count = 0
        self._sample_request_count = 0
        self._sample_count = 0
        self._eviction_count = 0
        self._duplicate_count = 0
        self._oversize_count = 0
        self._quota_deficit_count = 0
        self._maximum_observed_record_nbytes = 0
        self._sampled_actor_fresh_count: dict[EvidenceStratum, int] = {stratum: 0 for stratum in ACTOR_QUOTA_STRATA}
        self._sampled_lag_suppressed_critic_count: dict[
            EvidenceStratum,
            int,
        ] = {stratum: 0 for stratum in ACTOR_QUOTA_STRATA}
        self._generation = 0
        self._sample_lock = Lock()
        self._lock = Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._corpus.records)

    def put(self, record: EvidenceRecord) -> bool:
        """Publish one immutable record, evicting deterministic FIFO prefixes.

        Duplicate live incident IDs are ignored and counted.  An individual
        record larger than the configured byte capacity is rejected
        fail-closed because accepting it would make the configured bound
        untrue.
        """

        return self.put_many((record,)) == 1

    def put_many(self, records: tuple[EvidenceRecord, ...]) -> int:
        """Atomically publish one ordered episode batch.

        Validation and byte sizing happen before publication.  If any item is
        oversized, the complete batch is rejected and no earlier item leaks
        into replay.  Live and within-batch duplicate incident IDs retain the
        single-item ``put`` behavior: the duplicate occurrence is counted and
        ignored while later unique records remain eligible.

        The ordered batch is simulated against one immutable snapshot with the
        same FIFO/byte behavior as sequential ``put`` calls, then exactly one
        corpus/index value is built and published.  Records inserted early in
        a batch may therefore be deterministically evicted by later records
        when the batch itself exceeds capacity.
        """

        if not isinstance(records, tuple) or not all(isinstance(record, EvidenceRecord) for record in records):
            raise TypeError("failure evidence replay batch must be an EvidenceRecord tuple")
        if not records:
            return 0
        sizes = tuple(_record_storage_nbytes(record) for record in records)
        maximum_size = max(sizes)
        oversize_count = sum(size > self.byte_capacity for size in sizes)
        with self._lock:
            self._maximum_observed_record_nbytes = max(
                self._maximum_observed_record_nbytes,
                maximum_size,
            )
            if oversize_count:
                self._oversize_count += oversize_count
                raise ValueError("one failure evidence record exceeds replay byte capacity")

        # Corpus indexing is O(live records).  Construct the candidate value
        # outside the publication lock and retry only if another put/load won
        # the race.  This keeps both learner sampling and index construction
        # from blocking the collector's critical publication section.
        while True:
            with self._lock:
                generation = self._generation
                retained_records = self._corpus.records
                retained_sizes = self._record_sizes
                retained_incident_ids = set(self._incident_ids)
                storage_nbytes = self._storage_nbytes

            accepted_count = 0
            duplicate_count = 0
            eviction_count = 0
            for record, size in zip(records, sizes, strict=True):
                incident_id = record.incident.incident_id
                if incident_id in retained_incident_ids:
                    duplicate_count += 1
                    continue
                while retained_records and (
                    len(retained_records) >= self.capacity or storage_nbytes + size > self.byte_capacity
                ):
                    evicted = retained_records[0]
                    retained_records = retained_records[1:]
                    storage_nbytes -= retained_sizes[0]
                    retained_sizes = retained_sizes[1:]
                    retained_incident_ids.remove(evicted.incident.incident_id)
                    eviction_count += 1
                retained_records = (*retained_records, record)
                retained_sizes = (*retained_sizes, size)
                retained_incident_ids.add(incident_id)
                storage_nbytes += size
                accepted_count += 1

            if not accepted_count:
                with self._lock:
                    if generation != self._generation:
                        continue
                    self._duplicate_count += duplicate_count
                    return 0

            candidate_records = retained_records
            candidate_sizes = retained_sizes
            candidate_storage_nbytes = storage_nbytes
            candidate_corpus = ImmutableEvidenceCorpus(
                records=candidate_records,
            )
            candidate_incident_ids = frozenset(retained_incident_ids)
            candidate_metrics = candidate_corpus.metrics()
            candidate_stratum_counts = {item.stratum: item.count for item in candidate_metrics.stratum_counts}

            with self._lock:
                if generation != self._generation:
                    continue
                self._corpus = candidate_corpus
                self._record_sizes = candidate_sizes
                self._incident_ids = candidate_incident_ids
                self._storage_nbytes = candidate_storage_nbytes
                self._actor_actionable_records = candidate_metrics.actor_actionable_records
                self._direct_actor_eligible_records = candidate_metrics.direct_actor_eligible_records
                self._multi_edge_actor_eligible_records = candidate_metrics.multi_edge_actor_eligible_records
                self._risk_actor_eligible_records = candidate_metrics.risk_actor_eligible_records
                self._outcome_pair_count = candidate_metrics.outcome_pair_count
                self._stratum_counts = candidate_stratum_counts
                self._put_count += accepted_count
                self._put_batch_count += 1
                self._eviction_count += eviction_count
                self._duplicate_count += duplicate_count
                self._generation += 1
                return accepted_count

    def sample(
        self,
        batch_size: int,
        *,
        risk_actor_enabled: bool,
        quotas: tuple[StratumQuota, ...] = (),
        current_policy_version: int | None = None,
        policy_gradient_max_lag: int | None = None,
    ) -> EvidenceSample:
        """Sample critics broadly while satisfying actor quotas only from fresh data."""

        # ``ImmutableEvidenceCorpus.sample`` owns the canonical validation and
        # quota semantics.  The separate lock prevents concurrent sample calls
        # from racing the replay-owned RNG while still permitting put().
        with self._sample_lock:
            with self._lock:
                corpus = self._corpus
            result = corpus.sample(
                batch_size=batch_size,
                rng=self._rng,
                quotas=quotas,
                current_policy_version=current_policy_version,
                policy_gradient_max_lag=policy_gradient_max_lag,
                risk_actor_enabled=risk_actor_enabled,
            )
            with self._lock:
                self._sample_request_count += 1
                self._sample_count += len(result.records)
                self._quota_deficit_count += result.quota_diagnostics.total_deficit
                for status in result.quota_diagnostics.statuses:
                    if status.stratum not in ACTOR_QUOTA_STRATA:
                        continue
                    self._sampled_actor_fresh_count[status.stratum] += status.selected_actor_fresh
                    self._sampled_lag_suppressed_critic_count[status.stratum] += status.selected_stale_critic
            return result

    def snapshot(self) -> ImmutableEvidenceCorpus:
        """Return the currently published persistent corpus in O(1)."""

        with self._lock:
            return self._corpus

    def metrics(self) -> dict[str, int | str]:
        """Return JSON-ready replay and all-seven-strata diagnostics."""

        with self._lock:
            corpus = self._corpus
            result: dict[str, int | str] = {
                "version": FAILURE_EVIDENCE_REPLAY_VERSION,
                "size": len(corpus.records),
                "capacity": self.capacity,
                "storage_nbytes": self._storage_nbytes,
                "byte_capacity": self.byte_capacity,
                "put_count": self._put_count,
                "put_batch_count": self._put_batch_count,
                "sample_request_count": self._sample_request_count,
                "sample_count": self._sample_count,
                "eviction_count": self._eviction_count,
                "duplicate_count": self._duplicate_count,
                "oversize_count": self._oversize_count,
                "quota_deficit_count": self._quota_deficit_count,
                "maximum_observed_record_nbytes": (self._maximum_observed_record_nbytes),
                "actor_actionable_records": self._actor_actionable_records,
                "direct_actor_eligible_records": (self._direct_actor_eligible_records),
                "multi_edge_actor_eligible_records": (self._multi_edge_actor_eligible_records),
                "risk_actor_eligible_records": (self._risk_actor_eligible_records),
                "outcome_pair_count": self._outcome_pair_count,
            }
            for stratum, count in self._stratum_counts.items():
                result[f"stratum_{stratum.value.lower()}_size"] = count
            for stratum, prefix in _ACTOR_METRIC_PREFIX.items():
                result[f"sampled_{prefix}_actor_fresh_count"] = self._sampled_actor_fresh_count[stratum]
                result[f"sampled_{prefix}_lag_suppressed_critic_count"] = self._sampled_lag_suppressed_critic_count[
                    stratum
                ]
        return result

    def state_dict(self) -> dict[str, object]:
        """Return a strict exact-resume v4 state, including owned RNG."""

        with self._sample_lock:
            with self._lock:
                return {
                    "version": FAILURE_EVIDENCE_REPLAY_VERSION,
                    "schema_version": FAILURE_CREDIT_SCHEMA_VERSION,
                    "capacity": self.capacity,
                    "byte_capacity": self.byte_capacity,
                    "corpus": self._corpus.state_dict(),
                    "rng_state": deepcopy(self._rng.bit_generator.state),
                    "put_count": self._put_count,
                    "put_batch_count": self._put_batch_count,
                    "sample_request_count": self._sample_request_count,
                    "sample_count": self._sample_count,
                    "eviction_count": self._eviction_count,
                    "duplicate_count": self._duplicate_count,
                    "oversize_count": self._oversize_count,
                    "quota_deficit_count": self._quota_deficit_count,
                    "maximum_observed_record_nbytes": (self._maximum_observed_record_nbytes),
                    "sampled_direct_actor_fresh_count": (
                        self._sampled_actor_fresh_count[EvidenceStratum.DIRECT_WITNESS]
                    ),
                    "sampled_direct_lag_suppressed_critic_count": (
                        self._sampled_lag_suppressed_critic_count[EvidenceStratum.DIRECT_WITNESS]
                    ),
                    "sampled_multi_edge_actor_fresh_count": (
                        self._sampled_actor_fresh_count[EvidenceStratum.MULTI_EDGE_CYCLE]
                    ),
                    "sampled_multi_edge_lag_suppressed_critic_count": (
                        self._sampled_lag_suppressed_critic_count[EvidenceStratum.MULTI_EDGE_CYCLE]
                    ),
                    "sampled_risk_actor_fresh_count": (self._sampled_actor_fresh_count[EvidenceStratum.RISK_SEQUENCE]),
                    "sampled_risk_lag_suppressed_critic_count": (
                        self._sampled_lag_suppressed_critic_count[EvidenceStratum.RISK_SEQUENCE]
                    ),
                    "sampled_matched_pair_actor_fresh_count": (
                        self._sampled_actor_fresh_count[EvidenceStratum.MATCHED_OUTCOME_PAIR]
                    ),
                    "sampled_matched_pair_lag_suppressed_critic_count": (
                        self._sampled_lag_suppressed_critic_count[EvidenceStratum.MATCHED_OUTCOME_PAIR]
                    ),
                }

    def load_state_dict(self, payload: object) -> None:
        """Validate a complete v4 state before replacing this replay atomically."""

        if not isinstance(payload, dict):
            raise TypeError("failure evidence replay checkpoint must be an object")
        if set(payload) != _REPLAY_STATE_FIELDS or payload.get("version") != FAILURE_EVIDENCE_REPLAY_VERSION:
            raise ValueError("unsupported failure evidence replay checkpoint schema")
        if payload.get("schema_version") != FAILURE_CREDIT_SCHEMA_VERSION:
            raise ValueError("unsupported failure evidence contract schema")
        if payload.get("capacity") != self.capacity:
            raise ValueError("failure evidence replay checkpoint capacity differs")
        if payload.get("byte_capacity") != self.byte_capacity:
            raise ValueError("failure evidence replay checkpoint byte_capacity differs")

        corpus = ImmutableEvidenceCorpus.load_state_dict(payload.get("corpus"))
        if len(corpus.records) > self.capacity:
            raise ValueError("failure evidence replay checkpoint exceeds record capacity")
        record_sizes = tuple(_record_storage_nbytes(record) for record in corpus.records)
        storage_nbytes = sum(record_sizes)
        if storage_nbytes > self.byte_capacity:
            raise ValueError("failure evidence replay checkpoint exceeds byte capacity")

        counters: dict[str, int] = {}
        for name in (
            "put_count",
            "put_batch_count",
            "sample_request_count",
            "sample_count",
            "eviction_count",
            "duplicate_count",
            "oversize_count",
            "quota_deficit_count",
            "maximum_observed_record_nbytes",
            "sampled_direct_actor_fresh_count",
            "sampled_direct_lag_suppressed_critic_count",
            "sampled_multi_edge_actor_fresh_count",
            "sampled_multi_edge_lag_suppressed_critic_count",
            "sampled_risk_actor_fresh_count",
            "sampled_risk_lag_suppressed_critic_count",
            "sampled_matched_pair_actor_fresh_count",
            "sampled_matched_pair_lag_suppressed_critic_count",
        ):
            counters[name] = _integer(
                payload.get(name),
                label=f"failure evidence replay {name}",
            )
        if counters["put_count"] != (len(corpus.records) + counters["eviction_count"]):
            raise ValueError("failure evidence replay put/eviction accounting is inconsistent")
        if counters["put_batch_count"] > counters["put_count"]:
            raise ValueError("failure evidence replay batch accounting is inconsistent")
        if counters["sample_count"] > (counters["sample_request_count"] * self.capacity):
            raise ValueError("failure evidence replay sample accounting is inconsistent")
        for prefix in ("direct", "multi_edge", "risk", "matched_pair"):
            if (
                counters[f"sampled_{prefix}_actor_fresh_count"]
                + counters[f"sampled_{prefix}_lag_suppressed_critic_count"]
                > counters["sample_count"]
            ):
                raise ValueError("failure evidence replay actor-freshness accounting is inconsistent")
        largest_record = max(record_sizes, default=0)
        if counters["maximum_observed_record_nbytes"] < largest_record:
            raise ValueError("failure evidence replay maximum observed record size is inconsistent")

        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, dict):
            raise TypeError("failure evidence replay RNG checkpoint is invalid")
        probe = np.random.default_rng()
        try:
            probe.bit_generator.state = deepcopy(rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("failure evidence replay RNG checkpoint is invalid") from exc

        incident_ids = frozenset(record.incident.incident_id for record in corpus.records)
        corpus_metrics = corpus.metrics()
        stratum_counts = {item.stratum: item.count for item in corpus_metrics.stratum_counts}
        with self._sample_lock:
            with self._lock:
                self._corpus = corpus
                self._record_sizes = record_sizes
                self._incident_ids = incident_ids
                self._storage_nbytes = storage_nbytes
                self._actor_actionable_records = corpus_metrics.actor_actionable_records
                self._direct_actor_eligible_records = corpus_metrics.direct_actor_eligible_records
                self._multi_edge_actor_eligible_records = corpus_metrics.multi_edge_actor_eligible_records
                self._risk_actor_eligible_records = corpus_metrics.risk_actor_eligible_records
                self._outcome_pair_count = corpus_metrics.outcome_pair_count
                self._stratum_counts = stratum_counts
                self._rng.bit_generator.state = deepcopy(rng_state)
                self._put_count = counters["put_count"]
                self._put_batch_count = counters["put_batch_count"]
                self._sample_request_count = counters["sample_request_count"]
                self._sample_count = counters["sample_count"]
                self._eviction_count = counters["eviction_count"]
                self._duplicate_count = counters["duplicate_count"]
                self._oversize_count = counters["oversize_count"]
                self._quota_deficit_count = counters["quota_deficit_count"]
                self._maximum_observed_record_nbytes = counters["maximum_observed_record_nbytes"]
                self._sampled_actor_fresh_count = {
                    EvidenceStratum.DIRECT_WITNESS: counters["sampled_direct_actor_fresh_count"],
                    EvidenceStratum.MULTI_EDGE_CYCLE: counters["sampled_multi_edge_actor_fresh_count"],
                    EvidenceStratum.RISK_SEQUENCE: counters["sampled_risk_actor_fresh_count"],
                    EvidenceStratum.MATCHED_OUTCOME_PAIR: counters["sampled_matched_pair_actor_fresh_count"],
                }
                self._sampled_lag_suppressed_critic_count = {
                    EvidenceStratum.DIRECT_WITNESS: counters["sampled_direct_lag_suppressed_critic_count"],
                    EvidenceStratum.MULTI_EDGE_CYCLE: counters["sampled_multi_edge_lag_suppressed_critic_count"],
                    EvidenceStratum.RISK_SEQUENCE: counters["sampled_risk_lag_suppressed_critic_count"],
                    EvidenceStratum.MATCHED_OUTCOME_PAIR: counters[
                        "sampled_matched_pair_lag_suppressed_critic_count"
                    ],
                }
                self._generation += 1


__all__ = ["BoundedFailureCreditReplay"]
