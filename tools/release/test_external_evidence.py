from __future__ import annotations

import copy
import unittest
from datetime import datetime, timezone

from tools.release.check_external_evidence import (
    EvidenceError,
    validate_compatibility_evidence,
    validate_parity_evidence,
)

NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
COMMIT = "a" * 40
IDENTITIES = {
    "api_version": "2.0.0",
    "schema_version": "2026-07-13.1",
    "action_schema_version": "2.1.0",
    "legal_action_ordering_version": "2.0.0",
    "observation_schema_version": "5.0.0",
    "reward_schema_version": "2.0.0",
    "bridge_version": "0.8.0",
    "rl_version": "0.2.0",
    "headless_commit": "b" * 40,
    "headless_tree": "c" * 40,
}


def parity() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "gate": "live-headless-parity",
        "status": "passed",
        "source_commit": COMMIT,
        "recorded_at_utc": "2026-07-12T11:00:00Z",
        "identities": dict(IDENTITIES),
        "scenarios": {
            "count": 12,
            "legal_action_identity_match": True,
            "legal_action_order_match": True,
            "transition_fact_match": True,
            "external_reward_match": True,
        },
    }


def compatibility() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "gate": "compatibility-window",
        "status": "passed",
        "source_commit": COMMIT,
        "recorded_at_utc": "2026-07-12T11:00:00Z",
        "compatibility_releases": [
            {"version": "0.7.0", "commit": "d" * 40, "released_at_utc": "2026-06-01T00:00:00Z"},
            {"version": "0.7.1", "commit": "e" * 40, "released_at_utc": "2026-06-20T00:00:00Z"},
        ],
        "v1_usage": {
            "active_clients": 0,
            "observation_window_days": 21,
            "ended_at_utc": "2026-07-12T10:00:00Z",
        },
    }


class ExternalEvidenceTests(unittest.TestCase):
    def test_valid_evidence_passes(self) -> None:
        validate_parity_evidence(parity(), source_commit=COMMIT, identities=IDENTITIES, now=NOW)
        validate_compatibility_evidence(compatibility(), source_commit=COMMIT, now=NOW)

    def test_parity_identity_or_assertion_drift_fails_closed(self) -> None:
        wrong_identity = parity()
        assert isinstance(wrong_identity["identities"], dict)
        wrong_identity["identities"]["reward_schema_version"] = "legacy"
        with self.assertRaisesRegex(EvidenceError, "reward_schema_version"):
            validate_parity_evidence(
                wrong_identity,
                source_commit=COMMIT,
                identities=IDENTITIES,
                now=NOW,
            )

        failed_assertion = parity()
        assert isinstance(failed_assertion["scenarios"], dict)
        failed_assertion["scenarios"]["legal_action_order_match"] = False
        with self.assertRaisesRegex(EvidenceError, "legal_action_order_match"):
            validate_parity_evidence(
                failed_assertion,
                source_commit=COMMIT,
                identities=IDENTITIES,
                now=NOW,
            )

    def test_compatibility_requires_two_releases_and_zero_v1_clients(self) -> None:
        one_release = compatibility()
        assert isinstance(one_release["compatibility_releases"], list)
        one_release["compatibility_releases"] = one_release["compatibility_releases"][:1]
        with self.assertRaisesRegex(EvidenceError, "at least two"):
            validate_compatibility_evidence(one_release, source_commit=COMMIT, now=NOW)

        active = copy.deepcopy(compatibility())
        assert isinstance(active["v1_usage"], dict)
        active["v1_usage"]["active_clients"] = 1
        with self.assertRaisesRegex(EvidenceError, "zero active"):
            validate_compatibility_evidence(active, source_commit=COMMIT, now=NOW)

    def test_unknown_fields_stale_timestamp_and_wrong_commit_are_rejected(self) -> None:
        unknown = parity()
        unknown["notes"] = "unchecked"
        with self.assertRaisesRegex(EvidenceError, "fields mismatch"):
            validate_parity_evidence(unknown, source_commit=COMMIT, identities=IDENTITIES, now=NOW)

        stale = parity()
        stale["recorded_at_utc"] = "2026-01-01T00:00:00Z"
        with self.assertRaisesRegex(EvidenceError, "older than"):
            validate_parity_evidence(stale, source_commit=COMMIT, identities=IDENTITIES, now=NOW)

        wrong_commit = compatibility()
        wrong_commit["source_commit"] = "f" * 40
        with self.assertRaisesRegex(EvidenceError, "source_commit"):
            validate_compatibility_evidence(wrong_commit, source_commit=COMMIT, now=NOW)


if __name__ == "__main__":
    unittest.main()
