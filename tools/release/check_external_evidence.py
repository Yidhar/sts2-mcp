"""Validate external promotion evidence required by the protected release job.

The live/headless parity run and the two-release compatibility/telemetry window
cannot be manufactured by source CI.  This checker makes those external gates
machine-verifiable instead of reducing them to unchecked release notes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SCHEMA_VERSION = "1.0.0"


class EvidenceError(ValueError):
    """External evidence is missing, stale, or incompatible."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise EvidenceError(f"{label} fields mismatch; missing={missing}, unknown={unknown}")


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{label} must be a non-empty RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError(f"{label} is not a valid RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise EvidenceError(f"{label} must contain a timezone")
    return parsed.astimezone(timezone.utc)


def _load(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"{label} file is missing: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{label} is not valid JSON: {exc}") from exc


def expected_identities(root: Path = ROOT) -> dict[str, str]:
    contract = json.loads((root / "contracts" / "manifest.json").read_text(encoding="utf-8"))
    release = json.loads((root / "release-manifest.json").read_text(encoding="utf-8"))
    headless = json.loads((root / "third_party" / "sts2-ai.lock.json").read_text(encoding="utf-8"))
    return {
        "api_version": str(contract["api_version"]),
        "schema_version": str(contract["schema_version"]),
        "action_schema_version": str(contract["action_schema_version"]),
        "legal_action_ordering_version": str(contract["legal_action_ordering_version"]),
        "observation_schema_version": str(contract["observation_schema_version"]),
        "reward_schema_version": str(contract["reward_schema_version"]),
        "bridge_version": str(release["components"]["bridge"]),
        "rl_version": str(release["components"]["rl_trainer"]),
        "headless_commit": str(headless["commit"]),
        "headless_tree": str(headless["tree"]),
    }


def _validate_common(
    payload: dict[str, Any],
    *,
    gate: str,
    source_commit: str,
    now: datetime,
    max_age: timedelta,
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError(f"{gate} evidence schema_version is unsupported")
    if payload.get("gate") != gate or payload.get("status") != "passed":
        raise EvidenceError(f"{gate} evidence must identify the gate and have status=passed")
    if payload.get("source_commit") != source_commit:
        raise EvidenceError(f"{gate} evidence source_commit does not match the release commit")
    recorded = _timestamp(payload.get("recorded_at_utc"), f"{gate}.recorded_at_utc")
    if recorded > now + timedelta(minutes=5):
        raise EvidenceError(f"{gate} evidence timestamp is in the future")
    if now - recorded > max_age:
        raise EvidenceError(f"{gate} evidence is older than {max_age.days} days")


def validate_parity_evidence(
    payload: dict[str, Any],
    *,
    source_commit: str,
    identities: dict[str, str],
    now: datetime,
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "gate",
            "status",
            "source_commit",
            "recorded_at_utc",
            "identities",
            "scenarios",
        },
        "live-headless-parity",
    )
    _validate_common(
        payload,
        gate="live-headless-parity",
        source_commit=source_commit,
        now=now,
        max_age=timedelta(days=30),
    )
    observed = _object(payload["identities"], "live-headless-parity.identities")
    _exact_keys(observed, set(identities), "live-headless-parity.identities")
    for key, expected in identities.items():
        if observed.get(key) != expected:
            raise EvidenceError(f"live-headless-parity identity {key} does not match {expected}")

    scenarios = _object(payload["scenarios"], "live-headless-parity.scenarios")
    _exact_keys(
        scenarios,
        {
            "count",
            "legal_action_identity_match",
            "legal_action_order_match",
            "transition_fact_match",
            "external_reward_match",
        },
        "live-headless-parity.scenarios",
    )
    count = scenarios.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise EvidenceError("live-headless-parity must cover at least one scenario")
    for key in (
        "legal_action_identity_match",
        "legal_action_order_match",
        "transition_fact_match",
        "external_reward_match",
    ):
        if scenarios.get(key) is not True:
            raise EvidenceError(f"live-headless-parity scenario assertion failed: {key}")


def validate_compatibility_evidence(
    payload: dict[str, Any],
    *,
    source_commit: str,
    now: datetime,
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "gate",
            "status",
            "source_commit",
            "recorded_at_utc",
            "compatibility_releases",
            "v1_usage",
        },
        "compatibility-window",
    )
    _validate_common(
        payload,
        gate="compatibility-window",
        source_commit=source_commit,
        now=now,
        max_age=timedelta(days=45),
    )
    releases = payload["compatibility_releases"]
    if not isinstance(releases, list) or len(releases) < 2:
        raise EvidenceError("compatibility-window requires at least two compatibility releases")
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(releases):
        release = _object(raw, f"compatibility_releases[{index}]")
        _exact_keys(release, {"version", "commit", "released_at_utc"}, f"compatibility_releases[{index}]")
        version = release.get("version")
        commit = release.get("commit")
        if not isinstance(version, str) or not version.strip():
            raise EvidenceError(f"compatibility_releases[{index}].version is required")
        if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
            raise EvidenceError(f"compatibility_releases[{index}].commit must be a 40-character Git ID")
        _timestamp(release.get("released_at_utc"), f"compatibility_releases[{index}].released_at_utc")
        identities.add((version, commit))
    if len(identities) < 2:
        raise EvidenceError("compatibility releases must have two distinct version/commit identities")

    usage = _object(payload["v1_usage"], "compatibility-window.v1_usage")
    _exact_keys(usage, {"active_clients", "observation_window_days", "ended_at_utc"}, "v1_usage")
    if usage.get("active_clients") != 0:
        raise EvidenceError("compatibility-window requires zero active v1 clients")
    days = usage.get("observation_window_days")
    if isinstance(days, bool) or not isinstance(days, int) or days < 14:
        raise EvidenceError("v1 usage observation window must cover at least 14 days")
    ended = _timestamp(usage.get("ended_at_utc"), "v1_usage.ended_at_utc")
    if ended > now + timedelta(minutes=5):
        raise EvidenceError("v1 usage observation window ends in the future")


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument("--require-source-commit", default=None)
    args = parser.parse_args()
    source_commit = str(args.require_source_commit or current_commit()).lower()
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise SystemExit("required source commit must be a 40-character lowercase Git ID")
    now = datetime.now(timezone.utc)
    try:
        validate_parity_evidence(
            _load(args.parity.resolve(), "live-headless-parity"),
            source_commit=source_commit,
            identities=expected_identities(),
            now=now,
        )
        validate_compatibility_evidence(
            _load(args.compatibility.resolve(), "compatibility-window"),
            source_commit=source_commit,
            now=now,
        )
    except EvidenceError as exc:
        print(f"release evidence rejected: {exc}")
        return 1
    print("external release evidence ok: parity and compatibility gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
