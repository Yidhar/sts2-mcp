from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RECORD_PATH = Path(__file__).resolve().parents[3] / "tools" / "draft_recorder" / "record.py"
SPEC = importlib.util.spec_from_file_location("draft_recorder_record", RECORD_PATH)
assert SPEC is not None and SPEC.loader is not None
record = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = record
SPEC.loader.exec_module(record)


def session_payload(base_url: str = "http://127.0.0.1:27100/") -> dict[str, object]:
    return {
        "session_id": "example-session-0001",
        "base_url": base_url,
        "capability_tokens": {"player-control": "p" * 32},
        "api_versions": ["2.0.0"],
        "schema_version": "2026-07-13.1",
        "action_schema_version": "2.1.0",
        "legal_action_ordering_version": "2.0.0",
        "capabilities": ["player-control"],
    }


def test_cards_from_state_handles_nested_and_fallback_decks() -> None:
    nested = {
        "state": {
            "run": {
                "player": {
                    "deck": {
                        "cards": [
                            "CARD.STRIKE_R",
                            {"id": "CARD.DEFEND_R"},
                            {"card_id": "CARD.BASH"},
                            {"name": "CARD.UNKNOWN_NAME"},
                            {},
                        ]
                    }
                }
            }
        }
    }
    assert record.cards_from_state(nested) == [
        "CARD.STRIKE_R",
        "CARD.DEFEND_R",
        "CARD.BASH",
        "CARD.UNKNOWN_NAME",
        "unknown",
    ]
    assert record.cards_from_state({"deck": ["CARD.A", {"id": "CARD.B"}]}) == [
        "CARD.A",
        "CARD.B",
    ]
    assert record.cards_from_state({"state": {"run": None}}) == []


def test_multiset_added_preserves_after_order_and_duplicate_counts() -> None:
    before = ["A", "A", "B", "D"]
    after = ["A", "B", "A", "A", "C", "C"]
    assert record.multiset_added(before, after) == ["A", "C", "C"]
    assert record.multiset_added(after, before) == ["D"]
    assert record.multiset_added([], ["X", "X"]) == ["X", "X"]


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:27100/",
        "http://localhost:27100",
        "http://[::1]:27100/",
    ],
)
def test_session_load_accepts_only_loopback_http(tmp_path: Path, base_url: str) -> None:
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session_payload(base_url)), encoding="utf-8")
    session = record.Session.load(path)
    assert session.base_url == base_url.rstrip("/")
    assert session.token == "p" * 32
    assert session.session_id == "example-session-0001"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://0.0.0.0:27100",
        "http://192.168.1.10:27100",
        "https://127.0.0.1:27100",
        "http://localhost.example:27100",
    ],
)
def test_session_load_rejects_non_loopback_or_non_http(tmp_path: Path, base_url: str) -> None:
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session_payload(base_url)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="loopback HTTP"):
        record.Session.load(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("capability_tokens", {"legacy-privileged": "l" * 32}, "player-control credential"),
        ("capabilities", ["training"], "does not advertise player-control"),
        ("api_versions", ["legacy-v1"], "does not advertise API"),
        ("schema_version", "old", "schema_version"),
        ("action_schema_version", "old", "action_schema_version"),
        ("legal_action_ordering_version", "old", "legal_action_ordering_version"),
    ],
)
def test_session_load_rejects_legacy_or_incompatible_identity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = session_payload()
    payload[field] = value
    path = tmp_path / "session.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        record.Session.load(path)


def test_default_output_root_is_confined_to_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(artifact_root))
    assert record.default_output_root() == artifact_root / "draft-events"
    with pytest.raises(ValueError, match="below"):
        record.resolve_artifact_path(str(tmp_path / "outside"), default="draft-events")


class _Response:
    status_code = 200

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _Client:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def get(self, url: str, **_kwargs: object) -> _Response:
        self.urls.append(url)
        return _Response(self.payload)


def test_fetch_state_is_v2_only_and_checks_session_identity() -> None:
    session = record.Session("http://127.0.0.1:27100", "p" * 32, "example-session-0001")
    payload = {
        "session_id": session.session_id,
        "state_version": 3,
        "visibility": "player",
        "state": {},
        "legal_actions": [],
    }
    client = _Client(payload)
    assert record.fetch_state(client, session) == payload
    assert client.urls == ["http://127.0.0.1:27100/v2/state"]

    mismatched = dict(payload, session_id="different-session")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        record.fetch_state(_Client(mismatched), session)
