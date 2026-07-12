"""Record player-visible draft/deck transitions from bridge revision events."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import requests

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.artifacts.path_policy import (  # noqa: E402
    resolve_artifact_path,
    resolve_external_input_path,
)

SESSION_SCHEMA_VERSION = "2026-07-11.1"
API_VERSION = "2.0.0"
ACTION_SCHEMA_VERSION = "2.0.0"
LEGAL_ACTION_ORDERING_VERSION = "2.0.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_session_file() -> Path:
    override = os.getenv("STS2_BRIDGE_SESSION_FILE")
    if override:
        return Path(override).expanduser()
    appdata = os.getenv("APPDATA")
    if not appdata:
        raise RuntimeError("set STS2_BRIDGE_SESSION_FILE when APPDATA is unavailable")
    return Path(appdata) / "SlayTheSpire2" / "bridge" / "session.json"


def default_output_root() -> Path:
    return resolve_artifact_path(None, default="draft-events")


@dataclass(frozen=True)
class Session:
    base_url: str
    token: str
    session_id: str

    @classmethod
    def load(cls, path: Path) -> "Session":
        path = resolve_external_input_path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("bridge session descriptor must be a JSON object")

        base_url = str(payload.get("base_url") or "").rstrip("/")
        parsed = urlsplit(base_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise RuntimeError("bridge session base_url must use a valid loopback HTTP port") from exc
        if (
            parsed.scheme != "http"
            or (parsed.hostname or "").casefold() not in {"127.0.0.1", "localhost", "::1"}
            or port is None
            or not 1 <= port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError("bridge session base_url must be loopback HTTP")

        capabilities = payload.get("capabilities")
        api_versions = payload.get("api_versions")
        tokens = payload.get("capability_tokens")
        if not isinstance(capabilities, list) or "player-control" not in capabilities:
            raise RuntimeError("bridge session does not advertise player-control")
        if not isinstance(api_versions, list) or API_VERSION not in api_versions:
            raise RuntimeError(f"bridge session does not advertise API {API_VERSION}")
        if payload.get("schema_version") != SESSION_SCHEMA_VERSION:
            raise RuntimeError("bridge session schema_version is unsupported")
        if payload.get("action_schema_version") != ACTION_SCHEMA_VERSION:
            raise RuntimeError("bridge action_schema_version is unsupported")
        if payload.get("legal_action_ordering_version") != LEGAL_ACTION_ORDERING_VERSION:
            raise RuntimeError("bridge legal_action_ordering_version is unsupported")
        if not isinstance(tokens, dict):
            raise RuntimeError("bridge session capability_tokens is missing")
        token = tokens.get("player-control")
        if not isinstance(token, str) or len(token) < 32:
            raise RuntimeError("bridge session has no valid player-control credential")
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or len(session_id) < 16:
            raise RuntimeError("bridge session_id is malformed")
        return cls(base_url, token, session_id)


def cards_from_state(payload: dict[str, Any]) -> list[str]:
    state = payload.get("state", payload)
    run = state.get("run") if isinstance(state, dict) else None
    player = state.get("player") if isinstance(state, dict) else None
    if not isinstance(player, dict):
        player = run.get("player") if isinstance(run, dict) else None
    deck = player.get("deck") if isinstance(player, dict) else None
    if isinstance(deck, dict):
        deck = deck.get("cards")
    if not isinstance(deck, list):
        deck = state.get("deck") if isinstance(state, dict) else None
    result: list[str] = []
    for card in deck if isinstance(deck, list) else []:
        if isinstance(card, str):
            result.append(card)
        elif isinstance(card, dict):
            result.append(str(card.get("id") or card.get("card_id") or card.get("name") or "unknown"))
    return result


def multiset_added(before: Iterable[str], after: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    for value in before:
        counts[value] = counts.get(value, 0) + 1
    added: list[str] = []
    for value in after:
        if counts.get(value, 0):
            counts[value] -= 1
        else:
            added.append(value)
    return added


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def fetch_state(client: requests.Session, session: Session) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {session.token}"}
    response = client.get(session.base_url + "/v2/state", headers=headers, timeout=(2, 10))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("bridge v2 state response must be an object")
    if payload.get("session_id") != session.session_id:
        raise RuntimeError("bridge v2 state session identity mismatch")
    if payload.get("visibility") != "player":
        raise RuntimeError("bridge v2 state visibility is not player")
    if not isinstance(payload.get("state_version"), int) or int(payload["state_version"]) < 0:
        raise RuntimeError("bridge v2 state revision is malformed")
    if not isinstance(payload.get("state"), dict) or not isinstance(payload.get("legal_actions"), list):
        raise RuntimeError("bridge v2 state payload is malformed")
    return payload


def revision_stream(client: requests.Session, session: Session) -> Iterable[str]:
    headers = {
        "Authorization": f"Bearer {session.token}",
        "Accept": "text/event-stream",
    }
    response = client.get(
        session.base_url + "/v2/events",
        headers=headers,
        stream=True,
        timeout=(2, 90),
    )
    response.raise_for_status()
    with response:
        for raw in response.iter_lines(decode_unicode=True):
            if raw and raw.startswith("data:"):
                yield raw[5:].strip()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-file", default=str(default_session_file()))
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory below STS2_ARTIFACT_ROOT (default: draft-events)",
    )
    parser.add_argument("--retry-seconds", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    session_file = resolve_external_input_path(args.session_file)
    output_root = resolve_artifact_path(args.output_root, default="draft-events")

    previous: list[str] = []
    output = output_root / f"draft-events-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    while True:
        try:
            session = Session.load(session_file)
            with requests.Session() as client:
                initial = fetch_state(client, session)
                previous = cards_from_state(initial)
                for event in revision_stream(client, session):
                    state = fetch_state(client, session)
                    current = cards_from_state(state)
                    added = multiset_added(previous, current)
                    removed = multiset_added(current, previous)
                    if added or removed:
                        append_jsonl(
                            output,
                            {
                                "schema_version": "draft-event-v2",
                                "captured_at_utc": utc_now(),
                                "session_id": session.session_id,
                                "state_version": state.get("state_version"),
                                "event": event[:2048],
                                "cards_added": added,
                                "cards_removed": removed,
                            },
                        )
                    previous = current
        except KeyboardInterrupt:
            return
        except (OSError, ValueError, requests.RequestException, RuntimeError) as exc:
            print(f"draft recorder reconnecting after error: {type(exc).__name__}: {exc}")
            time.sleep(max(0.25, args.retry_seconds))


if __name__ == "__main__":
    main()
