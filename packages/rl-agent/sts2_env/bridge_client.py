"""HTTP client for the STS2 bridge mod's RL environment endpoints."""

import json
import os
import time
from pathlib import Path
from typing import Any

import requests


class BridgeError(Exception):
    """Raised when a bridge request fails."""

    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def _default_session_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "SlayTheSpire2" / "bridge" / "session.json"
    # Fallback for non-standard Windows setups
    userprofile = os.environ.get("USERPROFILE", "")
    return Path(userprofile) / "AppData" / "Roaming" / "SlayTheSpire2" / "bridge" / "session.json"


def _resolve_session_path() -> Path:
    override = os.environ.get("STS2_BRIDGE_SESSION_FILE")
    if override:
        return Path(override)
    return _default_session_path()


class BridgeClient:
    """Client for the STS2 bridge mod's RL environment HTTP API.

    Reads session.json to discover the bridge URL and auth token.
    All requests use Bearer token authentication and retry on transient errors.
    """

    MAX_RETRIES = 3
    RETRY_DELAY_S = 1.0
    HTTP_TIMEOUT_GRACE_MS = 10_000

    def __init__(self, session_path: str | Path | None = None):
        path = Path(session_path) if session_path else _resolve_session_path()
        self._session_path = path
        self._base_url: str = ""
        self._token: str = ""
        self._is_connected: bool = False
        self._load_session()

    def _load_session(self) -> None:
        """Read session.json and extract base_url and token."""
        try:
            data = json.loads(self._session_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise BridgeError(
                f"Session file not found: {self._session_path}. "
                "Is the STS2 bridge mod running?"
            )
        except json.JSONDecodeError as exc:
            raise BridgeError(f"Invalid JSON in session file: {exc}")

        if "base_url" not in data or "token" not in data:
            raise BridgeError(
                "Session file missing required fields (base_url, token)."
            )

        self._base_url = data["base_url"].rstrip("/")
        self._token = data["token"]

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        body: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> Any:
        """Send an HTTP request to the bridge with retry logic.

        Args:
            method: HTTP method (GET or POST).
            endpoint: Path relative to base_url (e.g. "env/spec").
            body: JSON body for POST requests.
            timeout_ms: Game-side timeout included in the body. The HTTP
                timeout is this value plus a grace period.

        Returns:
            Parsed JSON response body.

        Raises:
            BridgeError: On non-retryable HTTP errors or after retries exhausted.
        """
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._token}"}
        http_timeout_s = (
            (timeout_ms + self.HTTP_TIMEOUT_GRACE_MS) / 1000.0
            if timeout_ms is not None
            else 30.0
        )

        last_exc: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = requests.request(
                    method,
                    url,
                    headers=headers,
                    json=body if body is not None else None,
                    timeout=http_timeout_s,
                )

                if resp.status_code >= 400:
                    try:
                        resp_body = resp.json()
                    except Exception:
                        resp_body = resp.text
                    raise BridgeError(
                        f"Bridge returned HTTP {resp.status_code} for {method} /{endpoint}: "
                        f"{resp_body}",
                        status_code=resp.status_code,
                        response_body=resp_body,
                    )

                self._is_connected = True
                return resp.json()

            except BridgeError:
                # Don't retry application-level errors from the bridge
                raise
            except requests.ConnectionError as exc:
                self._is_connected = False
                last_exc = exc
            except requests.Timeout as exc:
                self._is_connected = False
                last_exc = exc

            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY_S)

        raise BridgeError(
            f"Bridge unreachable after {self.MAX_RETRIES} attempts "
            f"({method} /{endpoint}): {last_exc}"
        )

    # -- Public API --

    def get_spec(self) -> dict[str, Any]:
        """GET /env/spec -- returns environment specification."""
        return self._request("GET", "env/spec")

    def reset(
        self,
        character: str | None = None,
        defensive_buffs: bool = False,
        timeout_ms: int = 45_000,
    ) -> dict[str, Any]:
        """POST /env/reset -- start a new episode.

        Returns dict with keys:
            ok, episode_id, step_index, done, truncated, obs, legal_actions, info
        """
        body: dict[str, Any] = {"timeout_ms": timeout_ms}
        if character is not None:
            body["character"] = character
        if defensive_buffs:
            body["defensive_buffs"] = True
        return self._request("POST", "env/reset", body=body, timeout_ms=timeout_ms)

    def step(
        self,
        episode_id: str,
        action_index: int | None = None,
        action_id: str | None = None,
        timeout_ms: int = 20_000,
    ) -> dict[str, Any]:
        """POST /env/step -- execute an action in the current episode.

        Provide either action_index or action_id (not both).

        Returns dict with keys:
            ok, episode_id, step_index, reward, done, truncated, obs, legal_actions, info
        """
        if action_index is None and action_id is None:
            raise ValueError("Must provide either action_index or action_id.")

        body: dict[str, Any] = {
            "episode_id": episode_id,
            "timeout_ms": timeout_ms,
        }
        if action_index is not None:
            body["action_index"] = action_index
        if action_id is not None:
            body["action_id"] = action_id

        return self._request("POST", "env/step", body=body, timeout_ms=timeout_ms)

    def health(self) -> dict[str, Any]:
        """GET /health -- health check. Updates is_connected."""
        return self._request("GET", "health")

    def get_state(self) -> dict[str, Any]:
        """GET /state -- full game state."""
        return self._request("GET", "state")
