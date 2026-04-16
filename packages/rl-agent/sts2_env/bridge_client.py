"""HTTP client for the STS2 bridge mod's RL environment endpoints."""

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests

from .path_utils import default_bridge_session_dir, normalize_path, running_in_wsl


class BridgeError(Exception):
    """Raised when a bridge request fails."""

    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def _default_session_path() -> Path:
    return default_bridge_session_dir() / "session.json"


def _resolve_session_path() -> Path:
    override = os.environ.get("STS2_BRIDGE_SESSION_FILE")
    if override:
        normalized = normalize_path(override)
        assert normalized is not None
        return normalized
    return _default_session_path()


def _rewrite_loopback_url_for_wsl(base_url: str) -> str:
    """Optionally rewrite a Windows loopback bridge URL into a WSL-reachable URL.

    This is only applied when running inside WSL and the caller explicitly
    provides a host override such as the Windows host gateway IP
    (for example ``172.28.x.1``).
    """
    if not running_in_wsl():
        return base_url

    parsed = urlsplit(base_url)
    host = (parsed.hostname or "").strip().lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return base_url

    wsl_host = os.environ.get("STS2_BRIDGE_WSL_HOST")
    if not wsl_host:
        return base_url

    override_port = os.environ.get("STS2_BRIDGE_WSL_PORT")
    port = int(override_port) if override_port else (parsed.port or 80)
    netloc = f"{wsl_host}:{port}"
    rewritten = SplitResult(
        scheme=parsed.scheme or "http",
        netloc=netloc,
        path=parsed.path or "",
        query=parsed.query,
        fragment=parsed.fragment,
    )
    return urlunsplit(rewritten)


def _resolve_base_url(session_base_url: str, *, allow_env_override: bool = True) -> str:
    explicit_override = os.environ.get("STS2_BRIDGE_BASE_URL")
    if explicit_override and allow_env_override:
        return explicit_override.rstrip("/")

    return _rewrite_loopback_url_for_wsl(session_base_url.rstrip("/"))


class BridgeClient:
    """Client for the STS2 bridge mod's RL environment HTTP API.

    Reads session.json to discover the bridge URL and auth token.
    All requests use Bearer token authentication and retry on transient errors.
    """

    MAX_RETRIES = 3
    RETRY_DELAY_S = 1.0
    HTTP_TIMEOUT_GRACE_MS = 10_000

    def __init__(self, session_path: str | Path | None = None):
        normalized = normalize_path(session_path) if session_path else None
        path = normalized if normalized is not None else _resolve_session_path()
        self._allow_env_base_url_override = normalized is None
        self._session_path = path
        self._base_url: str = ""
        self._session_base_url: str = ""
        self._token: str = ""
        self._is_connected: bool = False
        self._session = requests.Session()
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

        self._session_base_url = str(data["base_url"]).rstrip("/")
        self._base_url = _resolve_base_url(
            self._session_base_url,
            allow_env_override=self._allow_env_base_url_override,
        )
        self._token = data["token"]
        self._session.headers.update({"Authorization": f"Bearer {self._token}"})

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def session_base_url(self) -> str:
        return self._session_base_url

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
        http_timeout_s = (
            (timeout_ms + self.HTTP_TIMEOUT_GRACE_MS) / 1000.0
            if timeout_ms is not None
            else 30.0
        )

        last_exc: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = self._session.request(
                    method,
                    url,
                    json=body if body is not None else None,
                    timeout=http_timeout_s,
                )

                if resp.status_code >= 400:
                    try:
                        resp_body = resp.json()
                    except Exception:
                        resp_body = resp.text
                    if (
                        resp.status_code == 401
                        and isinstance(resp_body, dict)
                        and str(resp_body.get("error") or "").strip() == "missing_or_invalid_token"
                        and attempt < self.MAX_RETRIES
                    ):
                        self._load_session()
                        time.sleep(self.RETRY_DELAY_S)
                        continue
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
            f"({method} /{endpoint}) via {self._base_url}: {last_exc}"
            + (
                " (WSL note: if session.json still points at http://127.0.0.1:<port>/, "
                "start the Windows WSL relay and export STS2_BRIDGE_BASE_URL=http://<windows-host-ip>:<relay-port>/)"
                if running_in_wsl() and self._session_base_url.startswith("http://127.0.0.1")
                else ""
            )
        )

    # -- Public API --

    def get_spec(self) -> dict[str, Any]:
        """GET /env/spec -- returns environment specification."""
        return self._request("GET", "env/spec")

    def reset(
        self,
        character: str | None = None,
        rebind_active_run: bool = False,
        force_fresh: bool = False,
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
        if rebind_active_run:
            body["rebind_active_run"] = True
        if force_fresh:
            body["force_fresh"] = True
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

    def combat_reset(
        self,
        *,
        character: str | None = None,
        encounter_id: str | None = None,
        seed: int | None = None,
        current_hp: int | None = None,
        max_hp: int | None = None,
        max_energy: int | None = None,
        deck: list[str] | None = None,
        deck_entries: list[dict[str, Any]] | None = None,
        relics: list[str] | None = None,
        potions: list[str] | None = None,
        gold: int | None = None,
        timeout_ms: int = 15_000,
    ) -> dict[str, Any]:
        """POST /env/combat_reset -- start a combat sandbox episode.

        Returns dict with keys:
            ok, episode_id, step_index, done, truncated, obs, legal_actions, info
        """
        body: dict[str, Any] = {"timeout_ms": timeout_ms}
        if character is not None:
            body["character"] = character
        if encounter_id is not None:
            body["encounter_id"] = encounter_id
        if seed is not None:
            body["seed"] = seed
        if current_hp is not None:
            body["current_hp"] = current_hp
        if max_hp is not None:
            body["max_hp"] = max_hp
        if max_energy is not None:
            body["max_energy"] = max_energy
        if deck is not None:
            body["deck"] = deck
        if deck_entries is not None:
            body["deck_entries"] = deck_entries
        if relics is not None:
            body["relics"] = relics
        if potions is not None:
            body["potions"] = potions
        if gold is not None:
            body["gold"] = gold
        return self._request("POST", "env/combat_reset", body=body, timeout_ms=timeout_ms)

    def combat_catalog(self) -> dict[str, Any]:
        """GET /env/combat_catalog -- list available combat encounters."""
        return self._request("GET", "env/combat_catalog")

    def health(self) -> dict[str, Any]:
        """GET /health -- health check. Updates is_connected."""
        return self._request("GET", "health")

    def export_static(
        self,
        *,
        output_dir: str | None = None,
        timeout_ms: int = 60_000,
    ) -> dict[str, Any]:
        """POST /static/export -- dump static game metadata to disk.

        Returns dict with keys like:
            ok, output_dir, items_path, manifest_path, counts
        """
        body: dict[str, Any] = {"timeout_ms": timeout_ms}
        if output_dir is not None:
            body["output_dir"] = output_dir
        return self._request("POST", "static/export", body=body, timeout_ms=timeout_ms)

    def get_state(self) -> dict[str, Any]:
        """GET /state -- full game state."""
        return self._request("GET", "state")
