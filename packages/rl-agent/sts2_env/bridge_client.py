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

    # 2026-04-27: bridge-outage recovery loop. When the burst-retry path
    # (MAX_RETRIES quick attempts) exhausts on ConnectionError/Timeout, the
    # game is likely fully crashed and the watchdog is restarting it. The
    # watchdog needs ~30-60s to relaunch + emit a fresh session.json, so the
    # 3x1s burst is far too short to ride that out. Poll for session-file
    # rotation + retry the request every BRIDGE_OUTAGE_POLL_S until the
    # total outage exceeds MAX_BRIDGE_OUTAGE_S, only then give up.
    MAX_BRIDGE_OUTAGE_S = 180.0
    BRIDGE_OUTAGE_POLL_S = 5.0

    def __init__(self, session_path: str | Path | None = None):
        normalized = normalize_path(session_path) if session_path else None
        path = normalized if normalized is not None else _resolve_session_path()
        self._allow_env_base_url_override = normalized is None
        self._session_path = path
        self._base_url: str = ""
        self._session_base_url: str = ""
        self._token: str = ""
        # Session-file freshness tracking so we can hot-reload after launcher
        # kills and restarts the bridge mod (new PID → new token → new
        # session_N.json written in place).
        self._session_mtime: float = 0.0
        self._is_connected: bool = False
        self._session = requests.Session()
        self._load_session()

    def _load_session(self) -> None:
        """Read session.json and extract base_url and token."""
        try:
            raw = self._session_path.read_text(encoding="utf-8")
            data = json.loads(raw)
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
        try:
            self._session_mtime = self._session_path.stat().st_mtime
        except OSError:
            self._session_mtime = 0.0

    def _session_file_changed_on_disk(self) -> bool:
        """Return True if session_N.json has been re-written since last load
        (e.g., launcher killed and restarted this instance and wrote a fresh
        token/port). Used to trigger a hot reload in the retry loop.
        """
        try:
            current_mtime = self._session_path.stat().st_mtime
        except OSError:
            return False
        # Treat any mtime forward-movement as a change. A new session file
        # written by the launcher after a kill+restart always has a later
        # mtime than the one we originally loaded.
        return current_mtime > self._session_mtime

    def _maybe_rebind_session(self, reason: str) -> bool:
        """If the session file has changed on disk, reload it and return True.

        Silently return False when nothing changed or the reload itself
        fails (we want the caller's retry/backoff to carry on rather than
        fatally raising in the middle of a transient network blip).
        """
        if not self._session_file_changed_on_disk():
            return False
        old_base_url = self._base_url
        old_token_prefix = self._token[:6] if self._token else ""
        try:
            self._load_session()
        except BridgeError:
            return False
        new_token_prefix = self._token[:6] if self._token else ""
        # Keep this print — it's the only operator-visible signal that the
        # client auto-rebound after a launcher-side restart. Without it the
        # restart is silent and very hard to correlate with watchdog events.
        print(
            f"[bridge-client] rebound session ({reason}): "
            f"{old_base_url} token={old_token_prefix}... -> "
            f"{self._base_url} token={new_token_prefix}..."
        )
        return True

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
        http_timeout_s = (
            (timeout_ms + self.HTTP_TIMEOUT_GRACE_MS) / 1000.0
            if timeout_ms is not None
            else 30.0
        )

        last_exc: Exception | None = None
        quick_attempts_left = self.MAX_RETRIES
        outage_deadline: float | None = None
        outage_announced = False
        total_attempts = 0

        while True:
            total_attempts += 1
            # Recompute per attempt so a rebind during the retry loop picks
            # up the new base_url. Token lives in self._session.headers
            # which _load_session() updated.
            url = f"{self._base_url}/{endpoint.lstrip('/')}"
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
                    if resp.status_code == 401:
                        # ANY 401 may indicate a stale token after a
                        # launcher/watchdog-side game restart that rotated
                        # the bridge token.  Try a fresh session reload first,
                        # fall back to forced reload on the canonical
                        # "missing_or_invalid_token" error code even if the
                        # session file mtime did not advance.  This handler
                        # MUST run during both the burst-retry phase AND the
                        # outage-recovery polling loop — if we only rebound
                        # during burst-retry, then a watchdog restart that
                        # happens after burst-retry is exhausted would leave
                        # us holding the stale token and dying on the next
                        # outage-recovery probe.
                        rebound = self._maybe_rebind_session(
                            reason=f"401 on {method} /{endpoint}",
                        )
                        is_token_error = (
                            isinstance(resp_body, dict)
                            and str(resp_body.get("error") or "").strip()
                            == "missing_or_invalid_token"
                        )
                        if not rebound and is_token_error:
                            # Legacy narrow path: re-read even without mtime
                            # change to preserve prior behavior on the one
                            # error code we know means "token rejected".
                            self._load_session()
                            rebound = True
                        if rebound:
                            if quick_attempts_left > 1:
                                quick_attempts_left -= 1
                                time.sleep(self.RETRY_DELAY_S)
                                continue
                            # Burst-retry exhausted.  If we are inside the
                            # outage-recovery polling window, treat the 401 as
                            # a continuation of the outage (bridge alive but
                            # token-rotated mid-restart) and keep polling.
                            if (
                                outage_deadline is not None
                                and time.monotonic() < outage_deadline
                            ):
                                time.sleep(self.BRIDGE_OUTAGE_POLL_S)
                                continue
                    raise BridgeError(
                        f"Bridge returned HTTP {resp.status_code} for {method} /{endpoint}: "
                        f"{resp_body}",
                        status_code=resp.status_code,
                        response_body=resp_body,
                    )

                self._is_connected = True
                if outage_announced:
                    print(
                        f"[bridge-client] outage recovered for {method} /{endpoint} "
                        f"after {total_attempts} attempts",
                        flush=True,
                    )
                return resp.json()

            except BridgeError:
                # Don't retry application-level errors from the bridge
                raise
            except requests.ConnectionError as exc:
                self._is_connected = False
                last_exc = exc
                self._maybe_rebind_session(
                    reason=f"ConnectionError on {method} /{endpoint}",
                )
            except requests.Timeout as exc:
                self._is_connected = False
                last_exc = exc
                self._maybe_rebind_session(
                    reason=f"Timeout on {method} /{endpoint}",
                )

            # Burst-retry phase: 3 attempts at 1s spacing for transient blips.
            if quick_attempts_left > 1:
                quick_attempts_left -= 1
                time.sleep(self.RETRY_DELAY_S)
                continue
            quick_attempts_left = 0

            # Outage-recovery phase: game likely crashed, watchdog needs
            # 30-60s to relaunch + emit fresh session.json. Poll until
            # MAX_BRIDGE_OUTAGE_S elapses, only then surface the failure.
            if outage_deadline is None:
                outage_deadline = time.monotonic() + self.MAX_BRIDGE_OUTAGE_S
                outage_announced = True
                print(
                    f"[bridge-client] burst-retry exhausted for {method} /{endpoint}; "
                    f"entering outage-recovery wait (up to {self.MAX_BRIDGE_OUTAGE_S:.0f}s) "
                    f"polling for watchdog to restart bridge...",
                    flush=True,
                )

            if time.monotonic() >= outage_deadline:
                break

            time.sleep(self.BRIDGE_OUTAGE_POLL_S)
            # Continue the while loop — next iteration will retry the request
            # against whatever base_url/token _maybe_rebind_session pulled in.

        raise BridgeError(
            f"Bridge unreachable after {total_attempts} attempts "
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
        seed: str | None = None,
        timeout_ms: int = 45_000,
    ) -> dict[str, Any]:
        """POST /env/reset -- start a new episode.

        If ``seed`` is supplied (10-char alphanumeric — canonicalized on the
        server via SeedHelper.CanonicalizeSeed), the bridge writes
        NGame.Instance.DebugSeedOverride before the embark trigger, fully
        determining map / encounters / card rewards / potion drops / monster
        AI / treasure relics / shuffle order for the new run. ``seed`` is
        ignored when ``rebind_active_run`` is True (run's RNG already baked).

        Requires bridge mod ``env_api_version >= bridge-env-v2-seed``.

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
        if seed is not None:
            body["seed"] = str(seed)
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
