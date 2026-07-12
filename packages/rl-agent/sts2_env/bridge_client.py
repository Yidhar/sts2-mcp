"""HTTP client for the STS2 bridge mod's RL environment endpoints."""

import hashlib
import ipaddress
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests

from sts2_rl.contracts.versions import (
    ACTION_SCHEMA_VERSION,
    API_VERSION,
    LEGAL_ACTION_ORDERING_VERSION,
    SCHEMA_VERSION,
)

from .path_utils import default_bridge_session_dir, normalize_path, running_in_wsl

_SUPPORTED_SESSION_CAPABILITIES = frozenset(
    {"player-control", "training", "legacy-privileged"}
)
_GAME_COMPATIBILITY_KEYS = frozenset(
    {
        "health",
        "startup_allowed",
        "error_code",
        "error_message",
        "profile_id",
        "assembly",
        "probes",
    }
)
_GAME_ASSEMBLY_KEYS = frozenset(
    {"name", "assembly_version", "informational_version", "module_version_id"}
)
_GAME_PROBE_KEYS = frozenset({"capability", "passed", "code", "detail"})


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


def _normalized_base_url(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise BridgeError("Bridge base URL must be an unauthenticated http:// host:port URL.")
    if parsed.query or parsed.fragment:
        raise BridgeError("Bridge base URL must not contain query or fragment components.")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")).rstrip("/")


def _is_loopback_host(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _session_log_hash(session_id: str) -> str:
    """Return a non-secret correlation label without exposing credentials."""

    value = str(session_id or "").encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:12] if value else "none"


def _url_port(value: str) -> int | None:
    try:
        return urlsplit(value).port
    except ValueError:
        return None


def _relay_allowlist() -> frozenset[str]:
    raw = os.environ.get("STS2_BRIDGE_RELAY_ALLOWLIST", "")
    allowed: set[str] = set()
    for value in raw.split(","):
        value = value.strip()
        if value:
            allowed.add(_normalized_base_url(value).casefold())
    return frozenset(allowed)


def _validate_base_url(value: str, *, explicit_relay: bool) -> str:
    normalized = _normalized_base_url(value)
    parsed = urlsplit(normalized)
    if _is_loopback_host(parsed.hostname or ""):
        return normalized
    if explicit_relay and normalized.casefold() in _relay_allowlist():
        return normalized
    raise BridgeError(
        "Non-loopback bridge URLs are denied unless the exact URL is listed in "
        "STS2_BRIDGE_RELAY_ALLOWLIST and selected via STS2_BRIDGE_BASE_URL."
    )


def _resolve_base_url(session_base_url: str, *, allow_env_override: bool = True) -> str:
    # Session descriptors are local authority and must always point at loopback.
    session_url = _validate_base_url(session_base_url, explicit_relay=False)
    explicit_override = os.environ.get("STS2_BRIDGE_BASE_URL")
    if explicit_override and allow_env_override:
        return _validate_base_url(explicit_override, explicit_relay=True)

    rewritten = _rewrite_loopback_url_for_wsl(session_url)
    if rewritten != session_url:
        return _validate_base_url(rewritten, explicit_relay=True)
    return session_url


def _required_bounded_string(
    value: Any,
    *,
    field: str,
    max_length: int,
) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise BridgeError(f"Session descriptor {field} is missing or malformed.")
    return value


def _validate_game_compatibility(data: dict[str, Any]) -> None:
    """Require a successful, structurally exact retail compatibility proof.

    The Bridge only publishes a descriptor after its assembly profile and all
    capability probes pass. Training clients must enforce the same invariant;
    otherwise a stale, hand-written, or partially upgraded descriptor could
    bypass the startup gate merely because its scoped token still looks valid.
    """

    compatibility = data.get("game_compatibility")
    if not isinstance(compatibility, dict) or set(compatibility) != _GAME_COMPATIBILITY_KEYS:
        raise BridgeError("Session descriptor game_compatibility is missing or malformed.")
    if (
        compatibility.get("health") != "ready"
        or compatibility.get("startup_allowed") is not True
        or compatibility.get("error_code") != ""
        or compatibility.get("error_message") != ""
    ):
        raise BridgeError("Session descriptor reports an incompatible game assembly.")
    _required_bounded_string(
        compatibility.get("profile_id"),
        field="game_compatibility.profile_id",
        max_length=128,
    )

    assembly = compatibility.get("assembly")
    if not isinstance(assembly, dict) or set(assembly) != _GAME_ASSEMBLY_KEYS:
        raise BridgeError("Session descriptor game_compatibility.assembly is malformed.")
    for field, max_length in (
        ("name", 128),
        ("assembly_version", 128),
        ("informational_version", 256),
    ):
        _required_bounded_string(
            assembly.get(field),
            field=f"game_compatibility.assembly.{field}",
            max_length=max_length,
        )
    module_version_id = _required_bounded_string(
        assembly.get("module_version_id"),
        field="game_compatibility.assembly.module_version_id",
        max_length=36,
    )
    try:
        parsed_module_version_id = uuid.UUID(module_version_id)
    except ValueError as exc:
        raise BridgeError(
            "Session descriptor game_compatibility.assembly.module_version_id is malformed."
        ) from exc
    if str(parsed_module_version_id) != module_version_id.lower():
        raise BridgeError(
            "Session descriptor game_compatibility.assembly.module_version_id is malformed."
        )

    game_assembly_version = data.get("game_assembly_version")
    if game_assembly_version is not None and game_assembly_version != assembly["assembly_version"]:
        raise BridgeError(
            "Session descriptor game assembly identities are internally inconsistent."
        )

    probes = compatibility.get("probes")
    if not isinstance(probes, list) or not probes:
        raise BridgeError("Session descriptor game_compatibility.probes is malformed.")
    seen_capabilities: set[str] = set()
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != _GAME_PROBE_KEYS:
            raise BridgeError("Session descriptor contains a malformed compatibility probe.")
        capability = _required_bounded_string(
            probe.get("capability"),
            field="game_compatibility.probes[].capability",
            max_length=256,
        )
        if capability in seen_capabilities:
            raise BridgeError("Session descriptor contains duplicate compatibility probes.")
        seen_capabilities.add(capability)
        if probe.get("passed") is not True or probe.get("code") != "capability_present":
            raise BridgeError("Session descriptor contains a failed compatibility probe.")
        _required_bounded_string(
            probe.get("detail"),
            field="game_compatibility.probes[].detail",
            max_length=1024,
        )


def _validate_session_descriptor(data: Any) -> tuple[str, dict[str, str], str]:
    """Validate the versioned identity before accepting any credential.

    A stale or partially written descriptor must not silently downgrade into
    the former ``token``/PID fallback identity.  This intentionally validates
    the cross-component contract constants before returning secrets to the
    request layer.
    """

    if not isinstance(data, dict):
        raise BridgeError("Session descriptor must be a JSON object.")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise BridgeError(f"Unsupported bridge session schema_version; expected {SCHEMA_VERSION}.")
    if data.get("action_schema_version") != ACTION_SCHEMA_VERSION:
        raise BridgeError(
            f"Unsupported bridge action_schema_version; expected {ACTION_SCHEMA_VERSION}."
        )
    if data.get("legal_action_ordering_version") != LEGAL_ACTION_ORDERING_VERSION:
        raise BridgeError(
            "Unsupported bridge legal_action_ordering_version; "
            f"expected {LEGAL_ACTION_ORDERING_VERSION}."
        )

    _validate_game_compatibility(data)

    api_versions = data.get("api_versions")
    if (
        not isinstance(api_versions, list)
        or API_VERSION not in api_versions
        or any(value not in {API_VERSION, "legacy-v1"} for value in api_versions)
    ):
        raise BridgeError(f"Session descriptor must advertise supported API {API_VERSION}.")

    raw_capabilities = data.get("capabilities")
    if not isinstance(raw_capabilities, list) or any(
        not isinstance(value, str) for value in raw_capabilities
    ):
        raise BridgeError("Session descriptor capabilities must be a string array.")
    capabilities = set(raw_capabilities)
    if (
        len(capabilities) != len(raw_capabilities)
        or "player-control" not in capabilities
        or not capabilities.issubset(_SUPPORTED_SESSION_CAPABILITIES)
    ):
        raise BridgeError("Session descriptor capabilities are unsupported or malformed.")

    raw_tokens = data.get("capability_tokens")
    if not isinstance(raw_tokens, dict) or set(raw_tokens) != capabilities:
        raise BridgeError("Session descriptor capability credentials do not match capabilities.")
    capability_tokens: dict[str, str] = {}
    for capability, token in raw_tokens.items():
        if not isinstance(capability, str) or not isinstance(token, str) or len(token) < 32:
            raise BridgeError("Session descriptor contains a malformed scoped credential.")
        capability_tokens[capability] = token

    legacy_token = data.get("token")
    if legacy_token is not None:
        if (
            not isinstance(legacy_token, str)
            or capabilities.isdisjoint({"legacy-privileged"})
            or capability_tokens.get("legacy-privileged") != legacy_token
        ):
            raise BridgeError("Session descriptor legacy credential is inconsistent with its scope.")

    session_id = data.get("session_id")
    if not isinstance(session_id, str) or len(session_id) < 16:
        raise BridgeError("Session descriptor session_id is missing or malformed.")
    pid = data.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise BridgeError("Session descriptor pid is missing or malformed.")
    base_url = data.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise BridgeError("Session descriptor base_url is missing or malformed.")
    return base_url, capability_tokens, session_id


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
        self._session_id: str = ""
        self._capability_tokens: dict[str, str] = {}
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
            ) from None
        except json.JSONDecodeError as exc:
            raise BridgeError(f"Invalid JSON in session file: {exc}") from exc

        base_url, capability_tokens, session_id = _validate_session_descriptor(data)
        self._session_base_url = base_url.rstrip("/")
        self._base_url = _resolve_base_url(
            self._session_base_url,
            allow_env_override=self._allow_env_base_url_override,
        )
        self._token = str(data.get("token") or "")
        self._capability_tokens = capability_tokens
        self._session_id = session_id
        self._session.headers.pop("Authorization", None)
        if self._token:
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
        old_session_hash = _session_log_hash(self._session_id)
        old_port = _url_port(self._base_url)
        try:
            self._load_session()
        except BridgeError:
            return False
        # Keep this print — it's the only operator-visible signal that the
        # client auto-rebound after a launcher-side restart. Without it the
        # restart is silent and very hard to correlate with watchdog events.
        print(
            f"[bridge-client] rebound session ({reason}): "
            f"session={old_session_hash} port={old_port} -> "
            f"session={_session_log_hash(self._session_id)} port={_url_port(self._base_url)}"
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

    @property
    def session_id(self) -> str:
        return self._session_id

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
        if not self._token:
            raise BridgeError(
                "Legacy endpoint requested but the session descriptor has no top-level token.",
                status_code=403,
                response_body={"error": "legacy_token_unavailable"},
            )
        http_timeout_s = (
            (timeout_ms + self.HTTP_TIMEOUT_GRACE_MS) / 1000.0
            if timeout_ms is not None
            else 30.0
        )

        last_exc: Exception | None = None
        quick_attempts_left = self.MAX_RETRIES
        transient_http_attempts_left = 12
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
                    # 2026-05-08 (recovery): 503 with pump_stalled /
                    # main_thread_stalled means the Godot main thread didn't
                    # tick within the bridge's 5s window — usually a GC
                    # pause, GPU-contention spike (training and rendering
                    # share the same GPU), or combat/death transition. Not
                    # a hard fault. Treat as transient and retry with a
                    # generous budget: observed worst-case ms_since_last_pump
                    # is ~18-20s under heavy GPU load, so we need ~30s of
                    # retry slack.
                    #
                    # 2026-05-13 (combat sandbox stability): the bridge can
                    # also transiently answer HTTP 502/504 during
                    # /env/combat_reset after a game-side reset hiccup
                    # (observed with an empty 502 body plus a Godot
                    # SharedSignalPool leak warning).  Treat gateway-class
                    # 5xx as bridge-outage candidates instead of killing the
                    # long-running trainer.
                    transient_5xx = False
                    if resp.status_code in {502, 504}:
                        transient_5xx = True
                    elif (
                        resp.status_code == 503
                        and isinstance(resp_body, dict)
                        and str(resp_body.get("error") or "").strip()
                        in {"pump_stalled", "main_thread_stalled"}
                    ):
                        transient_5xx = True
                    if transient_5xx:
                        last_exc = BridgeError(
                            f"Bridge returned HTTP {resp.status_code} for {method} /{endpoint}: "
                            f"{resp_body}",
                            status_code=resp.status_code,
                            response_body=resp_body,
                        )
                        self._is_connected = False
                        self._maybe_rebind_session(
                            reason=f"HTTP {resp.status_code} on {method} /{endpoint}",
                        )
                        # Use a local per-request counter so a previous
                        # request's transient-5xx budget cannot leak into
                        # this one. 3s sleep x 12 attempts = ~36s quick
                        # slack, enough for most render/GPU/reset stalls.
                        if transient_http_attempts_left > 0:
                            transient_http_attempts_left -= 1
                            time.sleep(3.0)
                            continue
                        # Quick transient-HTTP budget exhausted.  Fall into
                        # the same bounded outage-recovery window used by
                        # ConnectionError/Timeout so watchdog restarts and
                        # session-file rotations can be ridden out without an
                        # infinite retry loop.
                        if outage_deadline is None:
                            outage_deadline = time.monotonic() + self.MAX_BRIDGE_OUTAGE_S
                            outage_announced = True
                            print(
                                f"[bridge-client] transient HTTP {resp.status_code} retries "
                                f"exhausted for {method} /{endpoint}; entering "
                                f"outage-recovery wait (up to {self.MAX_BRIDGE_OUTAGE_S:.0f}s) "
                                f"polling for watchdog to restart bridge...",
                                flush=True,
                            )
                        if time.monotonic() < outage_deadline:
                            time.sleep(self.BRIDGE_OUTAGE_POLL_S)
                            continue
                        break
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

    def _request_no_retry(
        self,
        method: str,
        endpoint: str,
        *,
        body: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
        capability: str | None = None,
    ) -> Any:
        """Perform exactly one HTTP attempt.

        Mutation callers use this path when the endpoint is not idempotent.
        v2 callers may safely submit the same request_id again explicitly, but
        transport code never invents a second mutation attempt.
        """

        session = self._session
        if session is None:
            raise BridgeError("BridgeClient is closed.")
        http_timeout_s = (
            (timeout_ms + self.HTTP_TIMEOUT_GRACE_MS) / 1000.0
            if timeout_ms is not None
            else 30.0
        )
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        headers: dict[str, str] | None = None
        if capability is not None:
            token = self._capability_tokens.get(str(capability))
            if not token:
                raise BridgeError(
                    f"Bridge session does not grant capability {capability!r}.",
                    status_code=403,
                    response_body={"error": "capability_not_granted", "capability": capability},
                )
            headers = {"Authorization": f"Bearer {token}"}
        try:
            response = session.request(
                method,
                url,
                json=body if body is not None else None,
                timeout=http_timeout_s,
                headers=headers,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            self._is_connected = False
            raise BridgeError(
                f"Bridge {method} /{endpoint} failed after one attempt; "
                "mutation outcome may be unknown.",
                response_body={"outcome_unknown": True, "exception": type(exc).__name__},
            ) from exc

        try:
            payload = response.json()
        except (requests.JSONDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._is_connected = False
            raise BridgeError(
                f"Bridge returned non-JSON for {method} /{endpoint}.",
                status_code=response.status_code,
                response_body=response.text,
            ) from exc
        if response.status_code >= 400:
            raise BridgeError(
                f"Bridge returned HTTP {response.status_code} for {method} /{endpoint}: {payload}",
                status_code=response.status_code,
                response_body=payload,
            )
        self._is_connected = True
        return payload

    def reset_v2(
        self,
        *,
        request_id: str,
        session_id: str,
        expected_state_version: int,
        scenario: str,
        seed: str | int | None,
        options: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        if (
            isinstance(expected_state_version, bool)
            or not isinstance(expected_state_version, int)
            or expected_state_version < 0
        ):
            raise ValueError("expected_state_version must be a non-negative integer")
        return self._request_no_retry(
            "POST",
            "v2/env/reset",
            body={
                "request_id": request_id,
                "session_id": session_id,
                "expected_state_version": expected_state_version,
                "scenario": scenario,
                "seed": seed,
                "options": options,
            },
            timeout_ms=timeout_ms,
            capability="training",
        )

    def step_v2(
        self,
        *,
        request_id: str,
        session_id: str,
        episode_id: str,
        expected_step_index: int,
        action: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        return self._request_no_retry(
            "POST",
            "v2/env/step",
            body={
                "request_id": request_id,
                "session_id": session_id,
                "episode_id": episode_id,
                "expected_step_index": expected_step_index,
                "action": action,
            },
            timeout_ms=timeout_ms,
            capability="training",
        )

    def reset_no_retry(self, **kwargs: Any) -> dict[str, Any]:
        body = {key: value for key, value in kwargs.items() if value is not None}
        timeout_ms = int(body.get("timeout_ms", 45_000))
        return self._request_no_retry("POST", "env/reset", body=body, timeout_ms=timeout_ms)

    def step_no_retry(self, **kwargs: Any) -> dict[str, Any]:
        body = {key: value for key, value in kwargs.items() if value is not None}
        timeout_ms = int(body.get("timeout_ms", 20_000))
        return self._request_no_retry("POST", "env/step", body=body, timeout_ms=timeout_ms)

    def combat_reset_no_retry(self, **kwargs: Any) -> dict[str, Any]:
        body = {key: value for key, value in kwargs.items() if value is not None}
        timeout_ms = int(body.get("timeout_ms", 15_000))
        return self._request_no_retry("POST", "env/combat_reset", body=body, timeout_ms=timeout_ms)

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
        """Deprecated legacy catalog read for explicit compatibility only."""
        return self._request("GET", "env/combat_catalog")

    def health(self) -> dict[str, Any]:
        """Deprecated legacy health read for explicit compatibility only."""
        return self._request("GET", "health")

    def health_v2(self) -> dict[str, Any]:
        return self._request_no_retry("GET", "v2/health", capability="training")

    def get_spec_v2(self) -> dict[str, Any]:
        return self._request_no_retry("GET", "v2/env/spec", capability="training")

    def get_state_v2(self) -> dict[str, Any]:
        return self._request_no_retry("GET", "v2/env/state", capability="training")

    def combat_catalog_v2(self) -> dict[str, Any]:
        return self._request_no_retry("GET", "v2/env/combat_catalog", capability="training")

    def get_state(self) -> dict[str, Any]:
        """Deprecated legacy state read for explicit compatibility only."""
        return self._request("GET", "state")

    def close(self) -> None:
        """Release the pooled HTTP connection. Idempotent."""

        if not getattr(self, "_is_connected", False) and getattr(self, "_session", None) is None:
            return
        session = getattr(self, "_session", None)
        self._session = None
        self._is_connected = False
        if session is not None:
            session.close()

    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
