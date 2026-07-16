"""Idempotent typed backend for the in-process headless simulator client."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.contracts import (
    BackendCapabilities,
    CombatResetRequest,
    EnvironmentResult,
    ResetRequest,
    StepRequest,
)
from sts2_rl.transitions import TransitionFacts, derive_transition_facts

from .legacy import _invoke_legacy_method


class HeadlessProtocolError(RuntimeError):
    """A typed headless request violated session/idempotency/step ordering."""

    recoverable = False


class HeadlessRecoverableProtocolError(HeadlessProtocolError):
    """A dispatched mutation produced an uncommittable backend surface.

    Unlike request/session/idempotency errors, this exception identifies an
    infrastructure incident that an actor supervisor may recover from by
    discarding the current episode and replacing the entire backend/client
    session.  Reusing the poisoned backend is never a recovery operation.
    """

    recoverable = True

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        incident_kind: str,
        fingerprint: str,
        quarantine_path: str | None,
    ) -> None:
        super().__init__(message)
        self.operation = str(operation)
        self.incident_kind = str(incident_kind)
        self.fingerprint = str(fingerprint)
        self.quarantine_path = quarantine_path
        self.poisoned = True


class HeadlessBackendPoisonedError(HeadlessRecoverableProtocolError):
    """A caller attempted to reuse a backend after a mutation incident."""


DEFAULT_REQUEST_CACHE_SIZE = 65_536
DEFAULT_REQUEST_CACHE_TTL_S = 600.0
DEFAULT_PROTOCOL_QUARANTINE_DIR = "logs/headless-sim/protocol-incidents"
PROTOCOL_EVIDENCE_GZIP_THRESHOLD = 64 * 1024


class HeadlessBackend:
    """Typed, process-local projection over ``HeadlessSimBridgeClient``.

    The raw simulator remains synchronous, but this boundary supplies the v2
    invariants it does not natively implement: request-id replay, request body
    conflict detection, session scope, expected-step ordering, canonical
    transition facts, and external reward authority.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        session_id: str | None = None,
        request_cache_size: int = DEFAULT_REQUEST_CACHE_SIZE,
        request_cache_ttl_s: float = DEFAULT_REQUEST_CACHE_TTL_S,
        protocol_quarantine_dir: str | Path | None = None,
        clock: Callable[[], float] | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is None:
            from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient

            client = HeadlessSimBridgeClient(**client_kwargs)
        self._client = client
        resolved_session = str(session_id or getattr(client, "session_id", "") or "headless-sim")
        self._capabilities = BackendCapabilities(
            backend_name="headless_sim",
            session_id=resolved_session,
            supports_full_run=True,
            supports_combat_reset=True,
            supports_seed=True,
            privileged_training=True,
        )
        if int(request_cache_size) <= 0:
            raise ValueError("request_cache_size must be positive")
        if float(request_cache_ttl_s) <= 0.0:
            raise ValueError("request_cache_ttl_s must be positive")
        self._request_cache_size = int(request_cache_size)
        self._request_cache_ttl_s = float(request_cache_ttl_s)
        self._clock = clock or time.monotonic
        self._protocol_quarantine_dir = protocol_quarantine_dir
        self._request_cache: dict[str, tuple[str, EnvironmentResult, float]] = {}
        self._request_expiry: deque[tuple[float, str]] = deque()
        self._lock = threading.RLock()
        self._closed = False
        self._poisoned = False
        self._poison_reason: str | None = None
        self._poison_fingerprint: str | None = None
        self._poison_quarantine_path: str | None = None
        self._logical_step_index = 0
        self._state_version = 0
        self._episode_id = ""
        self._observation: dict[str, Any] | None = None

    @property
    def client(self) -> Any:
        return self._client

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def session_id(self) -> str:
        return self._capabilities.session_id

    @property
    def is_connected(self) -> bool:
        return (
            not self._closed
            and not self._poisoned
            and bool(getattr(self._client, "is_connected", True))
        )

    def health(self) -> dict[str, Any]:
        self._ensure_open()
        if self._poisoned:
            return {
                "ok": False,
                "backend": "headless_sim",
                "session_id": self.session_id,
                "poisoned": True,
                "poison_reason": self._poison_reason,
                "incident_fingerprint": self._poison_fingerprint,
                "quarantine_path": self._poison_quarantine_path,
            }
        payload = dict(self._client.health())
        payload.setdefault("backend", "headless_sim")
        payload.setdefault("session_id", self.session_id)
        return payload

    def get_spec(self) -> dict[str, Any]:
        self._ensure_open()
        return {
            "ok": True,
            "backend": "headless_sim",
            "api_version": self.capabilities.contract_version,
            "schema_version": self.capabilities.schema_version,
            "supports_full_run": True,
            "supports_combat_reset": True,
            "supports_seed": True,
            "request_id_dedupe": True,
            "request_id_dedupe_capacity": self._request_cache_size,
            "request_id_dedupe_ttl_s": self._request_cache_ttl_s,
            "expected_step_index": True,
            "reward_authority": "external-rl",
        }

    def get_state(self) -> dict[str, Any]:
        self._ensure_usable()
        payload = dict(self._client.get_state())
        # This is the typed adapter's mutation revision, not the simulator's
        # episode-local step index. It is monotonic for the backend lifetime.
        payload["state_version"] = self._state_version
        payload.setdefault("ok", True)
        payload.setdefault("capability", "training")
        payload.setdefault("session_id", self.session_id)
        return payload

    def combat_catalog(self) -> dict[str, Any]:
        self._ensure_usable()
        return dict(self._client.combat_catalog())

    @staticmethod
    def _fingerprint(request: ResetRequest | StepRequest | CombatResetRequest) -> str:
        return type(request).__name__ + ":" + json.dumps(
            asdict(request),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _purge_expired(self) -> None:
        now = self._clock()
        while self._request_expiry and self._request_expiry[0][0] <= now:
            expires_at, request_id = self._request_expiry.popleft()
            cached = self._request_cache.get(request_id)
            if cached is not None and cached[2] == expires_at:
                del self._request_cache[request_id]

    def _cached(self, request_id: str, fingerprint: str) -> EnvironmentResult | None:
        self._purge_expired()
        cached = self._request_cache.get(request_id)
        if cached is None:
            return None
        cached_fingerprint, result, _ = cached
        if cached_fingerprint != fingerprint:
            raise HeadlessProtocolError("request_id was reused with a different request body")
        return result

    def _require_cache_capacity(self) -> None:
        self._purge_expired()
        if len(self._request_cache) >= self._request_cache_size:
            raise HeadlessProtocolError(
                "request-id store capacity exceeded; unexpired identities cannot be evicted"
            )

    def _remember(self, request_id: str, fingerprint: str, result: EnvironmentResult) -> None:
        if request_id not in self._request_cache:
            self._require_cache_capacity()
        expires_at = self._clock() + self._request_cache_ttl_s
        self._request_cache[request_id] = (fingerprint, result, expires_at)
        self._request_expiry.append((expires_at, request_id))

    @staticmethod
    def _facts_payload(facts: TransitionFacts) -> dict[str, Any]:
        payload = asdict(facts)
        for key in (
            "cards_added",
            "cards_removed",
            "potions_added",
            "potions_removed",
            "relics_used",
        ):
            payload[key] = list(payload[key])
        return payload

    @staticmethod
    def _incident_projection(
        *,
        operation: str,
        error: BaseException,
        response: Any,
    ) -> dict[str, Any]:
        state: Mapping[str, Any] = {}
        wrapper: Mapping[str, Any] = response if isinstance(response, Mapping) else {}
        state_value = wrapper.get("state")
        if isinstance(state_value, Mapping):
            state = state_value
        elif isinstance(response, Mapping):
            observation_value = response.get("observation", response.get("obs"))
            state = observation_value if isinstance(observation_value, Mapping) else response
        actions = state.get("legal_actions", wrapper.get("legal_actions"))
        action_count = len(actions) if isinstance(actions, list | tuple) else None
        return {
            "operation": str(operation),
            "error_type": type(error).__name__,
            "settlement_status": wrapper.get("settlement_status"),
            "accepted": wrapper.get("accepted"),
            "action_committed": wrapper.get("action_committed"),
            "state_type": state.get("state_type"),
            "terminal": state.get("terminal", wrapper.get("done", wrapper.get("terminated"))),
            "truncated": state.get("truncated", wrapper.get("truncated")),
            "legal_action_count": action_count,
        }

    @staticmethod
    def _incident_fingerprint(projection: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(projection),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def _write_protocol_evidence(
        self,
        *,
        operation: str,
        request: ResetRequest | StepRequest | CombatResetRequest,
        error: BaseException,
        response: Any,
        projection: Mapping[str, Any],
        fingerprint: str,
    ) -> str | None:
        """Best-effort atomic incident evidence outside the source checkout."""

        try:
            directory = resolve_artifact_path(
                self._protocol_quarantine_dir,
                default=DEFAULT_PROTOCOL_QUARANTINE_DIR,
            )
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "created_unix_ns": time.time_ns(),
                "incident": dict(projection),
                "fingerprint": fingerprint,
                "backend": {
                    "session_id": self.session_id,
                    "logical_step_index": self._logical_step_index,
                    "state_version": self._state_version,
                    "episode_id": self._episode_id,
                },
                "request": {
                    "type": type(request).__name__,
                    "body": asdict(request),
                },
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "before_observation": self._observation,
                "raw_response": response,
            }
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            stem = (
                f"incident-{time.time_ns()}-{operation}-"
                f"sv{self._state_version}-step{self._logical_step_index}-{fingerprint}"
            )
            if len(encoded) >= PROTOCOL_EVIDENCE_GZIP_THRESHOLD:
                path = directory / f"{stem}.json.gz"
                temp = directory / f".{stem}.{os.getpid()}.{threading.get_ident()}.tmp"
                with gzip.open(temp, "wb", compresslevel=6) as handle:
                    handle.write(encoded)
            else:
                path = directory / f"{stem}.json"
                temp = directory / f".{stem}.{os.getpid()}.{threading.get_ident()}.tmp"
                with temp.open("wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(temp, path)
            return str(path)
        except Exception:
            # Evidence must never hide the primary protocol failure.  The
            # supervisor still receives the stable incident fingerprint.
            return None

    def _poison_after_mutation(
        self,
        *,
        operation: str,
        request: ResetRequest | StepRequest | CombatResetRequest,
        error: BaseException,
        raw_response: Any = None,
    ) -> HeadlessRecoverableProtocolError:
        response = raw_response
        if response is None:
            response = getattr(error, "response", None)
        projection = self._incident_projection(
            operation=operation,
            error=error,
            response=response,
        )
        fingerprint = self._incident_fingerprint(projection)
        quarantine_path = self._write_protocol_evidence(
            operation=operation,
            request=request,
            error=error,
            response=response,
            projection=projection,
            fingerprint=fingerprint,
        )
        self._poisoned = True
        self._poison_reason = f"{type(error).__name__}: {error}"
        self._poison_fingerprint = fingerprint
        self._poison_quarantine_path = quarantine_path
        return HeadlessRecoverableProtocolError(
            f"headless {operation} produced an uncommittable response; "
            f"backend session is poisoned (incident={fingerprint}, evidence={quarantine_path})",
            operation=operation,
            incident_kind=str(projection.get("error_type") or "protocol_error"),
            fingerprint=fingerprint,
            quarantine_path=quarantine_path,
        )

    @staticmethod
    def _validate_raw_surface(raw: Mapping[str, Any]) -> None:
        ok_value = raw.get("ok", True)
        if not isinstance(ok_value, bool) or not ok_value:
            raise ValueError("headless environment response is not ok=true")
        observation = raw.get("observation", raw.get("obs"))
        if not isinstance(observation, Mapping):
            raise TypeError("headless environment response has no observation object")
        actions = raw.get("legal_actions")
        if not isinstance(actions, list | tuple):
            raise TypeError("headless environment response has no legal_actions array")
        if any(not isinstance(action, Mapping) for action in actions):
            raise TypeError("headless environment response contains a non-object legal action")
        done_value = raw.get("done", raw.get("terminated", False))
        truncated_value = raw.get("truncated", False)
        if not isinstance(done_value, bool) or not isinstance(truncated_value, bool):
            raise TypeError("headless environment terminal flags must be exact booleans")
        terminal = done_value or truncated_value
        if terminal and actions:
            raise ValueError("terminal headless environment response returned legal actions")
        if not terminal and not actions:
            raise ValueError("non-terminal headless environment response returned zero legal actions")

    def _project(
        self,
        raw_payload: Mapping[str, Any],
        *,
        before_observation: Mapping[str, Any] | None,
        step_index: int,
        before_state_version: int,
        after_state_version: int,
    ) -> EnvironmentResult:
        raw = dict(raw_payload)
        self._validate_raw_surface(raw)
        observation_value = raw.get("observation", raw.get("obs"))
        observation = dict(observation_value) if isinstance(observation_value, Mapping) else {}
        info_value = raw.get("info")
        info = dict(info_value) if isinstance(info_value, Mapping) else {}
        terminated = bool(raw.get("terminated", raw.get("done", False)))
        truncated = bool(raw.get("truncated", False))
        terminal_reason_value = raw.get("terminal_reason", info.get("terminal_reason"))
        terminal_reason = str(terminal_reason_value) if terminal_reason_value is not None else None
        raw_backend_reward = float(raw.get("reward", 0.0) or 0.0)
        if before_observation is None:
            facts = TransitionFacts()
        else:
            facts = derive_transition_facts(
                before_observation,
                observation,
                terminated=terminated,
                terminal_reason=terminal_reason,
            )
        episode_id = str(raw.get("episode_id") or self._episode_id)
        transition = {
            "episode_id": episode_id,
            "step_index": int(step_index),
            "before_state_version": int(before_state_version),
            "after_state_version": int(after_state_version),
            "facts": self._facts_payload(facts),
        }
        info["reward_authority"] = "external-rl"
        info["sim_backend_reward_ignored"] = raw_backend_reward
        info.pop("reward_breakdown", None)
        raw.update(
            {
                "ok": bool(raw.get("ok", True)),
                "episode_id": episode_id,
                "step_index": int(step_index),
                "observation": observation,
                "obs": observation,
                "terminated": terminated,
                "done": terminated,
                "truncated": truncated,
                "terminal_reason": terminal_reason,
                "legal_actions": list(raw.get("legal_actions") or ()),
                "transition": transition,
                "transition_facts": transition,
                "reward": None,
                "reward_status": "not_computed",
                "reward_authority": "external-rl",
                "info": info,
            }
        )
        result = EnvironmentResult.from_legacy(raw)
        # Recheck the immutable projection rather than trusting legacy coercion
        # to preserve the terminal/actionable XOR invariant.
        if (result.terminated or result.truncated) and result.legal_actions:
            raise ValueError("terminal projected headless result returned legal actions")
        if not result.terminated and not result.truncated and not result.legal_actions:
            raise ValueError("non-terminal projected headless result returned zero legal actions")
        return result

    def _ensure_request_session(self, session_id: str) -> None:
        if str(session_id) != self.session_id:
            raise HeadlessProtocolError(
                f"request session {session_id!r} does not match backend session {self.session_id!r}"
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("headless backend is closed")

    def _ensure_usable(self) -> None:
        self._ensure_open()
        if self._poisoned:
            raise HeadlessBackendPoisonedError(
                "headless backend is poisoned and must be replaced",
                operation="reuse",
                incident_kind="backend_poisoned",
                fingerprint=self._poison_fingerprint or "unknown",
                quarantine_path=self._poison_quarantine_path,
            )

    def _ensure_reset_revision(self, expected_state_version: int | None) -> None:
        if expected_state_version is None:
            raise HeadlessProtocolError("typed reset request is missing expected_state_version")
        if expected_state_version != self._state_version:
            raise HeadlessProtocolError(
                f"expected_state_version={expected_state_version} does not match current "
                f"state_version={self._state_version}"
            )

    def reset(self, request: ResetRequest) -> EnvironmentResult:
        with self._lock:
            self._ensure_usable()
            self._ensure_request_session(request.session_id)
            fingerprint = self._fingerprint(request)
            cached = self._cached(request.request_id, fingerprint)
            if cached is not None:
                return cached
            self._ensure_reset_revision(request.expected_state_version)
            self._require_cache_capacity()
            raw: Any = None
            try:
                raw = _invoke_legacy_method(self._client.reset, request.to_legacy_kwargs())
                if not isinstance(raw, Mapping):
                    raise TypeError("headless reset must return a mapping")
                step_index = self._logical_step_index if request.rebind_active_run else 0
                before_state_version = self._state_version
                after_state_version = before_state_version + 1
                result = self._project(
                    raw,
                    before_observation=None,
                    step_index=step_index,
                    before_state_version=before_state_version,
                    after_state_version=after_state_version,
                )
            except Exception as exc:
                raise self._poison_after_mutation(
                    operation="reset",
                    request=request,
                    error=exc,
                    raw_response=raw,
                ) from exc
            self._logical_step_index = int(result.step_index)
            self._state_version = after_state_version
            self._episode_id = result.episode_id
            self._observation = dict(result.observation)
            self._remember(request.request_id, fingerprint, result)
            return result

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult:
        with self._lock:
            self._ensure_usable()
            self._ensure_request_session(request.session_id)
            fingerprint = self._fingerprint(request)
            cached = self._cached(request.request_id, fingerprint)
            if cached is not None:
                return cached
            self._ensure_reset_revision(request.expected_state_version)
            self._require_cache_capacity()
            raw: Any = None
            try:
                raw = _invoke_legacy_method(self._client.combat_reset, request.to_legacy_kwargs())
                if not isinstance(raw, Mapping):
                    raise TypeError("headless combat reset must return a mapping")
                before_state_version = self._state_version
                after_state_version = before_state_version + 1
                result = self._project(
                    raw,
                    before_observation=None,
                    step_index=0,
                    before_state_version=before_state_version,
                    after_state_version=after_state_version,
                )
            except Exception as exc:
                raise self._poison_after_mutation(
                    operation="combat_reset",
                    request=request,
                    error=exc,
                    raw_response=raw,
                ) from exc
            self._logical_step_index = 0
            self._state_version = after_state_version
            self._episode_id = result.episode_id
            self._observation = dict(result.observation)
            self._remember(request.request_id, fingerprint, result)
            return result

    def step(self, request: StepRequest) -> EnvironmentResult:
        with self._lock:
            self._ensure_usable()
            self._ensure_request_session(request.session_id)
            fingerprint = self._fingerprint(request)
            cached = self._cached(request.request_id, fingerprint)
            if cached is not None:
                return cached
            self._require_cache_capacity()
            if request.episode_id != self._episode_id:
                raise HeadlessProtocolError(
                    f"step episode {request.episode_id!r} does not match active episode {self._episode_id!r}"
                )
            if request.expected_step_index != self._logical_step_index:
                raise HeadlessProtocolError(
                    f"expected_step_index={request.expected_step_index} does not match current "
                    f"step_index={self._logical_step_index}"
                )
            before = dict(self._observation or {})
            raw: Any = None
            try:
                raw = _invoke_legacy_method(self._client.step, request.to_legacy_kwargs())
                if not isinstance(raw, Mapping):
                    raise TypeError("headless step must return a mapping")
                next_step = self._logical_step_index + 1
                before_state_version = self._state_version
                after_state_version = before_state_version + 1
                result = self._project(
                    raw,
                    before_observation=before,
                    step_index=next_step,
                    before_state_version=before_state_version,
                    after_state_version=after_state_version,
                )
            except Exception as exc:
                raise self._poison_after_mutation(
                    operation="step",
                    request=request,
                    error=exc,
                    raw_response=raw,
                ) from exc
            self._logical_step_index = next_step
            self._state_version = after_state_version
            self._episode_id = result.episode_id
            self._observation = dict(result.observation)
            self._remember(request.request_id, fingerprint, result)
            return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> HeadlessBackend:
        self._ensure_usable()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
