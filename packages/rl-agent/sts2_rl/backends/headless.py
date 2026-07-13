"""Idempotent typed backend for the in-process headless simulator client."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

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


DEFAULT_REQUEST_CACHE_SIZE = 65_536
DEFAULT_REQUEST_CACHE_TTL_S = 600.0


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
        self._request_cache: dict[str, tuple[str, EnvironmentResult, float]] = {}
        self._request_expiry: deque[tuple[float, str]] = deque()
        self._lock = threading.RLock()
        self._closed = False
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
        return not self._closed and bool(getattr(self._client, "is_connected", True))

    def health(self) -> dict[str, Any]:
        self._ensure_open()
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
        self._ensure_open()
        payload = dict(self._client.get_state())
        # This is the typed adapter's mutation revision, not the simulator's
        # episode-local step index. It is monotonic for the backend lifetime.
        payload["state_version"] = self._state_version
        payload.setdefault("ok", True)
        payload.setdefault("capability", "training")
        payload.setdefault("session_id", self.session_id)
        return payload

    def combat_catalog(self) -> dict[str, Any]:
        self._ensure_open()
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
        return EnvironmentResult.from_legacy(raw)

    def _ensure_request_session(self, session_id: str) -> None:
        if str(session_id) != self.session_id:
            raise HeadlessProtocolError(
                f"request session {session_id!r} does not match backend session {self.session_id!r}"
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("headless backend is closed")

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
            self._ensure_open()
            self._ensure_request_session(request.session_id)
            fingerprint = self._fingerprint(request)
            cached = self._cached(request.request_id, fingerprint)
            if cached is not None:
                return cached
            self._ensure_reset_revision(request.expected_state_version)
            self._require_cache_capacity()
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
            self._logical_step_index = int(result.step_index)
            self._state_version = after_state_version
            self._episode_id = result.episode_id
            self._observation = dict(result.observation)
            self._remember(request.request_id, fingerprint, result)
            return result

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult:
        with self._lock:
            self._ensure_open()
            self._ensure_request_session(request.session_id)
            fingerprint = self._fingerprint(request)
            cached = self._cached(request.request_id, fingerprint)
            if cached is not None:
                return cached
            self._ensure_reset_revision(request.expected_state_version)
            self._require_cache_capacity()
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
            self._logical_step_index = 0
            self._state_version = after_state_version
            self._episode_id = result.episode_id
            self._observation = dict(result.observation)
            self._remember(request.request_id, fingerprint, result)
            return result

    def step(self, request: StepRequest) -> EnvironmentResult:
        with self._lock:
            self._ensure_open()
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
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
