"""Explicit compatibility adapter for deprecated dictionary clients."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from sts2_rl.contracts import (
    BackendCapabilities,
    CombatResetRequest,
    EnvironmentResult,
    ResetRequest,
    StepRequest,
)


def _mark_legacy_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    info = result.get("info")
    info = dict(info) if isinstance(info, Mapping) else {}
    info["legacy_non_idempotent"] = True
    result["info"] = info
    return result


def _invoke_legacy_method(method: Any, kwargs: Mapping[str, Any]) -> Any:
    """Call a legacy method without forcing unsupported compatibility kwargs."""
    values = {key: value for key, value in kwargs.items() if value is not None}
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(**values)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return method(**values)
    accepted = {name for name in signature.parameters if name != "self"}
    return method(**{key: value for key, value in values.items() if key in accepted})


class LegacyClientBackend:
    """Owns one legacy client and exposes the typed EnvironmentBackend API."""

    def __init__(
        self,
        client: Any,
        *,
        backend_name: str,
        session_id: str | None = None,
    ) -> None:
        self._client = client
        resolved_session = str(
            session_id or getattr(client, "session_id", "") or f"{backend_name}-legacy"
        )
        self._capabilities = BackendCapabilities(
            backend_name=backend_name,
            session_id=resolved_session,
        )
        self._closed = False

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
        return dict(self._client.health())

    def get_spec(self) -> dict[str, Any]:
        self._ensure_open()
        method = getattr(self._client, "get_spec", None)
        if not callable(method):
            return {
                "ok": True,
                "backend": self._capabilities.backend_name,
                "api_version": self._capabilities.contract_version,
                "schema_version": self._capabilities.schema_version,
            }
        return dict(method())

    def get_state(self) -> dict[str, Any]:
        self._ensure_open()
        return dict(self._client.get_state())

    def reset(self, request: ResetRequest) -> EnvironmentResult:
        self._ensure_request_session(request.session_id)
        self._ensure_open()
        return EnvironmentResult.from_legacy(
            _invoke_legacy_method(self._client.reset, request.to_legacy_kwargs())
        )

    def step(self, request: StepRequest) -> EnvironmentResult:
        self._ensure_request_session(request.session_id)
        self._ensure_open()
        return EnvironmentResult.from_legacy(
            _invoke_legacy_method(self._client.step, request.to_legacy_kwargs())
        )

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult:
        self._ensure_request_session(request.session_id)
        self._ensure_open()
        return EnvironmentResult.from_legacy(
            _invoke_legacy_method(self._client.combat_reset, request.to_legacy_kwargs())
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _ensure_request_session(self, session_id: str) -> None:
        if str(session_id) != self.session_id:
            raise ValueError(
                f"request session {session_id!r} does not match backend session {self.session_id!r}"
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self._capabilities.backend_name} backend is closed")

    def __enter__(self) -> LegacyClientBackend:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
