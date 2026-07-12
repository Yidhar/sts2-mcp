"""Strict contract-v2 live bridge backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sts2_rl.contracts import CombatResetRequest, EnvironmentResult, ResetRequest, StepRequest
from sts2_rl.contracts.versions import API_VERSION, SCHEMA_VERSION

from .legacy import LegacyClientBackend, _invoke_legacy_method, _mark_legacy_result


class EnvironmentCommandError(RuntimeError):
    """A v2 environment command did not commit."""

    def __init__(self, envelope: Mapping[str, Any]):
        self.envelope = dict(envelope)
        self.status = str(envelope.get("status") or "unknown")
        error = envelope.get("error")
        message = error.get("message") if isinstance(error, Mapping) else None
        super().__init__(message or f"environment command ended with status {self.status!r}")


def _normalize_v2_training_payload(payload: Any, *, surface: str) -> dict[str, Any]:
    """Translate strict v2 wire handles back to the legacy/internal RL identity.

    The live adapter is the only transport boundary that performs this mapping.
    The Bridge retains ``action_handle`` on the v2 wire; the Python dispatch
    contract uses ``action_id`` internally without exposing that opaque value to
    model features.
    """

    def invalid(message: str) -> EnvironmentCommandError:
        return EnvironmentCommandError(
            {
                "status": "rejected_before_execution",
                "error": {
                    "code": "invalid_v2_environment_payload",
                    "message": f"v2 {surface} {message}",
                },
            }
        )

    if not isinstance(payload, Mapping):
        raise invalid("response is not an object")
    actions = payload.get("legal_actions")
    if not isinstance(actions, list):
        raise invalid("response has no legal_actions array")

    normalized_actions: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise invalid(f"legal_actions[{index}] is not an object")
        if "action_id" in action:
            raise invalid(f"legal_actions[{index}] uses forbidden action_id")
        action_handle = action.get("action_handle")
        if not isinstance(action_handle, str) or not action_handle.strip():
            raise invalid(f"legal_actions[{index}] has no non-empty action_handle")
        if len(action_handle) > 512:
            raise invalid(f"legal_actions[{index}] action_handle exceeds 512 characters")
        normalized_action = dict(action)
        normalized_action.pop("action_handle", None)
        normalized_action["action_id"] = action_handle
        normalized_actions.append(normalized_action)

    normalized = dict(payload)
    normalized["legal_actions"] = normalized_actions
    return normalized


def _unwrap_v2_result(
    envelope: Mapping[str, Any],
    *,
    expected_request_id: str,
) -> dict[str, Any]:
    if not isinstance(envelope, Mapping):
        raise EnvironmentCommandError(
            {
                "status": "rejected_before_execution",
                "error": {"message": "v2 command response must be an object"},
            }
        )
    identity_errors: list[str] = []
    if envelope.get("api_version") != API_VERSION:
        identity_errors.append("api_version")
    if envelope.get("schema_version") != SCHEMA_VERSION:
        identity_errors.append("schema_version")
    if envelope.get("request_id") != expected_request_id:
        identity_errors.append("request_id")
    if not isinstance(envelope.get("replayed_result"), bool):
        identity_errors.append("replayed_result")
    if identity_errors:
        raise EnvironmentCommandError(
            {
                **dict(envelope),
                "status": "rejected_before_execution",
                "error": {
                    "code": "v2_command_identity_mismatch",
                    "message": (
                        "v2 command response failed contract identity validation: "
                        + ", ".join(identity_errors)
                    ),
                },
            }
        )
    if not bool(envelope.get("ok")) or str(envelope.get("status")) != "committed":
        raise EnvironmentCommandError(envelope)
    result = envelope.get("result")
    if not isinstance(result, Mapping):
        raise EnvironmentCommandError(
            {**dict(envelope), "error": {"message": "committed command has no result object"}}
        )
    transition = result.get("transition")
    authority = str(result.get("reward_authority") or "")
    if not isinstance(transition, Mapping) or not isinstance(transition.get("facts"), Mapping):
        raise EnvironmentCommandError(
            {**dict(envelope), "error": {"message": "v2 result has no typed transition facts"}}
        )
    if authority != "external-rl":
        raise EnvironmentCommandError(
            {**dict(envelope), "error": {"message": "v2 result must delegate reward to external-rl"}}
        )
    return _normalize_v2_training_payload(result, surface="environment result")


class LiveBackend(LegacyClientBackend):
    """Typed adapter that prefers idempotent scoped v2 training endpoints."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        session_path: str | None = None,
        session_id: str | None = None,
        allow_legacy_fallback: bool = False,
    ) -> None:
        if client is None:
            from sts2_env.bridge_client import BridgeClient

            client = BridgeClient(session_path=session_path)
        super().__init__(client, backend_name="live_bridge", session_id=session_id)
        # Compatibility callers may opt into the deprecated v1 endpoints, but
        # the canonical training factory always passes False.  Keeping this
        # explicit prevents an unavailable v2 endpoint from silently issuing a
        # second, non-idempotent mutation on the mainline training path.
        self.allow_legacy_fallback = bool(allow_legacy_fallback)

    @staticmethod
    def _is_missing_v2(exc: Exception) -> bool:
        return int(getattr(exc, "status_code", 0) or 0) == 404

    def _legacy_call(self, method_name: str, kwargs: dict[str, Any]) -> EnvironmentResult:
        if not self.allow_legacy_fallback:
            raise RuntimeError(
                f"live v2 endpoint for {method_name} is unavailable; "
                "legacy mutation fallback is disabled"
            )
        method = getattr(self.client, f"{method_name}_no_retry", None)
        if not callable(method):
            method = getattr(self.client, method_name)
        return EnvironmentResult.from_legacy(
            _mark_legacy_result(_invoke_legacy_method(method, kwargs))
        )

    def health(self) -> dict[str, Any]:
        self._ensure_open()
        method = getattr(self.client, "health_v2", None)
        if callable(method):
            return dict(method())
        if self.allow_legacy_fallback:
            return super().health()
        raise RuntimeError("live v2 health endpoint is unavailable")

    def get_spec(self) -> dict[str, Any]:
        self._ensure_open()
        method = getattr(self.client, "get_spec_v2", None)
        if callable(method):
            return dict(method())
        if self.allow_legacy_fallback:
            return super().get_spec()
        raise RuntimeError("live v2 spec endpoint is unavailable")

    def get_state(self) -> dict[str, Any]:
        self._ensure_open()
        method = getattr(self.client, "get_state_v2", None)
        if callable(method):
            return _normalize_v2_training_payload(method(), surface="environment state")
        if self.allow_legacy_fallback:
            return super().get_state()
        raise RuntimeError("live v2 state endpoint is unavailable")

    def combat_catalog(self) -> dict[str, Any]:
        self._ensure_open()
        method = getattr(self.client, "combat_catalog_v2", None)
        if callable(method):
            return dict(method())
        if self.allow_legacy_fallback:
            return dict(self.client.combat_catalog())
        raise RuntimeError("live v2 combat catalog endpoint is unavailable")

    def _authenticated_v2_state_version(self) -> int:
        """Read the current reset revision through the training-scoped v2 API."""

        method = getattr(self.client, "get_state_v2", None)
        if not callable(method):
            raise RuntimeError(
                "live v2 reset requires authenticated GET /v2/env/state; endpoint is unavailable"
            )
        payload = _normalize_v2_training_payload(
            method(),
            surface="environment state",
        )
        if payload.get("ok") is not True or str(payload.get("capability") or "") != "training":
            raise EnvironmentCommandError(
                {
                    "status": "rejected_before_execution",
                    "error": {"message": "v2 state response is not training-authenticated"},
                }
            )
        revision = payload.get("state_version")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise EnvironmentCommandError(
                {
                    "status": "rejected_before_execution",
                    "error": {"message": "v2 state response has no valid state_version"},
                }
            )
        return revision

    @staticmethod
    def _validate_reset_revision(
        requested: int | None,
        current: int,
    ) -> int:
        if requested is None:
            raise EnvironmentCommandError(
                {
                    "status": "rejected_before_execution",
                    "error": {"message": "typed reset request is missing expected_state_version"},
                }
            )
        if requested != current:
            raise EnvironmentCommandError(
                {
                    "status": "rejected_before_execution",
                    "error": {
                        "message": (
                            f"reset revision is stale: expected_state_version={requested}, "
                            f"current_state_version={current}"
                        ),
                        "details": {
                            "expected_state_version": requested,
                            "current_state_version": current,
                        },
                    },
                }
            )
        return current

    def reset(self, request: ResetRequest) -> EnvironmentResult:
        self._ensure_request_session(request.session_id)
        self._ensure_open()
        method = getattr(self.client, "reset_v2", None)
        if callable(method):
            options = request.to_v2_options()
            try:
                current_revision = self._authenticated_v2_state_version()
                expected_state_version = self._validate_reset_revision(
                    request.expected_state_version,
                    current_revision,
                )
                envelope = method(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    expected_state_version=expected_state_version,
                    scenario=request.scenario,
                    seed=request.seed,
                    options=options,
                    timeout_ms=request.timeout_ms,
                )
                return EnvironmentResult.from_legacy(
                    _unwrap_v2_result(envelope, expected_request_id=request.request_id)
                )
            except Exception as exc:
                if not self._is_missing_v2(exc):
                    raise
        return self._legacy_call("reset", request.to_legacy_kwargs())

    def step(self, request: StepRequest) -> EnvironmentResult:
        self._ensure_request_session(request.session_id)
        self._ensure_open()
        method = getattr(self.client, "step_v2", None)
        if callable(method):
            action: dict[str, Any]
            if request.action_index is not None:
                action = {"action_index": request.action_index}
            elif request.action_id is not None:
                action = {"action_handle": request.action_id}
            else:
                raise EnvironmentCommandError(
                    {
                        "status": "rejected",
                        "error": {"message": "step request has no action"},
                    }
                )
            action["timeout_ms"] = request.timeout_ms
            try:
                envelope = method(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    episode_id=request.episode_id,
                    expected_step_index=request.expected_step_index,
                    action=action,
                    timeout_ms=request.timeout_ms,
                )
                return EnvironmentResult.from_legacy(
                    _unwrap_v2_result(envelope, expected_request_id=request.request_id)
                )
            except Exception as exc:
                if not self._is_missing_v2(exc):
                    raise
        return self._legacy_call("step", request.to_legacy_kwargs())

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult:
        self._ensure_request_session(request.session_id)
        self._ensure_open()
        method = getattr(self.client, "reset_v2", None)
        if callable(method):
            options = request.to_v2_options()
            seed = request.seed
            try:
                current_revision = self._authenticated_v2_state_version()
                expected_state_version = self._validate_reset_revision(
                    request.expected_state_version,
                    current_revision,
                )
                envelope = method(
                    request_id=request.request_id,
                    session_id=request.session_id,
                    expected_state_version=expected_state_version,
                    scenario="combat",
                    seed=seed,
                    options=options,
                    timeout_ms=request.timeout_ms,
                )
                return EnvironmentResult.from_legacy(
                    _unwrap_v2_result(envelope, expected_request_id=request.request_id)
                )
            except Exception as exc:
                if not self._is_missing_v2(exc):
                    raise
        return self._legacy_call("combat_reset", request.to_legacy_kwargs())
