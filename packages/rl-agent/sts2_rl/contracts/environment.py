"""Typed environment contracts aligned with contracts/environment.schema.json."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

from .versions import API_VERSION, SCHEMA_VERSION

ENVIRONMENT_CONTRACT_VERSION = API_VERSION
ENVIRONMENT_SCHEMA_VERSION = SCHEMA_VERSION
JsonObject = dict[str, Any]


def _new_request_id() -> str:
    return str(uuid4())


def _require_text(name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{name} is required")


def _require_uuid(name: str, value: str) -> None:
    _require_text(name, value)
    try:
        UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be a UUID") from exc


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend_name: str
    session_id: str
    contract_version: str = ENVIRONMENT_CONTRACT_VERSION
    schema_version: str = ENVIRONMENT_SCHEMA_VERSION
    supports_full_run: bool = True
    supports_combat_reset: bool = True
    supports_seed: bool = True
    privileged_training: bool = True

    def __post_init__(self) -> None:
        _require_text("backend_name", self.backend_name)
        _require_text("session_id", self.session_id)


@dataclass(frozen=True, slots=True)
class ResetRequest:
    request_id: str
    session_id: str
    scenario: str
    # Required by contract-v2. ``None`` is reserved for the explicit legacy
    # adapter; LiveBackend refuses to issue a v2 mutation without a revision
    # read from the authenticated /v2/env/state surface.
    expected_state_version: int | None = None
    character: str | None = None
    seed: str | int | None = None
    rebind_active_run: bool = False
    force_fresh: bool = False
    defensive_buffs: bool = False
    timeout_ms: int = 45_000

    def __post_init__(self) -> None:
        _require_uuid("request_id", self.request_id)
        _require_text("session_id", self.session_id)
        if self.scenario not in {"full-run", "combat"}:
            raise ValueError("scenario must be full-run or combat")
        if self.expected_state_version is not None and self.expected_state_version < 0:
            raise ValueError("expected_state_version must be non-negative")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

    @classmethod
    def from_legacy(
        cls,
        *,
        session_id: str = "legacy-local",
        scenario: str = "full-run",
        request_id: str | None = None,
        **kwargs: Any,
    ) -> ResetRequest:
        return cls(
            request_id=request_id or _new_request_id(),
            session_id=session_id,
            scenario=scenario,
            **kwargs,
        )

    def to_v2_options(self) -> JsonObject:
        return {
            "character": self.character,
            "rebind_active_run": self.rebind_active_run,
            "force_fresh": self.force_fresh,
            "defensive_buffs": self.defensive_buffs,
            "timeout_ms": self.timeout_ms,
        }

    def to_legacy_kwargs(self) -> JsonObject:
        if self.scenario != "full-run":
            raise ValueError("combat scenario must use CombatResetRequest on the legacy API")
        return {
            "character": self.character,
            "seed": self.seed,
            "rebind_active_run": self.rebind_active_run,
            "force_fresh": self.force_fresh,
            "defensive_buffs": self.defensive_buffs,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True, slots=True)
class StepRequest:
    request_id: str
    session_id: str
    episode_id: str
    expected_step_index: int
    action_index: int | None = None
    action_id: str | None = None
    timeout_ms: int = 20_000

    def __post_init__(self) -> None:
        _require_uuid("request_id", self.request_id)
        _require_text("session_id", self.session_id)
        _require_text("episode_id", self.episode_id)
        if (self.action_index is None) == (self.action_id is None):
            raise ValueError("provide exactly one of action_index or action_id")
        if self.action_index is not None and self.action_index < 0:
            raise ValueError("action_index must be non-negative")
        if self.expected_step_index < 0:
            raise ValueError("expected_step_index must be non-negative")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

    @classmethod
    def from_legacy(
        cls,
        *,
        episode_id: str,
        expected_step_index: int,
        session_id: str = "legacy-local",
        request_id: str | None = None,
        action_index: int | None = None,
        action_id: str | None = None,
        timeout_ms: int = 20_000,
    ) -> StepRequest:
        return cls(
            request_id=request_id or _new_request_id(),
            session_id=session_id,
            episode_id=episode_id,
            expected_step_index=expected_step_index,
            action_index=action_index,
            action_id=action_id,
            timeout_ms=timeout_ms,
        )

    def to_legacy_kwargs(self) -> JsonObject:
        return {
            "episode_id": self.episode_id,
            "action_index": self.action_index,
            "action_id": self.action_id,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True, slots=True)
class CombatResetRequest:
    request_id: str
    session_id: str
    # Combat reset is the combat scenario of the same v2 reset contract.
    expected_state_version: int | None = None
    character: str | None = None
    encounter_id: str | None = None
    seed: int | None = None
    current_hp: int | None = None
    max_hp: int | None = None
    max_energy: int | None = None
    deck: tuple[str, ...] | None = None
    deck_entries: tuple[Mapping[str, Any], ...] | None = None
    relics: tuple[str, ...] | None = None
    additional_relics: tuple[str, ...] | None = None
    potions: tuple[str, ...] | None = None
    gold: int | None = None
    timeout_ms: int = 15_000

    def __post_init__(self) -> None:
        _require_uuid("request_id", self.request_id)
        _require_text("session_id", self.session_id)
        if self.expected_state_version is not None and self.expected_state_version < 0:
            raise ValueError("expected_state_version must be non-negative")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        for name in ("current_hp", "max_hp", "max_energy", "gold"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_legacy(
        cls,
        *,
        session_id: str = "legacy-local",
        request_id: str | None = None,
        **kwargs: Any,
    ) -> CombatResetRequest:
        return cls(request_id=request_id or _new_request_id(), session_id=session_id, **kwargs)

    def to_v2_options(self) -> JsonObject:
        return {
            "character": self.character,
            "encounter_id": self.encounter_id,
            "current_hp": self.current_hp,
            "max_hp": self.max_hp,
            "max_energy": self.max_energy,
            "deck": list(self.deck) if self.deck is not None else None,
            "deck_entries": [dict(entry) for entry in self.deck_entries] if self.deck_entries is not None else None,
            "relics": list(self.relics) if self.relics is not None else None,
            "additional_relics": (
                list(self.additional_relics)
                if self.additional_relics is not None
                else None
            ),
            "potions": list(self.potions) if self.potions is not None else None,
            "gold": self.gold,
            "timeout_ms": self.timeout_ms,
        }

    def to_legacy_kwargs(self) -> JsonObject:
        return {
            "character": self.character,
            "encounter_id": self.encounter_id,
            "seed": self.seed,
            "current_hp": self.current_hp,
            "max_hp": self.max_hp,
            "max_energy": self.max_energy,
            "deck": list(self.deck) if self.deck is not None else None,
            "deck_entries": [dict(entry) for entry in self.deck_entries] if self.deck_entries is not None else None,
            "relics": list(self.relics) if self.relics is not None else None,
            "additional_relics": (
                list(self.additional_relics)
                if self.additional_relics is not None
                else None
            ),
            "potions": list(self.potions) if self.potions is not None else None,
            "gold": self.gold,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentTransition:
    episode_id: str
    step_index: int
    before_state_version: int
    after_state_version: int
    facts: JsonObject

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> EnvironmentTransition:
        facts = payload.get("facts")
        if not isinstance(facts, Mapping):
            raise ValueError("environment transition requires a facts object")
        transition = cls(
            episode_id=str(payload.get("episode_id") or ""),
            step_index=int(payload.get("step_index", 0)),
            before_state_version=int(payload.get("before_state_version", 0)),
            after_state_version=int(payload.get("after_state_version", 0)),
            facts=dict(facts),
        )
        _require_text("transition.episode_id", transition.episode_id)
        if min(
            transition.step_index,
            transition.before_state_version,
            transition.after_state_version,
        ) < 0:
            raise ValueError("transition indexes/versions must be non-negative")
        return transition

    def to_mapping(self) -> JsonObject:
        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "before_state_version": self.before_state_version,
            "after_state_version": self.after_state_version,
            "facts": dict(self.facts),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentResult:
    episode_id: str
    step_index: int
    observation: JsonObject
    legal_actions: tuple[JsonObject, ...] = ()
    transition: EnvironmentTransition | None = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    terminal_reason: str | None = None
    info: JsonObject = field(default_factory=dict)
    ok: bool = True
    raw: JsonObject = field(default_factory=dict, repr=False)

    @classmethod
    def from_legacy(cls, payload: Mapping[str, Any]) -> EnvironmentResult:
        if not isinstance(payload, Mapping):
            raise TypeError(f"environment response must be a mapping, got {type(payload).__name__}")
        obs = payload.get("obs", payload.get("observation"))
        if not isinstance(obs, Mapping):
            obs = {}
        actions = payload.get("legal_actions")
        if not isinstance(actions, list | tuple):
            actions = ()
        info = payload.get("info")
        if not isinstance(info, Mapping):
            info = {}
        transition_payload = payload.get("transition")
        if not isinstance(transition_payload, Mapping):
            alternate = payload.get("transition_facts")
            transition_payload = alternate if isinstance(alternate, Mapping) and isinstance(alternate.get("facts"), Mapping) else None
        reward_authority = str(payload.get("reward_authority") or info.get("reward_authority") or "")
        if reward_authority == "external-rl" and transition_payload is None:
            raise ValueError("external-rl environment result requires a typed transition")
        transition = (
            EnvironmentTransition.from_mapping(transition_payload)
            if isinstance(transition_payload, Mapping)
            else None
        )
        return cls(
            episode_id=str(payload.get("episode_id") or ""),
            step_index=int(payload.get("step_index", 0) or 0),
            observation=dict(obs),
            legal_actions=tuple(dict(item) for item in actions if isinstance(item, Mapping)),
            transition=transition,
            reward=float(payload.get("reward", 0.0) or 0.0),
            terminated=bool(payload.get("done", payload.get("terminated", False))),
            truncated=bool(payload.get("truncated", False)),
            terminal_reason=payload.get("terminal_reason"),
            info=dict(info),
            ok=bool(payload.get("ok", True)),
            raw=payload if isinstance(payload, dict) else dict(payload),
        )

    def to_legacy(self) -> JsonObject:
        # Preserve identity for explicit legacy compatibility adapters.  V2
        # envelopes are already unwrapped into a private result dict, so this
        # does not mutate the caller's command envelope.
        payload = self.raw
        payload.update(
            {
                "ok": self.ok,
                "episode_id": self.episode_id,
                "step_index": self.step_index,
                "reward": self.reward,
                "done": self.terminated,
                "truncated": self.truncated,
                "terminal_reason": self.terminal_reason,
                "obs": dict(self.observation),
                "legal_actions": [dict(action) for action in self.legal_actions],
                "info": dict(self.info),
            }
        )
        if self.transition is not None:
            transition = self.transition.to_mapping()
            payload["transition"] = transition
            payload["transition_facts"] = dict(transition)
        return payload


@runtime_checkable
class EnvironmentBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    @property
    def session_id(self) -> str: ...

    @property
    def is_connected(self) -> bool: ...

    def health(self) -> JsonObject: ...

    def get_spec(self) -> JsonObject: ...

    def get_state(self) -> JsonObject: ...

    def reset(self, request: ResetRequest) -> EnvironmentResult: ...

    def step(self, request: StepRequest) -> EnvironmentResult: ...

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult: ...

    def close(self) -> None: ...
