"""Shared environment lifecycle and canonical-transition migration seam."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .action_compact import compact_legal_actions
from .observation_v2 import MAX_ACTIONS
from .reward_constants import ENEMY_HP_SENTINEL_THRESHOLD

from sts2_rl.backends import LegacyClientBackend
from sts2_rl.contracts import (
    CombatResetRequest,
    EnvironmentBackend,
    EnvironmentResult,
    ResetRequest,
    StepRequest,
)
from sts2_rl.reward import (
    CanonicalTransition,
    LegacyRewardCalculator,
    RewardBreakdown,
    VersionedRewardCalculator,
    canonicalize_legacy_transition,
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class EnvironmentRuntimeMixin:
    """Common lifecycle behavior for full-run and combat Gym environments."""

    bridge: Any
    backend: EnvironmentBackend

    def _initialize_environment_runtime(
        self,
        *,
        backend: EnvironmentBackend | None = None,
        bridge: Any | None = None,
        backend_name: str | None = None,
    ) -> None:
        """Install the typed backend boundary used by every environment mutation.

        ``bridge`` remains a compatibility attribute for diagnostics and old
        tests, but reset/step/combat-reset always flow through ``self.backend``.
        The canonical MuZero factory injects a strict ``LiveBackend`` or a
        ``HeadlessBackend``; direct legacy clients are wrapped explicitly here.
        """
        if backend is not None and bridge is not None:
            raise ValueError("provide backend or bridge, not both")
        if backend is None:
            raw_bridge = bridge if bridge is not None else getattr(self, "bridge", None)
            if raw_bridge is None:
                raise ValueError("environment runtime requires a backend or bridge")
            inferred_name = backend_name or (
                "headless_sim"
                if str(getattr(raw_bridge, "base_url", "")).startswith("headless_sim://")
                else "legacy_bridge"
            )
            backend = LegacyClientBackend(
                raw_bridge,
                backend_name=inferred_name,
                session_id=str(getattr(raw_bridge, "session_id", "") or f"{inferred_name}-session"),
            )
        self.backend = backend
        # Preserve the historical public attribute without using it for
        # mutations.  This is useful for catalog probes and compatibility tests.
        self.bridge = getattr(backend, "client", backend)
        self._backend_step_index = 0
        self._environment_close_lock = threading.Lock()
        self._environment_closed = False
        self._last_canonical_transition: CanonicalTransition | None = None
        self._canonical_reward_calculator = LegacyRewardCalculator()

    def _backend_for_call(self) -> EnvironmentBackend:
        """Return the installed backend, lazily wrapping legacy test stubs.

        A number of focused unit tests construct environments via ``__new__``
        and assign ``env.bridge`` directly.  Supporting that harness does not
        weaken the canonical factory, which always injects a strict backend.
        """
        backend = getattr(self, "backend", None)
        raw_bridge = getattr(self, "bridge", None)
        if backend is not None:
            backend_client = getattr(backend, "client", backend)
            if raw_bridge is None or backend_client is raw_bridge:
                return backend
        if raw_bridge is None:
            raise RuntimeError("environment has no backend")
        inferred_name = (
            "headless_sim"
            if str(getattr(raw_bridge, "base_url", "")).startswith("headless_sim://")
            else "legacy_bridge"
        )
        backend = LegacyClientBackend(
            raw_bridge,
            backend_name=inferred_name,
            session_id=str(getattr(raw_bridge, "session_id", "") or f"{inferred_name}-session"),
        )
        self.backend = backend
        if not hasattr(self, "_backend_step_index"):
            self._backend_step_index = 0
        return backend

    def _backend_reset(self, **kwargs: Any) -> dict[str, Any]:
        backend = self._backend_for_call()
        request = ResetRequest.from_legacy(
            session_id=backend.session_id,
            scenario="full-run",
            expected_state_version=self._reset_state_version(backend),
            **kwargs,
        )
        result = backend.reset(request)
        self._backend_step_index = int(result.step_index)
        return result.to_legacy()

    def _backend_combat_reset(self, **kwargs: Any) -> dict[str, Any]:
        backend = self._backend_for_call()
        normalized = dict(kwargs)
        for key in ("deck", "relics", "potions"):
            value = normalized.get(key)
            if isinstance(value, list):
                normalized[key] = tuple(value)
        entries = normalized.get("deck_entries")
        if isinstance(entries, list):
            normalized["deck_entries"] = tuple(entries)
        request = CombatResetRequest.from_legacy(
            session_id=backend.session_id,
            expected_state_version=self._reset_state_version(backend),
            **normalized,
        )
        result = backend.combat_reset(request)
        self._backend_step_index = int(result.step_index)
        return result.to_legacy()

    @staticmethod
    def _reset_state_version(backend: EnvironmentBackend) -> int | None:
        """Read the revision used to authorize a typed reset.

        The explicit LegacyClientBackend migration seam has no v2 revision
        contract and never serializes this field. Every canonical backend must
        publish a non-negative integer revision; absence is a hard failure and
        is never replaced with a guessed zero.
        """

        if type(backend) is LegacyClientBackend:
            return None
        state = backend.get_state()
        if not isinstance(state, dict):
            raise RuntimeError("typed backend state response must be an object")
        revision = state.get("state_version")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise RuntimeError(
                "typed backend state response requires a non-negative integer state_version"
            )
        return revision

    def _backend_step(
        self,
        *,
        episode_id: str,
        action_id: str | None,
        timeout_ms: int,
        action_index: int | None = None,
    ) -> dict[str, Any]:
        normalized_action_id = str(action_id).strip() if action_id is not None else ""
        request = StepRequest.from_legacy(
            session_id=self._backend_for_call().session_id,
            episode_id=str(episode_id),
            expected_step_index=int(getattr(self, "_backend_step_index", 0)),
            action_id=normalized_action_id or None,
            action_index=None if normalized_action_id else action_index,
            timeout_ms=int(timeout_ms),
        )
        result = self._backend_for_call().step(request)
        self._backend_step_index = int(result.step_index)
        return result.to_legacy()

    @staticmethod
    def _uses_external_v2_reward(payload: dict[str, Any]) -> bool:
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        authority = str(
            payload.get("reward_authority")
            or info.get("reward_authority")
            or ""
        ).strip().lower()
        transition = payload.get("transition") or payload.get("transition_facts")
        return authority == "external-rl" and isinstance(transition, dict)

    def _record_backend_transition(
        self,
        payload: dict[str, Any],
        *,
        action_handle: str | None,
    ) -> RewardBreakdown:
        capabilities = getattr(getattr(self, "backend", None), "capabilities", None)
        backend_name = str(getattr(capabilities, "backend_name", "") or "legacy_bridge")
        transition = canonicalize_legacy_transition(
            payload,
            action_handle=action_handle,
            backend_name=backend_name,
        )
        self._last_canonical_transition = transition
        calculator: VersionedRewardCalculator
        if self._uses_external_v2_reward(payload):
            calculator = VersionedRewardCalculator()
        else:
            calculator = LegacyRewardCalculator()
        self._canonical_reward_calculator = calculator
        breakdown = calculator.evaluate(transition)
        info = payload.get("info")
        if not isinstance(info, dict):
            info = {}
            payload["info"] = info
        info["canonical_reward"] = {
            "total": breakdown.total,
            "components": dict(breakdown.components),
            "spec_version": breakdown.spec_version,
            "spec_fingerprint": breakdown.spec_fingerprint,
            "authority": "external-rl" if self._uses_external_v2_reward(payload) else "legacy-backend",
        }
        return breakdown

    @property
    def last_canonical_transition(self) -> CanonicalTransition | None:
        return self._last_canonical_transition

    def _close_environment_runtime(self) -> None:
        lock = getattr(self, "_environment_close_lock", None)
        if lock is None:
            return
        with lock:
            if getattr(self, "_environment_closed", False):
                return
            self._environment_closed = True
            target = getattr(self, "backend", None) or getattr(self, "bridge", None)
            close = getattr(target, "close", None)
            if callable(close):
                close()

    def action_masks(self) -> np.ndarray:
        from .hp_cost_safety import is_self_lethal_action  # noqa: WPS433

        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        raw_obs = self._last_obs_raw if isinstance(self._last_obs_raw, dict) else None
        for i, action in enumerate(self._legal_actions[:MAX_ACTIONS]):
            if not isinstance(action, dict):
                continue
            # P0-1: hard-mask self-lethal HP-cost actions in the run-mode env too.
            if is_self_lethal_action(action, raw_obs):
                continue
            mask[i] = True
        return mask
    def get_compact_legal_actions(self) -> list[dict[str, Any]]:
        return compact_legal_actions(self._legal_actions)
    def _normalize_action(self, action: Any) -> int | None:
        try:
            normalized = int(action)
        except (TypeError, ValueError):
            return None
        if normalized < 0:
            return None
        return normalized
    def _combat_enemy_total_hp(self, obs: dict[str, Any] | None) -> float:
        if not isinstance(obs, dict):
            return 0.0
        combat = obs.get("combat")
        if not isinstance(combat, dict):
            return 0.0
        enemies = combat.get("enemies")
        if not isinstance(enemies, list):
            return 0.0

        total = 0.0
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            hp = float(enemy.get("hp", enemy.get("current_hp")) or 0.0)
            if hp > ENEMY_HP_SENTINEL_THRESHOLD:
                continue
            total += hp
        return total
    @staticmethod
    def _source_preview_metric(source: dict[str, Any] | None, key: str) -> float:
        if not isinstance(source, dict):
            return 0.0
        effect_preview = source.get("effect_preview")
        if isinstance(effect_preview, dict) and effect_preview.get(key) is not None:
            try:
                return float(effect_preview.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(source.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    @staticmethod
    def _is_end_turn_action(action: dict[str, Any] | None) -> bool:
        if not isinstance(action, dict):
            return False
        action_id = str(action.get("action_id") or "").strip().lower()
        kind = str(action.get("kind") or "").strip().lower()
        family = ""
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        if isinstance(semantic, dict):
            family = str(semantic.get("family") or semantic.get("action_kind") or "").strip().lower()
        return action_id == "end_turn" or kind == "end_turn" or family == "end_turn"
    @staticmethod
    def _raw_card_is_affordable_combat_action(card: Any, energy: float) -> bool:
        if not isinstance(card, dict):
            return False

        card_type = str(card.get("type") or card.get("card_type") or "").strip().lower()
        card_id = str(card.get("id") or card.get("card_id") or "").strip().lower()
        title = str(card.get("title") or card.get("name") or "").strip().lower()
        blocked_type_fragments = ("status", "curse", "quest")
        if card_type in blocked_type_fragments:
            return False
        if any(fragment in card_id for fragment in blocked_type_fragments):
            return False
        if any(fragment in title for fragment in ("晕眩", "伤口", "灼伤", "虚无", "诅咒")):
            return False

        playable = card.get("is_playable")
        if playable is False:
            return False
        if playable is True:
            return True

        # Missing card type is common in compact bridge payloads/tests.  Treat
        # explicit combat card types as valid, and allow unknown type only if a
        # finite non-negative energy cost is present.
        if card_type and card_type not in {"attack", "skill", "power"}:
            return False

        cost: float | None = None
        for key in (
            "cost_for_turn",
            "resolved_energy_cost",
            "energy_cost",
            "canonical_energy_cost",
            "cost",
        ):
            if card.get(key) is None:
                continue
            parsed = _float(card.get(key), float("nan"))
            if parsed == parsed:
                cost = parsed
                break
        if cost is None:
            return False
        if cost < 0.0:
            # X-cost or special-cost cards may be playable, but they are not a
            # high-confidence proof that EndTurn is stale unless the bridge
            # explicitly sets is_playable=True.
            return False
        return cost <= energy + 1e-6
    def _safe_get_state(self) -> dict[str, Any] | None:
        try:
            state = self._backend_for_call().get_state()
        except Exception:
            return None
        return state if isinstance(state, dict) else None
    @staticmethod
    def _state_available_actions(state: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(state, dict):
            return []
        actions = state.get("available_actions")
        if not isinstance(actions, list):
            actions = state.get("legal_actions")
        if not isinstance(actions, list):
            return []
        return [action for action in actions if isinstance(action, dict)]
    def _planner_context(self) -> dict[str, Any]:
        context = self._run_memory.build_context(self._last_obs_raw, self._legal_actions)
        context["combat_memory"] = self._combat_memory.snapshot()
        return context
    @staticmethod
    def _python_timing(
        *,
        bridge_roundtrip: float,
        run_memory_update: float,
        obs_encode: float,
        aux_targets: float,
        info_build: float,
        total: float,
    ) -> dict[str, float]:
        return {
            "bridge_roundtrip": float(bridge_roundtrip),
            "run_memory_update": float(run_memory_update),
            "obs_encode": float(obs_encode),
            "aux_targets": float(aux_targets),
            "info_build": float(info_build),
            "total": float(total),
        }
