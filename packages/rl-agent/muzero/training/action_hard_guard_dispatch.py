"""Post-search hard-guard dispatch for self-play.

This module intentionally contains only wiring: it decides which existing guard
family should inspect the selected action after search/direct-policy selection.
The guard implementations themselves live under ``muzero.combat_quality`` or in
legacy trainer helpers while ``train.py`` is being decomposed.
"""

from __future__ import annotations

from typing import Any

from sts2_env.boss_mechanics import build_boss_mechanics_context


class ActionHardGuardDispatchMixin:
    """Route post-search selections through domain-specific hard guards.

    The dispatcher is fail-open by design.  A guard bug must not crash self-play;
    it should leave the chosen action unchanged and surface a small telemetry
    flag in ``search_stats`` for later diagnosis.
    """

    def _apply_post_search_action_hard_guards(
        self,
        *,
        decision_domain: str,
        action_idx: int,
        legal_actions: list[Any] | None,
        action_mask: Any,
        obs: dict[str, Any] | None,
        info: dict[str, Any] | None,
        search_stats: dict[str, Any],
    ) -> int:
        """Return the final action index after domain hard guards.

        ``obs`` is the encoded model observation and is currently only kept in
        the signature so future dispatchers do not need to grow ad-hoc imports.
        Combat guards read the raw bridge observation through trainer/env helper
        methods because encoded observations intentionally omit some mechanics.
        """

        del obs  # encoded obs is not needed by the current guard families.
        domain = str(decision_domain or "").strip().lower()
        stats = search_stats if isinstance(search_stats, dict) else {}
        try:
            idx = int(action_idx)
        except Exception:
            return action_idx

        if domain == "build":
            policy = str(getattr(self, "build_hard_guard_policy", "off") or "off").strip().lower()
            if policy not in {"full", "emergency"}:
                stats["build_hard_guard_policy_off"] = 1.0
                return idx
            guard = getattr(self, "_apply_build_action_hard_guards", None)
            if not callable(guard):
                return idx
            try:
                return int(
                    guard(
                        action_idx=idx,
                        legal_actions=legal_actions,
                        action_mask=action_mask,
                        search_stats=stats,
                    )
                )
            except Exception:
                stats["action_hard_guard_dispatch_error"] = 1.0
                return idx

        if domain == "route":
            if not bool(getattr(self, "route_safety_guard_enabled", False)):
                return idx
            guard = getattr(self, "_apply_route_action_hard_guards", None)
            if not callable(guard):
                return idx
            try:
                return int(
                    guard(
                        action_idx=idx,
                        legal_actions=legal_actions,
                        action_mask=action_mask,
                        search_stats=stats,
                    )
                )
            except Exception:
                stats["action_hard_guard_dispatch_error"] = 1.0
                return idx

        if domain == "combat":
            policy = str(getattr(self, "combat_hard_guard_policy", "off") or "off").strip().lower()
            if policy not in {"full", "emergency"}:
                stats["combat_hard_guard_policy_off"] = 1.0
                return idx
            guard = getattr(self, "_apply_combat_action_hard_guards", None)
            if not callable(guard):
                return idx
            try:
                raw_obs = None
                raw_getter = getattr(self, "_current_raw_combat_obs", None)
                if callable(raw_getter):
                    raw_obs = raw_getter()
                if not isinstance(raw_obs, dict):
                    env_unwrap = getattr(getattr(self, "env", None), "unwrapped", getattr(self, "env", None))
                    raw_obs = getattr(env_unwrap, "_last_obs_raw", None)
                boss_ctx = build_boss_mechanics_context(raw_obs) if isinstance(raw_obs, dict) else {}
                encounter = self._hard_guard_encounter_key(raw_obs, boss_ctx, info)
                return int(
                    guard(
                        action_idx=idx,
                        legal_actions=legal_actions,
                        action_mask=action_mask,
                        raw_obs=raw_obs if isinstance(raw_obs, dict) else None,
                        boss_ctx=boss_ctx,
                        encounter=encounter,
                        search_stats=stats,
                    )
                )
            except Exception:
                stats["action_hard_guard_dispatch_error"] = 1.0
                return idx

        return idx

    @staticmethod
    def _hard_guard_encounter_key(
        raw_obs: Any,
        boss_ctx: Any,
        info: dict[str, Any] | None,
    ) -> str:
        """Best-effort encounter id used by combat guards and metrics."""

        candidates: list[Any] = []
        if isinstance(boss_ctx, dict):
            candidates.extend(
                boss_ctx.get(key)
                for key in (
                    "encounter_key",
                    "encounter_id",
                    "boss_key",
                    "boss_id",
                    "name",
                )
            )
        if isinstance(info, dict):
            candidates.extend(
                info.get(key)
                for key in (
                    "encounter_id",
                    "encounter",
                    "room_model",
                    "roomModel",
                    "tier",
                )
            )
        if isinstance(raw_obs, dict):
            candidates.extend(
                raw_obs.get(key)
                for key in (
                    "encounter_id",
                    "encounter",
                    "room_model",
                    "roomModel",
                )
            )
            run = raw_obs.get("run") if isinstance(raw_obs.get("run"), dict) else {}
            if isinstance(run, dict):
                candidates.extend(
                    run.get(key)
                    for key in (
                        "encounter_id",
                        "encounter",
                        "room_model",
                        "roomModel",
                    )
                )
        for value in candidates:
            text = str(value or "").strip()
            if text:
                return text
        return ""


__all__ = ["ActionHardGuardDispatchMixin"]
