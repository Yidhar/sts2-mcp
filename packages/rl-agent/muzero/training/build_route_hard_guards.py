"""Build/route post-search hard guards for MuZero self-play.

This module owns non-combat hard-guard wiring that used to live in
``muzero.train``:

* build/campfire safety: force REST/HEAL instead of SMITH/other when HP is
  below the environment's low-HP threshold and a concrete heal option is legal;
* route safety: optional Act1 recovery guard that avoids high-risk elite paths
  when a lower-risk route-summary-scored alternative is currently legal.

The policy/scoring primitives remain in ``sts2_env.route_heuristic`` and the
trainer only provides observation/action adapters plus telemetry counters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from muzero.training.card_reward_guard import apply_card_reward_guard, card_reward_guard_metric_keys
from muzero.training.card_reward_pick_quality_guard import (
    apply_card_reward_pick_quality_guard,
    card_reward_pick_quality_guard_metric_keys,
)
from muzero.training.deck_upgrade_target_guard import (
    apply_deck_upgrade_target_guard,
    deck_upgrade_target_guard_metric_keys,
)
from muzero.training.rest_site_smith_guard import (
    apply_rest_site_smith_guard,
    rest_site_smith_guard_metric_keys,
)
from muzero.training.shop_action_guard import apply_shop_action_guard, shop_action_guard_metric_keys
from sts2_env.observation_v2 import MAX_ACTIONS
from sts2_env.reward_constants import REST_SITE_SKIP_HEAL_HP_THRESHOLD


class BuildRouteHardGuardMixin:
    """Mixin implementing build and route hard-guard dispatch targets."""

    @staticmethod
    def _route_safety_guard_metric_keys() -> tuple[str, ...]:
        return (
            "route_safety_guard_enabled",
            "route_safety_guard_applicable",
            "route_safety_guard_safe_available",
            "route_safety_guard_lower_risk_available",
            "route_safety_guard_applied",
            "route_safety_guard_override",
            "route_safety_guard_alignment_error",
            "route_safety_guard_selected_risk_class",
            "route_safety_guard_final_risk_class",
            "route_safety_guard_selected_forced_elite",
            "route_safety_guard_selected_immediate_elite",
            "route_safety_guard_low_hp_forced",
            "route_safety_guard_invalid_obs",
        )

    @staticmethod
    def _route_actions_alignment_error(
        compact_actions: list[Any] | None,
        full_actions: list[Any] | None,
        *,
        max_index: int,
    ) -> bool:
        """Best-effort positional-alignment check for compact vs full route actions.

        ``legal_actions`` in the trainer is often a compacted copy while the env
        wrapper keeps the full bridge action (with ``route_summary``) in
        ``_legal_actions``.  The guard uses full actions for route scoring but
        ultimately sends the compact-selected index to the env, so it must not
        override if the two arrays appear out of sync.

        If no comparable stable fields are present we trust index order (the
        bridge contract); if comparable fields are present and any disagree we
        report an alignment error and the caller must no-op.
        """

        if not isinstance(compact_actions, list) or not isinstance(full_actions, list):
            return False
        if max_index < 0:
            return False
        if len(compact_actions) <= max_index or len(full_actions) <= max_index:
            return True
        stable_keys = (
            "kind",
            "action_id",
            "target_index",
            "choice_index",
            "index",
            "node_id",
            "map_node_id",
            "x",
            "y",
            "col",
            "row",
            "floor",
        )
        for idx in range(max_index + 1):
            compact = compact_actions[idx]
            full = full_actions[idx]
            if not isinstance(compact, dict) or not isinstance(full, dict):
                continue
            compared = False
            for key in stable_keys:
                cv = compact.get(key)
                fv = full.get(key)
                if cv is None or fv is None:
                    continue
                compared = True
                if str(cv) != str(fv):
                    return True
            # If both are dict route actions but have no shared comparable
            # stable fields, keep trusting the positional bridge contract.
            _ = compared
        return False

    @staticmethod
    def _build_safety_guard_metric_keys() -> tuple[str, ...]:
        return (
            "build_hard_guard_policy_full",
            "build_hard_guard_policy_emergency",
            "build_hard_guard_policy_off",
            "build_safety_guard_enabled",
            "build_safety_guard_rest_low_hp_applicable",
            "build_safety_guard_rest_available",
            "build_safety_guard_rest_applied",
            "build_safety_guard_rest_override",
            "build_safety_guard_rest_selected_non_heal_low_hp",
            "build_safety_guard_rest_selected_heal",
            "build_safety_guard_invalid_obs",
            "build_safety_guard_alignment_error",
            "build_safety_guard_hp_ratio",
            "build_safety_guard_hp_threshold",
        )

    @staticmethod
    def _normalized_action_text_for_guard(action: Any) -> str:
        if not isinstance(action, dict):
            return ""
        parts: list[str] = []
        containers = [action]
        payload = action.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)
        for container in containers:
            for key in (
                "action_id",
                "kind",
                "action_type",
                "label",
                "title",
                "name",
                "selection",
                "selection_action",
                "option_type",
                "canonical_text",
                "description",
            ):
                value = container.get(key)
                if value is not None:
                    parts.append(str(value))
            option = container.get("option") if isinstance(container.get("option"), dict) else {}
            for key in ("option_id", "id", "type", "option_type", "title", "label", "name", "description", "is_enabled"):
                value = option.get(key)
                if value is not None:
                    parts.append(str(value))
            semantic = container.get("semantic") if isinstance(container.get("semantic"), dict) else {}
            for key in ("family", "domain", "semantic_key"):
                value = semantic.get(key)
                if value is not None:
                    parts.append(str(value))
        return " ".join(parts).strip().lower()

    @classmethod
    def _is_rest_site_build_action(cls, action: Any) -> bool:
        """Return true for an actual campfire/rest-site option action.

        Deliberately excludes map-route actions whose *future path* contains a
        rest site.  This guard only overrides the within-campfire choice:
        REST/HEAL vs SMITH/other.
        """

        if not isinstance(action, dict):
            return False
        containers = [action]
        payload = action.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)
        for container in containers:
            action_id = str(container.get("action_id") or "").strip().lower()
            # Bridge terminal/proceed actions appear on the rest_site surface
            # after HEAL/SMITH has already resolved.  They must not trigger the
            # low-HP rest hard guard or be counted as non-heal choices.
            if action_id in {
                "rest_site:proceed",
                "rest_site:continue",
                "rest_site:leave",
                "rest_site:done",
                "rest_site:close",
            }:
                return False
            if action_id.startswith("rest_site:proceed"):
                return False
        for container in containers:
            kind = str(container.get("kind") or container.get("action_type") or "").strip().lower()
            if kind in {"rest", "rest_site", "choose_rest_option"}:
                return True
            semantic = container.get("semantic") if isinstance(container.get("semantic"), dict) else {}
            family = str(semantic.get("family") or "").strip().lower()
            domain = str(semantic.get("domain") or "").strip().lower()
            if domain == "build" and family in {"rest", "rest_site"}:
                return True
            action_id = str(container.get("action_id") or "").strip().lower()
            if action_id.startswith("rest_site:") or action_id.startswith("choose_rest_option:"):
                return True
            if action_id.startswith("sim:choose_rest_option"):
                return True
        return False

    @classmethod
    def _is_rest_heal_build_action(cls, action: Any) -> bool:
        """Detect the concrete HEAL/REST campfire option.

        Avoid the old bug: a substring match for "rest" in ``rest_site:smith``
        or "Rest Site" must NOT count as healing.
        """

        if not isinstance(action, dict):
            return False

        containers = [action]
        payload = action.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)

        option_identity_tokens: list[str] = []
        option_title_tokens: list[str] = []
        option_description_tokens: list[str] = []
        action_id_tokens: list[str] = []
        label_tokens: list[str] = []
        family_tokens: list[str] = []
        semantic_key_tokens: list[str] = []
        for container in containers:
            option = container.get("option") if isinstance(container.get("option"), dict) else {}
            for key in ("option_type", "type", "id", "option_id"):
                value = option.get(key)
                if value is not None:
                    option_identity_tokens.append(str(value).strip().lower())
            value = container.get("option_type")
            if value is not None:
                option_identity_tokens.append(str(value).strip().lower())
            for key in ("title", "label", "name"):
                value = option.get(key)
                if value is not None:
                    option_title_tokens.append(str(value).strip().lower())
            for key in ("title", "label", "name"):
                value = container.get(key)
                if value is not None:
                    label_tokens.append(str(value).strip().lower())
            for key in ("description",):
                value = option.get(key)
                if value is not None:
                    option_description_tokens.append(str(value).strip().lower())
                value = container.get(key)
                if value is not None:
                    option_description_tokens.append(str(value).strip().lower())
            action_id = container.get("action_id")
            if action_id is not None:
                action_id_tokens.append(str(action_id).strip().lower())
            semantic = container.get("semantic") if isinstance(container.get("semantic"), dict) else {}
            family = semantic.get("family")
            if family is not None:
                family_tokens.append(str(family).strip().lower())
            semantic_key = semantic.get("semantic_key")
            if semantic_key is not None:
                semantic_key_tokens.append(str(semantic_key).strip().lower())

        heal_exact = {
            "rest",
            "heal",
            "healing",
            "sleep",
            "campfire_rest",
            "rest_option",
            "heal_option",
            "healrestsiteoption",
            "mend",
            "mendrestsiteoption",
            "休息",
            "治疗",
            "恢復",
            "恢复",
        }
        if any(token in heal_exact for token in option_identity_tokens):
            return True
        if any(token in heal_exact for token in option_title_tokens):
            return True
        if any(family in {"rest_heal", "heal"} for family in family_tokens):
            return True
        if any(semantic_key.endswith("|rest") or semantic_key.endswith("|heal") for semantic_key in semantic_key_tokens):
            return True

        # Action id encodes the concrete option token in live bridge paths; only
        # boundary/exact matches are accepted.
        if any(
            action_id in {"rest", "heal"}
            or action_id.endswith(":rest")
            or action_id.endswith(":heal")
            or ":rest:" in action_id
            or ":heal:" in action_id
            or action_id.endswith("=rest")
            or action_id.endswith("=heal")
            or action_id.endswith("=rest_option")
            or action_id.endswith("=heal_option")
            for action_id in action_id_tokens
        ):
            return True

        # Titles/descriptions can be localized, but never treat the generic
        # surface label "Rest Site" as a heal.  Require a concrete heal token.
        positive_substrings = (
            "heal",
            "healing",
            "restore hp",
            "restore health",
            "recover hp",
            "recover health",
            "回复生命",
            "恢復生命",
            "恢复生命",
            "治疗",
        )
        title_blob = " ".join(option_title_tokens + label_tokens + option_description_tokens).strip()
        if any(token in title_blob for token in positive_substrings):
            return True
        if (
            ("回复" in title_blob or "恢復" in title_blob or "恢复" in title_blob or "治療" in title_blob)
            and ("生命" in title_blob or "hp" in title_blob or "health" in title_blob)
        ):
            return True
        # English/Chinese exact label for the concrete option is safe; generic
        # "rest site" is not.
        return title_blob in {"rest", "休息"}

    def _apply_build_action_hard_guards(
        self,
        *,
        action_idx: int,
        legal_actions: list[Any] | None,
        action_mask: Any,
        search_stats: dict[str, Any],
    ) -> int:
        """Low-HP campfire hard guard for the Act1 recovery run.

        When the agent is already at a campfire and HP is below the same
        threshold used by the environment reward, force the concrete HEAL/REST
        option if it is legal.  This does not change route planning and fails
        open on missing HP/action-contract issues.
        """

        for key in self._build_safety_guard_metric_keys():
            search_stats.setdefault(key, 0.0)
        for key in card_reward_guard_metric_keys():
            search_stats.setdefault(key, 0.0)
        for key in card_reward_pick_quality_guard_metric_keys():
            search_stats.setdefault(key, 0.0)
        for key in shop_action_guard_metric_keys():
            search_stats.setdefault(key, 0.0)
        for key in rest_site_smith_guard_metric_keys():
            search_stats.setdefault(key, 0.0)
        for key in deck_upgrade_target_guard_metric_keys():
            search_stats.setdefault(key, 0.0)

        policy = str(getattr(self, "build_hard_guard_policy", "full") or "full").strip().lower()
        if policy not in {"full", "emergency", "off"}:
            policy = "full"
        search_stats["build_hard_guard_policy_full"] = 1.0 if policy == "full" else 0.0
        search_stats["build_hard_guard_policy_emergency"] = 1.0 if policy == "emergency" else 0.0
        search_stats["build_hard_guard_policy_off"] = 1.0 if policy == "off" else 0.0
        search_stats["build_safety_guard_enabled"] = 0.0 if policy == "off" else 1.0
        search_stats["build_safety_guard_hp_threshold"] = float(REST_SITE_SKIP_HEAL_HP_THRESHOLD)

        if policy == "off":
            return int(action_idx)
        if not isinstance(legal_actions, list) or len(legal_actions) == 0:
            return action_idx
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        except Exception:
            return action_idx
        legal_count = min(len(legal_actions), int(mask_np.shape[0]) if mask_np.size else 0, MAX_ACTIONS)
        if not (0 <= int(action_idx) < legal_count) or mask_np[int(action_idx)] <= 0:
            return action_idx

        env_unwrap = getattr(self.env, "unwrapped", self.env)
        raw = getattr(env_unwrap, "_last_obs_raw", None)
        full_legal_actions = getattr(env_unwrap, "_legal_actions", None)

        if policy == "full":
            # Card-reward anti-skip must run before the campfire HP-specific path:
            # reward surfaces often do not expose HP cleanly, but they do expose the
            # deck needed to decide whether skipping is pathological.
            action_idx = apply_card_reward_guard(
                action_idx=int(action_idx),
                legal_actions=legal_actions,
                full_legal_actions=full_legal_actions if isinstance(full_legal_actions, list) else None,
                action_mask=action_mask,
                raw_obs=raw if isinstance(raw, dict) else None,
                search_stats=search_stats,
            )
            action_idx = apply_card_reward_pick_quality_guard(
                action_idx=int(action_idx),
                legal_actions=legal_actions,
                full_legal_actions=full_legal_actions if isinstance(full_legal_actions, list) else None,
                action_mask=action_mask,
                raw_obs=raw if isinstance(raw, dict) else None,
                search_stats=search_stats,
            )
            action_idx = apply_shop_action_guard(
                action_idx=int(action_idx),
                legal_actions=legal_actions,
                full_legal_actions=full_legal_actions if isinstance(full_legal_actions, list) else None,
                action_mask=action_mask,
                raw_obs=raw if isinstance(raw, dict) else None,
                search_stats=search_stats,
            )
            action_idx = apply_rest_site_smith_guard(
                action_idx=int(action_idx),
                legal_actions=legal_actions,
                full_legal_actions=full_legal_actions if isinstance(full_legal_actions, list) else None,
                action_mask=action_mask,
                raw_obs=raw if isinstance(raw, dict) else None,
                search_stats=search_stats,
            )
            action_idx = apply_deck_upgrade_target_guard(
                action_idx=int(action_idx),
                legal_actions=legal_actions,
                full_legal_actions=full_legal_actions if isinstance(full_legal_actions, list) else None,
                action_mask=action_mask,
                raw_obs=raw if isinstance(raw, dict) else None,
                search_stats=search_stats,
            )

        hp, max_hp, hp_valid = self._player_hp_values(raw)
        if not hp_valid:
            search_stats["build_safety_guard_invalid_obs"] = 1.0
            return action_idx
        hp_ratio = self._player_hp_ratio_from_values(hp, max_hp)
        search_stats["build_safety_guard_hp_ratio"] = hp_ratio
        if hp_ratio >= float(REST_SITE_SKIP_HEAL_HP_THRESHOLD):
            return action_idx

        action_source = full_legal_actions if isinstance(full_legal_actions, list) else legal_actions
        if not isinstance(action_source, list) or len(action_source) < legal_count:
            search_stats["build_safety_guard_alignment_error"] = 1.0
            return action_idx

        selected_action = action_source[int(action_idx)]
        selected_is_rest_site = self._is_rest_site_build_action(selected_action)
        if not selected_is_rest_site:
            return action_idx
        if self._is_rest_heal_build_action(selected_action):
            search_stats["build_safety_guard_rest_selected_heal"] = 1.0
            return action_idx

        search_stats["build_safety_guard_rest_low_hp_applicable"] = 1.0
        search_stats["build_safety_guard_rest_selected_non_heal_low_hp"] = 1.0

        heal_indices: list[int] = []
        for idx in range(legal_count):
            if mask_np[idx] <= 0:
                continue
            candidate = action_source[idx]
            if self._is_rest_site_build_action(candidate) and self._is_rest_heal_build_action(candidate):
                heal_indices.append(int(idx))
        if not heal_indices:
            return action_idx

        search_stats["build_safety_guard_rest_available"] = 1.0
        override_idx = int(heal_indices[0])
        if not (0 <= override_idx < legal_count) or mask_np[override_idx] <= 0:
            search_stats["build_safety_guard_alignment_error"] = 1.0
            return action_idx

        search_stats["build_safety_guard_rest_applied"] = 1.0
        search_stats["build_safety_guard_rest_override"] = 1.0
        return override_idx

    def _apply_route_action_hard_guards(
        self,
        *,
        action_idx: int,
        legal_actions: list[Any] | None,
        action_mask: Any,
        search_stats: dict[str, Any],
    ) -> int:
        """Optional Act1 route safety hard guard.

        This is a narrow guardrail for the observed collapse pattern: the
        policy selects a forced/immediate/no-rest elite route at low HP while a
        route-summary-scored safe alternative is currently legal.  It is
        independent of the Phase 3 soft bias, which remains disabled by default.
        """

        for key in self._route_safety_guard_metric_keys():
            search_stats.setdefault(key, 0.0)
        if not getattr(self, "route_safety_guard_enabled", False):
            return action_idx
        search_stats["route_safety_guard_enabled"] = 1.0

        if not isinstance(legal_actions, list) or len(legal_actions) == 0:
            return action_idx
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        except Exception:
            return action_idx
        legal_count = min(len(legal_actions), int(mask_np.shape[0]) if mask_np.size else 0, MAX_ACTIONS)
        if not (0 <= int(action_idx) < legal_count) or mask_np[int(action_idx)] <= 0:
            return action_idx

        try:
            from sts2_env.deck_quality import deck_quality_v2_from_obs
            from sts2_env.route_heuristic import (
                choose_route_safety_override,
                count_non_empty_potions,
                rank_legal_route_actions,
            )

            env_unwrap = getattr(self.env, "unwrapped", self.env)
            raw = getattr(env_unwrap, "_last_obs_raw", None)
            full_legal_actions = getattr(env_unwrap, "_legal_actions", None) or legal_actions
            if not isinstance(raw, dict) or not isinstance(full_legal_actions, list):
                return action_idx
            if len(full_legal_actions) < legal_count:
                search_stats["route_safety_guard_alignment_error"] = 1.0
                return action_idx

            player = raw.get("player") if isinstance(raw.get("player"), dict) else {}
            run = raw.get("run") if isinstance(raw.get("run"), dict) else {}
            hp, max_hp, hp_valid = self._player_hp_values(raw)
            if not hp_valid:
                search_stats["route_safety_guard_invalid_obs"] = 1.0
                return action_idx
            ranked = rank_legal_route_actions(
                legal_actions=full_legal_actions,
                deck_quality=deck_quality_v2_from_obs(raw),
                hp=float(hp),
                max_hp=float(max_hp),
                gold=self._safe_float(player.get("gold")) if isinstance(player, dict) else 0.0,
                potion_count=count_non_empty_potions(player.get("potions")),
                floor=int(run.get("floor") or 0),
            )
            guard = choose_route_safety_override(ranked, int(action_idx))
            search_stats["route_safety_guard_applicable"] = 1.0 if guard.get("applicable") else 0.0
            search_stats["route_safety_guard_safe_available"] = 1.0 if guard.get("safe_available") else 0.0
            search_stats["route_safety_guard_lower_risk_available"] = 1.0 if guard.get("lower_risk_available") else 0.0
            search_stats["route_safety_guard_selected_risk_class"] = float(guard.get("selected_risk_class", 0.0) or 0.0)
            search_stats["route_safety_guard_final_risk_class"] = float(guard.get("final_risk_class", 0.0) or 0.0)
            search_stats["route_safety_guard_selected_forced_elite"] = 1.0 if guard.get("selected_forced_elite") else 0.0
            search_stats["route_safety_guard_selected_immediate_elite"] = 1.0 if guard.get("selected_immediate_elite") else 0.0
            search_stats["route_safety_guard_low_hp_forced"] = 1.0 if guard.get("low_hp_forced") else 0.0
            if not guard.get("override"):
                return action_idx

            override_idx = guard.get("override_idx")
            if not isinstance(override_idx, int) or not (0 <= override_idx < legal_count):
                search_stats["route_safety_guard_alignment_error"] = 1.0
                return action_idx
            if mask_np[int(override_idx)] <= 0:
                search_stats["route_safety_guard_alignment_error"] = 1.0
                return action_idx
            max_check = max(int(action_idx), int(override_idx))
            if self._route_actions_alignment_error(
                legal_actions,
                full_legal_actions,
                max_index=max_check,
            ):
                search_stats["route_safety_guard_alignment_error"] = 1.0
                return action_idx

            search_stats["route_safety_guard_applied"] = 1.0
            search_stats["route_safety_guard_override"] = 1.0
            action_idx = int(override_idx)
        except Exception as exc:
            print(
                f"[route_safety_guard] exception: {type(exc).__name__}: {exc}",
                flush=True,
            )
        return action_idx


__all__ = ["BuildRouteHardGuardMixin"]
