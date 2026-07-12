"""Full-run rest/event option classification and low-HP exposure policy."""

from __future__ import annotations

from typing import Any

from ._full_run_values import (
    EVENT_COMBAT_LOW_HP_THRESHOLD,
    EVENT_HP_LOSS_LOW_HP_THRESHOLD,
    _float,
)
from .reward_constants import (
    REST_SITE_SKIP_HEAL_HP_THRESHOLD,
)


class FullRunChoiceSafetyMixin:
    @staticmethod
    def _action_text_blob(action: dict[str, Any]) -> str:
        """Return a conservative lowercase text view of one legal action.

        The live bridge and the headless simulator do not use exactly the
        same rest-site schema.  For tonight's Act1 recovery guard we must
        recognize the *concrete* REST/HEAL option without repeating the old
        bug where the generic surface name "Rest Site" or action id
        ``rest_site:smith`` counted as healing.
        """
        if not isinstance(action, dict):
            return ""
        parts: list[str] = []
        containers = [action]
        payload = action.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)
        for container in containers:
            for key in (
                "kind",
                "action_type",
                "action_id",
                "label",
                "title",
                "name",
                "option_type",
                "canonical_text",
                "description",
            ):
                value = container.get(key)
                if value is not None:
                    parts.append(str(value))
            option = container.get("option") if isinstance(container.get("option"), dict) else {}
            for key in (
                "option_id",
                "id",
                "type",
                "option_type",
                "title",
                "label",
                "name",
                "description",
                "is_enabled",
            ):
                value = option.get(key)
                if value is not None:
                    parts.append(str(value))
        return " ".join(parts).strip().lower()

    @classmethod
    def _is_rest_site_choice_action(cls, action: Any) -> bool:
        """True only for an in-campfire option, not a map node containing rest."""
        if not isinstance(action, dict):
            return False
        containers = [action]
        payload = action.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)
        for container in containers:
            action_id = str(container.get("action_id") or "").strip().lower()
            # After a campfire option is resolved, the bridge exposes a
            # terminal/proceed action on the same rest_site surface.  That is
            # not a player choice between HEAL and SMITH.  Counting it as a
            # rest-site choice creates false rest_heal_exposure_miss and
            # false low-HP "skipped heal" penalties.
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
            action_id = str(container.get("action_id") or "").strip().lower()
            if (
                action_id.startswith("rest_site:")
                or action_id.startswith("choose_rest_option:")
                or action_id.startswith("sim:choose_rest_option")
            ):
                return True
        return False

    @classmethod
    def _is_rest_heal_choice_action(cls, action: Any) -> bool:
        """Detect the concrete HEAL/REST campfire option.

        Important negative examples:
        - ``rest_site:smith`` is NOT heal just because it contains "rest".
        - generic labels like "Rest Site" are NOT heal.
        """
        if not isinstance(action, dict):
            return False
        # HEAL and SMITH must be mutually exclusive even when compact bridge
        # payloads contain stale derived fields (e.g. title="锻造" plus
        # action_kind/is_heal from an earlier classifier).  Prefer concrete
        # smith/upgrade identity before accepting broad rest/heal wording; this
        # keeps the low-HP exposure filter, rest telemetry, and replay action
        # labels aligned with the executed campfire option.
        if cls._is_rest_smith_choice_action(action):
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

        # Live bridge payload uses internal ids/types:
        #   option_id="HEAL", option_type="HealRestSiteOption"
        # Previous code collapsed option_type before option_id, so
        # HealRestSiteOption never matched "heal" and the numeric Chinese
        # description ("回复18点生命值。") also missed "回复生命".
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

        # Require HP/health semantics for substring matches; the bare word
        # "rest" in "Rest Site" is intentionally ignored.
        positive_substrings = (
            "heal",
            "healing",
            "restore hp",
            "restore health",
            "recover hp",
            "recover health",
            "gain hp",
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
        return title_blob in {"rest", "休息"}

    @classmethod
    def _is_rest_smith_choice_action(cls, action: Any) -> bool:
        """Detect the concrete SMITH/UPGRADE campfire option for telemetry.

        This deliberately does not affect masking/guarding; it only gives
        TensorBoard a stable ``env/rest_smith_chosen`` counter so Act1
        recovery can tell "policy never sees smith" from "policy chooses heal
        because HP is low".
        """
        if not isinstance(action, dict):
            return False
        if not cls._is_rest_site_choice_action(action):
            return False
        blob = cls._action_text_blob(action)
        if not blob:
            return False
        smith_tokens = (
            "smith",
            "upgrade",
            "forge",
            "improve",
            "强化",
            "升级",
            "鍛造",
            "锻造",
        )
        return any(token in blob for token in smith_tokens)

    @staticmethod
    def _action_containers(action: Any) -> list[dict[str, Any]]:
        if not isinstance(action, dict):
            return []
        containers = [action]
        payload = action.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)
        return containers

    @classmethod
    def _is_event_option_action(cls, action: Any) -> bool:
        """True for concrete event-option choices exposed by bridge/sim."""
        for container in cls._action_containers(action):
            kind = str(container.get("kind") or container.get("action_type") or "").strip().lower()
            action_id = str(container.get("action_id") or "").strip().lower()
            surface = str(container.get("surface") or "").strip().lower()
            if kind == "event_option" or surface == "event":
                return True
            if (
                action_id.startswith("event_option:")
                or action_id.startswith("choose_event_option:")
                or action_id.startswith("sim:choose_event_option")
            ):
                return True
        return False

    @classmethod
    def _event_option_effect_deltas(cls, action: Any) -> dict[str, Any]:
        """Extract structured event deltas from live or nested action schemas."""
        for container in cls._action_containers(action):
            deltas = container.get("effect_deltas")
            if isinstance(deltas, dict):
                return deltas
            option = container.get("option") if isinstance(container.get("option"), dict) else {}
            deltas = option.get("effect_deltas")
            if isinstance(deltas, dict):
                return deltas
        return {}

    @classmethod
    def _is_event_combat_option(cls, action: Any) -> bool:
        """Detect optional event choices that lead into combat.

        Live bridge effect_deltas are preferred.  The text fallback deliberately
        includes the Chinese "我能打两个" pattern observed in Act1 deaths:
        that first event option leads to a forced singleton "战斗" surface, so
        waiting until the second screen is too late to save the run.
        """
        if not cls._is_event_option_action(action):
            return False
        deltas = cls._event_option_effect_deltas(action)
        if bool(deltas.get("enter_combat")):
            return True
        blob = cls._action_text_blob(action)
        if not blob:
            return False
        combat_tokens = (
            "enter combat",
            "start combat",
            "begin battle",
            "fight",
            "battle",
            "combat",
            "战斗",
            "戰鬥",
            "进入战斗",
            "進入戰鬥",
            "开始战斗",
            "開始戰鬥",
            "遭遇敌人",
            "遭遇敵人",
            "我能打",
            "打两个",
            "打兩個",
            "打一",
            "打二",
        )
        return any(token in blob for token in combat_tokens)

    @classmethod
    def _event_hp_delta_from_action(cls, action: Any) -> float:
        """Best-effort immediate HP delta advertised by an event option."""
        deltas = cls._event_option_effect_deltas(action)
        return _float(deltas.get("hp_delta"), 0.0)

    @classmethod
    def _is_event_hp_loss_option(
        cls,
        action: Any,
        obs: dict[str, Any] | None = None,
    ) -> bool:
        """Detect optional event branches that directly drain player HP.

        Prefer structured ``effect_deltas.hp_delta`` when the bridge/simulator
        provides it.  Some live events still expose only localized labels; the
        Act1 Slippery Bridge death loop is one of those surfaces.  Its risky
        continuation is text like "再撑一会" / "继续" while a safe alternative
        is visible, so keep a narrow room/text fallback instead of treating all
        generic "continue" event buttons as harmful.
        """
        if not cls._is_event_option_action(action):
            return False
        if cls._event_hp_delta_from_action(action) < 0.0:
            return True
        blob = cls._action_text_blob(action)
        if not blob:
            return False
        explicit_risk_tokens = (
            "失去生命",
            "失去生命值",
            "损失生命",
            "损失生命值",
            "lose hp",
            "lose health",
            "lose life",
            "take damage",
            "pay hp",
            "再撑一会",
            "再撐一會",
            "撑一会",
            "撐一會",
            "硬撑",
            "硬撐",
            "强撑",
            "強撐",
            "press on",
            "hold on",
            "keep going",
            "push onward",
            "go further",
        )
        if any(token in blob for token in explicit_risk_tokens):
            return True
        run = obs.get("run") if isinstance(obs, dict) and isinstance(obs.get("run"), dict) else {}
        room_model = str((run or {}).get("room_model") or "").upper()
        if "PUNCH_OFF" in room_model:
            # Punch Off has localized reward/greed branches that may not carry
            # structured effect_deltas yet.  Keep this text fallback
            # room-specific: generic "take" is far too broad on arbitrary
            # events, but in PUNCH_OFF it denotes the risky HP-cost steal/grab
            # branch seen in late-Act1 deaths.
            punch_off_risk_tokens = (
                "顺走",
                "順走",
                "偷",
                "偷走",
                "拿走",
                "抢",
                "搶",
                "take",
                "steal",
                "grab",
            )
            if any(token in blob for token in punch_off_risk_tokens):
                return True
        if "SLIPPERY_BRIDGE" in room_model:
            slippery_continue_tokens = (
                "继续",
                "繼續",
                "continue",
                "再走",
                "再試",
                "再试",
            )
            if any(token in blob for token in slippery_continue_tokens):
                return True
        return False

    @staticmethod
    def _hp_ratio_from_obs(obs: dict[str, Any] | None) -> tuple[float, bool]:
        if not isinstance(obs, dict):
            return 0.0, False
        player = obs.get("player") if isinstance(obs.get("player"), dict) else None
        if not isinstance(player, dict):
            combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
            player = combat.get("player") if isinstance(combat.get("player"), dict) else None
        if not isinstance(player, dict):
            return 0.0, False
        hp = _float(
            player.get("hp", player.get("current_hp", player.get("currentHealth"))),
            0.0,
        )
        max_hp_raw = player.get("max_hp")
        if max_hp_raw is None:
            max_hp_raw = player.get("maxHealth")
        if max_hp_raw is None:
            max_hp_raw = player.get("max_health")
        if max_hp_raw is None:
            max_hp_raw = player.get("maximum_hp")
        max_hp = _float(max_hp_raw, 0.0)
        if hp <= 0.0:
            return 0.0, False
        if max_hp <= 1.0:
            # Live bridge hiccups have produced full-run snapshots with
            # player.max_hp == 1 on mid-Act floors while the player is really
            # at critical HP.  Treat hp<=1/max_hp<=1 as critical instead of
            # as 100% HP; otherwise the rest-site exposure filter can leave
            # Smith/Leave visible at a campfire and train a death spiral.
            run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
            floor = max(
                _float(run.get("floor"), 0.0),
                _float(run.get("total_floor"), 0.0),
                _float(obs.get("snapshot_floor_number"), 0.0),
            )
            if floor > 0.0 and hp <= 1.0:
                return 0.0, True
            return 0.0, False
        return hp / max_hp, True

    def _apply_low_hp_rest_heal_exposure_filter(
        self,
        legal_actions: list[dict[str, Any]],
        obs: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Mask non-heal campfire choices at low HP before the policy sees them.

        The trainer-side override is useful telemetry, but it can fail when
        compact/full action alignment differs.  Filtering at EnvV2's legal
        action exposure layer is safer: the policy's chosen action index now
        truly corresponds to the executed HEAL/REST action, so replay does not
        learn that SMITH caused a heal transition.
        """
        if not isinstance(legal_actions, list) or not legal_actions:
            return legal_actions
        hp_ratio, hp_valid = self._hp_ratio_from_obs(obs)
        if not hp_valid or hp_ratio >= REST_SITE_SKIP_HEAL_HP_THRESHOLD:
            return legal_actions
        rest_indices = [
            idx for idx, action in enumerate(legal_actions)
            if self._is_rest_site_choice_action(action)
        ]
        if not rest_indices:
            return legal_actions
        self._episode_telemetry["rest_heal_exposure_low_hp"] += 1.0
        heal_indices = [
            idx for idx in rest_indices
            if self._is_rest_heal_choice_action(legal_actions[idx])
        ]
        if not heal_indices:
            self._episode_telemetry["rest_heal_exposure_miss"] += 1.0
            try:
                import json

                dump = {
                    "hp_ratio": hp_ratio,
                    "actions": [
                        {
                            "idx": idx,
                            "action_id": str(legal_actions[idx].get("action_id") or ""),
                            "kind": str(legal_actions[idx].get("kind") or ""),
                            "label": str(legal_actions[idx].get("label") or ""),
                            "option": legal_actions[idx].get("option"),
                            "payload": legal_actions[idx].get("payload"),
                            "text": self._action_text_blob(legal_actions[idx])[:500],
                        }
                        for idx in rest_indices
                        if isinstance(legal_actions[idx], dict)
                    ],
                }
                print(
                    "[rest_heal_exposure_miss] "
                    + json.dumps(dump, ensure_ascii=False, default=str)[:2000],
                    flush=True,
                )
            except Exception:
                pass
            return legal_actions
        self._episode_telemetry["rest_heal_exposure_heal_available"] += 1.0
        non_heal_rest_indices = [idx for idx in rest_indices if idx not in set(heal_indices)]
        if not non_heal_rest_indices:
            return legal_actions
        self._episode_telemetry["rest_heal_exposure_forced"] += 1.0
        # Keep non-rest actions if a mixed surface ever slips through; in the
        # normal actions phase _split_actions_phase_actions already returns
        # the rest-site group only.
        heal_set = set(heal_indices)
        rest_set = set(rest_indices)
        return [
            action
            for idx, action in enumerate(legal_actions)
            if idx not in rest_set or idx in heal_set
        ]

    def _apply_low_hp_event_combat_exposure_filter(
        self,
        legal_actions: list[dict[str, Any]],
        obs: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Hide optional event-combat branches when HP is too low.

        The policy was repeatedly dying before Act1 boss by choosing an event
        chain like "我能打两个" -> forced "战斗".  Once the chain advances to the
        singleton combat option the model has no alternative, so the guard must
        operate at the first optional event surface.  Forced/singleton event
        combats are still left visible to avoid deadlocking mandatory events.
        """
        if not isinstance(legal_actions, list) or not legal_actions:
            return legal_actions
        event_indices = [
            idx for idx, action in enumerate(legal_actions)
            if self._is_event_option_action(action)
        ]
        if not event_indices:
            return legal_actions
        combat_indices = [
            idx for idx in event_indices
            if self._is_event_combat_option(legal_actions[idx])
        ]
        if not combat_indices:
            return legal_actions
        self._episode_telemetry["event_combat_option_available"] += 1.0
        hp_ratio, hp_valid = self._hp_ratio_from_obs(obs)
        if not hp_valid or hp_ratio >= EVENT_COMBAT_LOW_HP_THRESHOLD:
            return legal_actions
        self._episode_telemetry["event_combat_option_low_hp_available"] += 1.0
        combat_set = set(combat_indices)
        noncombat_event_indices = [idx for idx in event_indices if idx not in combat_set]
        if not noncombat_event_indices:
            return legal_actions
        self._episode_telemetry["event_combat_option_safe_alternative"] += 1.0
        self._episode_telemetry["event_combat_option_masked_low_hp"] += 1.0
        return [
            action
            for idx, action in enumerate(legal_actions)
            if idx not in combat_set
        ]

    def _apply_low_hp_event_hp_loss_exposure_filter(
        self,
        legal_actions: list[dict[str, Any]],
        obs: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Hide optional event HP-loss branches when a safe exit exists.

        The live Act1 Slippery Bridge can expose a repeated low-HP option
        ("再撑一会") plus a safe alternative.  Existing event-combat masking did
        not catch it, so the policy could spend HP down to 0 before seeing the
        boss.  This guard is exposure-level (before policy/action indexing)
        and leaves singleton/forced event choices visible to avoid deadlocks.
        """
        if not isinstance(legal_actions, list) or not legal_actions:
            return legal_actions
        event_indices = [
            idx for idx, action in enumerate(legal_actions)
            if self._is_event_option_action(action)
        ]
        if not event_indices:
            return legal_actions
        risky_indices = [
            idx for idx in event_indices
            if self._is_event_hp_loss_option(legal_actions[idx], obs)
        ]
        if not risky_indices:
            return legal_actions
        self._episode_telemetry["event_hp_loss_option_available"] += 1.0
        hp_ratio, hp_valid = self._hp_ratio_from_obs(obs)
        if not hp_valid or hp_ratio >= EVENT_HP_LOSS_LOW_HP_THRESHOLD:
            return legal_actions
        self._episode_telemetry["event_hp_loss_option_low_hp_available"] += 1.0
        risky_set = set(risky_indices)
        safe_event_indices = [idx for idx in event_indices if idx not in risky_set]
        if not safe_event_indices:
            return legal_actions
        self._episode_telemetry["event_hp_loss_option_safe_alternative"] += 1.0
        self._episode_telemetry["event_hp_loss_option_masked_low_hp"] += 1.0
        return [
            action
            for idx, action in enumerate(legal_actions)
            if idx not in risky_set
        ]
