"""Compact rollout diagnostics split from the self-play orchestrator."""

from __future__ import annotations

from typing import Any

import numpy as np

from sts2_env.action_compact import compact_action_signature
from sts2_env.observation_v2 import MAX_ACTIONS


class SelfPlayDiagnosticsMixin:
    """Replay-safe potion and pre-step diagnostics for self-play."""


    @staticmethod
    def _diag_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _diag_nonempty_potion(potion: Any) -> bool:
        if isinstance(potion, dict):
            text = str(
                potion.get("id")
                or potion.get("potion_id")
                or potion.get("potionId")
                or potion.get("model_id")
                or potion.get("modelId")
                or potion.get("normalized_id")
                or potion.get("normalizedId")
                or potion.get("title")
                or potion.get("name")
                or potion.get("label")
                or potion.get("display_name")
                or potion.get("displayName")
                or potion.get("localized_title")
                or potion.get("localizedTitle")
                or ""
            ).strip().lower()
        else:
            text = str(potion or "").strip().lower()
        return text not in {"", "[empty]", "empty", "none", "null"}

    @classmethod
    def _compact_raw_potion_inventory(cls, raw_obs: dict[str, Any] | None, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return compact live potion slots for boss/elite death forensics.

        Death slices previously only recorded the selected action.  That made it
        impossible to prove whether a visually observed potion was actually
        present in the bridge observation, whether the bridge exposed a legal
        ``use_potion`` action, or whether the tactical guard missed it.  Keep the
        payload tiny so storing it on the trajectory does not inflate replay.
        """

        if not isinstance(raw_obs, dict):
            return []

        containers: list[tuple[str, Any]] = [("$", raw_obs)]
        for key in ("state", "raw_obs", "transition_state", "observation", "obs"):
            payload = raw_obs.get(key)
            if isinstance(payload, dict):
                containers.append((f"$.{key}", payload))

        def _candidate_lists(obs: dict[str, Any], prefix: str) -> list[tuple[str, Any]]:
            candidates: list[tuple[str, Any]] = [(f"{prefix}.potions", obs.get("potions"))]
            player = obs.get("player")
            if isinstance(player, dict):
                candidates.append((f"{prefix}.player.potions", player.get("potions")))
            run = obs.get("run")
            if isinstance(run, dict):
                candidates.append((f"{prefix}.run.potions", run.get("potions")))
            combat = obs.get("combat")
            if isinstance(combat, dict):
                candidates.append((f"{prefix}.combat.potions", combat.get("potions")))
                combat_player = combat.get("player")
                if isinstance(combat_player, dict):
                    candidates.append((f"{prefix}.combat.player.potions", combat_player.get("potions")))
            return candidates

        for prefix, obs in containers:
            if not isinstance(obs, dict):
                continue
            for source, potions in _candidate_lists(obs, prefix):
                if not isinstance(potions, list):
                    continue
                compact: list[dict[str, Any]] = []
                for slot, potion in enumerate(potions[: max(1, int(limit))]):
                    if not cls._diag_nonempty_potion(potion):
                        continue
                    if isinstance(potion, dict):
                        item: dict[str, Any] = {
                            "slot": int(slot),
                            "id": (
                                potion.get("id")
                                or potion.get("potion_id")
                                or potion.get("potionId")
                                or potion.get("model_id")
                                or potion.get("modelId")
                                or potion.get("normalized_id")
                                or potion.get("normalizedId")
                            ),
                            "title": (
                                potion.get("title")
                                or potion.get("name")
                                or potion.get("label")
                                or potion.get("display_name")
                                or potion.get("displayName")
                                or potion.get("localized_title")
                                or potion.get("localizedTitle")
                            ),
                            "rarity": potion.get("rarity"),
                            "empty": bool(potion.get("empty", False)),
                            "is_usable": potion.get("is_usable", potion.get("usable")),
                            "is_queued": potion.get("is_queued"),
                            "has_been_removed_from_state": potion.get("has_been_removed_from_state"),
                            "source": source,
                        }
                        compact.append({k: v for k, v in item.items() if v not in (None, "")})
                    else:
                        compact.append({"slot": int(slot), "title": str(potion), "source": source})
                if compact:
                    return compact
        return []

    def _compact_pre_step_combat_diagnostics(
        self,
        *,
        encoded_obs: dict[str, Any],
        raw_obs: dict[str, Any] | None,
        legal_actions: list[Any],
        action_mask: Any,
        selected_idx: int,
        encounter_tier: str,
        action_family: str,
    ) -> dict[str, Any]:
        """Small pre-step diagnostics embedded in death slices.

        This is intentionally narrower than raw observation dumps.  The primary
        regression it catches is: "boss death while a survival potion (Lucky
        Tonic/幸运药剂) was visible but not selected."  It records both sides of
        the question:

        * current potion inventory from the raw bridge state;
        * legal potion actions and their timing classification before env.step.
        """

        if not isinstance(legal_actions, list):
            legal_actions = []
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        except Exception:
            mask_np = np.zeros(0, dtype=np.float32)
        legal_count = min(len(legal_actions), int(mask_np.shape[0]) if mask_np.ndim == 1 else 0, MAX_ACTIONS)
        if legal_count <= 0:
            return {}

        inventory = self._compact_raw_potion_inventory(raw_obs, limit=5)
        potion_actions: list[dict[str, Any]] = []
        current_energy = self._diag_float(self._combat_energy(encoded_obs, raw_obs), 0.0)

        for idx in range(legal_count):
            if idx >= int(mask_np.shape[0]) or float(mask_np[idx]) <= 0.0:
                continue
            action = legal_actions[idx]
            if not isinstance(action, dict):
                continue
            family = self._semantic_family(action)
            if family not in {"use_potion", "potion"}:
                continue
            signature = compact_action_signature(action)
            timing: dict[str, Any] = {}
            try:
                profile = self._potion_timing_profile(
                    action,
                    int(idx),
                    None,
                    raw_obs,
                    legal_actions,
                    mask_np,
                    current_energy,
                )
            except Exception:
                profile = {}
            if isinstance(profile, dict):
                for key in (
                    "potion_id",
                    "urgent",
                    "prevent_lethal",
                    "prevent_major_loss",
                    "buffer_like",
                    "critical_hp_survival_tool",
                    "resource_survival_tool",
                    "low_urgency",
                    "save_recommended",
                    "lethal",
                    "damage",
                    "block",
                    "heal",
                    "prevent_damage",
                    "incoming",
                    "current_block",
                    "hp",
                    "max_hp",
                    "threat_gap",
                    "use_quality",
                    "waste_risk",
                    "save_value",
                ):
                    value = profile.get(key)
                    if isinstance(value, (bool, int, float, str)):
                        timing[key] = value
            potion_actions.append(
                {
                    "index": int(idx),
                    "action_id": signature.get("action_id"),
                    "title": signature.get("title"),
                    "potion_id": signature.get("potion_id") or timing.get("potion_id"),
                    "target_index": signature.get("target_index"),
                    "timing": {k: v for k, v in timing.items() if v not in (None, "")},
                }
            )

        selected_signature: dict[str, Any] = {}
        if 0 <= int(selected_idx) < len(legal_actions):
            selected_signature = compact_action_signature(legal_actions[int(selected_idx)])

        # Store boss/elite combat decisions even if potion_actions is empty: that
        # is how we can later distinguish "potion existed but no legal action"
        # from "no potion in the bridge state".  For hallway fights, keep only
        # potion-bearing decisions to avoid replay bloat.
        tier = str(encounter_tier or "").strip().lower()
        if tier not in {"elite", "boss"} and not potion_actions and not inventory:
            return {}

        incoming = current_block = hp = max_hp = threat_gap = 0.0
        hp_valid = False
        try:
            incoming, current_block, hp = self._incoming_damage_pressure(raw_obs)
            hp, max_hp, hp_valid = self._player_hp_values(raw_obs)
            threat_gap = max(0.0, float(incoming) - float(current_block))
        except Exception:
            pass

        return {
            "schema": "combat_pre_step_v1",
            "legal_action_count": int(legal_count),
            "legal_potion_action_count": int(len(potion_actions)),
            "raw_potion_count": int(len(inventory)),
            "raw_potions": inventory,
            "legal_potion_actions": potion_actions[:5],
            "selected": {
                "index": int(selected_idx),
                "family": str(action_family or ""),
                "action_id": selected_signature.get("action_id"),
                "title": selected_signature.get("title"),
                "potion_id": selected_signature.get("potion_id"),
            },
            "combat": {
                "hp": float(hp),
                "max_hp": float(max_hp),
                "hp_valid": bool(hp_valid),
                "incoming": float(incoming),
                "current_block": float(current_block),
                "threat_gap": float(threat_gap),
                "energy": float(current_energy),
            },
        }

    @staticmethod
    def _diag_is_lucky_potion_payload(payload: Any) -> bool:
        """Best-effort Lucky Tonic / 幸运药剂 detector for diagnostics payloads."""

        parts: list[str] = []

        def _collect(obj: Any, depth: int = 0) -> None:
            if depth > 4:
                return
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(key, str):
                        parts.append(key)
                    if isinstance(value, (dict, list, tuple)):
                        _collect(value, depth + 1)
                    elif value is not None:
                        parts.append(str(value))
            elif isinstance(obj, (list, tuple)):
                for value in obj[:16]:
                    _collect(value, depth + 1)
            elif obj is not None:
                parts.append(str(obj))

        _collect(payload)
        text = " ".join(parts).lower()
        return bool(
            "lucky_tonic" in text
            or "lucky tonic" in text
            or ("lucky" in text and "tonic" in text)
            or "幸运补剂" in text
            or "幸运药剂" in text
            or "幸運補劑" in text
            or "幸運藥劑" in text
        )

    @classmethod
    def _episode_potion_history_from_steps(
        cls,
        steps: list[dict[str, Any]],
        *,
        encounter_id: str = "",
        tail_limit: int = 8,
    ) -> dict[str, Any]:
        """Aggregate compact potion visibility/legality/selection for death slices.

        ``final_potion_dump`` alone is not enough to debug "died with Lucky
        unused": the potion might have appeared earlier, been legal only on some
        turns, been selected, or disappeared after a state-sync-delayed use.  This
        helper reconstructs a tiny, replay-safe history from the per-step
        ``decision_diagnostics`` stored before ``env.step``.
        """

        encounter_upper = str(encounter_id or "").strip().upper()
        max_tail = max(1, int(tail_limit))
        last_seen_potions: list[dict[str, Any]] = []
        last_seen_legal_potion_actions: list[dict[str, Any]] = []
        selected_potion_actions: list[dict[str, Any]] = []
        lucky_seen = False
        lucky_legal = False
        lucky_selected = False
        matched_steps = 0

        for step_index, step in enumerate(steps or []):
            if not isinstance(step, dict):
                continue
            if str(step.get("decision_domain") or "").strip().lower() != "combat":
                continue
            step_encounter = str(step.get("encounter_id") or "").strip().upper()
            if encounter_upper and step_encounter and step_encounter != encounter_upper:
                continue
            diagnostics = step.get("decision_diagnostics")
            if not isinstance(diagnostics, dict):
                continue
            matched_steps += 1

            raw_potions = diagnostics.get("raw_potions")
            if isinstance(raw_potions, list) and raw_potions:
                compact_raw = [p for p in raw_potions[:5] if isinstance(p, dict)]
                if compact_raw:
                    last_seen_potions = compact_raw
                    lucky_seen = lucky_seen or any(cls._diag_is_lucky_potion_payload(p) for p in compact_raw)

            legal_potions = diagnostics.get("legal_potion_actions")
            if isinstance(legal_potions, list) and legal_potions:
                compact_legal = [p for p in legal_potions[:5] if isinstance(p, dict)]
                if compact_legal:
                    last_seen_legal_potion_actions = compact_legal
                    lucky_legal = lucky_legal or any(cls._diag_is_lucky_potion_payload(p) for p in compact_legal)

            selected = diagnostics.get("selected")
            if isinstance(selected, dict):
                selected_family = str(selected.get("family") or "").strip().lower()
                selected_action_id = str(selected.get("action_id") or "").strip().lower()
                selected_is_potion = bool(
                    selected_family in {"use_potion", "potion"}
                    or selected_action_id.startswith("use_potion")
                    or selected.get("potion_id")
                )
                if selected_is_potion:
                    selected_record = {
                        "step_index": int(step_index),
                        "index": selected.get("index"),
                        "family": selected.get("family"),
                        "action_id": selected.get("action_id"),
                        "title": selected.get("title"),
                        "potion_id": selected.get("potion_id"),
                    }
                    selected_potion_actions.append(
                        {k: v for k, v in selected_record.items() if v not in (None, "")}
                    )
                    lucky_selected = lucky_selected or cls._diag_is_lucky_potion_payload(selected)

        return {
            "potion_history_schema": "episode_potion_history_v1",
            "potion_history_steps_this_combat": int(matched_steps),
            "last_seen_potions_this_combat": list(last_seen_potions[-5:]),
            "last_seen_legal_potion_actions_this_combat": list(last_seen_legal_potion_actions[-5:]),
            "selected_potion_actions_this_combat": list(selected_potion_actions[-max_tail:]),
            "lucky_seen_this_combat": bool(lucky_seen),
            "lucky_legal_this_combat": bool(lucky_legal),
            "lucky_selected_this_combat": bool(lucky_selected),
        }
