"""Death-slice JSONL diagnostic helpers for MuZero training.

Kept separate from :mod:`trainer_dumps` so the core diagnostics mixin stays
small and Act1 post-mortem logic can evolve without growing a mega-file.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np


class DeathSliceDumpMixin:
    """Helpers for bounded combat death-slice diagnostic dumps."""

    @staticmethod
    def _compact_death_slice_array(value: Any, *, head: int = 16) -> Any:
        """Return a bounded JSON-safe summary for large ndarray-like values."""

        try:
            arr = np.asarray(value)
        except Exception:
            return value
        if arr.dtype == object:
            return str(value)
        flat = arr.reshape(-1)
        summary: dict[str, Any] = {
            "shape": [int(dim) for dim in arr.shape],
            "size": int(flat.size),
        }
        if flat.size:
            try:
                numeric = flat.astype(np.float32, copy=False)
                summary.update(
                    {
                        "min": float(np.nanmin(numeric)),
                        "max": float(np.nanmax(numeric)),
                        "mean": float(np.nanmean(numeric)),
                        "sum": float(np.nansum(numeric)),
                        "head": [float(x) for x in numeric[:head].tolist()],
                    }
                )
            except Exception:
                summary["head"] = [str(x) for x in flat[:head].tolist()]
        return summary

    @classmethod
    def _compact_death_slice_obs(cls, obs: Any) -> dict[str, Any]:
        """Summarise the packed observation kept in replay without dumping it whole.

        Combat failure slices are post-mortem artifacts, not replay files.  Full
        token/vector observations can be large enough to make long sandbox runs
        noisy, so keep only shapes/statistics plus small scalar metadata.
        """

        if not isinstance(obs, dict):
            return {}
        out: dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, (int, float, bool, str)) or value is None:
                out[str(key)] = value
                continue
            if isinstance(value, np.ndarray):
                out[str(key)] = cls._compact_death_slice_array(value)
                continue
            if isinstance(value, (list, tuple)) and value and all(
                isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_)) for x in value[:64]
            ):
                out[str(key)] = cls._compact_death_slice_array(value)
                continue
            if isinstance(value, dict):
                small: dict[str, Any] = {}
                for sub_key, sub_value in list(value.items())[:24]:
                    if isinstance(sub_value, (int, float, bool, str)) or sub_value is None:
                        small[str(sub_key)] = sub_value
                    elif isinstance(sub_value, np.ndarray):
                        small[str(sub_key)] = cls._compact_death_slice_array(sub_value, head=8)
                if small:
                    out[str(key)] = small
        return out

    @staticmethod
    def _compact_death_slice_policy(
        *,
        action_mask: Any,
        search_policy: Any,
        selected_action: int,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """Compact selected/legal/top-policy information for a death slice."""

        try:
            mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        except Exception:
            mask = np.zeros(0, dtype=np.float32)
        try:
            policy = np.asarray(search_policy, dtype=np.float32).reshape(-1)
        except Exception:
            policy = np.zeros(0, dtype=np.float32)
        legal_indices = np.flatnonzero(mask > 0.0) if mask.size else np.asarray([], dtype=np.int64)
        top: list[dict[str, Any]] = []
        if policy.size:
            candidates = legal_indices if legal_indices.size else np.arange(policy.size)
            ranked = sorted(
                (int(idx) for idx in candidates if 0 <= int(idx) < policy.size),
                key=lambda idx: float(policy[idx]),
                reverse=True,
            )[: max(int(top_k), 0)]
            top = [{"idx": int(idx), "p": float(policy[idx])} for idx in ranked]
        selected_p = float(policy[int(selected_action)]) if 0 <= int(selected_action) < policy.size else 0.0
        return {
            "legal_count": int(legal_indices.size),
            "selected_policy_prob": selected_p,
            "top_policy": top,
        }

    @staticmethod
    def _compact_death_slice_search_stats(search_stats: Any) -> dict[str, Any]:
        """Keep the search/planner fields that explain tactical failure."""

        if not isinstance(search_stats, dict):
            return {}
        allowed_exact = {
            "num_simulations",
            "root_candidates",
            "root_selectable_children_mean",
            "mean_expanded_children",
            "mean_predicted_legal_count",
            "mean_surface_keep_count",
            "root_top1_visit_share",
            "root_visit_entropy",
            "root_value",
            "max_search_depth",
            "mean_leaf_depth",
            "direct_rollout_steps_used",
            "direct_rollout_branch_count_mean",
            "direct_rollout_root_valid_count",
            "direct_rollout_q_mean",
            "direct_rollout_objective_q_mean",
            "direct_rollout_risk_q_mean",
            "direct_rollout_uncertainty_mean",
            "direct_rollout_branch_disagreement_mean",
            "root_bias_scale",
            "root_bias_nonzero",
            "root_bias_abs_mean",
            "root_bias_max_abs",
            "root_bias_changed_top1",
            "root_bias_selected_action_delta",
            "combat_quality_energy",
            "combat_quality_incoming_damage",
            "combat_quality_positive_action_count",
            "combat_quality_urgent_positive_action_count",
            "combat_quality_deferable_positive_action_count",
        }
        out: dict[str, Any] = {}
        for key, value in search_stats.items():
            keep = (
                key in allowed_exact
                or str(key).startswith("combat_quality_")
                or str(key).startswith("offender/")
            )
            if not keep or not isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
                continue
            out[str(key)] = float(value)
        return out

    @staticmethod
    def _death_slice_lucky_text_match(value: Any) -> bool:
        """Best-effort Lucky/Fortune potion detector for post-mortem dumps.

        Death-slice logging is diagnostic-only; it should prefer catching a
        suspicious death over silently dropping it because an upstream metadata
        flag was not set.  Match the compact potion payload, selected action, or
        final inventory dump by both canonical ids and localized titles.
        """

        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
        text_u = text.upper()
        return bool(
            "LUCKY_TONIC" in text_u
            or "LUCKY TONIC" in text_u
            or "POTION.LUCK" in text_u
            or "FORTUNE" in text_u
            or "幸运药剂" in text
            or "幸运补剂" in text
            or "幸运药" in text
            or "幸运补" in text
            or "幸運" in text
        )

    @classmethod
    def _death_slice_lucky_count(cls, values: Any) -> int:
        if not isinstance(values, list):
            return 0
        return int(sum(1 for item in values if cls._death_slice_lucky_text_match(item)))

    @classmethod
    def _death_slice_lucky_used_in_rows(cls, values: Any) -> bool:
        if not isinstance(values, list):
            return False
        for item in values:
            if not cls._death_slice_lucky_text_match(item):
                continue
            action_id = ""
            if isinstance(item, dict):
                action_id = str(
                    item.get("action_id")
                    or item.get("selected_action_id")
                    or item.get("action")
                    or ""
                ).strip().lower()
            # Some historical rows only record the potion payload after a use.
            # Treat those as used instead of flagging a false "unused Lucky".
            if not action_id:
                return True
            if action_id.startswith("use_potion") or "use_potion" in action_id:
                return True
        return False

    def _dump_death_slice(
        self,
        *,
        trajectory: Any,
        encounter_id: str,
        encounter_tier: str,
        loss: bool,
        steps: list[dict[str, Any]],
        watch_only: bool = True,
        tail_len: int = 5,
        reason: str = "",
    ) -> None:
        """Append a death-slice record to ``diagnostics/death_slices/<eid>.jsonl``
        when a watch-listed or explicitly requested combat episode loses.

        Each record captures:
        * the last few combat steps before death (action / search_stats / mask)
        * episode-level metadata (encounter id, tier, length)
        * mechanic context for Kaiser / Insatiable / Ceremonial as a
          convenience so the slice is self-contained.

        The writer is bounded by ``_DEATH_SLICE_PER_ENCOUNTER_CAP`` so a
        long run cannot produce a multi-gigabyte JSONL. After the cap is
        reached the writer becomes a no-op for that encounter.
        """
        if not loss:
            return
        eid_upper = str(encounter_id or "").strip().upper()
        targets = tuple(str(x).upper() for x in getattr(self, "_DEATH_SLICE_TARGETS", ()))
        if not eid_upper:
            return
        if not isinstance(steps, list) or not steps:
            return
        metadata = trajectory.metadata if isinstance(getattr(trajectory, "metadata", None), dict) else {}
        final_lucky_potion_count = self._death_slice_lucky_count(metadata.get("final_potion_dump"))
        lucky_seen_anywhere = bool(
            bool(metadata.get("lucky_seen_this_combat"))
            or bool(metadata.get("lucky_legal_this_combat"))
            or final_lucky_potion_count > 0
            or self._death_slice_lucky_count(metadata.get("last_seen_potions_this_combat")) > 0
            or self._death_slice_lucky_count(metadata.get("last_seen_legal_potion_actions_this_combat")) > 0
            or self._death_slice_lucky_count(metadata.get("selected_potion_actions_this_combat")) > 0
            or self._death_slice_lucky_count(metadata.get("potion_use_transitions_this_combat")) > 0
        )
        lucky_selected_or_used = bool(
            bool(metadata.get("lucky_selected_this_combat"))
            or self._death_slice_lucky_used_in_rows(metadata.get("selected_potion_actions_this_combat"))
            or self._death_slice_lucky_used_in_rows(metadata.get("potion_use_transitions_this_combat"))
        )
        lucky_unused_survival_potion_death = bool(
            lucky_seen_anywhere
            and not lucky_selected_or_used
        )
        if bool(watch_only) and eid_upper not in targets and not lucky_unused_survival_potion_death:
            return

        if not hasattr(self, "_death_slice_counts"):
            self._death_slice_counts: dict[str, int] = {}
        if self._death_slice_counts.get(eid_upper, 0) >= self._DEATH_SLICE_PER_ENCOUNTER_CAP:
            return

        # Pull the last N combat decisions before death (or fewer if the
        # episode was very short).
        tail_n = max(1, min(int(tail_len or 5), 12))
        tail_steps = steps[-tail_n:]
        slice_steps: list[dict[str, Any]] = []
        for step in tail_steps:
            if not isinstance(step, dict):
                continue
            search_stats = step.get("search_stats") if isinstance(step.get("search_stats"), dict) else {}
            action_info = step.get("action_info") if isinstance(step.get("action_info"), dict) else {}
            decision_diagnostics = (
                step.get("decision_diagnostics")
                if isinstance(step.get("decision_diagnostics"), dict)
                else {}
            )
            selected_action = int(step.get("action") or 0)
            slice_steps.append(
                {
                    "family": self._step_family(step) or "",
                    "decision_domain": str(step.get("decision_domain") or ""),
                    "phase": str(step.get("phase") or ""),
                    "surface": str(step.get("surface") or ""),
                    "room_type": str(step.get("room_type") or ""),
                    "encounter_id": str(step.get("encounter_id") or ""),
                    "encounter_tier": str(step.get("encounter_tier") or ""),
                    "floor": step.get("floor"),
                    "act_id": step.get("act_id"),
                    "action_index": selected_action,
                    "reward": float(step.get("reward") or 0.0),
                    "root_value": float(step.get("root_value") or 0.0),
                    "wasteful_end_turn": bool(step.get("wasteful_end_turn") or False),
                    "settlement_bonus": float(step.get("settlement_bonus") or 0.0),
                    "action_info": action_info,
                    "policy_summary": self._compact_death_slice_policy(
                        action_mask=step.get("action_mask"),
                        search_policy=step.get("search_policy"),
                        selected_action=selected_action,
                    ),
                    "search_stats": self._compact_death_slice_search_stats(search_stats),
                    "decision_diagnostics": decision_diagnostics,
                    "obs_summary": self._compact_death_slice_obs(step.get("obs")),
                }
            )

        record = {
            "time": time.time(),
            "schema_version": 3,
            "global_step": int(getattr(self, "total_steps", 0)),
            "episode_id": int(getattr(self, "episode_count", 0)),
            "reason": str(reason or ""),
            "encounter_id": str(encounter_id or ""),
            "encounter_tier": str(encounter_tier or ""),
            "episode_steps": int(len(steps)),
            "tail_steps": slice_steps,
            "metadata_subset": {
                "episode_mode": metadata.get("episode_mode"),
                "death_floor": metadata.get("death_floor"),
                "max_floor": metadata.get("max_floor"),
                "max_act_id": metadata.get("max_act_id"),
                "max_floor_reached": metadata.get("max_floor_reached"),
                "episode_total_reward": metadata.get("episode_total_reward")
                or metadata.get("episode_reward"),
                "episode_length": metadata.get("episode_length"),
                "negative_reward_episode": metadata.get("negative_reward_episode"),
                "final_deck_cards_compact": metadata.get("final_deck_cards_compact"),
                "final_deck_quality_v2": metadata.get("final_deck_quality_v2"),
                "card_reward_seen_count": metadata.get("card_reward_seen_count"),
                "card_reward_pick_count": metadata.get("card_reward_pick_count"),
                "card_reward_skip_count": metadata.get("card_reward_skip_count"),
                "card_reward_other_count": metadata.get("card_reward_other_count"),
                "card_reward_pick_rate": metadata.get("card_reward_pick_rate"),
                "card_reward_skip_rate": metadata.get("card_reward_skip_rate"),
                "card_reward_consecutive_skip_current": metadata.get("card_reward_consecutive_skip_current"),
                "card_reward_consecutive_skip_max": metadata.get("card_reward_consecutive_skip_max"),
                "boss_rooms_seen": metadata.get("boss_rooms_seen"),
                "elite_rooms_seen": metadata.get("elite_rooms_seen"),
                "snapshot_sample_id": metadata.get("snapshot_sample_id"),
                "snapshot_run_id": metadata.get("snapshot_run_id"),
                "snapshot_floor_number": metadata.get("snapshot_floor_number"),
                "snapshot_build_id": metadata.get("snapshot_build_id"),
                "final_potion_count": metadata.get("final_potion_count"),
                "final_potion_dump": metadata.get("final_potion_dump"),
                "used_potion_count_this_combat": metadata.get("used_potion_count_this_combat"),
                "used_potion_count_final_info": metadata.get("used_potion_count_final_info"),
                "used_potion_count_transition_current": metadata.get("used_potion_count_transition_current"),
                "used_potion_count_episode": metadata.get("used_potion_count_episode"),
                "potion_history_schema": metadata.get("potion_history_schema"),
                "potion_history_steps_this_combat": metadata.get("potion_history_steps_this_combat"),
                "last_seen_potions_this_combat": metadata.get("last_seen_potions_this_combat"),
                "last_seen_legal_potion_actions_this_combat": metadata.get("last_seen_legal_potion_actions_this_combat"),
                "selected_potion_actions_this_combat": metadata.get("selected_potion_actions_this_combat"),
                "lucky_seen_this_combat": metadata.get("lucky_seen_this_combat"),
                "lucky_legal_this_combat": metadata.get("lucky_legal_this_combat"),
                "lucky_selected_this_combat": metadata.get("lucky_selected_this_combat"),
                "lucky_seen_anywhere_on_death": lucky_seen_anywhere,
                "lucky_selected_or_used_this_combat": lucky_selected_or_used,
                "final_lucky_potion_count": final_lucky_potion_count,
                "final_lucky_unused_on_death": bool(final_lucky_potion_count > 0 and not lucky_selected_or_used),
                "lucky_unused_survival_potion_death": lucky_unused_survival_potion_death,
                "potion_use_transitions_this_combat": metadata.get("potion_use_transitions_this_combat"),
                "potion_transition_sync_suspect_this_combat": metadata.get("potion_transition_sync_suspect_this_combat"),
            },
        }

        try:
            path = self._diagnostic_jsonl_path(
                f"death_slices/{eid_upper.replace('.', '_').lower()}.jsonl"
            )
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._death_slice_counts[eid_upper] = self._death_slice_counts.get(eid_upper, 0) + 1
        except Exception:
            return
