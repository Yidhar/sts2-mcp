"""Bounded JSONL diagnostic dump helpers for MuZero training.

Runtime diagnostics belong here instead of in ``muzero.train``.  All writers are
best-effort and bounded by trainer counters/caps so post-mortem instrumentation
cannot take down training or fill the disk during long runs.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from sts2_env.boss_mechanics import build_boss_mechanics_context
from sts2_env.observation_v2 import MAX_ACTIONS

from muzero.diagnostics.death_slice_dumps import DeathSliceDumpMixin


def _diagnostic_jsonl_path_for(owner: Any, filename: str) -> Path | None:
    """Resolve a diagnostics path without requiring a fully-built trainer.

    Diagnostic helpers are intentionally usable as bound methods on a tiny
    test/analysis object. Keeping path resolution standalone makes that
    compatibility explicit instead of losing records behind a broad exception.
    """

    run_paths = getattr(owner, "run_paths", None)
    if run_paths is not None and hasattr(run_paths, "diagnostic_jsonl"):
        try:
            return run_paths.diagnostic_jsonl(str(filename))
        except (OSError, TypeError, ValueError, AttributeError):
            pass
    log_dir = getattr(owner, "log_dir", None)
    if not log_dir:
        return None
    name = str(filename)
    if not name.endswith(".jsonl"):
        name = f"{name}.jsonl"
    return Path(log_dir) / "diagnostics" / name


def _report_diagnostic_failure(owner: Any, diagnostic: str, exc: Exception) -> None:
    """Make best-effort dump failures observable without stopping training."""

    failures = getattr(owner, "_diagnostic_dump_failures", None)
    if not isinstance(failures, dict):
        failures = {}
        try:
            setattr(owner, "_diagnostic_dump_failures", failures)
        except (AttributeError, TypeError):
            pass
    failures[str(diagnostic)] = int(failures.get(str(diagnostic), 0) or 0) + 1
    warnings.warn(
        f"diagnostic dump {diagnostic!r} failed: {type(exc).__name__}: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )


def _progress_values_from_obs(owner: Any, raw_obs: Any) -> tuple[float, float]:
    """Extract stable numeric floor/act values for diagnostic records."""

    obs = raw_obs if isinstance(raw_obs, dict) else {}
    run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
    if not run and isinstance(obs.get("transition_state"), dict):
        candidate = obs["transition_state"].get("run")
        if isinstance(candidate, dict):
            run = candidate

    def _number(value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return 0.0
        return out if math.isfinite(out) else 0.0

    floor = _number(
        run.get("floor", run.get("total_floor", obs.get("floor", obs.get("current_floor", 0.0))))
    )
    act_raw = run.get("act_id", run.get("act", obs.get("act_id", obs.get("act", 0.0))))
    parse_act = getattr(owner, "_parse_progress_act_id", None)
    if callable(parse_act):
        try:
            act = _number(parse_act(act_raw, run))
        except (TypeError, ValueError, KeyError, AttributeError):
            act = _number(act_raw)
    else:
        act = _number(act_raw)
    return floor, act


class DiagnosticDumpMixin(DeathSliceDumpMixin):
    """JSONL dump helpers mixed into MuZeroTrainer."""

    def _diagnostic_jsonl_path(self, filename: str) -> Path | None:
        """Return the canonical diagnostics JSONL path for this run.

        Prefer ``RunPaths`` when the trainer has it; keep the ``self.log_dir``
        fallback so old tests/lightweight mixin users do not need to construct a
        full trainer instance.
        """

        return _diagnostic_jsonl_path_for(self, filename)

    def _dump_card_reward_choice_diagnostic(self, payload: dict[str, Any] | None) -> None:
        """Append one card-reward choice diagnostic record.

        TensorBoard scalars can tell us that reward skip is high, but not
        whether the offered cards were true junk, orphan combo pieces, or
        useful future options that the model should have taken.  The guard
        stores a compact payload in ``search_stats``; this method is the bounded
        writer used by self-play after the final selected action is known.
        """

        if not isinstance(payload, dict):
            return
        if getattr(self, "_card_reward_choice_dump_disabled", False):
            return
        cap = int(getattr(self, "_card_reward_choice_dump_max", 100000))
        count = int(getattr(self, "_card_reward_choice_dump_count", 0) or 0)
        if cap > 0 and count >= cap:
            return
        try:
            record = {
                "time": time.time(),
                "global_step": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                **payload,
            }
            path = self._diagnostic_jsonl_path("card_reward_choices.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._card_reward_choice_dump_count = count + 1
        except Exception:
            return

    def _dump_death_deck_summary(
        self,
        *,
        final_info: dict[str, Any] | None,
        final_progress: dict[str, Any] | None,
        final_deck_cards: list[Any] | tuple[Any, ...] | None,
        final_deck_quality: dict[str, Any] | None,
        final_deck_compact: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
        card_reward_meta: dict[str, Any] | None,
        episode_reward: float,
        episode_length: int,
        max_floor: float,
        death_floor: float,
        act1_boss_seen: bool,
        act1_clear: bool,
        progress_snapshots: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    ) -> None:
        """Append one compact death-deck post-mortem record.

        TensorBoard scalars show when a dying deck is weak, but not which
        concrete cards caused the construction failure.  This bounded JSONL
        keeps card counts, deck-quality scalars, reward skip/pick metadata,
        and final floor context together for post-mortems.
        """

        if float(death_floor or 0.0) <= 0.0:
            return
        if getattr(self, "_death_deck_summary_dump_disabled", False):
            return
        cap = int(getattr(self, "_death_deck_summary_dump_max", 100000))
        count = int(getattr(self, "_death_deck_summary_dump_count", 0) or 0)
        if cap > 0 and count >= cap:
            return

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                out = float(value)
            except (TypeError, ValueError):
                return float(default)
            return out if math.isfinite(out) else float(default)

        def _safe_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        try:
            compact_cards = list(final_deck_compact or [])
            title_counts: dict[str, int] = {}
            upgraded_counts: dict[str, int] = {}
            for card in compact_cards:
                if not isinstance(card, dict):
                    continue
                title = str(card.get("title") or card.get("id") or "<unknown>").strip() or "<unknown>"
                title_counts[title] = int(title_counts.get(title, 0)) + 1
                if bool(card.get("upgraded")) or _safe_int(card.get("upgrade_level"), 0) > 0:
                    upgraded_counts[title] = int(upgraded_counts.get(title, 0)) + 1

            progress = dict(final_progress or {})
            final_info_phase = final_info.get("phase") if isinstance(final_info, dict) else None
            payload = {
                "time": time.time(),
                "kind": "death_deck_summary",
                "global_step": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                "episode": {
                    "reward": _safe_float(episode_reward),
                    "length": _safe_int(episode_length),
                    "max_floor": _safe_float(max_floor),
                    "death_floor": _safe_float(death_floor),
                    "act1_boss_seen": bool(act1_boss_seen),
                    "act1_clear": bool(act1_clear),
                    "final_phase": final_info_phase,
                },
                "final_progress": {
                    "floor": _safe_float(progress.get("floor")),
                    "act_id": _safe_float(progress.get("act_id")),
                    "room_type": progress.get("room_type"),
                    "room_model": progress.get("room_model"),
                    "encounter_id": progress.get("encounter_id"),
                    "hp": _safe_float(progress.get("hp")),
                    "max_hp": _safe_float(progress.get("max_hp")),
                    "hp_valid": bool(progress.get("hp_valid")),
                    "max_hp_suspicious": bool(progress.get("max_hp_suspicious")),
                },
                "deck": {
                    "size": len(final_deck_cards or []),
                    "compact_size": len(compact_cards),
                    "starter_count": _safe_float((final_deck_quality or {}).get("starter_count"), 0.0),
                    "starter_ratio": _safe_float((final_deck_quality or {}).get("starter_ratio"), 0.0),
                    "nonstarter_count": _safe_float((final_deck_quality or {}).get("nonstarter_count"), 0.0),
                    "strike_count": _safe_float((final_deck_quality or {}).get("strike_count"), 0.0),
                    "defend_count": _safe_float((final_deck_quality or {}).get("defend_count"), 0.0),
                    "upgraded_count": _safe_float((final_deck_quality or {}).get("upgraded_count"), 0.0),
                    "upgraded_ratio": _safe_float((final_deck_quality or {}).get("upgraded_ratio"), 0.0),
                    "starter_upgrade_count": _safe_float((final_deck_quality or {}).get("starter_upgrade_count"), 0.0),
                    "starter_upgrade_ratio": _safe_float((final_deck_quality or {}).get("starter_upgrade_ratio"), 0.0),
                    "title_counts": [
                        {
                            "title": title,
                            "count": int(count_value),
                            "upgraded_count": int(upgraded_counts.get(title, 0)),
                        }
                        for title, count_value in sorted(
                            title_counts.items(),
                            key=lambda kv: (-int(kv[1]), str(kv[0])),
                        )
                    ],
                    "cards": compact_cards,
                },
                "deck_quality": {
                    str(key): _safe_float(value, 0.0)
                    for key, value in (final_deck_quality or {}).items()
                    if isinstance(value, (bool, int, float, np.generic)) or value is None
                },
                "card_reward": {
                    str(key): _safe_float(value, 0.0)
                    for key, value in (card_reward_meta or {}).items()
                    if isinstance(value, (bool, int, float, np.generic)) or value is None
                },
                "progress_tail": [
                    dict(item)
                    for item in list(progress_snapshots or [])[-8:]
                    if isinstance(item, dict)
                ],
            }
            path = self._diagnostic_jsonl_path("death_deck_summary.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._death_deck_summary_dump_count = count + 1
        except Exception:
            return

    def _dump_combat_hard_guard_record(
        self,
        *,
        kind: str,
        raw_obs: Any,
        legal_actions: list[Any] | None,
        original_idx: int,
        override_idx: int,
        risk: float,
        countdown: float | None,
        encounter: str,
        lethal_exemption: bool,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append one record to the per-guard diagnostics jsonl file.

        ``kind`` is ``kaiser_facing`` or ``insatiable_escape``. The two
        files share schema except for the encounter-specific fields.
        """
        try:
            filename_map = {
                "kaiser_facing": "kaiser_facing_guard.jsonl",
                "insatiable_escape": "insatiable_escape_guard.jsonl",
                "x_cost_zero": "x_cost_zero_guard.jsonl",
                "hp_cost_margin": "hp_cost_margin_guard.jsonl",
                "potion_bad_use": "potion_bad_use_guard.jsonl",
                "boss_survival_potion": "boss_survival_potion_guard.jsonl",
                "boss_race_potion": "boss_race_potion_guard.jsonl",
                "late_normal_race_potion": "late_normal_race_potion_guard.jsonl",
                "refund_no_followup": "refund_no_followup_guard.jsonl",
                "selection_loop": "selection_loop_guard.jsonl",
            }
            filename = filename_map.get(str(kind), f"{kind}_guard.jsonl")
            path = self._diagnostic_jsonl_path(filename)
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
            player = raw_obs.get("player") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("player"), dict) else {}
            player_hp, player_max_hp, _player_hp_valid = self._player_hp_values(raw_obs if isinstance(raw_obs, dict) else None)

            def _safe_call(name: str, *args: Any, default: Any = None) -> Any:
                try:
                    fn = getattr(self, name, None)
                    if not callable(fn):
                        return default
                    return fn(*args)
                except Exception:
                    return default

            def _safe_float(value: Any, default: float | None = None) -> float | None:
                try:
                    if value is None:
                        return default
                    out = float(value)
                    if not math.isfinite(out):
                        return default
                    return out
                except Exception:
                    return default

            def _numeric_value(action: dict[str, Any], keys: tuple[str, ...]) -> float | None:
                return _safe_float(_safe_call("_action_numeric_value", action, keys, default=None))

            def _metric(action: dict[str, Any], name: str) -> float | None:
                return _safe_float(_safe_call("_action_metric", action, name, default=None))

            def _max_metric(action: dict[str, Any], names: tuple[str, ...], keys: tuple[str, ...] = ()) -> float | None:
                values: list[float] = []
                for name in names:
                    value = _metric(action, name)
                    if value is not None:
                        values.append(value)
                if keys:
                    value = _numeric_value(action, keys)
                    if value is not None:
                        values.append(value)
                return max(values) if values else None

            incoming_damage: float | None = None
            current_block: float | None = None
            try:
                incoming_damage, current_block, _ = self._incoming_damage_pressure(
                    raw_obs if isinstance(raw_obs, dict) else None
                )
                incoming_damage = _safe_float(incoming_damage)
                current_block = _safe_float(current_block)
            except Exception:
                incoming_damage = _safe_float(combat.get("incoming_damage") or combat.get("incoming"))
                current_block = _safe_float(
                    combat.get("block") or player.get("block") or combat.get("current_block")
                )
            floor_value = _safe_float(
                _safe_call("_combat_floor_value", raw_obs if isinstance(raw_obs, dict) else None, default=None)
            )
            if floor_value is None and isinstance(raw_obs, dict):
                floor_value = _safe_float(raw_obs.get("floor") or raw_obs.get("current_floor"))
            act_id = None
            if isinstance(raw_obs, dict):
                act_id = raw_obs.get("act_id") or raw_obs.get("act")
                if act_id is None and isinstance(raw_obs.get("run"), dict):
                    act_id = raw_obs["run"].get("act_id") or raw_obs["run"].get("act")

            def _summarise(idx: int) -> dict[str, Any]:
                if not isinstance(legal_actions, list) or not (0 <= idx < len(legal_actions)):
                    return {"index": idx, "missing": True}
                action = legal_actions[idx] if isinstance(legal_actions[idx], dict) else {}
                card = action.get("card") if isinstance(action.get("card"), dict) else {}
                potion = action.get("potion") if isinstance(action.get("potion"), dict) else {}
                try:
                    roles = sorted(str(x) for x in (_safe_call("_action_roles", action, default=set()) or []))
                except Exception:
                    roles = []
                title = (
                    action.get("title")
                    or card.get("title")
                    or card.get("name")
                    or potion.get("title")
                    or potion.get("name")
                    or action.get("label")
                )
                return {
                    "index": int(idx),
                    "family": self._semantic_family(action),
                    "kind": action.get("kind"),
                    "action_id": action.get("action_id"),
                    "label": action.get("label"),
                    "title": title,
                    "card_id": card.get("id"),
                    "card_title": card.get("title") or card.get("name"),
                    "potion_id": potion.get("id"),
                    "potion_title": potion.get("title") or potion.get("name"),
                    "cost": _safe_float(_safe_call("_action_cost_value", action, default=None)),
                    "damage": _max_metric(
                        action,
                        ("damage",),
                        ("damage", "attack_damage", "preview_damage", "expected_damage"),
                    ),
                    "total_damage": _max_metric(
                        action,
                        ("total_damage",),
                        ("total_damage", "typed_total_damage"),
                    ),
                    "block": _max_metric(
                        action,
                        ("block",),
                        ("block", "preview_block", "typed_block_amount"),
                    ),
                    "total_block": _max_metric(
                        action,
                        ("total_block",),
                        ("total_block", "typed_total_block"),
                    ),
                    "heal": _max_metric(
                        action,
                        ("heal",),
                        ("heal", "healing", "hp_gain", "typed_heal_amount"),
                    ),
                    "impact": _safe_float(_safe_call("_action_immediate_impact", action, default=None)),
                    "roles": roles,
                    "target_combat_id": self._action_target_combat_id(action),
                    "target_back_attack_position": self._action_target_back_attack_position(action, raw_obs),
                    "changes_facing": self._action_changes_facing_toward_target(action, raw_obs),
                }

            record = {
                "time": time.time(),
                "kind": kind,
                "episode_count": int(getattr(self, "episode_count", 0)),
                "total_steps": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                "global_step": int(getattr(self, "total_steps", 0)),
                "encounter": encounter,
                "legal_count": len(legal_actions) if isinstance(legal_actions, list) else 0,
                "floor": floor_value,
                "act_id": act_id,
                "incoming_damage": incoming_damage,
                "current_block": current_block,
                "kaiser_back_attack_risk": float(risk) if risk is not None else None,
                "insatiable_sandpit_countdown": (
                    float(countdown) if countdown is not None else None
                ),
                "lethal_exemption": bool(lethal_exemption),
                "player_facing": self._combat_player_facing(raw_obs),
                "player_hp": float(player_hp) if player_hp > 0.0 else player.get("current_hp") or player.get("hp"),
                "player_max_hp": float(player_max_hp) if player_max_hp > 0.0 else player.get("max_hp"),
                "energy": combat.get("energy"),
                "state_summary": {
                    "hp": float(player_hp) if player_hp > 0.0 else player.get("current_hp") or player.get("hp"),
                    "max_hp": float(player_max_hp) if player_max_hp > 0.0 else player.get("max_hp"),
                    "energy": combat.get("energy"),
                    "incoming_damage": incoming_damage,
                    "block": current_block,
                    "floor": floor_value,
                    "act_id": act_id,
                },
                "original_action": _summarise(int(original_idx)),
                "override_action": _summarise(int(override_idx)),
                "enemies": combat.get("enemies"),
            }
            if isinstance(extra, dict):
                record["extra"] = extra
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            return

    def _dump_kaiser_facing_diagnostic(
        self,
        raw_obs: Any,
        legal_actions: list[Any] | None,
        action_mask: np.ndarray,
        encounter: str,
        risk: float,
    ) -> None:
        if self._kaiser_facing_debug_dump_count >= 24:
            return
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
            combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
            rows = []
            for idx, action in enumerate(legal_actions or []):
                if idx >= MAX_ACTIONS or idx >= mask_np.shape[0] or mask_np[idx] <= 0:
                    continue
                if not isinstance(action, dict):
                    continue
                target = action.get("target") if isinstance(action.get("target"), dict) else {}
                rows.append({
                    "idx": idx,
                    "family": self._semantic_family(action),
                    "kind": action.get("kind"),
                    "action_id": action.get("action_id"),
                    "label": action.get("label"),
                    "target_combat_id": self._action_target_combat_id(action),
                    "target_name": action.get("target_name") or target.get("name"),
                    "target_side_or_back_attack_position": self._action_target_side(action, raw_obs),
                    "target_back_attack_position": self._action_target_back_attack_position(action, raw_obs),
                    "changes_facing": self._action_changes_facing_toward_target(action, raw_obs),
                    "explicit_facing": self._is_facing_change_action(action),
                    "roles": sorted(self._action_roles(action)),
                    "card": (action.get("card") if isinstance(action.get("card"), dict) else {}).get("title")
                        or (action.get("card") if isinstance(action.get("card"), dict) else {}).get("name"),
                    "potion": (action.get("potion") if isinstance(action.get("potion"), dict) else {}).get("title")
                        or (action.get("potion") if isinstance(action.get("potion"), dict) else {}).get("name"),
                    "keys": sorted(str(k) for k in action.keys()),
                })
            path = self._diagnostic_jsonl_path("kaiser_facing_legal_actions.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "time": time.time(),
                "episode_count": int(getattr(self, "episode_count", 0)),
                "total_steps": int(getattr(self, "total_steps", 0)),
                "encounter": encounter,
                "risk": float(risk),
                "player_facing": self._combat_player_facing(raw_obs),
                "combat_facing_raw": combat.get("facing"),
                "enemies": combat.get("enemies"),
                "actions": rows,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._kaiser_facing_debug_dump_count += 1
        except Exception:
            return

    def _dump_end_turn_diagnostic(
        self,
        raw_obs: Any,
        encoded_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        action_mask: np.ndarray,
        context: dict[str, Any],
        chosen_idx: int,
        encounter: str,
        wasteful_end_turn: bool,
    ) -> None:
        """One-shot per end_turn selection: dump full decision state for post-mortem.

        Written to <log_dir>/diagnostics/end_turn_selections.jsonl when the policy
        chooses end_turn.  Captures mask / playability / positive-action judgement
        per legal action so we can prove whether the policy was genuinely forced
        into end_turn (no positive action playable) or whether the detector mis-
        categorised a hand card and missed a real wasteful.
        """
        if self._end_turn_debug_dump_count >= 80:
            return
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
            raw_combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
            raw_player = raw_obs.get("player") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("player"), dict) else {}
            energy_raw = self._obs_energy(raw_obs)
            energy_encoded = self._obs_energy(encoded_obs)
            hand_cards = raw_combat.get("hand") if isinstance(raw_combat.get("hand"), list) else []

            rows: list[dict[str, Any]] = []
            for idx, action in enumerate(legal_actions or []):
                if idx >= MAX_ACTIONS:
                    break
                if not isinstance(action, dict):
                    continue
                mask_value = float(mask_np[idx]) if idx < mask_np.shape[0] else 0.0
                family = self._semantic_family(action)
                roles = sorted(self._action_roles(action))
                is_positive = self._is_positive_combat_action(action) if mask_value > 0 else False
                card = action.get("card") if isinstance(action.get("card"), dict) else {}
                rows.append({
                    "idx": idx,
                    "mask": mask_value,
                    "is_chosen": (idx == int(chosen_idx)),
                    "family": family,
                    "action_id": action.get("action_id"),
                    "title": action.get("title") or card.get("title"),
                    "is_positive": is_positive,
                    "roles": roles,
                    "damage": self._action_metric(action, "damage"),
                    "block": self._action_metric(action, "block"),
                    "cost": card.get("cost"),
                    "card_id": card.get("id"),
                })

            path = self._diagnostic_jsonl_path("end_turn_selections.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "time": time.time(),
                "episode_count": int(getattr(self, "episode_count", 0)),
                "total_steps": int(getattr(self, "total_steps", 0)),
                "encounter": encounter,
                "wasteful_end_turn": bool(wasteful_end_turn),
                "wasteful_context": {
                    "wasteful": bool(context.get("wasteful", False)),
                    "energy_from_context": float(context.get("energy", 0.0) or 0.0),
                    "end_turn_indices": list(context.get("end_turn_indices", [])),
                    "positive_indices": list(context.get("positive_indices", [])),
                    "positive_progress_count": int(context.get("positive_progress_count", 0) or 0),
                    "setup_scaling_indices": list(context.get("setup_scaling_indices", [])),
                    "zero_cost_positive": bool(context.get("zero_cost_positive", False)),
                    "severity": float(context.get("severity", 0.0) or 0.0),
                },
                "energy": {
                    "raw": energy_raw,
                    "encoded": energy_encoded,
                    "combat.energy": raw_combat.get("energy"),
                    "combat.max_energy": raw_combat.get("max_energy"),
                },
                "player_hp": raw_player.get("hp"),
                "combat_round": raw_combat.get("round"),
                "raw_hand_cards": [
                    {
                        "title": c.get("title") if isinstance(c, dict) else str(c),
                        "cost": c.get("cost") if isinstance(c, dict) else None,
                        "is_playable": c.get("is_playable") if isinstance(c, dict) else None,
                    }
                    for c in hand_cards[:10]
                ],
                "chosen_idx": int(chosen_idx),
                "legal_actions": rows,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._end_turn_debug_dump_count += 1
        except Exception:
            return

    def _dump_selected_end_turn_context(
        self,
        encoded_obs: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None,
        action_mask: np.ndarray,
        legal_actions: list[Any] | None,
        chosen_idx: int,
        context: dict[str, Any],
        search_policy: np.ndarray | None,
        search_stats: dict[str, Any] | None,
        action_diagnostics: dict[str, Any] | None,
        encounter: str,
        tier: str,
        pre_step_info: dict[str, Any] | None = None,
    ) -> None:
        """Append one JSONL entry per selected end_turn action.

        Designed to stay cheap: reuses precomputed ``context`` indices and
        existing search probabilities; does not loop legality classifiers again.
        """

        if getattr(self, "_end_turn_context_dump_disabled", False):
            return
        cap = int(getattr(self, "_end_turn_context_dump_max", 50000))
        if cap > 0 and self._end_turn_context_dump_count >= cap:
            return
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
            class_name, reason_flags = self._classify_end_turn_action(context, action_diagnostics)

            raw_combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
            raw_player = raw_obs.get("player") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("player"), dict) else {}
            if not raw_player and isinstance(raw_combat.get("player"), dict):
                raw_player = raw_combat.get("player")
            raw_run = raw_obs.get("run") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("run"), dict) else {}
            if not raw_run and isinstance(raw_obs, dict) and isinstance(raw_obs.get("transition_state"), dict):
                transition_run = raw_obs["transition_state"].get("run")
                if isinstance(transition_run, dict):
                    raw_run = transition_run
            pre_info = pre_step_info if isinstance(pre_step_info, dict) else {}
            pre_actionability = (
                pre_info.get("actionability")
                if isinstance(pre_info.get("actionability"), dict)
                else {}
            )
            pre_bridge_info = (
                pre_info.get("bridge_info")
                if isinstance(pre_info.get("bridge_info"), dict)
                else {}
            )
            pre_bridge_actionability = (
                pre_bridge_info.get("actionability")
                if isinstance(pre_bridge_info.get("actionability"), dict)
                else {}
            )

            def _safe_float_local(value: Any, default: float = 0.0) -> float:
                try:
                    out = float(value)
                except (TypeError, ValueError):
                    return float(default)
                return out if math.isfinite(out) else float(default)

            floor_value = (
                _safe_float_local(raw_run.get("floor", raw_run.get("total_floor")))
                if isinstance(raw_run, dict)
                else 0.0
            )
            parse_act = getattr(self, "_parse_progress_act_id", None)
            if callable(parse_act) and isinstance(raw_run, dict):
                try:
                    act_id_value = float(parse_act(raw_run.get("act_id"), raw_run))
                except Exception:
                    act_id_value = 0.0
            else:
                act_id_value = _safe_float_local(raw_run.get("act_id")) if isinstance(raw_run, dict) else 0.0
            incoming, block, hp = self._incoming_damage_pressure(raw_obs)
            energy_val = float(context.get("energy", 0.0) or 0.0)
            try:
                raw_hand = raw_combat.get("hand")
                if not isinstance(raw_hand, list) and isinstance(raw_player, dict):
                    raw_hand = raw_player.get("hand")
                if not isinstance(raw_hand, list):
                    raw_hand = []
                hand_count = len(raw_hand)
                if hand_count <= 0:
                    hand_count = int(_safe_float_local(raw_combat.get("hand_count", raw_combat.get("num_cards_in_hand")), 0.0))
                draw_count = len(raw_combat.get("draw_pile") or []) if isinstance(raw_combat.get("draw_pile"), list) else 0
                discard_count = len(raw_combat.get("discard_pile") or []) if isinstance(raw_combat.get("discard_pile"), list) else 0
                exhaust_count = len(raw_combat.get("exhaust_pile") or []) if isinstance(raw_combat.get("exhaust_pile"), list) else 0
            except Exception:
                raw_hand = []
                hand_count = draw_count = discard_count = exhaust_count = 0

            x_cost_count = 0
            zero_x_count = 0
            legal_non_end_turn_count = 0
            playable_card_action_count = 0
            try:
                legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
                for idx in range(legal_count):
                    if mask_np[idx] <= 0:
                        continue
                    action = (legal_actions or [])[idx]
                    family = self._semantic_family(action)
                    if family != "end_turn":
                        legal_non_end_turn_count += 1
                    if family == "play_card":
                        playable_card_action_count += 1
                        if self._is_x_cost_action(encoded_obs, idx, action):
                            x_cost_count += 1
                            if energy_val <= 0.05:
                                zero_x_count += 1
            except Exception:
                pass

            boss_context: dict[str, Any] = {}
            try:
                if isinstance(raw_obs, dict):
                    boss_ctx = build_boss_mechanics_context(raw_obs)
                    kaiser_back_attack_risk = self._kaiser_back_attack_risk_from_context(boss_ctx, raw_obs)
                    kaiser_back_attack_active = (
                        float(self._boss_context_max(boss_ctx, "back_attack_active"))
                        if self._is_kaiser_encounter_context(boss_ctx, raw_obs)
                        else 0.0
                    )
                    boss_context = {
                        "kaiser_back_attack_risk": float(kaiser_back_attack_risk),
                        "kaiser_back_attack_active": float(kaiser_back_attack_active),
                        "incoming_damage_multiplier_norm": float(self._boss_context_max(boss_ctx, "incoming_damage_multiplier_norm")),
                        "kaiser_facing_change_candidate_count": int(float(
                            (search_stats or {}).get("combat_quality_kaiser_facing_change_candidate_count", 0.0) or 0.0
                        )),
                        "ceremonial_one_card_lock": float(self._boss_context_max(boss_ctx, "one_card_lock")),
                        "ceremonial_stun_window": float(self._boss_context_max(boss_ctx, "stun_window")),
                    }
            except Exception:
                boss_context = {}

            top_actions: list[dict[str, Any]] = []
            try:
                policy_np = (
                    np.asarray(search_policy, dtype=np.float32).reshape(-1)
                    if search_policy is not None
                    else np.zeros(MAX_ACTIONS, dtype=np.float32)
                )
                # Rank legal actions by search_policy probability; fall back to mask order.
                legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0], policy_np.shape[0])
                ranked: list[tuple[int, float]] = []
                for idx in range(legal_count):
                    if mask_np[idx] <= 0:
                        continue
                    score = float(policy_np[idx]) if idx < policy_np.shape[0] else 0.0
                    ranked.append((idx, score))
                ranked.sort(key=lambda kv: kv[1], reverse=True)
                for rank, (idx, score) in enumerate(ranked[:6], start=1):
                    action = (legal_actions or [])[idx]
                    family = self._semantic_family(action) if isinstance(action, dict) else ""
                    card = action.get("card") if isinstance(action, dict) and isinstance(action.get("card"), dict) else {}
                    target = action.get("target") if isinstance(action, dict) and isinstance(action.get("target"), dict) else {}
                    tags: list[str] = []
                    if isinstance(action, dict):
                        if int(idx) in {int(i) for i in context.get("urgent_positive_indices") or []}:
                            tags.append("urgent_positive")
                        if int(idx) in {int(i) for i in context.get("deferable_positive_indices") or []}:
                            tags.append("deferable")
                        if int(idx) in {int(i) for i in context.get("setup_scaling_indices") or []}:
                            tags.append("setup_scaling")
                        if family == "end_turn":
                            tags.append("end_turn")
                        if family == "play_card" and self._is_x_cost_action(encoded_obs, idx, action):
                            tags.append("x_cost")
                            if energy_val <= 0.05:
                                tags.append("zero_energy_x_cost")
                        if self._is_kaiser_facing_change_action(action, raw_obs):
                            tags.append("kaiser_facing_change")
                    top_actions.append({
                        "rank": rank,
                        "action_idx": int(idx),
                        "is_chosen": int(idx) == int(chosen_idx),
                        "family": family,
                        "card_id": card.get("id") if isinstance(card, dict) else None,
                        "title": (action.get("title") if isinstance(action, dict) else None) or (card.get("title") if isinstance(card, dict) else None),
                        "target": target.get("name") if isinstance(target, dict) else None,
                        "score": float(score),
                        "tags": tags,
                    })
            except Exception:
                top_actions = []

            def _compact_card_local(card: Any) -> dict[str, Any]:
                if not isinstance(card, dict):
                    return {"repr": str(card)}
                return {
                    "id": card.get("id") or card.get("card_id"),
                    "title": card.get("title") or card.get("name"),
                    "instance_uuid": card.get("instance_uuid") or card.get("uuid") or card.get("runtime_id"),
                    "type": card.get("type") or card.get("card_type"),
                    "cost": card.get("cost"),
                    "cost_for_turn": card.get("cost_for_turn"),
                    "is_playable": card.get("is_playable"),
                    "upgrade_level": card.get("upgrade_level", card.get("upgrades")),
                    "damage": card.get("damage"),
                    "block": card.get("block"),
                }

            def _compact_action_local(idx: int, action: Any, *, score: float | None = None) -> dict[str, Any]:
                if not isinstance(action, dict):
                    return {"idx": int(idx), "repr": str(action), "score": score}
                semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
                card = action.get("card") if isinstance(action.get("card"), dict) else {}
                potion = action.get("potion") if isinstance(action.get("potion"), dict) else {}
                target = action.get("target") if isinstance(action.get("target"), dict) else {}
                family = self._semantic_family(action)
                return {
                    "idx": int(idx),
                    "is_chosen": int(idx) == int(chosen_idx),
                    "mask_legal": bool(idx < mask_np.shape[0] and mask_np[idx] > 0),
                    "family": family,
                    "kind": action.get("kind"),
                    "action_id": action.get("action_id"),
                    "title": action.get("title") or card.get("title") or potion.get("title") or potion.get("name"),
                    "card_id": card.get("id") or action.get("card_id"),
                    "card_title": card.get("title") or action.get("card_title"),
                    "potion_id": potion.get("id") or action.get("potion_id"),
                    "potion_title": potion.get("title") or potion.get("name"),
                    "cost": card.get("cost", action.get("cost")),
                    "target": target.get("name") or target.get("id") or action.get("target"),
                    "roles": semantic.get("roles") or semantic.get("tags"),
                    "semantic": {
                        "family": semantic.get("family") or semantic.get("action_kind"),
                        "domain": semantic.get("domain"),
                        "surface": semantic.get("surface"),
                        "selection": semantic.get("selection"),
                    } if semantic else {},
                    "damage": action.get("damage", card.get("damage")),
                    "block": action.get("block", card.get("block")),
                    "score": score,
                }

            legal_actions_compact: list[dict[str, Any]] = []
            selected_action_compact: dict[str, Any] = {}
            try:
                policy_np_full = (
                    np.asarray(search_policy, dtype=np.float32).reshape(-1)
                    if search_policy is not None
                    else np.zeros(MAX_ACTIONS, dtype=np.float32)
                )
                compact_count = min(len(legal_actions or []), MAX_ACTIONS, 32)
                for idx in range(compact_count):
                    score = float(policy_np_full[idx]) if idx < policy_np_full.shape[0] else None
                    row = _compact_action_local(idx, (legal_actions or [])[idx], score=score)
                    legal_actions_compact.append(row)
                    if idx == int(chosen_idx):
                        selected_action_compact = row
            except Exception:
                legal_actions_compact = []
                selected_action_compact = {}

            actionability_record: dict[str, Any] = {}
            try:
                raw_legal_action_count = int(pre_info.get("raw_legal_action_count", 0) or 0)
            except (TypeError, ValueError):
                raw_legal_action_count = 0
            try:
                blocked_action_drop_count = int(pre_info.get("blocked_action_drop_count", 0) or 0)
            except (TypeError, ValueError):
                blocked_action_drop_count = 0
            try:
                actionability_legal_non_end_turn_count = int(
                    pre_actionability.get(
                        "legal_non_end_turn_count",
                        pre_bridge_actionability.get("legal_non_end_turn_count", 0),
                    )
                    or 0
                )
            except (TypeError, ValueError):
                actionability_legal_non_end_turn_count = 0
            action_diag = action_diagnostics if isinstance(action_diagnostics, dict) else {}
            try:
                actionability_record = {
                    "transient_only_end_turn": bool(
                        pre_actionability.get(
                            "transient_only_end_turn",
                            pre_bridge_actionability.get("transient_only_end_turn", False),
                        )
                    ),
                    "frontier_stable": pre_actionability.get(
                        "frontier_stable",
                        pre_bridge_actionability.get("frontier_stable"),
                    ),
                    "legal_non_end_turn_count": actionability_legal_non_end_turn_count,
                    "raw_legal_action_count": raw_legal_action_count,
                    "blocked_action_drop_count": blocked_action_drop_count,
                    "combat_quality_prior_transient_only_end_turn": action_diag.get("combat_quality_prior_transient_only_end_turn"),
                    "combat_quality_post_step_frontier_leaked": action_diag.get("combat_quality_post_step_frontier_leaked"),
                    "combat_quality_post_step_frontier_suspicious_singleton": action_diag.get("combat_quality_post_step_frontier_suspicious_singleton"),
                    "frontier_pre_dispatch_end_turn_blocked": action_diag.get("frontier_pre_dispatch_end_turn_blocked"),
                }
            except Exception:
                actionability_record = {}

            positive_progress_count = int(context.get("positive_progress_count", 0) or 0)
            urgent_action_count = int(context.get("urgent_positive_count", 0) or 0)
            forced_no_non_end_turn_action = bool(class_name == "forced_end_turn" and legal_non_end_turn_count <= 0)
            benign_leftover_energy = bool(
                class_name == "forced_end_turn"
                and energy_val > 0.05
                and legal_non_end_turn_count <= 0
                and positive_progress_count <= 0
                and urgent_action_count <= 0
            )
            optional_only_nonurgent_action = bool(
                class_name != "bad_end_turn"
                and legal_non_end_turn_count > 0
                and urgent_action_count <= 0
            )
            singleton_end_turn_leak_suspect = bool(
                class_name == "forced_end_turn"
                and legal_non_end_turn_count <= 0
                and energy_val > 0.05
                and hand_count > 0
                and (
                    bool(actionability_record.get("transient_only_end_turn", False))
                    or actionability_record.get("frontier_stable") is False
                    or raw_legal_action_count <= 1
                    or blocked_action_drop_count > 0
                )
            )
            # Keep selected-EndTurn diagnostics reviewable: leftover energy after
            # all real actions are exhausted is a benign forced EndTurn, not an
            # empty-pass mistake.
            reason_flags = dict(reason_flags)
            reason_flags.update({
                "forced_no_non_end_turn_action": forced_no_non_end_turn_action,
                "benign_leftover_energy": benign_leftover_energy,
                "optional_only_nonurgent_action": optional_only_nonurgent_action,
                "singleton_end_turn_leak_suspect": singleton_end_turn_leak_suspect,
            })

            payload = {
                "time": time.time(),
                "global_step": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                "encounter_id": str(encounter or ""),
                "tier": str(tier or ""),
                "turn": raw_combat.get("round"),
                "selected_action_idx": int(chosen_idx),
                "selected_family": "end_turn",
                "end_turn_class": class_name,
                "reason_flags": reason_flags,
                "progress": {
                    "floor": float(floor_value),
                    "act_id": float(act_id_value),
                    "room_type": raw_run.get("room_type") or raw_run.get("room_type_name"),
                    "room_model": raw_run.get("room_model") or raw_run.get("room_id"),
                    "encounter_id_raw": raw_run.get("encounter_id") or raw_run.get("encounter"),
                    "screen": raw_obs.get("screen") if isinstance(raw_obs, dict) else None,
                    "phase": raw_obs.get("phase") if isinstance(raw_obs, dict) else None,
                    "state_version": raw_obs.get("state_version") if isinstance(raw_obs, dict) else None,
                },
                "player": {
                    "hp": float(hp),
                    "max_hp": float(self._safe_float(raw_player.get("max_hp"))) if isinstance(raw_player, dict) else 0.0,
                    "block": float(block),
                    "energy": float(energy_val),
                },
                "combat": {
                    "incoming_damage": float(incoming),
                    "hand_count": int(hand_count),
                    "draw_count": int(draw_count),
                    "discard_count": int(discard_count),
                    "exhaust_count": int(exhaust_count),
                },
                "counts": {
                    "legal_action_count": int(np.sum(mask_np > 0)),
                    "non_end_turn_action_count": int(legal_non_end_turn_count),
                    "playable_card_action_count": int(playable_card_action_count),
                    "playable_cards_left": int(len(context.get("positive_indices") or [])),
                    "positive_action_count": int(positive_progress_count),
                    "urgent_action_count": int(urgent_action_count),
                    "deferable_action_count": int(context.get("deferable_positive_count", 0) or 0),
                    "deferable_exhaust_count": int(context.get("deferable_exhaust_count", 0) or 0),
                    "x_cost_candidate_count": int(x_cost_count),
                    "zero_energy_x_cost_candidate_count": int(zero_x_count),
                    "potion_available_count": int(context.get("potion_available_count", 0) or 0),
                    "potion_urgent_count": int(context.get("potion_urgent_count", 0) or 0),
                    "raw_legal_action_count": int(raw_legal_action_count),
                    "blocked_action_drop_count": int(blocked_action_drop_count),
                    "actionability_legal_non_end_turn_count": int(actionability_legal_non_end_turn_count),
                },
                "actionability": actionability_record,
                "raw_hand_cards": [_compact_card_local(card) for card in (raw_hand or [])[:12]],
                "selected_action": selected_action_compact,
                "legal_actions_compact": legal_actions_compact,
                "boss_context": boss_context,
                "top_legal_actions": top_actions,
            }
            path = self._diagnostic_jsonl_path("end_turn_contexts.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._end_turn_context_dump_count += 1
        except Exception as exc:
            _report_diagnostic_failure(self, "end_turn_contexts", exc)
            return

    def _dump_loss_spike(
        self,
        *,
        step_k: int,
        future_world_aux_value: float,
        future_bank_state_value: float,
        future_bank_delta_value: float,
        batch_size: int | None,
        threshold: float = 30.0,
        max_dumps: int = 200,
        extra_losses: dict[str, float] | None = None,
        action_indices: Any = None,
        sample_tier_flags: dict[str, Any] | None = None,
    ) -> bool:
        """Dump a loss-spike record to ``diagnostics/loss_spikes.jsonl``.

        Triggered when any of the future_world / future_bank loss components
        exceeds ``threshold`` (default 100).  Capped at ``max_dumps`` per run
        so a runaway divergent batch cannot fill the disk.

        Per the P0-7 hardening spec the dump now also carries:

        * ``losses``: all related loss values (the three primary heads plus
          any extras the caller passes in).
        * ``trigger``: which loss component(s) crossed the threshold and the
          dominant component's name + value.
        * ``finite_guard``: NaN / Inf detection — when any value is
          non-finite the record is still written (with the offending values
          serialized as strings) and a ``quarantine_reason`` is set so the
          downstream offline analysis knows to discard the polluted update.
        * ``action_index_dist``: top action-index frequencies inside the
          batch — coarse handle on which action mix the offending batch was
          dominated by, available even before per-sample buffer metadata
          lands (deferred to a follow-up).
        * ``sample_tier``: tier-flag means (boss / elite / normal / weak)
          when the caller forwards the buffer's sample-tier flags so we can
          tell which encounter category is over-represented in spike batches.
        """
        try:
            primary_values = {
                "future_world_aux_loss": float(future_world_aux_value),
                "future_bank_state_loss": float(future_bank_state_value),
                "future_bank_delta_loss": float(future_bank_delta_value),
            }
            losses_block: dict[str, float] = dict(primary_values)
            if extra_losses:
                for key, val in extra_losses.items():
                    try:
                        losses_block[str(key)] = float(val)
                    except (TypeError, ValueError):
                        losses_block[str(key)] = float("nan")

            non_finite: dict[str, float] = {}
            for key, val in losses_block.items():
                if not math.isfinite(val):
                    non_finite[key] = val
            crossed = [k for k, v in primary_values.items() if math.isfinite(v) and v > threshold]
            if not crossed and not non_finite:
                return False
            count = getattr(self, "_loss_spike_dump_count", 0)
            if count >= max_dumps:
                # Spike still happened — caller must still skip the optimizer
                # step to avoid contamination. We just stop writing dumps.
                return True

            dominant_key = ""
            dominant_val = 0.0
            for k, v in primary_values.items():
                if math.isfinite(v) and v > dominant_val:
                    dominant_val = v
                    dominant_key = k

            quarantine_reason: list[str] = []
            if non_finite:
                quarantine_reason.append("non_finite_loss")
            if crossed:
                quarantine_reason.append("loss_above_threshold")

            action_index_dist: list[dict[str, int]] = []
            try:
                if action_indices is not None:
                    arr = action_indices.detach().cpu().tolist() if hasattr(action_indices, "detach") else list(action_indices)
                    counter: dict[int, int] = {}
                    for idx in arr:
                        try:
                            ki = int(idx)
                        except (TypeError, ValueError):
                            continue
                        counter[ki] = counter.get(ki, 0) + 1
                    top = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
                    action_index_dist = [{"action_index": k, "count": v} for k, v in top]
            except Exception:
                action_index_dist = []

            tier_summary: dict[str, float] = {}
            if isinstance(sample_tier_flags, dict):
                for tier_name, tensor in sample_tier_flags.items():
                    try:
                        if hasattr(tensor, "detach"):
                            tier_summary[tier_name] = float(tensor.detach().mean().item())
                        else:
                            tier_summary[tier_name] = float(sum(float(v) for v in tensor) / max(len(tensor), 1))
                    except Exception:
                        continue

            def _serialize(value: float) -> Any:
                return value if math.isfinite(value) else str(value)

            payload: dict[str, Any] = {
                "kind": "loss_spike",
                "schema_version": 2,
                "total_steps": int(getattr(self, "total_steps", 0)),
                "step_k": int(step_k),
                "batch_size": batch_size,
                "threshold": float(threshold),
                "wall_time": time.time(),
                "losses": {k: _serialize(v) for k, v in losses_block.items()},
                "trigger": {
                    "crossed_keys": crossed,
                    "dominant_key": dominant_key,
                    "dominant_value": _serialize(dominant_val),
                    "non_finite_keys": list(non_finite.keys()),
                },
                "finite_guard": {
                    "all_finite": not non_finite,
                    "non_finite_count": len(non_finite),
                },
                "quarantine_reason": quarantine_reason,
                # Legacy top-level fields kept for back-compat with existing
                # offline tooling reading v1 dumps.
                "future_world_aux_loss": _serialize(future_world_aux_value),
                "future_bank_state_loss": _serialize(future_bank_state_value),
                "future_bank_delta_loss": _serialize(future_bank_delta_value),
            }
            if action_index_dist:
                payload["action_index_dist"] = action_index_dist
            if tier_summary:
                payload["sample_tier"] = tier_summary

            path = _diagnostic_jsonl_path_for(self, "loss_spikes.jsonl")
            if path is None:
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._loss_spike_dump_count = count + 1
            tag = "non_finite" if non_finite else dominant_key.replace("_loss", "")
            print(
                f"[loss_spike] dump #{self._loss_spike_dump_count} step={payload['total_steps']} "
                f"step_k={step_k} dominant={tag} value={_serialize(dominant_val)} "
                f"reasons={quarantine_reason}",
                flush=True,
            )
            return True
        except Exception as exc:
            _report_diagnostic_failure(self, "loss_spikes", exc)
            return False

    def _dump_episode_end_anomaly(
        self,
        *,
        episode_reward: float,
        episode_length: int,
        terminated: bool,
        truncated: bool,
        max_floor: float,
        max_act_id: float,
        death_floor: float,
        act1_boss_seen: bool,
        act1_clear: bool,
        final_info: dict[str, Any] | None,
        progress_snapshots: list[dict[str, Any]],
        episode_telemetry: dict[str, float],
        trigger_reasons: list[str] | None = None,
        trajectory_steps: list[dict[str, Any]] | None = None,
        max_dumps: int = 500,
    ) -> None:
        """Dump compact episode-end records when Act1 signals disagree.

        This is deliberately on the episode boundary, not the hot step path.
        It gives us enough evidence to separate "actually died to boss" from
        bridge/telemetry contract bugs without turning every run into a giant
        raw-observation log.
        """
        try:
            count = int(getattr(self, "_episode_end_anomaly_dump_count", 0))
            if count >= int(max_dumps):
                return

            def _trim(value: Any, depth: int = 0) -> Any:
                if depth >= 4:
                    return "<truncated>"
                if isinstance(value, dict):
                    out: dict[str, Any] = {}
                    for idx, (key, val) in enumerate(value.items()):
                        if idx >= 80:
                            out["<more_keys>"] = len(value) - idx
                            break
                        out[str(key)] = _trim(val, depth + 1)
                    return out
                if isinstance(value, list):
                    return [_trim(v, depth + 1) for v in value[:40]] + (
                        [{"<more_items>": len(value) - 40}] if len(value) > 40 else []
                    )
                return value

            env_unwrapped = getattr(self.env, "unwrapped", self.env)
            raw_obs = getattr(env_unwrapped, "_last_obs_raw", None)
            raw_summary: dict[str, Any] = {}
            if isinstance(raw_obs, dict):
                for key in ("phase", "run", "player", "combat"):
                    if key in raw_obs:
                        raw_summary[key] = _trim(raw_obs.get(key))

            decision_tail: list[dict[str, Any]] = []
            try:
                for step in list(trajectory_steps or [])[-48:]:
                    if not isinstance(step, dict):
                        continue
                    action_info = step.get("action_info") if isinstance(step.get("action_info"), dict) else {}
                    semantic = action_info.get("semantic") if isinstance(action_info.get("semantic"), dict) else {}
                    family = str(semantic.get("family") or action_info.get("kind") or "")
                    phase = str(action_info.get("phase") or "")
                    # Keep all non-combat build/route decisions plus the last combat
                    # actions. This is compact enough for JSONL and is the missing
                    # evidence for floor 13/15 deaths: what cards/rest/shop choices
                    # led to the weak build.
                    decision_tail.append({
                        "family": family,
                        "phase": phase,
                        "title": action_info.get("title"),
                        "card_id": action_info.get("card_id"),
                        "card_cost": action_info.get("card_cost"),
                        "card_type": action_info.get("card_type"),
                        "reward_type": action_info.get("reward_type"),
                        "shop_action": action_info.get("shop_action"),
                        "selected_index": action_info.get("selected_index"),
                        "legal_action_count": action_info.get("legal_action_count"),
                        "reward": step.get("reward"),
                        "wasteful_end_turn": step.get("wasteful_end_turn"),
                    })
            except Exception:
                decision_tail = []

            payload = {
                "kind": "episode_end_anomaly",
                "schema_version": 1,
                "wall_time": time.time(),
                "episode_count": int(getattr(self, "episode_count", 0)),
                "total_steps": int(getattr(self, "total_steps", 0)),
                "episode_reward": float(episode_reward),
                "episode_length": int(episode_length),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "max_floor": float(max_floor),
                "max_act_id": float(max_act_id),
                "death_floor": float(death_floor),
                "act1_boss_seen": bool(act1_boss_seen),
                "act1_clear": bool(act1_clear),
                "trigger_reasons": list(trigger_reasons or []),
                "progress_tail": _trim(progress_snapshots[-12:]),
                "final_transition_state": _trim(
                    final_info.get("transition_state") if isinstance(final_info, dict) else None
                ),
                "final_phase": final_info.get("phase") if isinstance(final_info, dict) else None,
                "episode_telemetry": dict(episode_telemetry),
                "decision_tail": _trim(decision_tail),
                "final_raw_obs_summary": raw_summary,
            }
            path = self._diagnostic_jsonl_path("episode_end_anomalies.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._episode_end_anomaly_dump_count = count + 1
        except Exception:
            return

    def _dump_potion_transition(self, record: dict[str, Any] | None) -> None:
        """Append one JSONL line per use_potion transition emitted by combat_env."""

        if not isinstance(record, dict):
            return
        if getattr(self, "_potion_transition_dump_disabled", False):
            return
        cap = int(getattr(self, "_potion_transition_dump_max", 200000))
        if cap > 0 and getattr(self, "_potion_transition_dump_count", 0) >= cap:
            return
        try:
            payload = {
                "time": time.time(),
                "global_step": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                **record,
            }
            path = self._diagnostic_jsonl_path("potion_transitions.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._potion_transition_dump_count += 1
        except Exception:
            return

    def _dump_frontier_trace(self, record: dict[str, Any] | None) -> None:
        """Append one post-step frontier short-poll trace from combat_env.

        The trace is intentionally compact: it records the immediate
        bridge.step frontier, each short /state poll summary, and an optional
        soft-rebind frontier.  This lets us separate "really no playable cards"
        from "hand/action frontier was still settling" without dumping full raw
        observations on every End Turn.
        """

        if not isinstance(record, dict):
            return
        if getattr(self, "_frontier_trace_dump_disabled", False):
            return
        cap = int(getattr(self, "_frontier_trace_dump_max", 200000))
        if cap > 0 and getattr(self, "_frontier_trace_dump_count", 0) >= cap:
            return
        try:
            payload = {
                "time": time.time(),
                "global_step": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                **record,
            }
            path = self._diagnostic_jsonl_path("frontier_trace.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._frontier_trace_dump_count = int(getattr(self, "_frontier_trace_dump_count", 0) or 0) + 1
        except Exception:
            return

    def _dump_death_final_potions(
        self,
        *,
        loss: bool,
        final_potion_count: int,
        used_potion_count: int,
        adjusted_unused: int,
        encounter: str,
        tier: str,
        potion_dump: list[dict[str, Any]] | None,
    ) -> None:
        """Append a death-final potion accounting line to the same JSONL.

        Lets post-mortem confirm whether ``potion_unused_on_death`` flipped
        because of a missing transition record vs a real hoarded potion.
        """

        if getattr(self, "_potion_transition_dump_disabled", False):
            return
        cap = int(getattr(self, "_potion_transition_dump_max", 200000))
        if cap > 0 and getattr(self, "_potion_transition_dump_count", 0) >= cap:
            return
        try:
            payload = {
                "time": time.time(),
                "global_step": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                "event": "death_final_potions",
                "loss": bool(loss),
                "encounter_id": str(encounter or ""),
                "tier": str(tier or ""),
                "final_potion_count": int(final_potion_count),
                "used_potion_count": int(used_potion_count),
                "adjusted_unused": int(adjusted_unused),
                "potion_dump": list(potion_dump or []),
            }
            path = self._diagnostic_jsonl_path("potion_transitions.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._potion_transition_dump_count += 1
        except Exception:
            return

    @staticmethod
    def _classify_action_offenders(
        *,
        search_stats: dict[str, Any] | None,
        encounter: str,
        family: str,
    ) -> list[str]:
        """Return the offender types triggered by the selected action's stats.

        Pure function over stats already populated by ``_combat_action_quality_bias``
        and ``_selected_combat_quality_stats``.  Boss-specific offenders are gated
        by ``encounter`` so e.g. ``kaiser_*`` will never fire on a Construct fight.
        """

        s = search_stats if isinstance(search_stats, dict) else {}
        encounter_l = str(encounter or "").lower()
        family_l = str(family or "").lower()

        def gt(key: str, threshold: float = 0.5) -> bool:
            try:
                return float(s.get(key, 0.0) or 0.0) > threshold
            except (TypeError, ValueError):
                return False

        out: list[str] = []
        # ``combat_quality_wasteful_end_turn_selected`` is a legacy/broad
        # signal: it can include soft "left energy on the table" situations
        # where the strict EndTurn taxonomy later classifies the choice as
        # forced/unknown.  Do not surface that broad alias as a hard
        # ``bad_end_turn`` offender, otherwise diagnostics imply that benign
        # "cards are gone but energy remains" turns are model bugs.
        #
        # The strict hard-bad source of truth is
        # ``combat_quality_bad_end_turn_selected``; legacy wasteful rows remain
        # inspectable under a softer label.
        if gt("combat_quality_bad_end_turn_selected"):
            out.append("bad_end_turn")
        elif gt("combat_quality_wasteful_end_turn_selected"):
            out.append("soft_or_ambiguous_end_turn")
        if gt("combat_quality_strategic_defer_end_turn_selected"):
            out.append("strategic_defer_end_turn")
        if gt("combat_quality_strategic_skip_selected"):
            out.append("strategic_skip_selected")
        if gt("combat_quality_refund_no_followup_with_progress_selected"):
            out.append("refund_no_followup_with_progress_selected")
        if gt("combat_quality_refund_no_followup_selected"):
            out.append("refund_no_followup_selected")
        # Emit narrow pure-block buckets before the broad selected signal.
        # The broad ``card_pure_block_selected`` offender is useful for volume,
        # but the sandbox gate is keyed on ``bad_pure_block_selected``; without
        # these tags the offender file mostly says "Defend was played" and
        # hides whether it was actually bad, survival-justified, or no-alt.
        if gt("combat_quality_bad_pure_block_selected"):
            out.append("bad_pure_block_selected")
        if gt("combat_quality_insufficient_block_selected"):
            out.append("insufficient_block_selected")
        if gt("combat_quality_pure_block_progress_alternative_selected"):
            out.append("pure_block_progress_alternative_selected")
        if gt("combat_quality_pure_block_low_value_pressure_selected"):
            out.append("pure_block_low_value_pressure_selected")
        if gt("combat_quality_pure_block_no_alternative_selected"):
            out.append("pure_block_no_alternative_selected")
        if gt("combat_quality_pure_block_survival_justified_selected"):
            out.append("pure_block_survival_justified_selected")
        if gt("combat_quality_card_block_waste_selected"):
            out.append("card_block_waste_selected")
        if gt("combat_quality_card_pure_block_selected"):
            out.append("card_pure_block_selected")
        if gt("combat_quality_card_no_damage_pressure_selected"):
            out.append("card_no_damage_pressure_selected")
        if gt("combat_quality_zero_energy_x_cost_selected"):
            out.append("zero_energy_x_cost_selected")
        if gt("combat_quality_x_cost_bad_selected"):
            out.append("x_cost_low_value_selected")
        if gt("combat_quality_hp_cost_self_lethal_selected"):
            out.append("hp_cost_self_lethal_selected")
        if gt("combat_quality_hp_cost_low_margin_selected"):
            out.append("hp_cost_low_margin_selected")
        if family_l in {"use_potion", "potion"}:
            if (
                gt("combat_quality_potion_low_urgency_selected")
                or gt("combat_quality_potion_save_recommended_selected")
                or gt("combat_quality_potion_no_followup_selected")
                or gt("combat_quality_potion_block_waste_selected")
                or gt("combat_quality_potion_overkill_selected")
            ):
                out.append("low_quality_potion_selected")
        # End-turn while genuinely urgent potion answers exist.
        #
        # Do *not* treat ``potion_save_recommended_*`` as an unused-potion
        # offender.  That signal means the potion classifier believes holding
        # the potion is strategically valuable in the current low-threat state.
        # The old ``high_save_value_potion_unused`` tag inverted that semantic
        # and produced noisy offenders on high-HP / zero-incoming end turns.
        if family_l == "end_turn":
            if (
                gt("combat_quality_prior_transient_only_end_turn")
                or gt("combat_quality_post_step_frontier_leaked")
                or gt("combat_quality_post_step_frontier_suspicious_singleton")
                or gt("frontier_pre_dispatch_end_turn_blocked")
            ):
                out.append("singleton_end_turn_leak")
            try:
                urgent_potions = (
                    float(s.get("combat_quality_potion_urgent_count", 0.0) or 0.0)
                    + float(s.get("combat_quality_potion_lethal_count", 0.0) or 0.0)
                    + float(s.get("combat_quality_potion_prevent_lethal_count", 0.0) or 0.0)
                    + float(s.get("combat_quality_potion_mechanism_count", 0.0) or 0.0)
                )
                if urgent_potions >= 1.0:
                    out.append("urgent_potion_unused")
            except (TypeError, ValueError):
                pass

        is_kaiser = "kaiser" in encounter_l
        if is_kaiser:
            risk = float(s.get("combat_quality_kaiser_back_attack_risk", 0.0) or 0.0)
            if risk > 0.05:
                if gt("combat_quality_kaiser_risky_end_turn_selected"):
                    out.append("kaiser_risky_end_turn")
                facing_cands = float(s.get("combat_quality_kaiser_facing_change_candidate_count", 0.0) or 0.0)
                facing_sel = float(s.get("combat_quality_kaiser_facing_change_selected", 0.0) or 0.0)
                if facing_cands >= 1.0 and facing_sel < 0.5 and family_l != "end_turn":
                    out.append("kaiser_facing_missed")

        if "ceremonial" in encounter_l:
            if gt("combat_quality_ceremonial_low_impact_selected"):
                out.append("ceremonial_low_impact_under_lock")
            # P1-4 (recovery 2026-05-07): suppress false-positive
            # missed_stun_window. The detector at boss_ceremonial.py:295
            # marks the window as missed any time stun_window=True and the
            # selected action is end_turn / play_card without a high-impact
            # tag. That fires even when:
            #   - the policy is forced into end_turn (legal=1, no positive
            #     action playable)
            #   - the legal hand has zero high-impact alternatives
            # Per spec §P1-4 only emit when the agent had a real choice.
            stun_window = float(s.get("combat_quality_ceremonial_stun_window", 0.0) or 0.0)
            high_impact_sel = float(s.get("combat_quality_ceremonial_high_impact_selected", 0.0) or 0.0)
            high_impact_count = float(s.get("combat_quality_ceremonial_high_impact_count", 0.0) or 0.0)
            positive_count = float(s.get("combat_quality_positive_action_count", 0.0) or 0.0)
            forced_end_turn_avail = float(s.get("combat_quality_forced_end_turn_available", 0.0) or 0.0)
            if stun_window > 0.05 and high_impact_sel < 0.5:
                if (
                    high_impact_count >= 1.0
                    and positive_count > 0.0
                    and forced_end_turn_avail < 0.5
                ):
                    out.append("ceremonial_missed_stun_window")

        if "insatiable" in encounter_l:
            if gt("combat_quality_insatiable_frantic_escape_missed_at_1"):
                out.append("insatiable_frantic_escape_missed_at_1")
            elif gt("combat_quality_insatiable_frantic_escape_missed_lt3"):
                out.append("insatiable_frantic_escape_missed_lt3")
            if gt("combat_quality_insatiable_non_escape_at_1_selected"):
                out.append("insatiable_non_escape_at_1_nonlethal")
            if gt("combat_quality_strategic_skip_selected"):
                out.append("insatiable_strategic_skip")
            if gt("combat_quality_refund_no_followup_with_progress_selected") or gt("combat_quality_refund_no_followup_selected"):
                out.append("insatiable_strategic_skip")
            if gt("combat_quality_wasteful_end_turn_selected"):
                out.append("insatiable_strategic_skip")

        # Preserve insertion order on dedup so primary offenders rank first.
        seen: set[str] = set()
        deduped: list[str] = []
        for item in out:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _dump_action_offender(
        self,
        *,
        encoded_obs: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None,
        action_mask: np.ndarray,
        legal_actions: list[Any] | None,
        chosen_idx: int,
        chosen_action: Any,
        offender_types: list[str],
        search_stats: dict[str, Any] | None,
        encounter: str,
        tier: str,
    ) -> None:
        """Append one JSONL entry per (step, offender_type) pair."""

        if not offender_types:
            return
        if getattr(self, "_action_offender_dump_disabled", False):
            return
        cap = int(getattr(self, "_action_offender_dump_max", 100000))
        if cap > 0 and getattr(self, "_action_offender_dump_count", 0) >= cap:
            return
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
            raw_combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
            raw_player = raw_obs.get("player") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("player"), dict) else {}
            if not raw_player and isinstance(raw_combat.get("player"), dict):
                raw_player = raw_combat.get("player")
            floor_value, act_id_value = _progress_values_from_obs(self, raw_obs)
            incoming, block, hp = self._incoming_damage_pressure(raw_obs)
            family = self._semantic_family(chosen_action) if isinstance(chosen_action, dict) else ""
            card = chosen_action.get("card") if isinstance(chosen_action, dict) and isinstance(chosen_action.get("card"), dict) else {}
            potion = chosen_action.get("potion") if isinstance(chosen_action, dict) and isinstance(chosen_action.get("potion"), dict) else {}
            target = chosen_action.get("target") if isinstance(chosen_action, dict) and isinstance(chosen_action.get("target"), dict) else {}
            stats = search_stats if isinstance(search_stats, dict) else {}
            effective_tier = str(tier or "").strip()
            if not effective_tier:
                try:
                    effective_tier = str(self._combat_encounter_tier_from_raw(raw_obs) or "").strip()
                except Exception:
                    effective_tier = ""
            state_summary = {
                "floor": float(floor_value),
                "act_id": float(act_id_value),
                "energy": float(stats.get("combat_quality_energy", 0.0) or 0.0),
                "incoming_damage": float(incoming),
                "block": float(block),
                "hp": float(hp),
                "max_hp": float(self._safe_float(raw_player.get("max_hp"))) if isinstance(raw_player, dict) else 0.0,
                "legal_action_count": int(np.sum(mask_np > 0)),
                "positive_action_count": float(stats.get("combat_quality_positive_action_count", 0.0) or 0.0),
                "urgent_positive_action_count": float(stats.get("combat_quality_urgent_positive_action_count", 0.0) or 0.0),
                "deferable_positive_action_count": float(stats.get("combat_quality_deferable_positive_action_count", 0.0) or 0.0),
                "x_cost_available_count": float(stats.get("combat_quality_x_cost_available_count", 0.0) or 0.0),
                "zero_energy_x_cost_count": float(stats.get("combat_quality_zero_energy_x_cost_count", 0.0) or 0.0),
                "hp_cost_unblockable_value": float(stats.get("combat_quality_hp_cost_unblockable_value", 0.0) or 0.0),
                "hp_cost_self_lethal_selected": float(stats.get("combat_quality_hp_cost_self_lethal_selected", 0.0) or 0.0),
                "hp_cost_low_margin_selected": float(stats.get("combat_quality_hp_cost_low_margin_selected", 0.0) or 0.0),
                "card_block_waste_count": float(stats.get("combat_quality_card_block_waste_count", 0.0) or 0.0),
                "card_pure_block_count": float(stats.get("combat_quality_card_pure_block_count", 0.0) or 0.0),
                "card_no_damage_pressure_count": float(stats.get("combat_quality_card_no_damage_pressure_count", 0.0) or 0.0),
                "bad_pure_block_selected": float(stats.get("combat_quality_bad_pure_block_selected", 0.0) or 0.0),
                "insufficient_block_selected": float(stats.get("combat_quality_insufficient_block_selected", 0.0) or 0.0),
                "pure_block_progress_alternative_selected": float(stats.get("combat_quality_pure_block_progress_alternative_selected", 0.0) or 0.0),
                "pure_block_survival_justified_selected": float(stats.get("combat_quality_pure_block_survival_justified_selected", 0.0) or 0.0),
                "pure_block_no_alternative_selected": float(stats.get("combat_quality_pure_block_no_alternative_selected", 0.0) or 0.0),
                "pure_block_low_value_pressure_selected": float(stats.get("combat_quality_pure_block_low_value_pressure_selected", 0.0) or 0.0),
                "refund_no_followup_available": float(stats.get("combat_quality_refund_no_followup_available", 0.0) or 0.0),
                "refund_no_followup_guard_candidate_count": float(stats.get("combat_quality_refund_no_followup_guard_candidate_count", 0.0) or 0.0),
                "refund_no_followup_guard_no_alternative": float(stats.get("combat_quality_refund_no_followup_guard_no_alternative", 0.0) or 0.0),
                "refund_no_followup_progress_alternative_count": float(stats.get("combat_quality_refund_no_followup_progress_alternative_count", 0.0) or 0.0),
                "kaiser_back_attack_risk": float(stats.get("combat_quality_kaiser_back_attack_risk", 0.0) or 0.0),
                "kaiser_facing_change_candidate_count": float(stats.get("combat_quality_kaiser_facing_change_candidate_count", 0.0) or 0.0),
                "ceremonial_one_card_lock": float(stats.get("combat_quality_ceremonial_one_card_lock", 0.0) or 0.0),
                "ceremonial_stun_window": float(stats.get("combat_quality_ceremonial_stun_window", 0.0) or 0.0),
                "insatiable_sandpit_countdown": float(stats.get("combat_quality_insatiable_sandpit_countdown", 0.0) or 0.0),
                "insatiable_frantic_escape_available": float(stats.get("combat_quality_insatiable_frantic_escape_available", 0.0) or 0.0),
                "insatiable_escape_cycle_risk": float(stats.get("combat_quality_insatiable_escape_cycle_risk", 0.0) or 0.0),
            }
            guard_bits = {
                "no_pressure_available": float(stats.get("combat_quality_no_pressure_block_guard_available", 0.0) or 0.0),
                "no_pressure_applied": float(stats.get("combat_quality_no_pressure_block_guard_applied", 0.0) or 0.0),
                "no_pressure_override": float(stats.get("combat_quality_no_pressure_block_guard_override", 0.0) or 0.0),
                "no_pressure_candidate_count": float(stats.get("combat_quality_no_pressure_block_guard_candidate_count", 0.0) or 0.0),
                "no_pressure_no_alternative": float(stats.get("combat_quality_no_pressure_block_guard_no_alternative", 0.0) or 0.0),
                "no_pressure_progress_override_idx": float(stats.get("combat_quality_no_pressure_block_guard_progress_override_idx", -1.0) or -1.0),
                "no_pressure_progress_override_lock": float(stats.get("combat_quality_no_pressure_block_guard_progress_override_lock", 0.0) or 0.0),
                "survival_override": float(stats.get("combat_quality_survival_non_endturn_guard_override", 0.0) or 0.0),
                "survival_candidate_count": float(stats.get("combat_quality_survival_non_endturn_guard_candidate_count", 0.0) or 0.0),
                "hard_guard_override_any": float(stats.get("combat_quality_hard_guard_override_any", 0.0) or 0.0),
            }
            reason_flags = {
                key: float(stats.get(f"combat_quality_{key}", 0.0) or 0.0) > 0.5
                for key in (
                    "wasteful_end_turn_selected",
                    "strategic_defer_end_turn_selected",
                    "strategic_skip_selected",
                    "refund_no_followup_selected",
                    "refund_no_followup_with_progress_selected",
                    "refund_no_followup_progress_alternative_selected",
                    "refund_no_followup_no_alternative_selected",
                    "card_block_waste_selected",
                    "card_pure_block_selected",
                    "card_no_damage_pressure_selected",
                    "bad_pure_block_selected",
                    "insufficient_block_selected",
                    "pure_block_progress_alternative_selected",
                    "pure_block_survival_justified_selected",
                    "pure_block_no_alternative_selected",
                    "pure_block_low_value_pressure_selected",
                    "zero_energy_x_cost_selected",
                    "hp_cost_self_lethal_selected",
                    "hp_cost_low_margin_selected",
                    "potion_low_urgency_selected",
                    "potion_save_recommended_selected",
                    "potion_no_followup_selected",
                    "potion_block_waste_selected",
                    "potion_overkill_selected",
                    "kaiser_facing_change_selected",
                    "kaiser_pressure_selected",
                    "kaiser_risky_end_turn_selected",
                    "ceremonial_low_impact_selected",
                    "ceremonial_high_impact_selected",
                    "insatiable_frantic_escape_selected",
                    "insatiable_frantic_escape_missed_lt3",
                    "insatiable_frantic_escape_missed_at_1",
                    "insatiable_non_escape_at_1_selected",
                )
            }

            def _action_debug_summary(idx: int, action: Any, *, chosen: bool = False) -> dict[str, Any]:
                if not isinstance(action, dict):
                    return {"action_idx": int(idx), "is_chosen": bool(chosen), "invalid": True}
                action_card = action.get("card") if isinstance(action.get("card"), dict) else {}
                action_potion = action.get("potion") if isinstance(action.get("potion"), dict) else {}
                action_target = action.get("target") if isinstance(action.get("target"), dict) else {}
                family_i = self._semantic_family(action)
                try:
                    energy_for_profile = float(
                        stats.get("combat_quality_energy", 0.0)
                        or self._combat_energy(encoded_obs, raw_obs)
                        or 0.0
                    )
                except Exception:
                    energy_for_profile = 0.0
                try:
                    profile = self._classify_positive_combat_action(
                        action,
                        int(idx),
                        encoded_obs,
                        raw_obs,
                        legal_actions if isinstance(legal_actions, list) else [],
                        mask_np,
                        energy_for_profile,
                    )
                except Exception:
                    profile = {}
                try:
                    x_diag = self._x_cost_diagnostic(action, energy_for_profile)
                except Exception:
                    x_diag = {}
                try:
                    from sts2_env.hp_cost_safety import hp_cost_safety_view

                    hp_safety = hp_cost_safety_view(action, raw_obs)
                except Exception:
                    hp_safety = {}
                try:
                    lethal = bool(self._is_action_confirmed_lethal(action, raw_obs))
                except Exception:
                    lethal = False
                return {
                    "action_idx": int(idx),
                    "mask": float(mask_np[int(idx)]) if 0 <= int(idx) < mask_np.shape[0] else 0.0,
                    "is_chosen": bool(chosen),
                    "family": family_i,
                    "kind": action.get("kind"),
                    "action_id": action.get("action_id"),
                    "card_id": action_card.get("id") if isinstance(action_card, dict) else None,
                    "potion_id": action_potion.get("id") if isinstance(action_potion, dict) else None,
                    "title": action.get("title")
                    or (action_card.get("title") if isinstance(action_card, dict) else None)
                    or (action_potion.get("title") if isinstance(action_potion, dict) else None),
                    "target": action_target.get("name") if isinstance(action_target, dict) else None,
                    "target_combat_id": self._action_target_combat_id(action),
                    "cost": float(self._action_cost_value(action)) if family_i == "play_card" else 0.0,
                    "damage": float(
                        max(
                            self._action_metric(action, "damage"),
                            self._action_metric(action, "total_damage"),
                            self._action_numeric_value(
                                action,
                                ("damage", "total_damage", "attack_damage", "preview_damage", "expected_damage"),
                            ),
                        )
                    ),
                    "block": float(
                        max(
                            self._action_metric(action, "block"),
                            self._action_metric(action, "total_block"),
                            self._action_numeric_value(
                                action,
                                ("block", "total_block", "preview_block", "expected_block"),
                            ),
                        )
                    ),
                    "impact": float(self._action_immediate_impact(action)),
                    "roles": sorted(self._action_roles(action)),
                    "profile_flags": {
                        key: bool(profile.get(key, False))
                        for key in (
                            "positive",
                            "urgent",
                            "deferable",
                            "followup_missing",
                            "energy_without_followup",
                            "setup_followup_dependent",
                            "setup_followup_available",
                            "card_block_waste",
                            "card_pure_block",
                            "card_no_damage_pressure",
                        )
                    },
                    "x_cost_bad": float(x_diag.get("x_cost_bad", 0.0) or 0.0),
                    "hp_loss_unblockable": float(hp_safety.get("hp_loss_unblockable", 0.0) or 0.0),
                    "confirmed_lethal": bool(lethal),
                }

            alternatives: list[dict[str, Any]] = []
            try:
                count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
                for idx in range(count):
                    if idx == int(chosen_idx) or mask_np[idx] <= 0:
                        continue
                    cand = (legal_actions or [])[idx]
                    if not isinstance(cand, dict):
                        continue
                    alternatives.append(_action_debug_summary(int(idx), cand, chosen=False))
                    if len(alternatives) >= 12:
                        break
            except Exception:
                alternatives = []
            try:
                selected_detail = _action_debug_summary(int(chosen_idx), chosen_action, chosen=True)
            except Exception:
                selected_detail = {}
            path = self._diagnostic_jsonl_path("action_offenders.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            for offender_type in offender_types:
                payload = {
                    "time": time.time(),
                    "global_step": int(getattr(self, "total_steps", 0)),
                    "episode_id": int(getattr(self, "episode_count", 0)),
                    "floor": float(floor_value),
                    "act_id": float(act_id_value),
                    "encounter_id": str(encounter or ""),
                    "tier": effective_tier,
                    "turn": raw_combat.get("round"),
                    "offender_type": offender_type,
                    "selected_action_idx": int(chosen_idx),
                    "selected_action_id": chosen_action.get("action_id") if isinstance(chosen_action, dict) else None,
                    "selected_family": family,
                    "selected_card_id": card.get("id") if isinstance(card, dict) else None,
                    "selected_potion_id": potion.get("id") if isinstance(potion, dict) else None,
                    "selected_title": (chosen_action.get("title") if isinstance(chosen_action, dict) else None)
                        or (card.get("title") if isinstance(card, dict) else None)
                        or (potion.get("title") if isinstance(potion, dict) else None),
                    "selected_target": target.get("name") if isinstance(target, dict) else None,
                    "reason_flags": reason_flags,
                    "guard_bits": guard_bits,
                    "state_summary": state_summary,
                    "selected_action_detail": selected_detail,
                    "alternative_actions": alternatives,
                }
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                self._action_offender_dump_count = int(getattr(self, "_action_offender_dump_count", 0)) + 1
                if cap > 0 and self._action_offender_dump_count >= cap:
                    return
        except Exception as exc:
            _report_diagnostic_failure(self, "action_offenders", exc)
            return
