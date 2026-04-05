"""Utilities for parsing Slay the Spire 2 native .run history files.

The native history files are JSON payloads written by the game under:
    .../modded/profile1/saves/history/*.run

This module extracts three practical views:
    - summary: run-level metadata
    - floors: one normalized record per map-point/player-stats entry
    - decisions: flattened decision-centric records for offline analysis
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_run_history(path: str | Path) -> dict[str, Any]:
    run_path = Path(path)
    return json.loads(run_path.read_text(encoding="utf-8"))


def load_run_history_bytes(payload: bytes | str) -> dict[str, Any]:
    if isinstance(payload, bytes):
        text = payload.decode("utf-8-sig")
    else:
        text = payload
    return json.loads(text)


def extract_run_history(path: str | Path) -> dict[str, Any]:
    run_path = Path(path)
    raw = load_run_history(run_path)
    return _extract_run_history_from_loaded(raw, run_path)


def extract_run_history_bytes(payload: bytes | str, source_name: str) -> dict[str, Any]:
    raw = load_run_history_bytes(payload)
    return _extract_run_history_from_loaded(raw, Path(source_name))


def _extract_run_history_from_loaded(raw: dict[str, Any], source_path: Path) -> dict[str, Any]:
    summary = _extract_summary(raw, source_path)
    final_build = _extract_final_build(raw)
    floors = _extract_floor_records(raw, summary)
    decisions = _extract_decision_records(floors)
    return {
        "summary": summary,
        "final_build": final_build,
        "floors": floors,
        "decisions": decisions,
    }


def write_extracted_run_history(bundle: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    final_build_path = out_dir / "final_build.json"
    floors_path = out_dir / "floors.jsonl"
    decisions_path = out_dir / "decisions.jsonl"
    full_path = out_dir / "full.json"

    summary_path.write_text(
        json.dumps(bundle["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_build_path.write_text(
        json.dumps(bundle["final_build"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(floors_path, bundle["floors"])
    _write_jsonl(decisions_path, bundle["decisions"])
    full_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "summary": summary_path,
        "final_build": final_build_path,
        "floors": floors_path,
        "decisions": decisions_path,
        "full": full_path,
    }


def build_offline_training_samples(bundle: dict[str, Any], player_id: int | None = None) -> dict[str, list[dict[str, Any]]]:
    summary, floor_views = _build_player_floor_views(bundle, player_id=player_id)
    if not floor_views:
        return {"route_samples": [], "card_choice_samples": [], "build_samples": []}

    route_samples = _build_route_samples(summary, floor_views)
    card_choice_samples = _build_card_choice_samples(summary, floor_views)
    build_samples = _build_build_samples(summary, floor_views)
    return {
        "route_samples": route_samples,
        "card_choice_samples": card_choice_samples,
        "build_samples": build_samples,
    }


def build_offline_build_v2_samples(
    bundle: dict[str, Any],
    player_id: int | None = None,
) -> dict[str, Any]:
    summary, floor_views = _build_player_floor_views(bundle, player_id=player_id)
    task_rows = _empty_build_v2_task_rows()
    if not floor_views:
        return {
            "task_rows": task_rows,
            "audit": {},
        }

    task_rows["regular_card_reward"] = _build_regular_card_reward_v2_samples(summary, floor_views)
    task_rows["event_card_bundle"] = _build_event_card_bundle_v2_samples(summary, floor_views)
    task_rows["ancient_choice"] = _build_ancient_choice_v2_samples(summary, floor_views)
    task_rows["relic_choice_step"] = _build_relic_choice_step_v2_samples(summary, floor_views)
    task_rows["potion_choice_step"] = _build_potion_choice_step_v2_samples(summary, floor_views)
    task_rows["rest_action"] = _build_rest_action_v2_samples(summary, floor_views)
    task_rows["smith_target"] = _build_smith_target_v2_samples(summary, floor_views)
    task_rows["remove_card_step"] = _build_remove_card_step_v2_samples(summary, floor_views)
    task_rows["transform_card_step"] = _build_transform_card_step_v2_samples(summary, floor_views)
    task_rows["shop_relic_pick_step"] = _build_shop_relic_pick_step_v2_samples(summary, floor_views)
    task_rows["shop_potion_pick_step"] = _build_shop_potion_pick_step_v2_samples(summary, floor_views)
    task_rows["shop_remove_binary"] = _build_shop_remove_binary_v2_samples(summary, floor_views)
    task_rows["shop_remove_target_step"] = _build_shop_remove_target_step_v2_samples(summary, floor_views)
    task_rows["shop_bundle_aux"] = _build_shop_bundle_aux_v2_samples(summary, floor_views)
    return {
        "task_rows": task_rows,
        "audit": _build_build_v2_audit(summary, floor_views, task_rows),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _empty_build_v2_task_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "regular_card_reward": [],
        "event_card_bundle": [],
        "ancient_choice": [],
        "relic_choice_step": [],
        "potion_choice_step": [],
        "rest_action": [],
        "smith_target": [],
        "remove_card_step": [],
        "transform_card_step": [],
        "shop_relic_pick_step": [],
        "shop_potion_pick_step": [],
        "shop_remove_binary": [],
        "shop_remove_target_step": [],
        "shop_bundle_aux": [],
    }


def _build_player_floor_views(
    bundle: dict[str, Any],
    player_id: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    floors = bundle.get("floors") or []
    summary = bundle.get("summary") or {}
    if not floors or not summary:
        return summary, []

    final_build = bundle.get("final_build") or {}
    player_records = final_build.get("players") or []

    if player_id is None:
        player_id = floors[0].get("player_id")

    player_floors = [floor for floor in floors if floor.get("player_id") == player_id]
    if not player_floors:
        return summary, []

    player_build = next(
        (player for player in player_records if player.get("player_id") == player_id),
        player_records[0] if player_records else {},
    )

    deck_state = _infer_initial_deck_state(player_build, player_floors)
    relic_state = _infer_initial_relic_state(player_build, player_floors)

    floor_views: list[dict[str, Any]] = []
    for floor in player_floors:
        deck_before = _summarize_deck_instances(deck_state)
        derived = _derive_floor_numerics(floor)
        quality = _derive_floor_quality_flags(floor)
        relics_before = None if quality["relic_ids_before_approximate"] else list(relic_state)

        resolved_upgraded_cards, upgrade_label_reliable = _apply_floor_changes(deck_state, relic_state, floor)

        deck_after = _summarize_deck_instances(deck_state)
        relics_after = list(relic_state)
        floor_views.append(
            {
                **floor,
                "raw_upgraded_cards": list(floor.get("upgraded_cards") or []),
                "upgraded_cards": resolved_upgraded_cards,
                "upgrade_label_reliable": upgrade_label_reliable,
                "floor_number": floor["path_index"] + 1,
                "hp_before": derived["hp_before"],
                "hp_ratio_before": derived["hp_ratio_before"],
                "gold_before": derived["gold_before"],
                "quality_flags": quality,
                "deck_before": deck_before,
                "deck_after": deck_after,
                "relic_ids_before": relics_before,
                "relic_ids_after": relics_after,
            }
        )
    return summary, floor_views


def _build_v2_common_row(summary: dict[str, Any], floor: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "split": summary["split"],
        "build_id": summary.get("build_id"),
        "character": floor.get("character"),
        "ascension": summary.get("ascension"),
        "floor_number": floor.get("floor_number"),
        "act_index": floor.get("act_index"),
        "path_index": floor.get("path_index"),
        "map_point_type": floor.get("map_point_type"),
        "room_type": floor.get("room_type"),
        "room_model_id": floor.get("room_model_id"),
        "monster_ids": list(floor.get("monster_ids") or []),
        "turns_taken": floor.get("turns_taken"),
        "hp_before": floor.get("hp_before"),
        "current_hp": floor.get("current_hp"),
        "max_hp": floor.get("max_hp"),
        "gold_before": floor.get("gold_before"),
        "current_gold": floor.get("current_gold"),
        "gold_spent": floor.get("gold_spent"),
        "gold_gained": floor.get("gold_gained"),
        "deck_before": floor.get("deck_before"),
        "deck_after": floor.get("deck_after"),
        "relic_ids_before": floor.get("relic_ids_before"),
        "relic_ids_after": floor.get("relic_ids_after"),
        "quality_flags": floor.get("quality_flags"),
    }


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _remaining_candidate_counts(
    option_ids: list[str],
    selected_prefix_ids: list[str],
) -> Counter[str]:
    counts = Counter(str(value) for value in option_ids if value)
    for selected_id in selected_prefix_ids:
        key = str(selected_id)
        if counts.get(key, 0) > 0:
            counts[key] -= 1
    return counts


def _expand_deck_card_id_multiset(deck_summary: dict[str, Any] | None) -> list[str]:
    cards = (deck_summary or {}).get("cards") or []
    expanded: list[str] = []
    for card in cards:
        card_id = str(card.get("id") or "")
        count = int(card.get("count") or 0)
        if not card_id or count <= 0:
            continue
        expanded.extend([card_id] * count)
    return expanded


def _build_candidate_step_rows(
    *,
    summary: dict[str, Any],
    floor: dict[str, Any],
    task: str,
    choice_group_id: str,
    option_ids: list[str],
    selected_ids: list[str],
    option_kind: str,
    supervision_type: str,
    allow_skip: bool,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    option_ids = [str(value) for value in option_ids if value]
    selected_ids = [str(value) for value in selected_ids if value]
    option_order = _ordered_unique(option_ids)
    rows: list[dict[str, Any]] = []
    common = _build_v2_common_row(summary, floor)
    extra = extra_fields or {}

    if not option_order and not selected_ids:
        return []

    if not selected_ids:
        if not allow_skip:
            return []
        rows.append(
            {
                **common,
                **extra,
                "sample_id": f"{choice_group_id}:step:1",
                "choice_group_id": choice_group_id,
                "task": task,
                "decision_type": task,
                "option_kind": option_kind,
                "supervision_type": supervision_type,
                "option_ids": option_order,
                "option_counts": {key: int(value) for key, value in _remaining_candidate_counts(option_ids, []).items() if value > 0},
                "label_id": "<skip>",
                "skip_available": True,
                "selection_step_index": 0,
                "selection_steps_total": 1,
                "selected_prefix_ids": [],
                "selected_ids_full": [],
            }
        )
        return rows

    selected_prefix_ids: list[str] = []
    total_steps = len(selected_ids)
    for step_index, label_id in enumerate(selected_ids):
        remaining_counts = _remaining_candidate_counts(option_ids, selected_prefix_ids)
        candidate_ids = [candidate_id for candidate_id in option_order if remaining_counts.get(candidate_id, 0) > 0]
        if label_id not in candidate_ids:
            return []

        rows.append(
            {
                **common,
                **extra,
                "sample_id": f"{choice_group_id}:step:{step_index + 1}",
                "choice_group_id": choice_group_id,
                "task": task,
                "decision_type": task,
                "option_kind": option_kind,
                "supervision_type": supervision_type,
                "option_ids": candidate_ids,
                "option_counts": {key: int(remaining_counts[key]) for key in candidate_ids},
                "label_id": label_id,
                "skip_available": False,
                "selection_step_index": step_index,
                "selection_steps_total": total_steps,
                "selected_prefix_ids": list(selected_prefix_ids),
                "selected_ids_full": list(selected_ids),
            }
        )
        selected_prefix_ids.append(label_id)

    return rows


def _build_regular_card_reward_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        if floor.get("room_type") not in {"monster", "elite", "boss"}:
            continue
        options = floor.get("card_choices") or []
        if not options:
            continue
        option_ids = [item["card"]["id"] for item in options if item.get("card") and item["card"].get("id")]
        picked_ids = [item["card"]["id"] for item in options if item.get("was_picked") and item.get("card") and item["card"].get("id")]
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="regular_card_reward",
                choice_group_id=f"{summary['run_id']}:regular_card_reward:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=picked_ids,
                option_kind="card",
                supervision_type="single_pick_or_skip",
                allow_skip=True,
                extra_fields={
                    "options": options,
                    "picked_card_ids": picked_ids,
                },
            )
        )
    return rows


def _build_event_card_bundle_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        if floor.get("room_type") != "event":
            continue
        options = floor.get("card_choices") or []
        if not options:
            continue
        option_ids = [item["card"]["id"] for item in options if item.get("card") and item["card"].get("id")]
        picked_ids = [item["card"]["id"] for item in options if item.get("was_picked") and item.get("card") and item["card"].get("id")]
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="event_card_bundle",
                choice_group_id=f"{summary['run_id']}:event_card_bundle:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=picked_ids,
                option_kind="card",
                supervision_type="autoregressive_multiselect",
                allow_skip=True,
                extra_fields={
                    "options": options,
                    "picked_card_ids": picked_ids,
                },
            )
        )
    return rows


def _build_ancient_choice_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        choices = floor.get("ancient_choices") or []
        if not choices:
            continue
        option_ids = [item["text_key"] for item in choices if item.get("text_key")]
        selected_ids = [item["text_key"] for item in choices if item.get("was_chosen") and item.get("text_key")]
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="ancient_choice",
                choice_group_id=f"{summary['run_id']}:ancient_choice:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=selected_ids,
                option_kind="ancient",
                supervision_type="single_pick",
                allow_skip=False,
                extra_fields={"choices": choices},
            )
        )
    return rows


def _build_relic_choice_step_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        if floor.get("room_type") == "shop" or floor.get("ancient_choices"):
            continue
        choices = floor.get("relic_choices") or []
        if not choices:
            continue
        option_ids = [item["choice"] for item in choices if item.get("choice")]
        selected_ids = [item["choice"] for item in choices if item.get("was_picked") and item.get("choice")]
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="relic_choice_step",
                choice_group_id=f"{summary['run_id']}:relic_choice_step:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=selected_ids,
                option_kind="relic",
                supervision_type="autoregressive_multiselect",
                allow_skip=True,
                extra_fields={"choices": choices},
            )
        )
    return rows


def _build_potion_choice_step_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        if floor.get("room_type") == "shop":
            continue
        choices = floor.get("potion_choices") or []
        if not choices:
            continue
        option_ids = [item["choice"] for item in choices if item.get("choice")]
        selected_ids = [item["choice"] for item in choices if item.get("was_picked") and item.get("choice")]
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="potion_choice_step",
                choice_group_id=f"{summary['run_id']}:potion_choice_step:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=selected_ids,
                option_kind="potion",
                supervision_type="autoregressive_multiselect",
                allow_skip=True,
                extra_fields={"choices": choices},
            )
        )
    return rows


def _build_rest_action_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        selected_actions = [str(value) for value in floor.get("rest_site_choices") or [] if value]
        if len(selected_actions) != 1:
            continue
        row = {
            **_build_v2_common_row(summary, floor),
            "sample_id": f"{summary['run_id']}:rest_action:{floor['floor_number']}",
            "choice_group_id": f"{summary['run_id']}:rest_action:{floor['floor_number']}",
            "task": "rest_action",
            "decision_type": "rest_action",
            "option_kind": "rest_action",
            "supervision_type": "classification",
            "label_id": selected_actions[0],
            "skip_available": False,
            "selected_prefix_ids": [],
            "selected_ids_full": list(selected_actions),
            "rest_site_choices": list(floor.get("rest_site_choices") or []),
        }
        rows.append(row)
    return rows


def _build_smith_target_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        selected_ids = [str(value) for value in floor.get("upgraded_cards") or [] if value]
        if len(selected_ids) != 1 or not floor.get("upgrade_label_reliable"):
            continue
        option_ids: list[str] = []
        for card in (floor.get("deck_before") or {}).get("cards") or []:
            card_id = str(card.get("id") or "")
            count = int(card.get("count") or 0)
            upgraded_count = int(card.get("upgraded_count") or 0)
            if not card_id or count <= 0:
                continue
            if count > upgraded_count:
                option_ids.append(card_id)
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="smith_target",
                choice_group_id=f"{summary['run_id']}:smith_target:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=selected_ids,
                option_kind="card",
                supervision_type="single_pick",
                allow_skip=False,
                extra_fields={"raw_upgraded_cards": list(floor.get("raw_upgraded_cards") or [])},
            )
        )
    return rows


def _build_remove_card_step_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        selected_ids = [item["id"] for item in floor.get("cards_removed") or [] if item and item.get("id")]
        if not selected_ids:
            continue
        option_ids = _expand_deck_card_id_multiset(floor.get("deck_before"))
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="remove_card_step",
                choice_group_id=f"{summary['run_id']}:remove_card_step:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=selected_ids,
                option_kind="card",
                supervision_type="autoregressive_multiselect",
                allow_skip=False,
                extra_fields={"removed_cards": floor.get("cards_removed") or []},
            )
        )
    return rows


def _build_transform_card_step_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        transforms = floor.get("cards_transformed") or []
        selected_ids = [
            item["original_card"]["id"]
            for item in transforms
            if item.get("original_card") and item["original_card"].get("id")
        ]
        if not selected_ids:
            continue
        option_ids = _expand_deck_card_id_multiset(floor.get("deck_before"))
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="transform_card_step",
                choice_group_id=f"{summary['run_id']}:transform_card_step:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=selected_ids,
                option_kind="card",
                supervision_type="autoregressive_multiselect",
                allow_skip=False,
                extra_fields={"transforms": transforms},
            )
        )
    return rows


def _build_shop_relic_pick_step_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        if floor.get("room_type") != "shop":
            continue
        choices = floor.get("relic_choices") or []
        if not choices:
            continue
        option_ids = [item["choice"] for item in choices if item.get("choice")]
        selected_ids = [item["choice"] for item in choices if item.get("was_picked") and item.get("choice")]
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="shop_relic_pick_step",
                choice_group_id=f"{summary['run_id']}:shop_relic_pick_step:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=selected_ids,
                option_kind="relic",
                supervision_type="autoregressive_multiselect",
                allow_skip=True,
                extra_fields={"choices": choices},
            )
        )
    return rows


def _build_shop_potion_pick_step_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        if floor.get("room_type") != "shop":
            continue
        choices = floor.get("potion_choices") or []
        if not choices:
            continue
        option_ids = [item["choice"] for item in choices if item.get("choice")]
        selected_ids = [item["choice"] for item in choices if item.get("was_picked") and item.get("choice")]
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="shop_potion_pick_step",
                choice_group_id=f"{summary['run_id']}:shop_potion_pick_step:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=selected_ids,
                option_kind="potion",
                supervision_type="autoregressive_multiselect",
                allow_skip=True,
                extra_fields={"choices": choices},
            )
        )
    return rows


def _build_shop_remove_binary_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        if floor.get("room_type") != "shop":
            continue
        removed_ids = [item["id"] for item in floor.get("cards_removed") or [] if item and item.get("id")]
        row = {
            **_build_v2_common_row(summary, floor),
            "sample_id": f"{summary['run_id']}:shop_remove_binary:{floor['floor_number']}",
            "choice_group_id": f"{summary['run_id']}:shop_remove_binary:{floor['floor_number']}",
            "task": "shop_remove_binary",
            "decision_type": "shop_remove_binary",
            "option_kind": "shop_remove",
            "supervision_type": "binary_choice",
            "label_id": "remove_card" if removed_ids else "skip_remove",
            "skip_available": False,
            "selected_prefix_ids": [],
            "selected_ids_full": removed_ids,
            "shown_card_options": floor.get("card_choices") or [],
            "shown_relic_options": floor.get("relic_choices") or [],
            "shown_potion_options": floor.get("potion_choices") or [],
        }
        rows.append(row)
    return rows


def _build_shop_remove_target_step_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        if floor.get("room_type") != "shop":
            continue
        selected_ids = [item["id"] for item in floor.get("cards_removed") or [] if item and item.get("id")]
        if not selected_ids:
            continue
        option_ids = _expand_deck_card_id_multiset(floor.get("deck_before"))
        rows.extend(
            _build_candidate_step_rows(
                summary=summary,
                floor=floor,
                task="shop_remove_target_step",
                choice_group_id=f"{summary['run_id']}:shop_remove_target_step:{floor['floor_number']}",
                option_ids=option_ids,
                selected_ids=selected_ids,
                option_kind="card",
                supervision_type="autoregressive_multiselect",
                allow_skip=False,
                extra_fields={"removed_cards": floor.get("cards_removed") or []},
            )
        )
    return rows


def _build_shop_bundle_aux_v2_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for floor in floor_views:
        if floor.get("room_type") != "shop":
            continue
        shown_cards = floor.get("card_choices") or []
        shown_relics = floor.get("relic_choices") or []
        shown_potions = floor.get("potion_choices") or []
        bought_card_ids = [item["id"] for item in floor.get("cards_gained") or [] if item and item.get("id")]
        picked_relic_ids = [item["choice"] for item in shown_relics if item.get("was_picked") and item.get("choice")]
        picked_potion_ids = [item["choice"] for item in shown_potions if item.get("was_picked") and item.get("choice")]
        removed_card_ids = [item["id"] for item in floor.get("cards_removed") or [] if item and item.get("id")]
        available_card_ids = [item["card"]["id"] for item in shown_cards if item.get("card") and item["card"].get("id")]

        did_buy_card = bool(bought_card_ids)
        did_buy_relic = bool(picked_relic_ids)
        did_buy_potion = bool(picked_potion_ids)
        did_remove = bool(removed_card_ids)

        rows.append(
            {
                **_build_v2_common_row(summary, floor),
                "sample_id": f"{summary['run_id']}:shop_bundle_aux:{floor['floor_number']}",
                "choice_group_id": f"{summary['run_id']}:shop_bundle_aux:{floor['floor_number']}",
                "task": "shop_bundle_aux",
                "decision_type": "shop_bundle_aux",
                "supervision_type": "multi_binary_aux",
                "available_card_ids": available_card_ids,
                "available_relic_ids": [item["choice"] for item in shown_relics if item.get("choice")],
                "available_potion_ids": [item["choice"] for item in shown_potions if item.get("choice")],
                "bought_card_ids": bought_card_ids,
                "picked_relic_ids": picked_relic_ids,
                "picked_potion_ids": picked_potion_ids,
                "removed_card_ids": removed_card_ids,
                "did_buy_any_card": did_buy_card,
                "did_buy_any_relic": did_buy_relic,
                "did_buy_any_potion": did_buy_potion,
                "did_remove_card": did_remove,
                "leave_only": not (did_buy_card or did_buy_relic or did_buy_potion or did_remove),
                "shop_card_purchase_unsupported": bool(bought_card_ids) and not set(bought_card_ids).issubset(set(available_card_ids)),
            }
        )
    return rows


def _build_build_v2_audit(
    summary: dict[str, Any],
    floor_views: list[dict[str, Any]],
    task_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "run_id": summary.get("run_id"),
        "character": (summary.get("characters") or [None])[0],
        "regular_card_reward_rows": len(task_rows.get("regular_card_reward") or []),
        "event_card_bundle_rows": len(task_rows.get("event_card_bundle") or []),
        "rest_action_rows": len(task_rows.get("rest_action") or []),
        "smith_target_rows": len(task_rows.get("smith_target") or []),
        "remove_card_step_rows": len(task_rows.get("remove_card_step") or []),
        "transform_card_step_rows": len(task_rows.get("transform_card_step") or []),
        "shop_remove_binary_rows": len(task_rows.get("shop_remove_binary") or []),
        "shop_remove_target_step_rows": len(task_rows.get("shop_remove_target_step") or []),
        "shop_bundle_aux_rows": len(task_rows.get("shop_bundle_aux") or []),
    }

    regular_choice_groups = 0
    regular_invalid_multi_pick_groups = 0
    event_multi_pick_groups = 0
    rest_multi_action_anomalies = 0
    unsupported_shop_card_purchase_rows = 0
    duplicate_remove_groups = 0
    duplicate_transform_groups = 0
    shop_rows = 0

    for floor in floor_views:
        room_type = str(floor.get("room_type") or "")
        card_choices = floor.get("card_choices") or []
        if room_type in {"monster", "elite", "boss"} and card_choices:
            regular_choice_groups += 1
            regular_pick_count = sum(1 for item in card_choices if item.get("was_picked") and item.get("card"))
            if regular_pick_count > 1:
                regular_invalid_multi_pick_groups += 1
        if room_type == "event" and card_choices:
            event_pick_count = sum(1 for item in card_choices if item.get("was_picked") and item.get("card"))
            if event_pick_count > 1:
                event_multi_pick_groups += 1

        if len([value for value in floor.get("rest_site_choices") or [] if value]) > 1:
            rest_multi_action_anomalies += 1

        if room_type == "shop":
            shop_rows += 1
            shown_card_ids = {
                item["card"]["id"]
                for item in card_choices
                if item.get("card") and item["card"].get("id")
            }
            bought_card_ids = {
                item["id"]
                for item in floor.get("cards_gained") or []
                if item and item.get("id")
            }
            if bought_card_ids and not bought_card_ids.issubset(shown_card_ids):
                unsupported_shop_card_purchase_rows += 1

        removed_ids = [item["id"] for item in floor.get("cards_removed") or [] if item and item.get("id")]
        if len(removed_ids) != len(set(removed_ids)):
            duplicate_remove_groups += 1

        transformed_ids = [
            item["original_card"]["id"]
            for item in floor.get("cards_transformed") or []
            if item.get("original_card") and item["original_card"].get("id")
        ]
        if len(transformed_ids) != len(set(transformed_ids)):
            duplicate_transform_groups += 1

    audit.update(
        {
            "regular_card_reward_groups": regular_choice_groups,
            "regular_invalid_multi_pick_groups": regular_invalid_multi_pick_groups,
            "event_multi_pick_groups": event_multi_pick_groups,
            "rest_multi_action_anomalies": rest_multi_action_anomalies,
            "shop_rows": shop_rows,
            "unsupported_shop_card_purchase_rows": unsupported_shop_card_purchase_rows,
            "duplicate_remove_groups": duplicate_remove_groups,
            "duplicate_transform_groups": duplicate_transform_groups,
        }
    )
    return audit


def _extract_summary(run: dict[str, Any], source_path: Path) -> dict[str, Any]:
    players = run.get("players") or []
    path_segments = run.get("map_point_history") or []
    path_point_count = sum(len(segment) for segment in path_segments if isinstance(segment, list))
    run_id = str(run.get("start_time") or source_path.stem)

    return {
        "run_id": run_id,
        "split": _stable_dataset_split(run_id),
        "source_file": str(source_path),
        "source_name": source_path.name,
        "schema_version": run.get("schema_version"),
        "build_id": run.get("build_id"),
        "game_mode": run.get("game_mode"),
        "platform_type": run.get("platform_type"),
        "ascension": run.get("ascension"),
        "seed": run.get("seed"),
        "start_time": run.get("start_time"),
        "run_time_seconds": run.get("run_time"),
        "win": run.get("win"),
        "was_abandoned": run.get("was_abandoned"),
        "killed_by_encounter": run.get("killed_by_encounter"),
        "killed_by_event": run.get("killed_by_event"),
        "acts": list(run.get("acts") or []),
        "modifier_ids": list(run.get("modifiers") or []),
        "player_count": len(players),
        "characters": [player.get("character") for player in players],
        "final_deck_sizes": [len(player.get("deck") or []) for player in players],
        "final_relic_counts": [len(player.get("relics") or []) for player in players],
        "final_potion_counts": [len(player.get("potions") or []) for player in players],
        "path_segment_count": len(path_segments),
        "path_point_count": path_point_count,
    }


def _extract_final_build(run: dict[str, Any]) -> dict[str, Any]:
    players_out: list[dict[str, Any]] = []
    for player in run.get("players") or []:
        players_out.append(
            {
                "player_id": player.get("id"),
                "character": player.get("character"),
                "max_potion_slot_count": player.get("max_potion_slot_count"),
                "deck": [card for card in (_normalize_card_ref(card) for card in player.get("deck") or []) if card],
                "relics": [relic for relic in (_normalize_choice_ref(relic) for relic in player.get("relics") or []) if relic],
                "potions": [potion for potion in (_normalize_choice_ref(potion) for potion in player.get("potions") or []) if potion],
            }
        )

    return {"players": players_out}


def _extract_floor_records(run: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    players = run.get("players") or []
    player_by_id = {
        player.get("id"): player
        for player in players
        if player.get("id") is not None
    }

    floors: list[dict[str, Any]] = []
    path_index = 0

    for act_index, segment in enumerate(run.get("map_point_history") or [], start=1):
        if not isinstance(segment, list):
            continue

        for point_index_in_act, point in enumerate(segment):
            rooms = point.get("rooms") or []
            room = rooms[0] if rooms else {}
            player_stats = point.get("player_stats") or [{}]

            for player_stat_index, stat in enumerate(player_stats):
                player_id = stat.get("player_id")
                player = player_by_id.get(player_id, {})
                floor_record = {
                    "run_id": summary["run_id"],
                    "split": summary["split"],
                    "source_file": summary["source_file"],
                    "path_index": path_index,
                    "act_index": act_index,
                    "point_index_in_act": point_index_in_act,
                    "player_stat_index": player_stat_index,
                    "player_id": player_id,
                    "character": player.get("character"),
                    "map_point_type": point.get("map_point_type"),
                    "room_type": room.get("room_type"),
                    "room_model_id": room.get("model_id"),
                    "monster_ids": list(room.get("monster_ids") or []),
                    "turns_taken": room.get("turns_taken"),
                    "current_hp": stat.get("current_hp"),
                    "max_hp": stat.get("max_hp"),
                    "damage_taken": stat.get("damage_taken"),
                    "hp_healed": stat.get("hp_healed"),
                    "current_gold": stat.get("current_gold"),
                    "gold_gained": stat.get("gold_gained"),
                    "gold_spent": stat.get("gold_spent"),
                    "gold_lost": stat.get("gold_lost"),
                    "gold_stolen": stat.get("gold_stolen"),
                    "max_hp_gained": stat.get("max_hp_gained"),
                    "max_hp_lost": stat.get("max_hp_lost"),
                    "cards_gained": [card for card in (_normalize_card_ref(card) for card in stat.get("cards_gained") or []) if card],
                    "cards_removed": [card for card in (_normalize_card_ref(card) for card in stat.get("cards_removed") or []) if card],
                    "cards_transformed": [
                        {
                            "original_card": _normalize_card_ref(item.get("original_card")),
                            "final_card": _normalize_card_ref(item.get("final_card")),
                        }
                        for item in stat.get("cards_transformed") or []
                    ],
                    "upgraded_cards": list(stat.get("upgraded_cards") or []),
                    "rest_site_choices": list(stat.get("rest_site_choices") or []),
                    "card_choices": _normalize_card_choices(stat.get("card_choices") or []),
                    "relic_choices": _normalize_marked_choices(stat.get("relic_choices") or []),
                    "potion_choices": _normalize_marked_choices(stat.get("potion_choices") or []),
                    "event_choices": _normalize_event_choices(stat.get("event_choices") or []),
                    "ancient_choices": _normalize_ancient_choices(stat.get("ancient_choice") or []),
                }
                floors.append(floor_record)

            path_index += 1

    return floors


def _extract_decision_records(floors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []

    for floor in floors:
        is_shop = floor["room_type"] == "shop"
        has_ancient_choice = bool(floor["ancient_choices"])
        context = {
            "run_id": floor["run_id"],
            "split": floor["split"],
            "source_file": floor["source_file"],
            "path_index": floor["path_index"],
            "act_index": floor["act_index"],
            "point_index_in_act": floor["point_index_in_act"],
            "player_id": floor["player_id"],
            "character": floor["character"],
            "map_point_type": floor["map_point_type"],
            "room_type": floor["room_type"],
            "room_model_id": floor["room_model_id"],
            "monster_ids": floor["monster_ids"],
            "turns_taken": floor["turns_taken"],
            "current_hp": floor["current_hp"],
            "max_hp": floor["max_hp"],
            "damage_taken": floor["damage_taken"],
            "hp_healed": floor["hp_healed"],
            "current_gold": floor["current_gold"],
            "gold_gained": floor["gold_gained"],
            "gold_spent": floor["gold_spent"],
        }

        if floor["card_choices"] and not is_shop:
            picked = [item["card"] for item in floor["card_choices"] if item.get("was_picked")]
            decisions.append(
                {
                    **context,
                    "decision_type": "card_choice",
                    "options": floor["card_choices"],
                    "picked_cards": picked,
                    "skipped": not picked,
                }
            )

        if floor["relic_choices"] and not is_shop and not has_ancient_choice:
            picked = [item["choice"] for item in floor["relic_choices"] if item.get("was_picked")]
            decisions.append(
                {
                    **context,
                    "decision_type": "relic_choice",
                    "options": floor["relic_choices"],
                    "picked_choices": picked,
                }
            )

        if floor["potion_choices"] and not is_shop:
            picked = [item["choice"] for item in floor["potion_choices"] if item.get("was_picked")]
            decisions.append(
                {
                    **context,
                    "decision_type": "potion_choice",
                    "options": floor["potion_choices"],
                    "picked_choices": picked,
                }
            )

        if floor["event_choices"]:
            decisions.append(
                {
                    **context,
                    "decision_type": "event_choice",
                    "choices": floor["event_choices"],
                }
            )

        if floor["ancient_choices"]:
            picked = [item["text_key"] for item in floor["ancient_choices"] if item.get("was_chosen")]
            decisions.append(
                {
                    **context,
                    "decision_type": "ancient_choice",
                    "options": floor["ancient_choices"],
                    "picked_choices": picked,
                }
            )

        if floor["rest_site_choices"]:
            decisions.append(
                {
                    **context,
                    "decision_type": "rest_site_choice",
                    "choices": floor["rest_site_choices"],
                }
            )

        if floor["upgraded_cards"]:
            decisions.append(
                {
                    **context,
                    "decision_type": "upgrade",
                    "upgraded_cards": floor["upgraded_cards"],
                }
            )

        if floor["cards_removed"]:
            decisions.append(
                {
                    **context,
                    "decision_type": "card_remove",
                    "removed_cards": floor["cards_removed"],
                }
            )

        if floor["cards_transformed"]:
            decisions.append(
                {
                    **context,
                    "decision_type": "card_transform",
                    "transforms": floor["cards_transformed"],
                }
            )

        if is_shop:
            decisions.append(
                {
                    **context,
                    "decision_type": "shop_summary",
                    "cards_gained": floor["cards_gained"],
                    "cards_removed": floor["cards_removed"],
                    "gold_spent": floor["gold_spent"],
                    "shown_card_options": floor["card_choices"],
                    "shown_relic_options": floor["relic_choices"],
                    "shown_potion_options": floor["potion_choices"],
                }
            )

    return decisions


def _derive_floor_numerics(floor: dict[str, Any]) -> dict[str, Any]:
    current_hp = floor.get("current_hp")
    max_hp = floor.get("max_hp")
    damage_taken = floor.get("damage_taken") or 0
    hp_healed = floor.get("hp_healed") or 0
    gold_now = floor.get("current_gold")
    gold_gained = floor.get("gold_gained") or 0
    gold_spent = floor.get("gold_spent") or 0
    gold_lost = floor.get("gold_lost") or 0
    gold_stolen = floor.get("gold_stolen") or 0

    hp_before = None
    hp_ratio_before = None
    if isinstance(current_hp, int):
        hp_before = current_hp + damage_taken - hp_healed
        if floor.get("path_index") == 0 and hp_before <= 0 < current_hp:
            hp_before = current_hp
        hp_before = max(0, hp_before)
        if isinstance(max_hp, int) and max_hp > 0:
            hp_before = max(0, min(hp_before, max_hp))
        if isinstance(max_hp, int) and max_hp > 0:
            hp_ratio_before = max(0.0, min(hp_before / max_hp, 2.0))

    gold_before = None
    if isinstance(gold_now, int):
        gold_before = gold_now - gold_gained + gold_spent + gold_lost + gold_stolen
        gold_before = max(0, gold_before)

    return {
        "hp_before": hp_before,
        "hp_ratio_before": hp_ratio_before,
        "gold_before": gold_before,
    }


def _derive_floor_quality_flags(floor: dict[str, Any]) -> dict[str, bool]:
    is_initial_floor = floor.get("path_index") == 0
    return {
        "hp_before_estimated": isinstance(floor.get("current_hp"), int),
        "gold_before_estimated": isinstance(floor.get("current_gold"), int),
        "relic_ids_before_approximate": is_initial_floor and bool(floor.get("ancient_choices")),
        "initial_floor_sample": is_initial_floor,
    }


def _infer_initial_deck_state(player_build: dict[str, Any], floors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    initial_cards: list[dict[str, Any]] = []

    for card in player_build.get("deck") or []:
        if not isinstance(card, dict):
            continue
        if card.get("floor_added_to_deck", 1) == 1:
            initial_cards.append(
                _card_instance_from_ref(
                    card,
                    default_floor=1,
                    preserve_upgrade_level=False,
                )
            )

    for floor in floors:
        for card in floor.get("cards_removed") or []:
            if isinstance(card, dict) and card.get("floor_added_to_deck", 0) == 1:
                initial_cards.append(
                    _card_instance_from_ref(
                        card,
                        default_floor=1,
                        preserve_upgrade_level=False,
                    )
                )
        for item in floor.get("cards_transformed") or []:
            original = item.get("original_card")
            if isinstance(original, dict) and original.get("floor_added_to_deck", 0) == 1:
                initial_cards.append(
                    _card_instance_from_ref(
                        original,
                        default_floor=1,
                        preserve_upgrade_level=False,
                    )
                )

    return initial_cards


def _infer_initial_relic_state(player_build: dict[str, Any], floors: list[dict[str, Any]]) -> list[str]:
    relic_ids: list[str] = []

    for relic in player_build.get("relics") or []:
        if not isinstance(relic, dict):
            continue
        floor_added = relic.get("floor_added_to_deck", 1)
        relic_id = relic.get("id")
        if floor_added == 1 and relic_id and relic_id not in relic_ids:
            relic_ids.append(relic_id)

    if not relic_ids:
        first_floor = floors[0] if floors else {}
        for choice in first_floor.get("relic_choices") or []:
            relic_id = choice.get("choice")
            if choice.get("was_picked") and relic_id and relic_id not in relic_ids:
                relic_ids.append(relic_id)

    return relic_ids


def _card_instance_from_ref(
    card: dict[str, Any],
    default_floor: int,
    *,
    preserve_upgrade_level: bool = True,
) -> dict[str, Any]:
    return {
        "id": card.get("id"),
        "floor_added_to_deck": card.get("floor_added_to_deck", default_floor),
        "current_upgrade_level": (card.get("current_upgrade_level", 0) or 0) if preserve_upgrade_level else 0,
    }


def _apply_floor_changes(
    deck_state: list[dict[str, Any]],
    relic_state: list[str],
    floor: dict[str, Any],
) -> tuple[list[str], bool]:
    current_floor = floor["path_index"] + 1

    for choice in floor.get("relic_choices") or []:
        relic_id = choice.get("choice")
        if choice.get("was_picked") and relic_id and relic_id not in relic_state:
            relic_state.append(relic_id)

    for card in floor.get("cards_removed") or []:
        _pop_matching_card(deck_state, card)

    for item in floor.get("cards_transformed") or []:
        original = item.get("original_card")
        final_card = item.get("final_card")
        if isinstance(original, dict):
            _pop_matching_card(deck_state, original)
        if isinstance(final_card, dict):
            deck_state.append(_card_instance_from_ref(final_card, default_floor=current_floor))

    for card in floor.get("cards_gained") or []:
        if isinstance(card, dict):
            deck_state.append(_card_instance_from_ref(card, default_floor=current_floor))

    upgraded_cards = [str(value) for value in floor.get("upgraded_cards") or [] if value]
    for upgraded_id in upgraded_cards:
        _upgrade_matching_card(deck_state, upgraded_id)
    return upgraded_cards, _is_upgrade_label_reliable(floor, upgraded_cards)


def _is_upgrade_label_reliable(floor: dict[str, Any], upgraded_cards: list[str]) -> bool:
    if not upgraded_cards:
        return False
    room_type = str(floor.get("room_type") or "")
    rest_choices = {str(choice) for choice in floor.get("rest_site_choices") or [] if choice}

    # Campfire smith is the cleanest native "select card to upgrade" surface for
    # supervised labels. Keep deck-state reconstruction for multi-upgrade edge
    # cases, but only admit single-card smith samples into the upgrade task.
    if room_type == "rest_site" and "SMITH" in rest_choices:
        return len(upgraded_cards) == 1

    # Automatic upgrade outcomes from relics/events/purchases still belong in
    # reconstructed deck state, but they are not reliable action labels.
    return False


def _pop_matching_card(deck_state: list[dict[str, Any]], target: dict[str, Any]) -> None:
    target_id = target.get("id")
    target_floor = target.get("floor_added_to_deck")

    for index, card in enumerate(deck_state):
        if card.get("id") != target_id:
            continue
        if target_floor is not None and card.get("floor_added_to_deck") != target_floor:
            continue
        deck_state.pop(index)
        return

    for index, card in enumerate(deck_state):
        if card.get("id") == target_id:
            deck_state.pop(index)
            return


def _upgrade_matching_card(deck_state: list[dict[str, Any]], card_id: str) -> None:
    candidates = [card for card in deck_state if card.get("id") == card_id]
    if not candidates:
        return

    target = min(
        candidates,
        key=lambda card: (
            card.get("current_upgrade_level", 0),
            card.get("floor_added_to_deck", 10**9),
        ),
    )
    target["current_upgrade_level"] = int(target.get("current_upgrade_level", 0) or 0) + 1


def _summarize_deck_instances(deck_state: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    upgraded_counts: Counter[str] = Counter()
    max_upgrade_levels: dict[str, int] = {}

    for card in deck_state:
        card_id = str(card.get("id"))
        counts[card_id] += 1
        level = int(card.get("current_upgrade_level", 0) or 0)
        if level > 0:
            upgraded_counts[card_id] += 1
        max_upgrade_levels[card_id] = max(max_upgrade_levels.get(card_id, 0), level)

    cards = [
        {
            "id": card_id,
            "count": counts[card_id],
            "upgraded_count": upgraded_counts.get(card_id, 0),
            "max_upgrade_level": max_upgrade_levels.get(card_id, 0),
        }
        for card_id in sorted(counts)
    ]
    return {
        "deck_size": len(deck_state),
        "distinct_cards": len(cards),
        "upgraded_card_count": sum(1 for card in deck_state if int(card.get("current_upgrade_level", 0) or 0) > 0),
        "cards": cards,
    }


def _build_route_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if len(floor_views) < 2:
        return samples

    for index, floor in enumerate(floor_views[:-1]):
        next_floor = floor_views[index + 1]
        if floor["act_index"] != next_floor["act_index"]:
            continue

        samples.append(
            {
                "sample_id": f"{summary['run_id']}:route:{floor['floor_number']}",
                "run_id": summary["run_id"],
                "split": summary["split"],
                "build_id": summary["build_id"],
                "character": floor["character"],
                "ascension": summary["ascension"],
                "supervision_type": "chosen_path_only",
                "floor_number": floor["floor_number"],
                "act_index": floor["act_index"],
                "path_index": floor["path_index"],
                "current_map_point_type": floor["map_point_type"],
                "current_room_type": floor["room_type"],
                "current_room_model_id": floor["room_model_id"],
                "current_hp": floor["current_hp"],
                "max_hp": floor["max_hp"],
                "current_gold": floor["current_gold"],
                "deck_after": floor["deck_after"],
                "relic_ids_after": floor["relic_ids_after"],
                "damage_taken": floor["damage_taken"],
                "hp_healed": floor["hp_healed"],
                "turns_taken": floor["turns_taken"],
                "quality_flags": floor["quality_flags"],
                "next_map_point_type": next_floor["map_point_type"],
                "next_room_type": next_floor["room_type"],
                "next_room_model_id": next_floor["room_model_id"],
                "next_monster_ids": next_floor["monster_ids"],
            }
        )

    return samples


def _build_card_choice_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for floor in floor_views:
        if floor["room_type"] == "shop":
            continue
        if not floor["card_choices"]:
            continue

        picked = [item["card"] for item in floor["card_choices"] if item.get("was_picked")]
        option_card_ids = [item["card"]["id"] for item in floor["card_choices"] if item.get("card")]
        picked_card_ids = [item["id"] for item in picked if item]
        samples.append(
            {
                "sample_id": f"{summary['run_id']}:card_choice:{floor['floor_number']}",
                "run_id": summary["run_id"],
                "split": summary["split"],
                "build_id": summary["build_id"],
                "character": floor["character"],
                "ascension": summary["ascension"],
                "supervision_type": "full_choice_set",
                "floor_number": floor["floor_number"],
                "act_index": floor["act_index"],
                "path_index": floor["path_index"],
                "map_point_type": floor["map_point_type"],
                "room_type": floor["room_type"],
                "room_model_id": floor["room_model_id"],
                "monster_ids": floor["monster_ids"],
                "turns_taken": floor["turns_taken"],
                "hp_before": floor["hp_before"],
                "current_hp": floor["current_hp"],
                "max_hp": floor["max_hp"],
                "gold_before": floor["gold_before"],
                "current_gold": floor["current_gold"],
                "deck_before": floor["deck_before"],
                "deck_after": floor["deck_after"],
                "relic_ids_before": floor["relic_ids_before"],
                "relic_ids_after": floor["relic_ids_after"],
                "quality_flags": floor["quality_flags"],
                "option_card_ids": option_card_ids,
                "picked_card_ids": picked_card_ids,
                "options": floor["card_choices"],
                "picked_cards": picked,
                "skipped": not picked,
                "cards_gained": floor["cards_gained"],
            }
        )

    return samples


def _build_build_samples(summary: dict[str, Any], floor_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for floor in floor_views:
        common = {
            "run_id": summary["run_id"],
            "split": summary["split"],
            "build_id": summary["build_id"],
            "character": floor["character"],
            "ascension": summary["ascension"],
            "floor_number": floor["floor_number"],
            "act_index": floor["act_index"],
            "path_index": floor["path_index"],
            "map_point_type": floor["map_point_type"],
            "room_type": floor["room_type"],
            "room_model_id": floor["room_model_id"],
            "hp_before": floor["hp_before"],
            "current_hp": floor["current_hp"],
            "max_hp": floor["max_hp"],
            "gold_before": floor["gold_before"],
            "current_gold": floor["current_gold"],
            "gold_spent": floor["gold_spent"],
            "gold_gained": floor["gold_gained"],
            "deck_before": floor["deck_before"],
            "deck_after": floor["deck_after"],
            "relic_ids_before": floor["relic_ids_before"],
            "relic_ids_after": floor["relic_ids_after"],
            "quality_flags": floor["quality_flags"],
        }

        if floor["room_type"] == "shop":
            samples.append(
                {
                    **common,
                    "sample_id": f"{summary['run_id']}:shop:{floor['floor_number']}",
                    "decision_type": "shop",
                    "supervision_type": "shop_inventory_observed",
                    "available_card_ids": [item["card"]["id"] for item in floor["card_choices"] if item.get("card")],
                    "available_relic_ids": [item["choice"] for item in floor["relic_choices"] if item.get("choice")],
                    "available_potion_ids": [item["choice"] for item in floor["potion_choices"] if item.get("choice")],
                    "bought_card_ids": [item["id"] for item in floor["cards_gained"] if item],
                    "removed_card_ids": [item["id"] for item in floor["cards_removed"] if item],
                    "shown_card_options": floor["card_choices"],
                    "shown_relic_options": floor["relic_choices"],
                    "shown_potion_options": floor["potion_choices"],
                    "cards_gained": floor["cards_gained"],
                    "cards_removed": floor["cards_removed"],
                }
            )

        if floor["rest_site_choices"]:
            samples.append(
                {
                    **common,
                    "sample_id": f"{summary['run_id']}:rest_site:{floor['floor_number']}",
                    "decision_type": "rest_site",
                    "supervision_type": "chosen_action_only",
                    "selected_rest_actions": list(floor["rest_site_choices"]),
                    "choices": floor["rest_site_choices"],
                }
            )

        if floor["upgraded_cards"] and floor.get("upgrade_label_reliable"):
            samples.append(
                {
                    **common,
                    "sample_id": f"{summary['run_id']}:upgrade:{floor['floor_number']}",
                    "decision_type": "upgrade",
                    "supervision_type": "chosen_action_only",
                    "selected_card_ids": list(floor["upgraded_cards"]),
                    "upgraded_cards": floor["upgraded_cards"],
                }
            )

        if floor["cards_removed"]:
            samples.append(
                {
                    **common,
                    "sample_id": f"{summary['run_id']}:card_remove:{floor['floor_number']}",
                    "decision_type": "card_remove",
                    "supervision_type": "chosen_action_only",
                    "selected_card_ids": [item["id"] for item in floor["cards_removed"] if item],
                    "removed_cards": floor["cards_removed"],
                }
            )

        if floor["cards_transformed"]:
            samples.append(
                {
                    **common,
                    "sample_id": f"{summary['run_id']}:card_transform:{floor['floor_number']}",
                    "decision_type": "card_transform",
                    "supervision_type": "chosen_action_only",
                    "selected_card_ids": [
                        item["original_card"]["id"]
                        for item in floor["cards_transformed"]
                        if item.get("original_card")
                    ],
                    "result_card_ids": [
                        item["final_card"]["id"]
                        for item in floor["cards_transformed"]
                        if item.get("final_card")
                    ],
                    "transforms": floor["cards_transformed"],
                }
            )

        if floor["event_choices"]:
            samples.append(
                {
                    **common,
                    "sample_id": f"{summary['run_id']}:event_choice:{floor['floor_number']}",
                    "decision_type": "event_choice",
                    "supervision_type": "chosen_action_only",
                    "selected_event_title_keys": [
                        item["title_key"]
                        for item in floor["event_choices"]
                        if item.get("title_key")
                    ],
                    "choices": floor["event_choices"],
                }
            )

        if floor["ancient_choices"]:
            samples.append(
                {
                    **common,
                    "sample_id": f"{summary['run_id']}:ancient_choice:{floor['floor_number']}",
                    "decision_type": "ancient_choice",
                    "supervision_type": "full_choice_set",
                    "option_ids": [item["text_key"] for item in floor["ancient_choices"] if item.get("text_key")],
                    "selected_ids": [
                        item["text_key"]
                        for item in floor["ancient_choices"]
                        if item.get("was_chosen") and item.get("text_key")
                    ],
                    "choices": floor["ancient_choices"],
                }
            )

        if floor["relic_choices"] and floor["room_type"] != "shop" and not floor["ancient_choices"]:
            samples.append(
                {
                    **common,
                    "sample_id": f"{summary['run_id']}:relic_choice:{floor['floor_number']}",
                    "decision_type": "relic_choice",
                    "supervision_type": "full_choice_set",
                    "option_ids": [item["choice"] for item in floor["relic_choices"] if item.get("choice")],
                    "selected_ids": [
                        item["choice"]
                        for item in floor["relic_choices"]
                        if item.get("was_picked") and item.get("choice")
                    ],
                    "choices": floor["relic_choices"],
                }
            )

        if floor["potion_choices"] and floor["room_type"] != "shop":
            samples.append(
                {
                    **common,
                    "sample_id": f"{summary['run_id']}:potion_choice:{floor['floor_number']}",
                    "decision_type": "potion_choice",
                    "supervision_type": "full_choice_set",
                    "option_ids": [item["choice"] for item in floor["potion_choices"] if item.get("choice")],
                    "selected_ids": [
                        item["choice"]
                        for item in floor["potion_choices"]
                        if item.get("was_picked") and item.get("choice")
                    ],
                    "choices": floor["potion_choices"],
                }
            )

    return samples


def _normalize_card_ref(card: Any) -> dict[str, Any] | None:
    if not isinstance(card, dict):
        return None

    out = {"id": card.get("id")}
    if card.get("floor_added_to_deck") is not None:
        out["floor_added_to_deck"] = card.get("floor_added_to_deck")
    if card.get("current_upgrade_level") is not None:
        out["current_upgrade_level"] = card.get("current_upgrade_level")
    if out.get("id") is None:
        return None
    return out


def _normalize_choice_ref(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None

    out: dict[str, Any] = {}
    if entry.get("id") is not None:
        out["id"] = entry.get("id")
    if entry.get("floor_added_to_deck") is not None:
        out["floor_added_to_deck"] = entry.get("floor_added_to_deck")
    if entry.get("props") is not None:
        out["props"] = entry.get("props")
    return out or None


def _normalize_card_choices(card_choices: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in card_choices:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "card": _normalize_card_ref(item.get("card")),
                "was_picked": bool(item.get("was_picked")),
            }
        )
    return out


def _normalize_marked_choices(choices: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in choices:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "choice": item.get("choice"),
                "was_picked": bool(item.get("was_picked")),
            }
        )
    return out


def _normalize_event_choices(event_choices: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in event_choices:
        if not isinstance(item, dict):
            continue
        title = item.get("title") if isinstance(item.get("title"), dict) else {}
        out.append(
            {
                "title_key": title.get("key"),
                "title_table": title.get("table"),
                "variables": item.get("variables") or {},
            }
        )
    return out


def _normalize_ancient_choices(choices: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in choices:
        if not isinstance(item, dict):
            continue
        title = item.get("title") if isinstance(item.get("title"), dict) else {}
        out.append(
            {
                "text_key": item.get("TextKey"),
                "title_key": title.get("key"),
                "title_table": title.get("table"),
                "was_chosen": bool(item.get("was_chosen")),
            }
        )
    return out


def _stable_dataset_split(run_id: str) -> str:
    bucket = int(hashlib.md5(run_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"
