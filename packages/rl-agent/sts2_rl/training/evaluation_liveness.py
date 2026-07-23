"""Read-only greedy-policy liveness telemetry and fail-closed early guards.

The journal is evaluation data only.  This module never constructs a rollout,
replay item, reward, or action target.  Repeated validation gates may stop a
bad lineage, while a separate final-audit seed namespace remains untouched.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from .config import RuntimeConfig

LIVENESS_TELEMETRY_SCHEMA: Final = "sts2-greedy-liveness-telemetry-v1"
LIVENESS_GUARD_SCHEMA: Final = "sts2-greedy-liveness-guard-v1"


def _kind(action: object) -> str:
    if not isinstance(action, Mapping):
        return "unknown"
    # The concrete simulator operation distinguishes select/deselect/confirm
    # and claim/proceed; model_action_kind intentionally groups those actions.
    value = str(
        action.get(
        "action",
        action.get("kind", action.get("model_action_kind", "unknown")),
        )
    ).strip().lower()
    selection_operations = {
        "select_card": "select_card",
        "select_hand_card": "select_card",
        "select_card_option": "select_card",
        "combat_select_card": "select_card",
        "deselect_card": "deselect_card",
        "deselect_hand_card": "deselect_card",
        "deselect_card_option": "deselect_card",
        "combat_deselect_card": "deselect_card",
        "confirm_selection": "confirm_selection",
        "combat_confirm_selection": "confirm_selection",
        "cancel_selection": "cancel_selection",
        "combat_cancel_selection": "cancel_selection",
    }
    return selection_operations.get(
        value,
        value or "unknown",
    )


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        return None
    return result


def _decision_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"evaluation journal contains invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"evaluation journal line {line_number} must contain an object"
                )
            if value.get("event") != "decision":
                continue
            if value.get("record_kind") not in (None, "summary"):
                # Rich records duplicate compact decisions.
                continue
            result.append(value)
    return tuple(result)


def summarize_greedy_liveness_journal(path: str | Path) -> dict[str, Any]:
    """Aggregate deterministic-policy failure indicators from one journal."""

    records = _decision_records(path)
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    confirm_ready = 0
    confirm_selected = 0
    multi_action_end_turn = 0
    avoidable_end_turn_selected = 0
    reward_hubs = 0
    reward_claim_selected = 0
    reward_proceed_selected = 0
    reward_margins: list[float] = []

    for record in records:
        episode_id = str(record.get("episode_id", "")).strip()
        if episode_id:
            by_episode[episode_id].append(record)
        selected_kind = _kind(record.get("selected_action"))
        kinds_value = record.get("legal_action_kinds")
        kinds = kinds_value if isinstance(kinds_value, Mapping) else {}

        observation_value = record.get(
            "observation_summary",
            record.get("observation"),
        )
        observation = (
            observation_value
            if isinstance(observation_value, Mapping)
            else {}
        )
        selection_value = observation.get("card_selection")
        selection = (
            selection_value if isinstance(selection_value, Mapping) else {}
        )
        can_confirm = selection.get("can_confirm") is True

        topk_value = record.get("policy_topk")
        topk = (
            tuple(item for item in topk_value if isinstance(item, Mapping))
            if isinstance(topk_value, Sequence)
            and not isinstance(topk_value, str | bytes)
            else ()
        )
        topk_kinds = tuple(_kind(item.get("action")) for item in topk)
        if can_confirm or "confirm_selection" in topk_kinds:
            confirm_ready += 1
            confirm_selected += int(selected_kind == "confirm_selection")

        end_count = _integer(kinds.get("end_turn")) or 0
        total_legal = sum(
            value
            for value in (
                _integer(raw_value) for raw_value in kinds.values()
            )
            if value is not None and value > 0
        )
        if end_count > 0 and total_legal > end_count:
            multi_action_end_turn += 1
            avoidable_end_turn_selected += int(selected_kind == "end_turn")

        probabilities: dict[str, list[float]] = defaultdict(list)
        for item, action_kind in zip(topk, topk_kinds, strict=True):
            probability = _probability(item.get("probability"))
            if probability is not None:
                probabilities[action_kind].append(probability)
        if probabilities["claim_reward"] and probabilities["proceed"]:
            reward_hubs += 1
            reward_claim_selected += int(selected_kind == "claim_reward")
            reward_proceed_selected += int(selected_kind == "proceed")
            reward_margins.append(
                max(probabilities["proceed"])
                - max(probabilities["claim_reward"])
            )

    selection_cycle_episodes = 0
    combat_stall_episodes = 0
    noncombat_stall_episodes = 0
    for episode_records in by_episode.values():
        if not episode_records:
            continue
        last = episode_records[-1]
        deadlock_value = last.get("deadlock")
        deadlock = deadlock_value if isinstance(deadlock_value, Mapping) else {}
        deadlock_kind = str(deadlock.get("kind", "")).strip().lower()
        if deadlock_kind == "combat_no_net_progress":
            combat_stall_episodes += 1
        if deadlock_kind == "noncombat_no_durable_progress":
            noncombat_stall_episodes += 1

        cycle_span = _integer(deadlock.get("cycle_span"))
        tail_kinds = tuple(
            _kind(record.get("selected_action"))
            for record in episode_records[-16:]
        )
        alternating_selection_tail = (
            len(tail_kinds) >= 8
            and set(tail_kinds).issubset({"select_card", "deselect_card"})
            and all(
                left != right
                for left, right in pairwise(tail_kinds)
            )
        )
        if (
            cycle_span is not None
            and cycle_span <= 2
            and (
                _kind(last.get("selected_action"))
                in {"select_card", "deselect_card", "confirm_selection"}
                or alternating_selection_tail
            )
        ):
            selection_cycle_episodes += 1

    episode_count = len(by_episode)
    confirm_rate = (
        confirm_selected / confirm_ready if confirm_ready else None
    )
    avoidable_end_rate = (
        avoidable_end_turn_selected / multi_action_end_turn
        if multi_action_end_turn
        else None
    )
    return {
        "schema_version": LIVENESS_TELEMETRY_SCHEMA,
        "diagnostic_only": True,
        "training_samples_emitted": 0,
        "decision_count": len(records),
        "episode_count": episode_count,
        "confirm_ready_decisions": confirm_ready,
        "confirm_selected_decisions": confirm_selected,
        "confirm_ready_greedy_rate": confirm_rate,
        "confirm_ready_failure_rate": (
            1.0 - confirm_rate if confirm_rate is not None else None
        ),
        "multi_action_end_turn_decisions": multi_action_end_turn,
        "avoidable_end_turn_selected_decisions": avoidable_end_turn_selected,
        "multi_action_end_turn_greedy_rate": avoidable_end_rate,
        "reward_hub_decisions": reward_hubs,
        "reward_claim_selected_decisions": reward_claim_selected,
        "reward_proceed_selected_decisions": reward_proceed_selected,
        "reward_proceed_minus_best_claim_mean_margin": (
            statistics.fmean(reward_margins) if reward_margins else None
        ),
        "selection_cycle_episode_count": selection_cycle_episodes,
        "selection_cycle_episode_rate": (
            selection_cycle_episodes / episode_count if episode_count else 0.0
        ),
        "combat_progress_stall_episode_count": combat_stall_episodes,
        "noncombat_progress_stall_episode_count": noncombat_stall_episodes,
    }


def evaluate_liveness_guard(
    telemetry: Mapping[str, Any],
    config: RuntimeConfig,
) -> dict[str, Any]:
    """Return a deterministic stop decision for obvious liveness collapse.

    Floor, win rate, revival count and HP loss are deliberately absent.  Those
    outcome metrics are noisy at tiny early gates and must not stop training.
    """

    violations: list[dict[str, Any]] = []
    confirm_exposures = _integer(telemetry.get("confirm_ready_decisions")) or 0
    confirm_failure = telemetry.get("confirm_ready_failure_rate")
    if (
        confirm_exposures >= config.evaluation_guard_min_confirm_ready
        and isinstance(confirm_failure, int | float)
        and float(confirm_failure)
        >= config.evaluation_guard_max_confirm_failure_rate
    ):
        violations.append(
            {
                "kind": "confirm_ready_greedy_failure",
                "observed": float(confirm_failure),
                "threshold": config.evaluation_guard_max_confirm_failure_rate,
                "exposures": confirm_exposures,
            }
        )

    end_exposures = (
        _integer(telemetry.get("multi_action_end_turn_decisions")) or 0
    )
    end_rate = telemetry.get("multi_action_end_turn_greedy_rate")
    if (
        end_exposures >= config.evaluation_guard_min_multi_action_end_turn
        and isinstance(end_rate, int | float)
        and float(end_rate)
        >= config.evaluation_guard_max_multi_action_end_turn_rate
    ):
        violations.append(
            {
                "kind": "avoidable_end_turn_collapse",
                "observed": float(end_rate),
                "threshold": (
                    config.evaluation_guard_max_multi_action_end_turn_rate
                ),
                "exposures": end_exposures,
            }
        )

    cycle_rate = telemetry.get("selection_cycle_episode_rate")
    if (
        isinstance(cycle_rate, int | float)
        and float(cycle_rate)
        >= config.evaluation_guard_max_selection_cycle_episode_rate
    ):
        violations.append(
            {
                "kind": "selection_cycle_collapse",
                "observed": float(cycle_rate),
                "threshold": (
                    config.evaluation_guard_max_selection_cycle_episode_rate
                ),
                "episodes": _integer(telemetry.get("episode_count")) or 0,
            }
        )

    enabled = config.evaluation_liveness_guard_enabled
    return {
        "schema_version": LIVENESS_GUARD_SCHEMA,
        "enabled": enabled,
        "passed": not enabled or not violations,
        "stop_requested": enabled and bool(violations),
        "violations": violations,
        "outcome_metrics_used_for_stop": False,
        "thresholds": {
            "minimum_confirm_ready_decisions": (
                config.evaluation_guard_min_confirm_ready
            ),
            "minimum_multi_action_end_turn_decisions": (
                config.evaluation_guard_min_multi_action_end_turn
            ),
            "maximum_confirm_failure_rate": (
                config.evaluation_guard_max_confirm_failure_rate
            ),
            "maximum_multi_action_end_turn_rate": (
                config.evaluation_guard_max_multi_action_end_turn_rate
            ),
            "maximum_selection_cycle_episode_rate": (
                config.evaluation_guard_max_selection_cycle_episode_rate
            ),
        },
    }


__all__ = [
    "LIVENESS_GUARD_SCHEMA",
    "LIVENESS_TELEMETRY_SCHEMA",
    "evaluate_liveness_guard",
    "summarize_greedy_liveness_journal",
]
