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


def _shop_item_category(action: object) -> str | None:
    """Return the authoritative nested subtype for a shop purchase."""

    if _kind(action) != "shop_purchase" or not isinstance(action, Mapping):
        return None
    item_value = action.get("item")
    if not isinstance(item_value, Mapping):
        return None
    for key in ("category", "type", "item_type", "item_kind", "kind"):
        value = str(item_value.get(key, "")).strip().lower()
        if value:
            return value
    return None


def _deck_count(record: Mapping[str, Any]) -> int | None:
    observation_value = record.get(
        "observation_summary",
        record.get("observation"),
    )
    if not isinstance(observation_value, Mapping):
        return None
    player_value = observation_value.get("player")
    if not isinstance(player_value, Mapping):
        return None
    return _integer(player_value.get("deck_count"))


def _card_removal_outcomes(
    episode_records: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int, int]:
    """Count attempted, observed-complete, cancelled and unresolved removals.

    Completion is intentionally stricter than seeing ``confirm_selection``:
    the next actionable state must show a smaller deck.  This makes the metric
    an execution audit rather than a count of policy intentions.  Old compact
    journals without deck counts remain readable; their confirmed operations
    are conservatively classified as unresolved instead of being invented as
    successes.
    """

    attempts = 0
    completed = 0
    cancelled = 0
    pending_before_deck: int | None = None
    pending = False
    awaiting_post_confirm = False

    for record in episode_records:
        if awaiting_post_confirm:
            after_deck = _deck_count(record)
            if (
                pending_before_deck is not None
                and after_deck is not None
                and after_deck < pending_before_deck
            ):
                completed += 1
            pending = False
            awaiting_post_confirm = False
            pending_before_deck = None

        action = record.get("selected_action")
        if _shop_item_category(action) == "card_removal":
            # A second purchase before the prior transaction settles leaves
            # the prior attempt unresolved; the residual calculation below
            # records it without guessing why the UI changed.
            attempts += 1
            pending = True
            awaiting_post_confirm = False
            pending_before_deck = _deck_count(record)
            continue

        if not pending:
            continue
        action_kind = _kind(action)
        if action_kind == "cancel_selection":
            cancelled += 1
            pending = False
            pending_before_deck = None
        elif action_kind == "confirm_selection":
            awaiting_post_confirm = True

    unresolved = attempts - completed - cancelled
    if unresolved < 0:  # pragma: no cover - guarded by the state machine
        raise AssertionError("card-removal outcome counts became inconsistent")
    return attempts, completed, cancelled, unresolved


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
    selection_exit_ready = 0
    selection_exit_selected = 0
    multi_action_end_turn = 0
    avoidable_end_turn_selected = 0
    reward_hubs = 0
    reward_claim_selected = 0
    reward_proceed_selected = 0
    reward_margins: list[float] = []
    card_removal_legal_decisions = 0
    card_removal_legal_candidates = 0

    for record in records:
        episode_id = str(record.get("episode_id", "")).strip()
        if episode_id:
            by_episode[episode_id].append(record)
        selected_kind = _kind(record.get("selected_action"))
        kinds_value = record.get("legal_action_kinds")
        kinds = kinds_value if isinstance(kinds_value, Mapping) else {}
        semantics_value = record.get("legal_action_semantics")
        semantics = (
            semantics_value if isinstance(semantics_value, Mapping) else {}
        )
        removal_legal_count = _integer(
            semantics.get("shop_purchase:card_removal")
        )
        if removal_legal_count is not None and removal_legal_count > 0:
            card_removal_legal_decisions += 1
            card_removal_legal_candidates += removal_legal_count

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
        optional_cancel_exit = (
            selection.get("can_cancel") is True
            and _integer(selection.get("min_select")) == 0
        )

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
            selection_exit_ready += 1
            selection_exit_selected += int(
                selected_kind == "confirm_selection"
                or (
                    optional_cancel_exit
                    and selected_kind == "cancel_selection"
                )
            )

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
    cycle_episodes = 0
    liveness_failure_episodes = 0
    card_removal_cancel_cycle_episodes = 0
    card_removal_purchase_attempts = 0
    card_removal_completed = 0
    card_removal_cancelled = 0
    card_removal_unresolved = 0
    combat_stall_episodes = 0
    noncombat_stall_episodes = 0
    for episode_records in by_episode.values():
        if not episode_records:
            continue
        (
            episode_removal_attempts,
            episode_removal_completed,
            episode_removal_cancelled,
            episode_removal_unresolved,
        ) = _card_removal_outcomes(episode_records)
        card_removal_purchase_attempts += episode_removal_attempts
        card_removal_completed += episode_removal_completed
        card_removal_cancelled += episode_removal_cancelled
        card_removal_unresolved += episode_removal_unresolved
        last = episode_records[-1]
        deadlock_value = last.get("deadlock")
        deadlock = deadlock_value if isinstance(deadlock_value, Mapping) else {}
        if deadlock:
            liveness_failure_episodes += 1
        deadlock_kind = str(deadlock.get("kind", "")).strip().lower()
        if deadlock_kind == "combat_no_net_progress":
            combat_stall_episodes += 1
        if deadlock_kind == "noncombat_no_durable_progress":
            noncombat_stall_episodes += 1

        cycle_span = _integer(deadlock.get("cycle_span"))
        if cycle_span is not None and cycle_span > 0:
            cycle_episodes += 1
        tail_kinds = tuple(
            _kind(record.get("selected_action"))
            for record in episode_records[-16:]
        )
        tail_actions = tuple(
            record.get("selected_action")
            if isinstance(record.get("selected_action"), Mapping)
            else {}
            for record in episode_records[-16:]
        )
        has_card_removal_purchase = any(
            _shop_item_category(action) == "card_removal"
            for action in tail_actions
        )
        # Choosing a rest-site option and later cancelling a card selection is
        # authoritative evidence that the option opened a transaction. A
        # normal HP rest exits the room and cannot produce this same cycle.
        has_rest_site_transaction_entry = "choose_rest_option" in tail_kinds
        has_transaction_cancel_cycle = bool(
            cycle_span is not None
            and cycle_span > 0
            and "cancel_selection" in tail_kinds
            and (has_card_removal_purchase or has_rest_site_transaction_entry)
        )
        if (
            cycle_span is not None
            and cycle_span > 0
            and has_card_removal_purchase
            and "cancel_selection" in tail_kinds
        ):
            card_removal_cancel_cycle_episodes += 1
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
            and cycle_span > 0
            and (
                _kind(last.get("selected_action"))
                in {"select_card", "deselect_card", "confirm_selection"}
                or alternating_selection_tail
                or has_transaction_cancel_cycle
            )
        ):
            selection_cycle_episodes += 1

    episode_count = len(by_episode)
    confirm_rate = (
        confirm_selected / confirm_ready if confirm_ready else None
    )
    selection_exit_rate = (
        selection_exit_selected / selection_exit_ready
        if selection_exit_ready
        else None
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
        # Preserve the historical confirm-only fields above for report
        # compatibility.  The hard guard uses exit semantics: Cancel is a
        # valid exit when the simulator explicitly declares the window
        # optional (min_select=0, can_cancel=true).
        "selection_exit_ready_decisions": selection_exit_ready,
        "selection_exit_selected_decisions": selection_exit_selected,
        "selection_exit_greedy_rate": selection_exit_rate,
        "selection_exit_failure_rate": (
            1.0 - selection_exit_rate
            if selection_exit_rate is not None
            else None
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
        "cycle_episode_count": cycle_episodes,
        "cycle_episode_rate": (
            cycle_episodes / episode_count if episode_count else 0.0
        ),
        "liveness_failure_episode_count": liveness_failure_episodes,
        "liveness_failure_episode_rate": (
            liveness_failure_episodes / episode_count if episode_count else 0.0
        ),
        "card_removal_cancel_cycle_episode_count": (
            card_removal_cancel_cycle_episodes
        ),
        "card_removal_legal_decision_count": card_removal_legal_decisions,
        "card_removal_legal_candidate_count": card_removal_legal_candidates,
        "card_removal_purchase_attempt_count": card_removal_purchase_attempts,
        "card_removal_completed_count": card_removal_completed,
        "card_removal_cancelled_count": card_removal_cancelled,
        "card_removal_unresolved_count": card_removal_unresolved,
        "card_removal_completion_rate": (
            card_removal_completed / card_removal_purchase_attempts
            if card_removal_purchase_attempts
            else None
        ),
        "card_removal_cancel_rate": (
            card_removal_cancelled / card_removal_purchase_attempts
            if card_removal_purchase_attempts
            else None
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
    # Persisted pre-exit-semantics summaries contain only confirm_* fields.
    # Falling back keeps those old reports readable and preserves their guard
    # result while new summaries distinguish a legal optional Cancel.
    confirm_exposures = _integer(
        telemetry.get(
            "selection_exit_ready_decisions",
            telemetry.get("confirm_ready_decisions"),
        )
    ) or 0
    confirm_failure = telemetry.get(
        "selection_exit_failure_rate",
        telemetry.get("confirm_ready_failure_rate"),
    )
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

    episode_count = _integer(telemetry.get("episode_count")) or 0
    liveness_failure_rate = telemetry.get("liveness_failure_episode_rate")
    baseline_episodes = config.evaluation_guard_liveness_baseline_episodes
    baseline_failures = config.evaluation_guard_liveness_baseline_failures
    baseline_rate = (
        baseline_failures / baseline_episodes
        if baseline_episodes > 0
        else None
    )
    effective_liveness_threshold = (
        min(
            1.0,
            max(
                config.evaluation_guard_max_liveness_failure_episode_rate,
                baseline_rate
                + config.evaluation_guard_min_liveness_regression_rate,
            ),
        )
        if baseline_rate is not None
        else config.evaluation_guard_max_liveness_failure_episode_rate
    )
    if (
        episode_count >= config.evaluation_guard_min_liveness_episodes
        and isinstance(liveness_failure_rate, int | float)
        and float(liveness_failure_rate)
        >= effective_liveness_threshold
    ):
        violations.append(
            {
                "kind": "liveness_failure_collapse",
                "observed": float(liveness_failure_rate),
                "threshold": effective_liveness_threshold,
                "absolute_threshold": (
                    config.evaluation_guard_max_liveness_failure_episode_rate
                ),
                "baseline": (
                    {
                        "episodes": baseline_episodes,
                        "failures": baseline_failures,
                        "rate": baseline_rate,
                        "minimum_regression_rate": (
                            config.evaluation_guard_min_liveness_regression_rate
                        ),
                    }
                    if baseline_rate is not None
                    else None
                ),
                "episodes": episode_count,
                "failures": (
                    _integer(
                        telemetry.get("liveness_failure_episode_count")
                    )
                    or 0
                ),
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
            "minimum_liveness_episodes": (
                config.evaluation_guard_min_liveness_episodes
            ),
            "maximum_liveness_failure_episode_rate": (
                config.evaluation_guard_max_liveness_failure_episode_rate
            ),
            "liveness_baseline_failures": baseline_failures,
            "liveness_baseline_episodes": baseline_episodes,
            "minimum_liveness_regression_rate": (
                config.evaluation_guard_min_liveness_regression_rate
            ),
            "effective_liveness_failure_episode_rate": (
                effective_liveness_threshold
            ),
        },
    }


__all__ = [
    "LIVENESS_GUARD_SCHEMA",
    "LIVENESS_TELEMETRY_SCHEMA",
    "evaluate_liveness_guard",
    "summarize_greedy_liveness_journal",
]
