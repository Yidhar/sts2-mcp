"""Shared safe-progress candidate selection for combat hard guards.

Historically the no-pressure pure-block hard guard and the selected-side
``bad_pure_block_selected`` diagnostic each carried their own copy of the
"is there a safe progress alternative?" scan.  Once those drifted, TensorBoard
could correctly flag bad Defend choices while the guard reported
``candidate_count=0`` and never rewrote the action.

Keep the predicate here so metrics and hard guards use the same tactical
contract:

* only legal/affordable ``play_card`` actions;
* not bad zero-energy X-cost;
* no non-lethal HP-cost plays;
* classified positive by the existing combat-action profile;
* not pure block / block waste itself;
* immediate damage, lethal, impact, or a real progress role;
* skip defer/follow-up-dependent setup unless it is lethal/high impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sts2_env.hp_cost_safety import hp_cost_safety_view


PROGRESS_ROLE_NAMES = frozenset(
    {
        "attack",
        "damage",
        "debuff",
        "weak",
        "vulnerable",
        "poison",
        "stun",
        "artifact_strip",
        "lock",
        "mechanism",
        "facing_change",
        "scaling",
        "power",
        "draw",
        "discard",
        "resource",
        "energy",
    }
)


@dataclass(frozen=True)
class ProgressCandidate:
    """One legal, safe fight-progress alternative."""

    score: tuple[float, float, float, float, float]
    index: int
    damage: float
    impact: float
    cost: float
    lethal: bool
    title: str = ""


def _inc(rejection_counts: dict[str, int], reason: str) -> None:
    rejection_counts[reason] = int(rejection_counts.get(reason, 0)) + 1


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _action_title(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    card = action.get("card") if isinstance(action.get("card"), dict) else {}
    return str(action.get("title") or card.get("title") or card.get("name") or action.get("label") or "")


def _mask_allows(mask_np: Any, idx: int) -> bool:
    try:
        arr = np.asarray(mask_np, dtype=np.float32).reshape(-1)
        return idx < int(arr.shape[0]) and float(arr[idx]) > 0.0
    except Exception:
        return False


def collect_safe_progress_candidates(
    owner: Any,
    *,
    selected_idx: int,
    legal_count: int,
    legal_actions: list[Any],
    mask_np: Any,
    raw_obs: dict[str, Any] | None,
    current_energy: float,
    use_mask: bool = True,
    include_debug: bool = False,
) -> tuple[list[ProgressCandidate], dict[str, int]]:
    """Return legal safe progress alternatives using the shared CQ predicate.

    ``owner`` is normally ``MuZeroTrainer`` (or a lightweight test instance)
    and provides the existing action helper methods.  ``use_mask=False`` is for
    selected-side diagnostics where the real post-search mask is not threaded
    through; that call site intentionally asks "would the hand contain a safe
    progress alternative by the same semantic rules?" without claiming the
    hard guard had a real masked candidate.
    """

    rejection_counts: dict[str, int] = {}
    candidates: list[ProgressCandidate] = []

    try:
        count = max(0, int(legal_count))
    except (TypeError, ValueError):
        count = 0
    count = min(count, len(legal_actions) if isinstance(legal_actions, list) else 0)
    energy = _as_float(current_energy)

    for idx in range(count):
        if int(idx) == int(selected_idx):
            continue
        if use_mask and not _mask_allows(mask_np, int(idx)):
            _inc(rejection_counts, "mask")
            continue

        alt = legal_actions[idx]
        if not isinstance(alt, dict):
            _inc(rejection_counts, "missing_action")
            continue
        if owner._semantic_family(alt) != "play_card":
            _inc(rejection_counts, "not_play_card")
            continue

        cost = _as_float(owner._action_cost_value(alt))
        if cost > energy + 1e-6:
            _inc(rejection_counts, "unaffordable")
            continue

        try:
            x_diag = owner._x_cost_diagnostic(alt, energy)
            if _as_float((x_diag or {}).get("x_cost_bad", 0.0)) > 0.5:
                _inc(rejection_counts, "x_cost_bad")
                continue
        except Exception:
            # If diagnostics are unavailable in a lightweight test, do not make
            # the candidate disappear solely because instrumentation failed.
            pass

        lethal = bool(owner._is_action_confirmed_lethal(alt, raw_obs))
        try:
            safety = hp_cost_safety_view(alt, raw_obs)
            hp_loss = _as_float((safety or {}).get("hp_loss_unblockable", 0.0))
        except Exception:
            hp_loss = 0.0
        if hp_loss > 0.0 and not lethal:
            _inc(rejection_counts, "hp_cost_nonlethal")
            continue

        try:
            profile = owner._classify_positive_combat_action(
                alt,
                int(idx),
                None,
                raw_obs,
                legal_actions,
                mask_np,
                energy,
            )
        except Exception:
            profile = {}

        if not bool((profile or {}).get("positive", False)):
            _inc(rejection_counts, "not_positive")
            continue
        if bool((profile or {}).get("card_block_waste", False)) or bool(
            (profile or {}).get("card_pure_block", False)
        ):
            _inc(rejection_counts, "block_waste_or_pure_block")
            continue

        damage = max(
            _as_float(owner._action_metric(alt, "damage")),
            _as_float(owner._action_metric(alt, "total_damage")),
            _as_float(
                owner._action_numeric_value(
                    alt,
                    ("damage", "total_damage", "attack_damage", "preview_damage", "expected_damage"),
                )
            ),
        )
        impact = _as_float(owner._action_immediate_impact(alt))
        try:
            roles = set(owner._action_roles(alt))
        except Exception:
            roles = set()
        progress_roles = bool(roles.intersection(PROGRESS_ROLE_NAMES))
        high_damage = bool(damage >= 8.0 or impact >= 8.0)
        if not bool(lethal or damage > 0.0 or impact >= 3.0 or progress_roles):
            _inc(rejection_counts, "no_meaningful_progress")
            continue

        if bool((profile or {}).get("deferable", False)) and not (lethal or high_damage):
            _inc(rejection_counts, "deferable")
            continue
        if bool((profile or {}).get("followup_missing", False)) and not (lethal or high_damage):
            _inc(rejection_counts, "followup_missing")
            continue
        if bool((profile or {}).get("energy_without_followup", False)) and not (lethal or high_damage):
            _inc(rejection_counts, "energy_without_followup")
            continue
        if (
            bool((profile or {}).get("setup_followup_dependent", False))
            and not bool((profile or {}).get("setup_followup_available", False))
            and not (lethal or high_damage)
        ):
            _inc(rejection_counts, "setup_followup_missing")
            continue

        _inc(rejection_counts, "accepted")
        candidates.append(
            ProgressCandidate(
                score=(1.0 if lethal else 0.0, float(damage), float(impact), -float(cost), -float(idx)),
                index=int(idx),
                damage=float(damage),
                impact=float(impact),
                cost=float(cost),
                lethal=bool(lethal),
                title=_action_title(alt),
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    if not include_debug:
        return candidates, {}
    return candidates, rejection_counts


__all__ = ["PROGRESS_ROLE_NAMES", "ProgressCandidate", "collect_safe_progress_candidates"]
