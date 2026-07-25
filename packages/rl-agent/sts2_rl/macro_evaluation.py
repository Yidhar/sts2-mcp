"""Diagnostic-only macro-policy telemetry and fixed sensitivity probes.

The maintained learner deliberately has no hand-written deck, route, rest or
shop policy.  This module does not add one.  It answers two narrower questions:

* which macro decision surfaces were actually exposed during a held-out run,
  and how concentrated was the frozen policy on those surfaces; and
* does a frozen policy react at all when player-visible world facts change
  while the legal candidate set remains exactly fixed?

Both paths are evaluation-only.  They do not construct rollout/replay records,
do not name a preferred action, and do not feed diagnostics back into training.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

import torch
from torch import Tensor

from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path
from sts2_rl.checkpoints import validate_resume_checkpoint
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.models import RecurrentCandidateModel, RecurrentCandidateOutput
from sts2_rl.training.checkpointing import preflight_model_initialization
from sts2_rl.training.config import CONFIG_VERSION, TrainingConfig, training_config_from_mapping
from sts2_rl.training.trajectory import semantic_action_fingerprint

MacroSurface: TypeAlias = Literal["card_reward", "map", "rest", "shop"]
MACRO_SURFACES: Final[tuple[MacroSurface, ...]] = (
    "card_reward",
    "map",
    "rest",
    "shop",
)
MACRO_TELEMETRY_SCHEMA: Final = "sts2-macro-surface-telemetry-v1"
MACRO_SENSITIVITY_SCHEMA: Final = "sts2-macro-policy-sensitivity-v1"
FIXED_MACRO_PROBE_SUITE_VERSION: Final = "grounded-visible-facts-v1"
_V6_CONFIG_VERSION: Final = "sts2-relational-curriculum-config-v6"
_V7_CONFIG_VERSION: Final = "sts2-relational-curriculum-config-v7"
_V8_CONFIG_VERSION: Final = "sts2-relational-curriculum-config-v8"
_REVIEWED_DIAGNOSTIC_CONFIG_VERSIONS: Final = frozenset(
    {
        _V6_CONFIG_VERSION,
        _V7_CONFIG_VERSION,
        _V8_CONFIG_VERSION,
    }
)

_ACTION_KIND_TO_SURFACE: Final[dict[str, MacroSurface]] = {
    "card_reward": "card_reward",
    "map": "map",
    "rest_site": "rest",
    "shop": "shop",
}
_PHASE_TO_SURFACE: Final[dict[str, MacroSurface]] = {
    "card_reward": "card_reward",
    "map": "map",
    "navigation": "map",
    "route": "map",
    "rest": "rest",
    "rest_site": "rest",
    "campfire": "rest",
    "shop": "shop",
    "merchant": "shop",
}
_OUTPUT_VALUE_FIELDS: Final[tuple[str, ...]] = (
    "value",
    "combat_task_value",
    "act_task_value",
    "run_task_value",
    "combat_revival_cost_value",
    "act_revival_cost_value",
    "run_revival_cost_value",
)
_CANDIDATE_TENSOR_FIELDS: Final[tuple[str, ...]] = (
    "features",
    "type_ids",
    "role_ids",
    "owner_ids",
    "entity_ids",
    "entity_aux_ids",
    "zone_ids",
    "target_owner_ids",
    "target_entity_ids",
    "target_entity_aux_ids",
    "local_features",
    "local_mask",
    "local_type_ids",
    "local_role_ids",
    "local_owner_ids",
    "local_entity_ids",
    "local_entity_aux_ids",
    "local_zone_ids",
    "local_order_ids",
    "action_mask",
)
_WORLD_TENSOR_FIELDS: Final[tuple[str, ...]] = (
    "features",
    "mask",
    "type_ids",
    "role_ids",
    "owner_ids",
    "entity_ids",
    "entity_aux_ids",
    "zone_ids",
    "order_ids",
)


@dataclass(frozen=True, slots=True)
class MacroSensitivityCase:
    """One label-free, world-only paired policy probe."""

    name: str
    surface: MacroSurface
    changed_facts: tuple[str, ...]
    reference_observation: Mapping[str, Any]
    comparison_observation: Mapping[str, Any]
    legal_actions: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("macro sensitivity case name must be non-empty")
        if self.surface not in MACRO_SURFACES:
            raise ValueError(f"unsupported macro surface: {self.surface!r}")
        if not self.changed_facts:
            raise ValueError("macro sensitivity case must identify changed facts")
        if len(self.legal_actions) < 2:
            raise ValueError("macro sensitivity case requires at least two candidates")


@dataclass(frozen=True, slots=True)
class _EntropyBounds:
    lower: float
    upper: float
    exact: float | None
    top_probability: float
    top_gap_above_uniform: float
    recorded_mass: float
    recorded_count: int


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _finite_probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        return None
    return result


def _action_kind(action: object) -> str:
    if not isinstance(action, Mapping):
        return "unknown"
    value = action.get(
        "model_action_kind",
        action.get("kind", action.get("action", "unknown")),
    )
    return str(value).strip().lower() or "unknown"


def classify_macro_surface(record: Mapping[str, Any]) -> MacroSurface | None:
    """Classify one compact or rich journal decision without strategy rules."""

    kinds_value = record.get("legal_action_kinds")
    if isinstance(kinds_value, Mapping):
        for kind in _ACTION_KIND_TO_SURFACE:
            count = _integer(kinds_value.get(kind))
            if count is not None and count > 0:
                return _ACTION_KIND_TO_SURFACE[kind]

    legal_actions = record.get("legal_actions")
    if isinstance(legal_actions, Sequence) and not isinstance(legal_actions, str | bytes):
        kinds = {_action_kind(action) for action in legal_actions}
        for kind in _ACTION_KIND_TO_SURFACE:
            if kind in kinds:
                return _ACTION_KIND_TO_SURFACE[kind]

    observation = record.get("observation_summary", record.get("observation"))
    if not isinstance(observation, Mapping):
        return None
    phase = (
        str(
            observation.get(
                "phase",
                observation.get("decision_domain", observation.get("domain", "")),
            )
        )
        .strip()
        .lower()
    )
    return _PHASE_TO_SURFACE.get(phase)


def _entropy_bounds(
    *,
    candidate_count: int,
    topk: Sequence[Mapping[str, Any]],
) -> _EntropyBounds | None:
    if candidate_count <= 0:
        return None
    probabilities: list[float] = []
    candidate_indices: set[int] = set()
    for item in topk:
        probability = _finite_probability(item.get("probability"))
        candidate_index = _integer(item.get("candidate_index", item.get("index")))
        if probability is None or candidate_index is None:
            raise ValueError("macro journal policy_topk contains an invalid item")
        if candidate_index < 0 or candidate_index >= candidate_count:
            raise ValueError("macro journal policy_topk candidate index is out of range")
        if candidate_index in candidate_indices:
            raise ValueError("macro journal policy_topk repeats a candidate index")
        candidate_indices.add(candidate_index)
        probabilities.append(probability)
    if not probabilities:
        return None
    if len(probabilities) > candidate_count:
        raise ValueError("macro journal records more probabilities than candidates")

    probabilities.sort(reverse=True)
    recorded_mass = math.fsum(probabilities)
    tolerance = 1e-5
    if recorded_mass > 1.0 + tolerance:
        raise ValueError("macro journal policy probabilities sum above one")
    residual = max(0.0, 1.0 - min(recorded_mass, 1.0))
    missing = candidate_count - len(probabilities)
    if missing == 0:
        if residual > tolerance:
            raise ValueError("complete macro journal policy probabilities do not sum to one")
        entropy = -math.fsum(probability * math.log(probability) for probability in probabilities if probability > 0.0)
        normalized = entropy / math.log(candidate_count) if candidate_count > 1 else 0.0
        top = probabilities[0]
        return _EntropyBounds(
            lower=normalized,
            upper=normalized,
            exact=normalized,
            top_probability=top,
            top_gap_above_uniform=top - (1.0 / candidate_count),
            recorded_mass=recorded_mass,
            recorded_count=len(probabilities),
        )

    cap = probabilities[-1]
    if residual > (missing * cap) + tolerance:
        raise ValueError("macro journal top-k residual is inconsistent with descending policy probabilities")
    base_entropy = -math.fsum(probability * math.log(probability) for probability in probabilities if probability > 0.0)
    maximum_tail_entropy = -residual * math.log(residual / missing) if residual > 0.0 else 0.0
    minimum_tail_entropy = 0.0
    remaining = residual
    for _ in range(missing):
        allocated = min(cap, remaining)
        if allocated > 0.0:
            minimum_tail_entropy -= allocated * math.log(allocated)
        remaining -= allocated
        if remaining <= tolerance:
            break
    if remaining > tolerance:
        raise ValueError("macro journal residual could not fit below the top-k boundary")
    normalizer = math.log(candidate_count) if candidate_count > 1 else 1.0
    top = probabilities[0]
    return _EntropyBounds(
        lower=(base_entropy + minimum_tail_entropy) / normalizer,
        upper=(base_entropy + maximum_tail_entropy) / normalizer,
        exact=None,
        top_probability=top,
        top_gap_above_uniform=top - (1.0 / candidate_count),
        recorded_mass=recorded_mass,
        recorded_count=len(probabilities),
    )


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_macro_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate compact held-out journal decisions by macro surface."""

    decisions: dict[MacroSurface, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("event") != "decision":
            continue
        record_kind = record.get("record_kind")
        if record_kind not in (None, "summary"):
            # Rich snapshots repeat a compact decision and must never double
            # count exposure.
            continue
        surface = classify_macro_surface(record)
        if surface is not None:
            decisions[surface].append(record)

    surfaces: dict[str, Any] = {}
    for surface in MACRO_SURFACES:
        surface_records = decisions.get(surface, [])
        episodes: set[str] = set()
        seeds: set[int] = set()
        candidate_counts: Counter[int] = Counter()
        legal_kinds: Counter[str] = Counter()
        selected_kinds: Counter[str] = Counter()
        top_kinds: Counter[str] = Counter()
        entropy_lower: list[float] = []
        entropy_upper: list[float] = []
        entropy_exact: list[float] = []
        top_probabilities: list[float] = []
        top_gaps: list[float] = []
        recorded_mass: list[float] = []
        recorded_coverage: list[float] = []
        multi_candidate = 0
        probability_decisions = 0

        for record in surface_records:
            episode_id = record.get("episode_id")
            if isinstance(episode_id, str) and episode_id:
                episodes.add(episode_id)
            reset_seed = _integer(record.get("reset_seed"))
            if reset_seed is not None:
                seeds.add(reset_seed)
            candidate_count = _integer(
                record.get(
                    "semantic_candidate_count",
                    record.get("legal_action_count"),
                )
            )
            if candidate_count is None or candidate_count <= 0:
                continue
            candidate_counts[candidate_count] += 1
            if candidate_count > 1:
                multi_candidate += 1

            kinds_value = record.get("legal_action_kinds")
            if isinstance(kinds_value, Mapping):
                for kind, count_value in kinds_value.items():
                    count = _integer(count_value)
                    if count is not None and count > 0:
                        legal_kinds[str(kind)] += count
            selected_kinds[_action_kind(record.get("selected_action"))] += 1

            # Forced singletons have no policy target in the maintained
            # baseline.  Counting their trivial entropy of zero would make a
            # route surface with many forced transitions look spuriously
            # concentrated, so policy-distribution metrics are conditional on
            # there being an actual choice.
            if candidate_count <= 1:
                continue

            raw_topk = record.get("policy_topk")
            if not isinstance(raw_topk, Sequence) or isinstance(raw_topk, str | bytes):
                continue
            topk = [item for item in raw_topk if isinstance(item, Mapping)]
            bounds = _entropy_bounds(candidate_count=candidate_count, topk=topk)
            if bounds is None:
                continue
            probability_decisions += 1
            entropy_lower.append(bounds.lower)
            entropy_upper.append(bounds.upper)
            if bounds.exact is not None:
                entropy_exact.append(bounds.exact)
            top_probabilities.append(bounds.top_probability)
            top_gaps.append(bounds.top_gap_above_uniform)
            recorded_mass.append(bounds.recorded_mass)
            recorded_coverage.append(bounds.recorded_count / candidate_count)
            if topk:
                top_item = max(
                    topk,
                    key=lambda item: _finite_probability(item.get("probability")) or 0.0,
                )
                top_kinds[_action_kind(top_item.get("action"))] += 1

        surfaces[surface] = {
            "decision_count": len(surface_records),
            "episode_count": len(episodes),
            "held_out_seed_count": len(seeds),
            "multi_candidate_decision_count": multi_candidate,
            "forced_singleton_decision_count": (sum(candidate_counts.values()) - multi_candidate),
            "policy_probability_decision_count": probability_decisions,
            "exact_entropy_decision_count": len(entropy_exact),
            "candidate_count_histogram": {
                str(count): occurrences for count, occurrences in sorted(candidate_counts.items())
            },
            "legal_action_kind_counts": dict(sorted(legal_kinds.items())),
            "selected_action_kind_counts": dict(sorted(selected_kinds.items())),
            "top_action_kind_counts": dict(sorted(top_kinds.items())),
            "mean_top_probability": _mean(top_probabilities),
            "mean_top_gap_above_uniform": _mean(top_gaps),
            "mean_normalized_entropy_lower_bound": _mean(entropy_lower),
            "mean_normalized_entropy_upper_bound": _mean(entropy_upper),
            "mean_exact_normalized_entropy": _mean(entropy_exact),
            "mean_recorded_probability_mass": _mean(recorded_mass),
            "mean_recorded_candidate_coverage": _mean(recorded_coverage),
        }

    return {
        "schema_version": MACRO_TELEMETRY_SCHEMA,
        "diagnostic_only": True,
        "training_samples_emitted": 0,
        "surface_order": list(MACRO_SURFACES),
        "total_macro_decisions": sum(int(surface["decision_count"]) for surface in surfaces.values()),
        "surfaces": surfaces,
    }


def read_macro_journal(path: str | Path) -> dict[str, Any]:
    """Read one evaluation JSONL journal and return macro telemetry."""

    journal = Path(path).expanduser().resolve()
    with journal.open("r", encoding="utf-8") as handle:

        def _records() -> Iterable[Mapping[str, Any]]:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"macro journal contains invalid JSON at line {line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"macro journal line {line_number} must contain an object")
                yield value

        return summarize_macro_records(_records())


def _card(
    card_id: str,
    *,
    card_type: str,
    cost: int,
    upgraded: bool = False,
) -> dict[str, Any]:
    return {
        "id": card_id,
        "type": card_type,
        "cost": cost,
        "is_upgraded": upgraded,
    }


def _base_run_player(
    *,
    deck: Sequence[Mapping[str, Any]],
    hp: int = 52,
    gold: int = 120,
    floor: int = 12,
    act: int = 1,
) -> dict[str, Any]:
    return {
        "run": {"active": True, "floor": floor, "act": act},
        "player": {
            "character_id": "CHARACTER.IRONCLAD",
            "hp": hp,
            "max_hp": 80,
            "gold": gold,
            "deck": list(deck),
            "relics": [{"id": "RELIC.STARTER"}],
            "potions": [{"id": "POTION.TEST", "can_use": True}],
        },
    }


def fixed_macro_sensitivity_cases() -> tuple[MacroSensitivityCase, ...]:
    """Return the versioned, label-free fixed macro probe suite."""

    attack_deck = [
        _card("CARD.STRIKE", card_type="Attack", cost=1),
        _card("CARD.STRIKE", card_type="Attack", cost=1),
        _card("CARD.BASH", card_type="Attack", cost=2),
    ]
    skill_deck = [
        _card("CARD.DEFEND", card_type="Skill", cost=1),
        _card("CARD.DEFEND", card_type="Skill", cost=1),
        _card("CARD.DRAW", card_type="Skill", cost=1),
    ]
    offered_cards = (
        _card("CARD.OFFER.ATTACK", card_type="Attack", cost=1),
        _card("CARD.OFFER.SKILL", card_type="Skill", cost=1),
        _card("CARD.OFFER.POWER", card_type="Power", cost=2),
    )
    reward_actions: tuple[Mapping[str, Any], ...] = (
        *(
            {
                "kind": "select_card_reward",
                "model_action_kind": "card_reward",
                "model_action_variant": "select",
                "card": card,
            }
            for card in offered_cards
        ),
        {
            "kind": "skip_card_reward",
            "model_action_kind": "card_reward",
            "model_action_variant": "skip",
        },
    )
    reward_base = {
        "phase": "card_reward",
        "decision_domain": "build",
        **_base_run_player(deck=attack_deck),
        "rewards": {"visible": True, "cards": list(offered_cards)},
    }
    reward_deck = {
        **reward_base,
        **_base_run_player(deck=skill_deck),
    }
    reward_floor = {
        **reward_base,
        **_base_run_player(deck=attack_deck, floor=35, act=3),
    }

    map_actions: tuple[Mapping[str, Any], ...] = (
        {
            "kind": "choose_map_node",
            "model_action_kind": "map",
            "map_node": {
                "coord": {"x": 0, "y": 1},
                "point_type": "Monster",
            },
        },
        {
            "kind": "choose_map_node",
            "model_action_kind": "map",
            "map_node": {
                "coord": {"x": 1, "y": 1},
                "point_type": "Monster",
            },
        },
    )

    def map_observation(
        *,
        hp: int,
        left_downstream: str,
        right_downstream: str,
    ) -> dict[str, Any]:
        return {
            "phase": "map",
            "decision_domain": "route",
            **_base_run_player(deck=attack_deck, hp=hp),
            "map": {
                "is_open": True,
                "current_coord": {"x": 0, "y": 0},
                "points": [
                    {
                        "coord": {"x": 0, "y": 1},
                        "point_type": "Monster",
                        "children": [{"x": 0, "y": 2}],
                    },
                    {
                        "coord": {"x": 1, "y": 1},
                        "point_type": "Monster",
                        "children": [{"x": 1, "y": 2}],
                    },
                    {
                        "coord": {"x": 0, "y": 2},
                        "point_type": left_downstream,
                    },
                    {
                        "coord": {"x": 1, "y": 2},
                        "point_type": right_downstream,
                    },
                ],
            },
        }

    map_base = map_observation(
        hp=52,
        left_downstream="RestSite",
        right_downstream="Elite",
    )
    map_hp = map_observation(
        hp=14,
        left_downstream="RestSite",
        right_downstream="Elite",
    )
    map_topology = map_observation(
        hp=52,
        left_downstream="Elite",
        right_downstream="RestSite",
    )

    rest_options = (
        {
            "option_id": "REST",
            "option_type": "Rest",
            "heal_amount": 24,
            "is_enabled": True,
        },
        {
            "option_id": "SMITH",
            "option_type": "Smith",
            "is_enabled": True,
        },
    )
    rest_actions: tuple[Mapping[str, Any], ...] = tuple(
        {
            "kind": "choose_rest_site_option",
            "model_action_kind": "rest_site",
            "model_action_variant": str(option["option_type"]).lower(),
            "option": option,
        }
        for option in rest_options
    )

    def rest_observation(hp: int) -> dict[str, Any]:
        return {
            "phase": "rest_site",
            "decision_domain": "build",
            **_base_run_player(deck=attack_deck, hp=hp),
            "rest_site": {"visible": True, "options": list(rest_options)},
        }

    shop_items: tuple[dict[str, Any], ...] = (
        {
            "item_id": "SHOP.CARD",
            "item_kind": "card",
            "price": 75,
            "is_affordable": True,
            "is_stocked": True,
            "card": _card("CARD.SHOP.SKILL", card_type="Skill", cost=1),
        },
        {
            "item_id": "SHOP.RELIC",
            "item_kind": "relic",
            "price": 150,
            "is_affordable": True,
            "is_stocked": True,
            "relic": {"id": "RELIC.SHOP.TEST", "rarity": "Rare"},
        },
    )
    shop_actions: tuple[Mapping[str, Any], ...] = (
        *(
            {
                "kind": "shop_purchase",
                "model_action_kind": "shop",
                "model_action_variant": "buy",
                "item": item,
            }
            for item in shop_items
        ),
        {
            "kind": "shop_skip",
            "model_action_kind": "shop",
            "model_action_variant": "leave",
        },
    )

    def shop_observation(*, gold: int, deck: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        items = [
            {
                **item,
                "is_affordable": bool(gold >= int(item["price"])),
            }
            for item in shop_items
        ]
        return {
            "phase": "shop",
            "decision_domain": "build",
            **_base_run_player(deck=deck, gold=gold),
            "shop": {"visible": True, "gold": gold, "items": items},
        }

    return (
        MacroSensitivityCase(
            name="card_reward.deck_composition",
            surface="card_reward",
            changed_facts=("player.deck",),
            reference_observation=reward_base,
            comparison_observation=reward_deck,
            legal_actions=reward_actions,
        ),
        MacroSensitivityCase(
            name="card_reward.run_progress",
            surface="card_reward",
            changed_facts=("run.act", "run.floor"),
            reference_observation=reward_base,
            comparison_observation=reward_floor,
            legal_actions=reward_actions,
        ),
        MacroSensitivityCase(
            name="map.player_hp",
            surface="map",
            changed_facts=("player.hp",),
            reference_observation=map_base,
            comparison_observation=map_hp,
            legal_actions=map_actions,
        ),
        MacroSensitivityCase(
            name="map.visible_downstream_topology",
            surface="map",
            changed_facts=("map.points[].point_type",),
            reference_observation=map_base,
            comparison_observation=map_topology,
            legal_actions=map_actions,
        ),
        MacroSensitivityCase(
            name="rest.player_hp",
            surface="rest",
            changed_facts=("player.hp",),
            reference_observation=rest_observation(12),
            comparison_observation=rest_observation(68),
            legal_actions=rest_actions,
        ),
        MacroSensitivityCase(
            name="shop.player_gold",
            surface="shop",
            changed_facts=("player.gold", "shop.gold"),
            reference_observation=shop_observation(gold=160, deck=attack_deck),
            comparison_observation=shop_observation(gold=500, deck=attack_deck),
            legal_actions=shop_actions,
        ),
        MacroSensitivityCase(
            name="shop.deck_composition",
            surface="shop",
            changed_facts=("player.deck",),
            reference_observation=shop_observation(gold=220, deck=attack_deck),
            comparison_observation=shop_observation(gold=220, deck=skill_deck),
            legal_actions=shop_actions,
        ),
    )


def _tensor_fields_equal(
    left: object,
    right: object,
    fields: Sequence[str],
) -> bool:
    return all(torch.equal(cast(Tensor, getattr(left, field)), cast(Tensor, getattr(right, field))) for field in fields)


def _finite_candidate_logits(output: RecurrentCandidateOutput) -> Tensor:
    logits = output.policy_logits[0]
    mask = output.action_mask[0].bool()
    return logits[mask].float()


def _optional_candidate_values(
    output: RecurrentCandidateOutput,
    field: str,
) -> list[float] | None:
    value = getattr(output, field)
    if value is None:
        return None
    tensor = cast(Tensor, value)[0]
    mask = output.action_mask[0].bool()
    return [float(item) for item in tensor[mask].float().cpu().tolist()]


def evaluate_macro_sensitivity(
    model: RecurrentCandidateModel,
    encoder: GroundedObservationEncoder,
    *,
    cases: Sequence[MacroSensitivityCase] | None = None,
) -> dict[str, Any]:
    """Run fixed zero-memory pairs against one frozen policy."""

    selected_cases = tuple(cases or fixed_macro_sensitivity_cases())
    if not selected_cases:
        raise ValueError("macro sensitivity evaluation requires at least one case")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - maintained model has parameters
        raise ValueError("macro sensitivity model has no parameters") from exc

    previous_training = model.training
    case_results: list[dict[str, Any]] = []
    model.eval()
    try:
        with torch.no_grad():
            for case in selected_cases:
                reference = encoder.encode(
                    case.reference_observation,
                    case.legal_actions,
                    device=device,
                )
                comparison = encoder.encode(
                    case.comparison_observation,
                    case.legal_actions,
                    device=device,
                )
                reference_fingerprints = [
                    semantic_action_fingerprint(group.prototype) for group in reference.semantic_groups
                ]
                comparison_fingerprints = [
                    semantic_action_fingerprint(group.prototype) for group in comparison.semantic_groups
                ]
                if reference_fingerprints != comparison_fingerprints:
                    raise ValueError(f"macro sensitivity case {case.name!r} changed candidate semantics")
                if not _tensor_fields_equal(
                    reference.batch.candidates,
                    comparison.batch.candidates,
                    _CANDIDATE_TENSOR_FIELDS,
                ):
                    raise ValueError(f"macro sensitivity case {case.name!r} changed candidate tensors")
                world_changed = not _tensor_fields_equal(
                    reference.batch.world,
                    comparison.batch.world,
                    _WORLD_TENSOR_FIELDS,
                )
                if not world_changed:
                    raise ValueError(f"macro sensitivity case {case.name!r} did not change encoded world facts")

                reference_output = model(reference.batch)
                comparison_output = model(comparison.batch)
                reference_mask = reference_output.action_mask[0].bool()
                comparison_mask = comparison_output.action_mask[0].bool()
                if not torch.equal(reference_mask, comparison_mask):
                    raise ValueError(f"macro sensitivity case {case.name!r} changed the legality mask")
                reference_probabilities = reference_output.policy_probabilities()[0][reference_mask].float()
                comparison_probabilities = comparison_output.policy_probabilities()[0][comparison_mask].float()
                reference_greedy = int(
                    reference_output.greedy_action_indices()[0].item()
                )
                comparison_greedy = int(
                    comparison_output.greedy_action_indices()[0].item()
                )
                probability_delta = (comparison_probabilities - reference_probabilities).abs()
                logit_delta = (
                    _finite_candidate_logits(comparison_output) - _finite_candidate_logits(reference_output)
                ).abs()
                value_deltas = {
                    field: abs(
                        float(cast(Tensor, getattr(comparison_output, field))[0])
                        - float(cast(Tensor, getattr(reference_output, field))[0])
                    )
                    for field in _OUTPUT_VALUE_FIELDS
                }
                world_feature_delta: float | None = None
                if reference.batch.world.features.shape == comparison.batch.world.features.shape:
                    world_feature_delta = float(
                        (comparison.batch.world.features - reference.batch.world.features).abs().max().cpu()
                    )
                state_embedding_delta = float(
                    (comparison_output.state_embedding - reference_output.state_embedding).norm(p=2).cpu()
                )
                candidate_embedding_delta = float(
                    (comparison_output.candidate_embeddings - reference_output.candidate_embeddings).norm(p=2).cpu()
                )
                case_results.append(
                    {
                        "name": case.name,
                        "surface": case.surface,
                        "changed_facts": list(case.changed_facts),
                        "candidate_count": int(reference_mask.sum().item()),
                        "candidate_fingerprints": reference_fingerprints,
                        "encoded_world_changed": True,
                        "encoded_candidate_tensors_equal": True,
                        "world_feature_max_abs_delta": world_feature_delta,
                        "state_embedding_l2_delta": state_embedding_delta,
                        "candidate_embedding_l2_delta": candidate_embedding_delta,
                        "reference_probabilities": [float(value) for value in reference_probabilities.cpu().tolist()],
                        "comparison_probabilities": [float(value) for value in comparison_probabilities.cpu().tolist()],
                        "policy_total_variation": float(0.5 * probability_delta.sum().cpu()),
                        "policy_max_abs_probability_delta": float(probability_delta.max().cpu()),
                        "policy_max_abs_logit_delta": float(logit_delta.max().cpu()),
                        "top_candidate_changed": bool(
                            reference_greedy != comparison_greedy
                        ),
                        "value_abs_deltas": value_deltas,
                        "reference_transaction_q_values": _optional_candidate_values(
                            reference_output,
                            "transaction_q_values",
                        ),
                        "comparison_transaction_q_values": _optional_candidate_values(
                            comparison_output,
                            "transaction_q_values",
                        ),
                    }
                )
    finally:
        model.train(previous_training)

    by_surface: dict[str, Any] = {}
    for surface in MACRO_SURFACES:
        members = [item for item in case_results if item["surface"] == surface]
        variations = [float(item["policy_total_variation"]) for item in members]
        max_deltas = [float(item["policy_max_abs_probability_delta"]) for item in members]
        by_surface[surface] = {
            "case_count": len(members),
            "mean_policy_total_variation": _mean(variations),
            "maximum_policy_total_variation": max(variations) if variations else None,
            "mean_policy_max_abs_probability_delta": _mean(max_deltas),
            "maximum_policy_max_abs_probability_delta": (max(max_deltas) if max_deltas else None),
            "top_candidate_change_count": sum(bool(item["top_candidate_changed"]) for item in members),
        }
    return {
        "schema_version": MACRO_SENSITIVITY_SCHEMA,
        "suite_version": FIXED_MACRO_PROBE_SUITE_VERSION,
        "diagnostic_only": True,
        "training_samples_emitted": 0,
        "expected_action_labels": False,
        "recurrent_state_contract": "fresh_zero_state_per_pair",
        "case_count": len(case_results),
        "surfaces": by_surface,
        "cases": case_results,
    }


def _diagnostic_model_initialization_config(
    payload: Mapping[str, Any],
) -> TrainingConfig:
    """Interpret only reviewed config-default migrations for diagnostics.

    Frozen sensitivity loads network parameters through the same guarded
    model-initialization path used by training. A v18/v6 source predates the
    replay-sampling ``macro_sample_fraction`` field. A v19/v7 source predates
    v8 runtime schedules and scalar optimization/replay controls. V6--v8 all
    predate v9 environment-step liveness-probe milestones. None of those
    missing fields shape model tensors. Reviewed disabled defaults are filled
    explicitly before the version is advanced.

    This is explicitly not exact resume and does not relax any checkpoint,
    tensor, or grounding-encoding validation.
    """

    source_version = payload.get("version")
    if source_version == CONFIG_VERSION:
        return training_config_from_mapping(payload)
    if source_version not in _REVIEWED_DIAGNOSTIC_CONFIG_VERSIONS:
        raise ValueError(
            "macro checkpoint evaluation has no reviewed config migration "
            f"from {source_version!r} to {CONFIG_VERSION!r}"
        )
    migrated = deepcopy(dict(payload))
    raw_rollout = migrated.get("rollout")
    if not isinstance(raw_rollout, Mapping):
        raise ValueError(
            "reviewed macro checkpoint config migration requires a rollout table"
        )
    rollout = dict(raw_rollout)
    if "deterministic_probe_environment_steps" in rollout:
        raise ValueError(
            f"{source_version} macro checkpoint config unexpectedly contains "
            "deterministic_probe_environment_steps"
        )
    rollout["deterministic_probe_environment_steps"] = []
    migrated["rollout"] = rollout
    if source_version == _V6_CONFIG_VERSION:
        raw_episodic = migrated.get("episodic_learning")
        if not isinstance(raw_episodic, Mapping):
            raise ValueError(
                "reviewed macro checkpoint config migration requires an "
                "episodic_learning table"
            )
        episodic = dict(raw_episodic)
        if "macro_sample_fraction" in episodic:
            raise ValueError(
                "v6 macro checkpoint config unexpectedly contains "
                "macro_sample_fraction"
            )
        episodic["macro_sample_fraction"] = 0.0
        migrated["episodic_learning"] = episodic
    migrated["version"] = CONFIG_VERSION
    return training_config_from_mapping(migrated)


def evaluate_checkpoint_macro_sensitivity(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load only ``network.pt`` and run the fixed macro probe suite."""

    root = Path(checkpoint).expanduser().resolve()
    initial_validation = validate_resume_checkpoint(root)
    config_payload = initial_validation.metadata.get("training_config")
    if not isinstance(config_payload, dict):
        raise ValueError("checkpoint metadata has no training_config object")
    source_config_version = config_payload.get("version")
    config = _diagnostic_model_initialization_config(config_payload)
    validated = preflight_model_initialization(root, config=config)
    checkpoint_id = validated.manifest.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("checkpoint manifest has no checkpoint_id")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm macro evaluation requested but unavailable")
    model = RecurrentCandidateModel(
        config.model.to_model_config(),
        enable_transaction_heads=config.transaction_learning.enabled,
    ).to(resolved_device)
    state = torch.load(
        validated.root / "network.pt",
        map_location=resolved_device,
        weights_only=True,
    )
    if not isinstance(state, dict):
        raise ValueError("checkpoint network payload must be an object")
    model.load_state_dict(state, strict=True)
    encoder = GroundedObservationEncoder(config.model.to_encoding_config())
    result = evaluate_macro_sensitivity(model, encoder)
    training_state = validated.metadata.get("training_state")
    return {
        **result,
        "checkpoint": str(validated.root),
        "checkpoint_id": checkpoint_id,
        "source_training_config_version": source_config_version,
        "evaluation_training_config_version": config.version,
        "config_load_mode": "diagnostic_model_parameter_initialization",
        "training_state": training_state if isinstance(training_state, dict) else None,
        "network_source": "network.pt",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only macro telemetry for a journal and/or checkpoint",
    )
    parser.add_argument("--journal", type=Path, help="held-out trajectory JSONL")
    parser.add_argument("--checkpoint", type=Path, help="atomic checkpoint directory")
    parser.add_argument("--device", default="cpu", help="checkpoint probe device")
    parser.add_argument("--output", type=Path, help="write JSON instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.journal is None and args.checkpoint is None:
        raise SystemExit("at least one of --journal or --checkpoint is required")
    payload: dict[str, Any] = {
        "schema_version": "sts2-macro-evaluation-report-v1",
        "diagnostic_only": True,
        "training_samples_emitted": 0,
    }
    if args.journal is not None:
        journal = resolve_external_input_path(args.journal)
        payload["held_out_journal"] = read_macro_journal(journal)
    if args.checkpoint is not None:
        checkpoint = resolve_external_input_path(args.checkpoint)
        # Validation is intentionally repeated inside the evaluator so the
        # CLI never trusts an unverified network payload.
        validate_resume_checkpoint(checkpoint)
        payload["fixed_sensitivity"] = evaluate_checkpoint_macro_sensitivity(
            checkpoint,
            device=args.device,
        )
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        output = resolve_artifact_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"macro evaluation output already exists: {output}")
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = [
    "FIXED_MACRO_PROBE_SUITE_VERSION",
    "MACRO_SENSITIVITY_SCHEMA",
    "MACRO_SURFACES",
    "MACRO_TELEMETRY_SCHEMA",
    "MacroSensitivityCase",
    "classify_macro_surface",
    "evaluate_checkpoint_macro_sensitivity",
    "evaluate_macro_sensitivity",
    "fixed_macro_sensitivity_cases",
    "main",
    "read_macro_journal",
    "summarize_macro_records",
]
