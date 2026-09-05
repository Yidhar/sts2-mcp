from __future__ import annotations

import json
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import pytest

from sts2_baseline.objective import ActFrontloadedDeathRewardCalculator, TaskRewardCalculator
from sts2_rl.macro_root_evidence import (
    TRANSPORT_IDENTITY_KEYS,
    canonical_facts,
    first_difference,
    reward_state,
)


def test_canonical_facts_preserves_every_field_and_does_not_mutate_input() -> None:
    facts = {
        "episode_id": "old-episode",
        "state_version": 31,
        "step_index": 4,
        "observation": {
            "run": {"seed": "fixed-seed", "act": 2},
            "_training": {"revivals_used": 4, "player_hp_lost": 90.5, "revival_budget": 64},
            "player": {"instance_id": "p1", "runtime_id": "runtime-p1"},
            "cards": [{"instance_id": "c2", "owner_id": "p1"}, {"instance_id": "c1", "owner_id": "p1"}],
            "future_fact": {"id": 17, "timestamp": 123, "revision": 4, "_private": True},
        },
        "legal_actions": [{"action_handle": "select:c1", "target_id": "c1"}],
    }
    original = deepcopy(facts)
    canonical = canonical_facts(facts)
    assert canonical == facts
    assert facts == original
    canonical["observation"]["cards"][0]["owner_id"] = "changed"
    assert facts == original


@pytest.mark.parametrize("key", sorted(TRANSPORT_IDENTITY_KEYS))
def test_transport_named_fields_are_preserved_at_every_level(key: str) -> None:
    facts = {
        key: "root-value",
        "observation": {
            "_training": {key: 1},
            "entities": [{"instance_id": "p1", "modifier": {key: 2}}],
            "_sim_raw": {key: "raw-value", "entity": {key: 3}},
        },
    }
    assert canonical_facts(facts) == facts
    changed = deepcopy(facts)
    changed["observation"]["_training"][key] = 4
    difference = first_difference(canonical_facts(facts), canonical_facts(changed))
    assert difference is not None
    assert difference["path"] == f'$["observation"]["_training"]["{key}"]'


def test_entity_transport_named_field_changes_remain_visible() -> None:
    expected = {"entities": [{"instance_id": "p1", "modifier": {"state_version": 1}}]}
    actual = {"entities": [{"instance_id": "p1", "modifier": {"state_version": 2}}]}
    assert first_difference(canonical_facts(expected), canonical_facts(actual)) == {
        "path": '$["entities"][0]["modifier"]["state_version"]',
        "kind": "value_mismatch",
        "expected": 1,
        "actual": 2,
    }


def test_canonical_mapping_and_tuple_are_json_safe_and_order_is_retained() -> None:
    canonical = canonical_facts(MappingProxyType({"z": ({"b": 2, "a": 1},), "a": [2, 1, 2]}))
    assert list(canonical) == ["a", "z"]
    assert json.loads(json.dumps(canonical, allow_nan=False)) == {"a": [2, 1, 2], "z": [{"a": 1, "b": 2}]}
    assert list(canonical["z"][0]) == ["a", "b"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_facts_and_comparison_inputs_fail_closed(bad: float) -> None:
    with pytest.raises(ValueError, match=r'\$\["world"\]\[0\]'):
        canonical_facts({"world": [bad]})
    # Validate the entire payload, even if an earlier field already differs.
    with pytest.raises(ValueError, match="non-finite"):
        first_difference({"a": 1, "z": bad}, {"a": 2, "z": 0.0})
    with pytest.raises(ValueError, match="non-finite"):
        first_difference({"a": 1}, {"a": 2, "z": bad})


@pytest.mark.parametrize("bad", [{1: "not a string key"}, {"field": object()}, {"field": {1, 2}}, b"bytes"])
def test_non_json_evidence_is_rejected(bad: Any) -> None:
    with pytest.raises(TypeError):
        canonical_facts(bad)
    with pytest.raises(TypeError):
        first_difference(bad, None)


def test_cycles_fail_closed_but_shared_acyclic_values_are_allowed() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cyclic"):
        canonical_facts(cycle)
    shared = {"hp": 30}
    assert canonical_facts([shared, shared]) == [{"hp": 30}, {"hp": 30}]


def test_first_difference_is_exact_and_does_not_remove_transport_itself() -> None:
    assert first_difference({"session_id": "a"}, {"session_id": "b"}) == {
        "path": '$["session_id"]',
        "kind": "value_mismatch",
        "expected": "a",
        "actual": "b",
    }
    assert first_difference({"hp": 2.0}, {"hp": 2.00000001}) is not None
    assert first_difference({"unknown": 1}, {"unknown": 1}) is None
    assert first_difference({"b": 2, "a": 1}, {"a": 1, "b": 2}) is None


def test_first_difference_preserves_entity_associations_and_training_counters() -> None:
    before = {"_training": {"revivals_used": 2}, "cards": [{"instance_id": "c1", "owner_id": "p1"}]}
    after = deepcopy(before)
    after["_training"]["revivals_used"] = 3  # type: ignore[index]
    difference = first_difference(canonical_facts(before), canonical_facts(after))
    assert difference is not None
    assert difference["path"] == '$["_training"]["revivals_used"]'
    after = {"_training": {"revivals_used": 2}, "cards": [{"instance_id": "c1", "owner_id": "p2"}]}
    difference = first_difference(canonical_facts(before), canonical_facts(after))
    assert difference is not None
    assert difference["path"] == '$["cards"][0]["owner_id"]'


def test_difference_paths_are_unambiguous_and_mapping_order_is_deterministic() -> None:
    difference = first_difference({'a.b[0]"': [1, 2]}, {'a.b[0]"': [1, 3]})
    assert difference is not None
    assert difference["path"] == '$["a.b[0]\\\""][1]'
    difference = first_difference({"z": 2, "a": 1}, {"z": 5, "a": 3})
    assert difference is not None
    assert difference["path"] == '$["a"]'


def test_missing_key_is_not_confused_with_null_and_array_length_has_index() -> None:
    assert first_difference({"value": None}, {}) == {
        "path": '$["value"]',
        "kind": "missing_key",
        "expected_present": True,
        "actual_present": False,
    }
    assert first_difference([], [None]) == {
        "path": "$[0]",
        "kind": "length_mismatch",
        "expected_length": 0,
        "actual_length": 1,
    }
    assert first_difference([1, 2], [2, 1]) == {
        "path": "$[0]", "kind": "value_mismatch", "expected": 1, "actual": 2,
    }


@pytest.mark.parametrize("left,right", [(True, 1), (False, 0), (1, 1.0), ("1", 1), ([], {})])
def test_types_are_not_coerced_even_with_float_tolerances(left: Any, right: Any) -> None:
    result = first_difference(left, right, float_atol=1.0, float_rtol=1.0)
    assert result is not None
    assert result["kind"] == "type_mismatch"


def test_explicit_policy_float_tolerance_does_not_tolerate_integer_drift() -> None:
    assert first_difference([1.0, 2.0], [1.00001, 2.00001], float_atol=0.00002) is None
    assert first_difference([100.0], [100.1], float_rtol=0.002) is None
    assert first_difference([1.0], [1.001], float_atol=0.0001) is not None
    assert first_difference([1], [2], float_atol=10.0) is not None


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
def test_invalid_float_tolerances_are_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError):
        first_difference({}, {}, float_atol=tolerance)
    with pytest.raises(ValueError):
        first_difference({}, {}, float_rtol=tolerance)


def test_boolean_tolerance_is_rejected() -> None:
    with pytest.raises(TypeError):
        first_difference({}, {}, float_atol=True)


def test_reward_state_is_non_mutating_and_includes_act_local_course_context() -> None:
    calculator = ActFrontloadedDeathRewardCalculator(maximum_episode_steps=12_000, revival_budget=64, objective="run")
    cold = reward_state(calculator)
    assert cold["episode_started"] is False
    assert calculator._episode_started is False
    calculator.begin_episode()
    calculator._current_act = 2
    calculator._act_revivals = 3
    calculator._last_total_revivals = 8
    state = reward_state(calculator)
    assert state == {
        "reward_version": "sts2-run-survival-efficiency-v9",
        "objective": "run",
        "discount": 1.0,
        "maximum_episode_steps": 12_000,
        "revival_budget": 64,
        "episode_started": True,
        "current_act": 2,
        "act_revivals": 3,
        "last_total_revivals": 8,
    }
    assert calculator._revival_penalty(1) == pytest.approx(-(0.80 / 64 + 0.20 / (4 * 5)))
    assert state["act_revivals"] == 3
    fresh = ActFrontloadedDeathRewardCalculator(maximum_episode_steps=12_000, revival_budget=64, objective="run")
    assert first_difference(state, reward_state(fresh)) is not None


def test_reward_evidence_rejects_other_reward_contracts() -> None:
    with pytest.raises(TypeError, match="reward-v9"):
        reward_state(TaskRewardCalculator("run"))  # type: ignore[arg-type]
