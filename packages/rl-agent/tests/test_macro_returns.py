from __future__ import annotations

import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.joint.macro_returns import (  # noqa: E402
    fold_macro_smdp_edges,
    mixed_vtrace_lambda_targets,
    on_policy_macro_lambda_returns,
)


def _tensor(values: Any) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _case() -> dict[str, Any]:
    return {
        "rewards": _tensor([1, 2, 3, 4, 5, 6]),
        "discounts": _tensor([0.997, 1, 1, 0.997, 1, 0]),
        "values": _tensor([10, 11, 12, 13, 14, 15]),
        "target_log_probs": _tensor([math.log(0.2)] * 6),
        "behavior_log_probs": _tensor([math.log(0.2)] * 6),
        "control_domains": ("macro", "combat", "combat", "macro", "combat", "macro"),
        "trace_lambda": 0.7,
    }


@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0])
def test_on_policy_mixed_targets_equal_independently_folded_smdp_oracle(lam: float) -> None:
    case = _case()
    case["trace_lambda"] = lam
    mixed = mixed_vtrace_lambda_targets(**case)
    edges = fold_macro_smdp_edges(
        rewards=case["rewards"], discounts=case["discounts"], control_domains=case["control_domains"]
    )
    oracle = on_policy_macro_lambda_returns(edges, successor_values=_tensor([13, 15, 0]), trace_lambda=lam)
    assert edges.start_indices == (0, 3, 5)
    assert edges.stop_indices == (3, 5, 6)
    torch.testing.assert_close(mixed.value_targets[list(edges.start_indices)], oracle)
    torch.testing.assert_close(mixed.action_value_targets, mixed.value_targets)
    torch.testing.assert_close(mixed.trace_lambdas, _tensor([1, 1, lam, 1, lam, 1]))
    if lam == 0:
        torch.testing.assert_close(oracle, edges.rewards + edges.discounts * _tensor([13, 15, 0]))
    if lam == 1:
        expected = edges.rewards[0] + edges.discounts[0] * (
            edges.rewards[1] + edges.discounts[1] * edges.rewards[2]
        )
        torch.testing.assert_close(oracle[0], expected)


def test_fold_applies_the_supplied_floor_discount_exactly_once_inside_rewards() -> None:
    edges = fold_macro_smdp_edges(
        rewards=_tensor([1, 2, 3, 4]),
        discounts=_tensor([0.997, 1, 1, 0]),
        control_domains=("macro", "combat", "combat", "macro"),
    )
    torch.testing.assert_close(edges.rewards, _tensor([1 + 0.997 * 2 + 0.997 * 3, 4]))
    torch.testing.assert_close(edges.discounts, _tensor([0.997, 0]))


@pytest.mark.parametrize("lam", [0.0, 0.4, 1.0])
def test_extra_same_floor_combat_decisions_do_not_add_lambda_decay(lam: float) -> None:
    common = {"trace_lambda": lam}
    short = mixed_vtrace_lambda_targets(
        rewards=_tensor([1, 2, 4]), discounts=_tensor([1, 0.997, 0]), values=_tensor([9, 8, 7]),
        target_log_probs=_tensor([-1, -1, -1]), behavior_log_probs=_tensor([-1, -1, -1]),
        control_domains=("macro", "combat", "macro"), **common,
    )
    extended = mixed_vtrace_lambda_targets(
        rewards=_tensor([1, 0, 0, 2, 4]), discounts=_tensor([1, 1, 1, 0.997, 0]),
        values=_tensor([9, 12, -15, 8, 7]),
        target_log_probs=_tensor([-1] * 5), behavior_log_probs=_tensor([-1] * 5),
        control_domains=("macro", "combat", "combat", "combat", "macro"), **common,
    )
    torch.testing.assert_close(short.value_targets[[0, 2]], extended.value_targets[[0, 4]])


@pytest.mark.parametrize("split", [1, 2, 3, 5])
def test_reverse_chunking_is_identical_even_when_split_inside_combat(split: int) -> None:
    case = _case()
    case["target_log_probs"] = _tensor([math.log(value) for value in (0.1, 0.4, 0.05, 0.3, 0.15, 0.1)])
    full = mixed_vtrace_lambda_targets(**case)
    suffix = mixed_vtrace_lambda_targets(**{
        key: (value[split:] if key != "trace_lambda" else value) for key, value in case.items()
    })
    prefix = mixed_vtrace_lambda_targets(
        **{key: (value[:split] if key != "trace_lambda" else value) for key, value in case.items()},
        bootstrap_value=case["values"][split],
        bootstrap_domain=case["control_domains"][split],
        tail_value_target=suffix.value_targets[0],
    )
    torch.testing.assert_close(torch.cat((prefix.value_targets, suffix.value_targets)), full.value_targets)
    torch.testing.assert_close(torch.cat((prefix.action_value_targets, suffix.action_value_targets)), full.action_value_targets)


def test_intervening_combat_importance_ratios_cannot_be_dropped() -> None:
    case = _case()
    actual_probs = [0.2, 0.05, 0.1, 0.2, 0.04, 0.2]
    case["target_log_probs"] = _tensor([math.log(value) for value in actual_probs])
    corrected = mixed_vtrace_lambda_targets(**case)
    incorrect_macro_only = mixed_vtrace_lambda_targets(**{**case, "target_log_probs": case["behavior_log_probs"]})
    torch.testing.assert_close(corrected.clipped_rhos, _tensor([1, 0.25, 0.5, 1, 0.2, 1]))
    assert not torch.isclose(corrected.value_targets[0], incorrect_macro_only.value_targets[0])
    assert not torch.isclose(corrected.action_value_targets[0], incorrect_macro_only.action_value_targets[0])


def test_forced_root_target_excludes_its_ratio_but_preserves_next_macro_lambda_boundary() -> None:
    facts = {
        "rewards": _tensor([1, 4]), "discounts": _tensor([0.9, 0]), "values": _tensor([3, 5]),
        "behavior_log_probs": _tensor([math.log(0.5)] * 2),
        "control_domains": ("macro", "macro"), "trace_lambda": 0.3,
    }
    same = mixed_vtrace_lambda_targets(**facts, target_log_probs=_tensor([math.log(0.5)] * 2))
    rare = mixed_vtrace_lambda_targets(**facts, target_log_probs=_tensor([math.log(0.005), math.log(0.5)]))
    torch.testing.assert_close(rare.action_value_targets[0], same.action_value_targets[0])
    torch.testing.assert_close(rare.action_value_targets[0], _tensor(1 + 0.9 * (5 + 0.3 * (4 - 5))))
    assert not torch.isclose(rare.value_targets[0], same.value_targets[0])
    assert not torch.isclose(rare.action_value_targets[0], _tensor(1 + 0.9 * 4))
    torch.testing.assert_close(rare.action_value_targets[-1], _tensor(4.0))


def test_horizon_tail_keeps_real_bootstrap_instead_of_fabricating_a_terminal() -> None:
    case = _case()
    case["discounts"][-1] = 0.997
    with pytest.raises(ValueError, match="non-terminal tails require"):
        mixed_vtrace_lambda_targets(**case)
    actual = mixed_vtrace_lambda_targets(**case, bootstrap_value=_tensor(7.0), bootstrap_domain="combat")
    torch.testing.assert_close(actual.value_targets[-1], _tensor(6 + 0.997 * 7))
    edges = fold_macro_smdp_edges(
        rewards=case["rewards"], discounts=case["discounts"], control_domains=case["control_domains"]
    )
    oracle = on_policy_macro_lambda_returns(
        edges, successor_values=_tensor([13, 15, 7]), trace_lambda=0.7, bootstrap_domain="combat"
    )
    torch.testing.assert_close(actual.value_targets[list(edges.start_indices)], oracle)


def test_terminal_forbids_bootstrap_even_if_it_would_be_zero() -> None:
    with pytest.raises(ValueError, match="terminal sequences must not bootstrap"):
        mixed_vtrace_lambda_targets(**_case(), bootstrap_value=_tensor(0.0), bootstrap_domain="macro")


@pytest.mark.parametrize("field,bad", [("rewards", float("inf")), ("values", float("nan")),
                                     ("target_log_probs", 0.1), ("behavior_log_probs", float("nan")),
                                     ("discounts", -0.1)])
def test_nonfinite_or_invalid_inputs_fail_closed(field: str, bad: float) -> None:
    case = _case()
    case[field][0] = bad
    with pytest.raises(ValueError):
        mixed_vtrace_lambda_targets(**case)


@pytest.mark.parametrize("lam", [-0.1, 1.1, float("nan"), True])
def test_lambda_validation(lam: float) -> None:
    with pytest.raises(ValueError, match="trace_lambda"):
        mixed_vtrace_lambda_targets(**{**_case(), "trace_lambda": lam})


def test_extreme_log_ratios_are_clipped_stably_without_an_artificial_probability_floor() -> None:
    case = _case()
    case["target_log_probs"][0] = 0
    case["behavior_log_probs"][0] = -1000
    case["target_log_probs"][1] = -1000
    case["behavior_log_probs"][1] = 0
    output = mixed_vtrace_lambda_targets(**case)
    assert output.clipped_rhos[0].item() == 1
    assert output.clipped_rhos[1].item() == 0
    assert output.log_importance_ratios[0].item() == 1000
    assert torch.isfinite(output.value_targets).all()


def test_return_computation_never_retains_autograd() -> None:
    case = _case()
    for value in case.values():
        if isinstance(value, torch.Tensor):
            value.requires_grad_()
    result = mixed_vtrace_lambda_targets(**case)
    for value in (result.value_targets, result.action_value_targets, result.log_importance_ratios,
                  result.clipped_rhos, result.trace_coefficients, result.trace_lambdas):
        assert not value.requires_grad and value.grad_fn is None


def test_terminal_in_the_middle_cannot_be_stitched_into_another_episode() -> None:
    case = _case()
    case["discounts"][2] = 0
    with pytest.raises(ValueError, match="continue after a terminal"):
        mixed_vtrace_lambda_targets(**case)


def test_clipping_levels_have_valid_vtrace_order() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        mixed_vtrace_lambda_targets(**_case(), rho_clip=0.5, c_clip=1.0)


def test_folding_does_not_invent_a_macro_from_a_combat_only_suffix() -> None:
    edges = fold_macro_smdp_edges(
        rewards=_tensor([1, 2]), discounts=_tensor([1, 0]), control_domains=("combat", "combat")
    )
    assert edges.start_indices == edges.stop_indices == ()
    assert edges.rewards.numel() == edges.discounts.numel() == 0


def test_folding_rejects_output_overflow_even_if_individual_rewards_are_finite() -> None:
    maximum = torch.finfo(torch.float64).max
    with pytest.raises(FloatingPointError, match="non-finite"):
        fold_macro_smdp_edges(
            rewards=_tensor([maximum, maximum]), discounts=_tensor([1, 0]),
            control_domains=("macro", "combat"),
        )
