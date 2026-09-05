from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.joint.macro_targets import project_macro_policy  # noqa: E402


def _project(values=(0.0, 0.4, 1.0), prior=(0.6, 0.3, 0.1), max_kl=0.02, dtype=torch.float64):
    return project_macro_policy(
        action_value_targets=torch.tensor(values, dtype=dtype),
        teacher_log_probs=torch.tensor(prior, dtype=dtype).log(),
        max_kl=max_kl,
    )


def test_equal_outcomes_preserve_nonuniform_prior_and_skip_actor():
    result = _project(values=(4.0, 4.0, 4.0), max_kl=1.0)
    assert torch.equal(result.target_probabilities, result.teacher_probabilities)
    assert result.target_probabilities.tolist() == pytest.approx([0.6, 0.3, 0.1])
    assert result.value_target.item() == pytest.approx(4.0)
    assert not result.actor_enabled
    assert result.temperature is None
    assert result.kl == result.expected_improvement == 0.0


def test_single_candidate_skips_actor():
    result = _project(values=(2.0,), prior=(1.0,))
    assert result.target_probabilities.tolist() == [1.0]
    assert result.value_target.item() == 2.0
    assert not result.actor_enabled


def test_zero_budget_is_noop_not_teacher_cloning():
    result = _project(max_kl=0.0)
    assert torch.equal(result.target_probabilities, result.teacher_probabilities)
    assert not result.actor_enabled
    assert result.temperature is None


@pytest.mark.parametrize("budget", [1e-5, 0.002, 0.02, 0.2, 1.0])
def test_projection_enforces_forward_kl_and_improves_estimated_outcome(budget):
    result = _project(max_kl=budget)
    posterior, prior = result.target_probabilities, result.teacher_probabilities
    kl = (posterior * (posterior.log() - prior.log())).sum().item()
    assert posterior.sum().item() == pytest.approx(1.0)
    assert 0.0 < kl <= budget + 1e-12
    assert result.kl == pytest.approx(kl, abs=1e-12)
    assert result.actor_enabled
    assert result.expected_improvement > 0
    assert result.temperature is not None and result.temperature > 0
    assert result.target_probabilities[-1] > result.teacher_probabilities[-1]


def test_critic_targets_teacher_expectation_not_improved_posterior():
    result = _project()
    values = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float64)
    assert result.value_target.item() == pytest.approx(0.22)
    assert (result.target_probabilities * values).sum() > result.value_target


def test_units_and_translation_do_not_change_policy_projection():
    original = _project()
    shifted_scaled = _project(values=(12.0, 12.0 + 0.4 * 3, 15.0))
    assert torch.allclose(original.target_probabilities, shifted_scaled.target_probabilities, atol=1e-12)
    assert shifted_scaled.temperature == pytest.approx(3 * original.temperature)
    assert shifted_scaled.value_target.item() == pytest.approx(12.0 + 3 * original.value_target.item())


def test_candidate_permutation_is_equivariant():
    original = _project()
    permuted = _project(values=(1.0, 0.0, 0.4), prior=(0.1, 0.6, 0.3))
    assert torch.allclose(permuted.target_probabilities, original.target_probabilities[[2, 0, 1]])
    assert permuted.value_target.item() == pytest.approx(original.value_target.item())


def test_splitting_identical_candidate_outcome_keeps_aggregate_probability():
    # This is a semantic-refinement check, NOT an instruction to expand strict
    # equivalent candidates or to divide the prior by native branch counts.
    original = _project(values=(0.0, 1.0), prior=(0.8, 0.2))
    split = _project(values=(0.0, 1.0, 1.0), prior=(0.8, 0.05, 0.15))
    assert split.target_probabilities[0] == pytest.approx(original.target_probabilities[0])
    assert split.target_probabilities[1:].sum() == pytest.approx(original.target_probabilities[1])
    assert (split.target_probabilities[2] / split.target_probabilities[1]).item() == pytest.approx(3.0)


def test_unconstrained_greedy_limit_retains_prior_on_tied_maxima():
    result = _project(values=(0.0, 1.0, 1.0), prior=(0.8, 0.05, 0.15), max_kl=2.0)
    assert result.target_probabilities.tolist() == pytest.approx([0.0, 0.25, 0.75])
    assert result.temperature == 0.0
    assert result.kl == pytest.approx(-math.log(0.2))
    assert result.actor_enabled
    assert result.value_target.item() == pytest.approx(0.2)


def test_tiny_prior_is_not_clamped_and_can_gain_probability():
    result = project_macro_policy(
        action_value_targets=torch.tensor([1.0, 0.0], dtype=torch.float64),
        teacher_log_probs=torch.tensor([-1000.0, 0.0], dtype=torch.float64),
        max_kl=0.02,
    )
    assert result.teacher_probabilities[0] == 0.0  # natural exp underflow, no floor
    assert 0.0 < result.target_probabilities[0] < 0.001
    assert result.kl == pytest.approx(0.02, abs=1e-10)
    assert result.value_target.item() == 0.0
    assert result.actor_enabled
    assert result.temperature is not None and math.isfinite(result.temperature)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_projection_outputs_detached_input_dtype_and_device(dtype):
    values = torch.tensor([0.0, 1.0], dtype=dtype, requires_grad=True)
    logs = torch.tensor([0.3, 0.7], dtype=dtype, requires_grad=True).log()
    result = project_macro_policy(action_value_targets=values, teacher_log_probs=logs, max_kl=0.02)
    for tensor in (result.target_probabilities, result.teacher_probabilities, result.value_target):
        assert tensor.dtype == values.dtype
        assert tensor.device == values.device
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


@pytest.mark.parametrize("budget", [-0.1, math.nan, math.inf, True])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match="max_kl"):
        _project(max_kl=budget)


@pytest.mark.parametrize(
    ("values", "logs", "message"),
    [
        (torch.tensor([]), torch.tensor([]), "non-empty"),
        (torch.tensor([[0.0, 1.0]]), torch.tensor([0.2, 0.8]).log(), "rank-1"),
        (torch.tensor([0, 1]), torch.tensor([0.2, 0.8]).log(), "floating"),
        (torch.tensor([math.nan, 1.0]), torch.tensor([0.2, 0.8]).log(), "finite"),
        (torch.tensor([0.0, math.inf]), torch.tensor([0.2, 0.8]).log(), "finite"),
        (torch.tensor([0.0, 1.0]), torch.tensor([-math.inf, 0.0]), "finite"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0.0]), "match"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0.2, 0.8], dtype=torch.float64).log(), "match"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0.1, -1.0]), "<= 0"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0.1, 0.4]).log(), "normalized full"),
    ],
)
def test_invalid_candidate_inputs_fail_closed(values, logs, message):
    with pytest.raises(ValueError, match=message):
        project_macro_policy(action_value_targets=values, teacher_log_probs=logs, max_kl=0.02)


def test_representationally_tiny_update_skips_actor_instead_of_fake_improvement():
    result = _project(max_kl=1e-30, dtype=torch.float32)
    assert not result.actor_enabled
    assert torch.equal(result.target_probabilities, result.teacher_probabilities)
