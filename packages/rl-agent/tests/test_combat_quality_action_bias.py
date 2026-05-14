from __future__ import annotations

import numpy as np
import pytest

from muzero.combat_quality.action_bias import apply_card_block_waste_bias


def test_block_waste_gets_hard_penalty_when_progress_alternative_exists() -> None:
    bias = np.zeros(6, dtype=np.float32)
    mask = np.ones(6, dtype=np.float32)

    stats = apply_card_block_waste_bias(
        bias,
        action_mask=mask,
        card_block_waste_indices=[1],
        positive_indices=[1, 2],
        urgent_positive_indices=[3],
        end_turn_indices=[5],
        max_actions=6,
    )

    assert bias[1] == pytest.approx(-2.75)
    assert bias[2] == pytest.approx(0.20)
    assert bias[3] == pytest.approx(0.20)
    assert bias[5] == pytest.approx(0.0)
    assert stats == {
        "card_block_waste_bias_count": 1.0,
        "card_block_waste_bias_min": -2.75,
        "card_block_waste_hard_bias_applied": 1.0,
        "card_block_waste_progress_alternative": 1.0,
        "card_block_waste_progress_bonus_count": 2.0,
        "card_block_waste_progress_bonus_max": 0.20,
    }


def test_block_waste_keeps_soft_penalty_when_only_end_turn_is_alternative() -> None:
    bias = np.zeros(4, dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)

    stats = apply_card_block_waste_bias(
        bias,
        action_mask=mask,
        card_block_waste_indices=[1],
        positive_indices=[1],
        urgent_positive_indices=[],
        end_turn_indices=[3],
        max_actions=4,
    )

    assert bias[1] == pytest.approx(-1.10)
    assert bias[3] == pytest.approx(0.0)
    assert stats["card_block_waste_bias_count"] == 1.0
    assert stats["card_block_waste_hard_bias_applied"] == 0.0
    assert stats["card_block_waste_progress_bonus_count"] == 0.0


def test_masked_and_out_of_range_indices_are_ignored() -> None:
    bias = np.zeros(4, dtype=np.float32)
    mask = np.array([1, 0, 1, 1], dtype=np.float32)

    stats = apply_card_block_waste_bias(
        bias,
        action_mask=mask,
        card_block_waste_indices=[1, 2, 99, "bad"],
        positive_indices=[2, 3],
        urgent_positive_indices=[],
        end_turn_indices=[],
        max_actions=4,
    )

    assert bias.tolist() == pytest.approx([0.0, 0.0, -2.75, 0.20])
    assert stats["card_block_waste_bias_count"] == 1.0
    assert stats["card_block_waste_progress_bonus_count"] == 1.0


def test_bias_and_mask_must_be_one_dimensional() -> None:
    mask = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError, match="bias must be a 1-D array"):
        apply_card_block_waste_bias(
            np.zeros((2, 2), dtype=np.float32),
            action_mask=mask,
            card_block_waste_indices=[],
            max_actions=4,
        )

    with pytest.raises(ValueError, match="action_mask must be a 1-D array"):
        apply_card_block_waste_bias(
            np.zeros(4, dtype=np.float32),
            action_mask=np.ones((2, 2), dtype=np.float32),
            card_block_waste_indices=[],
            max_actions=4,
        )
