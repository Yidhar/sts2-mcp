from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("torch")

from sts2_rl.joint import HybridQLearner, HybridRun
from tests.test_hybrid_replay import _step


def test_macro_event_edge_folds_intervening_combat_facts() -> None:
    run = HybridRun(
        run_id="edge",
        steps=(
            _step("macro"),
            replace(_step("combat", encounter=1), reward=2.0, discount=0.5),
            replace(_step("macro", terminal=True), reward=3.0),
        ),
    )
    # Edge from node 0 to the next macro node owns rewards on nodes [0, 1].
    factual, discount = HybridQLearner._edge(run, 0, 2)
    assert factual == pytest.approx(1.0 + 0.9 * 2.0)
    assert discount == pytest.approx(0.9 * 0.5)


def test_one_absolute_revival_price_is_folded_into_every_domain_edge() -> None:
    run = HybridRun(
        run_id="death-price",
        steps=(
            _step("macro"),
            replace(
                _step("combat", encounter=1, revivals_delta=2),
                reward=2.0,
                discount=0.5,
            ),
            _step("macro", terminal=True),
        ),
    )
    factual, discount = HybridQLearner._edge(
        run,
        0,
        2,
        lambda_value=0.25,
    )
    assert factual == pytest.approx(1.0 + 0.9 * (2.0 - 0.5))
    assert discount == pytest.approx(0.9 * 0.5)
