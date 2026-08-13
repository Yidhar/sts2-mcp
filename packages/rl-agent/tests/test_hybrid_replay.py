from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.joint import HybridRun, HybridRunReplay, HybridStep  # noqa: E402
from tests.test_macro_q_learner import _snapshot  # noqa: E402


def _step(domain: str, *, encounter: int | None = None, terminal: bool = False) -> HybridStep:
    return HybridStep(
        snapshot=_snapshot(),
        action_index=0,
        reward=1.0,
        discount=0.0 if terminal else 0.9,
        terminal=terminal,
        surface="combat_card" if domain == "combat" else "map",
        branch="play" if domain == "combat" else "choose",
        control_domain=domain,  # type: ignore[arg-type]
        encounter_id=encounter,
    )


def _run(run_id: str = "run") -> HybridRun:
    return HybridRun(
        run_id=run_id,
        environment_steps=12,
        steps=(
            _step("macro"),       # 0
            _step("combat", encounter=1),  # 1
            _step("combat", encounter=1),  # 2
            _step("macro"),       # 3 clears tactical memory
            _step("combat", encounter=2),  # 4
            _step("combat", encounter=2),  # 5
            _step("macro", terminal=True),  # 6
        ),
    )


def test_replay_stores_one_run_and_returns_balanced_views() -> None:
    replay = HybridRunReplay(capacity_runs=4, macro_unroll=8, combat_unroll=8, seed=4)
    run = _run()
    replay.put(run)
    pair = replay.sample_pairs(1)[0]
    assert pair.macro.run is run
    assert pair.combat.run is run
    assert all(run.steps[index].control_domain == "macro" for index in pair.macro.learn_indices)
    assert all(run.steps[index].control_domain == "combat" for index in pair.combat.learn_indices)


def test_macro_prefix_and_unroll_are_counted_in_macro_nodes() -> None:
    replay = HybridRunReplay(capacity_runs=4, macro_unroll=2, combat_unroll=8, seed=1)
    replay.put(_run())
    for _ in range(20):
        view = replay.sample_pairs(1)[0].macro
        anchor = view.learn_indices[0]
        assert view.prefix_indices == tuple(index for index in (0, 3, 6) if index < anchor)
        assert len(view.learn_indices) <= 2


def test_combat_view_never_stitches_encounters() -> None:
    replay = HybridRunReplay(capacity_runs=4, macro_unroll=2, combat_unroll=8, seed=3)
    replay.put(_run())
    for _ in range(20):
        view = replay.sample_pairs(1)[0].combat
        encounter_ids = {_run().steps[index].encounter_id for index in view.learn_indices}
        assert len(encounter_ids) == 1
        anchor = view.learn_indices[0]
        encounter = _run().steps[anchor].encounter_id
        assert view.prefix_indices == tuple(
            index
            for index in range(anchor)
            if _run().steps[index].control_domain == "macro"
            or (
                _run().steps[index].control_domain == "combat"
                and _run().steps[index].encounter_id == encounter
            )
        )


def test_replay_stops_reusing_exhausted_views_until_fresh_data_arrives() -> None:
    replay = HybridRunReplay(
        capacity_runs=4,
        macro_unroll=2,
        combat_unroll=2,
        maximum_view_reuse=1,
        seed=7,
    )
    replay.put(_run())
    # Run-first sampling consumes at most one view pair from this run per
    # update. Three macro anchors make it the smaller corpus.
    assert replay.sample_pairs(3)
    assert replay.sample_pairs(3)
    assert replay.sample_pairs(3)
    assert replay.sample_pairs(1) == ()
    assert replay.metrics()["maximum_view_reuse"] == 1


def test_replay_state_roundtrip_preserves_freshness_budget() -> None:
    replay = HybridRunReplay(
        capacity_runs=4,
        macro_unroll=2,
        combat_unroll=2,
        maximum_view_reuse=3,
        seed=9,
    )
    replay.put(_run())
    replay.sample_pairs(2)
    restored = HybridRunReplay(
        capacity_runs=4,
        macro_unroll=2,
        combat_unroll=2,
        maximum_view_reuse=3,
        seed=0,
    )
    restored.load_state_dict(replay.state_dict())
    assert restored.metrics() == replay.metrics()
    assert restored.state_dict()["runs"][0].environment_steps == 12
