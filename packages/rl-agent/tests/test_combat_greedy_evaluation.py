from __future__ import annotations

from dataclasses import replace

from sts2_rl.combat_greedy_evaluation import (
    CombatOutcome,
    PairedCombatResult,
    choose_greedy_candidate,
    combat_state_fingerprint,
    evaluate_one_step,
    summarize_pairs,
)


def _result(
    *,
    hp: int = 80,
    block: int = 0,
    enemy_hp: int = 40,
    enemy_max_hp: int = 40,
    incoming: int = 10,
    terminal: bool = False,
    reason: str | None = None,
    combat: bool = True,
) -> dict[str, object]:
    observation = {
        "state_type": "monster" if combat else "combat_rewards",
        "decision_domain": "combat" if combat else "rewards",
        "player": {"hp": hp, "max_hp": 80, "block": block},
        "combat": {
            "in_progress": combat,
            "enemies": [
                {
                    "model_id": "MONSTER.TEST",
                    "instance_id": "enemy-1",
                    "hp": enemy_hp,
                    "max_hp": enemy_max_hp,
                    "intents": [{"total_damage": incoming}],
                }
            ],
        },
        "run": {"act": 1, "floor": 1},
        "_training": {"revivals_used": 0, "player_hp_lost": 0},
    }
    actions = (
        []
        if terminal or not combat
        else [
            {"kind": "play_card", "is_enabled": True, "_sim_raw": {"action": "play_card", "index": 0}},
            {"kind": "end_turn", "is_enabled": True, "_sim_raw": {"action": "end_turn"}},
        ]
    )
    return {
        "episode_id": "episode",
        "obs": observation,
        "legal_actions": actions,
        "done": terminal,
        "truncated": False,
        "terminal_reason": reason,
    }


def _outcome(agent: str, *, won: bool, hp_loss: float = 0.0) -> CombatOutcome:
    return CombatOutcome(
        agent=agent,
        won=won,
        lost=not won,
        truncated=False,
        deadlocked=False,
        no_progress_stall=False,
        terminal_reason="combat_victory" if won else "run_defeat",
        start_hp=80.0,
        end_hp=80.0 - hp_loss,
        hp_loss=hp_loss,
        exact_hp_loss=hp_loss,
        revivals=0,
        turns=2,
        actions=5,
        elapsed_s=0.1,
        final_state_type="combat_rewards" if won else "game_over",
        action_trace=(),
        recurrent_state=None,
    )


def test_one_step_evaluator_prefers_survival_over_small_damage_gain() -> None:
    before = _result()
    safe = _result(hp=80, block=10, enemy_hp=35)
    reckless = _result(hp=60, block=0, enemy_hp=25)
    assert evaluate_one_step(before, safe).score > evaluate_one_step(before, reckless).score


def test_one_step_evaluator_terminal_ordering_is_explicit() -> None:
    before = _result()
    victory = _result(enemy_hp=0, terminal=True, reason="combat_victory", combat=False)
    defeat = _result(hp=0, terminal=True, reason="run_defeat")
    assert evaluate_one_step(before, victory).score > 100_000
    assert evaluate_one_step(before, defeat).score < -100_000


def test_choose_greedy_candidate_probes_each_enabled_legal_action() -> None:
    before = _result()
    probes: list[int] = []

    def probe(position: int) -> dict[str, object]:
        probes.append(position)
        return _result(enemy_hp=35 if position == 0 else 39)

    choice = choose_greedy_candidate(before, probe)
    assert probes == [0, 1]
    assert choice.position == 0
    assert len(choice.probes) == 2


def test_combat_fingerprint_ignores_episode_transport_identity() -> None:
    first = _result()
    second = _result()
    second["episode_id"] = "other-episode"
    assert combat_state_fingerprint(first) == combat_state_fingerprint(second)


def test_combat_fingerprint_excludes_full_run_transport_seed_only() -> None:
    first = _result()
    second = _result()
    first_obs = first["obs"]
    second_obs = second["obs"]
    assert isinstance(first_obs, dict) and isinstance(second_obs, dict)
    first_obs["run"]["seed"] = "reset-a"
    second_obs["run"]["seed"] = "reset-b"
    assert combat_state_fingerprint(first) == combat_state_fingerprint(second)
    second_obs["run"]["floor"] = 2
    assert combat_state_fingerprint(first) != combat_state_fingerprint(second)


def test_single_pair_is_smoke_only_not_a_bottleneck_diagnosis() -> None:
    pair = PairedCombatResult(
        pair_index=0,
        evaluation_seed=1001,
        act=1,
        floor=1,
        encounter_id="test",
        room_type="monster",
        initial_fingerprint="root",
        revival_budget=0,
        rl=_outcome("rl", won=True),
        greedy=_outcome("greedy", won=False),
    )
    assert summarize_pairs([pair])["diagnosis"] == "insufficient_data"


def test_summary_reports_combat_micro_underperformance_when_rl_loses_every_discordant_pair() -> None:
    pairs = [
        PairedCombatResult(
            pair_index=index,
            evaluation_seed=1001 + index * 2,
            act=1,
            floor=1,
            encounter_id="test",
            room_type="monster",
            initial_fingerprint=str(index),
            revival_budget=0,
            rl=_outcome("rl", won=False, hp_loss=80.0),
            greedy=_outcome("greedy", won=True, hp_loss=10.0),
        )
        for index in range(12)
    ]
    summary = summarize_pairs(pairs)
    assert summary["diagnosis"] == "combat_micro_underperformance"
    assert summary["paired"]["greedy_only_win"] == 12


def test_summary_is_macro_supported_when_rl_is_never_worse() -> None:
    base = PairedCombatResult(
        pair_index=0,
        evaluation_seed=1001,
        act=1,
        floor=1,
        encounter_id="test",
        room_type="monster",
        initial_fingerprint="root",
        revival_budget=0,
        rl=_outcome("rl", won=True),
        greedy=_outcome("greedy", won=False),
    )
    pairs = [replace(base, pair_index=index, evaluation_seed=1001 + index * 2) for index in range(12)]
    assert summarize_pairs(pairs)["diagnosis"] == "macro_bottleneck_supported"
