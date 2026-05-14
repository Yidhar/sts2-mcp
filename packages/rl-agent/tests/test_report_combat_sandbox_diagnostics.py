import json

from scripts.report_combat_sandbox_diagnostics import (
    encounter_id,
    filter_recent_by_step,
    is_forced_energy_left_no_playable,
    is_hard_bad_incoming_end_turn,
    priority_rows,
    render_report,
    row_step,
    selected_title,
)


def test_filter_recent_by_step_uses_max_step_window():
    rows = [{"global_step": 10}, {"global_step": 50}, {"global_step": 120}, {"no_step": True}]

    recent = filter_recent_by_step(rows, 50)

    assert {"global_step": 10} not in recent
    assert {"global_step": 50} not in recent
    assert {"global_step": 120} in recent
    assert {"no_step": True} in recent


def test_selected_title_prefers_nested_action_info():
    assert selected_title({"action_info": {"title": "全身撞击"}}) == "全身撞击"


def test_encounter_id_supports_snapshot_fallback():
    assert encounter_id({"snapshot": {"encounter_id": "ENCOUNTER.TEST"}}) == "encounter.test"


def test_priority_rows_weights_deaths_above_offenders():
    offenders = [{"encounter_id": "A"} for _ in range(20)] + [{"encounter_id": "B"}]
    deaths = [{"encounter_id": "B"} for _ in range(3)]

    rows = priority_rows(offenders, deaths, limit=2)

    assert rows[0][0] == "b"
    assert rows[0][1:] == (3, 1, 31)


def test_row_step_accepts_training_step():
    assert row_step({"training_step": 42.0}) == 42


def test_strict_end_turn_helpers_do_not_treat_leftover_energy_as_bad():
    forced = {
        "end_turn_class": "forced_end_turn",
        "player": {"energy": 4, "block": 0},
        "combat": {"incoming_damage": 0},
        "counts": {
            "legal_action_count": 1,
            "playable_cards_left": 0,
            "positive_action_count": 0,
            "urgent_action_count": 0,
        },
    }
    bad = {
        "end_turn_class": "bad_end_turn",
        "player": {"energy": 1, "block": 3},
        "combat": {"incoming_damage": 14},
        "counts": {"urgent_action_count": 2},
    }

    assert is_forced_energy_left_no_playable(forced)
    assert not is_hard_bad_incoming_end_turn(forced)
    assert is_hard_bad_incoming_end_turn(bad)


def test_forced_energy_left_allows_non_progress_legal_actions():
    forced_with_non_progress_actions = {
        "end_turn_class": "forced_end_turn",
        "player": {"energy": 3, "block": 0},
        "combat": {"incoming_damage": 0, "hand_count": 0},
        "counts": {
            "legal_action_count": 3,
            "playable_cards_left": 0,
            "positive_action_count": 0,
            "urgent_action_count": 0,
        },
        "top_legal_actions": [
            {"family": "end_turn", "title": "End Turn"},
            {"family": "use_potion", "title": "能量药水", "tags": []},
            {"family": "play_card", "title": "黏液", "tags": []},
        ],
    }

    assert is_forced_energy_left_no_playable(forced_with_non_progress_actions)


def test_render_report_includes_strict_end_turn_taxonomy(tmp_path):
    run_dir = tmp_path / "run"
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True)
    (diagnostics / "action_offenders.jsonl").write_text("", encoding="utf-8")
    (diagnostics / "end_turn_contexts.jsonl").write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {
                    "global_step": 100,
                    "encounter_id": "encounter.slimed_berserker_normal",
                    "turn": 3,
                    "end_turn_class": "forced_end_turn",
                    "player": {"hp": 76, "max_hp": 76, "block": 0, "energy": 4},
                    "combat": {"incoming_damage": 0, "hand_count": 0},
                    "counts": {
                        "legal_action_count": 1,
                        "playable_cards_left": 0,
                        "positive_action_count": 0,
                        "urgent_action_count": 0,
                    },
                    "top_legal_actions": [{"family": "end_turn", "title": "End Turn"}],
                },
                {
                    "global_step": 101,
                    "encounter_id": "encounter.corpse_slugs_normal",
                    "turn": 2,
                    "end_turn_class": "bad_end_turn",
                    "player": {"hp": 70, "max_hp": 80, "block": 3, "energy": 1},
                    "combat": {"incoming_damage": 14, "hand_count": 3},
                    "counts": {
                        "legal_action_count": 4,
                        "playable_cards_left": 3,
                        "positive_action_count": 3,
                        "urgent_action_count": 2,
                    },
                    "top_legal_actions": [
                        {"family": "play_card", "title": "防御", "tags": ["urgent_positive"]},
                        {"family": "end_turn", "title": "End Turn"},
                    ],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = render_report(run_dir, recent_window=1000, top_n=5)

    assert "## Strict EndTurn taxonomy" in report
    assert "| strict_bad_end_turn | 1 |" in report
    assert "| hard_bad_incoming_end_turn | 1 |" in report
    assert "| forced_energy_left_no_playable | 1 |" in report
    assert "encounter.corpse_slugs_normal" in report
    assert "encounter.slimed_berserker_normal" in report
