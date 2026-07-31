from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _shadow_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "validate_semantics_shadow.py"
    spec = importlib.util.spec_from_file_location("validate_semantics_shadow", path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib invariant
        raise RuntimeError("could not load semantic shadow command")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event_option(index: int, option_id: str) -> dict[str, object]:
    return {
        "action": "choose_event_option",
        "kind": "choose_event_option",
        "model_action_kind": "event_option",
        "action_index": index,
        "index": index,
        "option": {
            "index": index,
            "text_key": option_id,
            "is_proceed": False,
            "is_locked": False,
        },
    }


def _event_observation(page: str) -> dict[str, object]:
    return {
        "phase": "event",
        "screen": "EVENT",
        "decision_domain": "build",
        "run": {
            "act": 1,
            "floor": 9,
            "room_type": "event",
            "room_model_id": "LINGER9",
        },
        "event": {
            "event_id": "LINGER9",
            "description_key": page,
            "options": [],
        },
        "player": {
            "hp": 40,
            "max_hp": 80,
            "gold": 99,
            "deck": [],
            "relics": [],
            "potions": [],
        },
    }


def _write_journal(
    path: Path,
    *,
    semantic_candidate_count: int,
) -> None:
    actions = (
        _event_option(0, "LINGER9.option.LOOP"),
        _event_option(1, "LINGER9.option.EXIT"),
    )
    record = {
        "event": "decision_snapshot",
        "record_kind": "rich_snapshot",
        "semantic_candidate_count": semantic_candidate_count,
        "observation": _event_observation("LINGER9.page.MAIN"),
        "legal_actions": actions,
        "result_observation": _event_observation("DEATH_WARNING.page.WARNING"),
        "result_legal_actions": (
            {
                "action": "confirm",
                "kind": "confirm",
                "model_action_kind": "confirm",
                "action_index": 0,
            },
        ),
        "deadlock": None,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_shadow_audit_is_read_only_and_accepts_real_shape_records(
    tmp_path: Path,
) -> None:
    module = _shadow_module()
    journal = tmp_path / "trajectory.jsonl"
    _write_journal(journal, semantic_candidate_count=2)
    before = journal.read_bytes()

    report = module.audit_trajectory_semantics((journal,))

    assert report["status"] == "passed"
    assert report["read_only"] is True
    assert report["training_authority"] is False
    assert report["counts"]["rich_decision_snapshots"] == 1
    assert report["counts"]["semantic_candidates"] == 2
    assert report["surfaces"] == {"event": 1}
    assert report["errors"] == []
    assert journal.read_bytes() == before


def test_shadow_audit_fails_closed_on_candidate_contract_drift(
    tmp_path: Path,
) -> None:
    module = _shadow_module()
    journal = tmp_path / "trajectory.jsonl"
    _write_journal(journal, semantic_candidate_count=99)

    report = module.audit_trajectory_semantics((journal,))

    assert report["status"] == "failed"
    assert len(report["errors"]) == 1
    assert report["errors"][0]["stage"] == "semantic_kernel"
    assert "candidate count differs" in report["errors"][0]["message"]
