from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.map_reproduction import MapReplayUnavailable, SeedMapReplayCache


def _append(path: Path, *rows: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")


def _decision(episode_id: str, step: int, index: int, state_type: str) -> dict[str, object]:
    return {
        "event": "decision",
        "episode_id": episode_id,
        "step_index": step,
        "selected_action": {
            "action_index": index,
            "action_handle": f"sim:{index}:test",
        },
        "observation_summary": {"state_type": state_type},
    }


def _map_observation(act: int) -> dict[str, object]:
    return {
        "state_type": "map",
        "run": {"act": act},
        "map": {
            "current_coord": {"x": 3, "y": 0},
            "nodes": [
                {
                    "coord": {"x": 3, "y": 0},
                    "point_type": "ancient",
                    "children": [{"col": 1, "row": 1}],
                    "unreviewed": "must-not-leak",
                },
                {
                    "coord": {"x": 1, "y": 1},
                    "point_type": "monster",
                    "children": [],
                },
            ],
        },
    }


class _FakeClient:
    def __init__(self, executable: Path, calls: dict[str, Any]) -> None:
        self.executable = executable
        self.calls = calls
        self.responses = [
            {
                "episode_id": "replay-episode",
                "obs": {"state_type": "event", "run": {"act": 1}},
                "legal_actions": [{}, {}],
            },
            {
                "episode_id": "replay-episode",
                "obs": _map_observation(1),
                "legal_actions": [{}],
            },
            {
                "episode_id": "replay-episode",
                "obs": _map_observation(2),
                "legal_actions": [{}],
            },
            {
                "episode_id": "replay-episode",
                "obs": _map_observation(2),
                "legal_actions": [],
            },
        ]
        self.position = 0

    def __enter__(self) -> _FakeClient:
        self.calls["entered"] = self.calls.get("entered", 0) + 1
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def reset(self, **kwargs: object) -> dict[str, Any]:
        self.calls["reset"] = kwargs
        return self.responses[0]

    def step(
        self,
        episode_id: str,
        action_index: int | None = None,
        action_id: str | None = None,
        timeout_ms: int = 20_000,
    ) -> dict[str, Any]:
        self.calls.setdefault("steps", []).append((episode_id, action_index, action_id, timeout_ms))
        self.position += 1
        return self.responses[self.position]


def _identity(executable: Path) -> None:
    executable.write_bytes(b"apphost")
    Path(f"{executable}.identity.json").write_text(
        json.dumps(
            {
                "schema_version": "1.2.0",
                "source": {"commit": "source-commit"},
                "binary": {"sha256": "apphost-sha"},
                "managed_binary": {"sha256": "managed-sha"},
            }
        ),
        encoding="utf-8",
    )


def _detail() -> dict[str, object]:
    return {
        "seed": 7001,
        "character": "IRONCLAD",
        "ascension": 7,
        "floors": [{"act": 1}, {"act": 2}],
        "route": [],
        "provenance": {
            "simulator": {
                "binary_sha256": "apphost-sha",
                "managed_binary_sha256": "managed-sha",
            }
        },
    }


def test_seed_map_replay_uses_recorded_actions_and_memory_cache(tmp_path: Path) -> None:
    episode_id = "heldout-seed-7001-attempt-1:episode"
    journal = tmp_path / "evaluation-step-000000001.jsonl"
    _append(
        journal,
        _decision(episode_id, 0, 1, "event"),
        _decision(episode_id, 1, 0, "map"),
        _decision(episode_id, 2, 0, "map"),
        {"event": "evaluation_attempt_completed", "episode_id": episode_id},
    )
    executable = tmp_path / "HeadlessSim.exe"
    _identity(executable)
    calls: dict[str, Any] = {}

    def factory(path: Path) -> _FakeClient:
        assert path == executable
        return _FakeClient(path, calls)

    cache = SeedMapReplayCache(executable, maximum_entries=2, client_factory=factory)
    first = cache.load(journal, episode_id=episode_id, detail=_detail())
    second = cache.load(journal, episode_id=episode_id, detail=_detail())

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert calls["entered"] == 1
    assert calls["reset"] == {
        "character": "IRONCLAD",
        "seed": "7001",
        "ascension_level": 7,
        "force_fresh": True,
        "training_revival_budget": -1,
        "timeout_ms": 45_000,
    }
    assert [step[1] for step in calls["steps"]] == [1, 0, 0]
    assert [item["act"] for item in first["map_topologies"]] == [1, 2]
    assert first["map_topologies"][0]["nodes"][0] == {
        "coord": {"x": 3, "y": 0},
        "point_type": "ancient",
        "children": [{"x": 1, "y": 1}],
    }
    assert "unreviewed" not in json.dumps(first["map_topologies"])
    assert first["reproduction"]["actions_available"] == 3
    assert first["reproduction"]["actions_replayed"] == 3
    assert first["reproduction"]["writes_artifacts"] is False


def _rest_observation(
    *,
    screen: str,
    upgraded: bool = False,
    selected: bool = False,
) -> dict[str, object]:
    card = {
        "index": 14,
        "id": "CARD.SETUP_STRIKE",
        "name": "SETUP_STRIKE.title",
        "is_upgraded": upgraded,
        "type": "Attack",
        "rarity": "Common",
    }
    observation: dict[str, object] = {
        "state_type": "rest_site" if screen == "REST_SITE" else "card_selection",
        "screen": screen,
        "run": {"act": 1, "floor": 9, "room_type": "rest_site"},
        "player": {
            "hp": 49,
            "max_hp": 85,
            "gold": 51,
            "deck": [card],
            "relics": [],
            "potions": [],
        },
    }
    if screen == "CARD_SELECTION":
        observation["card_selection"] = {
            "operation_type": "upgrade",
            "selected_cards": [card] if selected else [],
        }
    return observation


class _RestTransactionClient(_FakeClient):
    def __init__(self, executable: Path, calls: dict[str, Any]) -> None:
        super().__init__(executable, calls)
        card = {
            "index": 14,
            "id": "CARD.SETUP_STRIKE",
            "name": "SETUP_STRIKE.title",
            "is_upgraded": False,
        }
        self.responses = [
            {
                "episode_id": "replay-episode",
                "obs": _rest_observation(screen="REST_SITE"),
                "legal_actions": [
                    {"action": "choose_rest_option", "option": {"id": "HEAL"}},
                    {"action": "choose_rest_option", "option": {"id": "SMITH"}},
                ],
            },
            {
                "episode_id": "replay-episode",
                "obs": _rest_observation(screen="CARD_SELECTION"),
                "legal_actions": [{"action": "select_card", "card": card}],
            },
            {
                "episode_id": "replay-episode",
                "obs": _rest_observation(screen="CARD_SELECTION", selected=True),
                "legal_actions": [{"action": "confirm_selection", "label": "Confirm"}],
            },
            {
                "episode_id": "replay-episode",
                "obs": {
                    **_rest_observation(screen="REST_SITE", upgraded=True),
                    **_map_observation(1),
                },
                "legal_actions": [],
            },
        ]


def test_seed_replay_recovers_exact_rest_transaction_outcomes(tmp_path: Path) -> None:
    episode_id = "heldout-seed-7001-attempt-1:episode"
    journal = tmp_path / "evaluation-step-000000001.jsonl"
    _append(
        journal,
        _decision(episode_id, 0, 1, "rest_site"),
        _decision(episode_id, 1, 0, "card_selection"),
        _decision(episode_id, 2, 0, "card_selection"),
        {"event": "evaluation_attempt_completed", "episode_id": episode_id},
    )
    executable = tmp_path / "HeadlessSim.exe"
    _identity(executable)
    calls: dict[str, Any] = {}

    detail = _detail()
    detail["floors"] = [{"act": 1}]
    cache = SeedMapReplayCache(
        executable,
        client_factory=lambda path: _RestTransactionClient(path, calls),
    )
    result = cache.load(journal, episode_id=episode_id, detail=detail)

    assert [fact["semantic_label"] for fact in result["macro_actions"]] == [
        "锻造升级",
        "选择升级: 预备打击",
        "完成升级: 预备打击+",
    ]
    assert result["macro_actions"][1]["entity"]["entity_id"] == "CARD.SETUP_STRIKE"
    assert result["macro_actions"][2]["effects"]["deck_count"] == 0
    assert result["macro_actions_omitted"] == 0


def test_seed_map_replay_fails_closed_on_managed_binary_mismatch(tmp_path: Path) -> None:
    episode_id = "heldout-seed-7001-attempt-1:episode"
    journal = tmp_path / "evaluation-step-000000001.jsonl"
    _append(
        journal,
        _decision(episode_id, 0, 0, "map"),
        {"event": "evaluation_attempt_completed", "episode_id": episode_id},
    )
    executable = tmp_path / "HeadlessSim.exe"
    _identity(executable)
    detail = _detail()
    provenance = detail["provenance"]
    assert isinstance(provenance, dict)
    simulator = provenance["simulator"]
    assert isinstance(simulator, dict)
    simulator["managed_binary_sha256"] = "other-managed-sha"
    cache = SeedMapReplayCache(executable)

    with pytest.raises(MapReplayUnavailable, match="托管实现"):
        cache.load(journal, episode_id=episode_id, detail=detail)
