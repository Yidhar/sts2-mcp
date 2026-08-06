from __future__ import annotations

import json
from pathlib import Path

from sts2_rl.episode_monitoring import HeldoutJournalCache, journal_descriptor, parse_heldout_journal


def _append(path: Path, *events: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _observation(
    *,
    act: int,
    floor: int,
    room_type: str,
    hp: int,
    screen: str,
    room_model_id: str | None = None,
    combat_round: int | None = None,
) -> dict[str, object]:
    return {
        "screen": screen,
        "phase": screen.lower(),
        "run": {
            "act": act,
            "floor": floor,
            "room_type": room_type,
            "room_model_id": room_model_id,
            "character_id": "IRONCLAD",
            "ascension_level": 0,
        },
        "player": {"character": "IRONCLAD", "hp": hp, "max_hp": 80, "gold": 99},
        "combat": {
            "in_progress": combat_round is not None,
            "round": combat_round,
        },
        "map": {"current_coord": [floor, 2]},
        "card_reward_selection": {"cards_count": 3, "can_skip": True},
    }


def _decision(
    episode_id: str,
    step: int,
    *,
    observation: dict[str, object],
    action: dict[str, object],
    hp_lost: int,
    revivals: int,
    outcome: str = "ongoing",
    candidates: list[dict[str, object]] | None = None,
    deadlock: dict[str, object] | None = None,
    legal_action_count: int = 2,
) -> dict[str, object]:
    rows = candidates or [action]
    return {
        "event": "decision",
        "episode_id": episode_id,
        "step_index": step,
        "reset_seed": 7001,
        "player_hp_lost": hp_lost,
        "revivals_used": revivals,
        "outcome": outcome,
        "selected_candidate_index": 0,
        "selected_action": action,
        "policy_topk": [
            {
                "candidate_index": index,
                "probability": 0.7 if index == 0 else 0.3 / max(1, len(rows) - 1),
                "multiplicity": 1,
                "action": candidate,
            }
            for index, candidate in enumerate(rows)
        ],
        "observation_summary": observation,
        "deadlock": deadlock,
        "legal_action_count": legal_action_count,
        "value": -0.25,
    }


def _journal(path: Path) -> None:
    episode_id = "heldout-seed-7001-attempt-1:sim-episode"
    header = {
        "event": "evaluation_started",
        "evaluation_gate": 75_000,
        "gate_kind": "validation",
        "actual_environment_steps": 75_123,
        "policy_version": 733,
        "policy_model_state_sha256": "abc",
        "deterministic": True,
        "evaluation_seeds": [7001],
        "journal_version": "compact-v2",
        "config": {"version": "v32", "fingerprint_sha256": "config-sha"},
        "encoding": {"version": "v4", "fingerprint_sha256": "encoding-sha"},
        "checkpoint_association": {
            "relation": "exact",
            "load_mode": "exact_resume",
            "last_committed_checkpoint": {
                "path": "checkpoints/run/periodic-step-000075000",
                "checkpoint_id": "checkpoint-id",
                "manifest_sha256": "manifest-sha",
            },
        },
        "simulator": {
            "backend": "native",
            "simulator_identity": {
                "schema_version": "sim-v2",
                "source": {"commit": "source-commit", "tree": "source-tree"},
                "binary": {"sha256": "binary-sha"},
                "managed_binary": {"sha256": "managed-binary-sha"},
            },
        },
    }
    _append(
        path,
        header,
        {"event": "evaluation_attempt_started", "evaluation_seed": 7001, "attempt": 1},
        _decision(
            episode_id,
            0,
            observation=_observation(act=1, floor=1, room_type="event", hp=80, screen="EVENT"),
            action={"action": "choose_event_option", "index": 2},
            hp_lost=0,
            revivals=0,
        ),
        _decision(
            episode_id,
            1,
            observation=_observation(act=1, floor=1, room_type="map", hp=80, screen="MAP"),
            action={
                "action": "choose_map_node",
                "index": 0,
                "col": 1,
                "row": 2,
                "map_node": {"coord": {"x": 1, "y": 2}, "index": 0, "point_type": "monster"},
            },
            candidates=[
                {
                    "action": "choose_map_node",
                    "index": 0,
                    "col": 1,
                    "row": 2,
                    "map_node": {"coord": {"x": 1, "y": 2}, "index": 0, "point_type": "monster"},
                },
                {"action": "choose_map_node", "index": 1, "coord": {"x": 4, "y": 2}},
                {
                    "action": "choose_map_node",
                    "index": 2,
                    "map_node": {"row": 2, "col": 6, "point_type": "elite"},
                },
            ],
            hp_lost=0,
            revivals=0,
        ),
        _decision(
            episode_id,
            2,
            observation=_observation(
                act=1,
                floor=2,
                room_type="monster",
                room_model_id="CULTIST",
                hp=80,
                screen="COMBAT",
                combat_round=1,
            ),
            action={"action": "play_card", "card_id": "STRIKE", "index": 0},
            hp_lost=0,
            revivals=0,
        ),
        _decision(
            episode_id,
            3,
            observation=_observation(
                act=1,
                floor=2,
                room_type="monster",
                room_model_id="CULTIST",
                hp=72,
                screen="COMBAT",
                combat_round=3,
            ),
            action={"action": "end_turn"},
            hp_lost=8,
            revivals=0,
        ),
        _decision(
            episode_id,
            4,
            observation=_observation(act=1, floor=2, room_type="reward", hp=72, screen="REWARDS"),
            action={"action": "select_card_reward", "card_id": "BASH", "card_rarity": "Uncommon"},
            candidates=[
                {"action": "select_card_reward", "card_id": "BASH", "card_rarity": "Uncommon"},
                {"action": "select_card_reward", "card_id": "ANGER", "card_rarity": "Common"},
                {"action": "skip_card_reward", "label": "Skip"},
            ],
            hp_lost=8,
            revivals=0,
        ),
        _decision(
            episode_id,
            5,
            observation=_observation(act=1, floor=2, room_type="reward", hp=72, screen="REWARDS"),
            action={"action": "claim_reward", "index": 0},
            hp_lost=8,
            revivals=0,
            legal_action_count=1,
        ),
        _decision(
            episode_id,
            6,
            observation=_observation(act=1, floor=2, room_type="reward", hp=72, screen="CARD_REWARD"),
            action={"action": "skip_card_reward", "label": "Skip"},
            candidates=[
                {"action": "select_card_reward", "card_id": "FEED", "card_rarity": "Rare"},
                {"action": "skip_card_reward", "label": "Skip"},
            ],
            hp_lost=8,
            revivals=0,
        ),
        _decision(
            episode_id,
            7,
            observation=_observation(
                act=1,
                floor=3,
                room_type="elite",
                room_model_id="GREMLIN_NOB",
                hp=72,
                screen="COMBAT",
                combat_round=1,
            ),
            action={"action": "play_card", "card_id": "DEFEND", "index": 1},
            hp_lost=8,
            revivals=0,
        ),
        _decision(
            episode_id,
            8,
            observation=_observation(
                act=1,
                floor=3,
                room_type="elite",
                room_model_id="GREMLIN_NOB",
                hp=80,
                screen="COMBAT",
                combat_round=8,
            ),
            action={"action": "end_turn"},
            hp_lost=88,
            revivals=1,
            outcome="deadlock",
            deadlock={"cycle_span": 2, "occurrences": 6, "first_step": 7, "last_step": 8},
        ),
        {
            "event": "decision_snapshot",
            "episode_id": episode_id,
            "huge_payload": "ignored",
        },
        {
            "event": "evaluation_attempt_completed",
            "episode_id": episode_id,
            "evaluation_seed": 7001,
            "attempt": 1,
            "steps": 9,
        },
    )


def test_parse_heldout_journal_projects_route_rewards_floors_and_deadlock(tmp_path: Path) -> None:
    run = tmp_path / "run"
    journal = run / "evaluation-step-000075000.jsonl"
    _journal(journal)

    parsed = parse_heldout_journal(journal, parent=run)

    assert parsed.parsed_rows == 13
    assert parsed.malformed_rows == 0
    assert parsed.provenance["game_version"] is None
    assert parsed.provenance["policy_version"] == 733
    assert parsed.provenance["checkpoint"]["name"] == "periodic-step-000075000"
    assert parsed.provenance["simulator"]["source_commit"] == "source-commit"
    assert parsed.provenance["simulator"]["managed_binary_sha256"] == "managed-binary-sha"
    assert len(parsed.episodes) == 1
    index = parsed.episodes[0]
    assert index["outcome"] == "deadlock"
    assert index["revivals"] == 1
    assert index["player_hp_lost"] == 88
    assert index["normal_combats"] == 1
    assert index["elite_combats"] == 1
    assert index["combat_victories"] == 1
    assert index["combat_failures"] == 1

    detail = parsed.details[str(index["episode_id"])]
    assert detail["route"][0]["selected"]["column"] == 1
    assert detail["route"][0]["selected"]["row"] == 2
    assert detail["route"][0]["selected"]["coord"] == {"x": 1, "y": 2}
    assert detail["route"][0]["selected"]["map_node"]["point_type"] == "monster"
    assert detail["route"][0]["candidates"][1]["column"] == 4
    assert detail["route"][0]["candidates"][1]["row"] == 2
    assert detail["route"][0]["candidates"][2]["coord"] == {"row": 2, "col": 6}
    assert detail["route"][0]["candidates"][2]["map_node"]["point_type"] == "elite"
    assert detail["route"][0]["destination"] == {
        "act": 1,
        "floor": 2,
        "room_type": "monster",
        "room_model_id": "CULTIST",
    }
    assert detail["map_topologies"] == []
    assert detail["card_rewards"][0]["selected"]["card_id"] == "BASH"
    assert detail["card_rewards"][0]["selected"]["display_name"] == "痛击"
    assert {candidate["card_id"] for candidate in detail["card_rewards"][0]["candidates"]} == {
        "BASH",
        "ANGER",
        None,
    }
    assert (
        next(
            candidate["display_name"]
            for candidate in detail["card_rewards"][0]["candidates"]
            if candidate["card_id"] == "ANGER"
        )
        == "愤怒"
    )
    assert detail["card_rewards"][1]["skipped"] is True
    normal_floor = detail["floors"][1]
    assert normal_floor["room_type"] == "monster"
    assert normal_floor["hp_loss_delta"] == 8
    assert normal_floor["revivals_delta"] == 0
    assert normal_floor["combat_result"] == "victory"
    claim = next(item for item in normal_floor["macro_decisions"] if item["kind"] == "claim_reward")
    assert claim["forced"] is True
    assert claim["automatic"] is True
    assert claim["automatic_reason"] == "only_legal_action"
    elite_floor = detail["floors"][2]
    assert elite_floor["hp_loss_delta"] == 80
    assert elite_floor["revivals_delta"] == 1
    assert elite_floor["combat_result"] == "failure"
    assert detail["termination_reason"] == "deadlock_cycle"
    assert detail["anomalies"][0]["cycle_span"] == 2
    assert [item["kind"] for item in detail["anomalies"][0]["cycle_actions"]] == [
        "play_card",
        "end_turn",
    ]
    assert detail["final_loadout"]["coverage"]["kind"] == "unavailable"
    assert detail["snapshot_coverage"]["rich_snapshot_count"] == 1


def _card(
    card_id: str,
    *,
    quantity: int,
    upgraded: bool = False,
    enchantments: list[dict[str, object]] | None = None,
    afflictions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": card_id,
        "name": f"{card_id}.title" + ("+" if upgraded else ""),
        "type": "Attack",
        "rarity": "Basic",
        "cost": 1,
        "star_cost": -1,
        "is_upgraded": upgraded,
        "quantity": quantity,
        "enchantments": enchantments or [],
        "afflictions": afflictions or [],
        "transport_only_secret": "must-not-leak",
    }


def _rich_observation(
    *,
    act: int,
    floor: int,
    hp: int,
    gold: int,
    deck: list[dict[str, object]],
    nodes: list[dict[str, object]],
) -> dict[str, object]:
    observation = _observation(act=act, floor=floor, room_type="map", hp=hp, screen="MAP")
    observation["player"] = {
        "character": "IRONCLAD",
        "hp": hp,
        "max_hp": 88,
        "gold": gold,
        "deck": deck,
        "relics": [
            {
                "id": "RELIC.BURNING_BLOOD",
                "name": "BURNING_BLOOD.title",
                "rarity": "Starter",
                "status": "Normal",
                "stack_count": 1,
                "secret": "must-not-leak",
            }
        ],
        "potions": [
            {
                "potion_id": "POTION.FIRE",
                "name": "FIRE_POTION.title",
                "rarity": "Common",
                "usage": "CombatOnly",
                "index": 0,
                "secret": "must-not-leak",
            },
            {
                "potion_id": "POTION.SKILL_POTION",
                "name": "SKILL_POTION.title",
                "rarity": "Uncommon",
                "usage": "CombatOnly",
                "index": 1,
                "secret": "must-not-leak",
            },
        ],
    }
    observation["map"] = {
        "current_coord": {"x": 1, "y": floor},
        "nodes": nodes,
        "hidden_seed": "must-not-leak",
    }
    return observation


def test_rich_snapshots_project_latest_loadout_without_counting_steps_or_maps(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    journal = run / "evaluation-step-000075000.jsonl"
    _journal(journal)
    episode_id = "heldout-seed-7001-attempt-1:sim-episode"
    nimble = {
        "id": "NIMBLE",
        "class_name": "Nimble",
        "type": "enchantment",
        "amount": 2,
        "display_amount": 2,
        "status": "Normal",
        "should_glow_gold": True,
        "secret": "must-not-leak",
    }
    bound = {
        "id": "AFFLICTION.BOUND",
        "class_name": "Bound",
        "modifier_type": "affliction",
        "amount": 1,
        "has_overlay": True,
        "secret": "must-not-leak",
    }
    act_one = _rich_observation(
        act=1,
        floor=2,
        hp=72,
        gold=99,
        deck=[_card("CARD.STRIKE", quantity=2)],
        nodes=[
            {
                "coord": {"x": 0, "y": 1},
                "point_type": "monster",
                "children": [{"row": 2, "col": 1, "secret": "must-not-leak"}],
                "secret": "must-not-leak",
            }
        ],
    )
    act_two_before = _rich_observation(
        act=2,
        floor=18,
        hp=61,
        gold=140,
        deck=[_card("CARD.STRIKE", quantity=1)],
        nodes=[
            {
                "coord": {"x": 3, "y": 0},
                "point_type": "ancient",
                "children": [{"row": 1, "col": 0}, {"row": 1, "col": 3}],
            },
            {"coord": {"x": 0, "y": 1}, "point_type": "monster", "children": []},
        ],
    )
    act_two_after = _rich_observation(
        act=2,
        floor=18,
        hp=58,
        gold=121,
        deck=[
            _card("CARD.STRIKE", quantity=2),
            _card("CARD.STRIKE", quantity=3),
            _card("CARD.STRIKE", quantity=1, upgraded=True),
            _card("CARD.STRIKE", quantity=2, enchantments=[nimble]),
            _card("CARD.BASH", quantity=1, afflictions=[bound]),
        ],
        nodes=[],
    )
    _append(
        journal,
        {
            "event": "decision_snapshot",
            "record_kind": "rich_snapshot",
            "episode_id": episode_id,
            "step_index": 3,
            "snapshot_reasons": ["periodic"],
            "observation": act_one,
        },
        {
            "event": "decision_snapshot",
            "record_kind": "rich_snapshot",
            "episode_id": episode_id,
            "step_index": 8,
            "snapshot_reasons": ["episode_last"],
            "observation": act_two_before,
            "result_observation": act_two_after,
        },
    )

    parsed = parse_heldout_journal(journal, parent=run)
    detail = parsed.details[episode_id]

    # The two rich records are evidence only; the compact decision count remains 9.
    assert detail["steps"] == 9
    assert parsed.episodes[0]["steps"] == 9
    loadout = detail["final_loadout"]
    assert loadout["source"] == "decision_snapshot.result_observation.player"
    assert loadout["coverage"]["kind"] == "latest_available_rich_snapshot"
    assert loadout["coverage"]["episode_last_snapshot"] is True
    assert loadout["coverage"]["complete"] is True
    assert (loadout["hp"], loadout["max_hp"], loadout["gold"]) == (58, 88, 121)
    assert len(loadout["deck"]) == 4
    plain_strike = next(
        card
        for card in loadout["deck"]
        if card["card_id"] == "CARD.STRIKE"
        and card["is_upgraded"] is False
        and not card["enchantments"]
    )
    assert plain_strike["quantity"] == 5
    upgraded_strike = next(card for card in loadout["deck"] if card["is_upgraded"] is True)
    assert upgraded_strike["quantity"] == 1
    enchanted_strike = next(card for card in loadout["deck"] if card["enchantments"])
    assert enchanted_strike["quantity"] == 2
    assert enchanted_strike["enchantments"][0]["id"] == "NIMBLE"
    afflicted_bash = next(card for card in loadout["deck"] if card["card_id"] == "CARD.BASH")
    assert afflicted_bash["afflictions"][0]["id"] == "AFFLICTION.BOUND"
    assert loadout["relics"][0]["relic_id"] == "RELIC.BURNING_BLOOD"
    assert loadout["relics"][0]["display_name"] == "燃烧之血"
    assert loadout["potions"][0]["potion_id"] == "POTION.FIRE"
    assert loadout["potions"][1]["potion_id"] == "POTION.SKILL_POTION"
    assert loadout["potions"][1]["display_name"] == "技能药水"
    assert "secret" not in loadout["relics"][0]
    assert "transport_only_secret" not in plain_strike

    # Maps are reconstructed from seed + recorded actions by the explicit
    # replay endpoint.  Snapshot map payloads are intentionally ignored, so
    # old v4 journals and current journals share one contract.
    assert detail["map_topologies"] == []
    assert detail["snapshot_coverage"] == {
        "source": "decision_snapshot",
        "coverage": "bounded_rich_snapshot_projection",
        "rich_snapshot_count": 3,
    }


def test_descriptor_is_first_line_only_and_cache_invalidates_by_stat(tmp_path: Path) -> None:
    run = tmp_path / "run"
    journal = run / "evaluation-step-000075000.jsonl"
    _journal(journal)
    descriptor = journal_descriptor(
        journal,
        parent=run,
        journal_kind="evaluation",
        gate=75_000,
        complete=True,
    )
    assert descriptor is not None
    assert descriptor["episode_count"] == 1
    assert descriptor["complete"] is True

    cache = HeldoutJournalCache(maximum_entries=1)
    first = cache.load(journal, parent=run)
    second = cache.load(journal, parent=run)
    assert second is first
    _append(journal, {"event": "ignored_tail"})
    third = cache.load(journal, parent=run)
    assert third is not first


def test_parser_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    run = tmp_path / "run"
    outside = tmp_path / "outside.jsonl"
    _journal(outside)
    try:
        parse_heldout_journal(outside, parent=run)
    except ValueError as exc:
        assert "safe regular child" in str(exc)
    else:
        raise AssertionError("outside journal unexpectedly accepted")

    run.mkdir()
    link = run / "evaluation-step-000075000.jsonl"
    try:
        link.symlink_to(outside)
    except OSError:
        return
    try:
        parse_heldout_journal(link, parent=run)
    except ValueError as exc:
        assert "safe regular child" in str(exc)
    else:
        raise AssertionError("symlink journal unexpectedly accepted")
