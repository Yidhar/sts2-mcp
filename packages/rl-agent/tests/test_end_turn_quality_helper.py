from __future__ import annotations

from sts2_env.end_turn_quality import strict_end_turn_waste_context


def _obs(*, energy: int = 2, hp: int = 50, block: int = 0, incoming: int = 0, enemy_hp: int = 20) -> dict:
    return {
        "player": {"hp": hp, "block": block},
        "combat": {
            "energy": energy,
            "enemies": [
                {
                    "id": "e1",
                    "combat_id": "e1",
                    "hp": enemy_hp,
                    "intent": {"total_damage": incoming},
                }
            ],
        },
    }


def _end_turn() -> dict:
    return {"action_id": "end_turn", "kind": "end_turn"}


def _card(title: str, *, cost: int = 1, damage: int = 0, block: int = 0, draw: int = 0) -> dict:
    effect_preview = {}
    if damage:
        effect_preview["damage"] = damage
        effect_preview["total_damage"] = damage
    if block:
        effect_preview["block"] = block
    if draw:
        effect_preview["draw"] = draw
    return {
        "action_id": f"play:{title}",
        "kind": "play_card",
        "target": {"combat_id": "e1"},
        "card": {
            "title": title,
            "name": title,
            "cost": cost,
            "effect_preview": effect_preview,
        },
    }


def _potion(title: str, *, damage: int = 0, block: int = 0, heal: int = 0) -> dict:
    effect_preview = {}
    if damage:
        effect_preview["damage"] = damage
        effect_preview["total_damage"] = damage
    if block:
        effect_preview["block"] = block
    if heal:
        effect_preview["heal"] = heal
    return {
        "action_id": f"potion:{title}",
        "kind": "use_potion",
        "target": {"combat_id": "e1"},
        "potion": {"title": title, "effect_preview": effect_preview},
    }


def test_leftover_energy_after_all_cards_are_done_is_benign() -> None:
    """User case: hand was genuinely exhausted; unspent energy alone is OK."""

    ctx = strict_end_turn_waste_context(_obs(energy=3, incoming=18), [_end_turn()], _end_turn())

    assert ctx["selected_is_end_turn"] is True
    assert ctx["energy"] == 3.0
    assert ctx["non_end_turn_action_count"] == 0
    assert ctx["playable_card_count"] == 0
    assert ctx["urgent_positive_action_count"] == 0
    assert ctx["wasteful_end_turn_selected"] is False
    assert ctx["benign_leftover_energy"] is True


def test_leftover_energy_with_only_optional_potion_is_not_wasteful() -> None:
    """A non-urgent potion left after cards are gone should not train as bad EndTurn."""

    legal = [_potion("铁心药水", block=10), _end_turn()]
    ctx = strict_end_turn_waste_context(_obs(energy=2, incoming=0), legal, _end_turn())

    assert ctx["non_end_turn_action_count"] == 1
    assert ctx["positive_action_count"] == 1
    assert ctx["urgent_positive_action_count"] == 0
    assert ctx["wasteful_end_turn_selected"] is False


def test_pure_block_without_attack_pressure_is_not_wasteful() -> None:
    legal = [_card("防御", block=5), _end_turn()]
    ctx = strict_end_turn_waste_context(_obs(energy=2, incoming=0), legal, _end_turn())

    assert ctx["positive_action_count"] == 1
    assert ctx["urgent_positive_action_count"] == 0
    assert ctx["wasteful_end_turn_selected"] is False


def test_skipping_block_against_real_incoming_damage_is_wasteful() -> None:
    legal = [_card("防御", block=5), _end_turn()]
    ctx = strict_end_turn_waste_context(_obs(energy=2, incoming=12, block=0), legal, _end_turn())

    assert ctx["urgent_positive_action_count"] == 1
    assert ctx["urgent_positive_reasons"] == ["prevent_damage"]
    assert ctx["wasteful_end_turn_selected"] is True


def test_skipping_available_lethal_is_wasteful_even_with_no_incoming_damage() -> None:
    legal = [_card("打击", damage=6), _end_turn()]
    ctx = strict_end_turn_waste_context(_obs(energy=1, incoming=0, enemy_hp=6), legal, _end_turn())

    assert ctx["urgent_positive_action_count"] == 1
    assert ctx["urgent_positive_reasons"] == ["lethal"]
    assert ctx["wasteful_end_turn_selected"] is True
