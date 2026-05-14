from __future__ import annotations

from muzero.strategy import action_features as af


def test_semantic_family_falls_back_to_action_id_and_kind() -> None:
    assert af.semantic_family({"action_id": "end_turn"}) == "end_turn"
    assert af.semantic_family({"kind": "play_card"}) == "play_card"
    assert af.semantic_family({"action_id": "play_card:3"}) == "play_card"
    assert af.semantic_family({"action_id": "use_potion:0"}) == "use_potion"
    assert af.semantic_family({"semantic": {"family": " MAP "}, "kind": "play_card"}) == "map"


def test_action_source_prefers_nested_card_payload() -> None:
    card = {"id": "strike", "cost": 1, "type": "Attack"}
    action = {"kind": "play_card", "card": card, "cost": 9}
    assert af.action_source(action) is card
    assert af.action_source({"kind": "map"}) == {"kind": "map"}
    assert af.action_source(None) == {}


def test_action_metric_reads_typed_and_compact_fields() -> None:
    action = {
        "semantic": {
            "card_effect_profile": {
                "total_damage": 7,
                "typed_gain_energy_amount": 2,
            }
        },
        "card": {
            "card_effect_profile": {
                "total_block": 5,
                "typed_draw_amount": 1,
            }
        },
    }
    assert af.action_metric(action, "damage") == 7.0
    assert af.action_metric(action, "block") == 5.0
    assert af.action_metric(action, "energy") == 2.0
    assert af.action_metric(action, "draw") == 1.0


def test_zero_cost_excludes_x_cost() -> None:
    assert af.is_zero_cost_action({"card": {"cost": 0}}) is True
    assert af.is_zero_cost_action({"card_cost": "0"}) is True
    assert af.is_zero_cost_action({"card": {"cost": "X"}}) is False
    assert af.is_zero_cost_action({"card": {"cost": 1}}) is False


def test_positive_combat_action_uses_type_roles_metrics_and_facing_change() -> None:
    assert af.is_positive_combat_action({"action_id": "end_turn"}) is False
    assert af.is_positive_combat_action({"kind": "discard_potion"}) is False
    assert af.is_positive_combat_action({"kind": "play_card", "card": {"type": "Status"}}) is False
    assert af.is_positive_combat_action({"kind": "play_card", "card": {"type": "Attack"}}) is True
    assert af.is_positive_combat_action({"kind": "play_card", "semantic": {"roles": ["draw"]}}) is True
    assert af.is_positive_combat_action({"kind": "play_card", "damage": 3}) is True
    assert (
        af.is_positive_combat_action(
            {"kind": "play_card", "card": {"type": "Special"}},
            is_facing_change_action=lambda _action: True,
        )
        is True
    )


def test_exhaust_ethereal_retain_flags_preserve_legacy_fallbacks() -> None:
    assert af.is_exhausting_action({"semantic": {"roles": ["exhaust"]}}) is True
    assert af.is_exhausting_action({"card": {"will_exhaust": True}}) is True
    assert af.is_ethereal_action({"semantic": {"roles": ["ethereal"]}}) is True
    assert af.is_ethereal_action({"description": "??"}) is True
    assert af.is_ethereal_action({"description": "虚无"}) is True
    assert af.is_retain_action({"semantic": {"roles": ["retain"]}}) is True
    assert af.is_retain_action({"description": "??"}) is True
    assert af.is_retain_action({"description": "保留"}) is True


def test_immediate_impact_rewards_progress_and_penalizes_hp_loss() -> None:
    attack = {"damage": 8, "block": 4, "draw": 1, "energy_gain": 1}
    costly = {"damage": 8, "hp_loss": 6}
    assert af.action_immediate_impact(attack) > af.action_immediate_impact(costly)
