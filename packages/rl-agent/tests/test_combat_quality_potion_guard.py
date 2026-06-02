from muzero.combat_quality.potion_guard import (
    boss_race_potion_traits,
    boss_zero_energy_block_potion_escape,
    boss_zero_energy_liquid_escape,
    lagavulin_setup_liquid_escape,
    lagavulin_setup_window_for_potion,
    potion_identity_text_for_guard,
    raw_potion_payload_for_guard,
)
from sts2_env.aux_targets import compute_trait_targets
from sts2_env.potion_profiles import get_potion_profile, reload_registry
from sts2_env.potion_timing import _player_hp_triplet_from_raw


def test_shared_potion_timing_hp_ratio_never_treats_max_hp_one_as_full_health():
    assert _player_hp_triplet_from_raw({"player": {"hp": 50, "max_hp": 1}})[2] == 0.0
    assert _player_hp_triplet_from_raw({"player": {"hp": 1, "max_hp": 1}})[2] == 0.0
    assert _player_hp_triplet_from_raw({"player": {"currentHealth": 40, "maxHealth": 80}})[2] == 0.5


def test_aux_trait_potion_timing_handles_suspicious_max_hp_one():
    obs = {
        "player": {
            "hp": 50,
            "max_hp": 1,
            "potions": [{"title": "Fire Potion"}],
        },
        "combat": {
            "energy": 3,
            "block": 0,
            "enemies": [{"hp": 30, "max_hp": 30, "intent": {"total_damage": 0}}],
        },
    }
    action = {
        "kind": "use_potion",
        "action_id": "use_potion:0:enemy:0",
        "potion": {"title": "Fire Potion"},
        "source": {"damage": 20},
    }

    targets = compute_trait_targets(obs, action, legal_actions_before=[action])

    assert targets.shape[0] > 1
    assert 0.0 <= float(targets[1]) <= 1.0


def test_boss_race_potion_traits_strength_damage_and_empty_liquid():
    strength = {"potion_id": "STRENGTH_POTION", "effect_family": ["strength", "scaling"]}
    assert boss_race_potion_traits({"potion": {"title": "Strength Potion"}}, strength)["candidate"]
    assert boss_race_potion_traits({"potion": {"title": "Strength Potion"}}, strength)["strength_like"]

    damage = {"potion_id": "FIRE_POTION", "damage": 20}
    assert boss_race_potion_traits({"potion": {"title": "Fire Potion"}}, damage)["damage_like"]
    assert boss_race_potion_traits({"potion": {"title": "Fire Potion"}}, damage)["candidate"]

    empty_liquid = {"potion_id": "LIQUID_MEMORIES", "retrieve_from_discard_like": True}
    traits = boss_race_potion_traits({"potion": {"title": "Liquid Memories"}}, empty_liquid)
    assert traits["invalid_context"]
    assert not traits["candidate"]


def test_lucky_tonic_registry_marks_buffer_survival_tool():
    reload_registry()
    profile = get_potion_profile("POTION.LUCKY_TONIC")
    effect = profile["effect_profile"]

    assert effect["prevent_damage"] >= 1.0
    assert "buffer" in {str(x).lower() for x in profile["effect_family"]}
    assert "survival" in {str(x).lower() for x in profile["semantic_tags"]}
    assert "boss_survival_tool" in {str(x).lower() for x in profile["timing_tags"]}


def test_boss_race_potion_traits_lucky_tonic_is_survival_not_idle_setup():
    profile = {
        "potion_id": "POTION.LUCKY_TONIC",
        "effect_family": ["buff", "buffer", "prevent_damage"],
        "semantic_tags": ["buff", "defense", "survival", "buffer"],
        "timing_tags": ["prevent_major_loss_tool", "survival_tool", "boss_survival_tool"],
        "prevent_damage": 1.0,
        "hp_valid": True,
    }

    traits = boss_race_potion_traits(
        {"kind": "use_potion", "potion": {"id": "POTION.LUCKY_TONIC", "title": "幸运补剂"}},
        profile,
    )

    assert traits["candidate"]
    assert traits["buffer_like"]
    assert traits["survival_like"]
    assert traits["prevent_damage_like"]
    # Buffer is intentionally not treated as an idle boss setup potion; the
    # boss guard may spend it in a race/survival window, but should save it on
    # a harmless asleep/idle setup turn.
    assert not traits["setup_like"]


def test_boss_race_potion_traits_lucky_tonic_title_only_zh_is_survival():
    """Bridge fallbacks sometimes expose only a localized potion title."""

    traits = boss_race_potion_traits(
        {
            "kind": "use_potion",
            "action_id": "use_potion:0:self",
            "potion": {"title": "幸运药剂"},
        },
        {"hp_valid": True, "effect_family": [], "semantic_tags": [], "timing_tags": []},
    )

    assert traits["candidate"]
    assert traits["buffer_like"]
    assert traits["survival_like"]
    assert "幸运药剂" in traits["identity"]


def test_slot_only_potion_identity_resolves_lucky_from_raw_player_inventory():
    action = {"kind": "use_potion", "action_id": "use_potion:0:self"}
    raw_obs = {
        "player": {
            "potions": [
                {"id": "POTION.LUCKY_TONIC", "title": "幸运药剂"},
            ]
        }
    }

    payload = raw_potion_payload_for_guard(raw_obs, action)
    identity = potion_identity_text_for_guard(action, {"effect_family": [], "semantic_tags": []}, raw_obs)
    traits = boss_race_potion_traits(action, {"hp_valid": True}, raw_obs)

    assert payload["id"] == "POTION.LUCKY_TONIC"
    assert "potion.lucky_tonic" in identity
    assert "幸运药剂" in identity
    assert traits["candidate"]
    assert traits["buffer_like"]
    assert traits["survival_like"]


def test_lagavulin_setup_window_and_liquid_escape_are_narrow():
    raw_obs = {
        "combat": {
            "enemies": [
                {
                    "id": "LAGAVULIN_MATRIARCH",
                    "current_hp": 190,
                    "intent": {"total_damage": 0},
                    "powers": [{"id": "ASLEEP_POWER"}],
                }
            ]
        }
    }
    assert lagavulin_setup_window_for_potion(
        raw_obs,
        encounter_tier="boss",
        encounter_hint="Lagavulin Matriarch",
        threat_gap=0,
        current_energy=0,
        no_non_potion_alt=True,
    )
    assert lagavulin_setup_liquid_escape(
        {"potion": {"title": "Liquid Memories"}},
        {"potion_id": "LIQUID_MEMORIES", "retrieve_from_discard_like": True},
        raw_obs,
        encounter_tier="boss",
        encounter_hint="Lagavulin Matriarch",
        threat_gap=0,
        current_energy=0,
        no_non_potion_alt=True,
    )
    assert not lagavulin_setup_window_for_potion(
        raw_obs,
        encounter_tier="normal",
        encounter_hint="Lagavulin Matriarch",
        threat_gap=0,
        current_energy=0,
        no_non_potion_alt=True,
    )


def test_boss_zero_energy_liquid_escape_requires_pressure_or_critical_hp():
    profile = {"potion_id": "LIQUID_MEMORIES", "retrieve_from_discard_like": True}

    assert boss_zero_energy_liquid_escape(
        {"potion": {"title": "Liquid Memories"}},
        profile,
        encounter_tier="boss",
        hp=9,
        max_hp=30,
        hp_ratio=0.3,
        threat_gap=9,
        current_energy=0,
        no_non_potion_alt=True,
    )
    assert not boss_zero_energy_liquid_escape(
        {"potion": {"title": "Liquid Memories"}},
        profile,
        encounter_tier="boss",
        hp=25,
        max_hp=91,
        hp_ratio=25 / 91,
        threat_gap=0,
        current_energy=0,
        no_non_potion_alt=True,
    )


def test_boss_zero_energy_block_potion_escape_requires_real_block_value():
    fortifier = {"potion_id": "FORTIFIER_POTION", "block": 8, "effect_family": ["block"]}
    assert boss_zero_energy_block_potion_escape(
        {"potion": {"title": "Fortifier"}},
        fortifier,
        encounter_tier="boss",
        threat_gap=6,
        current_energy=0,
        no_non_potion_alt=True,
    )

    noop = {"potion_id": "FORTIFIER_POTION", "block": 0, "amplify_block_noop": True, "effect_family": ["block"]}
    assert not boss_zero_energy_block_potion_escape(
        {"potion": {"title": "Fortifier"}},
        noop,
        encounter_tier="boss",
        threat_gap=6,
        current_energy=0,
        no_non_potion_alt=True,
    )
