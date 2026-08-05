from __future__ import annotations

from sts2_rl.entity_localization import localized_entity_name


def test_canonical_game_data_titles_cover_dashboard_entity_families() -> None:
    assert localized_entity_name("ANGER", kind="card") == "愤怒"
    assert localized_entity_name("CARD.ARMAMENTS", kind="card") == "武装"
    assert localized_entity_name("CARD.BARRICADE.title+", kind="card") == "壁垒"
    assert localized_entity_name("RELIC.BURNING_BLOOD", kind="relic") == "燃烧之血"
    assert localized_entity_name("SKILL_POTION", kind="potion") == "技能药水"


def test_unknown_entity_id_fails_open_to_original_identifier() -> None:
    assert localized_entity_name("CARD.FUTURE_UNKNOWN", kind="card") == "CARD.FUTURE_UNKNOWN"
    assert localized_entity_name("FUTURE_UNKNOWN", kind="relic") == "FUTURE_UNKNOWN"
    assert localized_entity_name(None, kind="potion", fallback="未记录") == "未记录"
