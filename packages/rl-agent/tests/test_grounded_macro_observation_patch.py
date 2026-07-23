from __future__ import annotations

import json

from sts2_rl.simulator_identity import repository_root, sha256_file

PATCH_NAME = "0007-grounded-macro-observation-contract.patch"
DTO_PATH = "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunApiStateDtos.cs"
BUILDER_PATH = "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunApiStateBuilder.cs"


def _locked_patch() -> str:
    root = repository_root()
    lock = json.loads((root / "third_party" / "sts2-ai.lock.json").read_text(encoding="utf-8"))
    matching = [
        record for record in lock["patches"] if record["path"].endswith(PATCH_NAME)
    ]
    assert len(matching) == 1
    patch_path = root / matching[0]["path"]
    assert matching[0]["sha256"] == sha256_file(patch_path)
    return patch_path.read_text(encoding="utf-8")


def _sections(patch: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for section in patch.split("diff --git ")[1:]:
        before, after = section.splitlines()[0].split()
        assert before.startswith("a/") and after.startswith("b/")
        assert before[2:] == after[2:]
        sections[after[2:]] = section
    return sections


def _added(section: str) -> str:
    return "\n".join(
        line[1:]
        for line in section.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def test_grounded_macro_observation_patch_is_locked_and_scope_bounded() -> None:
    sections = _sections(_locked_patch())
    assert set(sections) == {BUILDER_PATH, DTO_PATH}

    combined = "\n".join(_added(section) for section in sections.values())
    # The producer exposes only current authoritative state. It must not add a
    # route scorer, future rollout, text parser, or RNG/future-state leak.
    for forbidden in (
        "route_score",
        "deck_strength",
        "Regex.",
        ".Split(",
        "Description.Parse",
        "runState.Rng",
        "FutureReward",
        "FutureShop",
    ):
        assert forbidden not in combined


def test_grounded_macro_observation_patch_exposes_lossless_visible_map_and_boss() -> None:
    sections = _sections(_locked_patch())
    dto = _added(sections[DTO_PATH])
    builder = _added(sections[BUILDER_PATH])

    assert "public string? next_boss_id { get; set; }" in dto
    assert "public sealed class FullRunApiMapCoord" in dto
    assert "public sealed class FullRunApiMapNode" in dto
    assert "public FullRunApiMapCoord? current_coord { get; set; }" in dto
    assert "public List<FullRunApiMapNode> nodes { get; set; }" in dto
    assert "public List<FullRunApiMapCoord> children { get; set; }" in dto

    assert "runState.Act.PullNextEncounter(RoomType.Boss).Id.Entry" in builder
    assert "runState?.CurrentMapCoord is { } currentCoord" in builder
    assert "nodes = snapshot.MapNodes.Select" in builder
    assert "children = node.Children.Select" in builder
    assert "col = child.Col" in builder
    assert "row = child.Row" in builder
    assert ".Take(" not in builder


def test_grounded_macro_observation_patch_uses_authoritative_nullable_heal_preview() -> None:
    sections = _sections(_locked_patch())
    dto = _added(sections[DTO_PATH])
    builder = _added(sections[BUILDER_PATH])

    assert "public decimal? heal_amount { get; set; }" in dto
    assert "option is HealRestSiteOption" in builder
    assert "HealRestSiteOption.GetHealAmount(player)" in builder
    assert "SafeValue<decimal?>" in builder
    assert "option.Description" not in "\n".join(
        line for line in builder.splitlines() if "heal_amount" in line
    )


def test_grounded_macro_observation_patch_normalizes_shop_entities_without_scores() -> None:
    sections = _sections(_locked_patch())
    dto = _added(sections[DTO_PATH])
    builder = _added(sections[BUILDER_PATH])

    for declaration in (
        "public string? type { get; set; }",
        "public int price { get; set; }",
        "public bool is_affordable { get; set; }",
        "public bool is_on_sale { get; set; }",
        "public FullRunApiCardOption? card { get; set; }",
        "public FullRunApiRelicOption? relic { get; set; }",
        "public FullRunApiPotionState? potion { get; set; }",
    ):
        assert declaration in dto

    assert "item.type = item.category;" in builder
    assert "price = entry.Cost" in builder
    assert "is_affordable = entry.EnoughGold" in builder
    assert "item.is_on_sale = cardEntry.IsOnSale" in builder
    assert "ToApiCardOption(cardEntry.CreationResult.Card, item.index)" in builder
    assert "ToApiRelicOption(relicEntry.Model, item.index)" in builder
    assert "ToApiPotionState(potionEntry.Model, item.index, canUseInCombat: false)" in builder
    assert "score" not in builder.lower()
