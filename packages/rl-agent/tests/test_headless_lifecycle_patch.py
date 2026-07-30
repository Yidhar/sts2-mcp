from __future__ import annotations

import json
from pathlib import Path

from sts2_rl.simulator_identity import repository_root, sha256_file

PATCH_NAME = "0010-headless-event-and-post-combat-lifecycle.patch"


def _locked_patch() -> tuple[Path, str]:
    root = repository_root()
    lock = json.loads(
        (root / "third_party" / "sts2-ai.lock.json").read_text(encoding="utf-8")
    )
    matching = [
        record
        for record in lock["patches"]
        if Path(record["path"]).name == PATCH_NAME
    ]
    assert len(matching) == 1
    path = root / matching[0]["path"]
    assert matching[0]["sha256"] == sha256_file(path)
    return path, path.read_text(encoding="utf-8")


def _sections(patch: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for section in patch.split("diff --git ")[1:]:
        before, after = section.splitlines()[0].split()
        assert before.startswith("a/") and after.startswith("b/")
        assert before[2:] == after[2:]
        result[after[2:]] = section
    return result


def _added_source(section: str) -> str:
    return "\n".join(
        line[1:]
        for line in section.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def test_patch_guards_all_direct_event_ngame_paths_used_by_headless() -> None:
    _, patch = _locked_patch()
    sections = _sections(patch)
    event_paths = {
        "src/Core/Models/Events/Amalgamator.cs",
        "src/Core/Models/Events/DenseVegetation.cs",
        "src/Core/Models/Events/PunchOff.cs",
    }
    assert event_paths.issubset(sections)

    amalgamator = _added_source(sections["src/Core/Models/Events/Amalgamator.cs"])
    assert amalgamator.count("NGame.Instance?.ScreenShakeTrauma") == 4

    dense_vegetation = _added_source(
        sections["src/Core/Models/Events/DenseVegetation.cs"]
    )
    assert "NGame.Instance?.ScreenRumble" in dense_vegetation
    assert "NDebugAudioManager.Instance?.Play" in dense_vegetation
    assert "NDebugAudioManager.Instance?.Stop" in dense_vegetation

    punch_off = _added_source(sections["src/Core/Models/Events/PunchOff.cs"])
    assert "NGame.Instance?.ScreenShakeTrauma" in punch_off

    for path in event_paths:
        added = _added_source(sections[path])
        assert "NGame.Instance.Screen" not in added


def test_patch_mirrors_primary_enemy_victory_and_rejects_pending_as_settled() -> None:
    _, patch = _locked_patch()
    sections = _sections(patch)
    contract_path = (
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSettlementContract.cs"
    )
    facade_path = (
        "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunSimulatorRuntimeFacade.cs"
    )
    assert {contract_path, facade_path}.issubset(sections)

    contract = _added_source(sections[contract_path])
    assert "BlocksPostCombatContinuation" in contract
    assert "return isAlive && isPrimaryEnemy;" in contract
    assert "IsCombatFollowupSettled" in contract
    assert "&& !isPostCombatPending" in contract
    assert "Fabricator can die to thorns while a secondary Zapbot remains alive" in contract

    facade = _added_source(sections[facade_path])
    assert facade.count("FullRunSettlementContract.BlocksPostCombatContinuation(") == 2
    assert "enemy.IsAlive" in facade
    assert "enemy.IsPrimaryEnemy" in facade
    assert facade.count("HasSettledCombatFollowup(signature, observed)") == 5
    assert "phase1_post_combat_pending_exhausted" in facade
    assert "Post-combat continuation remained pending" in facade
