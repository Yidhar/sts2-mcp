from __future__ import annotations

import json

from sts2_rl.simulator_identity import repository_root, sha256_file

PATCH_NAME = "0008-guard-unowned-shop-potion-context.patch"
BUILDER_PATH = "STS2AI/ENV/Sim/HeadlessSim/Simulation/FullRunApiStateBuilder.cs"
SHOP_PATCH_NAME = "0007-grounded-macro-observation-contract.patch"


def _locked_patch(name: str) -> str:
    root = repository_root()
    lock = json.loads((root / "third_party" / "sts2-ai.lock.json").read_text(encoding="utf-8"))
    matching = [record for record in lock["patches"] if record["path"].endswith(name)]
    assert len(matching) == 1
    patch_path = root / matching[0]["path"]
    assert matching[0]["sha256"] == sha256_file(patch_path)
    return patch_path.read_text(encoding="utf-8")


def test_shop_potions_do_not_dereference_an_unowned_potion_context() -> None:
    guard_patch = _locked_patch(PATCH_NAME)
    shop_patch = _locked_patch(SHOP_PATCH_NAME)

    assert (
        "ToApiPotionState(potionEntry.Model, item.index, canUseInCombat: false)"
        in shop_patch
    )
    assert (
        "-\t\t\tcan_throw_at_ally = potion.CanThrowAtAlly(),"
        in guard_patch
    )
    assert (
        "+\t\t\tcan_throw_at_ally = canUseInCombat && potion.CanThrowAtAlly(),"
        in guard_patch
    )
    assert guard_patch.count("CanThrowAtAlly()") == 2
    assert guard_patch.count("diff --git ") == 1
    assert BUILDER_PATH in guard_patch
