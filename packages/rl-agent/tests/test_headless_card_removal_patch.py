from __future__ import annotations

import json
from pathlib import Path

from sts2_rl.simulator_identity import repository_root, sha256_file

PATCH_NAME = "0011-single-use-headless-card-removal.patch"
ENTRY_PATH = "src/Core/Entities/Merchant/MerchantCardRemovalEntry.cs"


def test_card_removal_success_retires_the_domain_entry_without_ui_authority() -> None:
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
    patch_path = root / matching[0]["path"]
    assert matching[0]["sha256"] == sha256_file(patch_path)

    patch = patch_path.read_text(encoding="utf-8")
    assert patch.count("diff --git ") == 1
    assert f"a/{ENTRY_PATH} b/{ENTRY_PATH}" in patch
    assert "+\t\t\tSetUsed();" in patch
    assert "NRun.Instance?.MerchantRoom?.Inventory.OnCardRemovalUsed();" in patch

    # The authoritative domain mutation must happen before the optional UI
    # projection.  Headless simulation intentionally has no NRun instance.
    assert patch.index("+\t\t\tSetUsed();") < patch.index(
        "NRun.Instance?.MerchantRoom?.Inventory.OnCardRemovalUsed();"
    )
    assert "goldToSpend" not in "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
