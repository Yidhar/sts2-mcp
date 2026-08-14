from __future__ import annotations

import json
from pathlib import Path

from sts2_rl.simulator_identity import repository_root, sha256_file

PATCH_NAME = "0012-complete-game-action-after-queue-pop.patch"
ENTRY_PATH = "src/Core/GameActions/GameAction.cs"


def test_action_leaves_queue_before_completion_continuations_run() -> None:
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

    after_finished = "this.AfterFinished?.Invoke(this);"
    completion = "_completionSource.SetResult();"
    added_completion = next(
        line for line in patch.splitlines() if line.startswith("+") and completion in line
    )
    assert after_finished in patch
    assert patch.index(after_finished) < patch.index(added_completion)
