"""Static regression guard for combat-sandbox snapshot encounter filtering.

``--encounter-pool`` previously constrained only the fallback sampler.  Snapshot
rows still carried their own ``encounter_id`` and CombatSandboxEnv reset prefers
that row value, so the training run leaked into every encounter contained in the
dataset.  This test deliberately checks the training launcher wiring, not the
dataset module (which already supports ``encounter_ids``).
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_train_snapshot_pool_from_path_receives_encounter_pool_filter() -> None:
    train_path = Path(__file__).resolve().parents[1] / "muzero" / "train.py"
    # ``train.py`` may carry a UTF-8 BOM on Windows checkouts; use utf-8-sig
    # so this guard remains about launcher wiring rather than file encoding.
    tree = ast.parse(train_path.read_text(encoding="utf-8-sig"))

    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "from_path"
            and isinstance(func.value, ast.Name)
            and func.value.id == "CombatSnapshotPool"
        ):
            calls.append(node)

    assert calls, "expected train.py to construct CombatSnapshotPool.from_path(...)"
    keyword_values = {
        kw.arg: ast.unparse(kw.value)
        for call in calls
        for kw in call.keywords
        if kw.arg is not None
    }
    assert keyword_values.get("encounter_ids") == "encounter_pool or None"
