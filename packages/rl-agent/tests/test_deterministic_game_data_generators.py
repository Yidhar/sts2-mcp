"""Regression tests for byte-deterministic committed-data generators.

These tests use tiny synthetic inputs and redirect every output to pytest's
temporary directory.  They intentionally do not regenerate ``game-data``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = RL_AGENT_ROOT.parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

AUDITED_SCRIPT_PATHS = (
    REPOSITORY_ROOT / "tools" / "game_data" / "build_manifest.py",
    REPOSITORY_ROOT / "tools" / "game_data" / "verify_manifest.py",
    RL_AGENT_ROOT / "import_sts2_exporter_items.py",
)


def _load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_no_invocation_path(tree: dict[str, bytes], *path_tokens: str) -> None:
    text = "\n".join(blob.decode("utf-8") for blob in tree.values())
    assert "generated_at_utc" not in text
    for token in path_tokens:
        assert token not in text


@pytest.mark.parametrize("path", AUDITED_SCRIPT_PATHS, ids=lambda path: path.name)
def test_committable_generators_do_not_read_the_wall_clock(path: Path) -> None:
    """A timestamp cannot accidentally reappear behind a current-time call."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    assert "generated_at_utc" not in source
    assert "_utc_now" not in source
    assert "SOURCE_DATE_EPOCH" not in source
    assert not any(
        isinstance(node, ast.Import | ast.ImportFrom)
        and (
            (isinstance(node, ast.ImportFrom) and node.module == "datetime")
            or (
                isinstance(node, ast.Import)
                and any(alias.name == "datetime" for alias in node.names)
            )
        )
        for node in ast.walk(tree)
    )


def test_exporter_import_is_byte_stable_across_input_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    importer = _load_script(
        "test_deterministic_import_sts2_exporter_items",
        RL_AGENT_ROOT / "import_sts2_exporter_items.py",
    )
    payload = {
        "mod": {"id": "determinism-fixture", "version": "1"},
        "cards": [
            {
                "id": "CARD.DETERMINISM_FIXTURE",
                "name": "Determinism Fixture",
                "color": "ironclad",
                "rarity": "Common",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": "1",
                "upgrades": 0,
                "description": "Deal 3 damage.",
                "semanticTags": ["damage"],
                "semanticSignals": {"damage": 3},
            }
        ],
    }
    input_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    sources = (
        tmp_path / "machine-a-root" / "first-export-name.json",
        tmp_path / "machine-b-root" / "renamed-export-input.json",
    )
    for source in sources:
        source.parent.mkdir(parents=True)
        source.write_bytes(input_bytes)

    results: list[dict[str, bytes]] = []
    for index, source in enumerate(sources):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", str(index + 1))
        run_root = tmp_path / f"import-run-{index}"
        game_data_root = run_root / "game-data"
        artifact_root = run_root / "artifacts"
        monkeypatch.setenv("STS2_GAME_DATA_ROOT", str(game_data_root))
        monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(artifact_root))
        content_dir = game_data_root / "generated"
        dataset_dir = artifact_root / "datasets"
        assert (
            importer.main(
                [
                    "--items",
                    str(source),
                    "--content-dir",
                    str(content_dir),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--no-copy-raw",
                ]
            )
            == 0
        )
        results.append(_tree_bytes(run_root))

    assert results[0] == results[1]
    generated_text = "\n".join(
        blob.decode("utf-8") for blob in results[0].values()
    )
    assert "semantic_tags" not in generated_text
    assert "semantic_signals" not in generated_text
    _assert_no_invocation_path(
        results[0],
        "machine-a-root",
        "machine-b-root",
        "first-export-name.json",
        "renamed-export-input.json",
    )


def test_game_data_manifest_is_byte_stable_across_checkout_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_script(
        "test_deterministic_build_game_data_manifest",
        REPOSITORY_ROOT / "tools" / "game_data" / "build_manifest.py",
    )
    manifests: list[bytes] = []
    for index in range(2):
        checkout = tmp_path / f"checkout-on-machine-{index}"
        data_root = checkout / "game-data"
        (data_root / "raw").mkdir(parents=True)
        (data_root / "generated").mkdir()
        (checkout / "third_party").mkdir()
        (checkout / "third_party" / "sts2-ai.lock.json").write_text(
            '{"commit": "0123456789abcdef"}\n',
            encoding="utf-8",
        )
        (data_root / "raw" / "input.json").write_bytes(b'{"value": 1}\n')
        (data_root / "generated" / "output.json").write_bytes(b'{"value": 2}\n')

        monkeypatch.setenv("SOURCE_DATE_EPOCH", str(index + 100))
        monkeypatch.setattr(builder, "ROOT", checkout)
        monkeypatch.setattr(builder, "DATA_ROOT", data_root)
        builder.main()
        manifests.append((data_root / "manifest.json").read_bytes())

    assert manifests[0] == manifests[1]
    document = json.loads(manifests[0])
    assert "generated_at_utc" not in document
    assert [entry["path"] for entry in document["files"]] == [
        "raw/input.json",
        "generated/output.json",
    ]
    assert "checkout-on-machine" not in manifests[0].decode("utf-8")


def test_committed_static_card_catalog_contains_no_policy_annotations() -> None:
    path = REPOSITORY_ROOT / "game-data" / "generated" / "cards.static.generated.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    forbidden = {
        "keep",
        "observations",
        "prior",
        "priors",
        "quality",
        "quality_score",
        "score",
        "semantic_signals",
        "semantic_tags",
        "strategy",
        "task_summaries",
    }

    def collect_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return {
                *(str(key).lower() for key in value),
                *(
                    child_key
                    for child in value.values()
                    for child_key in collect_keys(child)
                ),
            }
        if isinstance(value, list):
            return {
                child_key
                for child in value
                for child_key in collect_keys(child)
            }
        return set()

    assert collect_keys(payload).isdisjoint(forbidden)
