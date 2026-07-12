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
    RL_AGENT_ROOT / "tools" / "generate_card_effect_profiles.py",
    RL_AGENT_ROOT / "tools" / "audit_card_mechanism_coverage.py",
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
        isinstance(node, (ast.Import, ast.ImportFrom))
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


def test_card_effect_profile_generator_is_byte_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _load_script(
        "test_deterministic_generate_card_effect_profiles",
        RL_AGENT_ROOT / "tools" / "generate_card_effect_profiles.py",
    )
    monkeypatch.setattr(generator, "load_catalog", lambda: [{"id": "CARD.B"}, {"id": "CARD.A"}])
    monkeypatch.setattr(
        generator,
        "build_profile",
        lambda row: {
            "id": row["id"],
            "operations": [],
            "training_tags": ["determinism_fixture"],
        },
    )

    outputs: list[bytes] = []
    for index in range(2):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", str(index + 10))
        output = tmp_path / f"effect-run-{index}" / "card_effect_profiles.generated.json"
        monkeypatch.setattr(generator, "OUT_PATH", output)
        generator.main()
        outputs.append(output.read_bytes())

    assert outputs[0] == outputs[1]
    assert outputs[0].endswith(b"\n")
    document = json.loads(outputs[0])
    assert "generated_at_utc" not in document
    assert list(document["cards"]) == ["CARD.A", "CARD.B"]
    assert str(REPOSITORY_ROOT) not in outputs[0].decode("utf-8")


def test_card_mechanism_audit_is_byte_stable_across_input_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _load_script(
        "test_deterministic_audit_card_mechanism_coverage",
        RL_AGENT_ROOT / "tools" / "audit_card_mechanism_coverage.py",
    )
    monkeypatch.setattr(audit, "_load_catalog", lambda: {})

    input_bytes = b'{"cards": []}\n'
    sources = (
        tmp_path / "audit-machine-a" / "first-items-name.json",
        tmp_path / "audit-machine-b" / "renamed-items-input.json",
    )
    for source in sources:
        source.parent.mkdir(parents=True)
        source.write_bytes(input_bytes)

    results: list[dict[str, bytes]] = []
    for index, source in enumerate(sources):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", str(index + 20))
        out_dir = tmp_path / f"audit-run-{index}"
        monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(out_dir))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit_card_mechanism_coverage.py",
                "--items",
                str(source),
                "--out-dir",
                str(out_dir),
                "--report",
                str(out_dir / "card-mechanism-coverage-audit.md"),
            ],
        )
        audit.main()
        results.append(_tree_bytes(out_dir))

    assert results[0] == results[1]
    _assert_no_invocation_path(
        results[0],
        "audit-machine-a",
        "audit-machine-b",
        "first-items-name.json",
        "renamed-items-input.json",
    )
    assert all(b"Generated:" not in blob for blob in results[0].values())


def test_card_mechanism_checked_in_publication_has_fixed_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _load_script(
        "test_checked_in_card_mechanism_publication_boundaries",
        RL_AGENT_ROOT / "tools" / "audit_card_mechanism_coverage.py",
    )
    repository = tmp_path / "checkout"
    (repository / "docs" / "generated").mkdir(parents=True)
    monkeypatch.setattr(audit, "PROJECT_ROOT", repository)

    out_dir, report, output_uri = audit._resolve_output_paths(
        publish_checked_in_docs=True,
        out_dir=None,
        report=None,
    )
    assert out_dir == repository / "docs" / "generated"
    assert report == out_dir / "card-mechanism-coverage-audit.md"
    assert output_uri == "repo://docs/generated"

    with pytest.raises(ValueError, match="cannot be combined"):
        audit._resolve_output_paths(
            publish_checked_in_docs=True,
            out_dir=str(tmp_path / "arbitrary-source-path"),
            report=None,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        audit._resolve_output_paths(
            publish_checked_in_docs=True,
            out_dir=None,
            report=str(tmp_path / "arbitrary-report.md"),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_card_mechanism_coverage.py",
            "--publish-checked-in-docs",
            "--items",
            str(tmp_path / "arbitrary-input.json"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        audit.main()
    assert exc_info.value.code == 2


def test_card_mechanism_checked_in_input_normalization_is_semantic() -> None:
    audit = _load_script(
        "test_checked_in_card_mechanism_input_normalization",
        RL_AGENT_ROOT / "tools" / "audit_card_mechanism_coverage.py",
    )
    payload = {
        "CARD.FIXTURE": {
            "id": "CARD.FIXTURE",
            "title": "Fixture",
            "color": "ironclad",
            "rarity": "Common",
            "type": "Attack",
            "target": "AnyEnemy",
            "upgrade_level_texts": {
                "0": {
                    "canonical_text": "Fixture base",
                    "description": "Deal 3 damage.",
                    "effect": "3 damage",
                    "energy_cost": 1,
                    "energy_cost_text": "1",
                    "keywords": [],
                    "semantic_signals": {"damage": 3},
                    "semantic_tags": ["damage"],
                },
                "1": {
                    "canonical_text": "Fixture upgraded",
                    "description": "Deal 5 damage.",
                    "effect": "5 damage",
                    "energy_cost": 1,
                    "energy_cost_text": "1",
                    "keywords": [],
                    "semantic_signals": {"damage": 5},
                    "semantic_tags": ["damage"],
                },
            },
        },
        "__meta__": {"schema": "fixture"},
    }

    rows, source_kind = audit._card_rows(payload)
    assert source_kind == "checked-in-card-data"
    assert [row["upgrades"] for row in rows] == [0, 1]
    assert rows[0]["semanticSignals"] == {"damage": 3}
    assert rows[1]["description"] == "Deal 5 damage."
