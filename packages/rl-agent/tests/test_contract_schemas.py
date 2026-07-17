from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

from sts2_rl.contracts import ResetRequest
from sts2_rl.contracts.versions import (
    ACTION_SCHEMA_VERSION,
    API_VERSION,
    SCHEMA_VERSION,
)
from sts2_rl.encoding import MODEL_ACTION_KIND_VOCABULARY

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_ROOT = REPO_ROOT / "contracts"
MANIFEST_PATH = CONTRACT_ROOT / "manifest.json"


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _walk_refs(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "$ref" and isinstance(child, str):
                yield child
            yield from _walk_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_refs(child)


def test_manifest_versions_match_generated_python_source() -> None:
    manifest = _load(MANIFEST_PATH)
    assert manifest["api_version"] == API_VERSION
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["action_schema_version"] == ACTION_SCHEMA_VERSION


def test_environment_model_action_kind_contract_matches_grounded_encoder() -> None:
    schema = _load(CONTRACT_ROOT / "schemas" / "environment.schema.json")
    legal_action = schema["$defs"]["legal_action"]
    model_action_kind = schema["$defs"]["model_action_kind"]

    assert "model_action_kind" in legal_action["required"]
    assert legal_action["properties"]["model_action_kind"] == {
        "$ref": "#/$defs/model_action_kind"
    }
    assert set(model_action_kind["enum"]) == MODEL_ACTION_KIND_VOCABULARY


def test_manifest_lists_parseable_schemas_and_resolvable_local_refs() -> None:
    manifest = _load(MANIFEST_PATH)
    schema_paths = [CONTRACT_ROOT / relative for relative in manifest["schemas"]]
    assert schema_paths
    schemas = [_load(path) for path in schema_paths]
    ids = {schema["$id"] for schema in schemas}
    filenames = {path.name for path in schema_paths}

    for path, schema in zip(schema_paths, schemas, strict=True):
        assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
        assert schema.get("$id") in ids
        for reference in _walk_refs(schema):
            parsed = urlparse(reference)
            if parsed.scheme:
                assert reference in ids
            elif parsed.path:
                assert Path(parsed.path).name in filenames, (
                    f"{path.name} has unresolved local ref {reference!r}"
                )
            else:
                assert parsed.fragment, f"{path.name} has an empty local ref"


@pytest.mark.parametrize(
    ("fixture_name", "schema_name"),
    [
        ("session.player-control.json", "session-descriptor.schema.json"),
        ("health.ok.json", "health.schema.json"),
        ("command.play-card.json", "command.schema.json"),
        ("command-result.committed.json", "command-result.schema.json"),
        ("transition.combat-victory.json", "transition.schema.json"),
        ("transition.run-victory.json", "transition.schema.json"),
    ],
)
def test_contract_fixture_validates_with_cross_file_registry(
    fixture_name: str,
    schema_name: str,
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")

    manifest = _load(MANIFEST_PATH)
    schemas = [_load(CONTRACT_ROOT / relative) for relative in manifest["schemas"]]
    registry = referencing.Registry()
    for schema in schemas:
        registry = registry.with_resource(
            schema["$id"],
            referencing.Resource.from_contents(schema),
        )

    schema = next(item for item in schemas if item["$id"].endswith("/" + schema_name))
    fixture = _load(CONTRACT_ROOT / "fixtures" / fixture_name)
    validator = jsonschema.Draft202012Validator(
        schema,
        registry=registry,
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(fixture), key=lambda error: list(error.path))
    assert errors == [], "\n".join(error.message for error in errors)


def test_transition_outcome_scope_fields_are_required() -> None:
    schema = _load(CONTRACT_ROOT / "schemas" / "transition.schema.json")
    assert set(schema["properties"]["facts"]["required"]) >= {
        "combat_result",
        "run_result",
        "terminal_reason",
    }


def test_environment_reset_fixture_matches_schema_and_typed_request() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load(CONTRACT_ROOT / "schemas" / "environment.schema.json")
    fixture = _load(CONTRACT_ROOT / "fixtures" / "environment.reset-request.json")
    jsonschema.Draft202012Validator(
        schema["$defs"]["reset_request"],
        format_checker=jsonschema.FormatChecker(),
    ).validate(fixture)

    request = ResetRequest(
        request_id=fixture["request_id"],
        session_id=fixture["session_id"],
        scenario=fixture["scenario"],
        expected_state_version=fixture["expected_state_version"],
        seed=fixture["seed"],
        character=fixture["options"]["character"],
        force_fresh=fixture["options"]["force_fresh"],
        timeout_ms=fixture["options"]["timeout_ms"],
    )
    assert request.expected_state_version == 42
    assert request.to_v2_options()["timeout_ms"] == 30_000
