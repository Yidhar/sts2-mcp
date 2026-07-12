"""Mandatory dependency-free validation for versioned contract assets."""

from __future__ import annotations

import copy
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "contracts"
OPENAPI = CONTRACTS / "openapi" / "bridge-v2.openapi.json"


class ContractValidationError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractValidationError(f"{path}: expected a JSON object")
    return value


def resolve_pointer(document: Any, fragment: str, label: str) -> Any:
    if not fragment:
        return document
    if not fragment.startswith("/"):
        raise ContractValidationError(f"{label}: unsupported JSON pointer #{fragment}")
    current = document
    for raw in fragment[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ContractValidationError(f"{label}: unresolved JSON pointer #{fragment}")
    return current


def resolve_ref(
    reference: str,
    document_path: Path,
    root_document: dict[str, Any],
) -> tuple[Any, Path, dict[str, Any]]:
    relative, marker, fragment = reference.partition("#")
    if relative:
        target_path = (document_path.parent / relative).resolve()
        if not target_path.is_file():
            raise ContractValidationError(
                f"{document_path.relative_to(ROOT)}: unresolved ref {reference!r}"
            )
        target_root = load_json(target_path)
    else:
        target_path = document_path
        target_root = root_document
    target = resolve_pointer(
        target_root,
        fragment if marker else "",
        f"{document_path}:{reference}",
    )
    return target, target_path, target_root


def type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise ContractValidationError(f"unsupported JSON Schema type {expected!r}")


def is_valid(value: Any, schema: Any, path: Path, root: dict[str, Any]) -> bool:
    try:
        validate(value, schema, path, root, "$")
        return True
    except ContractValidationError:
        return False


def validate_format(value: str, name: str, location: str) -> None:
    try:
        if name == "uuid":
            uuid.UUID(value)
        elif name == "date-time":
            datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractValidationError(
            f"{location}: {value!r} is not a valid {name}"
        ) from exc


def validate(
    value: Any,
    schema: Any,
    document_path: Path,
    root_document: dict[str, Any],
    location: str,
) -> None:
    if schema is True:
        return
    if schema is False:
        raise ContractValidationError(f"{location}: rejected by false schema")
    if not isinstance(schema, dict):
        raise ContractValidationError(f"{location}: schema must be an object or boolean")

    if isinstance(schema.get("$ref"), str):
        target, path, root = resolve_ref(
            schema["$ref"], document_path, root_document
        )
        validate(value, target, path, root, location)

    for branch in schema.get("allOf", []):
        validate(value, branch, document_path, root_document, location)
    if "anyOf" in schema and not any(
        is_valid(value, branch, document_path, root_document)
        for branch in schema["anyOf"]
    ):
        raise ContractValidationError(f"{location}: does not match any anyOf branch")
    if "oneOf" in schema:
        matches = sum(
            is_valid(value, branch, document_path, root_document)
            for branch in schema["oneOf"]
        )
        if matches != 1:
            raise ContractValidationError(
                f"{location}: matches {matches} oneOf branches, expected one"
            )
    if "not" in schema and is_valid(
        value, schema["not"], document_path, root_document
    ):
        raise ContractValidationError(f"{location}: matches forbidden not schema")
    if "if" in schema:
        condition = is_valid(value, schema["if"], document_path, root_document)
        selected = schema.get("then") if condition else schema.get("else")
        if selected is not None:
            validate(value, selected, document_path, root_document, location)

    if "const" in schema and value != schema["const"]:
        raise ContractValidationError(
            f"{location}: expected {schema['const']!r}, got {value!r}"
        )
    if "enum" in schema and value not in schema["enum"]:
        raise ContractValidationError(
            f"{location}: {value!r} is not one of {schema['enum']!r}"
        )

    expected = schema.get("type")
    if expected is not None:
        choices = [expected] if isinstance(expected, str) else expected
        if not any(type_matches(value, item) for item in choices):
            raise ContractValidationError(
                f"{location}: expected type {choices!r}, got {type(value).__name__}"
            )

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ContractValidationError(f"{location}: string is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ContractValidationError(f"{location}: string is too long")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ContractValidationError(f"{location}: pattern mismatch")
        if isinstance(schema.get("format"), str):
            validate_format(value, schema["format"], location)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ContractValidationError(f"{location}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ContractValidationError(f"{location}: above maximum")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ContractValidationError(f"{location}: too few array items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ContractValidationError(f"{location}: too many array items")
        if schema.get("uniqueItems"):
            canonical = [json.dumps(item, sort_keys=True) for item in value]
            if len(canonical) != len(set(canonical)):
                raise ContractValidationError(f"{location}: duplicate array items")
        if "items" in schema:
            for index, item in enumerate(value):
                validate(
                    item,
                    schema["items"],
                    document_path,
                    root_document,
                    f"{location}[{index}]",
                )
        if "contains" in schema and not any(
            is_valid(item, schema["contains"], document_path, root_document)
            for item in value
        ):
            raise ContractValidationError(f"{location}: contains constraint failed")

    if isinstance(value, dict):
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ContractValidationError(
                f"{location}: missing required properties {missing!r}"
            )
        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key in value:
                validate(
                    value[key],
                    child_schema,
                    document_path,
                    root_document,
                    f"{location}.{key}",
                )
        extras = set(value) - set(properties)
        additional = schema.get("additionalProperties", True)
        if additional is False and extras:
            raise ContractValidationError(
                f"{location}: unexpected properties {sorted(extras)!r}"
            )
        if isinstance(additional, dict):
            for key in extras:
                validate(
                    value[key],
                    additional,
                    document_path,
                    root_document,
                    f"{location}.{key}",
                )
        if len(value) < schema.get("minProperties", 0):
            raise ContractValidationError(f"{location}: too few object properties")


def walk_refs(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "$ref" and isinstance(child, str):
                yield child
            yield from walk_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_refs(child)


def validate_ref_graph(document: dict[str, Any], path: Path) -> None:
    for reference in walk_refs(document):
        resolve_ref(reference, path, document)


def assert_rejected(
    value: Any,
    schema_ref: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    label: str,
) -> None:
    schema, path, root = resolve_ref(schema_ref, manifest_path, manifest)
    if is_valid(value, schema, path, root):
        raise ContractValidationError(
            f"negative assertion unexpectedly validated: {label}"
        )


def validate_semantic_guards(
    manifest: dict[str, Any],
    manifest_path: Path,
    fixtures: dict[str, dict[str, Any]],
) -> None:
    command = copy.deepcopy(fixtures["fixtures/command.play-card.json"])
    command["capability"] = "training"
    assert_rejected(
        command,
        "schemas/command.schema.json",
        manifest_path,
        manifest,
        "training capability submitted to player command",
    )

    alias = copy.deepcopy(fixtures["fixtures/command.play-card.json"])
    handle = alias["command"].pop("action_handle")
    alias["command"]["action_id"] = handle
    assert_rejected(
        alias,
        "schemas/command.schema.json",
        manifest_path,
        manifest,
        "v2 player command using legacy action_id alias",
    )

    missing_deadline = copy.deepcopy(fixtures["fixtures/command.play-card.json"])
    missing_deadline.pop("deadline_utc")
    assert_rejected(
        missing_deadline,
        "schemas/command.schema.json",
        manifest_path,
        manifest,
        "v2 player command without deadline_utc",
    )

    null_action_handle = copy.deepcopy(fixtures["fixtures/command.play-card.json"])
    null_action_handle["command"]["action_handle"] = None
    assert_rejected(
        null_action_handle,
        "schemas/command.schema.json",
        manifest_path,
        manifest,
        "v2 player command with a null action_handle",
    )

    excessive_wait = copy.deepcopy(fixtures["fixtures/command.play-card.json"])
    excessive_wait["command"]["wait_after_ms"] = 5_001
    assert_rejected(
        excessive_wait,
        "schemas/command.schema.json",
        manifest_path,
        manifest,
        "v2 player command with wait_after_ms above 5000",
    )

    leaked_state = copy.deepcopy(fixtures["fixtures/state.player-control.json"])
    leaked_state["state"]["target_priority_hints"] = ["enemy-1"]
    assert_rejected(
        leaked_state,
        "schemas/state.schema.json",
        manifest_path,
        manifest,
        "player state with a non-allowlisted policy hint",
    )

    oversized_event = copy.deepcopy(fixtures["fixtures/event.frontier.json"])
    oversized_event["state"] = leaked_state
    assert_rejected(
        oversized_event,
        "schemas/event.schema.json",
        manifest_path,
        manifest,
        "frontier event carrying a full state payload",
    )

    reset = copy.deepcopy(fixtures["fixtures/environment.reset-request.json"])
    reset.pop("expected_state_version")
    assert_rejected(
        reset,
        "schemas/environment.schema.json#/$defs/reset_request",
        manifest_path,
        manifest,
        "reset without expected_state_version",
    )

    step_alias = copy.deepcopy(fixtures["fixtures/environment.step-request.json"])
    step_alias["action"] = {"action_id": "combat:end-turn"}
    assert_rejected(
        step_alias,
        "schemas/environment.schema.json#/$defs/step_request",
        manifest_path,
        manifest,
        "v2 environment step using legacy action_id alias",
    )

    step_dual = copy.deepcopy(fixtures["fixtures/environment.step-request.json"])
    step_dual["action"]["action_index"] = 0
    assert_rejected(
        step_dual,
        "schemas/environment.schema.json#/$defs/step_request",
        manifest_path,
        manifest,
        "v2 environment step using both action_index and action_handle",
    )

    for fixture_name, schema_ref in (
        (
            "fixtures/environment.result.json",
            "schemas/environment.schema.json#/$defs/result",
        ),
        (
            "fixtures/environment.state.json",
            "schemas/environment.schema.json#/$defs/state_response",
        ),
    ):
        legal_actions = fixtures[fixture_name]["legal_actions"]
        if not legal_actions or any(
            "action_handle" not in action or "action_id" in action
            for action in legal_actions
        ):
            raise ContractValidationError(
                f"{fixture_name}: v2 legal actions must use action_handle only"
            )
        legacy_identity = copy.deepcopy(fixtures[fixture_name])
        action = legacy_identity["legal_actions"][0]
        action["action_id"] = action.pop("action_handle")
        assert_rejected(
            legacy_identity,
            schema_ref,
            manifest_path,
            manifest,
            f"{fixture_name} using legacy action_id identity",
        )

    legacy_spec_shape = copy.deepcopy(fixtures["fixtures/environment.spec.json"])
    legacy_spec_shape["action_encoding"]["legal_action_shape"] = [
        "idx",
        "action_id",
        "kind",
    ]
    assert_rejected(
        legacy_spec_shape,
        "schemas/environment.schema.json#/$defs/spec_response",
        manifest_path,
        manifest,
        "v2 environment spec advertising legacy action_id",
    )

    committed = copy.deepcopy(fixtures["fixtures/command-result.committed.json"])
    committed["ok"] = False
    assert_rejected(
        committed,
        "schemas/command-result.schema.json",
        manifest_path,
        manifest,
        "committed result with ok=false",
    )

    rejected = copy.deepcopy(fixtures["fixtures/command-result.rejected.json"])
    rejected["error"] = None
    assert_rejected(
        rejected,
        "schemas/command-result.schema.json",
        manifest_path,
        manifest,
        "rejected result without structured error",
    )

    player = copy.deepcopy(fixtures["fixtures/session.player-control.json"])
    player["token"] = "x" * 32
    assert_rejected(
        player,
        "schemas/session-descriptor.schema.json",
        manifest_path,
        manifest,
        "top-level token while legacy-v1 is disabled",
    )

    missing_game_compatibility = copy.deepcopy(
        fixtures["fixtures/session.player-control.json"]
    )
    del missing_game_compatibility["game_compatibility"]
    assert_rejected(
        missing_game_compatibility,
        "schemas/session-descriptor.schema.json",
        manifest_path,
        manifest,
        "v2 session without game compatibility evidence",
    )

    degraded_game_compatibility = copy.deepcopy(
        fixtures["fixtures/session.player-control.json"]
    )
    degraded_game_compatibility["game_compatibility"]["health"] = "degraded"
    degraded_game_compatibility["game_compatibility"]["startup_allowed"] = False
    assert_rejected(
        degraded_game_compatibility,
        "schemas/session-descriptor.schema.json",
        manifest_path,
        manifest,
        "v2 session published after a denied compatibility decision",
    )

    unexpected_game_compatibility_field = copy.deepcopy(
        fixtures["fixtures/session.player-control.json"]
    )
    unexpected_game_compatibility_field["game_compatibility"]["assembly"]["mvid"] = (
        "97f10687-c306-4798-ab75-8b9f23f34dfb"
    )
    assert_rejected(
        unexpected_game_compatibility_field,
        "schemas/session-descriptor.schema.json",
        manifest_path,
        manifest,
        "game compatibility evidence with an undeclared assembly field",
    )

    failed_game_compatibility_probe = copy.deepcopy(
        fixtures["fixtures/session.player-control.json"]
    )
    failed_game_compatibility_probe["game_compatibility"]["probes"][0]["passed"] = False
    assert_rejected(
        failed_game_compatibility_probe,
        "schemas/session-descriptor.schema.json",
        manifest_path,
        manifest,
        "v2 session containing a failed compatibility probe",
    )

    public = copy.deepcopy(fixtures["fixtures/health.public.json"])
    public["session_id"] = "must-not-be-public"
    assert_rejected(
        public,
        "schemas/health.schema.json#/$defs/public",
        manifest_path,
        manifest,
        "public health leaking session identity",
    )


def validate_openapi(openapi: dict[str, Any]) -> None:
    validate_ref_graph(openapi, OPENAPI)
    required = {
        "/health",
        "/v2/health",
        "/v2/state",
        "/v2/commands",
        "/v2/commands/{requestId}",
        "/v2/events",
        "/v2/env/spec",
        "/v2/env/state",
        "/v2/env/combat_catalog",
        "/v2/env/reset",
        "/v2/env/step",
    }
    paths = openapi.get("paths")
    if not isinstance(paths, dict) or not required.issubset(paths):
        raise ContractValidationError(
            f"OpenAPI missing paths: {sorted(required - set(paths or {}))}"
        )
    if paths["/health"]["get"].get("security") != []:
        raise ContractValidationError("GET /health must be explicitly public")

    health_security = paths["/v2/health"]["get"].get("security") or []
    actual = {
        next(iter(item))
        for item in health_security
        if isinstance(item, dict) and len(item) == 1
    }
    expected = {
        "PlayerControlBearer",
        "TrainingBearer",
        "LegacyPrivilegedBearer",
    }
    if actual != expected:
        raise ContractValidationError(
            "GET /v2/health must accept every published scoped bearer"
        )

    for endpoint in (
        "/v2/env/spec",
        "/v2/env/state",
        "/v2/env/combat_catalog",
        "/v2/env/reset",
        "/v2/env/step",
    ):
        operation = next(
            value
            for key, value in paths[endpoint].items()
            if key in {"get", "post"}
        )
        if operation.get("security") != [{"TrainingBearer": []}]:
            raise ContractValidationError(
                f"{endpoint} must require TrainingBearer"
            )

    environment_wire_description = json.dumps(
        {key: paths[key] for key in required if key.startswith("/v2/env/")},
        sort_keys=True,
    )
    if "action_handle" not in environment_wire_description or "action_id" in environment_wire_description:
        raise ContractValidationError(
            "OpenAPI v2 environment operations must document action_handle without the legacy action_id alias"
        )


def main() -> int:
    try:
        manifest_path = CONTRACTS / "manifest.json"
        manifest = load_json(manifest_path)
        schema_paths = [CONTRACTS / item for item in manifest.get("schemas", [])]
        if not schema_paths:
            raise ContractValidationError("manifest must list schemas")

        identifiers: set[str] = set()
        for path in schema_paths:
            schema = load_json(path)
            if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
                raise ContractValidationError(f"{path}: wrong JSON Schema dialect")
            identifier = schema.get("$id")
            if not isinstance(identifier, str) or not identifier:
                raise ContractValidationError(f"{path}: missing $id")
            if identifier in identifiers:
                raise ContractValidationError(f"duplicate schema $id: {identifier}")
            identifiers.add(identifier)
            validate_ref_graph(schema, path)

        mapping = manifest.get("fixtures")
        if not isinstance(mapping, dict) or not mapping:
            raise ContractValidationError("manifest must map fixtures to schemas")
        actual = {
            path.relative_to(CONTRACTS).as_posix()
            for path in (CONTRACTS / "fixtures").glob("*.json")
        }
        declared = set(mapping)
        if actual != declared:
            raise ContractValidationError(
                "fixture manifest mismatch: "
                f"undeclared={sorted(actual - declared)}, "
                f"missing={sorted(declared - actual)}"
            )

        fixtures: dict[str, dict[str, Any]] = {}
        for relative, schema_ref in sorted(mapping.items()):
            fixture = load_json(CONTRACTS / relative)
            schema, path, root = resolve_ref(
                schema_ref, manifest_path, manifest
            )
            validate(fixture, schema, path, root, "$")
            fixtures[relative] = fixture

        validate_semantic_guards(manifest, manifest_path, fixtures)
        validate_openapi(load_json(OPENAPI))
    except (ContractValidationError, json.JSONDecodeError, OSError) as exc:
        print(f"contract validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"contracts ok: {len(schema_paths)} schemas, "
        f"{len(fixtures)} fixtures, OpenAPI and semantic guards"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
