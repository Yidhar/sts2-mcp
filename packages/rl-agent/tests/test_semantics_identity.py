from __future__ import annotations

import pytest

from sts2_rl.semantics import (
    SemanticCollisionError,
    SemanticContractError,
    SemanticKey,
    SemanticKeyIndex,
    canonical_payload_bytes,
)


def test_semantic_key_keeps_canonical_payload_as_authority() -> None:
    first = SemanticKey.from_payload(
        namespace="test",
        schema_version="v1",
        payload={"b": [2, 3], "a": {"x": 1}},
    )
    second = SemanticKey.from_payload(
        namespace="test",
        schema_version="v1",
        payload={"a": {"x": 1}, "b": [2, 3]},
    )

    assert first == second
    assert first.payload == {"a": {"x": 1}, "b": [2, 3]}
    first.verify()


def test_semantic_key_index_fails_closed_on_digest_collision() -> None:
    index = SemanticKeyIndex(lambda payload: "forced-collision")
    index.intern(namespace="test", schema_version="v1", payload={"value": 1})

    with pytest.raises(SemanticCollisionError, match="digest collision"):
        index.intern(
            namespace="test",
            schema_version="v1",
            payload={"value": 2},
        )


def test_semantic_payload_rejects_non_json_and_non_finite_content() -> None:
    with pytest.raises(SemanticContractError, match="string keys"):
        canonical_payload_bytes({1: "not-a-string-key"})
    with pytest.raises(SemanticContractError, match="non-finite"):
        canonical_payload_bytes({"value": float("nan")})
    with pytest.raises(SemanticContractError, match="JSON-compatible"):
        canonical_payload_bytes({"value": object()})
