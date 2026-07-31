"""Canonical, collision-audited identities for decision semantics.

The digest is an index, never the semantic authority.  Equality retains the
canonical payload bytes and :class:`SemanticKeyIndex` rejects a digest that is
ever observed with different canonical content.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

SEMANTIC_KEY_CONTRACT_VERSION: Final = "sts2-semantic-key-v1"

DigestFunction = Callable[[bytes], str]


class SemanticContractError(ValueError):
    """Raised when external data cannot be represented without guessing."""


class SemanticCollisionError(RuntimeError):
    """Raised when one digest is observed for two different payloads."""


class CoarseActionCollisionError(RuntimeError):
    """Raised when a coarse action identity is not injective in one decision."""


def _canonical_value(value: Any, *, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                location = ".".join(path) or "<root>"
                raise SemanticContractError(f"semantic mappings require string keys at {location}")
            projected[raw_key] = _canonical_value(
                child,
                path=(*path, raw_key),
            )
        return projected
    if isinstance(value, list | tuple):
        return [_canonical_value(child, path=(*path, str(index))) for index, child in enumerate(value)]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            location = ".".join(path) or "<root>"
            raise SemanticContractError(f"semantic payload contains non-finite number at {location}")
        if value == 0.0:
            return 0
        if value.is_integer() and abs(value) <= 2**53:
            return int(value)
        return value
    location = ".".join(path) or "<root>"
    raise SemanticContractError(
        f"semantic payload supports only JSON-compatible values, got {type(value).__name__} at {location}"
    )


def canonical_payload_bytes(payload: Any) -> bytes:
    """Return the sole authoritative serialization for a semantic payload."""

    projected = _canonical_value(payload)
    return json.dumps(
        projected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticKey:
    """A versioned semantic identity retaining both payload and digest."""

    namespace: str
    schema_version: str
    canonical_payload: bytes
    digest: str

    @classmethod
    def from_payload(
        cls,
        *,
        namespace: str,
        schema_version: str,
        payload: Any,
        digest_function: DigestFunction = _sha256,
    ) -> SemanticKey:
        if not namespace.strip():
            raise SemanticContractError("semantic key namespace must be non-empty")
        if not schema_version.strip():
            raise SemanticContractError("semantic key schema_version must be non-empty")
        canonical = canonical_payload_bytes(payload)
        digest = digest_function(canonical)
        if not digest or not isinstance(digest, str):
            raise SemanticContractError("semantic digest function must return a non-empty string")
        return cls(
            namespace=namespace,
            schema_version=schema_version,
            canonical_payload=canonical,
            digest=digest,
        )

    @property
    def payload(self) -> Any:
        """Decode the canonical payload for audits and durable manifests."""

        return json.loads(self.canonical_payload)

    def verify(self, digest_function: DigestFunction = _sha256) -> None:
        """Fail if the payload and stored digest no longer agree."""

        actual = digest_function(self.canonical_payload)
        if actual != self.digest:
            raise SemanticCollisionError(
                "semantic key digest does not match its canonical payload: "
                f"namespace={self.namespace!r} expected={self.digest!r} "
                f"actual={actual!r}"
            )


class SemanticKeyIndex:
    """Intern keys while checking digest-to-payload injectivity."""

    def __init__(self, digest_function: DigestFunction = _sha256) -> None:
        self._digest_function = digest_function
        self._payloads: dict[tuple[str, str, str], bytes] = {}

    def intern(
        self,
        *,
        namespace: str,
        schema_version: str,
        payload: Any,
    ) -> SemanticKey:
        key = SemanticKey.from_payload(
            namespace=namespace,
            schema_version=schema_version,
            payload=payload,
            digest_function=self._digest_function,
        )
        index_key = (key.namespace, key.schema_version, key.digest)
        previous = self._payloads.get(index_key)
        if previous is not None and previous != key.canonical_payload:
            raise SemanticCollisionError(
                "semantic digest collision detected: "
                f"namespace={namespace!r} schema={schema_version!r} "
                f"digest={key.digest!r}"
            )
        self._payloads[index_key] = key.canonical_payload
        return key

    def validate(self, key: SemanticKey) -> None:
        key.verify(self._digest_function)
        index_key = (key.namespace, key.schema_version, key.digest)
        previous = self._payloads.get(index_key)
        if previous is not None and previous != key.canonical_payload:
            raise SemanticCollisionError("semantic key conflicts with an interned canonical payload")

    def __len__(self) -> int:
        return len(self._payloads)


@dataclass(frozen=True, slots=True)
class IdentityTriple:
    """Three deliberately different identities for one node or action.

    ``exact`` retains all reward- and transition-relevant semantic facts.
    ``loop`` removes reviewed costs and other non-control churn for recurrence.
    ``comparison`` retains policy-relevant resources for factual outcome
    comparisons while still removing transport and presentation noise.
    """

    exact: SemanticKey
    loop: SemanticKey
    comparison: SemanticKey


def verify_coarse_action_injectivity(
    actions: Sequence[IdentityTriple],
    *,
    strict_equivalence_fingerprints: Sequence[str | None],
) -> None:
    """Reject unreviewed many-to-one loop identities in one decision.

    A repeated loop identity is accepted only when every member was already
    collapsed into one strict group.  Callers therefore pass one identity per
    group; any duplicate here is necessarily ambiguous and fails closed.
    """

    if len(actions) != len(strict_equivalence_fingerprints):
        raise ValueError("action identities and strict equivalence metadata must align")
    seen: dict[tuple[str, str, str], tuple[int, SemanticKey]] = {}
    for index, identities in enumerate(actions):
        loop = identities.loop
        lookup = (loop.namespace, loop.schema_version, loop.digest)
        previous = seen.get(lookup)
        if previous is None:
            seen[lookup] = (index, loop)
            continue
        previous_index, previous_key = previous
        if previous_key.canonical_payload != loop.canonical_payload:
            raise SemanticCollisionError("loop action digest collision survived SemanticKey validation")
        raise CoarseActionCollisionError(
            "distinct learned candidates share one loop action identity: "
            f"candidate_indexes=({previous_index},{index}) "
            f"digest={loop.digest!r}"
        )
