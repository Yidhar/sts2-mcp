# V2 cutover checklist

## Baseline

- [x] Preserve the pre-refactor dirty diff and hashes.
- [x] Establish a shared contract manifest and fixtures.
- [x] Establish a versioned game-data package.
- [ ] Record live fixtures for every decision surface.

## Bridge

- [x] Publish atomic versioned session descriptors.
- [x] Provide bounded idempotent commands and status lookup.
- [x] Serialize environment reset/step with expected step index.
- [x] Expose game-thread health and bounded revision events.
- [x] Split player-control and training capabilities.

## MCP

- [x] Default to minimal player-control tools.
- [x] Redact session credentials from all output.
- [x] Use the standard MCP SDK and generated validation.
- [x] Preserve stable public tool names through adapters.

## RL

- [x] Use one typed backend protocol for live and headless execution.
- [x] Use one canonical transition and reward calculator.
- [x] Record contract/content/reward hashes in replay and checkpoints.
- [x] Remove tactical post-hoc action/target rewriting from the default path.

Local evidence for the completed items is executable rather than narrative:

- `mods/sts2-bridge/tests/BridgeCore.Tests` covers strict DTO parsing, capability
  isolation, command replay/capacity/deadline semantics, bounded events, explicit
  player projection, session lifecycle, and the retail compatibility gate.
- `packages/mcp-server/test` covers SDK registration, minimal/debug profiles,
  credential redaction, strict v2 envelopes, legacy one-shot unknown outcomes, and
  exact duplicate replay wire construction.
- `packages/rl-agent/tests` covers the shared backend protocol, transition/reward
  authority, guard-off defaults, artifact boundaries, and hash/identity-gated exact
  checkpoint resume versus explicit weights-only migration.

## Removal gate

Legacy v1 may be removed only when contract, idempotency, concurrency,
live/headless parity, checkpoint migration, and end-to-end tests are green and
v1 client telemetry has reached zero.

The following promotion gates remain deliberately unchecked and cannot be inferred
from unit tests: live fixtures for every decision surface, live/headless promotion
evidence against the pinned dependency, two measured compatibility releases, and a
minimum 14-day v1-usage window ending with zero active clients. The protected release
workflow validates those external records with
`tools/release/check_external_evidence.py`; absence or stale identity blocks release.
