# ADR 0002: Serialize game mutations through an idempotent command actor

- Status: accepted
- Date: 2026-07-11

HTTP request concurrency is not a game-state concurrency model. All mutations
use a required request id, session id, capability, expected revision, and
deadline. Validation, current-state action resolution, execution, and result
commit form one game-thread transaction. Results are retained in a bounded TTL
dedupe store and can be queried by request id.
