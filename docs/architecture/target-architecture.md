# STS2 MCP target architecture

Status: accepted for the 2.0 refactor.

## Dependency direction

```text
contracts -> bridge-mod / mcp-server / rl-trainer
game-data -> offline catalog and release-audit tools only
```

The bridge and MCP server must never import from the RL package, and the grounded
trainer must not read the static game-data catalog. Game-specific
reflection is isolated behind a versioned game adapter. Runtime artifacts are
stored outside the source checkout.

The retail adapter registry matches exact assembly identity (including
informational version and module version ID), not only the reusable assembly
version. Its required type/member probes run before Harmony and before HTTP. No
profile match, a missing capability, a probe exception, or patch activation
failure leaves compatibility health degraded and exposes no Bridge listener or
mutation endpoint. A successful session descriptor records the profile,
identity, and probe results. Runtime UI/private reflection remains localized in
adapter partials and still requires live fixture validation for each supported
profile.

## Runtime domains

### Player control

Exposes player-visible state and explicit legal actions. It cannot reset a run,
select a seed, start a combat sandbox, invoke AutoSlay, export catalogs, or
expose draw-pile order.

### Training

Exposes reset, step, seed, sandbox, and diagnostic state under an independent
scoped capability. Environment sessions are single-owner actors with an
episode id and expected step index.

### Catalog tooling

Static export and content generation are build tools, not game mutation HTTP
endpoints.

## Mutation model

Every mutation enters one bounded command actor. The actor performs session,
capability, deadline, request-id, and expected-revision checks on the game
thread; resolves a stable action handle against the current state; executes it;
and records the terminal command result. Replaying a request id returns the
recorded result without executing again.

## Training ownership

The bridge emits state and transition facts. The RL package is the sole owner
of observation encoding, action encoding, reward, curriculum, replay, model,
and checkpoint semantics. Live and headless environments implement one typed
`EnvironmentBackend` contract and share one versioned reward calculator.
