# MCP package development guide

## Purpose

This package is the typed player-control MCP adapter. It does not own game reward semantics,
RL episode state, training policy, knowledge files, journals, static catalog generation, or
game-internal reflection.

## Required commands

```powershell
npm ci
npm run typecheck
npm test
```

`dist/` is ignored generated output. The root MCP configuration invokes `index.js` directly;
the launcher validates the generated contract-version copy and performs a local TypeScript build
when output is missing or stale. Never commit `dist/`.

## Dependency direction

```text
contracts / game-data
          ↓
config → session → bridge-client
                    ↓
presentation ← workflows ← tools
                    ↓
             tool-registry
                    ↓
       official MCP SDK server
```

Do not add imports or path fallbacks from `rl-agent`; shared metadata belongs only in
repository-level `game-data/generated`.

## Mutation rules

1. Every gameplay mutation is bound to an integer revision.
2. `strict=false` is forbidden.
3. Never retry a legacy mutation after conflict, timeout, connection reset, or 5xx.
4. Prefer control-v2 request IDs and command-status reconciliation.
5. A workflow may issue multiple deliberate commands, but it must re-read state and bind every
   next command to the new revision.
6. Do not expose reset/sandbox/training operations outside the debug profile.
7. HTTP success is transport-only. A v2 mutation succeeds only for an internally consistent
   `ok=true`, `status=committed` envelope with an object result.
8. For v2, scoped tokens are mandatory. Never fall back to the legacy top-level token.
9. Canonical legal-action identity is `handle`; aliases exist only for legacy-only sessions.

## Secret rules

Never return or log:

- session token;
- Authorization header;
- complete raw session descriptor;
- cookies, API keys, or passwords.

All outward errors pass through `errorBody` and `toolResult`; all serialization passes through
`redactSecrets`/`safeStringify`. Add a redaction regression test for any new secret-bearing
field.

## Module ownership

- `protocol/server.ts`: transport and SDK lifecycle only.
- `session.ts`: session parsing, liveness, identity, URL allowlist.
- `bridge-client.ts`: HTTP and v1/v2 command semantics only.
- `workflows.ts`: legal action selection and polling primitives.
- `presentation.ts`: output projection, never policy or reward.
- `tools/control.ts`: normal player-control surface.
- `tools/strategic.ts`: optional views and ordered commands.
- `tools/training.ts`: privileged debug-only surface.
- `tool-registry.ts`: the sole profile membership authority.

Keep individual production files below roughly 1,500 lines. Do not rebuild a monolith in
`index.js` or a tool module.
