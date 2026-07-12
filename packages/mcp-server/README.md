# STS2 MCP Server

A local Model Context Protocol server for the Slay the Spire 2 Bridge mod.

Version 0.5 is a contracts-first TypeScript rewrite. The server uses the official
`@modelcontextprotocol/sdk`, validates every tool input with Zod, and keeps transport,
session discovery, Bridge HTTP access, presentation, workflows, and tool registration in
separate modules.

## Security and mutation guarantees

- The default tool profile is **`minimal`**.
- Bridge bearer tokens are never returned by tools or structured errors. Status returns only
  `token_present` and a short session identity hash.
- Session `base_url` is restricted to HTTP(S) loopback addresses. A session file cannot make
  the MCP server forward credentials to an arbitrary host.
- `sts2_perform_action` requires `expected_state_version`; `strict=false` is rejected.
- A legacy `/action` mutation is attempted **once**. HTTP conflict, connection failure, and
  timeout are returned to the caller and are never automatically retried.
- When the session advertises `control-v2`, mutations use `/v2/commands`, include a
  `request_id` plus a required deadline of at most 120 seconds, and can be
  queried with `sts2_get_command_status`. Action handles are non-empty and
  `wait_after_ms` remains bounded to 0-5000.
- A v2 session must provide the requested scoped token. The client never substitutes the
  top-level legacy credential for `player-control` or `training`.
- HTTP 200 is not treated as command success: only `ok=true`, `status=committed`, and an
  object result are successful. Rejected, pending, malformed, and outcome-unknown envelopes
  become structured MCP errors.
- Training reset/step tools are available only in the explicit `debug` profile.
- Journal, knowledge-file access, observation persistence, and the old in-process RL state
  machine are deliberately outside this control-plane server.

A timeout from a legacy mutation means the outcome is unknown. Refresh state and reconcile;
do not issue the same gameplay intent again automatically.

## Layout

```text
index.js                  built-code compatibility launcher
smoke-test.js             structured health-check compatibility launcher
src/
  server.ts               process lifecycle
  protocol/server.ts      official MCP SDK and stdio transport
  config.ts               profiles, paths, timeouts, package/contract versions
  session.ts              secure session discovery and validation
  bridge-client.ts        typed HTTP client and v1/v2 command semantics
  redaction.ts            recursive credential redaction
  presentation.ts         output shaping
  workflows.ts            legal-action selection and state waits
  tool-registry.ts        profile-owned tool surface
  tools/control.ts        minimal control tools
  tools/strategic.ts      strategic views and ordered sequences
  tools/training.ts       privileged debug-only environment tools
test/                     node:test unit and SDK integration tests
dist/                     ignored local build output created on demand
```

## Install, build, and test

Requires the Node/npm versions pinned by the repository `.nvmrc` and `packageManager` field.

```powershell
npm ci
npm run typecheck
npm test
```


`npm test` performs a clean TypeScript build, runs the compiled `node:test` suite, and removes
local `dist/` output. `index.js` and `smoke-test.js` perform an on-demand build when the
ignored output is absent or older than the TypeScript sources.
The suite covers:

- official SDK initialize/tools/call behavior;
- minimal-profile exposure;
- token redaction;
- stale-session structured errors and nonzero smoke exit status;
- loopback-only session URLs;
- v2 request IDs and strict revisions;
- the real Bridge v2 canonical `{handle, kind, coord, option_index, ...}` legal-action wire;
- committed/rejected/pending/outcome-unknown command envelope semantics;
- scoped-token fail-closed behavior and debug-only v2 environment reads;
- one-shot legacy mutation behavior;
- hard request timeouts.

Run the stdio server:

```powershell
npm run build
node index.js
```

Run a health check against the current session:

```powershell
node smoke-test.js
```

The smoke command exits `0` only when a control-v2 session advertises the required API/schema,
authenticated `/v2/health` reports both transport and game-thread readiness, and authenticated
`/v2/state` returns a valid integer revision plus canonical legal-action handles. It emits one
redacted JSON error to stderr and exits `1` for any discovery, identity, contract, auth, health,
or state failure.

## Configuration

| Variable / flag | Meaning | Default |
|---|---|---|
| `STS2_BRIDGE_SESSION_FILE` | Session descriptor written by the Bridge | `%APPDATA%/SlayTheSpire2/bridge/session.json` |
| `STS2_MCP_PROFILE` | `minimal`, `strategic`, or `debug` | `minimal` |
| `STS2_TOOL_PROFILE` / `STS2_MCP_TOOL_PROFILE` | Legacy aliases for `STS2_MCP_PROFILE` | unset |
| `--profile=<name>` | Per-process profile override | unset |
| `STS2_MCP_HTTP_TIMEOUT_MS` | Bridge HTTP timeout | `10000` |
| `STS2_MCP_WAIT_POLL_MS` | Read-only wait polling interval | `200` |

`--profile` takes precedence over the environment. Privileged `debug` must be spelled
exactly; removed aliases such as `full` and `default` fail closed to `minimal` when no
other valid profile is configured.

## Profiles

### minimal

Player-visible status/state/action tools, one-shot control commands, safe workflow helpers,
and read-only wait tools. This is the normal MCP surface.

### strategic

Everything in `minimal`, plus deck/map views and ordered card/combat sequences. Ordered
sequences are deliberate commands, not retries: state is fetched after every committed step
and the next step is bound to the newly observed revision.

### debug

Everything in `strategic`, plus privileged environment spec/state/combat-catalog/reset/step
and training-command-status tools. Reset requires an expected state revision. Step requires
`episode_id` and `expected_step_index`. All v2 requests use only the `training` token. This
profile must not be used as the normal player-control configuration.

## Structured errors

Tools return MCP `isError: true` with both text JSON and structured content:

```json
{
  "ok": false,
  "error": {
    "code": "stale_session",
    "message": "The recorded bridge process is not alive.",
    "retryable": false,
    "details": {
      "pid": 1234,
      "session_id_hash": "0123456789ab"
    }
  }
}
```

No error details contain the complete session descriptor, token, Authorization header, or
request headers.

## Compatibility changes from 0.4

- Default profile changed from `debug` to `minimal`.
- The 15k-line hand-written JSON-RPC server was removed.
- The official MCP SDK now owns framing, initialization, cancellation, ping, and validation.
- `sts2_perform_action.expected_state_version` is required.
- `strict=false` is no longer accepted.
- State-conflict auto-retry was removed.
- RL environment tools moved from `strategic` to `debug` and have stricter mutation inputs.
- Journal, knowledge, observation persistence, AutoSlay, and legacy RL smoke runners are not
  part of this package's core runtime.
- Stale-session smoke checks now fail the process instead of printing success-like output.

These are intentional safety and ownership changes rather than accidental API drift.
