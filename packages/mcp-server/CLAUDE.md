# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A stdio MCP server (`index.js`, ~14k lines, zero dependencies) that bridges AI agents to a running Slay the Spire 2 instance via an in-game C# bridge mod. The server speaks newline-delimited JSON-RPC over stdin/stdout and proxies game state and actions through HTTP to the bridge mod.

## Commands

```bash
# Run the MCP server (stdio transport)
node index.js

# Run the smoke test (spawns server as child, exercises initialize + tool calls)
node smoke-test.js

# Run the AutoSlay monitor/launcher
node autoslay-runner.js              # start AutoSlay and monitor
node autoslay-runner.js --monitor-only
node autoslay-runner.js --fresh-run  # abandon current save first
```

No build step, no bundler, no transpiler. The server is a single CommonJS file with only Node.js built-in dependencies.

## Architecture

### Single-file server (`index.js`)

Everything lives in one file with no external npm dependencies. The major sections, in order:

1. **Constants** (lines 1-63) — timing, polling intervals, settle windows, topic enums, profile names
2. **Tool definitions** (`TOOL_DEFINITIONS` array, lines 64-846) — JSON Schema for every MCP tool
3. **Tool profiles** (lines 848-894) — `minimal`, `strategic`, `debug` subsets of tool names; selected via `STS2_MCP_TOOL_PROFILE` env var
4. **Stdio transport** (lines 896-961) — newline-delimited JSON-RPC reader/writer on stdin/stdout, with end-turn batching via `dequeueNextMessageForExecution`
5. **Message dispatch** (`handleMessage` / `handleToolCall`, lines 1059-1210) — routes JSON-RPC methods (`initialize`, `tools/list`, `tools/call`) to tool handlers
6. **Tool handlers** (lines 1257-3700) — one `async function *Tool(args)` per tool: `getBridgeStatus`, `getStateTool`, `performActionTool`, `runCombatSequenceTool`, `resolveRoomRewardsTool`, `resolveRestSiteTool`, `resolveCardSelectionTool`, `resolveShopVisitTool`, `travelToCoordinateTool`, `waitForChangeTool`, `waitUntilActionableTool`, etc.
7. **Bridge HTTP client** (lines 3668-3812) — `bridgeRequestJson`, `performBridgeAction`, session file reading, auth headers
8. **Post-action settlement** (lines 3812-4700) — state polling and stability detection after actions; different strategies for end-turn, combat actions, screen transitions, map travel
9. **Card/potion/shop action resolution** (lines 4700-6900) — re-matching planned actions against live state after hand reindex or draw changes; fingerprinting for stable identity
10. **Agent text formatting** (lines 6900-7700) — `summarize*ForAgent` functions that produce compact text payloads for LLM consumption
11. **State compaction** (lines 7800-10900) — `compactPayloadForOutput`, `summarizeStateForAgent`, `buildDeckPayloadForAgent`, `buildMapRoutesPayload`
12. **Bridge event client** (`BridgeEventClient` class, lines 11021-11270) — SSE stream consumer for real-time state updates from the bridge
13. **HTTP transport** (lines 11270-11460) — `bridgeRequestJson`, `readJsonResponseBody`, timeout handling
14. **Journal system** (lines 12000-12650) — per-run markdown journal with write/read/summarize/list tools
15. **Observation system** (lines 12650-13100) — provenance-tagged entity observations (cards, relics, events, enemies)
16. **Knowledge system** (lines 13100-13852) — strategy knowledge base with topic resolution, section parsing, search, and slice reads

### Key classes

- **`BridgeEventClient`** — SSE client that maintains a persistent connection to the bridge's `/events` endpoint, caches latest state, and resolves waiters when state changes
- **`ToolPayloadError`** — structured error with `code` and `details` for agent-readable error payloads
- **`BridgeHttpError`** — HTTP-level bridge communication errors

### Bridge discovery

The server does not hardcode a port. It reads `session.json` written by the in-game mod:
- Default: `%APPDATA%\SlayTheSpire2\bridge\session.json`
- Override: `STS2_BRIDGE_SESSION_FILE` env var

### Tool profiles

Set `STS2_MCP_TOOL_PROFILE` to one of: `minimal`, `strategic`, `debug` (default exposes all tools). Each profile is a subset of `TOOL_DEFINITIONS`.

### Combat sequence rematching

The `runCombatSequenceTool` function is the most complex handler. When executing a multi-step combat sequence (`play_card` + `use_potion` + `end_turn`), it:
1. Plans all steps against the initial hand state
2. Executes step-by-step, re-fetching state after each action
3. Rematches remaining planned steps against the new hand (cards may have shifted indices after draws/exhausts)
4. Uses fingerprinting (`buildPlayCardActionFingerprint`, `buildUsePotionActionFingerprint`) for stable identity across reindexing

### Post-action settlement

After executing bridge actions, the server waits for the game to reach a stable state. Different action types use different settlement strategies (`getPostActionSettleStrategy`):
- **end_turn**: polls until enemy intents resolve and new hand is drawn
- **combat actions**: waits for card resolution animations
- **screen transitions**: waits for the new screen to stabilize
- **map travel**: waits for map snapshot consistency

### Diagnostics log

Written to `%APPDATA%\SlayTheSpire2\bridge\mcp-stdio.log` (override: `STS2_MCP_LOG_FILE`). Records process start, stdin chunks, parsed requests, and response summaries.

## Environment Variables

| Variable | Purpose |
|---|---|
| `STS2_BRIDGE_SESSION_FILE` | Override bridge session file path |
| `STS2_MCP_LOG_FILE` | Override MCP diagnostic log path |
| `STS2_MCP_TOOL_PROFILE` | Tool profile: `minimal`, `strategic`, or `debug` |

## Supporting files

- **`smoke-test.js`** — spawns the server as a child process, runs initialize handshake, calls `sts2_get_bridge_status` / `sts2_get_state` / `sts2_list_actions`, prints results
- **`autoslay-runner.js`** — spawns the server, optionally starts the game's built-in AutoSlay, monitors state until terminal (victory/defeat/timeout)
- **`knowledge/`** — markdown strategy guides (card tiers, boss patterns, route planning, etc.) served by `sts2_get_knowledge` and related tools

## Transport

Stdio, newline-delimited JSON-RPC. One JSON object per line. No `Content-Length` framing.

## Important patterns

- All tool handlers return via `asToolResult(payload, isError)` which wraps in `{ content: [{ type: "text", text: JSON.stringify(payload) }], isError }`
- The server prioritizes non-end-turn messages to avoid race conditions (end-turn batching window: 150ms)
- `state_version` and `state_hash` provide idempotency guards — callers can pass `expected_state_version` to reject stale actions
- Combat sequence tools (`sts2_play_card_sequence`, `sts2_execute_combat_sequence`) should always be preferred over parallel `sts2_perform_action` calls for consecutive combat plays
