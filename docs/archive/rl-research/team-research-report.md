# Team Research Report

> Update on 2026-03-16:
> Parts of this report have been superseded by live web verification and local
> build validation captured in `docs/research-notes.md`.
> In particular, the earlier uncertainty around mod installation path,
> entrypoint shape, native project structure, and the move from local build
> `v0.98.3` assumptions to live `v0.99.1` verification should now be read
> together with the newer notes before making implementation decisions.

Compiled: 2026-03-16
Research scope: Five parallel research agents investigated the technical landscape
for the sts2_mcp bridge project.

## Research Team Summary

| Agent | Domain | Confidence |
|-------|--------|------------|
| Agent 1 | STS2 Modding Ecosystem | Medium (training knowledge, no live web) |
| Agent 2 | MCP Protocol & TypeScript SDK | High (well-documented spec) |
| Agent 3 | Transport Layer & Security | High (established .NET patterns) |
| Agent 4 | Godot 4.5 + C# Mod Architecture | High (Godot docs + patterns) |
| Agent 5 | STS2 Game Internals | Medium (inference from STS1 + local evidence) |

All agents were limited to project docs and training knowledge (cutoff May 2025).
Live web access was not available. Findings that require Phase 0 validation are
marked with **[VALIDATE]**.

---

## 1. STS2 Modding Ecosystem

### Mod Loading Mechanism

STS2 ships with **first-party mod support** built by MegaCrit. This is NOT
GodotModLoader. Evidence:

- The game has a "Mods" menu in the main menu for enable/disable
- The game tracks mod identity in multiplayer ("Mod mismatch" in logs)
- `0Harmony.dll` ships with the game, confirming Harmony is a supported dependency
- MegaCrit published an official mod template on GitHub **[VALIDATE]**:
  `https://github.com/MegaCrit/slay-the-spire-2-mod-template`

### Mod Structure (Expected)

```text
MyModName/
  manifest.json        # mod metadata (ID, version, name, dependencies)
  MyModName.dll        # compiled .NET 9 C# class library
  (optional assets/configs)
```

**[VALIDATE]** Clone or inspect the official template to confirm exact structure.

### Mod Installation Path

Two candidate locations:

1. `<game_install>/mods/` (most likely)
2. `%APPDATA%/SlayTheSpire2/mods/` (possible alternative)

**[VALIDATE]** Check if `E:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2\mods\` exists.

### Public Modding API

MegaCrit provides a public API NuGet package for mod authors. Expected features:

- Mod entry point interfaces or base classes
- Type definitions for cards, relics, enemies, combat state
- Registration APIs for custom content
- Event hooks for game events

**[VALIDATE]** Search NuGet for the exact package name; inspect official template `.csproj`.

### Mod Entrypoint (Expected Pattern)

```csharp
// Option A: Inheritance
public class MyMod : SomeMegaCritBaseClass
{
    public override void OnInitialize() { }
}

// Option B: Attribute-based
[Mod("my.mod.id", "My Mod", "1.0.0")]
public class MyMod
{
    public void Initialize() { }
}
```

**[VALIDATE]** This is the single most critical Phase 0 unknown.

### Limitations

- Multiplayer mod checking is active -- bridge mod may need "client-side only" flag
- Game updates during early access can break mod compatibility
- Mods load at startup; no hot-reload
- Must target .NET 9 and GodotSharp 4.5.1

---

## 2. MCP Protocol & TypeScript SDK

### Protocol Overview

- Version: `2025-03-26` (latest known)
- JSON-RPC 2.0 based client-server model
- Three primitives: **Tools** (our focus), Resources, Prompts
- Transport: **stdio** (recommended for our use case)

### Server Implementation Pattern

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "sts2-mcp",
  version: "0.1.0",
});

server.tool(
  "sts2_get_state",
  "Get the current visible game state",
  { scene_filter: z.string().optional().describe("Filter to a specific scene type") },
  async ({ scene_filter }) => {
    const state = await bridgeClient.getState();
    return {
      content: [{ type: "text", text: JSON.stringify(state, null, 2) }],
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

### Tool Design Best Practices

- Use `snake_case` with `sts2_` prefix
- Define input schemas with Zod (auto-converted to JSON Schema)
- Use `.describe()` on every field
- Return structured JSON as text content
- Use `isError: true` for logical errors (bridge down, stale state)
- **Never log to stdout** -- use stderr (`console.error`)

### Tool Annotations (New in 2025-03-26)

```typescript
server.tool("sts2_perform_action", description, schema, {
  destructive: true,
  idempotent: false,
  readOnlyHint: false,
  openWorld: false,
}, handler);
```

### Error Handling (Two Levels)

1. **Protocol errors**: JSON-RPC errors (handled by SDK)
2. **Tool errors**: Return `{ isError: true, content: [...] }`

Recommended error schema:
```json
{
  "error": "stale_state",
  "message": "State version 1842 is stale; current version is 1845",
  "bridge_connected": true,
  "state_version": 1845
}
```

### Client Integration (Claude Code)

```json
{
  "mcpServers": {
    "sts2": {
      "command": "node",
      "args": ["E:/game/project/sts2_mcp/packages/mcp-server/dist/index.js"],
      "env": {
        "STS2_BRIDGE_URL": "http://127.0.0.1:27100"
      }
    }
  }
}
```

### Recommended MCP Server Layout

```text
packages/mcp-server/
  src/
    index.ts              # entry point
    server.ts             # McpServer setup, tool registration
    bridge-client.ts      # HTTP client for bridge
    tools/
      get-state.ts
      list-actions.ts
      perform-action.ts
      end-turn.ts
      bridge-status.ts
    types.ts
  package.json
  tsconfig.json
```

---

## 3. Transport Layer & Security

### Loopback HTTP (Recommended for v0)

| Property | Detail |
|----------|--------|
| Binding | `http://localhost:{port}/` -- no admin needed on Win10+ |
| Firewall | **Loopback traffic is exempt from Windows Firewall** -- no popup |
| Latency | 0.2-1.5ms for small JSON payloads |
| Debuggability | Easy with curl, PowerShell, browser tools |
| Backend | `System.Net.HttpListener` backed by `http.sys` kernel driver |

### Threading Model (Critical)

**All Godot scene tree and node operations are main-thread-only.**

Recommended pattern for HTTP handler accessing game state:

1. `HttpListener` runs in background `Task` via `GetContextAsync()` loop
2. Request handler enqueues work item with `TaskCompletionSource<T>`
3. Mod's `_Process()` (main thread, every frame) drains `ConcurrentQueue`
4. Handler awaits the `Task` completion

```text
HTTP Thread                     Main Thread (_Process)
    |                                |
    |--enqueue(request, TCS)-------->|
    |                                |--read game state
    |                                |--TCS.SetResult(response)
    |<--await TCS.Task---------------|
    |--send HTTP response            |
```

### Named Pipes (Upgrade Path)

| Property | Named Pipes | Loopback HTTP |
|----------|-------------|---------------|
| Latency | 0.05-0.3ms | 0.2-1.5ms |
| Port conflicts | None (namespace-based) | Possible |
| Admin needed | No | No (localhost) |
| Debuggability | Harder | Easy |
| Node.js client | Requires `net` module | Built-in `fetch` |

Decision: **Start with HTTP, upgrade to pipes only if friction arises.**

### WebSockets (Phase 2+ Enhancement)

- Push model for real-time state change notifications
- Keep HTTP for request/response; add WS for events
- Not needed for v0 of a turn-based game

### Security: Token Exchange

**Recommended: File-based exchange (v0)**

Bridge writes at startup:
```text
%APPDATA%/SlayTheSpire2/bridge/session.json
```

```json
{
  "port": 27100,
  "token": "kT9x...",
  "pid": 12345,
  "bridge_version": "0.1.0",
  "started_utc": "2026-03-16T..."
}
```

Token generation:
```csharp
var bytes = RandomNumberGenerator.GetBytes(32);  // CSPRNG
var token = Convert.ToBase64String(bytes)
    .Replace('+', '-').Replace('/', '_').TrimEnd('=');
```

File ACL set to current-user-only. MCP server reads file at startup.

### Idempotency: Double Guard

1. **state_version check** (runs first): Reject with 409 if caller's version is stale
2. **request_id dedup** (runs second): Return cached response if same UUID seen

| Scenario | HTTP Status | Body |
|----------|-------------|------|
| Action executed | 200 | `{ok: true, state_version: N+1}` |
| Duplicate request_id | 200 | Cached response |
| Stale state_version | 409 | `{ok: false, error: "stale_state"}` |
| Invalid action | 422 | `{ok: false, error: "invalid_action"}` |
| Bridge not ready | 503 | `{ok: false, error: "not_ready"}` |

### MCP Server Reconnection State Machine

```text
DISCONNECTED --[session.json found, PID alive]--> CONNECTING
CONNECTING   --[GET /health OK]-----------------> CONNECTED
CONNECTED    --[connection error]---------------> DISCONNECTED
```

- Backoff: 1s, 2s, 4s, 8s, cap at 30s
- On reconnect: reset all cached state
- Health check interval: 5-10s proactive ping

---

## 4. Godot 4.5 + C# Mod Architecture

### Scene Graph Access from C#

```csharp
// Get scene tree from anywhere
SceneTree tree = (SceneTree)Engine.GetMainLoop();

// Current active scene
Node currentScene = tree.CurrentScene;

// Traverse children
foreach (Node child in node.GetChildren()) { }

// Find by path or name
var target = node.GetNode<T>("path/to/child");
var found = node.FindChild("name", recursive: true, owned: true);

// Find by group
var nodes = tree.GetNodesInGroup("group_name");

// Read properties dynamically
var value = node.Get("property_name");
```

### Scene Detection

**Polling approach (simplest, fine for v0):**
```csharp
public override void _Process(double delta)
{
    string sceneName = GetTree().CurrentScene?.Name;
    Type sceneType = GetTree().CurrentScene?.GetType();
    // Compare against last known scene to detect transitions
}
```

**Harmony approach (more reliable, for later):**
Patch the game's scene manager transition method for exact change events.

### Harmony Usage in STS2

`0Harmony.dll` ships with the game. Use for:

- **State change hooks**: Postfix patches on `StartTurn()`, `EndTurn()`, `PlayCard()` etc.
  to increment `state_version` and capture events
- **Scene transitions**: Patch scene manager for reliable detection

Do NOT use for:
- Bulk state reading (prefer direct reads from public APIs/scene graph)
- Anything that can be done through the public mod API

### MonoMod.Backports

- Part of the MonoMod ecosystem
- Underlies modern Harmony's method detour mechanism
- Already loaded in the game process
- **Do not bundle duplicate copies** -- reference the game's versions

### Best Practices for Mod Stability

- Wrap all mod code in try-catch at top level (unhandled exceptions crash the game)
- Keep `_Process()` work minimal -- only process when pending requests exist
- Initialize HTTP listener in `_Ready()`, clean up in `_ExitTree()`
- Use `CancellationToken` for async loop shutdown
- Version-check the game at startup, fail loudly if unsupported
- Isolate game-specific type access behind an adapter interface

---

## 5. STS2 Game Internals

### STS1 to STS2 Migration Map

| STS1 (Java) | STS2 (C#/Godot) | Role |
|-------------|-----------------|------|
| `AbstractDungeon` | Unknown (autoload singleton?) | Overall run state |
| `AbstractPlayer` | Unknown | Player entity |
| `AbstractMonster` / `MonsterGroup` | Unknown | Enemy state |
| `AbstractCard` | Unknown | Card base class |
| `CardGroup` | Unknown | Hand, draw, discard, exhaust piles |
| `AbstractRoom` | Unknown | Current room/encounter |
| `MapRoomNode` | Unknown | Map navigation |
| `AbstractRelic` / `AbstractPotion` | Unknown | Inventory items |
| `RewardItem` | Unknown | Reward entries |

### Expected Architecture Patterns

- **Manager singletons** (Godot autoloads) for combat, run, map state
- **Node-based entity hierarchy** for enemies, cards, UI
- **Data-driven design** using Godot Resources (`.tres`/`.res`) for card/relic definitions
- **Intent system** for enemy behavior with enum for attack types

### Decompilation Strategy

| Tool | Best For |
|------|----------|
| **ILSpy** (recommended) | Full .NET 9 support, free, C# decompilation |
| **dnSpyEx** | Live debugging + decompilation, attach to running process |
| **dotPeek** | Good decompilation, integrates with Rider |
| **ilspycmd** | CLI bulk namespace/class extraction |

### Phase 0 Decompilation Workflow

1. Open `sts2.dll` in ILSpy
2. Export full namespace tree
3. Search for singletons (static instance properties, autoload patterns)
4. Search for "Combat", "Battle", "Card", "Enemy", "Map", "Reward", "Event", "Choice"
5. Look for Harmony attributes or patch classes
6. Look for mod loading code (interface/base class that mods implement)

---

## 6. Cross-Cutting Findings & Consensus

### Architecture Validation

All five research threads **confirm the architecture document's decisions are sound**:

- Two-process design (bridge mod + MCP server) is correct
- Loopback HTTP for v0 is the right choice
- Visible-state-only default is appropriate
- The five planned MCP tools are well-scoped

### Key Technical Risks Identified

| Risk | Severity | Mitigation |
|------|----------|------------|
| Unknown mod entrypoint format | **High** | Inspect official template / decompile sts2.dll |
| Main-thread-only Godot API access | **High** | Use ConcurrentQueue + _Process() pattern |
| Game update breaks Harmony hooks | Medium | Isolate hooks behind adapter; version check at startup |
| HttpListener URL reservation on some Windows configs | Low | Use `http://localhost:port/` (default ACL) |
| Multiplayer mod mismatch rejection | Low | Mark as client-only mod if category exists |

### Highest Priority Phase 0 Actions

Ordered by impact:

1. **Decompile `sts2.dll` with ILSpy** -- find mod loader, battle manager, state classes
2. **Clone/inspect MegaCrit's official mod template** -- confirm entrypoint, manifest, API package
3. **Locate mods directory** on disk -- check game install and user data
4. **Examine `RemoveMultiplayerPlayerLimit` mod** -- learn real mod structure
5. **Check exact Harmony and MonoMod versions** in `sts2.deps.json`
6. **Verify MCP SDK version** with `npm view @modelcontextprotocol/sdk version`

---

## 7. Information Gaps

The following require live investigation (web access or game directory inspection):

- Exact mod template repository URL and contents
- Exact NuGet API package name for STS2 modding
- Exact manifest.json schema
- Actual namespace/class hierarchy in sts2.dll
- Current MCP SDK version and any post-May-2025 changes
- Whether STS2 community has a modding wiki or Discord
- Whether Steam Workshop integration exists for v0.98.3

These gaps do not block architecture decisions but DO block code scaffolding.
Phase 0 must resolve at minimum items 1-3 before writing bridge mod code.
