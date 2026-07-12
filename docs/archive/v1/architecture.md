# Architecture

## Scope

This document describes the recommended architecture for a local MCP integration
with Slay the Spire 2.

The design target is a narrow, reliable control plane that can read current
state and execute legal actions while staying inside the boundaries of what a
human player can normally observe.

## Assumptions

These assumptions are based on local inspection of the installed game build and
current community ecosystem indicators:

- The installed game is a Godot + C# build
- The game was built with `Godot 4.5.1 Mono`
- The runtime targets .NET 9
- The game natively supports mods
- The current community workflow is local loading through the install `mods`
  directory rather than Steam Workshop
- Community mods already exist for the game
- The game already tracks mod mismatch in multiplayer logs
- The machine already has `dotnet 9`, `node 22`, and `npm`

Any assumption that later proves false should be captured as an architecture
decision record and the bridge design should be updated accordingly.

## Design Goals

- High reliability in the presence of UI animation and window movement
- Clear separation between game integration and MCP presentation
- Safe defaults that do not expose hidden state
- Deterministic command execution with idempotency protection
- Local-only transport and easy debugging

## Design Constraints

- The bridge must coexist with the retail game binary
- The bridge should tolerate ordinary game updates failing closed
- The system must not depend on a foreground window or fixed resolution
- The MCP surface should not leak arbitrary code execution into the game process
- The first version should optimize for developer velocity over maximum elegance

## Mod Packaging And Entry Point

The current tutorial ecosystem makes the packaging contract concrete enough to
design around.

### Packaging Shape

The bridge should be built as a native Slay the Spire 2 mod project:

- a Godot C# project
- a project metadata file named `<modid>.json`
- a compiled `dll`
- an exported `pck`

Current implementation note:

- the repository keeps `project.godot` for later `pck` export work
- the first bridge DLL is currently compiled as a plain `.NET 9` class library
  because the current code path does not require compile-time Godot APIs

The effective deployed layout is:

```text
<Sts2Dir>/
  mods/
    sts2-bridge/
      sts2-bridge.dll
      sts2-bridge.pck
```

The tutorial indicates that `dll` and `pck` are the core runtime artifacts of a
mod. For `sts2_mcp`, this means the bridge must be treated as a real game mod,
not merely a loose helper assembly.

### Minimal C# Entry Point

The minimal initializer pattern is:

```csharp
[ModInitializer("Init")]
public class Entry
{
    public static void Init()
    {
        var harmony = new Harmony("sts2.bridge");
        harmony.PatchAll();
    }
}
```

Implications for this project:

- the bridge can start without reverse-engineering a custom boot path
- Harmony is available when patches become necessary
- tutorial examples mention `ScriptManagerBridge.LookupScriptsInAssembly(...)`,
  but the live local build should only use it if the symbol actually resolves
  during compilation
- the current scaffold omits that call because the bridge can build and start
  without it, while the symbol itself does not currently resolve from the local
  reference set

### BaseLib Position

`BaseLib` is useful for adding new content, configuration UI, and unified
registration behavior, but the current public tutorial explicitly notes that it
is optional for patch-only mods.

That means:

- `BaseLib` is out of scope for the first bridge milestone
- `sts2_mcp` should only introduce `BaseLib` later if it materially improves
  non-bridge features

## Recommended Topology

```text
Codex
  -> stdio MCP
MCP server (Node/TypeScript)
  -> loopback HTTP
Bridge mod (C# inside game process)
  -> game state readers and action executors
Slay the Spire 2
```

### Why Two Processes

Keeping the MCP server outside the game process has several benefits:

- MCP dependencies stay decoupled from the game runtime
- Crashes in the external server do not take down the game
- Tool output formatting and schema evolution are easier in Node/TypeScript
- The bridge mod can stay focused on state extraction and action execution

## Build And Debug Workflow

The current tutorial suggests a straightforward native workflow:

- reference `sts2.dll` and `0Harmony.dll` from the game's
  `data_sts2_windows_x86_64` directory
- use `.csproj` output rules to copy the built `dll` into the game's `mods`
  directory
- export a matching `pck` from Godot into the same mod folder
- use a `launch_*.bat` file with `--log` for local debugging
- place `steam_appid.txt` with `2868840` in the game root for local runs when
  needed
- use the in-game console opened with `~` for manual testing

For `sts2_mcp`, this should be the standard development loop until a better
automation path is proven.

## Transport Decision

### Recommended For v0: Loopback HTTP

Use loopback HTTP bound to `127.0.0.1` with a per-session bearer token generated
by the bridge mod at startup.

Do not require the external MCP server to know the bridge port in advance.
Instead, the bridge should write a session registry file such as:

- `%APPDATA%\\SlayTheSpire2\\bridge\\session.json`

That file should carry the current `port`, `base_url`, `token`, `pid`,
`session_id`, and `bridge_version`.

Why this is the recommended first choice:

- Easy to inspect with curl, PowerShell, or browser tools
- Easy to consume from Node without native Windows dependencies
- Easy to version and log
- Already supported by the installed runtime surface

The external MCP server should treat the session file as the discovery source
of truth and the health endpoint as the liveness check.

### Possible Later Upgrade: Named Pipes

If loopback HTTP produces Windows firewall friction or if tighter local-only
semantics become important, move the bridge transport to named pipes. The MCP
server contract should remain stable so only the adapter layer changes.

## Trust Model

The external MCP server is trusted. The model using the tools is not fully
trusted.

That implies:

- The MCP server must validate all action parameters
- The bridge mod must also validate them and reject invalid calls
- The bridge mod should never expose raw object method invocation
- All mutating actions should require the caller to provide a recent
  `state_version`

## Visibility Policy

Default mode is `visible-state-only`.

Allowed:

- Current HP, block, energy, statuses
- Current hand contents and playability
- Current enemies, visible intents, HP, block, statuses
- Draw pile count, discard pile count, exhaust pile count
- Visible rewards, map nodes, shop inventory, event choices
- Potions, relics, deck list, upgrade state when shown or inherently public

Not allowed in default mode:

- Future draw order
- Internal random seeds
- AI data not surfaced to the player
- Hidden encounter tables
- Developer-only debug state

A later debug mode may expose more, but it should be off by default and clearly
marked in the tool output.

## Domain Model

The bridge should normalize game state into a small schema family rather than
dumping raw object graphs.

### Core State Envelope

```json
{
  "bridge_version": "0.1.0",
  "game_version": "v0.99.1",
  "session_id": "5ce8...",
  "state_version": 1842,
  "scene": "combat",
  "timestamp_utc": "2026-03-16T00:00:00Z",
  "visible_only": true
}
```

### Combat State

```json
{
  "combat": {
    "turn": 3,
    "phase": "player_turn",
    "player": {
      "hp": 52,
      "max_hp": 71,
      "block": 7,
      "energy": 2,
      "statuses": [
        { "id": "strength", "name": "Strength", "amount": 2 }
      ]
    },
    "enemies": [
      {
        "id": "enemy_0",
        "name": "Fuzzy Wurm",
        "hp": 18,
        "max_hp": 26,
        "block": 0,
        "intent": {
          "kind": "attack",
          "value": 7,
          "hits": 1
        },
        "statuses": []
      }
    ],
    "hand": [
      {
        "instance_id": "card_14",
        "card_id": "strike",
        "name": "Strike",
        "cost": 1,
        "upgraded": false,
        "playable": true,
        "targeting": "single_enemy"
      }
    ],
    "draw_pile_count": 8,
    "discard_pile_count": 4,
    "exhaust_pile_count": 0
  }
}
```

### Choice State

The same top-level schema should support rewards, events, campfires, map
choices, shop choices, and card picks:

```json
{
  "choice": {
    "kind": "card_reward",
    "prompt": "Choose a card",
    "options": [
      { "id": "opt_0", "label": "Shrug It Off" },
      { "id": "opt_1", "label": "Pommel Strike" },
      { "id": "opt_skip", "label": "Skip" }
    ]
  }
}
```

## Action Model

The bridge should expose explicit legal actions, not generic UI gestures.

### Canonical Action Types

- `play_card`
- `end_turn`
- `choose_option`
- `select_map_node`
- `use_potion`
- `open_reward`
- `skip_reward`
- `proceed`
- `cancel`

### Legal Action Enumeration

The bridge should always provide a `legal_actions` list derived from current
state. Each action entry should include:

- `action_id`
- `kind`
- required arguments
- human-readable summary
- whether a target is required
- any obvious failure reason if currently disabled

Example:

```json
{
  "legal_actions": [
    {
      "action_id": "play:card_14",
      "kind": "play_card",
      "card_instance_id": "card_14",
      "requires_target": true,
      "valid_targets": ["enemy_0"]
    },
    {
      "action_id": "end_turn",
      "kind": "end_turn",
      "requires_target": false
    }
  ]
}
```

## Idempotency And Concurrency

The biggest operational risk is double-executing an action after the state has
already advanced. To prevent that:

- Every mutable call must include `state_version`
- The bridge rejects calls if `state_version` is stale
- Every mutable call should accept a client-generated `request_id`
- The bridge records recently executed `request_id` values and returns the prior
  result if the same request is replayed

## Bridge API Sketch

### Read-only

- `GET /health`
- `GET /state`
- `GET /actions`
- `GET /events?after=<seq>`

### Mutating

- `POST /actions/play-card`
- `POST /actions/end-turn`
- `POST /actions/choose-option`
- `POST /actions/use-potion`
- `POST /actions/proceed`

### Example Request

```json
{
  "state_version": 1842,
  "request_id": "b35b6450-0d56-4fa6-8f5e-d2b90ec8700d",
  "card_instance_id": "card_14",
  "target_id": "enemy_0"
}
```

### Example Response

```json
{
  "ok": true,
  "accepted": true,
  "state_version": 1843,
  "summary": "Played Strike on Fuzzy Wurm for 6 damage"
}
```

## MCP Tool Mapping

The external MCP server should expose a stable, small tool surface:

### `sts2_get_bridge_status`

Returns:

- whether the game process is connected
- bridge version
- game version
- visible-only mode
- last observed scene

### `sts2_get_state`

Returns normalized state for the active scene.

### `sts2_list_actions`

Returns current `legal_actions` and short summaries.

### `sts2_perform_action`

Executes a supplied action by `kind` plus validated arguments.

### `sts2_end_turn`

Thin convenience wrapper around `sts2_perform_action`.

## Scene Coverage Priority

### Must-Have For v0.1

- Main menu detection
- Map choice
- Combat
- Card reward
- Generic choice screen
- Continue/proceed transitions

### Nice-To-Have After v0.1

- Shop
- Campfire
- Potion usage outside combat if allowed
- Event-specific structured parsing
- Multiplayer read-only diagnostics

## Failure Modes And Recovery

Expected failures:

- Bridge mod not loaded
- Game updated and hook points changed
- Current scene not yet supported
- Action rejected because state advanced
- Action rejected because target or option is no longer valid

Required behavior:

- Return structured errors
- Include current scene and state version in every error
- Never attempt a best-effort fallback click inside the game window

## Logging

Both layers should log.

Bridge mod logs:

- startup and shutdown
- session token creation
- current game version
- unsupported scene warnings
- action acceptance and rejection

MCP server logs:

- bridge connectivity
- tool calls
- request and response summaries
- latency

Sensitive data such as session tokens must never be logged in plaintext.

## Testing Strategy

### Bridge Tests

- Unit tests for state normalization
- Contract tests for action validation
- Manual smoke tests against a live game instance

### MCP Tests

- Schema validation tests
- tool integration tests with mocked bridge responses
- live smoke tests against a local bridge process

### End-to-End Tests

- start game
- verify bridge detected
- read state in map scene
- enter combat
- play a safe card
- end turn
- choose a reward

## Open Questions

- Which existing public APIs inside `sts2.dll` are sufficient for state reads
  and action execution before patching anything?
- Which classes own battle state, map state, and reward state in `sts2.dll`?
- Which parts of the bridge can be implemented with public registration or model
  APIs versus Harmony patches?
- Is HTTP acceptable long-term, or will named pipes be necessary on Windows?

## Recommended Next Step

Do not start by building the full MCP server. Start by proving that a tiny
native mod can load and return a single JSON payload for current scene and turn
state. That is the main technical risk reducer.
