# sts2-mcp 🌌

[![English](https://img.shields.io/badge/lang-English-blue.svg)](#) [![Chinese](https://img.shields.io/badge/lang-中文-red.svg)](./README.zh-CN.md)

`sts2-mcp` is a high-performance, local control stack for **Slay the Spire 2**. It bridges the gap between the game's internal state and external AI agents using the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

Unlike traditional screen-scraping or OCR-based solutions, `sts2-mcp` uses a **native C# bridge mod** to extract precise game data and expose legal actions directly, ensuring 100% accuracy and sub-millisecond latency.

---

## 🏗️ Architecture

The project consists of two primary components:

### 1. `mods/sts2-bridge` (The "Sensors & Actuators")
A C#/.NET 9 mod that injects directly into the Slay the Spire 2 process.
- **State Serialization**: Converts complex in-game objects (Run, Combat, Rewards, Maps) into clean JSON.
- **Action Execution**: Directly invokes game methods to perform cards plays, selections, and navigation.
- **Discovery**: Automatically creates a session file for the MCP server to find and connect to.

### 2. `packages/mcp-server` (The "Interface")
A Node.js 22 server that implements the standard MCP.
- **Tool Mapping**: Translates bridge HTTP endpoints into standard MCP tools.
- **Safety**: Implements `state_version` guarding to prevent "stale" actions (e.g., trying to play a card that was already exhausted).
- **Optimization**: Handles complex batching for rewards, shops, and campfires to minimize LLM round trips.

---

## 🛠️ MCP Tools Reference

The server exposes the following tools to any MCP-compliant agent (like Claude Desktop or Antigravity):

| Tool | Description |
| :--- | :--- |
| `sts2_get_state` | Fetches the full current game state (Screen, Health, Deck, Relics, etc.). |
| `sts2_get_deck` | Returns the full master deck for lower-frequency strategic planning. |
| `sts2_list_actions` | Lists all currently legal actions available to the player. |
| `sts2_perform_action` | Executes a single action by its unique ID, with optional `return_state_after` for the full raw post-action bridge state. |
| `sts2_play_card_sequence` | Plays multiple cards in order, with automatic hand re-indexing. |
| `sts2_execute_combat_sequence` | Runs mixed combat sequences that can combine card plays, potion use, and `end_turn` in one call. |
| `sts2_resolve_room_rewards` | Claims all gold/potions and optionally picks a card in one call. |
| `sts2_resolve_rest_site` | Performs a rest/smith and proceeds back to the map. |
| `sts2_resolve_card_selection` | Handles card pick/skip/transform screens perfectly. |
| `sts2_pick_option` | Picks indexed reward/event/rest/card-selection options without raw action-id guessing. |
| `sts2_travel_to_coordinate` | Resolves cleanup, waits for a stable map snapshot, and then travels to a coordinate. |
| `sts2_resolve_shop_visit` | Buys multiple items and removes a card in a single batch. |
| `sts2_env_spec` | Describes the compact RL-oriented environment contract exposed by the RL branch, including observation layout, action encoding, and reward shaping. |
| `sts2_env_reset` | Starts a fresh standard run from the main menu and returns the first actionable RL observation plus legal actions. |
| `sts2_env_step` | Executes one RL step, waits for the next stable actionable or terminal state, and returns `obs/reward/done/truncated/legal_actions`. |

---

## 🚀 Getting Started

### Prerequisites
- **OS**: Windows (Slay the Spire 2 is currently Windows-only).
- **Runtime**: [Node.js 22+](https://nodejs.org/).
- **Game**: A legal copy of Slay the Spire 2.

### Option A: Using Pre-compiled Release (Recommended)
1. Download the latest `sts2-bridge` release (DLL and associated files) from the Releases page.
2. Create a folder named `sts2-bridge` inside your game's `mods` directory.
   - Example: `<PATH_TO_STS2>\mods\sts2-bridge`
3. Place the downloaded files into that folder.

### Option B: Building from Source
If you prefer to compile it yourself, you will also need the [.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0).
Set your game installation path and build the project:

```powershell
$env:STS2_DIR = "<PATH_TO_STS2>"
dotnet build .\mods\sts2-bridge\sts2-bridge.csproj
```
*Note: This will automatically copy the build output to the game's `mods\sts2-bridge` folder.*

### Next Steps...

1. **Launch the Game**: Run Slay the Spire 2. The bridge will initialize and create a session file at `%APPDATA%\SlayTheSpire2\bridge\session.json`.
2. **Start the MCP Server**:
```powershell
node .\packages\mcp-server\index.js
```

---

## 🗺️ Roadmap & TODO

- [ ] **Better Tools**: Expanding the MCP toolset for more granular state queries and complex action sequences.
- [ ] **Multiplayer Support**: Enabling agents to interact with or manage cooperative/multiplayer gameplay mechanics.
- [ ] **In-Game Dialogue Integration**: Injecting AI strategy text and reasoning directly into the game's dialogue boxes for a more immersive, visual experience.

---

## 📝 Configuration

The MCP server looks for the bridge session file automatically. You can override it via environment variables:
- `STS2_BRIDGE_SESSION_FILE`: Path to a custom session JSON.

Combat usage note:
- Do not parallelize consecutive combat plays with repeated `sts2_perform_action` calls.
- Use `sts2_play_card_sequence` for card-only turns.
- Use `sts2_execute_combat_sequence` when a turn mixes card plays, potion use, or `end_turn`.

---

## 🤖 Unified RL Pipeline

The RL branch now uses one checkpoint lineage across all training stages:

1. combat sandbox PPO
2. offline build/route pretraining
3. full-run PPO

The entry point is [packages/rl-agent/train_pipeline.py](./packages/rl-agent/train_pipeline.py). It passes the output checkpoint of each stage directly into the next one.

Example:

```powershell
python .\packages\rl-agent\train_pipeline.py `
  --dataset-root .\datasets\parquet `
  --character ironclad `
  --session-file "$env:APPDATA\SlayTheSpire2\bridge\session.json" `
  --stage2-partition-kind build_family `
  --stage2-partition-value v0.98_to_v0.99.1
```

If you do not pass `--stage1-encounter-pool`, stage 1 now defaults to a starter-deck-friendly early Act 1 weak pool derived from exported run history:

- `ENCOUNTER.SLIMES_WEAK`
- `ENCOUNTER.SHRINKER_BEETLE_WEAK`
- `ENCOUNTER.FUZZY_WURM_CRAWLER_WEAK`
- `ENCOUNTER.NIBBITS_WEAK`

The fixed eval holdout pool also defaults to unseen early Act 1 weak encounters:

- `ENCOUNTER.CORPSE_SLUGS_WEAK`
- `ENCOUNTER.SLUDGE_SPINNER_WEAK`
- `ENCOUNTER.SEAPUNK_WEAK`
- `ENCOUNTER.TOADPOLES_WEAK`

If you already have a sandbox checkpoint and want to skip stage 1:

```powershell
python .\packages\rl-agent\train_pipeline.py `
  --dataset-root .\datasets\parquet `
  --start-checkpoint .\pipeline_runs\some_run\stage1_sandbox\checkpoints\final `
  --stop-after offline
```

---

## ⚖️ Disclaimer & License

**Disclaimer**: This is an unofficial community project. It is not affiliated with, endorsed by, or associated with Mega Crit or the developers of Slay the Spire 2. Use at your own risk.

**License**: [MIT](./LICENSE)
