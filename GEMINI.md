# GEMINI.md - Project Context

## Project Overview
`sts2-mcp` is a high-performance, local control stack for **Slay the Spire 2 (STS2)**. It provides a bridge between the game's internal state and external AI agents using the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

The project consists of three main components:
1.  **`mods/sts2-bridge` (C#/.NET 9)**: A native mod that injects into STS2 to extract state (Run, Combat, Rewards, Maps) and execute actions (Card plays, selections, navigation).
2.  **`packages/mcp-server` (Node.js 22)**: A server that implements the MCP, translating bridge HTTP/JSON endpoints into standardized MCP tools for AI agents.
3.  **`packages/rl-agent` (Python)**: A Reinforcement Learning (RL) agent using Stable Baselines 3 (PPO). It features a unified three-stage training pipeline:
    *   Stage 1: Combat sandbox PPO.
    *   Stage 2: Offline build/route pretraining.
    *   Stage 3: Full-run PPO.

## Architecture
- **State Extraction**: 100% accurate, sub-millisecond latency extraction via direct memory/object access in C#.
- **Action Execution**: Direct invocation of game methods, bypassing UI/OCR limitations.
- **MCP Tooling**: Exposes tools for state retrieval (`sts2_get_state`), deck management (`sts2_get_deck`), action listing (`sts2_list_actions`), and batch action execution (`sts2_execute_combat_sequence`).
- **RL Environment**: A custom Gymnasium environment (`packages/rl-agent/sts2_env/env_v2.py`) that interacts with the bridge.

## Building and Running

### 1. `sts2-bridge` (Mod)
Requires [.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0) and the environment variable `STS2_DIR` set to your game's installation path.
```powershell
$env:STS2_DIR = "<PATH_TO_STS2>"
dotnet build .\mods\sts2-bridge\sts2-bridge.csproj
```
The build output is automatically copied to the game's `mods\sts2-bridge` folder.

### 2. `mcp-server` (Server)
Requires [Node.js 22+](https://nodejs.org/).
```powershell
# From packages/mcp-server
npm install
npm start
```

### 3. `rl-agent` (AI)
Requires Python 3.11+ and dependencies from `packages/rl-agent/requirements.txt`.
```powershell
# From packages/rl-agent
pip install -r requirements.txt
python train_pipeline.py --dataset-root ..\..\datasets\parquet --character ironclad
```

## Key Files and Directories
- `mods/sts2-bridge/Scripts/Entry.cs`: Mod entry point and initialization logic.
- `packages/mcp-server/index.js`: Core MCP server implementation.
- `packages/rl-agent/train_pipeline.py`: Main entry point for the three-stage RL training.
- `packages/rl-agent/sts2_env/env_v2.py`: The RL environment definition.
- `datasets/`: Storage for offline training data (parquet format).
- `schemas/`: Definitions for communication between the bridge and the server.

## Development Conventions
- **Accuracy First**: Rely on the bridge for state data; avoid OCR or screen-scraping.
- **Atomic Actions**: Prefer high-level, batchable actions (like `sts2_execute_combat_sequence`) to minimize LLM round trips and prevent state staleness.
- **Modularity**: Maintain strict separation between the game-level bridge, the interface-level MCP server, and the agent-level RL logic.
- **Safety**: Use `state_version` in the bridge/MCP to ensure actions are only applied to the state for which they were generated.
