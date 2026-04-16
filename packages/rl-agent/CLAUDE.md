# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An RL training system for Slay the Spire 2. A Gymnasium environment communicates with the game via HTTP through the sibling `mcp-server` bridge mod. The active training path uses a search-free omni-attention policy (`STS2OmniAttentionPolicy`) trained with `AuxMaskablePPO` (masked PPO with 6 auxiliary supervision heads).

## Commands

```bash
# Install (no setup.py/pyproject.toml — flat module structure)
pip install -r requirements.txt

# Run tests
pytest tests/ -v
pytest tests/test_observation_v3.py -v                  # single file
pytest tests/test_observation_v3.py::TestClass::test_fn  # single test

# Train (primary active path)
python train_attention_policy.py \
  --total-timesteps 200000 --n-envs 4 --collector-mode async \
  --combat-sandbox --snapshot-pool /path/to/snapshots \
  --device cuda --amp --amp-dtype bf16

# Evaluate a checkpoint
python evaluate_attention_policy.py checkpoints_attention/step_NNN \
  --episodes 10 --combat-sandbox --device cuda --deterministic

# Launch parallel game instances
python launcher.py --instances 4 --game-exe "E:/SlayTheSpire2.exe"

# TensorBoard
tensorboard --logdir logs_attention
```

Training runs in WSL with ROCm via `.venv-wsl-rocm/` and the `run_longtrain_*.sh` shell scripts. Windows venv is `venv/`.

## Architecture

### Data flow

```
STS2 Game Instance(s)  <-- HTTP -->  Gymnasium Env  -->  AsyncReadyCollector  -->  AuxMaskablePPO
       (bridge mod)                 (combat_env.py       (N worker threads)      (omni_attention_
                                     or env_v2.py)                                policy.py)
```

### Environments

Two Gymnasium environments wrap the bridge HTTP API:

- **`SlayTheSpire2EnvV2`** (`env_v2.py`) — full-run episodes (map traversal, shops, events, combat)
- **`CombatSandboxEnv`** (`combat_env.py`) — combat-only via `/env/combat_reset` with snapshot parameters (deck, relics, encounter). This is the primary training environment.

Both produce dict observations and action masks. Legal actions come from the bridge; `MaskablePPO` enforces masking during sampling.

### Observation system

`observation_v3.py` (`WorldTokenObservationEncoder`) encodes game state as a fixed-size token sequence:

- **320 world tokens** across roles (hand cards, enemies, relics, deck, route nodes, etc.)
- **24 candidate local tokens** for action-specific context
- Each token: 96-d numeric features + 64-d text features = 160-d
- 66 token types, with owner/role/zone/order metadata fields
- Constant: `OBSERVATION_API_VERSION = "attention_obs_v2"`

The base class `DenseObservationEncoder` in `observation_common.py` defines `MAX_ACTIONS = 127` and shared encoding utilities.

### Neural architecture

`STS2OmniAttentionPolicy` (extends `MaskableActorCriticPolicy`):

1. **EntityTokenEmbedder** — projects numeric + text features, adds type/owner/role/zone/order embeddings
2. **5 World Banks** (TransformerEncoderBlocks) — `runtime`, `support`, `enemy`, `build`, `route`; tokens are routed to banks by role/zone IDs defined in `WORLD_BANK_ROLE_IDS`/`WORLD_BANK_ZONE_IDS`
3. **Cross-attention** between banks
4. **CandidateDecoder** — attends world context to candidate local tokens
5. **Policy head** (masked categorical) + **value head** + **6 auxiliary heads**

Building blocks live in `attention_blocks.py`: `TransformerEncoderBlock`, `CrossAttentionBlock`, `CandidateDecoderBlock`, `EntityPooling`, `RelationBias`.

### Auxiliary supervision

`AuxMaskablePPO` (`aux_maskable_ppo.py`) extends SB3's `MaskablePPO` with 6 auxiliary loss terms built by `aux_targets.py`:

| Head group | Count | Examples |
|---|---|---|
| objective | 3 | immediate_reward, damage_dealt, defense_value |
| transition | 8 | next_player_hp_ratio, next_enemy_hp_ratio, ... |
| traits | 8 | energy_line, potion_line, enemy_risk_line, ... |
| build | 8 | frontload_fit, defense_fit, scaling_fit, ... |
| selection | 6 | source_hand, source_draw, target_mine, ... |
| route | 8 | safe_value, elite_value, rest_value, ... |

Each has an independent loss coefficient (`--aux-*-coef` flags).

### Data collection

`AsyncReadyCollector` (`async_ready_collector.py`) runs N environments in separate threads. Workers push ready observations to a shared queue; the main thread batches predictions and dispatches actions. Handles watchdog timeouts and automatic environment restarts.

### Content registry

`content_registry.py` merges game metadata from `content/*.json` files (cards, enemies, relics, potions). Three layers: `*.generated.json` < `*.static.generated.json` < curated `*.json`. Provides `get_card_metadata()`, `get_enemy_metadata()`, `build_live_*_semantic_text()`.

### Combat snapshots

`CombatSnapshotPool` (`combat_snapshot_dataset.py`) loads offline encounter datasets (JSONL/Parquet) for curriculum-based combat training. Supports sampling modes: `uniform`, `encounter_balanced`, `tier_weighted_encounter_balanced` with configurable tier weights.

### Checkpoints

Saved as `model.safetensors` + `metadata.json` in versioned directories. `checkpoint.py` handles save/load with forward-compatible state dict loading and checkpoint rotation. Metadata includes `observation_api_version`, `collector_mode`, and full policy kwargs for reproducibility.

### Bridge client

`BridgeClient` (`bridge_client.py`) reads session files (`session_*.json` with `base_url` + `token`) from `%APPDATA%\SlayTheSpire2\bridge\`. Supports WSL path rewriting. Multi-instance training uses `session_0.json` through `session_N.json`, one per game process.

### Text encoding

`TextEncoder` (`text_encoder.py`) wraps `sentence-transformers` (model: `BAAI/bge-small-zh-v1.5`, 512-d). Thread-safe singleton with persistent disk cache + in-memory cache. Deduplicates batch encoding per step.

### Semantic actions

`semantic_action.py` defines a vocabulary of action families (play_card, use_potion, end_turn, map, shop, smith, etc.) with semantic roles (attack, block, draw, scaling, etc.) independent of bridge action ordering. `SEMANTIC_ACTION_DIM = 49`.

## Key constants

| Constant | Value | Location |
|---|---|---|
| `MAX_ACTIONS` | 127 | `observation_common.py` |
| `MAX_WORLD_TOKENS` | 320 | `observation_v3.py` |
| `MAX_CANDIDATE_LOCAL_TOKENS` | 24 | `observation_v3.py` |
| `TOKEN_FEAT_DIM` | 160 (96+64) | `observation_v3.py` |
| `NUM_TOKEN_TYPES` | 66 | `observation_v3.py` |
| `ENTITY_HASH_BUCKETS` | 8192 | `observation_v3.py` |
| `SEMANTIC_ACTION_DIM` | 49 | `semantic_action.py` |
| `RUN_MEMORY_DIM` | 48 | `run_memory.py` |
| `TEXT_DIM` | 512 | `text_encoder.py` |

## Environment variables

| Variable | Purpose |
|---|---|
| `STS2_BRIDGE_SESSION_FILE` | Override default bridge session file path |
| `STS2_BRIDGE_INSTANCE_ID` | Instance index (0, 1, 2, ...) for multi-instance training |
| `PYTORCH_ALLOC_CONF` | Set `expandable_segments:True` for long training runs |

## Important patterns

- Env wrapping chain: `CombatSandboxEnv` -> `ActionMasker(mask_fn)` -> `Monitor`
- Model composition: `AuxMaskablePPO(STS2OmniAttentionPolicy)`
- Reward shaping is defined inline in `combat_env.py` and `env_v2.py` (enemy HP deltas, player HP loss, end-turn waste penalties)
- Observation encoding is the most complex subsystem — `observation_v3.py` is ~3000 lines. Token type assignment, owner/role/zone routing, and numeric feature packing are tightly coupled
- The `observation_common.py` base class is ~1500 lines of shared logic (action encoding, decision context, pile encoding)
- `RunMemoryTracker` (`run_memory.py`) maintains persistent deck/relic/potion/floor context across steps within an episode
- Mixed precision: `--amp --amp-dtype bf16` uses `torch.autocast` in the forward pass with gradient scaling
- Tests mock the bridge and text encoder; they don't require a running game instance
