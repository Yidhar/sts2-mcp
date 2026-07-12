# MuZero / Search-free World Model

This folder now contains the STS2 MuZero family: the historical MCTS path is
still available for compatibility and ablations, but the active target is an
**attention-first / JEPA-style / search-free planner**.

Chinese version: [`README.zh-CN.md`](./README.zh-CN.md)

## Current position

- **Combat sandbox stays mandatory**: it is the high-throughput combat trainer.
  Full-run collection is too slow to replace it.
- **MCTS stays as compatibility / comparison path**: old checkpoints, old replay
  analysis, and build/route control experiments can still use it.
- **Combat is moving search-free**: `action_rollout_planner` runs shared dynamics
  inside the model and produces Q-like / lookahead value scores without online
  tree-search latency.
- **`token_memory_v1` is the target architecture**: hand, draw, discard, exhaust,
  potions, relics, energy/X-cost, HP/block, enemy buffs/intents/reactions, global
  deck/build state, and route state are tokenized into banked attention and
  persistent memory slots.
- **Mixed precision is part of the training path**: `--mixed-precision auto`
  uses CUDA bf16 when supported, otherwise CUDA fp16 + GradScaler; CPU auto stays
  fp32/off.  The token-memory default training window is now `--batch-size 32`
  and `--unroll-steps 3`; the older 128×5 fp32 setting is too activation-heavy.
- **Search-free rollout uses bucketed tensors**: `--action-rollout-buckets`
  defaults to `8,16,32,64,80`, so root legal actions and latent beam branches
  are padded to allocator-stable buckets before heavy dynamics/value heads run.
  Continuation pruning is also root-bucket grouped top-k, not a per-root
  `nonzero`/Python-list/`cat` loop.
  This is the ROCm/HIP-friendly Scheme-B path for reducing reserved-vs-allocated
  fragmentation without falling back to MCTS.

## Layout

```text
packages/rl-agent/muzero/
├── README.md
├── README.zh-CN.md
├── __init__.py
├── train.py                    # canonical module entrypoint; thin CLI/trainer compatibility wiring
├── evaluate.py                 # checkpoint evaluation
├── analyze_replay.py           # replay analysis
├── eval_latent_probes.py       # linear probes for HP/energy/piles/enemies/route/etc.
├── training/                   # orchestration/path/file-budget helpers; no combat policy logic
│   ├── paths.py                # RunPaths / StrategyModulePaths / HeuristicSearchModulePaths
│   └── file_budget.py          # 2,000-line per-file budget guard
├── strategy/                   # combat/card/potion/HP/X-cost strategy features and helpers
│   └── encounters/             # boss/enemy-specific mechanics split by encounter
├── combat_quality/             # no-pressure pure-block, end-turn, hard-guard, offender metrics
├── route_heuristics/           # route graph/path candidates/scoring/safety/bias
├── search/                     # root-prior/search glue; no STS2 card-specific policy logic
├── diagnostics/                # diagnostic dump schemas/helpers; runtime files stay under log_dir
└── sts2_env/
    ├── __init__.py
    ├── mcts.py                 # compatible MCTS / ablation path
    ├── muzero_buffer.py        # replay buffer / trajectories
    ├── muzero_model.py         # MuZeroNetwork + search-free rollout planner
    ├── token_memory.py         # token world memory encoder / banked attention / token dynamics
    ├── latent_regularizers.py  # JEPA/SIGReg-style latent Gaussian regularizers
    └── semantic_rollout.py
```

### File-governance rules

- Do not add new strategy or heuristic-search logic directly to `train.py`;
  keep it as a shrinking CLI/trainer wiring layer.
- New Python files must stay below 2,000 lines; if a file crosses 1,500 lines,
  split a submodule before adding more behavior.
- Runtime artifact paths (logs, checkpoints, diagnostics, replay buffers) go
  through `training.paths.RunPaths`.
- Strategy source paths go through `training.paths.StrategyModulePaths`:
  `strategy_file(...)`, `encounter_file(...)`, and `combat_quality_file(...)`.
- Route/search source paths go through
  `training.paths.HeuristicSearchModulePaths`: `route_heuristic_file(...)` and
  `search_file(...)`.
- Use `PolicyModulePaths` only as a temporary aggregate during migration.
- Guard command:

```bash
cd packages/rl-agent
: "${STS2_ARTIFACT_ROOT:?Set STS2_ARTIFACT_ROOT to an absolute external directory}"
"$STS2_ARTIFACT_ROOT/environments/wsl-rocm/bin/python" \
  scripts/check_muzero_file_budget.py --quiet-ok
```

Only explicitly documented historical budget exemptions are allowlisted; any new
over-budget file fails the guard.

## Main entrypoints

```bash
cd packages/rl-agent

# Training
python -m muzero.train --obs-mode token_v3 --model-arch token_memory_v1 \
  --mixed-precision auto --batch-size 32 --unroll-steps 3 ...

# Exact continuation: atomic manifest, every SHA-256, and all identities must match.
python -m muzero.train --resume-from <checkpoint_dir> ...

# New lineage from compatible weights only (optimizer/replay/counters are not loaded).
python -m muzero.train --resume-from <old_checkpoint_dir> --warm-start \
  --checkpoint-migration-id sts2-weights-only-v1 ...

# Explicit search-free combat direct policy
python -m muzero.train --combat-sandbox --combat-direct-policy \
  --combat-rollout-steps 3 --combat-rollout-beam-width 2 \
  --action-rollout-buckets 8,16,32,64,80 ...

# In auto mode, token_memory_v1 + combat sandbox defaults to direct policy.
# Force legacy MCTS for ablations with --combat-policy-mode mcts.

# Evaluation / replay analysis
python -m muzero.evaluate ...
python -m muzero.analyze_replay ...

# Latent probes: verify that latent state encodes HP/energy/piles/enemies/route
python -m muzero.eval_latent_probes --checkpoint <checkpoint_dir> \
  --markdown-output runs/latent_probe.md
```

See `docs/runbooks/checkpoint-resume.md`. Exact resume fails closed; legacy sources
also require `--allow-legacy-checkpoint` and are accepted only for weights-only
warm-start.

## Model and method summary

### 1. Token-world memory encoder

`token_memory_v1` does not treat the observation as one monolithic dense vector.
It splits the world into explicit token banks:

- `runtime`: player state, resources, hand, draw pile, discard pile, exhaust
  pile, play pile, cycle plan, energy budget
- `support`: relics, potions, support graph
- `enemy`: enemy core state, intent, powers/buffs, traits, reactions
- `build`: master deck, reward/shop/upgrade/transform options
- `route`: route summary, map nodes, route risk/value
- `powers`: power slots and card keywords
- `history`: recent action / turn summaries

Encoding flow:

1. self-attend world tokens;
2. self/cross-attend candidate action query + local action context;
3. self-attend candidate set;
4. run **explicit banked world cross-attention**: action queries route to
   runtime/support/enemy/build/route/powers/history banks before attending to
   the selected bank memory;
5. write into persistent memory slots, each with a bank identity;
6. flatten the memory slots into MuZero hidden state while also emitting action
   embeddings.

### 2. Shared text / numeric trunks

`world_embedder`, `query_embedder`, and `local_embedder` no longer repeat modal
projection work:

- `shared_text_trunk` handles token text vectors;
- `shared_numeric_trunk` handles token numeric vectors;
- `_project_shared_modal_trunk_with_reuse(...)` concatenates world/query/local
  rows, unique-deduplicates repeated vectors, projects once, and scatters the
  result back.

### 3. Dynamics + JEPA world modeling

The recurrent MuZero step predicts:

- next hidden state;
- reward / reward components;
- policy/value/objective heads;
- next legal surface / decision domain / phase;
- future world-bank state;
- future token-slot state/mask/type/zone/source;
- trainable `surprise`.

Training losses now include:

- `loss/jepa_next_hidden`: dynamics next hidden vs. representation(next_obs);
- `loss/latent_gaussian_reg`: Gaussian/isotropy latent regularization to reduce
  representation collapse;
- `loss/surprise`: hidden prediction error plus future-world/surface auxiliary
  miss, supervised into the surprise head;
- future world-bank/token-slot auxiliary losses so the latent state must carry
  piles, relics, potions, enemies, build state, and route state.

### 4. Search-free action rollout planner

`MuZeroNetwork.action_rollout_planner(...)` is the direct combat planning path:

1. run every real legal root action through shared dynamics;
2. use shared value/objective value heads for one-step Q;
3. use latent policy + predicted legal surface for multi-step beam continuation;
4. aggregate back to root actions:
   - `planner_q`
   - `planner_objective_q`
   - `planner_risk_q`
   - `planner_uncertainty`
5. keep allocator-stable shapes in the direct planner:
   - root legal actions are padded to `--action-rollout-buckets`;
   - latent branch dynamics/value calls are padded to the same bucket ladder;
   - continuation pruning uses root-bucket grouped top-k instead of per-root
     dynamic `nonzero`/`cat`.
6. compute uncertainty from:
   - trainable surprise
   - next-surface entropy
   - latent norm drift
   - branch Q disagreement

The combat direct policy blends Q/objective/risk bonuses and subtracts
uncertainty from policy logits, replacing online MCTS for combat when
`--combat-direct-policy` is enabled.

## Parameter counts

Current default parameter counts:

- `MuZeroNetwork(obs_mode="token_v3", model_arch="token_memory_v1")`: **23,084,242**
- `MuZeroNetwork(obs_mode="dense_v2", model_arch="dense_v1")`: **14,261,639**

## Compatibility policy

The refactor removed the legacy package and the public sts2_env MuZero wrapper
modules. Canonical implementation imports and module entrypoints now live under
muzero/. Missing old paths are intentional and fail fast.

The only retained top-level utility wrapper is analyze_muzero_replay.py, which
delegates to muzero.analyze_replay for operator compatibility. New code and
documentation must use these module entrypoints:

- python -m muzero.train
- python -m muzero.evaluate
- python -m muzero.analyze_replay
- python -m muzero.eval_latent_probes

## Training guidance

Start the new search-free work from combat sandbox:

- run `--combat-sandbox` (auto mode enables direct policy for token_memory_v1);
- watch:
  - `loss/jepa_next_hidden`
  - `loss/latent_gaussian_reg`
  - `loss/surprise`
  - `metric/surprise_target_offset`
  - `direct_rollout_uncertainty_mean`
  - `direct_rollout_branch_disagreement_mean`
- periodically run `eval_latent_probes.py` to confirm hidden/memory slots expose
  HP, energy, piles, enemy intent/power, support, build, and route information.
