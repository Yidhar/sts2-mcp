# RL agent contributor guide

This package contains the relational recurrent legal-candidate V-trace v4
baseline. The only maintained training entry point is:

```text
python -m sts2_rl.train
```

The old MuZero/token-memory/MCTS stack, PPO trainers, semantic/future-world
rollouts, objective heads, offline shadow losses, tactical guards and
game-specific policy rewrites were deleted. Do not recreate them behind a
compatibility facade.

## Install and verify

```powershell
python -m pip install -r requirements-bootstrap.lock
python -m pip install -r requirements-dev.lock
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
python -m ruff check sts2_rl sts2_baseline launcher.py launcher_watchdog.py
python -m mypy sts2_rl sts2_baseline
python -m pytest tests -q -p no:cacheprovider
python -m sts2_rl.train --dry-run
```

WSL/ROCm uses the externally pinned artifacts installed by
`scripts/bootstrap_wsl_rocm.sh`. Launch the live baseline with
`scripts/train_grounded_wsl_rocm.sh` and the headless native-revival preheat
with `scripts/train_preheat_wsl_rocm.sh`. Never silently substitute a
Torch/ROCm build or fall back to CPU.

## Runtime ownership

```text
sts2_rl.train
  -> typed LiveBackend or HeadlessBackend
  -> factual relational GroundedObservationEncoder
  -> RecurrentCandidateModel (exact relation grounding + run/combat GRU + masked policy + online value + long-horizon values)
  -> fixed, profile-selected sts2_baseline reward contract
  -> bounded FIFO SequenceUnroll queue
  -> VTraceLearner + bounded detached complete-episode replay
  -> atomic checkpoint
```

- `sts2_rl/contracts/` owns reset/step/result/capability boundaries.
- `sts2_rl/backends/` owns live/headless protocol adapters.
- `sts2_rl/encoding/` projects factual runtime state and candidates into
  definition, instance/relation, zone and numeric channels; it must not
  interpret boss/card/route strategy or fabricate future outcomes.
- `sts2_rl/models/` owns the candidate-independent world encoder and grounded
  candidate scorer.
- `sts2_baseline/` owns immutable, versioned task rewards plus sequence-unroll
  and FIFO contracts.
- `sts2_rl/training/` owns configuration, collection, learning, evaluation and
  checkpoint composition.
- `sts2_env/` is transport-only. It must not contain Gym environments,
  observations, reward shaping, tactical rules or learner models.

## Hard architecture invariants

- World encoding cannot read legal candidates.
- Candidate order and opaque action handles are not model features.
- Definition identity and concrete runtime identity are separate channels.
  Candidate source/target relations must bind to the corresponding world
  entity; modifiers, powers and intents must retain their factual owner.
- Known physical/decision zones use reviewed fixed IDs. Draw composition is an
  unordered player-visible set; hidden pile order is never a model input.
- Run-scale memory updates outside combat. Combat-scale memory updates inside
  combat and clears on exit; keep the combined recurrent tensor ABI stable.
- Only the authoritative environment legality mask can suppress an action.
- No MCTS, planner, root bias, action rewrite or policy distillation is allowed
  in the baseline.
- Each profile selects exactly one immutable reward identity. Backend scalars
  are never accepted as targets (the headless adapter strips them and may
  retain a diagnostic); objective vectors and settlement bonuses are rejected.
- Native-revival preheat may inject only the configured native revival relic.
  The simulator re-arms only that training copy after the game's native death
  hook/flash/50%-heal path. Revival and actual HP removed come from exact
  monotonic simulator counters under `observation._training`, never from HP
  increase inference or a hand-written combat policy. Those underscore facts
  are not model inputs.
- Reward never pays for damage dealt, enemy-HP change or cards played. The
  maintained preheat runs the complete headless game flow and ranks final run
  outcome plus forward run distance first, then bounded exact run-scoped
  HP-loss/revival costs and a small decision cost.
- `scripts/train_preheat_wsl_rocm.sh` launches the native-revival full game
  directly. Act boundaries are episode metrics, never behavior gates or
  curriculum truncations. The short binary/schema/runtime-mechanics preflight
  validates transport facts only; it does not score a policy.
- Forced singleton actions generate no policy target or policy-gradient term.
- Main V-trace data is consumed once in FIFO order. Two bounded,
  training-partition-only sidecars are maintained. Transaction replay stores
  factual select/deselect/confirm/cancel transitions. Complete-episode replay
  stores immutable CPU snapshots and authoritative combat/Act/run outcomes;
  it never retains GPU tensors, hidden states, autograd graphs, held-out
  diagnostics, or fabricated counterfactual actions.
- With complete-episode replay enabled, runtime may retain exactly one fetched
  FIFO batch so the episode is committed before its tail batch learns. The
  batch must be flushed before maintenance/final checkpointing and may never be
  consumed twice. Recorded collection must reject held-out evaluation seeds.
- Complete-episode replay reconstructs the current split-GRU state from an
  exact sparse prefix under `torch.no_grad()`. Both prefix and suffix forward
  passes use deterministic evaluation-mode dropout semantics, while only the
  configured short, contiguous suffix may enter autograd; the caller's model
  mode must always be restored. Source episode length therefore cannot grow
  accelerator activation memory. Replay is bounded by episode count, total
  bytes, per-episode bytes and per-episode sampling quota.
- Run completion/progress is the primary long-horizon objective. Revival cost
  may train value/policy only on a factually successful horizon. Before the
  task value classifies success and enters the explicit primary-tie tolerance,
  its already-weighted policy signal is capped below the absolute primary
  residual. Only inside that narrow learned-success stratum may a small
  nominal-primary floor preserve the revival tie-break at zero task advantage;
  failed/censored horizons still receive no cost label and early failure must
  never become the cheap option. Forced singleton actions receive value labels
  but no replay policy gradient.
- The recurrent model keeps one legal-candidate policy, the legacy online
  V-trace scalar value, and candidate-independent combat/Act/run task and
  revival-cost value heads. Optional transaction effect/delta/Q heads train the shared factual
  representation. Transaction liveness must also optimize normalized policy
  logits directly: completed unique factual steps are preferred and exact
  repeated semantic node/action cycles are avoided. Deselect stays legal; no
  action rewrite, prompt/card ID rule, or forced confirmation is allowed.
- Evaluation uses generic semantic state/action recurrence detection and writes
  diagnostic trajectories; diagnostics never become training samples.

## Contracts, artifacts and checkpoints

Live mutations require a fresh UUID, exact session identity and current
state/step revision. Never guess a revision or retry an unknown mutation under a
new identity. Player-control and training credentials remain separate and must
never appear in logs.

All mutable output lives below `STS2_ARTIFACT_ROOT`, outside the checkout.
Checkpoint publication is atomic and hashes learner model, actor model, optimizer,
the pending rollout queue, bounded transaction replay, bounded complete-episode
replay (when enabled), and metadata. Exact resume
rejects contract, reward projection, dependency lock, encoding, model/config or
payload drift; optional static catalog provenance is not a gate. A declared
`model_parameter_initialization` may inherit only shape-compatible network tensors
into a new lineage; optimizer, queue, RNG, counters, and both replay sidecars reset.
Older compatible checkpoints may initialize all shared network tensors while
the complete all-or-none long-horizon head group starts fresh; this is never
reported as exact resume.

## Change discipline

1. Preserve the world/candidate separation and permutation tests.
2. Add generic facts or tensor fields, never card/boss-specific branches.
3. Keep reward changes versioned in source; do not expose coefficient CLI flags.
4. Run focused tests, then full pytest, Ruff, strict mypy and CLI dry-run.
5. Report live-game/ROCm validation separately from synthetic tests.
6. Do not claim Act 1 performance without held-out long-run evaluation evidence.

See `docs/rl-grounded-baseline.md` for the normative architecture and curriculum.
