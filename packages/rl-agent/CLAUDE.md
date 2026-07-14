# RL agent contributor guide

This package contains the recurrent grounded legal-candidate V-trace v2
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
  -> structural GroundedObservationEncoder
  -> RecurrentCandidateModel (GRU + masked policy + scalar value)
  -> fixed, profile-selected sts2_baseline reward contract
  -> bounded FIFO SequenceUnroll queue
  -> VTraceLearner
  -> atomic checkpoint
```

- `sts2_rl/contracts/` owns reset/step/result/capability boundaries.
- `sts2_rl/backends/` owns live/headless protocol adapters.
- `sts2_rl/encoding/` structurally hashes raw state and candidates; it must not
  interpret boss/card/route strategy.
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
- Training data is consumed once in FIFO order; replay sampling, PER and
  long-lived sample retention are forbidden.
- The recurrent model has exactly a legal-candidate policy head and one scalar
  value head. Q, reward-prediction and terminal-prediction heads are forbidden.
- Evaluation uses generic semantic state/action recurrence detection and writes
  diagnostic trajectories; diagnostics never become training samples.

## Contracts, artifacts and checkpoints

Live mutations require a fresh UUID, exact session identity and current
state/step revision. Never guess a revision or retry an unknown mutation under a
new identity. Player-control and training credentials remain separate and must
never appear in logs.

All mutable output lives below `STS2_ARTIFACT_ROOT`, outside the checkout.
Checkpoint publication is atomic and hashes learner model, actor model, optimizer,
the pending rollout queue and
metadata. Exact resume rejects contract, reward projection, dependency lock,
encoding, model/config or payload drift; optional static catalog provenance is not a
gate. Old model checkpoints are not migration inputs.

## Change discipline

1. Preserve the world/candidate separation and permutation tests.
2. Add generic facts or tensor fields, never card/boss-specific branches.
3. Keep reward changes versioned in source; do not expose coefficient CLI flags.
4. Run focused tests, then full pytest, Ruff, strict mypy and CLI dry-run.
5. Report live-game/ROCm validation separately from synthetic tests.
6. Do not claim Act 1 performance without held-out long-run evaluation evidence.

See `docs/rl-grounded-baseline.md` for the normative architecture and curriculum.
