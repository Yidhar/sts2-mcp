# RL agent contributor guide

This package is the architecture-v2 reinforcement-learning runtime for STS2. The
maintained training entry point is:

```text
python -m muzero.train
```

The deleted PPO, omni-attention trainer, top-level MuZero wrappers, low-memory
supervisor, and static-export client are not compatibility surfaces. Do not recreate
or call them. General-purpose attention/auxiliary modules that remain under
`sts2_env/` are dependencies of the active token-memory model, not evidence that the
old trainer is supported.

## Install and verify

Use the exact lock for the selected platform. For Windows/CPU development:

```powershell
python -m pip install -r requirements-bootstrap.lock
python -m pip install -r requirements-dev.lock
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
python -m ruff check sts2_rl launcher.py launcher_watchdog.py
python -m mypy sts2_rl
python -m pytest tests -q -p no:cacheprovider
python scripts/check_muzero_file_budget.py
python -m muzero.train --help
```

WSL/ROCm uses `requirements-wsl-rocm.txt` plus the exact external wheel artifacts
installed by `scripts/bootstrap_wsl_rocm.sh`. Never silently substitute a different
Torch/ROCm build.

## Runtime layout

```text
trainer/controller
  -> sts2_rl.EnvironmentBackend
       -> LiveBackend -> Bridge /v2/env/* with training capability
       -> HeadlessBackend -> pinned HeadlessSim process
  -> canonical TransitionFacts
  -> VersionedRewardCalculator
  -> replay / model / atomic checkpoint
```

- `sts2_rl/contracts/` owns typed reset, step, result, capability, and transition
  boundaries.
- `sts2_rl/backends/` owns live/headless protocol adapters and idempotent operation
  identity.
- `sts2_rl/reward/` is the canonical reward authority for v2 transitions.
- `sts2_rl/checkpoints/` owns atomic writes, checksums, completion markers, and
  provenance.
- `sts2_rl/training/` owns typed configuration and backend/service construction.
- `sts2_env/` contains the still-used Gym/observation/environment compatibility
  implementation while callers migrate to the typed boundary. It must not invent a
  second wire contract or reward authority.
- `muzero/` owns the maintained model, search/planner, replay, learner, collector,
  evaluation, diagnostics, and training CLI.

## Contract and capability rules

- Live v2 reset/step use a fresh UUID plus current `session_id` and the required
  expected state/step revision. Never guess a revision and never retry a timed-out
  mutation with a new request ID.
- Player-control and training tokens are independent. RL live environment operations
  use only the training token and loopback URL.
- Unknown API/schema/action/reward identities fail closed.
- Headless and live backends return the same typed result shape and canonical facts.
- The Bridge emits facts; this package alone calculates v2 reward. Legacy shaping may
  run only on an explicitly identified legacy result.
- Post-search tactical action rewriting is off by default. Structural legality,
  process safety, and explicit deadlock recovery are the only default runtime
  constraints; any opt-in override must record selected and executed actions.

## Artifacts and checkpoints

All runtime output belongs under `STS2_ARTIFACT_ROOT`, which must resolve outside the
source checkout. If the variable is absent, maintained entry points use the external
user default `~/.sts2-artifacts`; they must never silently create `runs/`,
`checkpoints/`, `logs_*`, `human_demos/`, datasets, or reports inside this package.
Relative CLI artifact paths resolve beneath that artifact root.

Checkpoints are directories committed atomically. A resumable checkpoint records at
least:

- model, optimizer, scheduler, replay, and global step;
- parent checkpoint identity and hashes;
- Git commit and dirty-state digest;
- contract, action-ordering, observation, reward, and game-data identities;
- Python/ROCm lock hashes and runtime/toolchain identity.

Missing or incompatible provenance fails closed. Do not infer compatibility from a
filename, directory name, or matching tensor count.

## Process lifecycle

- Launchers own only processes whose PID, creation marker, and executable identity all
  match the registry entry.
- Never use global `taskkill`, PID-only cleanup, or kill an unrelated process after PID
  reuse.
- Relay and child-process URLs remain loopback/allowlist constrained.
- Session descriptors, bearer tokens, Authorization values, or token prefixes must
  never appear in logs, exceptions, test snapshots, or diagnostics.
- Startup, reset, step, and shutdown all have bounded deadlines; failed children are
  killed and reaped only after ownership verification.

## Change discipline

1. Change the shared contract or game-data package first when the wire/data identity
   changes.
2. Add a fixture/characterization test before moving environment, observation, or
   tensor-ordering code.
3. Preserve legal-action ordering, token taxonomy/order, buffer layout, model state
   dict keys, and checkpoint tensor shapes during structural splits.
4. Keep source files below the enforced budget. Do not grow an allowlisted legacy
   giant; remove allowlist entries as files are decomposed.
5. Run focused tests, then the full suite, Ruff, strict mypy, dependency checks, and
   the file-budget gate.
6. State explicitly whether a real live-game Bridge and a real ROCm device were
   validated; unit/headless results are not substitutes.

See the repository-level `docs/architecture.md`, `docs/implementation-plan.md`,
`docs/migration/v2-cutover.md`, and `docs/runbooks/` for normative ownership, migration,
release, and operational procedures.
