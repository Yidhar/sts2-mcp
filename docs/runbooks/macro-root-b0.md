# B0: same-action macro-root reconstruction

Status: standalone diagnostic implemented. This is a prerequisite experiment,
not a change to the recurrent V-trace learner. It does not implement long-range
lambda returns, counterfactual action comparison, or distillation.

## Question and current-course fit

Can a fixed full-run seed and a frozen greedy semantic policy reconstruct a
card-reward root, including its observable world, legal candidates, run/combat
memory and stateful reward-v9 history? Does executing the same action and a
bounded identical continuation reproduce the same evidence?

The current course is [hybrid recurrent V-trace](../architecture/hybrid-recurrent-vtrace-recovery-v1.md),
not the archived IQL candidate-Q learner. This probe therefore uses the current
hybrid actor-critic for inference, the production semantic compiler/executor,
and reward v9. Scores are **policy logits**, never candidate Q values.

The runner is `python -m sts2_rl.macro_root_diagnostic`, from the RL package
directory. Its three source modules are:

- `sts2_rl/macro_root_diagnostic.py`: bounded two-process runner and evidence journal.
- `sts2_rl/macro_root_policy.py`: frozen publication loading / explicit IQL initialization.
- `sts2_rl/macro_root_evidence.py`: exact JSON comparison and read-only reward state.

## What actually runs

1. Validate the full-run, native-revival-preheat, run-objective course and the
   pinned simulator identity. No live backend is allowed.
2. Load one fixed publication in evaluation/inference mode. There is no learner,
   optimizer, training queue, replay store, target network, or publication writer.
3. Start an isolated HeadlessSim process; fresh-reset the full run with the
   explicit seed and configured revival budget. Replay reward v9 from reset,
   including the Act-local revival ordinal. Do **not** reset it at the macro root.
4. Execute the frozen greedy policy over the production semantic decision graph.
   Mechanical picker/confirm suffixes do not add recurrent policy forwards.
   Unsupported strategic surfaces, encoding collisions and executor failures
   fail closed; no legacy strategic fallback is used.
5. Identify the requested occurrence of a macro `reward` surface with `take` and
   `skip` alternatives. Execute its selected action plus the configured number
   of subsequent semantic decisions. Stop before dispatch at the next meaningful
   frontier, after mechanical suffixes have settled. That frontier is scored
   for recurrent evidence but its action is **not executed**.
6. Close the process. Start a second isolated process with a fresh calculator
   and fresh recurrent state. Re-execute the same prefix. Compare each action
   and its current dispatch binding **before** issuing it; compare each resulting
   transition and reward. Stop on the first mismatch, without retry/fallback.

Strict-group indices are not native legal-action indices. Every actual step
resolves the current group's representative and its current handle/position.
Handles from the reference process are never used to dispatch in the replay.

## Evidence and failure policy

Each pass writes a JSONL stream, retaining only one reference row at a time:

- complete observable DTOs (including `_sim_raw` and `_training`), ordered legal
  actions, strict groups, semantic candidates, selected branch/target/plan;
- native step index and typed transition facts;
- pre/post semantic recurrent memory and candidate policy logits;
- reward-v9 breakdown, calculator state, and native durable-floor discount.

Facts compare exactly, including numeric types, entity instance IDs, array order,
unknown fields and multiplicity. Only policy float arrays use `atol=1e-6` and
`rtol=1e-5`; shape, dtype labels, selected actions and non-finite values are not
excused by those tolerances.

Transport identity keys are excluded **only at the known observation root,
observation `_sim_raw` root, and typed transition root**. The exact key list and
the three scopes are saved in the summary. Identically named fields inside an
entity, `_training`, action candidate or transition facts remain evidence.
The pure `canonical_facts()` helper never deletes keys. Never widen normalization
to generic ID/hash/timestamp suffixes to make a failing probe pass.

Outputs are exclusively in a **new child directory of `STS2_ARTIFACT_ROOT`**,
outside the checkout. Existing output directories are rejected, not overwritten.

| Summary status | Interpretation |
| --- | --- |
| `matched` | Both bounded passes reached the root and matched through the requested next frontier, or a factual full-run terminal after the root. |
| `mismatch` | First evidence divergence; summary includes row, section and JSON path. Actual mismatching evidence remains in replay JSONL. |
| `inconclusive` | No root, unknown environment truncation, or a native-step/time/trace-byte budget. This is **not** task failure and not a zero-bootstrap target. |
| `failed` | Publication, simulator, protocol, semantic executor, non-finite inference or another infrastructure/contract error. |

The CLI returns 0 only for `matched`; all other summary statuses return 2.
Invalid CLI/config/output-path arguments are rejected before a run starts.
All acquired backend instances are closed in `finally` blocks. Budgets are per
pass; wall time excludes model loading and process construction, whose existing
transport startup/RPC timeouts still apply. The cooperative wall check cannot
preempt a running model kernel; it checks again before dispatch and after a step.

`native_floor_discount` is evidence, not a new learning time scale. It uses the
existing durable-floor contract (`0.997 ** max(delta_floor, 0)`, terminal zero).
No native-step gamma, lambda target, tail-value estimate, or branch ranking is
computed by B0.

## Running one bounded probe

Use the project's existing environment with Torch, NumPy and the RL package.
The host's default Python is not assumed to contain the training dependencies.
For WSL, set `STS2_ARTIFACT_ROOT` to its **WSL absolute path**, not a Windows
drive spelling. The pinned HeadlessSim executable is resolved by the existing
resolver, or supplied explicitly via `--sim-exe`.

From `packages/rl-agent`:

```bash
# Configure these externally; do not put machine-specific paths in source.
# STS2_ARTIFACT_ROOT: the existing runtime artifact root
# PYTHON: the existing Torch-enabled interpreter
# IQL_PARENT: the healthy policy-100 publication/network.pt

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -m sts2_rl.macro_root_diagnostic \
  --config config/experiments/hybrid_vtrace_v1_reward_v9_budget64_on_policy.toml \
  --initialize-from-iql "$IQL_PARENT" \
  --device cpu \
  --seed 13400001 \
  --root-occurrence 1 \
  --continuation-decisions 2 \
  --max-native-steps 256 \
  --timeout-seconds 150 \
  --output-dir "$STS2_ARTIFACT_ROOT/analysis/macro-root-b0-example-new"
```

For an actual current-course V-trace publication, replace
`--initialize-from-iql "$IQL_PARENT"` with `--model-path "$VTRACE_PUBLICATION"`.
The two arguments are mutually exclusive. Device selection is explicit; no
automatic CPU fallback is offered by the CLI.

The old IQL publication is only an **in-memory model initialization**:
representation, actor, value and split memory weights are inherited; the old
Q head is ignored. Missing destination weights are rejected. The source reward
v8 identity and destination reward v9 identity are separately recorded, as are
`exact_resume=false` and `training_state_loaded=false`. Policy version 100 is
source provenance, not a claim that V-trace has performed 100 updates.

Seed 13400001 is now consumed diagnostic material, not a future final-audit seed.
Two deterministic reconstructions of it are not two independent statistical
samples. Do not promote roots or candidates from one seed into separate
train/validation/test partitions.

## Initial bounded observation (2026-09-05)

The healthy IQL policy-100 parent transferred all 238 actor-critic tensors, with
no fresh tensors and only the four obsolete candidate-Q tensors ignored.
Using CPU, budget 64, seed 13400001, and the first card-reward root:

- The root was semantic decision 22 on durable floor 2, after 22 native steps.
- Candidates were TREMBLE, TWIN_STRIKE, INFERNAL_BLADE and skip; greedy inference
  selected TREMBLE. **This is not a judgement that TREMBLE is the best card.**
- The root action and two subsequent semantic decisions reached an event frontier
  on floor 3. Each pass dispatched 26 native steps and performed 25 semantic
  forwards (the final forward was diagnostic-only, without dispatch).
- The bounded two-process smoke matched, including a rerun after tightening the
  transport scope. Its reference and replay JSONL files were byte-identical
  (stronger than the configured policy tolerance, for this one seed only).
  The implementation also has
  focused tests for mismatches, budgets, mechanical suffixes, dispatch binding,
  transport scope, reward reconstruction and backend cleanup.

The final-code evidence is under the runtime artifact root, in
`analysis/macro-root-b0-20260905-smoke03`: `matched`, 55 evidence rows per pass,
and byte-identical 2,824,285-byte reference/replay traces. Earlier integration
runs `analysis/macro-root-b0-20260905-smoke01` and
`analysis/macro-root-b0-20260905-smoke02` are retained, not overwritten.
Inspect `summary.json` for actual status; do not infer a pass from the existence
of a directory. These reruns reuse one seed and add no independent sample count.

The final focused test run passed 181 tests: 104 new B0 tests and 77 related
existing actor-critic/authority/V-trace/pipeline/reward regressions. Ruff passed
for all seven new Python files, and strict mypy with silent dependency imports
passed for all three production modules. No formal training was started.

This first root had zero revivals. It does not empirically cover rebuilding
after a death or across an Act boundary; those require additional targeted
roots even though the stateful reward implementation and unit checks are reused.
It also has no production mechanical suffix in the observed short continuation;
the suffix contract is covered by a focused smith/select/confirm regression.

## What this does not establish / next gate

Matching observable facts does not prove equality of hidden simulator RNG,
provide an exact full-run clone API, or prove correctness for untested roots.
The summary explicitly sets `hidden_rng_clone_verified=false`. B0 supplies no
evidence yet that a different root action produces a better policy outcome.

The next bounded experiment can compare different candidates **at roots whose
same-action control has passed**, keeping the same frozen continuation policy
and clearly defined horizon/censoring. It should remain offline diagnostic
evidence initially. Broad-root coverage, revival/Act-boundary reconstruction,
branch determinism and uncertainty/seed partitioning must be settled before
designing long-range V-trace-compatible credit or actor distillation. Do not
reintroduce an IQL replay/target-Q learner merely to use this diagnostic.
