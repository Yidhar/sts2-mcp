# Synchronous macro-credit training: operational gates

Status: **next bounded reward-root training pilot is ready**, based on real GPU
updates, independent-process cohort-boundary resume, held-out bounded execution,
and the final checkpoint/publication/forward audit on 2026-09-05. This limited
readiness is not value-calibration certification, policy-quality promotion,
simulator-clone certification, or approval for large unattended long training.

## Ownership and why the old asynchronous launcher is not used

The new course freezes one teacher while collecting a complete cohort, performs
one bounded-gradient optimizer update, and immediately commits a checkpoint.
Only then may its derived actor+V publication advance. It has no asynchronous
spool, episode-boundary-only publication reload, or unbounded wait-for-both-domain
GPU batch inherited from the old hybrid V-trace launcher.

The independent publication format is
`sts2-hybrid-macro-credit-publication-v1`, not a V-trace learner publication.
Actor logits and candidate-independent scalar value retain the model ABI.

## Before starting

- All mutable files must resolve below the external `STS2_ARTIFACT_ROOT`, never
  inside the checkout. The run directory has one writer. Source initialization
  files are read-only; use a distinct checkpoint and publication destination.
- Record the complete effective course configuration, not only overrides. This
  includes seed recipe/partitions, cohort and optimizer budgets, simulator and
  dependency identities, encoding/compiler versions, model configuration,
  immutable reward-v9 identity, learner version and lineage.
- No omitted field may silently fall back to a changed default on continuation.
- Teacher evidence, training samples and held-out evaluation outputs remain
  separate. Observable B0 reconstruction is not certification of simulator
  hidden RNG or a universal state clone.
- Measure peak RAM/VRAM for the actual concurrent course, including any evaluation
  worker. Bound candidate size, autograd suffix/microbatch, total cohort bytes,
  root/branch rollout budget and wall-clock time before long collection.
- Define execution-failure gates separately from measured quality/promotion
  thresholds. A completed evaluation process is not itself a quality pass.

## Commit and recovery protocol

`sts2_rl.joint.macro_checkpoint` owns the cohort checkpoint format and atomic
publication. The checkpoint stores full course/model/reward/lineage/learner
identity, model and optimizer, committed update, next seed index, initialization
provenance, and Python/numpy/Torch CPU/all visible CUDA RNG states.

1. Save an update-zero boundary before collecting the first cohort.
2. Keep teacher parameters fixed while collecting a cohort.
3. On a successful bounded update, save its new `update` and `next_seed_index`
   immediately. A cohort that aborts must not advance the committed cursor.
4. File content is flushed and fsynced before same-directory `os.replace`;
   POSIX also fsyncs the directory after rename. Failure before replacement
   leaves the previous checkpoint intact. A failure after replacement is a
   commit/durability uncertainty: inspect the checkpoint, never blindly repeat
   an update based on the publication alone.
5. Rebuild publication from the committed checkpoint, replacing a missing or
   stale cache. The checkpoint is always recovery authority.

Resume strictly rejects differing full identity, model tensors, optimizer
parameter layout/hyperparameters, non-finite values and runtime environment.
Python/numpy/Torch versions, selected CPU/CUDA device, accelerator count/names
and CUDA/ROCm runtime must agree. CPU-to-GPU migration is a new initialization,
not exact continuation. Matching this metadata still does not promise bitwise
GPU determinism.

Only the **cohort boundary** is recoverable. No mid-run simulator snapshot,
partial cohort, action-in-flight, world/reward/recurrent state, asynchronous
spool or live simulator RNG is serialized. On interruption, discard the
uncommitted work and re-collect from the previous committed seed cursor and
random streams. Course-owned numpy `Generator` instances must be regenerated
from the recorded per-run seed recipe; the protocol restores global numpy RNG,
not arbitrary hidden application objects.

## Stop and readiness gates

- A cooperative stop either completes the cohort/update/commit, or explicitly
  aborts its uncommitted work and retains the preceding boundary. Never label
  an unfinished update or budget-exhausted cohort `completed`.
- On startup and resume, reject active competing writers and inspect the last
  committed cursor before producing new outputs.
- Focused tests must cover save/load plus the next optimizer step, all random
  streams, malformed optimizer non-mutation, identity/runtime mismatch,
  non-finite rejection, source protection, precommit write failure, and stale
  publication reconstruction.
- Before declaring the next training ready, attach real short-course evidence:
  a finite actual optimizer update, publication reload, bounded memory/time,
  interruption/recovery at a cohort boundary, and a separate held-out execution
  check. Synthetic tests alone do not satisfy this gate.

## Executable and bounded command

The explicit entry point is `python -m sts2_rl.train_macro_credit`; its independent
configuration is `config/experiments/hybrid_macro_credit_v1_reward_v9.toml`.
The current stage collects **reward-surface roots**, with every executable root
candidate and 12 post-root macro decisions by default. It is not an all-surfaces
coverage claim. Natural macro and combat rows must both exist before learning.
There is no epsilon mixture: the frozen teacher samples its policy distribution.

Example from the RL package directory in the pinned WSL environment:

```bash
export STS2_ARTIFACT_ROOT=/mnt/e/game/project/sts2_mcp_artifacts/runtime
export HSA_ENABLE_DXG_DETECTION=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PYTHON="$STS2_ARTIFACT_ROOT/environments/wsl-rocm/bin/python"
RUN="$STS2_ARTIFACT_ROOT/macro_credit_runs/first-bounded-course"
# IQL_SOURCE must name the reviewed external, policy-logits IQL publication.
"$PYTHON" -m sts2_rl.train_macro_credit \
  --config config/experiments/hybrid_macro_credit_v1_reward_v9.toml \
  --output-dir "$RUN" --lineage-id macro-credit-first \
  --device cuda --initialize-from-iql "$IQL_SOURCE" --max-updates 1
```

`--max-updates N` is a limit on **additional committed updates in this invocation**,
not a silently changed training objective. To continue this same course:

```bash
"$PYTHON" -m sts2_rl.train_macro_credit \
  --config config/experiments/hybrid_macro_credit_v1_reward_v9.toml \
  --output-dir "$RUN" --lineage-id macro-credit-first \
  --device cuda --resume "$RUN/checkpoint.pt" --max-updates 1
```

Initialization and resume are mutually exclusive. New initialization refuses
any existing output directory. Resume requires that directory's own checkpoint.
No `auto` device selection or CPU fallback exists. `--set dotted.key=TOML_VALUE`
only accepts known fields; its effective values are part of strict resume
identity. Changing a learning/collection budget creates a new course rather
than silently resuming the old one.

Add `--dry-run` for static, read-only config/path/device/simulator-identity
preflight. It does not load source weights, construct a learner, start a
simulator, create output files, or claim training readiness. A real bounded
training completion reports `quality_certified=false` and
`long_run_certified=false`; a separate readiness report must evaluate those
remaining gates.

Training seeds are exactly `2 * (6700000 + next_seed_index)`. A successful
cohort advances the cursor by `roots_per_cohort`; a partial cohort does not.
Held-out odd seeds and existing diagnostic result files are never imported.

Disk size/free-space checks run at most every 0.5 seconds during trace writes
and are forced at root/update boundaries. Single-trace byte limits are strict;
the total-directory gate is cooperative and can overshoot within one checking
interval. Time limits are also cooperative around simulator operations and
learner chunks, not preemption of an individual GPU kernel. A known exhausted
budget prevents checkpoint commit; an uncommitted learner object is discarded.

`source_manifest.json` records hashes of an explicit selection of course,
learner, checkpoint and direct model/semantics/encoding source files, plus Git
HEAD as provenance. It is marked **partial**, not a complete dependency lock.
Selected code hashes enter strict identity; tests/docs and Git HEAD do not, so
adding validation reports alone does not invalidate continuation.

Ctrl-C aborts uncommitted work and preserves the last boundary. A hard kill can
leave `.writer.lock`; verify the old process and simulator are stopped before
manually removing that stale lock. Do not remove it while a writer is active.
Resume preserves aborted attempt evidence and creates a new attempt directory.

## Measured launch evidence

### GPU optimizer update and independent-process resume: passed

Experiment lineage: `macro-credit-v1-20260905-smoke01`.
External run directory:
[macro-credit-v1-20260905-smoke01](E:/game/project/sts2_mcp_artifacts/runtime/hybrid_runs/macro-credit-v1-20260905-smoke01).
The default 12-post-root-macro horizon was retained; this was not shortened into
a transport-only dry run. Both cohorts contain one complete four-candidate
reward root and natural rows from both domains.

| Recorded metric | Initial process: update 0 → 1 | Independent resume process: update 1 → 2 |
| --- | ---: | ---: |
| Training seed | 13400000 | 13400002 |
| Teacher version / policy lag | 0 / 0 | 1 / 0 |
| Natural macro / combat steps | 20 / 121 | 17 / 67 |
| Backward chunks / maximum graph steps | 9 / 16 | 6 / 16 |
| CPU cohort snapshot bytes | 5,953,008 | 3,931,062 |
| Finite parameter-delta L2 norm | 0.1084993430 | 0.08279072995 |
| Maximum teacher-likelihood reconstruction error | 2.8277472e-7 | 3.7299614e-7 |
| Joint loss | 0.5980033565 | 0.5242816410 |
| Learner elapsed seconds | 19.748962 | 14.667777 |
| Learner CUDA peak allocated bytes | 317,266,432 | 357,869,056 |
| Learner CUDA peak reserved bytes | 354,418,688 | 379,584,512 |
| Process RSS historical high-water bytes | 1,940,942,848 | 1,993,113,600 |

The two authoritative learner/coverage records are
[update 1](E:/game/project/sts2_mcp_artifacts/runtime/hybrid_runs/macro-credit-v1-20260905-smoke01/cohorts/update-000000-attempt-c958399ab9694bfe9065bc8dd4fe6481/update.json)
and
[update 2](E:/game/project/sts2_mcp_artifacts/runtime/hybrid_runs/macro-credit-v1-20260905-smoke01/cohorts/update-000001-attempt-ffc3e23fbaab49aeb44a3469f4f651b9/update.json).
Separate invocation completion is recorded in
[initial-update.txt](E:/game/project/sts2_mcp_artifacts/runtime/analysis/macro-credit-readiness-20260905/initial-update.txt)
and
[resume-update.txt](E:/game/project/sts2_mcp_artifacts/runtime/analysis/macro-credit-readiness-20260905/resume-update.txt).

**Resource measurement scope:** CUDA peak statistics are reset inside the
learner. The values above are learner-update-window peaks, not complete
collection-plus-learning course GPU peaks. RSS is a process-lifetime historical
high-water value, not current RSS or an isolated update allocation. These
measurements cannot certify all later candidate distributions, larger cohorts,
concurrent evaluators, or an unattended long run.

The projection KL of approximately 0.02 in each record is `q || teacher`, not
a guarantee on the fitted actor. Separately measured updated-policy KL to its
teacher was 0.0062846343 and 0.0036592713. Neither quantity demonstrates improved
held-out policy quality.

### Publication reconstruction before resumed collection: passed

Before resume, the existing derived publication was moved aside to
`publication-before-resume.pt`. The resume process reconstructed policy version
1 from the checkpoint **before** collecting the next cohort with teacher 1.
All **238 online tensors** were exactly equal to the preserved publication.
The independent observation and equality result are in
[publication-repair-proof.json](E:/game/project/sts2_mcp_artifacts/runtime/analysis/macro-credit-readiness-20260905/publication-repair-proof.json).
This proves the tested derived-cache recovery path, not simulator mid-run
restoration or arbitrary crash-stage equivalence.

The current committed
[status](E:/game/project/sts2_mcp_artifacts/runtime/hybrid_runs/macro-credit-v1-20260905-smoke01/status.json)
records **update 2**, **next_seed_index 2**. Therefore the next training seed is
**13400004**. The saved source-manifest set has not changed during these two
invocations; documentation-only evidence additions do not change that identity.

### Related regression suite: passed

**455 related tests passed in 47.56 seconds**, recorded in
[regression-tests.txt](E:/game/project/sts2_mcp_artifacts/runtime/analysis/macro-credit-readiness-20260905/regression-tests.txt).
This is the selected related regression suite, not a claim that every test in
the repository or every live-game/release gate has passed.

### Final-publication held-out bounded execution: passed

The update-2 publication completed the B1 check on **odd seed 13400003**, separate
from training seeds 13400000 and 13400002. The
[held-out summary](E:/game/project/sts2_mcp_artifacts/runtime/analysis/macro-credit-heldout-20260905-smoke01/summary.json)
records `status=complete`, `same_action_gate=matched`, and all **four executable
root candidates** under the same fixed stopping rule. Each pass executed
12 post-root macro decisions, advanced from root floor 2 to ending floor 5,
and recorded three post-root combat exits. Native steps were 84 for reference,
84 for same-action control, and 86 / 86 / 90 for the three alternative actions:
**430 native steps total**. The reference action supplies the fourth candidate.

This is a **greedy diagnostic-only** continuation, not the stochastic training
teacher. The evidence explicitly records `training_label_eligible=false` and
`tail_value_used=false`; it must not flow back into learner targets. One held-out
seed passing execution does not establish policy improvement, general clone
correctness, complete-run performance, or all-surface coverage.

### Final checkpoint/publication and independent forward audit: passed

The independent
[checkpoint-publication-audit.json](E:/game/project/sts2_mcp_artifacts/runtime/analysis/macro-credit-readiness-20260905/checkpoint-publication-audit.json)
records:

- Committed update **2**, next seed cursor **2**, next training seed **13400004**.
- All **238 model tensors** in checkpoint and publication are exactly equal.
- A newly constructed model and optimizer passed strict restore, including
  **238 parameter optimizer-state entries**.
- All **18 selected source hashes** still match their recorded manifest.
- **80 held-out semantic snapshots**, re-encoded in reset-to-sequence order,
  were forwarded through the publication adapter and the independently
  restored learner. Maximum logit error and maximum recurrent-hidden error
  were both **0.0** on that complete audited sequence.
- Held-out execution passed; the audit explicitly sets
  `next_bounded_reward_root_training_ready=true`, `quality_certified=false`
  and `long_training_certified=false`.

The audited artifact hashes are:

| Artifact | SHA-256 |
| --- | --- |
| `checkpoint.pt` | `4c0ef8aa7533747cdb4ea8adf67481cb06dde0e83f2a15e151ab1bb4b7858d35` |
| `publication.pt` | `f8cc50295330291a4a989eb371f664d450cea86506ea6a8cc9b11b729332804a` |

The audited environment was Python **3.12.3**, Numpy **2.4.3**, Torch
**2.9.1+rocm7.2.1.gitff65f5bc**, device **cuda:0**, with one **AMD Radeon RX 7900
XTX**, HIP **7.2.53211-e1a6bc5663**. Resume must retain the recorded runtime and
course identity rather than silently changing the installed framework/device.

All processes from these bounded invocations have exited. The final Windows
`HeadlessSim` process query returned no processes, and `.writer.lock` was absent.
The run contains **31 files / 440,655,630 bytes**, including its preserved
update-1 recovery checkpoint and publication. These are closure-time facts;
recheck ownership, free disk and hardware availability before a later launch.

**Decision:** the tested, unchanged reward-root course may proceed to the next
bounded pilot using the command below. This does **not** approve value
calibration claims, quality-based promotion, all-surface/full-run performance
claims, or large unattended long training. Complete-course GPU peak sampling
and broader resource/quality coverage remain unmeasured. The exact forward
match is scoped to the audited sequence and is not hidden simulator RNG or
universal clone certification.

### Prepared next bounded resume command — not executed here

The checks above have passed for this bounded pilot. Reconfirm the writer is
stopped and use the unchanged course/source/runtime identity:

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
export STS2_ARTIFACT_ROOT=/mnt/e/game/project/sts2_mcp_artifacts/runtime
PYTHON="$STS2_ARTIFACT_ROOT/environments/wsl-rocm/bin/python"
RUN="$STS2_ARTIFACT_ROOT/hybrid_runs/macro-credit-v1-20260905-smoke01"
SIM="$STS2_ARTIFACT_ROOT/dependencies/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"
HSA_ENABLE_DXG_DETECTION=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" -m sts2_rl.train_macro_credit \
  --config config/experiments/hybrid_macro_credit_v1_reward_v9.toml \
  --output-dir "$RUN" --lineage-id macro-credit-v1-20260905-smoke01 \
  --device cuda --sim-exe "$SIM" \
  --resume "$RUN/checkpoint.pt" --max-updates 20
```

This schedules at most **20 additional committed updates**, beginning at teacher
2 / seed 13400004; any failed gate or exhausted budget aborts rather than
skipping a seed. It is a prepared bounded continuation, not a launch performed
by this documentation update. Use Ctrl-C for an operator stop, then inspect the
last checkpoint before resuming; never infer commit authority from publication
or terminal output alone.

## Subsequent authorized pilot launch — 2026-09-05

After the readiness checks above, the operator requested Git submission, a
monitoring panel, and the next training run. Core training code was committed as
`66bf5c207fd5498245aac38e1ce36f4136a18290`. The prepared continuation was then
actually launched at **2026-09-05 03:23:25 UTC / 11:23:25 Asia/Shanghai**:

- Resume the same audited lineage at update **2**, seed cursor **2** and seed
  **13400004**; do not reinitialize the parent or replay update zero.
- At most **20 additional committed updates**, hence target update at most
  **22**. Resource/legality/reconstruction failures still abort rather than
  skip a seed or automatically retry.
- The pinned WSL/ROCm environment, complete course identity, and selected
  source hashes are unchanged. A later docs/monitoring commit is not a model
  or learner identity change.
- The process is launched with a hidden Windows host window, independently of
  the interactive command session. No recurring restart or notification
  automation was created.

Invocation records are under the external artifact directory
`analysis/macro-credit-pilot-20260905-20updates/`: `launch.json` records the
Windows host PID and Git revision; `process.json` records the actual WSL Python
PID/arguments/start time; `stdout.log` and `stderr.log` retain process output.
`exit.json` is written only when that invocation exits. The training run's own
`status.json`, `cohorts/*/update.json`, and committed checkpoint retain their
existing roles. A quiet stdout log is not a stalled-run diagnosis.

The launch-time PIDs were Windows `wsl.exe` **74140** and WSL Python **271**.
These numbers are observations, not reusable process identities. Before any
manual stop, inspect `process.json` and verify the live command still names
this invocation's `launch.py`; only then send SIGINT to that verified WSL PID.
Inspect the final checkpoint/status afterward and never remove an active
writer lock. A missing publication remains a rebuildable derivative, not an
instruction to restart training from an earlier version.

### Initial continuation observations

The first new update **3** was independently read back from both checkpoint
and publication: all **238 model tensors** were finite and exactly equal, and
the checkpoint retained **238 optimizer parameter states**, cursor **3**, and
the original **18 selected source hashes**. This verifies a real resumed
commit, not just process creation.

Update **7** subsequently committed at **11:38:12 Asia/Shanghai**, then the
course entered collection for the next cohort. Its learner elapsed time was
**340.76 seconds**, much slower than updates 3/4/5/6
(16.27/22.81/26.28/61.42 seconds). During the delay, a read-only process sample
observed the main thread waiting in `p9_client_rpc`, consistent with a WSL
mounted-filesystem RPC wait. This does not establish a GPU deadlock or identify
the sole source of the slowdown. The course's repeated output-directory
resource checks are one code-level I/O candidate for later profiling, not a
proven diagnosis. Update 7 still reported finite loss **0.586758**, parameter
delta L2 **0.0488583**, graph span **16**, and policy lag **0**.

Treat this as a performance warning for the bounded pilot. Do not claim all
20 additional updates completed, sustained healthy throughput, or improved
policy quality from these startup observations. No runtime changes, process
restart, seed skipping, or recovery rollback were performed during inspection.
