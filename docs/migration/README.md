# Architecture-v2 migration guide

This guide covers operational migration from the historical Bridge/MCP/RL stack to
contract API `2.0.0`. It is not permission to delete local artifacts or resume an old
checkpoint without validation.

## Before changing anything

1. Stop the game Bridge, MCP server, trainers, launch supervisors, and data writers.
2. Preserve the current dirty Git diff on a dedicated branch or patch.
3. Back up saves and irreplaceable checkpoint/replay/dataset directories.
4. Generate an artifact inventory and verify its output location.
5. Record the active Bridge, MCP, contract, game-data, checkpoint, and third-party
   versions.
6. Rotate session credentials after upgrading any component that may have printed a
   historical session descriptor.

The files under `baseline-2026-07-11/`, `pre-move-artifacts.json`, and
`legacy-worktree-sha256.txt` are protected migration evidence.

## Player-control migration

| Historical behavior | V2 behavior |
|---|---|
| Default/debug-size MCP tool surface | Default `minimal`; `strategic` and `debug` explicit |
| Optional state version and `strict=false` | `expected_state_version` required; non-strict rejected |
| Client retries after conflict/transport failure | V2 reuses the same request ID and queries status |
| Legacy action identifier/closure | Stable handle re-resolved in the game-thread transaction |
| One shared bearer token | Separate `player-control` and optional `training` tokens |
| Full/hidden state mixed with player state | `/v2/state` is player-visible |

When only legacy v1 is advertised, the MCP server sends a mutation once. A timeout
means the outcome is unknown. Refresh state and reconcile. Do not issue a second
request for the same intent.

## MCP configuration migration

Use [`.mcp.example.json`](../../.mcp.example.json) and replace only
`<REPOSITORY_ROOT>`. Do not copy another developer's `.mcp.json`, session path, log
path, or token. Keep `STS2_MCP_PROFILE=minimal` for normal play.

Rebuild after upgrading:

```powershell
Set-Location .\packages\mcp-server
npm ci
npm run typecheck
npm test
```

Removed from the core control plane: journal/knowledge files, observation
persistence, the old Node RL state machine, AutoSlay, and obsolete live runners.
Deploy these as separate tools only if a current owner and contract exist.

## Bridge migration

- A normal build no longer deploys; use `-p:Sts2Deploy=true` deliberately.
- Training v2 is disabled unless `STS2_BRIDGE_ENABLE_TRAINING_V2=true`.
- Session descriptors advertise scoped tokens and API identities.
- Existing `/health`, `/state`, `/action`, `/events`, `/env/*`, and static export
  belong to `legacy-v1`; migrate callers to `/v2/*`.
- Run Bridge core tests before a live deployment, then record live scene fixtures.

Do not infer retail-game compatibility solely from Bridge core tests.

## RL and checkpoint migration

The maintained entry point is `python -m muzero.train`. Do not create new automation
for PPO `train_pipeline.py` or historical attention scripts.

An old checkpoint/replay may be resumed only if a supported migrator verifies:

- contract API/schema;
- legal-action ordering;
- observation schema;
- reward spec/hash;
- game-data manifest/hash;
- model architecture and tensor shapes;
- optimizer/replay/scheduler/global-step lineage.

Missing or incompatible identity fails closed. Never silently remap an old action
index or apply a different reward to the same replay without creating a new dataset
identity.

Exact resume and partial/legacy weights initialization are separate operations. See
[`../runbooks/checkpoint-resume.md`](../runbooks/checkpoint-resume.md) for the mandatory
atomic/hash gates and the explicit `sts2-weights-only-v1` migration workflow.

## Artifact migration

Use the artifact runbook. The move script is dry-run by default and refuses to
overwrite existing destinations. `-Mode Execute` must be used only after all writers are
stopped, backup and inventory are verified, and the target is outside the checkout.

## Rollback

- Keep the pre-refactor patch/tag and artifact inventory.
- Do not downgrade a live session by reusing a v2 descriptor with v1 binaries.
- Stop all processes, restore a matched Bridge/MCP pair, and let the Bridge publish a
  fresh descriptor/token.
- Restore artifacts by manifest rather than copying only the latest checkpoint file.
- Record the rollback reason and failing contract/fixture before continuing.

## V1 removal

V1 cannot be removed merely because v2 code exists. Follow
[`v2-cutover.md`](./v2-cutover.md): contract, concurrency, idempotency, capability,
parity, checkpoint, CI, live E2E, and zero-v1-client gates must all pass.
