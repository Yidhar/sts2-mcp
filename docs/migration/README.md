# Architecture-v2 migration guide

This guide covers operational migration from the historical Bridge/MCP control plane
to contract API `2.0.0`. The failed pre-reboot RL implementation, checkpoints,
replay, datasets, inventories, and worktree patches are deliberately outside the
supported migration surface.

## Before changing anything

1. Stop the game Bridge, MCP server, trainers, launch supervisors, and data writers.
2. Preserve unrelated user work on a dedicated branch or patch.
3. Back up saves and irreplaceable non-RL assets.
4. Record the active Bridge, MCP, contract, game-data, and third-party
   versions.
5. Rotate session credentials after upgrading any component that may have printed a
   historical session descriptor.

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

The maintained entry point is `python -m sts2_rl.train`. Do not create new automation
for deleted MuZero/token-memory/MCTS, PPO, planner or historical attention scripts.

Old learner checkpoints and replay are intentionally **not** migration inputs for
the grounded baseline. Exact resume applies only to checkpoints produced by the
new architecture and verifies:

- contract API/schema and legal-action ordering;
- observation schema and grounded encoding fingerprint;
- reward spec/hash;
- grounded model architecture/config and tensor shapes;
- dependency-lock and resolved-device identity; and
- optimizer/replay/counters/pending-update/RNG lineage.

When present and valid, the static `game-data` manifest is recorded as optional audit
provenance, not an exact-resume identity or runtime dependency, because the grounded
encoder and model do not read it.

Missing or incompatible identity fails closed. Never remap an old action index,
load old tensor shapes, or apply the new reward to old replay.

There is no partial or legacy weights path. The separate `--initialize-from` operation
accepts only a complete RL 0.3 checkpoint with an identical model and encoding contract,
then creates a fresh optimizer/replay/counter lineage. See
[`../runbooks/checkpoint-resume.md`](../runbooks/checkpoint-resume.md) for the atomic
hash and identity gates.

## Artifact migration

Use the artifact runbook. The move script is dry-run by default and refuses to
overwrite existing destinations. `-Mode Execute` must be used only after all writers are
stopped, backup and inventory are verified, and the target is outside the checkout.

This applies to saves and current v2 artifacts only. Historical RL checkpoints,
replay, datasets, experiment patches, and generated policy annotations have no
supported import path into the grounded baseline.

## Rollback

- Keep a tagged, matched Bridge/MCP control-plane release if rollback is required.
- Do not downgrade a live session by reusing a v2 descriptor with v1 binaries.
- Stop all processes, restore a matched Bridge/MCP pair, and let the Bridge publish a
  fresh descriptor/token.
- Restore only artifacts whose manifest belongs to that matched control-plane release.
- Record the rollback reason and failing contract/fixture before continuing.

## V1 removal

V1 cannot be removed merely because v2 code exists. Follow
[`v2-cutover.md`](./v2-cutover.md): contract, concurrency, idempotency, capability,
parity, checkpoint, CI, live E2E, and zero-v1-client gates must all pass.
