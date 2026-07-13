# Artifact inventory and externalization runbook

This runbook externalizes current architecture-v2 assets and dependency checkouts.
Artifacts from the deleted MuZero/PPO/offline/demo routes are not migration inputs and
are deliberately absent from the move map; remove them rather than archiving them as a
supported lineage.

## Scope

Runtime artifacts include:

- checkpoint and optimizer directories;
- pending rollout queues contained in complete current atomic checkpoints and current datasets;
- TensorBoard and diagnostic logs;
- virtual environments and downloaded runtimes;
- PIDs and launch-command files;
- static exports and temporary decompilations;
- release DLL/ZIP output;
- local analysis/probe reports, WSL/MCP smoke output, reverse-engineering scratch,
  nested third-party checkouts, and abandoned editor/game-project sandboxes.

The target root must be outside the repository and is configured as
`STS2_ARTIFACT_ROOT`.

## Preconditions

1. Stop trainers, supervisors, game instances, Bridge, MCP, recorders, and sync jobs.
2. Verify no process is writing into a source directory.
3. Ensure the destination has enough free space.
4. Back up current high-value artifacts independently.
5. Deactivate repository-local virtual environments. The inventory interpreter
   must be outside every mapped source (pass `-PythonExe` explicitly if needed),
   because post-move verification runs after mapped sources are moved.

## Inventory

Choose an external path and write the inventory there, not into the repository:

```powershell
python .\tools\artifacts\inventory.py `
  --layout source `
  --output '<ARTIFACT_ROOT>\pre-move-inventory.json'
```

Evidence output may not overlap the source inventory root or any mapped artifact
destination. Keep post-move inventory/comparison files at the artifact-root top level,
as shown below, rather than under `runs/`, `datasets/`, or another measured payload.

Inventory schema 2 records every mapping separately, including files, directories,
empty directories, symlinks, junctions, and Windows reparse records. It never follows
a link or reparse point. Any `lstat`, directory-enumeration, link-read, or hash error is
recorded and makes the command exit non-zero; an inventory with `complete: false` is
not migration evidence. Review each entry, totals, largest files, link details,
structure fingerprint, and available content hash. The default hashes files up to
8 MiB plus key metadata. It is not a substitute for a storage-level backup or a full
content hash of large replay/optimizer files. The inventory also records the SHA-256
of the inventory implementation and the move map; comparisons reject a tool or map
revision mismatch.

Inventories made with the former schema 1 tool are not comparable with schema 2 and
must be regenerated after any change to `move-map.json`.

## Dry run

```powershell
.\tools\artifacts\move_to_artifact_root.ps1 `
  -ArtifactRoot '<ARTIFACT_ROOT>' -Mode DryRun
```

The tool prints the exact source/destination/kind triples and exits without moving
them. It rejects filesystem roots, source/target overlap in either direction,
symlinks/junctions/mount points in either root path, overlapping sources, duplicate
destinations, parent/child destination namespace collisions, cross-volume moves,
and ignored runtime residue that has no reviewed mapping. The target and every
existing path component must be a normal filesystem directory, not a reparse point.

Externalization requires the checkout and destination to be on the same physical
volume so each directory or file move is an exact atomic rename. The reviewed
source-to-canonical mapping is versioned in `tools/artifacts/move-map.json`. It covers
only current generic quarantine, release, MCP smoke, and pinned dependency paths. Any
retired learner checkpoint, replay, demo, cache, or virtual environment is an unmapped
residue and blocks the move until it is deleted. Rebuild usable environments at their
final external paths from the exact dependency locks/wheel pins.

## Execute

Only after the preconditions, backup, inventory, and dry run are accepted:

```powershell
.\tools\artifacts\move_to_artifact_root.ps1 `
  -ArtifactRoot '<ARTIFACT_ROOT>' `
  -Mode Execute
```

The script refuses an existing destination before the batch and immediately before
every move. It uses the exact .NET file/directory move APIs, which fail instead of
PowerShell's directory-nesting behavior. Before the first move it creates a complete
source inventory and writes a local, crash-updated
`.sts2-source-externalization.json` journal beneath the artifact root. Each journal
entry includes filesystem kind, counts, bytes, and the pre-move structure
fingerprint. After the moves it:

1. requires every forbidden repository-residue pattern to be empty;
2. creates an artifact-layout inventory without following links/reparse points;
3. compares every source mapping to its exact destination mapping;
4. records the destination counts/fingerprint and marks entries `verified` only if
   the comparison succeeds.

The internal evidence files are `.sts2-pre-move-inventory.json`,
`.sts2-post-move-inventory.json`, and `.sts2-inventory-comparison.json`. A partial run
must be recovered from the journal rather than blindly rerun. When the journal remains
at `status: moving`, the pre-inventory exists, and neither post-inventory nor comparison
has been published, resume the exact same root with:

```powershell
.\tools\artifacts\move_to_artifact_root.ps1 `
  -ArtifactRoot '<ARTIFACT_ROOT>' `
  -Mode Resume
```

Resume verifies the repository/root identity, journal schema, inventory-tool hash, and
move-map hash before doing work. Every journal entry must be in exactly one of two
states: source present/destination absent, or source absent/destination present. It
refuses missing-both, present-both, completed post evidence, an edited map/tool, or an
unknown journal status. A fixed-path recovery rename completed after a process failure
is not trusted on sight: Resume only marks it moved provisionally, then recreates the
full artifact inventory and requires the original per-mapping comparison to pass.

After verification the script sets the user-level `STS2_ARTIFACT_ROOT`; it cannot
change the environment of the calling shell, so restart shells/processes or set the
process variable explicitly. CI/test fixtures may pass `-SkipUserEnvironmentUpdate`;
do not use that switch for an operator migration unless the environment is managed by
another explicit provisioning step. The evidence contains machine-local absolute paths
and must not be committed.

## Verify

1. Confirm the script-generated comparison has `status: ok` and no inventory error.
2. Independently compare file/byte/link counts and critical metadata/checkpoint hashes.
3. Run checkpoint inspection and a bounded resume smoke without starting a long run.
4. Verify logs, replay, and new checkpoints are written under the external root.
5. Run repository hygiene and confirm no runtime artifact is tracked.

The same inventory tool can inspect the external tree after the move:

```powershell
python .\tools\artifacts\inventory.py `
  --root '<ARTIFACT_ROOT>' `
  --layout artifact `
  --output '<ARTIFACT_ROOT>\post-move-inventory.json'
```

Artifact-layout inventory assigns nested mapped paths to exactly one entry, so
canonical parents such as `runs/` and `datasets/` do not double-count migrated
archives placed below them. Each parent reserves the first namespace
component used by a child mapping; the move preflight proves that namespace did not
exist in the parent's source tree. This permits direct per-mapping comparison rather
than relying only on aggregate totals.

For an independent comparison, generate source and artifact inventories with the
same `--hash-limit-mib` and run:

```powershell
python .\tools\artifacts\inventory.py `
  --compare '<SOURCE_INVENTORY>' '<ARTIFACT_INVENTORY>' `
  --output '<ARTIFACT_ROOT>\inventory-comparison.json'
```

The command exits non-zero for a missing destination, kind/count/byte/fingerprint
difference, link/reparse difference, content-hash difference, map-hash difference,
hash-limit difference, duplicate mapping, or any inventory error. Separately verify
the move journal has only `verified` entries.

## Failure handling

- Do not rerun with `-Mode Execute` against a partially populated destination.
- Use `-Mode Resume` only for its exact journal/root and only after stopping all writers.
- Do not delete a source, destination, journal, or inventory after a failed comparison.
- Stop all writers and compare source/destination inventories.
- Preserve both copies until the mismatch is explained.
- Never use a recursive delete as rollback.
- Restore by explicit manifest entries and re-run checkpoint validation.

## Retention

Keep documented policies for baseline, best, milestone, migration, and failure
checkpoints. Delete an artifact only after its lineage, replacement, retention class,
and backup status are known. Git should retain only manifests/checksums, never the
artifact itself.
