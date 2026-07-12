# Release validation and evidence runbook

Architecture-v2 release evidence is generated outside the checkout. Source Git never
contains built DLLs, ZIP files, virtual environments, third-party checkouts, or the
generated release SBOM/provenance files.

## Hard gates

A release is blocked unless all of the following are true:

1. The source tree is committed and clean, and component declarations match
   `release-manifest.json`.
2. Contract, game-data, generated-file, dependency-lock, Markdown-link, repository
   hygiene, MCP, Bridge-core/full-build, and RL gates pass.
3. The candidate Bridge/MCP pair passes the self-hosted live-game smoke against the
   exact installed DLL and manifest.
4. Checkpoint/backend parity and any release-relevant resume migration have a recorded
   result.
5. Every `third_party/*.lock.json` has an approved license and explicitly allows
   distribution.

`tools/third_party/check_licenses.py` intentionally fails while review is pending. Do
not change a lock to `approved` merely to make CI green; attach the actual legal review
and license identity first.

## Deterministic SBOM

Generate the CycloneDX 1.6 dependency inventory into an external candidate directory:

```powershell
python .\tools\release\build_sbom.py `
  --output '<ARTIFACT_ROOT>\release\release-sbom.cdx.json'
```

The generator reads the release manifest, npm lock, Python locks, ROCm wheel artifact
lock, and third-party Git locks. It has no wall-clock or machine-path fields, merges
identical cross-profile packages, and fails closed on ambiguous identity/hash/license
metadata. Re-running it against unchanged inputs must produce identical bytes.

## Candidate build and provenance

Build Bridge only on an authorized runner with the exact retail references. Normal
build remains non-deploying:

```powershell
$env:STS2_DIR = '<PATH_TO_STS2>'
dotnet build .\mods\sts2-bridge\sts2-bridge.csproj `
  --configuration Release -p:Sts2Deploy=false
```

Copy candidate files to `<ARTIFACT_ROOT>\release\` without adding them to Git. Once the
tree is clean, attest source, identity files, SBOM, and every distributable candidate:

```powershell
python .\tools\release\build_provenance.py `
  --require-clean `
  --artifact '<ARTIFACT_ROOT>\release\release-sbom.cdx.json' `
  --artifact '<ARTIFACT_ROOT>\release\sts2-bridge.dll' `
  --artifact '<ARTIFACT_ROOT>\release\sts2-bridge.json' `
  --output '<ARTIFACT_ROOT>\release\release-provenance.json'
```

The provenance contains SHA-256 and byte size for each artifact plus source commit,
tree, CI identity, toolchains, and hashes of release/contract/data/dependency identity
files. Never use a dirty-tree provenance document for publication.

## GitHub workflow

`.github/workflows/release-validation.yml` has independent source, MCP, Bridge-core,
and RL jobs. Only after all four pass does the protected
`[self-hosted, Windows, sts2, release]` job:

1. validate external live/headless parity and two-release compatibility/v1-telemetry
   evidence against the exact source commit and contract/dependency identities;
2. full-build the Bridge against retail references;
3. require the installed DLL/manifest to be byte-identical to that candidate;
4. run the full MCP suite and a live strict-revision mutation twice with the same
   request ID/deadline, proving the second result is retained replay rather than a
   second mutation;
5. package the Bridge DLL/manifest and MCP tarball, then attest those candidate bytes,
   live result, external evidence, and deterministic SBOM; and
6. enforce the third-party license gate.

The dispatch operator must prepare a safe game screen and supply its reviewed legal
`live_action_handle`. The protected runner must define absolute files in
`STS2_RELEASE_PARITY_EVIDENCE` and `STS2_RELEASE_COMPATIBILITY_EVIDENCE`; their strict
formats are enforced by `tools/release/check_external_evidence.py`. GitHub Actions are
pinned to full commit SHAs. A failed license or external evidence gate does not
authorize a release; candidate evidence may still upload under `if: always()` for
diagnosis.

## Publication checklist

- Verify candidate hashes against `release-provenance.json` after upload/download.
- Verify the SBOM reports the expected component versions and any unapproved dependency
  as non-distributable/`NOASSERTION`.
- Publish matched Bridge DLL/manifest and MCP source/package only; do not bundle a
  restored third-party checkout or retail assemblies.
- Attach migration notes, exact live-game result, known incompatibilities, and rollback
  instructions.
- Retain the source commit, evidence files, and immutable artifact URL as one release
  record.
- If any byte changes, regenerate all candidate evidence and rerun every gate.
