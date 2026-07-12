# Third-party dependencies

Third-party source and prebuilt runtimes are not vendored into this repository.
Each dependency is represented by a lock file containing its upstream URL,
commit, expected layout, and license review state.

Restore `sts2-ai` with:

```powershell
$env:STS2_ARTIFACT_ROOT = '<ABSOLUTE_PATH_OUTSIDE_THE_REPOSITORY>'
python tools/third_party/restore_sts2_ai.py
```

The default checkout is
`<STS2_ARTIFACT_ROOT>/dependencies/sts2-ai`; dependency source is never restored
inside the source checkout. The command refuses to replace an existing checkout
and verifies the commit, Git tree, clean/detached state, canonical project, and
submodules. Re-run verification without network access using:

```powershell
python tools/third_party/restore_sts2_ai.py --verify-only
```

During the one-time architecture-v2 migration only, audit the old ignored
checkout before externalizing it with an explicit absolute, resolved path:

```powershell
$legacyCheckout = (Resolve-Path -LiteralPath '.\third_party\sts2-ai').Path
python tools/third_party/restore_sts2_ai.py --verify-only --destination $legacyCheckout
```

Relative `--verify-only` destinations are rejected to avoid changing meaning
with the caller's working directory. Non-verification commands require every
destination—including absolute overrides—to remain below
`STS2_ARTIFACT_ROOT`.

Distribution remains disabled while a lock file has
`license_status: review-required`.
