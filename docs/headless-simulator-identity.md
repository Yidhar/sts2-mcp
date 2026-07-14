# HeadlessSim build identity

Formal headless training is fail-closed on simulator provenance. A file named
`HeadlessSim.exe` or a matching assembly product version is not sufficient:
the executable must have a verified `<executable>.identity.json` sidecar that
also binds the managed `HeadlessSim.dll` containing the simulator logic.

The identity binds all of the following:

- the upstream URL, full Git commit, and full Git tree from
  `third_party/sts2-ai.lock.json`;
- the canonical HeadlessSim project selected by that lock;
- a clean, detached source checkout with matching submodules at build time;
- the required `Release` / `net9.0` build configuration and .NET SDK version;
- the native apphost file name, byte size, and SHA-256 digest;
- the managed implementation file name, byte size, and SHA-256 digest.

## Build the pinned simulator

Restore the locked checkout first. Then run the builder on the external
checkout (never copy that dependency into this repository):

```powershell
python packages/rl-agent/scripts/build_pinned_headless_sim.py `
  --source "$env:STS2_ARTIFACT_ROOT\runtime\dependencies\sts2-ai"
```

The builder produces the canonical Release executable and managed assembly and
writes their identity next to the executable. It does not touch an already-
running Debug simulator, so an old smoke run can finish independently.

## Start formal headless training

```powershell
python -m sts2_rl.train `
  --profile default `
  --backend headless `
  --sim-exe E:\path\to\HeadlessSim\bin\Release\net9.0\HeadlessSim.exe
```

Before launching a subprocess, the CLI resolves the executable to an absolute
path, compares the sidecar source fields to the repository lock, hashes both
the current apphost and managed assembly bytes, and rejects any mismatch. It
then pins that resolved path into the effective training configuration and
records a preflight audit under
`<STS2_ARTIFACT_ROOT>/logs/simulator-preflight/`.

`--sim-identity` may select an explicit sidecar location. There is deliberately
no production flag that permits an unverified simulator. Unit tests remain
independent of this gate by injecting an in-memory typed backend; `--dry-run`
also continues to validate only the model/configuration and launches nothing.
