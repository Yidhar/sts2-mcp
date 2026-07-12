# Development and verification runbook

Run commands from the repository root unless a section changes directory.

## Fast repository checks

```powershell
python .\tools\contracts\check_contracts.py
python .\tools\game_data\verify_manifest.py
python .\tools\release\check_versions.py
python -m unittest tools.release.test_build_sbom -v
python .\tools\ci\check_dependency_locks.py
python .\tools\ci\check_markdown_links.py
python .\tools\ci\check_generated.py
python .\tools\ci\check_repository.py
```

`check_generated.py` is read-only. Use the relevant generator only when changing the
source manifest/data intentionally.

## MCP server

```powershell
Set-Location .\packages\mcp-server
npm ci
npm run typecheck
npm test
node smoke-test.js
Set-Location ..\..
```

`smoke-test.js` needs a current Bridge session and exits nonzero for a missing, stale,
or unhealthy session. It must not print credentials.

The self-hosted live-smoke runner must run GitHub Actions Runner 2.327.1 or newer so
the repository's SHA-pinned Node 24 action runtimes can start.

## Bridge core and full build

Dependency-free core:

```powershell
dotnet run --project .\mods\sts2-bridge\tests\BridgeCore.Tests\BridgeCore.Tests.csproj --configuration Release
```

Full build:

```powershell
$env:STS2_DIR = '<PATH_TO_STS2>'
dotnet build .\mods\sts2-bridge\sts2-bridge.csproj
```

Normal build does not deploy. Add `-p:Sts2Deploy=true` only when the operator intends
to replace the live mod. Stop the game first if its DLL is locked.

After deployment, prepare a screen with an operator-reviewed safe legal action and set
`STS2_LIVE_SMOKE_ACTION_HANDLE` to its exact handle. The live smoke validates
`/v2/health`, session capabilities, a read-only state, one strict-revision command,
and an identical duplicate-request replay (same request ID and deadline), then records
the game/Bridge/contract versions. Also verify clean shutdown separately.

## RL

```powershell
Set-Location .\packages\rl-agent
$env:PIP_EXTRA_INDEX_URL = 'https://download.pytorch.org/whl/cpu'
python -m pip install -r requirements-bootstrap.lock
python -m pip install -r requirements-dev.lock
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
python -m ruff check sts2_rl sts2_baseline launcher.py launcher_watchdog.py
python -m mypy sts2_rl sts2_baseline
python -m pytest tests -q -p no:cacheprovider
python -m sts2_rl.train --dry-run
Set-Location ..\..
```

Live or headless smoke runs are separate from unit/contract tests. A long training run
requires green backend lifecycle, live/headless parity, reward identity, checkpoint
resume, artifact-path checks, and the fixed odd held-out seed namespace with per-seed
logs. Dry-run and unit tests do not establish Act 1 performance.

## Contract and game-data changes

After changing `contracts/manifest.json`, regenerate language constants:

```powershell
python .\tools\contracts\generate_contract_versions.py
python .\tools\contracts\check_contracts.py
python .\tools\ci\check_generated.py
```

After changing `game-data/raw` or an intentional generated data file:

```powershell
python .\tools\game_data\build_manifest.py
python .\tools\game_data\verify_manifest.py
```

Never edit a generated documentation page merely to make a check pass.

## Before finishing a change

- Re-run the affected component suite and fast repository checks.
- Confirm no session descriptor, token, PID, log, checkpoint, binary, or virtual
  environment was added.
- Confirm examples contain no developer-specific absolute path.
- Confirm tests/builds did not leave generated output or caches tracked.
- Update contracts, component README, migration notes, and changelog when behavior or
  compatibility changes.
- State explicitly whether live-game validation was or was not performed.
