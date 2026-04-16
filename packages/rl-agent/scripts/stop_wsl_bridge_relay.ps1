$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stateFile = Join-Path $repoRoot "tmp\wsl_bridge_relay_state.json"

if (-not (Test-Path $stateFile)) {
    Write-Output '{"ok":true,"stopped":false,"reason":"state_file_missing"}'
    exit 0
}

try {
    $state = Get-Content $stateFile -Raw | ConvertFrom-Json
} catch {
    Remove-Item -Force $stateFile -ErrorAction SilentlyContinue
    Write-Output '{"ok":true,"stopped":false,"reason":"state_file_invalid"}'
    exit 0
}

$stopped = $false
if ($state.pid) {
    $proc = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Id $proc.Id -Force
        $stopped = $true
    }
}

Remove-Item -Force $stateFile -ErrorAction SilentlyContinue

Write-Output ([ordered]@{
    ok = $true
    stopped = $stopped
    pid = $state.pid
} | ConvertTo-Json -Compress)
