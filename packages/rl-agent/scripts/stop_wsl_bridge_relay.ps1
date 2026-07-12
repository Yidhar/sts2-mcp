$ErrorActionPreference = "Stop"

function Get-RelayStateFile {
    $localRoot = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
    if (-not $localRoot) { $localRoot = [IO.Path]::GetTempPath() }
    return (Join-Path (Join-Path $localRoot "STS2BridgeRelay") "wsl_bridge_relay_state.json")
}

function Get-CommandLineHash {
    param([string]$CommandLine)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes([string]$CommandLine)
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-OwnedProcess {
    param([int]$ProcessId)
    return Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
}

function Test-RelayProcessIdentity {
    param($State)
    if (-not $State.pid) {
        return [pscustomobject]@{ Exists = $false; Matches = $false; Reason = "pid_missing"; Process = $null }
    }
    $process = Get-OwnedProcess -ProcessId ([int]$State.pid)
    if (-not $process) {
        return [pscustomobject]@{ Exists = $false; Matches = $false; Reason = "process_missing"; Process = $null }
    }
    foreach ($field in @("process_creation_time_utc", "executable_path", "script_path", "instance_id", "command_line_sha256")) {
        if (-not $State.$field) {
            return [pscustomobject]@{ Exists = $true; Matches = $false; Reason = "identity_field_missing_$field"; Process = $process }
        }
    }
    $actualExe = [IO.Path]::GetFullPath([string]$process.ExecutablePath)
    $expectedExe = [IO.Path]::GetFullPath([string]$State.executable_path)
    if (-not $actualExe.Equals($expectedExe, [StringComparison]::OrdinalIgnoreCase)) {
        return [pscustomobject]@{ Exists = $true; Matches = $false; Reason = "executable_mismatch"; Process = $process }
    }
    $actualCreation = ([DateTime]$process.CreationDate).ToUniversalTime()
    $expectedCreation = ([DateTime]::Parse([string]$State.process_creation_time_utc)).ToUniversalTime()
    if ([Math]::Abs(($actualCreation - $expectedCreation).TotalSeconds) -gt 1.0) {
        return [pscustomobject]@{ Exists = $true; Matches = $false; Reason = "creation_time_mismatch"; Process = $process }
    }
    $commandLine = [string]$process.CommandLine
    if (-not $commandLine.Contains([string]$State.script_path) -or
        -not $commandLine.Contains([string]$State.instance_id) -or
        (Get-CommandLineHash -CommandLine $commandLine) -ne [string]$State.command_line_sha256) {
        return [pscustomobject]@{ Exists = $true; Matches = $false; Reason = "command_identity_mismatch"; Process = $process }
    }
    return [pscustomobject]@{ Exists = $true; Matches = $true; Reason = "matched"; Process = $process }
}

$stateFile = Get-RelayStateFile
if (-not (Test-Path -LiteralPath $stateFile)) {
    Write-Output '{"ok":true,"stopped":false,"reason":"state_file_missing"}'
    exit 0
}
try {
    $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
} catch {
    Write-Output '{"ok":false,"stopped":false,"reason":"state_file_invalid"}'
    exit 2
}
$identity = Test-RelayProcessIdentity -State $state
if (-not $identity.Exists) {
    Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue
    Write-Output ([ordered]@{
        ok = $true
        stopped = $false
        pid = $state.pid
        reason = "process_missing"
    } | ConvertTo-Json -Compress)
    exit 0
}
if (-not $identity.Matches) {
    Write-Output ([ordered]@{
        ok = $false
        stopped = $false
        pid = $state.pid
        reason = "identity_mismatch"
        detail = $identity.Reason
    } | ConvertTo-Json -Compress)
    exit 3
}
Stop-Process -Id ([int]$state.pid) -Force
Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue
Write-Output ([ordered]@{
    ok = $true
    stopped = $true
    pid = $state.pid
    instance_id = $state.instance_id
} | ConvertTo-Json -Compress)
