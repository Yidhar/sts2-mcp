[CmdletBinding()]
param(
    [string]$Session = "sts2-macro-g4",
    [string]$Lineage = "durable-afterstate-v2-b64-g4",
    [string]$ArtifactRoot = "",
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [ValidateRange(15, 3600)]
    [int]$SyncIntervalSeconds = 60,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$packageRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $packageRoot "..\.."))

if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    if (-not [string]::IsNullOrWhiteSpace($env:STS2_ARTIFACT_ROOT)) {
        $ArtifactRoot = $env:STS2_ARTIFACT_ROOT
    }
    else {
        $repositoryParent = Split-Path -Parent $repositoryRoot
        $repositoryName = Split-Path -Leaf $repositoryRoot
        $ArtifactRoot = Join-Path $repositoryParent "${repositoryName}_artifacts\runtime"
    }
}
$ArtifactRoot = [System.IO.Path]::GetFullPath($ArtifactRoot)
if (-not (Test-Path -LiteralPath $ArtifactRoot -PathType Container)) {
    throw "Artifact root does not exist: $ArtifactRoot"
}

$venvPython = Join-Path $packageRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $venvPython
}
else {
    (Get-Command python -ErrorAction Stop).Source
}

$monitorDirectory = Join-Path $ArtifactRoot "monitor"
New-Item -ItemType Directory -Force -Path $monitorDirectory | Out-Null
$syncPidPath = Join-Path $monitorDirectory "colab-g4-sync-$Lineage.pid"

$alreadyRunning = $false
if (Test-Path -LiteralPath $syncPidPath -PathType Leaf) {
    $rawPid = (Get-Content -LiteralPath $syncPidPath -Raw).Trim()
    if ($rawPid -match '^\d+$') {
        $existingProc = Get-Process -Id ([int]$rawPid) -ErrorAction SilentlyContinue
        if ($null -ne $existingProc -and $existingProc.ProcessName -match '^python') {
            $alreadyRunning = $true
            Write-Host "Colab G4 dashboard sync daemon is already running (PID: $rawPid)."
        }
    }
}

if (-not $alreadyRunning) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdoutLog = Join-Path $monitorDirectory "colab-g4-sync-$timestamp.stdout.log"
    $stderrLog = Join-Path $monitorDirectory "colab-g4-sync-$timestamp.stderr.log"
    $syncScript = Join-Path $PSScriptRoot "sync_colab_g4_to_dashboard.py"
    $syncArgs = @(
        "`"$syncScript`"",
        "--session", "`"$Session`"",
        "--lineage", "`"$Lineage`"",
        "--artifact-root", "`"$ArtifactRoot`"",
        "--watch",
        "--interval", $SyncIntervalSeconds.ToString()
    ) -join " "

    $syncProc = Start-Process `
        -FilePath $python `
        -ArgumentList $syncArgs `
        -WorkingDirectory $packageRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru

    Set-Content -LiteralPath $syncPidPath -Value $syncProc.Id -Encoding ascii
    Write-Host "Started Colab G4 sync daemon (PID: $($syncProc.Id)) -> $stdoutLog"
}

$dashboardLauncher = Join-Path $PSScriptRoot "start_training_dashboard.ps1"
$dashParams = @{
    ArtifactRoot = $ArtifactRoot
    Port         = $Port
}
if ($NoBrowser) {
    $dashParams["NoBrowser"] = $true
}
& $dashboardLauncher @dashParams
