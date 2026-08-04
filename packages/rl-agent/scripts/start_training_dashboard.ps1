[CmdletBinding()]
param(
    [string]$ArtifactRoot = "",
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [ValidateRange(60, 86400)]
    [int]$StaleSeconds = 900,
    [switch]$NoBrowser,
    [switch]$Foreground
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

function Test-DashboardHealthForArtifactRoot {
    param(
        [Parameter(Mandatory = $true)]$Health,
        [Parameter(Mandatory = $true)][string]$ExpectedArtifactRoot
    )

    if ($Health.schema -ne "sts2-training-dashboard-v1" -or $Health.read_only -ne $true) {
        return $false
    }
    if ([string]::IsNullOrWhiteSpace([string]$Health.artifact_root)) {
        return $false
    }
    try {
        $runningArtifactRoot = [System.IO.Path]::GetFullPath([string]$Health.artifact_root)
        return [string]::Equals(
            $runningArtifactRoot.TrimEnd('\', '/'),
            $ExpectedArtifactRoot.TrimEnd('\', '/'),
            [System.StringComparison]::OrdinalIgnoreCase
        )
    }
    catch {
        return $false
    }
}

function Test-DashboardPython {
    param([Parameter(Mandatory = $true)][string]$Candidate)

    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        return $false
    }
    Push-Location $packageRoot
    try {
        & $Candidate -c "import sts2_rl.monitor_dashboard" 2>$null
        return $LASTEXITCODE -eq 0
    }
    finally {
        Pop-Location
    }
}

$venvPython = Join-Path $packageRoot ".venv\Scripts\python.exe"
$systemPython = (Get-Command python -ErrorAction Stop).Source
$python = if (Test-DashboardPython -Candidate $venvPython) {
    $venvPython
}
elseif (Test-DashboardPython -Candidate $systemPython) {
    $systemPython
}
else {
    throw "Neither the package virtual environment nor system Python can import sts2_rl.monitor_dashboard."
}

$url = "http://127.0.0.1:$Port/"
$healthUrl = "${url}api/v1/health"
$health = $null
try {
    $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2
}
catch {
    # A refused connection or timeout is the normal startup path. If another
    # service owns the port, the fail-closed Python bind below reports it.
    $health = $null
}
if ($null -ne $health) {
    if (Test-DashboardHealthForArtifactRoot -Health $health -ExpectedArtifactRoot $ArtifactRoot) {
        Write-Host "STS2 training dashboard is already running: $url"
        if (-not $NoBrowser) {
            Start-Process $url
        }
        return
    }
    throw "Port $Port is occupied by another service or by a dashboard for a different artifact root."
}

$arguments = @(
    "-m",
    "sts2_rl.monitor_dashboard",
    "--artifact-root",
    $ArtifactRoot,
    "--host",
    "127.0.0.1",
    "--port",
    $Port.ToString(),
    "--stale-seconds",
    $StaleSeconds.ToString()
)
if ($Foreground -and -not $NoBrowser) {
    $arguments += "--open-browser"
}

if ($Foreground) {
    Push-Location $packageRoot
    try {
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Dashboard process exited with code $LASTEXITCODE."
        }
        return
    }
    finally {
        Pop-Location
    }
}

$monitorDirectory = Join-Path $ArtifactRoot "monitor"
New-Item -ItemType Directory -Force -Path $monitorDirectory | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutLog = Join-Path $monitorDirectory "dashboard-$Port-$timestamp.stdout.log"
$stderrLog = Join-Path $monitorDirectory "dashboard-$Port-$timestamp.stderr.log"

# Start-Process receives one Windows command-line string. Implement the
# CommandLineToArgvW escaping rules so quotes and trailing backslashes cannot
# alter an artifact-root argument.
function ConvertTo-WindowsCommandLineArgument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)

    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }
    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * ($backslashes * 2 + 1)))
            [void]$builder.Append('"')
        }
        else {
            [void]$builder.Append(('\' * $backslashes))
            [void]$builder.Append($character)
        }
        $backslashes = 0
    }
    [void]$builder.Append(('\' * ($backslashes * 2)))
    [void]$builder.Append('"')
    return $builder.ToString()
}

$quotedArguments = $arguments | ForEach-Object {
    ConvertTo-WindowsCommandLineArgument -Value $_
}
$process = Start-Process `
    -FilePath $python `
    -ArgumentList ($quotedArguments -join " ") `
    -WorkingDirectory $packageRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$ready = $false
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if ($process.HasExited) {
        break
    }
    Start-Sleep -Milliseconds 250
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 1
        if (
            Test-DashboardHealthForArtifactRoot `
                -Health $health `
                -ExpectedArtifactRoot $ArtifactRoot
        ) {
            $ready = $true
            break
        }
    }
    catch {
        # The server may still be indexing the first metrics file.
    }
}

if (-not $ready) {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        [void]$process.WaitForExit(5000)
    }
    $errorTail = if (Test-Path -LiteralPath $stderrLog) {
        (Get-Content -LiteralPath $stderrLog -Tail 20) -join [Environment]::NewLine
    }
    else {
        "No stderr log was created."
    }
    throw "Dashboard did not become healthy. See $stderrLog`n$errorTail"
}

$pidPath = Join-Path $monitorDirectory "dashboard-$Port.pid"
Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Write-Host "STS2 training dashboard: $url"
Write-Host "PID: $($process.Id)"
Write-Host "Logs: $stdoutLog"
if (-not $NoBrowser) {
    Start-Process $url
}
