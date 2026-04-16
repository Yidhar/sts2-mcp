param(
    [string]$SessionFile = "$env:APPDATA\SlayTheSpire2\bridge\session.json",
    [string]$ListenHost = "0.0.0.0",
    [int]$ListenPort = 0,
    [string]$PythonExe = "",
    [switch]$QuietJson
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-DefaultListenHost {
    return "0.0.0.0"
}

function Resolve-Python {
    param([string]$RepoRoot, [string]$PythonHint)

    if ($PythonHint) {
        return (Resolve-Path $PythonHint).Path
    }

    $candidates = @(
        (Join-Path $RepoRoot "venv\Scripts\python.exe"),
        "C:\Python313\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "Could not find a usable Windows python.exe to host the WSL bridge relay."
}

function Get-RelayStateFile {
    param([string]$RepoRoot)
    $stateDir = Join-Path $RepoRoot "tmp"
    if (-not (Test-Path $stateDir)) {
        New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
    }
    return (Join-Path $stateDir "wsl_bridge_relay_state.json")
}

function Stop-StaleRelay {
    param([string]$StateFile)
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*bridge_wsl_relay.py*" } |
        ForEach-Object {
            try {
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            } catch {
            }
        }

    if (-not (Test-Path $StateFile)) {
        return
    }

    try {
        $state = Get-Content $StateFile -Raw | ConvertFrom-Json
    } catch {
        Remove-Item -Force $StateFile -ErrorAction SilentlyContinue
        return
    }

    if ($state.pid) {
        $proc = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $proc.Id -Force
            Start-Sleep -Milliseconds 300
        }
    }

    Remove-Item -Force $StateFile -ErrorAction SilentlyContinue
}

$repoRoot = Get-RepoRoot
$relayScript = Join-Path $repoRoot "scripts\bridge_wsl_relay.py"
$stateFile = Get-RelayStateFile -RepoRoot $repoRoot

if (-not (Test-Path $SessionFile)) {
    throw "STS2 session file not found: $SessionFile"
}

$session = Get-Content $SessionFile -Raw | ConvertFrom-Json
if (-not $session.base_url) {
    throw "Session file missing base_url: $SessionFile"
}

if (-not $ListenHost) {
    $ListenHost = Get-DefaultListenHost
}

if ($ListenPort -le 0) {
    $basePort = 0
    if ($session.port) {
        $basePort = [int]$session.port
    } elseif ($session.base_url -match ":(\d+)/?$") {
        $basePort = [int]$Matches[1]
    }
    if ($basePort -le 0) {
        $basePort = 27100
    }
    $ListenPort = $basePort + 1000
}

Stop-StaleRelay -StateFile $stateFile

$pythonExe = Resolve-Python -RepoRoot $repoRoot -PythonHint $PythonExe
$stdoutLog = Join-Path $repoRoot "tmp\wsl_bridge_relay.out.log"
$stderrLog = Join-Path $repoRoot "tmp\wsl_bridge_relay.err.log"

$arguments = @(
    $relayScript,
    "--target-base-url", ([string]$session.base_url),
    "--listen-host", $ListenHost,
    "--listen-port", ([string]$ListenPort),
    "--state-file", $stateFile,
    "--quiet"
)

$proc = Start-Process -FilePath $pythonExe -ArgumentList $arguments -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

$deadline = (Get-Date).AddSeconds(10)
$bindUrl = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
    if (Test-Path $stateFile) {
        try {
            $state = Get-Content $stateFile -Raw | ConvertFrom-Json
            $bindUrl = [string]$state.base_url
            break
        } catch {
        }
    }
}

if (-not $bindUrl) {
    try {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }
    throw "WSL bridge relay failed to start within timeout. See: $stderrLog"
}

$healthBindHost = if ($ListenHost -eq "0.0.0.0") { "127.0.0.1" } else { $ListenHost }
$healthUrl = "http://${healthBindHost}:$ListenPort/health"
$headers = @{ Authorization = "Bearer $($session.token)" }
$healthDeadline = (Get-Date).AddSeconds(10)
$healthOk = $false
while ((Get-Date) -lt $healthDeadline) {
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -Headers $headers -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) {
            $healthOk = $true
            break
        }
    } catch {
    }
    Start-Sleep -Milliseconds 350
}

if (-not $healthOk) {
    try {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }
    throw "WSL bridge relay started but /health did not become ready: $healthUrl"
}

$wslBaseUrl = if ($ListenHost -eq "0.0.0.0") {
    "http://host.docker.internal:$ListenPort/"
} else {
    $bindUrl
}

$result = [ordered]@{
    ok = $true
    pid = $proc.Id
    base_url = $wslBaseUrl
    bind_url = $bindUrl
    listen_host = $ListenHost
    listen_port = $ListenPort
    target_base_url = [string]$session.base_url
    session_file = (Resolve-Path $SessionFile).Path
    state_file = $stateFile
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
}

if ($QuietJson) {
    Write-Output ($result | ConvertTo-Json -Depth 4 -Compress)
} else {
    Write-Output ($result | ConvertTo-Json -Depth 4)
}
