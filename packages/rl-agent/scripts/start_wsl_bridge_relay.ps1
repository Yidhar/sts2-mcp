param(
    [string]$SessionFile = "$env:APPDATA\SlayTheSpire2\bridge\session.json",
    [string]$ListenHost = "127.0.0.1",
    [int]$ListenPort = 0,
    [string[]]$AllowClientCidr = @(),
    [string]$PythonExe = "",
    [int]$MaxBodyBytes = 4194304,
    [int]$MaxResponseBytes = 33554432,
    [int]$MaxConcurrency = 8,
    [double]$TargetTimeoutSeconds = 30.0,
    [switch]$QuietJson
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Get-CheckoutRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}

function Get-ArtifactRoot {
    param([string]$CheckoutRoot)
    $configured = [string]$env:STS2_ARTIFACT_ROOT
    if ([string]::IsNullOrWhiteSpace($configured)) {
        $userProfile = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
        if ([string]::IsNullOrWhiteSpace($userProfile)) {
            throw "Could not resolve a user profile; set STS2_ARTIFACT_ROOT explicitly."
        }
        $configured = Join-Path $userProfile ".sts2-artifacts"
    }
    if (-not [IO.Path]::IsPathRooted($configured)) {
        throw "STS2_ARTIFACT_ROOT must be absolute: $configured"
    }
    $root = [IO.Path]::GetFullPath($configured)
    $anchor = [IO.Path]::GetPathRoot($root)
    if ($root.Equals($anchor, [StringComparison]::OrdinalIgnoreCase) -or
        $root.Equals($CheckoutRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $root.StartsWith($CheckoutRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        $CheckoutRoot.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Artifact root must be disjoint from the source checkout: $root"
    }
    return $root
}

function Get-DefaultListenHost {
    return "127.0.0.1"
}

function Resolve-Python {
    param([string]$ArtifactRoot, [string]$PythonHint)
    if ($PythonHint) {
        return (Resolve-Path $PythonHint).Path
    }
    $candidates = @(
        (Join-Path $ArtifactRoot "environments\windows-python\Scripts\python.exe"),
        "C:\Python313\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
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
    $localRoot = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
    if (-not $localRoot) {
        $localRoot = [IO.Path]::GetTempPath()
    }
    $stateDir = Join-Path $localRoot "STS2BridgeRelay"
    if (-not (Test-Path -LiteralPath $stateDir)) {
        New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
    }
    return (Join-Path $stateDir "wsl_bridge_relay_state.json")
}

function Protect-RelayStatePath {
    param([string]$Path, [switch]$Directory)
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    if ($Directory) {
        $security = New-Object Security.AccessControl.DirectorySecurity
        $inheritance = [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit
        $rule = New-Object Security.AccessControl.FileSystemAccessRule(
            $identity,
            [Security.AccessControl.FileSystemRights]::FullControl,
            $inheritance,
            [Security.AccessControl.PropagationFlags]::None,
            [Security.AccessControl.AccessControlType]::Allow
        )
    } else {
        $security = New-Object Security.AccessControl.FileSecurity
        $rule = New-Object Security.AccessControl.FileSystemAccessRule(
            $identity,
            [Security.AccessControl.FileSystemRights]::FullControl,
            [Security.AccessControl.AccessControlType]::Allow
        )
    }
    $security.SetAccessRuleProtection($true, $false)
    $security.AddAccessRule($rule)
    Set-Acl -LiteralPath $Path -AclObject $security
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

function Stop-StaleRelay {
    param([string]$StateFile)
    if (-not (Test-Path -LiteralPath $StateFile)) {
        return
    }
    try {
        $state = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
    } catch {
        Remove-Item -LiteralPath $StateFile -Force -ErrorAction SilentlyContinue
        return
    }
    $identity = Test-RelayProcessIdentity -State $state
    if (-not $identity.Exists) {
        Remove-Item -LiteralPath $StateFile -Force -ErrorAction SilentlyContinue
        return
    }
    if (-not $identity.Matches) {
        throw "Refusing to terminate PID $($state.pid): relay identity check failed ($($identity.Reason))."
    }
    Stop-Process -Id ([int]$state.pid) -Force
    Start-Sleep -Milliseconds 300
    Remove-Item -LiteralPath $StateFile -Force -ErrorAction SilentlyContinue
}

function Resolve-AllowedClientCidrs {
    param([string]$HostAddress, [string[]]$Configured)
    if ($Configured.Count -gt 0) {
        return @($Configured)
    }
    $result = New-Object System.Collections.Generic.List[string]
    $result.Add("127.0.0.0/8")
    if ($HostAddress -ne "127.0.0.1" -and $HostAddress -ne "::1") {
        $interface = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object { $_.IPAddress -eq $HostAddress } |
            Select-Object -First 1
        if (-not $interface) {
            throw "Cannot derive an allowed WSL subnet for listen host $HostAddress; pass -AllowClientCidr explicitly."
        }
        $result.Add("$HostAddress/$($interface.PrefixLength)")
    }
    return $result.ToArray()
}

$repoRoot = Get-RepoRoot
$checkoutRoot = Get-CheckoutRoot
$artifactRoot = Get-ArtifactRoot -CheckoutRoot $checkoutRoot
$relayScript = (Resolve-Path (Join-Path $repoRoot "scripts\bridge_wsl_relay.py")).Path
$stateFile = Get-RelayStateFile
Protect-RelayStatePath -Path (Split-Path -Parent $stateFile) -Directory

if (-not (Test-Path -LiteralPath $SessionFile)) {
    throw "STS2 session file not found: $SessionFile"
}
$sessionPath = (Resolve-Path $SessionFile).Path
$session = Get-Content -LiteralPath $sessionPath -Raw | ConvertFrom-Json
$trainingToken = [string]$session.capability_tokens.training
if (-not $session.base_url -or -not $trainingToken) {
    throw "Session file missing base_url or scoped training capability token: $sessionPath"
}
if (-not $ListenHost) {
    $ListenHost = Get-DefaultListenHost
}
if ($ListenHost -in @("0.0.0.0", "::", "[::]")) {
    throw "Wildcard relay binding is forbidden. Use loopback or an explicit WSL interface address."
}
$allowedCidrs = Resolve-AllowedClientCidrs -HostAddress $ListenHost -Configured $AllowClientCidr

if ($ListenPort -le 0) {
    $basePort = if ($session.port) { [int]$session.port } else { 0 }
    if ($basePort -le 0 -and [string]$session.base_url -match ":(\d+)/?$") {
        $basePort = [int]$Matches[1]
    }
    if ($basePort -le 0) { $basePort = 27100 }
    $ListenPort = $basePort + 1000
}

Stop-StaleRelay -StateFile $stateFile
$pythonExe = Resolve-Python -ArtifactRoot $artifactRoot -PythonHint $PythonExe
$stateDir = Split-Path -Parent $stateFile
$stdoutLog = Join-Path $stateDir "wsl_bridge_relay.out.log"
$stderrLog = Join-Path $stateDir "wsl_bridge_relay.err.log"
$instanceId = [Guid]::NewGuid().ToString("D")

$arguments = @(
    $relayScript,
    "--target-base-url", ([string]$session.base_url),
    "--listen-host", $ListenHost,
    "--listen-port", ([string]$ListenPort),
    "--state-file", $stateFile,
    "--auth-session-file", $sessionPath,
    "--max-body-bytes", ([string]$MaxBodyBytes),
    "--max-response-bytes", ([string]$MaxResponseBytes),
    "--max-concurrency", ([string]$MaxConcurrency),
    "--target-timeout-s", ([string]$TargetTimeoutSeconds),
    "--instance-id", $instanceId,
    "--quiet"
)
foreach ($cidr in $allowedCidrs) {
    $arguments += @("--allow-client-cidr", [string]$cidr)
}

$process = Start-Process -FilePath $pythonExe -ArgumentList $arguments -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

$deadline = (Get-Date).AddSeconds(10)
$state = $null
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
    if (Test-Path -LiteralPath $stateFile) {
        try {
            Protect-RelayStatePath -Path $stateFile
            $candidate = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
            if ([int]$candidate.pid -eq $process.Id -and [string]$candidate.instance_id -eq $instanceId) {
                $state = $candidate
                break
            }
        } catch {
        }
    }
}
if (-not $state) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    throw "WSL bridge relay failed to start within timeout. See: $stderrLog"
}

$cim = Get-OwnedProcess -ProcessId $process.Id
if (-not $cim) {
    throw "Relay process exited before identity capture. See: $stderrLog"
}
$actualExe = [IO.Path]::GetFullPath([string]$cim.ExecutablePath)
if (-not $actualExe.Equals([IO.Path]::GetFullPath($pythonExe), [StringComparison]::OrdinalIgnoreCase) -or
    -not ([string]$cim.CommandLine).Contains($relayScript) -or
    -not ([string]$cim.CommandLine).Contains($instanceId)) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    throw "Relay process identity did not match the requested executable/script/instance."
}
$state | Add-Member -NotePropertyName process_creation_time_utc -NotePropertyValue (([DateTime]$cim.CreationDate).ToUniversalTime().ToString("o")) -Force
$state | Add-Member -NotePropertyName command_line_sha256 -NotePropertyValue (Get-CommandLineHash -CommandLine ([string]$cim.CommandLine)) -Force
$state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $stateFile -Encoding UTF8
Protect-RelayStatePath -Path $stateFile

$healthUrl = "http://${ListenHost}:$ListenPort/_relay/health"
$headers = @{ Authorization = "Bearer $trainingToken" }
$healthDeadline = (Get-Date).AddSeconds(10)
$healthOk = $false
while ((Get-Date) -lt $healthDeadline) {
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -Headers $headers -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) { $healthOk = $true; break }
    } catch {
    }
    Start-Sleep -Milliseconds 350
}
if (-not $healthOk) {
    $identity = Test-RelayProcessIdentity -State $state
    if ($identity.Matches) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
    throw "WSL bridge relay started but relay health did not become ready: $healthUrl"
}

$result = [ordered]@{
    ok = $true
    pid = $process.Id
    instance_id = $instanceId
    base_url = [string]$state.base_url
    bind_url = [string]$state.base_url
    listen_host = $ListenHost
    listen_port = $ListenPort
    allowed_client_cidrs = $allowedCidrs
    target_base_url = [string]$session.base_url
    session_file = $sessionPath
    state_file = $stateFile
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
}
if ($QuietJson) {
    Write-Output ($result | ConvertTo-Json -Depth 6 -Compress)
} else {
    Write-Output ($result | ConvertTo-Json -Depth 6)
}
