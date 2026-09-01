param(
    [Parameter(Mandatory = $true)]
    [string]$EnvironmentFileWsl,
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,
    [string]$Distro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"
$launcher = "/mnt/e/game/project/sts2_mcp/packages/rl-agent/scripts/train_hybrid_vtrace_parallel_wsl_rocm.sh"
$command = "source '$EnvironmentFileWsl' && exec bash '$launcher'"
$exitCode = 1
try {
    & wsl.exe -d $Distro -- bash -lc $command
    $exitCode = $LASTEXITCODE
}
finally {
    $statusPath = Join-Path $RunRoot "status\trainer.json"
    $state = $null
    if (Test-Path -LiteralPath $statusPath) {
        try {
            $state = Get-Content -Raw -LiteralPath $statusPath | ConvertFrom-Json
        }
        catch {
            $state = $null
        }
    }
    if ($null -eq $state -or $state.state -notin @("complete", "failed")) {
        $payload = [ordered]@{
            pipeline_id = if ($null -ne $state) { $state.pipeline_id } else { $null }
            state = "interrupted"
            error = "native_wsl_exit:$exitCode"
            unix_s = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
        }
        $payload | ConvertTo-Json -Compress | Set-Content -LiteralPath $statusPath
    }
}
exit $exitCode
