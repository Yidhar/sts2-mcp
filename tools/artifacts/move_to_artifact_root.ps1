param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactRoot,
    [Parameter(Mandatory = $true)]
    [ValidateSet("DryRun", "Execute", "Resume")]
    [string]$Mode,
    [string]$PythonExe = "",
    [switch]$SkipUserEnvironmentUpdate
)

$ErrorActionPreference = "Stop"

function Get-NormalizedFullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($Path))
}

function Test-IsWithin {
    param(
        [Parameter(Mandatory = $true)][string]$Child,
        [Parameter(Mandatory = $true)][string]$Parent,
        [switch]$AllowEqual
    )
    $childPath = Get-NormalizedFullPath $Child
    $parentPath = Get-NormalizedFullPath $Parent
    if ($AllowEqual -and $childPath.Equals($parentPath, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    return $childPath.StartsWith(
        $parentPath + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Assert-DisjointPaths {
    param(
        [Parameter(Mandatory = $true)][string]$First,
        [Parameter(Mandatory = $true)][string]$Second,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if ((Test-IsWithin -Child $First -Parent $Second -AllowEqual) -or
        (Test-IsWithin -Child $Second -Parent $First -AllowEqual)) {
        throw "$Label paths must be disjoint: $First <> $Second"
    }
}

function Assert-NoReparsePathComponents {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $cursor = Get-NormalizedFullPath $Path
    while (-not (Test-Path -LiteralPath $cursor)) {
        $parent = Split-Path -Parent $cursor
        if (-not $parent -or $parent -eq $cursor) {
            throw "$Label has no existing filesystem ancestor: $Path"
        }
        $cursor = Get-NormalizedFullPath $parent
    }
    while ($true) {
        $item = Get-Item -LiteralPath $cursor -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$Label contains a symlink, junction, mount point, or other reparse component: $cursor"
        }
        $parent = Split-Path -Parent $cursor
        if (-not $parent -or $parent -eq $cursor) {
            break
        }
        $cursor = Get-NormalizedFullPath $parent
    }
}

function Resolve-ConfinedPath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Relative,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if ([string]::IsNullOrWhiteSpace($Relative) -or [IO.Path]::IsPathRooted($Relative)) {
        throw "$Label must be a non-empty relative path: $Relative"
    }
    $normalized = $Relative.Replace('\', '/')
    $parts = @($normalized.Split('/'))
    if ($parts.Count -eq 0 -or ($parts | Where-Object { $_ -in @('', '.', '..') }).Count -gt 0) {
        throw "$Label contains an empty or traversal component: $Relative"
    }
    $rootPath = Get-NormalizedFullPath $Root
    $resolved = Get-NormalizedFullPath (Join-Path $rootPath $Relative)
    if (-not (Test-IsWithin -Child $resolved -Parent $rootPath)) {
        throw "$Label escaped its root: $Relative -> $resolved"
    }
    return $resolved
}

function Get-ForbiddenResidue {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)]$Patterns
    )
    $found = New-Object System.Collections.Generic.List[string]
    foreach ($rawPattern in $Patterns) {
        $pattern = ([string]$rawPattern).Replace('\', '/')
        if ([string]::IsNullOrWhiteSpace($pattern) -or [IO.Path]::IsPathRooted($pattern)) {
            throw "forbidden residue pattern must be relative: $rawPattern"
        }
        $parts = @($pattern.Split('/'))
        if (($parts | Where-Object { $_ -in @('', '.', '..') }).Count -gt 0) {
            throw "forbidden residue pattern contains an unsafe component: $rawPattern"
        }
        $leaf = $parts[-1]
        if ($parts.Count -gt 1) {
            $parentRelative = [string]::Join('/', $parts[0..($parts.Count - 2)])
            if ($parentRelative.IndexOfAny([char[]]'*?[]') -ge 0) {
                throw "wildcards are allowed only in the final residue component: $rawPattern"
            }
            $parent = Resolve-ConfinedPath -Root $RepositoryRoot -Relative $parentRelative -Label "residue parent"
        } else {
            $parent = $RepositoryRoot
        }
        if (-not (Test-Path -LiteralPath $parent)) {
            continue
        }
        Assert-NoReparsePathComponents -Path $parent -Label "residue parent"
        foreach ($item in Get-ChildItem -LiteralPath $parent -Force) {
            if ($item.Name -like $leaf) {
                $found.Add((Get-NormalizedFullPath $item.FullName))
            }
        }
    }
    return @($found | Sort-Object -Unique)
}

function Get-PythonExecutable {
    param([string]$Hint)
    if ($Hint) {
        return (Resolve-Path -LiteralPath $Hint).Path
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Python is required to create and compare migration inventories."
    }
    return $command.Source
}

function Invoke-Inventory {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string]$Script,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $summary = & $Python $Script @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Artifact inventory command failed with exit code $LASTEXITCODE`: $summary"
    }
    if ($summary) {
        Write-Output $summary
    }
}

$repoRoot = Get-NormalizedFullPath ((Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path)
$targetRoot = Get-NormalizedFullPath $ArtifactRoot
$targetAnchor = Get-NormalizedFullPath ([IO.Path]::GetPathRoot($targetRoot))
if ($targetRoot.Equals($targetAnchor, [StringComparison]::OrdinalIgnoreCase)) {
    throw "ArtifactRoot must not be a filesystem root: $targetRoot"
}
Assert-DisjointPaths -First $targetRoot -Second $repoRoot -Label "artifact/source"
Assert-NoReparsePathComponents -Path $repoRoot -Label "source repository"
Assert-NoReparsePathComponents -Path $targetRoot -Label "artifact target"

$mapPath = Join-Path $PSScriptRoot "move-map.json"
$map = Get-Content -LiteralPath $mapPath -Raw | ConvertFrom-Json
if ($map.schema_version -ne "1.0.0" -or $null -eq $map.entries -or $null -eq $map.forbidden_residue_patterns) {
    throw "Unsupported or malformed artifact move map: $mapPath"
}

$allSpecs = @()
foreach ($spec in $map.entries) {
    $relative = [string]$spec.source
    $destinationRelative = [string]$spec.destination
    $category = [string]$spec.category
    if ([string]::IsNullOrWhiteSpace($category)) {
        throw "Artifact move-map entries require a non-empty category."
    }
    $source = Resolve-ConfinedPath -Root $repoRoot -Relative $relative -Label "source"
    $destination = Resolve-ConfinedPath -Root $targetRoot -Relative $destinationRelative -Label "destination"
    $allSpecs += [pscustomobject]@{
        Source = $source
        Destination = $destination
        SourceRelative = $relative.Replace('\', '/')
        DestinationRelative = $destinationRelative.Replace('\', '/')
        Category = $category
    }
}

if (($allSpecs.Source | Sort-Object -Unique).Count -ne $allSpecs.Count) {
    throw "Artifact move map resolves multiple entries to the same source."
}
if (($allSpecs.Destination | Sort-Object -Unique).Count -ne $allSpecs.Count) {
    throw "Artifact move map resolves multiple entries to the same destination."
}
for ($leftIndex = 0; $leftIndex -lt $allSpecs.Count; $leftIndex++) {
    for ($rightIndex = $leftIndex + 1; $rightIndex -lt $allSpecs.Count; $rightIndex++) {
        $left = $allSpecs[$leftIndex]
        $right = $allSpecs[$rightIndex]
        if ((Test-IsWithin -Child $left.Source -Parent $right.Source -AllowEqual) -or
            (Test-IsWithin -Child $right.Source -Parent $left.Source -AllowEqual)) {
            throw "Artifact move-map sources overlap: $($left.SourceRelative) <> $($right.SourceRelative)"
        }
    }
}

$moves = @()
foreach ($spec in $allSpecs) {
    if (-not (Test-Path -LiteralPath $spec.Source)) {
        continue
    }
    Assert-NoReparsePathComponents -Path $spec.Source -Label "mapped source"
    $item = Get-Item -LiteralPath $spec.Source -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Mapped source root must not be a reparse point: $($spec.Source)"
    }
    $kind = if ($item.PSIsContainer) { "directory" } else { "file" }
    if (-not [IO.Path]::GetPathRoot($spec.Source).Equals(
            [IO.Path]::GetPathRoot($targetRoot),
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Source and ArtifactRoot must be on the same volume for atomic moves: $($spec.Source) -> $targetRoot"
    }
    $moves += [pscustomobject]@{
        Source = $spec.Source
        Destination = $spec.Destination
        SourceRelative = $spec.SourceRelative
        DestinationRelative = $spec.DestinationRelative
        Category = $spec.Category
        Kind = $kind
    }
}

# Reserve the first destination component below every mapped parent. This is
# stricter than merely checking the exact child path: it prevents two source
# trees from being merged into one namespace and makes inventories one-to-one.
foreach ($parent in $allSpecs) {
    if (-not (Test-Path -LiteralPath $parent.Source)) {
        continue
    }
    $parentItem = Get-Item -LiteralPath $parent.Source -Force
    foreach ($child in $allSpecs) {
        if ($child.Destination -eq $parent.Destination -or
            -not (Test-IsWithin -Child $child.Destination -Parent $parent.Destination)) {
            continue
        }
        if (-not $parentItem.PSIsContainer) {
            throw "A file destination cannot contain another mapping: $($parent.DestinationRelative)"
        }
        $suffix = $child.Destination.Substring(
            $parent.Destination.Length + [IO.Path]::DirectorySeparatorChar.ToString().Length
        )
        $reservedName = ($suffix -split '[/\\]', 2)[0]
        $collision = Join-Path $parent.Source $reservedName
        if (Test-Path -LiteralPath $collision) {
            throw (
                "Mapped parent source already contains reserved child destination namespace: " +
                "$($parent.SourceRelative)/$reservedName (reserved by $($child.DestinationRelative))"
            )
        }
    }
}

$moves = @(
    $moves | Sort-Object `
        @{ Expression = { ($_.DestinationRelative -split '[/\\]').Count } }, `
        DestinationRelative
)

$residueBefore = @(Get-ForbiddenResidue -RepositoryRoot $repoRoot -Patterns $map.forbidden_residue_patterns)
$unmappedResidue = @(
    $residueBefore | Where-Object {
        $residuePath = $_
        -not ($moves | Where-Object {
            Test-IsWithin -Child $residuePath -Parent $_.Source -AllowEqual
        })
    }
)
if ($unmappedResidue.Count -gt 0) {
    throw "Forbidden runtime residue is not covered by the move map:`n  - $($unmappedResidue -join "`n  - ")"
}

$python = Get-NormalizedFullPath (Get-PythonExecutable -Hint $PythonExe)
foreach ($move in $moves) {
    if (Test-IsWithin -Child $python -Parent $move.Source -AllowEqual) {
        throw (
            "Inventory Python is inside a source that this migration will move: $python. " +
            "Deactivate the repository virtual environment or pass -PythonExe outside all mapped roots."
        )
    }
}

foreach ($move in $moves) {
    Write-Output "$($move.SourceRelative) -> $($move.DestinationRelative) [$($move.Category); $($move.Kind)]"
}
Write-Output "Forbidden residue roots covered by this move: $($residueBefore.Count)"
Write-Output "Inventory Python: $python"
if ($Mode -eq "DryRun") {
    Write-Output "Dry run only. Re-run with -Mode Execute after explicit approval of the reviewed paths."
    exit 0
}

New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null
Assert-NoReparsePathComponents -Path $targetRoot -Label "created artifact target"
$resolvedTarget = Get-NormalizedFullPath ((Resolve-Path -LiteralPath $targetRoot).Path)
if (-not $resolvedTarget.Equals($targetRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "ArtifactRoot physical path changed after creation: $targetRoot -> $resolvedTarget"
}
Assert-DisjointPaths -First $resolvedTarget -Second $repoRoot -Label "artifact/source"

$journalPath = Join-Path $targetRoot ".sts2-source-externalization.json"
$preInventoryPath = Join-Path $targetRoot ".sts2-pre-move-inventory.json"
$postInventoryPath = Join-Path $targetRoot ".sts2-post-move-inventory.json"
$comparisonPath = Join-Path $targetRoot ".sts2-inventory-comparison.json"
$inventoryScript = Join-Path $PSScriptRoot "inventory.py"

function Write-Journal {
    $temporary = "$journalPath.tmp"
    $journal | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8NoBOM
    Move-Item -LiteralPath $temporary -Destination $journalPath -Force
}

if ($Mode -eq "Resume") {
    foreach ($requiredEvidence in @($journalPath, $preInventoryPath)) {
        if (-not (Test-Path -LiteralPath $requiredEvidence)) {
            throw "Resume requires existing migration evidence: $requiredEvidence"
        }
    }
    foreach ($unfinishedEvidence in @($postInventoryPath, $comparisonPath)) {
        if (Test-Path -LiteralPath $unfinishedEvidence) {
            throw "Resume refuses ambiguous post-move evidence: $unfinishedEvidence"
        }
    }
    $journal = Get-Content -LiteralPath $journalPath -Raw | ConvertFrom-Json
    $preInventory = Get-Content -LiteralPath $preInventoryPath -Raw | ConvertFrom-Json
    if ($journal.schema_version -ne "2.0.0" -or $journal.status -ne "moving" -or
        -not $journal.repository.Equals($repoRoot, [StringComparison]::OrdinalIgnoreCase) -or
        -not $journal.artifact_root.Equals($targetRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Resume journal identity/status does not match this repository and artifact root."
    }
    if (-not $preInventory.complete -or
        $journal.inventory_tool_sha256 -ne $preInventory.inventory_tool_sha256 -or
        $journal.move_map_sha256 -ne $preInventory.move_map_sha256 -or
        (Get-FileHash -LiteralPath $inventoryScript -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            ([string]$preInventory.inventory_tool_sha256).ToLowerInvariant() -or
        (Get-FileHash -LiteralPath $mapPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            ([string]$preInventory.move_map_sha256).ToLowerInvariant()) {
        throw "Resume evidence no longer matches the inventory tool or move map."
    }
    foreach ($entry in $journal.entries) {
        $sourceExists = Test-Path -LiteralPath $entry.source
        $destinationExists = Test-Path -LiteralPath $entry.destination
        if ($entry.status -eq "moved") {
            if ($sourceExists -or -not $destinationExists) {
                throw "Moved journal entry has an invalid source/destination state: $($entry.source)"
            }
        } elseif ($entry.status -eq "planned") {
            if ($sourceExists -and -not $destinationExists) {
                continue
            }
            if (-not $sourceExists -and $destinationExists) {
                # A fixed-path recovery rename may have completed after the
                # original process failed but before Resume was invoked.
                $entry.status = "moved"
                continue
            }
            throw "Planned journal entry has an ambiguous source/destination state: $($entry.source)"
        } else {
            throw "Resume journal contains unsupported entry status: $($entry.status)"
        }
    }
    Write-Journal
} else {
    foreach ($reservedPath in @($journalPath, $preInventoryPath, $postInventoryPath, $comparisonPath)) {
        if (Test-Path -LiteralPath $reservedPath) {
            throw "Refusing to overwrite existing migration evidence: $reservedPath"
        }
    }
    foreach ($move in $moves) {
        if (Test-Path -LiteralPath $move.Destination) {
            throw "Refusing to overwrite existing artifact destination: $($move.Destination)"
        }
    }
    Invoke-Inventory -Python $python -Script $inventoryScript -Arguments @(
        "--root", $repoRoot,
        "--layout", "source",
        "--hash-limit-mib", "0",
        "--output", $preInventoryPath
    )
    $preInventory = Get-Content -LiteralPath $preInventoryPath -Raw | ConvertFrom-Json
    if (-not $preInventory.complete) {
        throw "Pre-move inventory is incomplete: $preInventoryPath"
    }
    $journal = [ordered]@{
        schema_version = "2.0.0"
        repository = $repoRoot
        artifact_root = $targetRoot
        inventory_tool_sha256 = [string]$preInventory.inventory_tool_sha256
        move_map_sha256 = [string]$preInventory.move_map_sha256
        pre_inventory = $preInventoryPath
        post_inventory = $postInventoryPath
        inventory_comparison = $comparisonPath
        started_at_utc = [DateTimeOffset]::UtcNow.ToString("O")
        completed_at_utc = $null
        status = "moving"
        entries = @(
            $moves | ForEach-Object {
                $move = $_
                $snapshot = $preInventory.entries | Where-Object {
                    $_.source_relative_path -eq $move.SourceRelative -and
                    $_.destination_relative_path -eq $move.DestinationRelative
                } | Select-Object -First 1
                if ($null -eq $snapshot -or -not $snapshot.exists) {
                    throw "Pre-move inventory has no existing entry for $($move.SourceRelative)"
                }
                [ordered]@{
                    source_relative_path = $move.SourceRelative
                    destination_relative_path = $move.DestinationRelative
                    category = $move.Category
                    kind = $move.Kind
                    source = $move.Source
                    destination = $move.Destination
                    source_snapshot = [ordered]@{
                        files = [int64]$snapshot.files
                        directories = [int64]$snapshot.directories
                        links = [int64]$snapshot.links
                        bytes = [int64]$snapshot.bytes
                        structure_fingerprint_sha256 = [string]$snapshot.structure_fingerprint_sha256
                    }
                    destination_snapshot = $null
                    status = "planned"
                }
            }
        )
    }
    Write-Journal
}

foreach ($move in $moves) {
    if (-not (Test-Path -LiteralPath $move.Source)) {
        throw "Mapped source disappeared before its move: $($move.Source)"
    }
    if (Test-Path -LiteralPath $move.Destination) {
        throw "Exact destination appeared before move; refusing to nest or merge: $($move.Destination)"
    }
    Assert-NoReparsePathComponents -Path $move.Source -Label "mapped source before move"
    Assert-NoReparsePathComponents -Path $move.Destination -Label "mapped destination before move"
    $sourceItem = Get-Item -LiteralPath $move.Source -Force
    $actualKind = if ($sourceItem.PSIsContainer) { "directory" } else { "file" }
    if ($actualKind -ne $move.Kind -or
        ($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Mapped source kind changed before move: $($move.Source)"
    }
    $parent = Split-Path -Parent $move.Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Assert-NoReparsePathComponents -Path $parent -Label "destination parent"
    if (Test-Path -LiteralPath $move.Destination) {
        throw "Exact destination appeared after parent creation: $($move.Destination)"
    }
    if ($move.Kind -eq "directory") {
        [IO.Directory]::Move($move.Source, $move.Destination)
    } else {
        [IO.File]::Move($move.Source, $move.Destination)
    }
    if ((Test-Path -LiteralPath $move.Source) -or -not (Test-Path -LiteralPath $move.Destination)) {
        throw "Atomic move did not produce the exact expected source/destination state: $($move.Source) -> $($move.Destination)"
    }
    $entry = $journal.entries | Where-Object { $_.source -eq $move.Source } | Select-Object -First 1
    $entry.status = "moved"
    Write-Journal
}

$residueAfter = @(Get-ForbiddenResidue -RepositoryRoot $repoRoot -Patterns $map.forbidden_residue_patterns)
if ($residueAfter.Count -gt 0) {
    throw "Post-move forbidden runtime residue remains:`n  - $($residueAfter -join "`n  - ")"
}

Invoke-Inventory -Python $python -Script $inventoryScript -Arguments @(
    "--root", $targetRoot,
    "--layout", "artifact",
    "--hash-limit-mib", "0",
    "--output", $postInventoryPath
)
Invoke-Inventory -Python $python -Script $inventoryScript -Arguments @(
    "--compare", $preInventoryPath, $postInventoryPath,
    "--output", $comparisonPath
)
$postInventory = Get-Content -LiteralPath $postInventoryPath -Raw | ConvertFrom-Json
foreach ($entry in $journal.entries) {
    if (Test-Path -LiteralPath $entry.source) {
        throw "Post-move verification found source still present: $($entry.source)"
    }
    if (-not (Test-Path -LiteralPath $entry.destination)) {
        throw "Post-move verification found destination missing: $($entry.destination)"
    }
    $snapshot = $postInventory.entries | Where-Object {
        $_.source_relative_path -eq $entry.source_relative_path -and
        $_.destination_relative_path -eq $entry.destination_relative_path
    } | Select-Object -First 1
    if ($null -eq $snapshot) {
        throw "Post-move inventory has no entry for $($entry.destination_relative_path)"
    }
    $entry.destination_snapshot = [ordered]@{
        files = [int64]$snapshot.files
        directories = [int64]$snapshot.directories
        links = [int64]$snapshot.links
        bytes = [int64]$snapshot.bytes
        structure_fingerprint_sha256 = [string]$snapshot.structure_fingerprint_sha256
    }
    $entry.status = "verified"
}
$journal.status = "complete"
$journal.completed_at_utc = [DateTimeOffset]::UtcNow.ToString("O")
Write-Journal

if (-not $SkipUserEnvironmentUpdate) {
    [Environment]::SetEnvironmentVariable("STS2_ARTIFACT_ROOT", $targetRoot, "User")
}
Write-Output "Moved and inventory-verified $($moves.Count) artifact roots to $targetRoot"
Write-Output "Local move journal: $journalPath"
Write-Output "Restart shells/processes or set STS2_ARTIFACT_ROOT explicitly before further runtime work."
