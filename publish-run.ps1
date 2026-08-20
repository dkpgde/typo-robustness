param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$Message
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot
$OutputRoots = @('results', 'artifacts', 'figures')

function Invoke-Git {
    param([Parameter(Mandatory)][string[]]$Arguments)

    & git -C $RepoRoot @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($Arguments -join ' ')"
    }
}

& git -C $RepoRoot diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    throw 'The index already contains staged changes. Commit or unstage them before running this script.'
}
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to inspect the Git index.'
}

$Branch = (& git -C $RepoRoot branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or -not $Branch) {
    throw 'The repository is on a detached HEAD or its current branch could not be determined.'
}

Invoke-Git -Arguments @('remote', 'get-url', 'origin') | Out-Null

$RunId = $null
$ConfigFiles = Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'artifacts') -Filter 'config_*.json' -File |
    Sort-Object Name -Descending

foreach ($ConfigFile in $ConfigFiles) {
    if ($ConfigFile.Name -notmatch '^config_(\d{8}_\d{6})\.json$') {
        continue
    }

    $CandidateId = $Matches[1]
    $RequiredFiles = @(
        "results/metrics_$CandidateId.csv"
        "results/resource_usage_$CandidateId.csv"
        "results/predictions_$CandidateId.csv"
        "results/mcnemar_holm_$CandidateId.csv"
        "results/relative_f1_degradation_test_$CandidateId.csv"
        "figures/absolute_performance_$CandidateId.png"
        "figures/f1_degradation_delta_$CandidateId.png"
    )

    if ($RequiredFiles.Where({ -not (Test-Path -LiteralPath (Join-Path $RepoRoot $_) -PathType Leaf) }).Count) {
        continue
    }

    $Config = Get-Content -LiteralPath $ConfigFile.FullName -Raw | ConvertFrom-Json
    # Word and character models are always present; tokenizer variants are config-driven.
    $ExpectedModels = 2
    if ($Config.bpe_vocab_sizes) {
        $ExpectedModels += @($Config.bpe_vocab_sizes).Count
    }
    elseif ($null -ne $Config.subword_max_tokens) {
        $ExpectedModels++
    }
    $ExpectedMetricRows = $Config.training_seeds.Count * $Config.corruption_levels.Count * $ExpectedModels
    $ExpectedResourceRows = $Config.training_seeds.Count * $ExpectedModels
    $MetricRows = @(Import-Csv -LiteralPath (Join-Path $RepoRoot "results/metrics_$CandidateId.csv")).Count
    $ResourceRows = @(Import-Csv -LiteralPath (Join-Path $RepoRoot "results/resource_usage_$CandidateId.csv")).Count

    if ($MetricRows -eq $ExpectedMetricRows -and $ResourceRows -eq $ExpectedResourceRows) {
        $RunId = $CandidateId
        break
    }
}

if (-not $RunId) {
    throw 'No completed experiment run was found.'
}

$SourcePathspecs = @('.')
foreach ($OutputRoot in $OutputRoots) {
    $SourcePathspecs += ":(exclude)$OutputRoot"
    $SourcePathspecs += ":(exclude)$OutputRoot/**"
}
Invoke-Git -Arguments (@('add', '-A', '--') + $SourcePathspecs)

$LatestOutputPaths = foreach ($OutputRoot in $OutputRoots) {
    Get-ChildItem -LiteralPath (Join-Path $RepoRoot $OutputRoot) -Recurse -File |
        Where-Object { $_.Name -like "*$RunId*" } |
        ForEach-Object { $_.FullName.Substring($RepoRoot.Length + 1).Replace('\', '/') }
}
$LatestOutputPaths = @($LatestOutputPaths | Sort-Object -Unique)

if (-not $LatestOutputPaths.Count) {
    throw "No output files were found for run $RunId."
}

$StageableOutputPaths = foreach ($Path in $LatestOutputPaths) {
    & git -C $RepoRoot check-ignore -q -- $Path
    if ($LASTEXITCODE -eq 1) {
        $Path
    }
    elseif ($LASTEXITCODE -ne 0) {
        throw "Unable to check ignore rules for $Path."
    }
}

if ($StageableOutputPaths) {
    Invoke-Git -Arguments (@('add', '-A', '--') + @($StageableOutputPaths))
}

$StagedPaths = @(Invoke-Git -Arguments @('diff', '--cached', '--name-only'))
$UnexpectedOutputs = $StagedPaths | Where-Object {
    $_ -match '^(results|artifacts|figures)/' -and $_ -notmatch [regex]::Escape($RunId)
}
if ($UnexpectedOutputs) {
    throw "Refusing to commit output files from another run: $($UnexpectedOutputs -join ', ')"
}
if (-not $StagedPaths.Count) {
    throw 'There are no changes to commit.'
}

Write-Host "Latest completed run: $RunId"
Write-Host 'Git status after staging:'
Invoke-Git -Arguments @('status', '--short')

$Confirmation = (Read-Host 'Proceed with commit and push? [y/N]').Trim().ToLowerInvariant()
if ($Confirmation -notin @('y', 'yes')) {
    Invoke-Git -Arguments @('restore', '--staged', '--', '.')
    Write-Host 'Cancelled. The changes staged by this script were unstaged.'
    exit 0
}

Invoke-Git -Arguments @('commit', '-m', $Message)
Invoke-Git -Arguments @('push', 'origin', $Branch)

Write-Host "Committed and pushed $Branch with run $RunId."
