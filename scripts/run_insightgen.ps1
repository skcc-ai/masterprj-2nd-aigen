<#!
.SYNOPSIS
  Run InsightGen (Agent 2) pipeline.

.DESCRIPTION
  PowerShell wrapper to execute agents/insightgen/cli.py with sensible defaults.
  Supports overriding via parameters or environment variables.

.PARAMETER Db
  Database URL (default: sqlite:///data/structsynth_code.db)

.PARAMETER SrcRoot
  Source root to analyze (default: .)

.PARAMETER OutDir
  Output directory base (default: data/insightgen)

.PARAMETER JobId
  Job identifier; if not set, timestamped ID will be used.

.PARAMETER Entries
  Number of entry patterns (default: 5)

.PARAMETER Sinks
  Number of sink patterns (default: 5)

.PARAMETER MaxDepth
  Max traversal depth (default: 5)

.PARAMETER MaxPaths
  Max paths per entry-sink pair (default: 3)

.PARAMETER TopK
  Top K symbols to retrieve (default: 8)
#>

[CmdletBinding()]
param(
    [string]$DatabaseUrl,
    [string]$SourceRoot,
    [string]$OutputDir,
    [string]$Job,
    [int]$EntriesCount,
    [int]$SinksCount,
    [int]$MaxTraversalDepth,
    [int]$MaxPathsPerPair,
    [int]$TopKSymbols
)

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Err($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Map parameters to internal names
$Db = $DatabaseUrl
$SrcRoot = $SourceRoot
$OutDir = $OutputDir
$JobId = $Job
$Entries = $EntriesCount
$Sinks = $SinksCount
$MaxDepth = $MaxTraversalDepth
$MaxPaths = $MaxPathsPerPair
$TopK = $TopKSymbols

# Resolve defaults from env or fallbacks
if (-not $Db) { $Db = $env:INSIGHTGEN_DB_URL }
if (-not $Db) { $Db = 'sqlite:///data/structsynth_code.db' }

if (-not $SrcRoot) { $SrcRoot = $env:SRC_ROOT }
if (-not $SrcRoot) { $SrcRoot = '.' }

if (-not $OutDir) { $OutDir = $env:OUT_DIR }
if (-not $OutDir) { $OutDir = 'data/insightgen' }

if (-not $JobId) { $JobId = $env:JOB_ID }
if (-not $JobId) { $JobId = "run-$(Get-Date -Format 'yyyyMMdd-HHmmss')" }

if (-not $Entries) { $Entries = $(if ($env:ENTRIES) { [int]$env:ENTRIES } else { 5 }) }
if (-not $Sinks) { $Sinks = $(if ($env:SINKS) { [int]$env:SINKS } else { 5 }) }
if (-not $MaxDepth) { $MaxDepth = $(if ($env:MAX_DEPTH) { [int]$env:MAX_DEPTH } else { 5 }) }
if (-not $MaxPaths) { $MaxPaths = $(if ($env:MAX_PATHS_PER_ENTRY) { [int]$env:MAX_PATHS_PER_ENTRY } else { 3 }) }
if (-not $TopK) { $TopK = $(if ($env:TOPK_SYMBOLS) { [int]$env:TOPK_SYMBOLS } else { 8 }) }

# Resolve paths relative to repo root
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Info "Repo root: $repoRoot"
Write-Info "DB: $Db"
Write-Info "SRC_ROOT: $SrcRoot"
Write-Info "OUT_DIR: $OutDir"
Write-Info "JOB_ID: $JobId"
Write-Info "Entries: $Entries, Sinks: $Sinks, MaxDepth: $MaxDepth, MaxPaths: $MaxPaths, TopK: $TopK"

# Ensure output base exists
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Resolve Python executable
$pythonExe = $env:PYTHON
if (-not $pythonExe) {
    try { $pythonExe = (Get-Command python -ErrorAction Stop).Path } catch {}
}
if (-not $pythonExe) {
    try { $pythonExe = (Get-Command py -ErrorAction Stop).Path } catch {}
}
if (-not $pythonExe) {
    Write-Err "Python executable not found in PATH. Install Python or set $env:PYTHON."
    exit 1
}

# Build args array to avoid PSReadLine/input issues
$argsList = @(
    '-m', 'agents.insightgen.cli',
    '--db', $Db,
    '--src-root', $SrcRoot,
    '--out-dir', $OutDir,
    '--job-id', $JobId,
    '--entries', $Entries.ToString(),
    '--sinks', $Sinks.ToString(),
    '--max-depth', $MaxDepth.ToString(),
    '--max-paths', $MaxPaths.ToString(),
    '--topk', $TopK.ToString()
)

Write-Info "Running: $pythonExe $($argsList -join ' ')"

try {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $pythonExe
    # Build single arguments string for Windows PowerShell compatibility
    $quotedArgs = $argsList | ForEach-Object {
        if ($_ -is [string] -and $_.Contains(' ')) { '"' + $_.Replace('"','\"') + '"' } else { $_ }
    }
    $psi.Arguments = ($quotedArgs -join ' ')
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()

    if ($stdout) { Write-Host $stdout }
    if ($stderr) { Write-Err $stderr }

    if ($proc.ExitCode -ne 0) {
        throw "InsightGen exited with code $($proc.ExitCode)"
    }

    # cli.py prints out_dir on success; fallback to constructed path
    $outDirPrinted = ($stdout.Trim().Split([Environment]::NewLine) | Where-Object { $_ -ne '' } | Select-Object -Last 1)
    if (-not $outDirPrinted) { $outDirPrinted = Join-Path $OutDir $JobId }

    Write-Info "Output directory: $outDirPrinted"
    if (Test-Path $outDirPrinted) {
        Write-Info "Files generated:"
        Get-ChildItem -Path $outDirPrinted | ForEach-Object { Write-Host " - $($_.Name)" }
    } else {
        Write-Err "Output directory not found: $outDirPrinted"
    }

} catch {
    Write-Err $_
    exit 1
}


