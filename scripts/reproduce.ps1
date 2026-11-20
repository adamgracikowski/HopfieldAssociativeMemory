<#
.SYNOPSIS
    Recompute the experiments and redraw their figures.

.DESCRIPTION
    PowerShell counterpart of scripts/reproduce.sh. Runs every "#   cargo run"
    line of data/experiments/<name>/*.toml from src/compute (with --force),
    then every "#   uv run" line from src/analysis. The benchmark is skipped
    unless -Benchmark is given.

    Requirements: rustup (the toolchain of src/compute/rust-toolchain.toml)
    and uv on PATH. If script execution is blocked, run it with:
        powershell -ExecutionPolicy Bypass -File scripts\reproduce.ps1

.PARAMETER Experiments
    Names of the experiment directories to run (default: all), e.g.
    -Experiments noise,cats-noise.

.PARAMETER Benchmark
    Also rerun the benchmark measurements.

.PARAMETER FiguresOnly
    Only redraw the figures; the recall figures need runs/ from an earlier
    full run.

.EXAMPLE
    pwsh -File scripts/reproduce.ps1

.EXAMPLE
    pwsh -File scripts/reproduce.ps1 -Experiments stability,noise -FiguresOnly
#>
[CmdletBinding()]
param(
    [string[]]$Experiments,
    [switch]$Benchmark,
    [switch]$FiguresOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$experimentsDir = Join-Path $root 'data/experiments'
$compute = Join-Path $root 'src/compute'
$analysis = Join-Path $root 'src/analysis'

$all = Get-ChildItem -Directory $experimentsDir | Sort-Object Name | ForEach-Object Name
if ($Experiments) {
    $unknown = $Experiments | Where-Object { $_ -notin $all }
    if ($unknown) { throw "unknown experiments: $($unknown -join ', ') (available: $($all -join ', '))" }
    $selected = $all | Where-Object { $_ -in $Experiments }
} else {
    $selected = $all
}

function Get-Commands([string]$prefix) {
    foreach ($name in $selected) {
        $dir = Join-Path $experimentsDir $name
        $files = @(Get-ChildItem (Join-Path $dir 'config.toml') -ErrorAction SilentlyContinue) +
            @(Get-ChildItem $dir -Filter '*.toml' | Where-Object Name -ne 'config.toml' | Sort-Object Name)
        foreach ($file in $files) {
            foreach ($line in Get-Content $file) {
                if ($line -like "#   $prefix *") {
                    [pscustomobject]@{ Experiment = $name; Command = $line.Substring(4) }
                }
            }
        }
    }
}

function Invoke-Step([string]$directory, [string]$command) {
    Write-Host "[$directory] $command" -ForegroundColor Cyan
    $words = $command -split '\s+'
    Push-Location $directory
    try {
        & $words[0] $words[1..($words.Length - 1)]
        if ($LASTEXITCODE -ne 0) { throw "command failed with exit code ${LASTEXITCODE}: $command" }
    } finally {
        Pop-Location
    }
}

$started = Get-Date

if (-not $FiguresOnly) {
    Invoke-Step $compute 'cargo build --release'
    foreach ($item in Get-Commands 'cargo run') {
        if ($item.Experiment -eq 'benchmark' -and -not $Benchmark) {
            Write-Host "[skip] benchmark measurements (use -Benchmark)" -ForegroundColor Yellow
            continue
        }
        Invoke-Step $compute "$($item.Command) --force"
    }
}

Invoke-Step $analysis 'uv sync --locked'
foreach ($item in Get-Commands 'uv run') {
    Invoke-Step $analysis $item.Command
}

$elapsed = (Get-Date) - $started
Write-Host ("Done in {0:hh\:mm\:ss}. Check the changes with: git status" -f $elapsed) -ForegroundColor Green
