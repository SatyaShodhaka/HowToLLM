<#
.SYNOPSIS
  Install uv (if needed) and sync project dependencies.

.PARAMETER Dev
  Also install dev dependencies.
#>
param(
  [switch] $Dev
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
Set-Location $projectRoot

# Install uv if not present
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "uv not found. Installing uv..."
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
  # Refresh PATH for this session
  $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
}

Write-Host "uv version: $(uv --version)"

# Initialize project if not already
if (-not (Test-Path "uv.lock")) {
  Write-Host "Initializing uv project..."
  uv init --no-readme 2>$null
}

# Sync dependencies
Write-Host "Syncing dependencies..."
if ($Dev) {
  uv sync --dev
} else {
  uv sync
}

Write-Host ""
Write-Host "Setup complete!"
Write-Host "To activate the environment:  & .\.venv\Scripts\Activate.ps1"
Write-Host "Or run commands via:          uv run <command>"