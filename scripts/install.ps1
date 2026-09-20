# Install Anvira Runtime (Windows). Installs the runtime only - NO models are downloaded.
#   .\scripts\install.ps1                      # from this checkout
#   .\scripts\install.ps1 -Source C:\path\anvira-runtime-1.0.0.zip
param([string]$Source = "", [switch]$Yes)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $py) { Write-Error "Python 3.10+ is required. Install it from https://www.python.org/downloads/ and re-run."; exit 3 }

if (-not $Yes) {
    $answer = Read-Host "Install Anvira Runtime (a shared local AI runtime for Anvira apps)? [y/N]"
    if ($answer -notmatch '^(y|yes)$') { Write-Host "Cancelled. Nothing was changed."; exit 0 }
}
if (-not $Source) { $Source = $repo }
$env:PYTHONPATH = Join-Path $repo "sdk\python"
& $py -m anvira_client install --source $Source
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done. Next:  anvira runtime start   |   anvira doctor   |   anvira ui"
