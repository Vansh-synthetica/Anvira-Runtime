<#
  Publish Anvira Runtime to GitHub: create the repo, push the source, build the packages, create the release.

  Run from the repository root (the "Finalized" folder) in PowerShell, after `gh auth login`:

      .\scripts\publish.ps1 -Owner <your-github-username> [-Private] [-LlamaCpu <dir>] [-LlamaCuda <dir>]

  * The repo is named Anvira-Runtime. It is created PRIVATE unless you pass -Public (apps download releases anonymously only
    when it is public).
  * Source goes to git; the packages (about 45 MB core + about 650 MB GPU pack) go to a GitHub Release as assets, because binaries
    never belong in git history.
  * -LlamaCpu / -LlamaCuda: folders holding llama-server builds (default: the ones Anvira already downloaded on this PC).
#>
param(
  [Parameter(Mandatory = $true)][string]$Owner,
  [switch]$Public,
  [string]$Repo = "Anvira-Runtime",
  [string]$LlamaCpu = "",
  [string]$LlamaCuda = "",
  [string]$Python = "python",
  [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

gh auth status 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Not logged in to GitHub. Run:  gh auth login" }

# 1. point the SDKs at this repository
$slug = "$Owner/$Repo"
foreach ($f in @("sdk\python\anvira_client\release.py", "sdk\typescript\index.js")) {
  (Get-Content $f -Raw).Replace("OWNER/Anvira-Runtime", $slug) | Set-Content $f -NoNewline -Encoding utf8
}

# 2. build the packages
$version = (Select-String -Path runtime\anvira_runtime\version.py -Pattern 'RUNTIME_VERSION\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
if (-not $SkipBuild) {
  $args = @("scripts\build_portable.py", "--out", "dist")
  if ($LlamaCpu) { $args += @("--llama-cpu", $LlamaCpu) }
  if ($LlamaCuda) { $args += @("--llama-cuda", $LlamaCuda) }
  & $Python @args
  if ($LASTEXITCODE -ne 0) { throw "package build failed" }
}

# 3. git: commit and push the source
if (-not (Test-Path .git)) { git init -b main | Out-Null }
git add -A
git -c user.name="$Owner" -c user.email="$Owner@users.noreply.github.com" commit -m "Anvira Runtime $version" | Out-Null
$vis = if ($Public) { "--public" } else { "--private" }
gh repo view $slug 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { gh repo create $slug $vis --source . --remote origin --description "Shared local runtime for Anvira apps: ORCHA + Nomi + AICL, models, CLI, SDKs" }
git push -u origin main

# 4. release with the packages
$assets = Get-ChildItem dist\AnviraRuntime-$version-*.zip, dist\SHA256SUMS, dist\manifest.json | ForEach-Object { $_.FullName }
gh release create "v$version" @assets --repo $slug --title "Anvira Runtime $version" --notes "Windows x64 self-contained package (own Python, ORCHA, Nomi, AICL, CPU llama-server). The -cuda zip adds NVIDIA GPU acceleration; unzip it into the same folder. SHA256SUMS lists checksums."
Write-Host "Done: https://github.com/$slug/releases/tag/v$version"
