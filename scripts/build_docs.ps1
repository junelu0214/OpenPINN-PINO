param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    & $Python scripts/build_docs.py
    if ($LASTEXITCODE -ne 0) { throw "Documentation pipeline failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

