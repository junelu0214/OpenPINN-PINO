param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$RunRoot = Join-Path $RepoRoot "runs/check_$RunStamp"
Push-Location $RepoRoot
try {
    & $Python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "pytest failed with exit code $LASTEXITCODE" }

    & $Python -m openpinn run --problem poisson --method baseline --steps 5 --output-dir (Join-Path $RunRoot "poisson")
    if ($LASTEXITCODE -ne 0) { throw "Poisson smoke failed with exit code $LASTEXITCODE" }

    & $Python -m openpinn run --problem burgers --method baseline --steps 5 --output-dir (Join-Path $RunRoot "burgers")
    if ($LASTEXITCODE -ne 0) { throw "Burgers smoke failed with exit code $LASTEXITCODE" }

    & $Python -m openpinn run --problem burgers --method adaptive_sampling --variant residual_adaptive --steps 2 --output-dir (Join-Path $RunRoot "adaptive_sampling_variant")
    if ($LASTEXITCODE -ne 0) { throw "Adaptive-sampling variant smoke failed with exit code $LASTEXITCODE" }

    & $Python scripts/validate_method_configurator.py
    if ($LASTEXITCODE -ne 0) { throw "Method-configurator validation failed with exit code $LASTEXITCODE" }

    & $Python scripts/build_docs.py
    if ($LASTEXITCODE -ne 0) { throw "Documentation pipeline failed with exit code $LASTEXITCODE" }
    Write-Host "Smoke artifacts written to $RunRoot"
}
finally {
    Pop-Location
}
