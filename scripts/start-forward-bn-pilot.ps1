$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repoRoot
try {
    conda run -n dra python -m src.forward_pilot serve --open
}
finally {
    Pop-Location
}
