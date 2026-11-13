$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    python -m agentgate run --tier smoke --agent config\agent-search.yaml --dataset dataset\golden_dataset.csv
    $agentgateExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $agentgateExitCode
