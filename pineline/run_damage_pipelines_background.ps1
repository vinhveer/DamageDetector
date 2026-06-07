param(
    [int]$GdinoTileBatchSize = 2,
    [int]$GdinoServiceWorkers = 1,
    [int]$GdinoServiceQueueSize = 2,
    [int]$GdinoServiceBatchSize = 1,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

function Invoke-PipelineStep {
    param(
        [string]$Name,
        [string[]]$PythonArgs
    )

    Write-Output "=== $Name START $(Get-Date -Format o) ==="
    & python @PythonArgs
    $code = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
    Write-Output "=== $Name EXIT=$code $(Get-Date -Format o) ==="
    return $code
}

$commonArgs = @(
    "--device", $Device,
    "--gdino-service-workers", [string]$GdinoServiceWorkers,
    "--gdino-tile-batch-size", [string]$GdinoTileBatchSize,
    "--gdino-service-queue-size", [string]$GdinoServiceQueueSize,
    "--gdino-service-batch-size", [string]$GdinoServiceBatchSize
)

$dinoArgs = @("-m", "pineline.dino_cutout.step1_gdino_detect.run") + $commonArgs
$dinoExit = Invoke-PipelineStep -Name "dino_cutout" -PythonArgs $dinoArgs
if ($dinoExit -ne 0) {
    Write-Output "Skipping house_cutout because dino_cutout failed."
    exit $dinoExit
}

$houseArgs = @("-m", "pineline.house_cutout.step2_gdino_detect.run") + $commonArgs
$houseExit = Invoke-PipelineStep -Name "house_cutout_step2" -PythonArgs $houseArgs
exit $houseExit
