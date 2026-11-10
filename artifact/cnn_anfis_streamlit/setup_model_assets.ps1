param(
    [Parameter(Mandatory=$true)]
    [string] $SourceDir
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$dest = Join-Path $here "model_assets"
$expected = @{
    "e2a50_best_efficientnet_b0.pth" = "2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9"
    "anfis_r16_seed52.pth" = "020663d064b6e7cb5b12b70d84974dd652c7d78c3759d1f696ade08de786dc18"
    "train_feature_scaler.npz" = "290c5e6b418bdfcb8fe5436d313cfecdc8661f5fbc5447a4529b2e6d889fb43f"
}
New-Item -ItemType Directory -Force -Path $dest | Out-Null
foreach ($filename in $expected.Keys) {
    $source = Join-Path $SourceDir $filename
    $target = Join-Path $dest $filename
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing original source asset: $source" }
    $actual = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected[$filename]) {
        throw "STOP: source checkpoint/scaler fingerprint differs from the frozen source: $filename"
    }
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        $existing = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($existing -ne $expected[$filename]) {
            throw "STOP: existing app model asset differs from expected frozen source; refusing overwrite: $target"
        }
    } else {
        Copy-Item -LiteralPath $source -Destination $target
    }
    Write-Host "PASS: verified original $filename"
}
Write-Host "PASS: original selected CNN / ANFIS / scaler assets are ready at $dest"
Write-Host "NOTE: Notebook12 original 18-asset pretest freeze remains untouched."
