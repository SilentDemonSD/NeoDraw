param([string]$Sdk = $env:RYZEN_AI_INSTALLATION_PATH)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$root = $PSScriptRoot
if (-not $Sdk) { $Sdk = Join-Path $root "sdk\PFiles64\RyzenAI\1.7.0" }
if (-not (Test-Path (Join-Path $Sdk "voe-4.0-win_amd64"))) {
    throw "Ryzen AI SDK 1.7.0 not found at '$Sdk'. Install it or pass -Sdk <path>."
}

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { uv venv --python 3.12 (Join-Path $root ".venv") }

$wheels = "onnxruntime_vitisai-*.whl", "voe-*.whl" | ForEach-Object { (Get-Item (Join-Path $Sdk $_)).FullName }
uv pip install --python $python @wheels -r (Join-Path $root "requirements.txt")
uv pip install --python $python --no-deps mediapipe==1.0.1

$site = Join-Path $root ".venv\Lib\site-packages"
Copy-Item (Join-Path $site "voe-*.data\data\lib\site-packages\onnxruntime\capi\*") (Join-Path $site "onnxruntime\capi") -Recurse -Force

$config = Join-Path $root "neodraw.toml"
if (-not (Test-Path $config) -or -not (Select-String -Path $config -Pattern '^\s*sdk\s*=' -Quiet)) {
    Add-Content -Path $config -Value "sdk = '$((Resolve-Path $Sdk).Path)'"
}

Push-Location $root
try {
    & $python -m neodraw fetch
    & $python -m tests.test_interaction
    & $python -m neodraw check
} finally {
    Pop-Location
}
