$ErrorActionPreference = 'Stop'

$BaseDir = "data/boltz_rcsb"
$TargetsDir = Join-Path $BaseDir "targets"
$MsaDir = Join-Path $BaseDir "msa"

New-Item -ItemType Directory -Force -Path $TargetsDir | Out-Null
New-Item -ItemType Directory -Force -Path $MsaDir | Out-Null

# URLs from Boltz training docs (update if they change).
$TargetsUrl = "https://example.com/rcsb_processed_targets.tar"
$MsaUrl = "https://example.com/rcsb_processed_msa.tar"
$SymmUrl = "https://example.com/symmetry.pkl"

$TargetsTar = Join-Path $BaseDir "rcsb_processed_targets.tar"
$MsaTar = Join-Path $BaseDir "rcsb_processed_msa.tar"
$SymmPath = Join-Path $BaseDir "symmetry.pkl"

Invoke-WebRequest -Uri $TargetsUrl -OutFile $TargetsTar
Invoke-WebRequest -Uri $MsaUrl -OutFile $MsaTar
Invoke-WebRequest -Uri $SymmUrl -OutFile $SymmPath

tar -xf $TargetsTar -C $TargetsDir
tar -xf $MsaTar -C $MsaDir

Write-Host "Boltz RCSB data downloaded to $BaseDir"
