@echo off
setlocal enabledelayedexpansion

set BASE_DIR=data\boltz_rcsb
set TARGETS_DIR=%BASE_DIR%\targets
set MSA_DIR=%BASE_DIR%\msa

if not exist %TARGETS_DIR% mkdir %TARGETS_DIR%
if not exist %MSA_DIR% mkdir %MSA_DIR%

REM URLs from Boltz training docs (update if they change).
set TARGETS_URL=https://example.com/rcsb_processed_targets.tar
set MSA_URL=https://example.com/rcsb_processed_msa.tar
set SYMM_URL=https://example.com/symmetry.pkl

curl -L %TARGETS_URL% -o %BASE_DIR%\rcsb_processed_targets.tar
curl -L %MSA_URL% -o %BASE_DIR%\rcsb_processed_msa.tar
curl -L %SYMM_URL% -o %BASE_DIR%\symmetry.pkl

tar -xf %BASE_DIR%\rcsb_processed_targets.tar -C %TARGETS_DIR%
tar -xf %BASE_DIR%\rcsb_processed_msa.tar -C %MSA_DIR%

echo Boltz RCSB data downloaded to %BASE_DIR%
endlocal
