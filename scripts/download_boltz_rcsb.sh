#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="data/boltz_rcsb"
TARGETS_DIR="$BASE_DIR/targets"
MSA_DIR="$BASE_DIR/msa"

mkdir -p "$TARGETS_DIR" "$MSA_DIR"

# URLs from Boltz training docs (update if they change).
TARGETS_URL="https://example.com/rcsb_processed_targets.tar"
MSA_URL="https://example.com/rcsb_processed_msa.tar"
SYMM_URL="https://example.com/symmetry.pkl"

curl -L "$TARGETS_URL" -o "$BASE_DIR/rcsb_processed_targets.tar"
curl -L "$MSA_URL" -o "$BASE_DIR/rcsb_processed_msa.tar"
curl -L "$SYMM_URL" -o "$BASE_DIR/symmetry.pkl"

tar -xf "$BASE_DIR/rcsb_processed_targets.tar" -C "$TARGETS_DIR"
tar -xf "$BASE_DIR/rcsb_processed_msa.tar" -C "$MSA_DIR"

echo "Boltz RCSB data downloaded to $BASE_DIR"
