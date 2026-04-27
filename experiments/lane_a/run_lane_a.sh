#!/usr/bin/env bash
# =============================================================================
# Lane A — MEDS-native supervised benchmark with meds-torch
#
# Prerequisites on the server:
#   pip install meds-torch MEDS-transforms
#
# Steps:
#   1. Preprocess MIMIC-IV with MEDS-transforms (K-MIMIC is already done)
#   2. Train on K-MIMIC (within-dataset)
#   3. Train on MIMIC-IV  (within-dataset, used for cross-cohort in Lane B)
# =============================================================================

set -euo pipefail

# ------------- paths (edit if needed) ---------------------------------------
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
KMIMIC_MEDS_DIR="$REPO_DIR/data/triplet_tensors"
MIMIC_RAW_DIR="$REPO_DIR/data/MEDS_cohort"
MIMIC_TENSOR_DIR="$REPO_DIR/data/mimic_triplet_tensors"

TASK_KMIMIC="inhospital_mortality_24h"
TASK_MIMIC="mortality/in_hospital/first_24h"

RESULTS_DIR="$REPO_DIR/experiments/lane_a/results"
mkdir -p "$RESULTS_DIR"

# ------------- Step 1: preprocess MIMIC with MEDS-transforms ----------------
# Skip if already done (directory exists and has the tensorization flag).
if [ -f "$MIMIC_TENSOR_DIR/.logs/tensorization.done" ]; then
    echo "[Step 1] MIMIC preprocessing already done — skipping."
else
    echo "[Step 1] Preprocessing MIMIC-IV with MEDS-transforms..."
    export MEDS_DIR="$MIMIC_RAW_DIR"
    export MODEL_DIR="$MIMIC_TENSOR_DIR"
    MEDS_transforms-runner \
        --config-dir "$REPO_DIR/experiments/lane_a/configs" \
        --config-name triplet_preproc \
        'hydra.searchpath=[pkg://MEDS_transforms.configs]'
    echo "[Step 1] Done."
fi

# ------------- Step 2: train on K-MIMIC ------------------------------------
echo
echo "[Step 2] Training on K-MIMIC (within-dataset)..."
meds-torch-train \
    --config-dir "$REPO_DIR/experiments/lane_a/configs" \
    --config-name kmimic_train \
    MEDS_cohort_dir="$KMIMIC_MEDS_DIR" \
    task_name="$TASK_KMIMIC" \
    output_dir="$RESULTS_DIR/kmimic" \
    seed=42
echo "[Step 2] Done. Results in $RESULTS_DIR/kmimic"

# ------------- Step 3: train on MIMIC-IV ------------------------------------
echo
echo "[Step 3] Training on MIMIC-IV..."
meds-torch-train \
    --config-dir "$REPO_DIR/experiments/lane_a/configs" \
    --config-name mimic_train \
    MEDS_cohort_dir="$MIMIC_TENSOR_DIR" \
    task_name="$TASK_MIMIC" \
    output_dir="$RESULTS_DIR/mimic" \
    seed=42
echo "[Step 3] Done. Results in $RESULTS_DIR/mimic"

echo
echo "All Lane A runs complete."
echo "Results: $RESULTS_DIR"
