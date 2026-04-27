#!/usr/bin/env bash
# =============================================================================
# Lane B — XGBoost cross-cohort transfer baseline
#
# Prerequisites:
#   pip install xgboost scikit-learn pandas pyarrow pyyaml
#
# Steps:
#   1. Extract first-24h features from K-MIMIC (fast — 3 small files)
#   2. Extract first-24h features from MIMIC train split (292 shards — ~10 min)
#   3. Train XGBoost: within-dataset K-MIMIC + cross-cohort MIMIC → K-MIMIC
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
FEATURES_DIR="$REPO_DIR/experiments/lane_b/features"
RESULTS_DIR="$REPO_DIR/experiments/lane_b/results"

# ------------- Step 1: extract K-MIMIC features ----------------------------
echo "[Step 1] Extracting K-MIMIC features (all splits)..."
python "$REPO_DIR/experiments/lane_b/feature_extract.py" \
    --meds_dir      "$REPO_DIR/data/output" \
    --labels_path   "$REPO_DIR/data/labels/inhospital_mortality_24h" \
    --dataset       kmimic \
    --concepts      "$REPO_DIR/experiments/concepts.yaml" \
    --splits        train tuning held_out \
    --output_dir    "$FEATURES_DIR/kmimic"
echo "[Step 1] Done."

# ------------- Step 2: extract MIMIC features (train + tuning) -------------
echo
echo "[Step 2] Extracting MIMIC features (train + tuning — may take ~10 min)..."
python "$REPO_DIR/experiments/lane_b/feature_extract.py" \
    --meds_dir      "$REPO_DIR/data/MEDS_cohort" \
    --labels_path   "$REPO_DIR/data/MEDS_cohort/tasks/mortality/in_hospital/first_24h.parquet" \
    --splits_path   "$REPO_DIR/data/MEDS_cohort/metadata/subject_splits.parquet" \
    --dataset       mimic \
    --concepts      "$REPO_DIR/experiments/concepts.yaml" \
    --splits        train tuning \
    --output_dir    "$FEATURES_DIR/mimic"
echo "[Step 2] Done."

# ------------- Step 3: train XGBoost ----------------------------------------
echo
echo "[Step 3] Training XGBoost (within + cross-cohort)..."
python "$REPO_DIR/experiments/lane_b/train_xgb.py" \
    --kmimic_features "$FEATURES_DIR/kmimic" \
    --mimic_features  "$FEATURES_DIR/mimic" \
    --output_dir      "$RESULTS_DIR"
echo "[Step 3] Done. Results in $RESULTS_DIR/metrics.json"
