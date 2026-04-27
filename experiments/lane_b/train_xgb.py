"""
Lane B — XGBoost training and evaluation for cross-cohort mortality prediction.

Runs two evaluation scenarios:

  Within-dataset  : train on K-MIMIC train,   evaluate on K-MIMIC held_out
  Cross-cohort    : train on MIMIC train,      evaluate on K-MIMIC held_out

Reports for each run:
  Primary   : AUROC, AUPRC
  Secondary : Brier score, calibration slope
  Subgroup  : AUROC by sex (M/F) and by age band (18-44, 45-64, 65-74, 75+)

Outputs are written to --output_dir:
  metrics.json           complete metric table
  predictions_*.parquet  predictions for each run × split

Usage:
    python experiments/lane_b/train_xgb.py \\
        --kmimic_features  experiments/lane_b/features/kmimic \\
        --mimic_features   experiments/lane_b/features/mimic \\
        --output_dir       experiments/lane_b/results
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

AGE_BANDS = [(18, 45, "18-44"), (45, 65, "45-64"), (65, 75, "65-74"), (75, 999, "75+")]

# XGBoost hyperparameters — tuned for moderate-size tabular medical data.
# scale_pos_weight is set at training time to handle class imbalance.
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(features_dir: Path, split: str) -> pd.DataFrame | None:
    path = features_dir / f"features_{split}.parquet"
    if not path.exists():
        log.warning(f"Missing: {path}")
        return None
    return pd.read_parquet(path)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"subject_id", "prediction_time", "label", "split", "age", "sex_male"}
    return [c for c in df.columns if c not in exclude]


def prepare_xy(df: pd.DataFrame):
    feature_cols = get_feature_cols(df)
    # Include age and sex_male as features (they may be in df separately)
    extra = [c for c in ["age", "sex_male"] if c in df.columns]
    all_cols = extra + feature_cols
    X = df[all_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    return X, y, all_cols


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_prob, label: str = "") -> dict:
    metrics = {}
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        log.warning(f"{label}: no variation in labels, skipping metrics.")
        return {"auroc": None, "auprc": None, "brier": None, "calib_slope": None,
                "n": int(len(y_true)), "n_pos": int(y_true.sum())}

    metrics["auroc"]  = float(roc_auc_score(y_true, y_prob))
    metrics["auprc"]  = float(average_precision_score(y_true, y_prob))
    metrics["brier"]  = float(brier_score_loss(y_true, y_prob))
    metrics["n"]      = int(len(y_true))
    metrics["n_pos"]  = int(y_true.sum())
    metrics["prevalence"] = float(y_true.mean())

    # Calibration slope (logistic regression of logit(prob) on label)
    try:
        from sklearn.linear_model import LogisticRegression
        eps = 1e-6
        logit = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))
        lr = LogisticRegression(fit_intercept=True, max_iter=1000)
        lr.fit(logit.reshape(-1, 1), y_true)
        metrics["calib_slope"] = float(lr.coef_[0][0])
    except Exception:
        metrics["calib_slope"] = None

    return metrics


def subgroup_metrics(df_test: pd.DataFrame, y_prob: np.ndarray) -> dict:
    results = {}
    df_test = df_test.copy()
    df_test["_prob"] = y_prob

    # Sex subgroups
    if "sex_male" in df_test.columns:
        for sex_val, sex_label in [(1.0, "male"), (0.0, "female")]:
            mask = df_test["sex_male"] == sex_val
            sub = df_test[mask]
            if len(sub) >= 10:
                results[f"sex_{sex_label}"] = compute_metrics(
                    sub["label"].values, sub["_prob"].values, label=f"sex_{sex_label}"
                )

    # Age band subgroups
    if "age" in df_test.columns:
        for lo, hi, band_label in AGE_BANDS:
            mask = (df_test["age"] >= lo) & (df_test["age"] < hi)
            sub = df_test[mask]
            if len(sub) >= 10:
                results[f"age_{band_label}"] = compute_metrics(
                    sub["label"].values, sub["_prob"].values, label=f"age_{band_label}"
                )

    return results


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_xgb(
    X_train, y_train,
    X_val=None, y_val=None,
) -> XGBClassifier:
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = float(neg / max(pos, 1))
    log.info(f"  Train: {len(y_train):,} samples, {int(pos)} positives "
             f"(scale_pos_weight={spw:.1f})")

    params = dict(XGB_PARAMS, scale_pos_weight=spw)
    model = XGBClassifier(**params)

    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )
    return model


def run_experiment(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    log.info(f"\n{'='*60}")
    log.info(f"Experiment: {name}")
    log.info(f"{'='*60}")

    X_train, y_train, cols = prepare_xy(train_df)

    X_val, y_val = None, None
    if val_df is not None and len(val_df) > 0:
        X_val, y_val, _ = prepare_xy(val_df)

    X_test, y_test, _ = prepare_xy(test_df)

    model = train_xgb(X_train, y_train, X_val, y_val)

    # Predictions on test set
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Metrics
    overall = compute_metrics(y_test, y_prob_test, label=name)
    subgroups = subgroup_metrics(test_df, y_prob_test)

    log.info(f"  AUROC={overall.get('auroc', 'N/A'):.4f}  "
             f"AUPRC={overall.get('auprc', 'N/A'):.4f}  "
             f"Brier={overall.get('brier', 'N/A'):.4f}")
    for sg_name, sg_m in subgroups.items():
        if sg_m.get("auroc") is not None:
            log.info(f"  [{sg_name}] AUROC={sg_m['auroc']:.4f}  n={sg_m['n']}")

    # Save predictions
    preds_df = test_df[["subject_id", "prediction_time", "label"]].copy()
    preds_df["prob"] = y_prob_test
    preds_df.to_parquet(output_dir / f"predictions_{name}.parquet", index=False)

    # Feature importance (top 20)
    importance = dict(zip(cols, model.feature_importances_))
    top20 = dict(sorted(importance.items(), key=lambda x: -x[1])[:20])
    log.info(f"  Top features: {list(top20.keys())[:10]}")

    return {
        "name": name,
        "overall": overall,
        "subgroups": subgroups,
        "feature_importance_top20": top20,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost within-dataset and cross-cohort experiments."
    )
    parser.add_argument("--kmimic_features", required=True,
                        help="Directory with K-MIMIC features_{split}.parquet files.")
    parser.add_argument("--mimic_features",  default=None,
                        help="Directory with MIMIC features_{split}.parquet files "
                             "(required for cross-cohort run).")
    parser.add_argument("--output_dir", default="experiments/lane_b/results",
                        help="Directory for metrics.json and prediction files.")
    parser.add_argument("--skip_cross_cohort", action="store_true",
                        help="Skip the MIMIC→K-MIMIC transfer experiment.")
    args = parser.parse_args()

    kmimic_dir = Path(args.kmimic_features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load K-MIMIC splits -----------------------------------------------
    log.info("Loading K-MIMIC features...")
    km_train  = load_split(kmimic_dir, "train")
    km_tuning = load_split(kmimic_dir, "tuning")
    km_test   = load_split(kmimic_dir, "held_out")

    if km_train is None or km_test is None:
        raise FileNotFoundError("K-MIMIC train or held_out features not found.")

    log.info(f"  K-MIMIC train={len(km_train):,}  "
             f"tuning={len(km_tuning) if km_tuning is not None else 0:,}  "
             f"held_out={len(km_test):,}")

    all_results = []

    # ---- Experiment 1: within-dataset K-MIMIC ------------------------------
    result_km = run_experiment(
        name="kmimic_within",
        train_df=km_train,
        val_df=km_tuning,
        test_df=km_test,
        output_dir=output_dir,
    )
    all_results.append(result_km)

    # ---- Experiment 2: cross-cohort MIMIC → K-MIMIC ------------------------
    if not args.skip_cross_cohort:
        if args.mimic_features is None:
            log.warning("--mimic_features not provided; skipping cross-cohort run.")
        else:
            mimic_dir = Path(args.mimic_features)
            log.info("Loading MIMIC features (train split)...")
            m_train  = load_split(mimic_dir, "train")
            m_tuning = load_split(mimic_dir, "tuning")

            if m_train is None:
                log.warning("MIMIC train features not found; skipping cross-cohort.")
            else:
                log.info(f"  MIMIC train={len(m_train):,}")

                # Align feature columns (intersection) so MIMIC model can be
                # applied to K-MIMIC test features.
                km_feat_cols = set(get_feature_cols(km_test)) | {"age", "sex_male"}
                m_feat_cols  = set(get_feature_cols(m_train)) | {"age", "sex_male"}
                shared_cols  = sorted(km_feat_cols & m_feat_cols)
                log.info(f"  Shared feature columns: {len(shared_cols)} "
                         f"(K-MIMIC has {len(km_feat_cols)}, MIMIC has {len(m_feat_cols)})")

                def align(df):
                    extra = [c for c in shared_cols if c not in df.columns]
                    for c in extra:
                        df[c] = np.nan
                    return df

                m_train_aligned  = align(m_train[["subject_id", "prediction_time", "label", "split"] +
                                                  [c for c in shared_cols if c in m_train.columns]])
                km_test_aligned  = align(km_test[["subject_id", "prediction_time", "label", "split"] +
                                                  [c for c in shared_cols if c in km_test.columns]])
                m_tuning_aligned = align(m_tuning[["subject_id", "prediction_time", "label", "split"] +
                                                   [c for c in shared_cols if c in m_tuning.columns]]) \
                                   if m_tuning is not None else None

                result_cross = run_experiment(
                    name="mimic_to_kmimic",
                    train_df=m_train_aligned,
                    val_df=m_tuning_aligned,
                    test_df=km_test_aligned,
                    output_dir=output_dir,
                )
                all_results.append(result_cross)

    # ---- Save metrics ------------------------------------------------------
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nMetrics saved to {metrics_path}")

    # ---- Summary table ---------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  {'Experiment':<25} {'AUROC':>8} {'AUPRC':>8} {'Brier':>8} {'N_test':>8}")
    print("  " + "-" * 66)
    for r in all_results:
        m = r["overall"]
        auroc = f"{m['auroc']:.4f}" if m.get("auroc") is not None else "  N/A "
        auprc = f"{m['auprc']:.4f}" if m.get("auprc") is not None else "  N/A "
        brier = f"{m['brier']:.4f}" if m.get("brier") is not None else "  N/A "
        print(f"  {r['name']:<25} {auroc:>8} {auprc:>8} {brier:>8} {m['n']:>8,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
