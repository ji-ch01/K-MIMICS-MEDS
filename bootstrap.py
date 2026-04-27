"""Bootstrap 95% CIs for Lane B (XGBoost) benchmark metrics.

Usage:
    python bootstrap.py

Reads saved predictions from experiments/lane_b/results/ and outputs
point estimates + 95% bootstrap CIs for AUROC, AUPRC, and Brier score.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

N_BOOTSTRAP = 2000
SEED = 42
rng = np.random.default_rng(SEED)


def bootstrap_ci(y_true, y_prob, metric_fn, n=N_BOOTSTRAP, ci=0.95):
    """Return (point_estimate, lower, upper) via percentile bootstrap."""
    point = metric_fn(y_true, y_prob)
    scores = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        yt, yp = y_true[sample], y_prob[sample]
        if len(np.unique(yt)) < 2:
            continue  # skip degenerate resamples (no positives or no negatives)
        scores.append(metric_fn(yt, yp))
    alpha = (1 - ci) / 2
    lo = np.percentile(scores, 100 * alpha)
    hi = np.percentile(scores, 100 * (1 - alpha))
    return point, lo, hi


def fmt(point, lo, hi):
    return f"{point:.3f} [{lo:.3f}-{hi:.3f}]"


models = {
    "XGBoost (K-MIMIC within)":    "experiments/lane_b/results/predictions_kmimic_within.parquet",
    "XGBoost (MIMIC-IV->K-MIMIC)":  "experiments/lane_b/results/predictions_mimic_to_kmimic.parquet",
}

print(f"Bootstrap 95% CIs  (n_bootstrap={N_BOOTSTRAP}, seed={SEED})\n")
print(f"{'Model':<30}  {'AUROC':^22}  {'AUPRC':^22}  {'Brier':^22}")
print("-" * 95)

results = {}
for name, path in models.items():
    df = pd.read_parquet(path)
    y_true = df["label"].values.astype(int)
    y_prob = df["prob"].values.astype(float)

    auroc  = bootstrap_ci(y_true, y_prob, roc_auc_score)
    auprc  = bootstrap_ci(y_true, y_prob, average_precision_score)
    brier  = bootstrap_ci(y_true, y_prob,
                          lambda yt, yp: brier_score_loss(yt, yp))

    results[name] = dict(auroc=auroc, auprc=auprc, brier=brier)
    print(f"{name:<30}  {fmt(*auroc):^22}  {fmt(*auprc):^22}  {fmt(*brier):^22}")

print()
print("Notes:")
print(f"  n_patients = {len(df)}, n_positives = {y_true.sum()}")
print("  Degenerate bootstrap resamples (no positives) are skipped.")
print("  meds-torch CIs not computed (no saved per-patient predictions).")

# Save for paper
import json
out = {}
for name, v in results.items():
    out[name] = {
        metric: {"point": round(vals[0], 4), "ci_lo": round(vals[1], 4), "ci_hi": round(vals[2], 4)}
        for metric, vals in v.items()
    }
with open("experiments/lane_b/results/bootstrap_ci.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved to experiments/lane_b/results/bootstrap_ci.json")
