"""Train and evaluate the injury-risk model (time-based split)."""
import json
import logging

import numpy as np
import pandas as pd

import config
from src.storage.duckdb_io import read_parquet
from src.features.build_features import FEATURE_COLS
from src.utils.logging import setup_logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
import joblib

FEATURES_DIR = config.DATA_DIR / "features"
REPORTS_DIR = config.ROOT / "reports"
TRAIN_SEASONS = ["2022-23", "2023-24"]
TEST_SEASONS = ["2024-25"]

log = logging.getLogger("train")


def _split(tbl):
    tbl = tbl[tbl["ACWR"].notna()].copy()
    train = tbl[tbl["SEASON"].isin(TRAIN_SEASONS)]
    test = tbl[tbl["SEASON"].isin(TEST_SEASONS)]
    Xtr = train[FEATURE_COLS].astype(float)
    Xte = test[FEATURE_COLS].astype(float)
    ytr = train["TARGET"].astype(int)
    yte = test["TARGET"].astype(int)
    return Xtr, Xte, ytr, yte, test


def _evaluate(name, model, Xte, yte):
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "precision": round(precision_score(yte, pred, zero_division=0), 3),
        "recall": round(recall_score(yte, pred, zero_division=0), 3),
        "f1": round(f1_score(yte, pred, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(yte, proba), 3),
        "pr_auc": round(average_precision_score(yte, proba), 3),
    }


def main():
    setup_logging()
    tbl = read_parquet(FEATURES_DIR / "model_table.parquet")
    Xtr, Xte, ytr, yte, test = _split(tbl)

    base_rate = yte.mean()
    log.info("train %d rows (%.1f%% pos) | test %d rows (%.1f%% pos)",
             len(Xtr), 100 * ytr.mean(), len(Xte), 100 * base_rate)

    base = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    base.fit(Xtr, ytr)

    pos_weight = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    sample_weight = np.where(ytr == 1, pos_weight, 1.0)
    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=42,
    )
    gbm.fit(Xtr, ytr, sample_weight=sample_weight)

    results = [
        _evaluate("logreg_baseline", base, Xte, yte),
        _evaluate("hist_gbm", gbm, Xte, yte),
    ]

    print("\n" + "=" * 68)
    print(f"INJURY-RISK MODEL — test on {TEST_SEASONS[0]} "
          f"(base rate {base_rate:.1%})")
    print("=" * 68)
    print(f"{'model':<18}{'precision':>10}{'recall':>9}{'f1':>7}{'roc_auc':>9}{'pr_auc':>8}")
    print("-" * 68)
    for r in results:
        print(f"{r['model']:<18}{r['precision']:>10}{r['recall']:>9}"
              f"{r['f1']:>7}{r['roc_auc']:>9}{r['pr_auc']:>8}")
    print("-" * 68)
    print(f"PR-AUC lift over base rate (gbm): "
          f"{results[1]['pr_auc'] - base_rate:.3f}")

    cm = confusion_matrix(yte, (gbm.predict_proba(Xte)[:, 1] >= 0.5).astype(int))
    print(f"\nGBM confusion matrix @0.5  [[TN FP][FN TP]]:\n{cm}")

    perm = permutation_importance(
        gbm, Xte, yte, scoring="average_precision",
        n_repeats=10, random_state=42,
    )
    order = np.argsort(perm.importances_mean)[::-1]
    print("\nTop features (permutation importance, PR-AUC drop):")
    for i in order[:8]:
        print(f"  {FEATURE_COLS[i]:<18} {perm.importances_mean[i]:+.4f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(gbm, REPORTS_DIR / "gbm_model.joblib")
    metrics = {
        "test_season": TEST_SEASONS[0],
        "base_rate": round(float(base_rate), 4),
        "results": results,
        "top_features": [
            {"feature": FEATURE_COLS[i],
             "importance": round(float(perm.importances_mean[i]), 4)}
            for i in order[:8]
        ],
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("saved model + metrics to %s", REPORTS_DIR)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        feats = [FEATURE_COLS[i] for i in order[::-1]]
        vals = [perm.importances_mean[i] for i in order[::-1]]
        plt.figure(figsize=(7, 5))
        plt.barh(feats, vals, color="#2a78d6")
        plt.xlabel("Permutation importance (PR-AUC drop)")
        plt.title("Injury-risk model — feature importance")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=140)
        log.info("saved feature_importance.png")
    except Exception as e:
        log.info("skipped importance plot (%s)", e)


if __name__ == "__main__":
    main()
