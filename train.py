"""Train and evaluate Logistic Regression and Random Forest on the haor data.

Usage:  python train.py
Outputs (models/): one .joblib per model, metrics.json, cv_results.csv,
feature_importance.csv
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common

warnings.filterwarnings("ignore")
SEED = 42


def build_models() -> dict:
    # NDWI has 5 missing values -> fill with the median inside each pipeline
    # (done inside CV, so no information leaks from the test folds).
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000)),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, max_depth=5,
                                   min_samples_leaf=2, random_state=SEED)),
    }


def importance(model, features) -> pd.Series:
    est = model.steps[-1][1]
    if hasattr(est, "feature_importances_"):         # Random Forest
        vals = est.feature_importances_
    else:                                            # Logistic Regression
        vals = np.abs(est.coef_[0])
    return pd.Series(100 * vals / vals.sum(), index=features)


def main():
    df = common.load_raw()
    X, y = common.make_xy(df)
    common.MODEL_DIR.mkdir(exist_ok=True)
    print(f"Data: {len(X)} events, {int(y.sum())} flood / {int((1 - y).sum())} non-flood")
    print(f"Features ({X.shape[1]}): {list(X.columns)}\n")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    all_metrics, rows, imps = {}, [], {}
    for name, model in build_models().items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        m = {"model": name, "n_events": int(len(X)), "n_flood": int(y.sum()),
             "cv": "5-fold stratified, repeated 10 times"}
        for s in scoring:
            m[s] = float(res[f"test_{s}"].mean())
            m[f"{s}_std"] = float(res[f"test_{s}"].std())

        # out-of-fold predictions -> confusion matrix
        pred = cross_val_predict(model, X, y, cv=skf)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        m["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

        print(f"{name}\n  accuracy  {m['accuracy']:.3f} ± {m['accuracy_std']:.3f}"
              f"\n  precision {m['precision']:.3f} ± {m['precision_std']:.3f}"
              f"\n  recall    {m['recall']:.3f} ± {m['recall_std']:.3f}"
              f"\n  f1-score  {m['f1']:.3f} ± {m['f1_std']:.3f}"
              f"\n  roc-auc   {m['roc_auc']:.3f}"
              f"\n  confusion TN={tn} FP={fp} FN={fn} TP={tp}\n")

        model.fit(X, y)                       # final fit on all data
        imps[name] = importance(model, list(X.columns))
        joblib.dump({"model": model, "features": list(X.columns), "name": name},
                    common.model_path(name))
        all_metrics[name] = m
        rows.append({k: m[k] for k in ["model", "accuracy", "precision", "recall",
                                       "f1", "roc_auc"]})

    pd.DataFrame(rows).to_csv(common.CV_RESULTS_PATH, index=False)
    pd.DataFrame(imps).to_csv(common.IMPORTANCE_PATH)
    common.METRICS_PATH.write_text(json.dumps(all_metrics, indent=2))
    print(f"Saved models and metrics to {common.MODEL_DIR}/")


if __name__ == "__main__":
    main()
