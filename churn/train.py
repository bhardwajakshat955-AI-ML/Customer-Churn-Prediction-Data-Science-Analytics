"""Train, compare and persist churn models.

Usage:
    python -m churn.train                # real IBM Telco CSV in data/raw/
    python -m churn.train --use-sample   # synthetic demo data
"""
import argparse
import json
import logging
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config
from .data import load_clean
from .preprocessing import build_preprocessor
from .risk import segment

log = logging.getLogger("churn.train")


def candidate_models() -> dict[str, Pipeline]:
    rs = config.RANDOM_STATE
    return {
        "Logistic Regression": Pipeline([("prep", build_preprocessor(True)), ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=rs))]),
        "Decision Tree": Pipeline([("prep", build_preprocessor(False)), ("clf", DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=30, class_weight="balanced", random_state=rs))]),
        "Random Forest": Pipeline([("prep", build_preprocessor(False)), ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=rs))]),
        "Gradient Boosting": Pipeline([("prep", build_preprocessor(False)), ("clf", GradientBoostingClassifier(
            random_state=rs))]),
    }


def test_metrics(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def main(use_sample: bool = False) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df, data_path = load_clean(use_sample)
    log.info("Loaded %d rows from %s (churn rate %.1f%%)", len(df), data_path, 100 * df[config.TARGET].mean())

    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    results, fitted = {}, {}
    for name, pipe in candidate_models().items():
        # CV runs on the TRAINING split only; preprocessing is refit inside each fold.
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        pipe.fit(X_train, y_train)
        m = test_metrics(y_test, pipe.predict_proba(X_test)[:, 1])
        m.update(cv_roc_auc_mean=cv_auc.mean(), cv_roc_auc_std=cv_auc.std())
        results[name], fitted[name] = m, pipe
        log.info("%-20s CV AUC %.3f | test AUC %.3f recall %.3f", name, cv_auc.mean(), m["roc_auc"], m["recall"])

    # Selection uses cross-validated ROC-AUC on training data only (test set stays untouched).
    best = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    model = fitted[best]
    proba = model.predict_proba(X_test)[:, 1]
    log.info("Selected model: %s", best)

    config.MODELS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    meta = {
        "model_name": best,
        "selection_criterion": "highest 5-fold CV ROC-AUC on training split",
        "data_file": str(data_path.relative_to(config.ROOT)),
        "synthetic_data": bool(use_sample),
        "n_rows": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "churn_rate": float(y.mean()),
        "features": config.FEATURES,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "random_state": config.RANDOM_STATE,
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        "threshold": 0.5,
        "test_metrics": results[best],
    }
    config.METADATA_PATH.write_text(json.dumps(meta, indent=2))
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {"synthetic_data": bool(use_sample), "results": results, "selected": best}, indent=2))

    # Markdown table + classification report
    lines = ["| Model | CV ROC-AUC (mean ± std) | Test Accuracy | Precision | Recall | F1 | Test ROC-AUC |",
             "|---|---|---|---|---|---|---|"]
    for n, m in results.items():
        star = " ✅" if n == best else ""
        lines.append(f"| {n}{star} | {m['cv_roc_auc_mean']:.3f} ± {m['cv_roc_auc_std']:.3f} | {m['accuracy']:.3f} | "
                     f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} |")
    note = "> ⚠️ Computed on SYNTHETIC sample data.\n\n" if use_sample else ""
    report = classification_report(y_test, (proba >= 0.5).astype(int), target_names=["Stayed", "Churned"])
    (config.REPORTS_DIR / "metrics.md").write_text(
        f"# Model comparison\n\n{note}" + "\n".join(lines) +
        f"\n\nSelected: **{best}** (highest CV ROC-AUC). Test split: {len(X_test)} rows, threshold 0.5.\n\n"
        f"```\n{report}\n```\n\nConfusion matrix [[TN, FP], [FN, TP]]: "
        f"{confusion_matrix(y_test, (proba >= 0.5).astype(int)).tolist()}\n")

    # Evaluation plots for the selected model
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, (proba >= 0.5).astype(int), display_labels=["Stayed", "Churned"],
                                            cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix – {best}")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)
    for cls, fname, title in [(RocCurveDisplay, "roc_curve.png", "ROC curve"),
                              (PrecisionRecallDisplay, "precision_recall_curve.png", "Precision–recall curve")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        cls.from_predictions(y_test, proba, name=best, ax=ax)
        ax.set_title(f"{title} – {best}"); fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / fname, dpi=150); plt.close(fig)

    # Risk segmentation on the held-out test set
    seg_df = X_test.assign(**{config.TARGET: y_test.values, "churn_probability": proba})
    seg = segment(seg_df)
    seg.to_csv(config.REPORTS_DIR / "risk_segments.csv")
    log.info("Risk segments (test set):\n%s", seg.to_string())
    log.info("Saved model to %s and reports to %s", config.MODEL_PATH, config.REPORTS_DIR)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-sample", action="store_true", help="use synthetic demo data")
    main(ap.parse_args().use_sample)
