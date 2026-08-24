"""
Model training module for Customer Churn Prediction.

Trains multiple classifiers, runs 5-fold cross-validation,
selects the best model by ROC-AUC (with Accuracy as tie-breaker),
and persists the winning model to disk.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

from evaluate import (
    evaluate_model,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_model_performance,
    plot_roc_curves,
    print_comparison_table,
)
from preprocessing import run_preprocessing
from utils import (
    C,
    S,
    box,
    get_paths,
    info,
    ok,
    print_section,
    save_artifact,
    set_plot_style,
)

log = logging.getLogger("churn_train")

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False


def get_models(random_state: int = 42) -> dict[str, Any]:
    """Return classifiers to train and compare."""
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
            class_weight="balanced",
        ),
        "SVM": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=random_state, class_weight="balanced"),
            ensemble=False,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    return models


def run_cross_validation(
    models: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Stratified K-fold CV for every model."""
    print_section(f"{n_folds}-Fold Cross Validation")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    rows = []

    for name, model in models.items():
        print(f"  {C.DIM}• Cross-validating {name}...{C.RESET}", end="\r")
        acc_scores = cross_val_score(
            model, X, y, cv=skf, scoring="accuracy", n_jobs=1
        )
        auc_scores = cross_val_score(
            model, X, y, cv=skf, scoring="roc_auc", n_jobs=1
        )
        rows.append(
            {
                "Model": name,
                "CV Accuracy (mean)": acc_scores.mean(),
                "CV Accuracy (std)": acc_scores.std(),
                "CV ROC-AUC (mean)": auc_scores.mean(),
                "CV ROC-AUC (std)": auc_scores.std(),
            }
        )
        print(
            f"  {C.GREEN}{S.CHECK}{C.RESET} {name:<22}  "
            f"Acc={C.BOLD}{acc_scores.mean():.4f}{C.RESET}±{acc_scores.std():.4f}  "
            f"AUC={C.BOLD}{auc_scores.mean():.4f}{C.RESET}±{auc_scores.std():.4f}"
        )

    return pd.DataFrame(rows)


def train_and_evaluate(
    models: dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list] = None,
) -> tuple[pd.DataFrame, dict, str, Any]:
    """Fit every model, evaluate, and pick the best by ROC-AUC."""
    print_section("Training & Evaluating Models")
    set_plot_style()

    metrics_rows = []
    fitted: dict[str, Any] = {}
    y_probs: dict[str, np.ndarray] = {}
    y_preds: dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"\n  {C.CYAN}▶ Training {name}...{C.RESET}")
        model.fit(X_train, y_train)
        fitted[name] = model

        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test, name)
        metrics_rows.append(metrics)
        y_preds[name] = y_pred
        if y_proba is not None:
            y_probs[name] = y_proba

    metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
    print_comparison_table(metrics_df)

    ranked = metrics_df.sort_values(by=["ROC-AUC", "Accuracy"], ascending=False)
    best_name = ranked.index[0]
    best_model = fitted[best_name]
    best = ranked.loc[best_name]

    box(
        f"BEST MODEL → {best_name}",
        [
            f"ROC-AUC    {best['ROC-AUC']:.4f}",
            f"Accuracy   {best['Accuracy']:.4f}",
            f"Precision  {best['Precision']:.4f}",
            f"Recall     {best['Recall']:.4f}",
            f"F1 Score   {best['F1 Score']:.4f}",
        ],
    )

    images = get_paths()["images"]
    plot_model_performance(metrics_df, save_path=images / "model_performance.png")
    ok("Model performance dashboard saved")
    plot_roc_curves(y_test, y_probs, save_path=images / "roc_curve.png")
    ok("ROC curves saved")
    plot_confusion_matrices(y_test, y_preds, save_path=images / "confusion_matrix.png")
    ok("Confusion matrices saved")

    if "Random Forest" in fitted and feature_names is not None:
        plot_feature_importance(
            fitted["Random Forest"],
            feature_names,
            top_n=10,
            save_path=images / "feature_importance.png",
        )
        ok("Feature importance saved")

    return metrics_df, fitted, best_name, best_model


def save_best_model(model: Any, name: str) -> None:
    """Persist the best model as model/best_model.pkl."""
    paths = get_paths()
    artifact = {"model": model, "model_name": name}
    save_artifact(artifact, paths["best_model"])
    ok(f"Best model ('{name}') → model/best_model.pkl")


def main(skip_preprocess_banner: bool = False) -> None:
    """Run training from raw CSV to saved best model."""
    data = run_preprocessing(quiet=skip_preprocess_banner)
    models = get_models()

    y_train = np.asarray(data["y_train"], dtype=int)
    y_test = np.asarray(data["y_test"], dtype=int)

    info(f"Train size: {len(y_train):,}  |  Test size: {len(y_test):,}")
    info(f"Models: {', '.join(models.keys())}")

    cv_df = run_cross_validation(models, data["X_train_scaled"], y_train)

    metrics_df, fitted, best_name, best_model = train_and_evaluate(
        models,
        data["X_train_scaled"],
        data["X_test_scaled"],
        y_train,
        y_test,
        feature_names=data["feature_names"],
    )

    save_best_model(best_model, best_name)

    metrics_path = get_paths()["root"] / "model" / "metrics.csv"
    metrics_df.to_csv(metrics_path)
    cv_df.to_csv(get_paths()["root"] / "model" / "cv_scores.csv", index=False)
    ok("Metrics → model/metrics.csv")


if __name__ == "__main__":
    main()
