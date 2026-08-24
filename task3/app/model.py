"""Load Task-1 churn model artifacts and run inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "best_model.pkl"
PREPROCESSOR_PATH = ROOT / "model" / "preprocessor.pkl"
METRICS_PATH = ROOT / "model" / "metrics.csv"

FEATURE_ORDER = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

_model = None
_model_name: str | None = None
_preprocessor = None
_metrics: dict | None = None


def _load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    df = pd.read_csv(METRICS_PATH)
    row = df.sort_values(by=["ROC-AUC", "Accuracy"], ascending=False).iloc[0]
    return {
        "model_name": str(row["Model"]),
        "roc_auc": float(row["ROC-AUC"]),
        "accuracy": float(row["Accuracy"]),
        "recall": float(row["Recall"]),
        "precision": float(row["Precision"]),
        "f1": float(row["F1 Score"]),
    }


def load_model() -> None:
    global _model, _model_name, _preprocessor, _metrics

    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Missing model/best_model.pkl or model/preprocessor.pkl "
            "(copy from Customer-Churn-Prediction Task 1)."
        )

    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict) and "model" in artifact:
        _model = artifact["model"]
        _model_name = artifact.get("model_name", "Unknown")
    else:
        _model = artifact
        _model_name = "Unknown"

    _preprocessor = joblib.load(PREPROCESSOR_PATH)
    _metrics = _load_metrics()
    if _model_name == "Unknown" and _metrics.get("model_name"):
        _model_name = _metrics["model_name"]


def get_model_meta() -> dict:
    if _model is None:
        load_model()
    return {
        "model_name": _model_name,
        "roc_auc": (_metrics or {}).get("roc_auc"),
        "accuracy": (_metrics or {}).get("accuracy"),
        "recall": (_metrics or {}).get("recall"),
        "precision": (_metrics or {}).get("precision"),
        "f1": (_metrics or {}).get("f1"),
        "features": FEATURE_ORDER,
    }


def predict(features: dict, threshold: float = 0.5) -> dict:
    if _model is None or _preprocessor is None:
        load_model()

    row = {k: features[k] for k in FEATURE_ORDER}
    df = pd.DataFrame([row])
    return predict_dataframe(df, threshold=threshold)[0]


def predict_dataframe(df: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    if _model is None or _preprocessor is None:
        load_model()

    work = df.copy()
    for col in ("customerID", "Churn"):
        if col in work.columns:
            work = work.drop(columns=[col])

    missing = [c for c in FEATURE_ORDER if c not in work.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    work = work[FEATURE_ORDER].copy()
    work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce").fillna(0.0)
    work["SeniorCitizen"] = work["SeniorCitizen"].astype(int)
    work["tenure"] = work["tenure"].astype(int)
    work["MonthlyCharges"] = work["MonthlyCharges"].astype(float)

    X = _preprocessor.transform(work)
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)[:, 1]
    else:
        proba = _model.predict(X).astype(float)

    results = []
    for p in proba:
        prediction = int(float(p) >= threshold)
        results.append(
            {
                "churn_prediction": prediction,
                "churn_label": "Churn" if prediction == 1 else "No Churn",
                "churn_probability": round(float(p), 4),
                "model_name": _model_name or "Unknown",
            }
        )
    return results

