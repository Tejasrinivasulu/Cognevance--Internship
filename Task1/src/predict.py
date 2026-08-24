"""
Prediction module for Customer Churn Prediction.

Loads the saved best model and preprocessor to score new customer records.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from utils import C, banner, get_paths, load_artifact, ok, print_section


def load_best_model(model_path: Optional[str] = None) -> tuple:
    """Load the persisted best model artifact. Returns (model, model_name)."""
    path = model_path or str(get_paths()["best_model"])
    artifact = load_artifact(path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact.get("model_name", "Unknown")
    return artifact, "Unknown"


def load_preprocessor(preprocessor_path: Optional[str] = None):
    """Load the fitted ColumnTransformer used during training."""
    path = preprocessor_path or str(get_paths()["preprocessor"])
    return load_artifact(path)


def predict_churn(
    data: Union[pd.DataFrame, dict, str],
    model_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict churn for one or more customers."""
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("data must be a DataFrame, dict, or CSV path (str).")

    for col in ("customerID", "Churn"):
        if col in df.columns:
            df = df.drop(columns=[col])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    model, model_name = load_best_model(model_path)
    preprocessor = load_preprocessor(preprocessor_path)

    try:
        X = preprocessor.transform(df)
    except Exception as exc:
        raise ValueError(
            "Failed to transform input features. "
            "Ensure columns match the training schema.\n"
            f"Details: {exc}"
        ) from exc

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X).astype(float)

    preds = (proba >= threshold).astype(int)

    result = df.copy()
    result["churn_probability"] = np.round(proba, 4)
    result["churn_prediction"] = preds
    result["churn_label"] = result["churn_prediction"].map(
        {0: "No Churn", 1: "Churn"}
    )

    ok(f"Predictions with '{model_name}' for {len(result)} record(s)")
    return result


def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Predict telecom customer churn using the saved best model."
    )
    parser.add_argument("--input", "-i", required=True, help="CSV with customer features.")
    parser.add_argument("--output", "-o", default=None, help="Save predictions CSV.")
    parser.add_argument("--threshold", "-t", type=float, default=0.5)
    args = parser.parse_args(argv)

    banner("CUSTOMER CHURN — PREDICT", "Score customers with the saved best model")
    results = predict_churn(args.input, threshold=args.threshold)

    print_section("Sample Predictions")
    preview = results[["churn_probability", "churn_prediction", "churn_label"]].head(15)
    print(preview.to_string())

    churn_n = int(results["churn_prediction"].sum())
    print()
    print(
        f"  {C.BOLD}Summary:{C.RESET} {churn_n} / {len(results)} predicted to churn "
        f"({churn_n / len(results):.1%})"
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
        ok(f"Results saved → {out}")


if __name__ == "__main__":
    main()
