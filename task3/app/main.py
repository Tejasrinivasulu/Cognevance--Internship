"""Flask API + multi-page UI for Customer Churn Prediction (Task 3)."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from app.model import FEATURE_ORDER, get_model_meta, load_model, predict, predict_dataframe

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "examples" / "samples"

app = Flask(__name__)


@app.get("/")
def landing():
    return render_template("landing.html")


@app.get("/predict-page")
def predict_page():
    return render_template("predict.html")


@app.get("/result")
def result_page():
    return render_template("result.html")


@app.get("/api")
def api_info():
    return jsonify(
        {
            "message": "Customer Churn Prediction API",
            "framework": "Flask",
            "ui_landing": "/",
            "ui_predict": "/predict-page",
            "ui_result": "/result",
            "health": "/health",
            "model": "/model",
            "predict": "POST /predict",
            "predict_csv": "POST /predict/csv",
            "sample_csv": "/samples/sample_customers.csv",
        }
    )


@app.get("/health")
def health():
    try:
        meta = get_model_meta()
        return jsonify(
            {
                "status": "ok",
                "model_loaded": True,
                "model_name": meta.get("model_name"),
                "roc_auc": meta.get("roc_auc"),
                "accuracy": meta.get("accuracy"),
            }
        )
    except FileNotFoundError:
        return jsonify({"status": "degraded", "model_loaded": False}), 503


@app.get("/model")
def model_info():
    try:
        return jsonify(get_model_meta())
    except FileNotFoundError as exc:
        return jsonify({"detail": str(exc)}), 503


@app.get("/samples/<path:filename>")
def samples(filename: str):
    return send_from_directory(SAMPLES_DIR, filename)


@app.get("/sample-data")
def sample_data_json():
    """Return one sample customer row for the single-record form."""
    path = SAMPLES_DIR / "sample_customers.csv"
    if not path.exists():
        return jsonify({"detail": "Sample CSV not found."}), 404
    row = pd.read_csv(path).iloc[0].to_dict()
    # Normalize types for JSON form
    row["SeniorCitizen"] = int(row["SeniorCitizen"])
    row["tenure"] = int(row["tenure"])
    row["MonthlyCharges"] = float(row["MonthlyCharges"])
    try:
        row["TotalCharges"] = float(row["TotalCharges"])
    except (TypeError, ValueError):
        row["TotalCharges"] = 0.0
    return jsonify(row)


@app.post("/predict")
def predict_churn():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"detail": "Send JSON body with customer features."}), 400

    missing = [k for k in FEATURE_ORDER if k not in payload]
    if missing:
        return jsonify({"detail": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        result = predict(payload)
    except FileNotFoundError as exc:
        return jsonify({"detail": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"detail": f"Prediction failed: {exc}"}), 400

    return jsonify(result)


@app.post("/predict/csv")
def predict_csv():
    if "file" not in request.files:
        return jsonify({"detail": "Upload a CSV file in form field 'file'."}), 400

    upload = request.files["file"]
    if not upload or not upload.filename:
        return jsonify({"detail": "Empty filename."}), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"detail": "Only .csv files are supported."}), 400

    try:
        raw = upload.read()
        df = pd.read_csv(io.BytesIO(raw))
        if df.empty:
            return jsonify({"detail": "CSV has no rows."}), 400
        if len(df) > 500:
            return jsonify({"detail": "Limit is 500 rows per upload."}), 400

        preds = predict_dataframe(df)
        out = df.copy()
        out["churn_prediction"] = [p["churn_prediction"] for p in preds]
        out["churn_label"] = [p["churn_label"] for p in preds]
        out["churn_probability"] = [p["churn_probability"] for p in preds]

        churn_n = int(out["churn_prediction"].sum())
        return jsonify(
            {
                "rows": int(len(out)),
                "churn_count": churn_n,
                "no_churn_count": int(len(out) - churn_n),
                "model_name": preds[0]["model_name"] if preds else None,
                "predictions": out.to_dict(orient="records"),
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"detail": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        return jsonify({"detail": f"CSV prediction failed: {exc}"}), 400


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8000, debug=False)
