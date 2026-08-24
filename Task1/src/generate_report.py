"""
Generate a professional PDF report from project metrics and results.

Usage:
    python src/generate_report.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils import get_paths


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontSize=22,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1a365d"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            parent=styles["Heading1"],
            fontSize=14,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor("#1a365d"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHead",
            parent=styles["Heading2"],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#2c5282"),
        )
    )
    return styles


def _img(path: Path, width=6.2 * inch):
    if not path.exists():
        return Paragraph(f"<i>[Image not found: {path.name}]</i>", _styles()["BodyJust"])
    # Preserve aspect roughly
    return Image(str(path), width=width, height=width * 0.55)


def _metrics_table(metrics_path: Path):
    if not metrics_path.exists():
        return Paragraph("<i>Metrics file not available.</i>", _styles()["BodyJust"])
    df = pd.read_csv(metrics_path, index_col=0)
    header = ["Model"] + list(df.columns)
    data = [header]
    for idx, row in df.iterrows():
        data.append([str(idx)] + [f"{v:.4f}" for v in row.values])
    table = Table(data, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.Color(0.9, 0.93, 0.97)]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def generate_pdf(output_path: Path | None = None) -> Path:
    paths = get_paths()
    out = output_path or (paths["root"] / "report.pdf")
    styles = _styles()
    images = paths["images"]
    metrics_path = paths["root"] / "model" / "metrics.csv"

    story = []

    # Cover
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("Customer Churn Prediction<br/>using Machine Learning", styles["CoverTitle"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            "A Supervised Classification Project on the Telco Customer Churn Dataset",
            ParagraphStyle("sub", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11),
        )
    )
    story.append(Spacer(1, 0.5 * inch))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y')}",
            ParagraphStyle("date", parent=styles["Normal"], alignment=TA_CENTER),
        )
    )
    story.append(PageBreak())

    # 1. Introduction
    story.append(Paragraph("1. Introduction", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Customer churn — the loss of subscribers who discontinue a service — is one of "
            "the most critical business problems in the telecommunications industry. Acquiring "
            "a new customer typically costs several times more than retaining an existing one. "
            "This project builds an end-to-end supervised machine learning pipeline that predicts "
            "whether a telecom customer is likely to churn, enabling proactive retention strategies.",
            styles["BodyJust"],
        )
    )

    # 2. Problem Statement
    story.append(Paragraph("2. Problem Statement", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Given demographic, account, and service-usage attributes of a telecom customer, "
            "predict a binary outcome: <b>Churn = Yes</b> or <b>Churn = No</b>. The solution must "
            "be accurate, interpretable, and production-ready (serialized model + reusable scripts).",
            styles["BodyJust"],
        )
    )

    # 3. Dataset Description
    story.append(Paragraph("3. Dataset Description", styles["SectionHead"]))
    story.append(
        Paragraph(
            "The project uses the publicly available <b>Telco Customer Churn</b> dataset "
            "(IBM / Kaggle). It contains approximately 7,043 customer records and 21 columns "
            "covering demographics (gender, senior citizen, partner, dependents), services "
            "(phone, internet, streaming, support add-ons), billing (contract, payment method, "
            "monthly/total charges), and the binary target <b>Churn</b>.",
            styles["BodyJust"],
        )
    )
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Rows: ~7,043 customers", styles["BodyJust"])),
                ListItem(Paragraph("Features: 19 predictors + 1 target", styles["BodyJust"])),
                ListItem(Paragraph("Target imbalance: ~27% churn rate", styles["BodyJust"])),
                ListItem(Paragraph("Source: Kaggle / IBM Telco Customer Churn", styles["BodyJust"])),
            ],
            bulletType="bullet",
        )
    )

    # 4. Data Cleaning
    story.append(Paragraph("4. Data Cleaning", styles["SectionHead"]))
    story.append(
        Paragraph(
            "TotalCharges was coerced to numeric (blank strings for brand-new customers "
            "became missing values and those rows were dropped). Duplicate records were removed. "
            "The identifier column <b>customerID</b> was discarded. The target label was mapped "
            "from Yes/No to 1/0. Categorical features were one-hot encoded and numeric features "
            "were standardized with StandardScaler via a sklearn ColumnTransformer.",
            styles["BodyJust"],
        )
    )

    # 5. EDA
    story.append(Paragraph("5. Exploratory Data Analysis", styles["SectionHead"]))
    story.append(
        Paragraph(
            "EDA revealed a moderate class imbalance, strong associations between short tenure / "
            "month-to-month contracts and churn, and higher monthly charges among churners. "
            "Fiber optic internet users show elevated churn relative to DSL or no-internet customers.",
            styles["BodyJust"],
        )
    )
    story.append(Paragraph("Target Distribution", styles["SubHead"]))
    story.append(_img(images / "target_distribution.png", width=5.5 * inch))
    story.append(Paragraph("Correlation Heatmap", styles["SubHead"]))
    story.append(_img(images / "heatmap.png", width=5.8 * inch))
    story.append(PageBreak())
    story.append(Paragraph("Contract Type vs Churn", styles["SubHead"]))
    story.append(_img(images / "contract_vs_churn.png", width=5.5 * inch))
    story.append(Paragraph("Internet Service vs Churn", styles["SubHead"]))
    story.append(_img(images / "internet_service_vs_churn.png", width=5.5 * inch))

    # 6. Feature Engineering
    story.append(Paragraph("6. Feature Engineering", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Features (X) and target (y) were separated. An 80:20 stratified train-test split "
            "preserved the churn ratio. All encoding and scaling were fit on the training set "
            "only to prevent data leakage. Five-fold stratified cross-validation was used to "
            "estimate generalization before final hold-out evaluation.",
            styles["BodyJust"],
        )
    )

    # 7. Models
    story.append(Paragraph("7. Models Used", styles["SectionHead"]))
    story.append(
        Paragraph(
            "The following classifiers were trained and compared: Logistic Regression, "
            "Random Forest, Decision Tree, Support Vector Machine (RBF), K-Nearest Neighbors, "
            "and XGBoost (when available). Class weights were applied where supported to mitigate "
            "class imbalance.",
            styles["BodyJust"],
        )
    )

    # 8. Cross Validation
    story.append(Paragraph("8. Cross Validation", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Stratified 5-Fold Cross Validation was applied on the training set for Accuracy "
            "and ROC-AUC. Mean and standard deviation of fold scores are saved in "
            "<b>model/cv_scores.csv</b> for reproducibility.",
            styles["BodyJust"],
        )
    )

    # 9. Performance Comparison
    story.append(Paragraph("9. Performance Comparison", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Each model was evaluated on the hold-out test set using Accuracy, Precision, "
            "Recall, F1 Score, and ROC-AUC. Confusion matrices and ROC curves were generated "
            "for visual comparison.",
            styles["BodyJust"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(_metrics_table(metrics_path))
    story.append(PageBreak())

    story.append(Paragraph("ROC Curves", styles["SubHead"]))
    story.append(_img(images / "roc_curve.png", width=5.5 * inch))
    story.append(Paragraph("Confusion Matrices", styles["SubHead"]))
    story.append(_img(images / "confusion_matrix.png", width=6.0 * inch))

    # 10. Results / Feature Importance
    story.append(Paragraph("10. Results &amp; Feature Importance", styles["SectionHead"]))
    story.append(
        Paragraph(
            "The best model was selected automatically using ROC-AUC as the primary metric "
            "and Accuracy as the tie-breaker, then serialized to <b>model/best_model.pkl</b>. "
            "Random Forest feature importances highlight tenure, MonthlyCharges, TotalCharges, "
            "and contract-related indicators among the strongest predictors of churn.",
            styles["BodyJust"],
        )
    )
    story.append(_img(images / "feature_importance.png", width=5.5 * inch))

    # 11. Conclusion
    story.append(Paragraph("11. Conclusion", styles["SectionHead"]))
    story.append(
        Paragraph(
            "This project delivers a complete, GitHub-ready churn prediction system: cleaned "
            "data, exploratory analysis, multiple classifiers, rigorous evaluation, feature "
            "importance insights, and a deployable saved model with a prediction CLI. The "
            "pipeline demonstrates that supervised learning can effectively identify at-risk "
            "telecom customers for targeted retention campaigns.",
            styles["BodyJust"],
        )
    )

    # 12. Future Scope
    story.append(Paragraph("12. Future Scope", styles["SectionHead"]))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Hyperparameter tuning with GridSearchCV / Optuna", styles["BodyJust"])),
                ListItem(Paragraph("Handle imbalance with SMOTE or cost-sensitive learning", styles["BodyJust"])),
                ListItem(Paragraph("Deploy as a REST API (FastAPI) or Streamlit dashboard", styles["BodyJust"])),
                ListItem(Paragraph("Add customer lifetime value (CLV) estimation", styles["BodyJust"])),
                ListItem(Paragraph("Monitor model drift in production", styles["BodyJust"])),
            ],
            bulletType="bullet",
        )
    )

    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Customer Churn Prediction Report",
    )
    doc.build(story)
    print(f"PDF report written to {out}")
    return out


if __name__ == "__main__":
    generate_pdf()
