"""
Model evaluation module for Customer Churn Prediction.

Computes classification metrics, prints comparison tables, and
generates professional evaluation plots (ROC, confusion matrix,
feature importance, histograms, EDA charts).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils import (
    C,
    S,
    ensure_dir,
    get_paths,
    info,
    ok,
    print_section,
    print_table,
    set_plot_style,
)

log = logging.getLogger("churn_evaluate")


def _proba_ok(y_proba: Optional[np.ndarray]) -> bool:
    """Return True if the probability array is usable for ROC-AUC."""
    return y_proba is not None and len(np.unique(y_proba)) > 1


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> tuple[dict, np.ndarray, Optional[np.ndarray]]:
    """
    Evaluate a fitted classifier and return metrics + predictions.

    Returns
    -------
    tuple
        (metrics_dict, y_pred, y_proba_or_None)
    """
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": (
            roc_auc_score(y_test, y_proba) if _proba_ok(y_proba) else np.nan
        ),
    }

    print(f"\n  {C.BOLD}{model_name}{C.RESET}")
    print(f"  {'─' * 40}")
    for key, value in metrics.items():
        if key != "Model":
            print(f"    {key:<12} {C.BOLD}{value:.4f}{C.RESET}")
    print(f"\n  {C.DIM}Classification Report:{C.RESET}")
    report = classification_report(
        y_test, y_pred, target_names=["No Churn", "Churn"]
    )
    for line in report.splitlines():
        print(f"    {line}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {C.DIM}Confusion Matrix:{C.RESET}  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

    return metrics, y_pred, y_proba


def print_comparison_table(metrics_df: pd.DataFrame) -> None:
    """Pretty-print a side-by-side metrics comparison table."""
    print_section("Model Comparison Table")
    headers = ["Model"] + list(metrics_df.columns)
    best_auc = metrics_df["ROC-AUC"].idxmax()
    rows = []
    for name, row in metrics_df.iterrows():
        label = f"{S.STAR} {name}" if name == best_auc else f"  {name}"
        rows.append([label] + [f"{v:.4f}" for v in row.values])
    # Find ROC-AUC column index for highlight
    highlight = headers.index("ROC-AUC") if "ROC-AUC" in headers else -1
    print_table(headers, rows, highlight_col=highlight)
    info(f"Winner (highest ROC-AUC): {best_auc}")


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------
def plot_roc_curves(
    y_test: np.ndarray,
    y_probs: dict[str, np.ndarray],
    save_path: Optional[Path] = None,
) -> None:
    """Plot ROC curves for every model that produced probability estimates."""
    set_plot_style()
    plt.figure(figsize=(9, 7))

    for name, proba in y_probs.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Customer Churn Models")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path)
        log.info("Saved ROC curve -> %s", save_path)
    plt.close()


def plot_confusion_matrices(
    y_test: np.ndarray,
    y_preds: dict[str, np.ndarray],
    save_path: Optional[Path] = None,
) -> None:
    """Plot a grid of confusion matrices for all models."""
    set_plot_style()
    n = len(y_preds)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (name, pred) in zip(axes, y_preds.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        ax.set_title(name)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path)
        log.info("Saved confusion matrices -> %s", save_path)
    plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: list,
    top_n: int = 10,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Plot and return the top-N feature importances from a tree-based model.

    Parameters
    ----------
    model : Any
        Fitted model exposing ``feature_importances_``.
    feature_names : list
        Names corresponding to each feature index.
    top_n : int
        Number of top features to display.
    save_path : Path, optional
        Where to save the figure.

    Returns
    -------
    pd.DataFrame
        Top-N features ranked by importance.
    """
    if not hasattr(model, "feature_importances_"):
        log.warning("Model has no feature_importances_ attribute.")
        return pd.DataFrame()

    set_plot_style()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    # Strip ColumnTransformer prefixes for readability
    importances.index = (
        importances.index.str.replace(r"^(num|cat)__", "", regex=True)
    )
    top = importances.nlargest(top_n).sort_values()

    plt.figure(figsize=(9, 6))
    top.plot(kind="barh", color="#2E86AB")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances — Random Forest")
    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path)
        log.info("Saved feature importance -> %s", save_path)
    plt.close()

    print_section(f"Top {top_n} Important Features (Random Forest)")
    ranked = importances.nlargest(top_n)
    for i, (feat, imp) in enumerate(ranked.items(), 1):
        bar = S.BAR * max(1, int(imp * 40))
        print(f"  {i:2d}. {feat:<35s} {imp:.4f}  {C.CYAN}{bar}{C.RESET}")

    return ranked.reset_index().rename(
        columns={"index": "Feature", 0: "Importance"}
    )


def plot_model_performance(
    metrics_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    primary: str = "Accuracy",
) -> None:
    """
    Save a single report-ready dashboard: Customer Churn Model Performance.

    Highlights the model with the highest ``primary`` metric (default Accuracy)
    and compares all classifiers side by side.
    """
    set_plot_style()
    df = metrics_df.copy()
    if "Model" in df.columns:
        df = df.set_index("Model")

    best_name = df[primary].idxmax()
    best = df.loc[best_name]

    navy = "#0F2C59"
    teal = "#1B7F79"
    gold = "#C9A227"
    slate = "#4A5568"

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    fig.suptitle(
        "Customer Churn Model Performance",
        fontsize=20,
        fontweight="bold",
        color=navy,
        y=0.97,
    )
    fig.text(
        0.5,
        0.915,
        f"Best by {primary}:  {best_name}    |    Hold-out test set (20% stratified)",
        ha="center",
        fontsize=11,
        color=slate,
    )

    # KPI cards
    kpi_ax = fig.add_axes([0.06, 0.58, 0.88, 0.28])
    kpi_ax.axis("off")
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    values = [float(best[c]) for c in labels]
    card_w, gap = 0.175, 0.03
    start_x = 0.02
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = start_x + i * (card_w + gap)
        rect = plt.Rectangle(
            (x, 0.12),
            card_w,
            0.78,
            transform=kpi_ax.transAxes,
            facecolor="#F4F7FB",
            edgecolor="#D0D7E2",
            linewidth=1.2,
            clip_on=False,
        )
        kpi_ax.add_patch(rect)
        kpi_ax.text(
            x + card_w / 2,
            0.72,
            lab.upper(),
            transform=kpi_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color=slate,
            fontweight="bold",
        )
        display = f"{val * 100:.1f}%" if lab != "ROC-AUC" else f"{val:.3f}"
        kpi_ax.text(
            x + card_w / 2,
            0.38,
            display,
            transform=kpi_ax.transAxes,
            ha="center",
            va="center",
            fontsize=22,
            color=navy if lab != primary else teal,
            fontweight="bold",
        )

    # Accuracy comparison bars
    ax = fig.add_axes([0.10, 0.12, 0.82, 0.40])
    acc = df["Accuracy"].sort_values()
    colors = [gold if idx == best_name else teal for idx in acc.index]
    bars = ax.barh(acc.index, acc.values * 100, color=colors, height=0.62, edgecolor="none")
    ax.set_xlabel("Accuracy (%)", color=slate)
    ax.set_xlim(0, 100)
    ax.set_title("Accuracy by Model", loc="left", color=navy, fontsize=13, pad=8)
    ax.axvline(95, color="#C0564A", linestyle="--", linewidth=1.2, label="95% target")
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, acc.values):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            fontsize=10,
            color=navy,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.03,
        "Source: Telco Customer Churn hold-out test set (20% stratified split)",
        ha="center",
        fontsize=8,
        color=slate,
    )

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, facecolor="white")
        log.info("Saved model performance dashboard -> %s", save_path)
    plt.close()


# ---------------------------------------------------------------------------
# Exploratory Data Analysis plots
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, images_dir: Optional[Path] = None) -> None:
    """
    Generate all required EDA visualizations and save them to ``images/``.

    Plots produced
    --------------
    - Target class distribution
    - Histograms of numeric features
    - Boxplots of numeric features by churn
    - Correlation heatmap
    - Pairplot of key numerics
    - Monthly Charges vs Churn
    - Contract Type vs Churn
    - Internet Service vs Churn
    """
    set_plot_style()
    images = Path(images_dir) if images_dir else get_paths()["images"]
    ensure_dir(images)

    print_section("Exploratory Data Analysis")
    colors = ["#4C72B0", "#DD8452"]

    # --- Target distribution ---
    plt.figure(figsize=(7, 5))
    counts = df["Churn"].value_counts().sort_index()
    plot_df = pd.DataFrame(
        {"Churn": ["No Churn", "Churn"], "Count": counts.values}
    )
    ax = sns.barplot(
        data=plot_df,
        x="Churn",
        y="Count",
        hue="Churn",
        palette=colors,
        legend=False,
    )
    for i, v in enumerate(counts.values):
        ax.text(i, v + 40, f"{v} ({v / len(df):.1%})", ha="center")
    plt.title("Target Class Distribution (Churn)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(images / "target_distribution.png")
    plt.close()

    # --- Histograms ---
    numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric = [c for c in numeric if c != "Churn"]
    n = len(numeric)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric):
        sns.histplot(df[col], kde=True, ax=ax, color="#2E86AB")
        ax.set_title(col)
    fig.suptitle("Numeric Feature Histograms", y=1.02)
    plt.tight_layout()
    plt.savefig(images / "histogram.png")
    plt.close()
    log.info("Saved histograms")

    # --- Boxplots ---
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric):
        data_no = df.loc[df["Churn"] == 0, col]
        data_yes = df.loc[df["Churn"] == 1, col]
        bp = ax.boxplot(
            [data_no, data_yes],
            tick_labels=["No", "Yes"],
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel("Churn")
        ax.set_title(f"{col} by Churn")
    fig.suptitle("Boxplots of Numeric Features by Churn", y=1.02)
    plt.tight_layout()
    plt.savefig(images / "boxplot.png")
    plt.close()

    # --- Correlation heatmap ---
    # Encode categoricals temporarily for a numeric correlation view
    df_corr = df.copy()
    for col in df_corr.select_dtypes(include=["object"]).columns:
        df_corr[col] = pd.factorize(df_corr[col])[0]
    plt.figure(figsize=(14, 11))
    corr = df_corr.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.4,
        annot_kws={"size": 7},
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(images / "heatmap.png")
    plt.close()
    log.info("Saved correlation heatmap")

    # --- Pairplot (key numeric features) ---
    pair_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "Churn"] if c in df.columns]
    pair_df = df[pair_cols].copy()
    pair_df["Churn"] = pair_df["Churn"].map({0: "No", 1: "Yes"})
    g = sns.pairplot(
        pair_df,
        hue="Churn",
        palette={"No": "#4C72B0", "Yes": "#DD8452"},
        diag_kind="kde",
        corner=True,
    )
    g.fig.suptitle("Pairplot of Key Numeric Features", y=1.02)
    g.savefig(images / "pairplot.png")
    plt.close("all")
    log.info("Saved pairplot")

    # --- Monthly Charges vs Churn ---
    plt.figure(figsize=(8, 5))
    data_no = df.loc[df["Churn"] == 0, "MonthlyCharges"]
    data_yes = df.loc[df["Churn"] == 1, "MonthlyCharges"]
    bp = plt.boxplot(
        [data_no, data_yes],
        tick_labels=["No Churn", "Churn"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.title("Monthly Charges vs Churn")
    plt.ylabel("MonthlyCharges")
    plt.tight_layout()
    plt.savefig(images / "monthly_charges_vs_churn.png")
    plt.close()

    # --- Contract Type vs Churn ---
    plt.figure(figsize=(9, 5))
    contract_churn = (
        pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    )
    contract_churn.columns = ["No Churn", "Churn"]
    contract_churn.plot(kind="bar", stacked=False, color=colors, ax=plt.gca())
    plt.title("Contract Type vs Churn Rate (%)")
    plt.ylabel("Percentage")
    plt.xlabel("Contract")
    plt.xticks(rotation=15)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(images / "contract_vs_churn.png")
    plt.close()

    # --- Internet Service vs Churn ---
    plt.figure(figsize=(9, 5))
    internet_churn = (
        pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
    )
    internet_churn.columns = ["No Churn", "Churn"]
    internet_churn.plot(kind="bar", stacked=False, color=colors, ax=plt.gca())
    plt.title("Internet Service vs Churn Rate (%)")
    plt.ylabel("Percentage")
    plt.xlabel("Internet Service")
    plt.xticks(rotation=15)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(images / "internet_service_vs_churn.png")
    plt.close()

    print(f"  EDA figures saved → {images}")
