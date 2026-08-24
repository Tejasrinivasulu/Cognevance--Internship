"""
Data preprocessing module for Customer Churn Prediction.

Handles loading, cleaning, encoding, and scaling of the Telco
Customer Churn dataset so that it is ready for model training.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from utils import get_paths, logger, print_section, save_artifact

# Re-export logger under module name for clarity
log = logging.getLogger("churn_preprocessing")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Telco Customer Churn CSV into a Pandas DataFrame.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV. Defaults to ``dataset/Telco-Customer-Churn.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    path = filepath or str(get_paths()["data_csv"])
    try:
        df = pd.read_csv(path)
        log.info("Loaded dataset from %s  | shape=%s", path, df.shape)
        return df
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Place Telco-Customer-Churn.csv inside the dataset/ folder."
        ) from exc


def display_dataset_info(df: pd.DataFrame, quiet: bool = False) -> None:
    """Print shape, dtypes, head, and summary statistics."""
    if quiet:
        return
    print_section("Dataset Information")
    print(f"  Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    print("\n  Column dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<22} {dtype}")
    print("\n  First 5 rows:")
    print(df.head().to_string(index=True))
    print_section("Summary Statistics")
    print(df.describe(include="all").T.to_string())
    print("\n  Missing values per column:")
    for col, val in df.isnull().sum().items():
        print(f"    {col:<22} {val}")


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco dataset.

    Steps
    -----
    1. Convert ``TotalCharges`` to numeric (blank strings become NaN).
    2. Drop rows with missing values.
    3. Remove duplicate records.
    4. Drop the ``customerID`` identifier column.
    5. Encode the binary target ``Churn`` as 0 / 1.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()

    # TotalCharges often contains whitespace for brand-new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    log.info("Dropped rows with missing values (NaNs removed: %s)", missing_before)

    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    log.info("Removed %s duplicate record(s)", dup_count)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        log.info("Dropped column: customerID")

    # Map target to binary integers
    if df["Churn"].dtype == object or str(df["Churn"].dtype).startswith("str"):
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df["Churn"] = df["Churn"].astype(int)

    # SeniorCitizen is already 0/1 but ensure int type
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    log.info("Cleaned dataset shape: %s", df.shape)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Feature / target split & encoding helpers
# ---------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify numeric and categorical feature columns (excluding target).

    Returns
    -------
    tuple[list, list]
        (numeric_cols, categorical_cols)
    """
    target = "Churn"
    feature_df = df.drop(columns=[target])

    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # SeniorCitizen is numeric but semantically categorical; keep as numeric
    # (already 0/1) so StandardScaler can handle it.
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: list,
    categorical_cols: list,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that one-hot encodes categoricals and
    standard-scales numerical features.

    Parameters
    ----------
    numeric_cols : list
        Names of numeric columns.
    categorical_cols : list
        Names of categorical columns.

    Returns
    -------
    ColumnTransformer
        Fitted-ready sklearn transformer.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def encode_with_label_encoder(df: pd.DataFrame, categorical_cols: list) -> Tuple[pd.DataFrame, dict]:
    """
    Alternative encoding path using LabelEncoder for each categorical column.

    Useful for tree-based models and for EDA-friendly numeric DataFrames.

    Returns
    -------
    tuple
        (encoded DataFrame, dict of fitted LabelEncoders)
    """
    df_enc = df.copy()
    encoders: dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


# ---------------------------------------------------------------------------
# 4. Full pipeline helpers
# ---------------------------------------------------------------------------
def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and target (y)."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """
    Perform an 80:20 stratified train-test split.

    Stratification preserves the churn class ratio in both sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def fit_transform_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, list]:
    """
    Fit the preprocessor on training data and transform both splits.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, fitted_preprocessor, feature_names)
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Recover readable feature names after one-hot encoding
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:  # pragma: no cover - older sklearn fallback
        feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

    if save_path:
        save_artifact(preprocessor, save_path)

    log.info(
        "Transformed features | train=%s  test=%s  n_features=%s",
        X_train_t.shape,
        X_test_t.shape,
        len(feature_names),
    )
    return X_train_t, X_test_t, preprocessor, feature_names


def run_preprocessing(
    filepath: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_preprocessor: bool = True,
    quiet: bool = False,
) -> dict:
    """
    End-to-end preprocessing pipeline.

    Returns a dictionary with cleaned data, splits, scaled arrays,
    preprocessor, and feature metadata — ready for training.
    """
    if not quiet:
        print_section("Running Preprocessing Pipeline")

    df_raw = load_data(filepath)
    display_dataset_info(df_raw, quiet=quiet)

    df = clean_data(df_raw)
    numeric_cols, categorical_cols = get_feature_columns(df)

    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    preproc_path = str(get_paths()["preprocessor"]) if save_preprocessor else None
    X_train_t, X_test_t, preprocessor, feature_names = fit_transform_features(
        X_train, X_test, numeric_cols, categorical_cols, save_path=preproc_path
    )

    return {
        "df": df,
        "df_raw": df_raw,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_t,
        "X_test_scaled": X_test_t,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print_section("Preprocessing Complete")
    print(f"Train shape : {result['X_train_scaled'].shape}")
    print(f"Test shape  : {result['X_test_scaled'].shape}")
    print(f"Churn rate  : {result['y'].mean():.2%}")
