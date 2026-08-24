"""
End-to-end pipeline runner with clear, attractive terminal output.

Usage:
    python src/run_pipeline.py
    run.bat
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import run_eda
from preprocessing import clean_data, load_data
from train import main as train_main
from utils import banner, finish, get_paths, ok, print_section, step


def main() -> None:
    """Run EDA, then train / evaluate / save the best model."""
    banner(
        "CUSTOMER CHURN PREDICTION",
        "Machine Learning Classification Pipeline",
    )

    paths = get_paths()
    total_steps = 4

    step(1, total_steps, "Loading & cleaning dataset")
    df_raw = load_data()
    ok(f"Loaded {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    df = clean_data(df_raw)
    ok(f"Cleaned shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    ok(f"Churn rate: {df['Churn'].mean():.1%}")

    step(2, total_steps, "Exploratory Data Analysis (saving plots)")
    run_eda(df, images_dir=paths["images"])
    ok(f"EDA figures saved → {paths['images']}")

    step(3, total_steps, "Training & evaluating models")
    train_main(skip_preprocess_banner=True)

    step(4, total_steps, "Finished")
    finish(paths)


if __name__ == "__main__":
    main()
