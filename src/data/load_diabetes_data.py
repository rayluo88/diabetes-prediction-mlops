"""
Diabetes Dataset Loader from UCI ML Repository

This module loads the UCI Pima Indian Diabetes dataset using scikit-learn
and saves it to the local data directory for MLOps pipeline processing.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_pima_diabetes_dataset():
    """
    Load the Pima Indian Diabetes dataset from OpenML.

    Returns:
        pd.DataFrame: Complete dataset with features and target
    """
    logger.info("Loading Pima Indian Diabetes dataset from OpenML...")

    try:
        # Load Pima Indian Diabetes dataset (dataset_id: 37)
        diabetes = fetch_openml(data_id=37, as_frame=True, parser="auto")

        # Create DataFrame
        df = diabetes.frame

        # Rename target column for consistency
        if "class" in df.columns:
            df = df.rename(columns={"class": "target"})

        # Convert target to binary (0/1)
        df["target"] = (df["target"] == "tested_positive").astype(int)

        logger.info(f"Dataset loaded successfully: {df.shape}")
        logger.info(f"Features: {list(df.columns[:-1])}")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")

        return df

    except Exception as e:
        logger.error(f"Error loading from OpenML: {e}")
        logger.info("Creating synthetic diabetes dataset as fallback...")
        return create_synthetic_diabetes_dataset()


def create_synthetic_diabetes_dataset():
    """
    Create a synthetic diabetes dataset matching UCI specifications.

    Returns:
        pd.DataFrame: Synthetic diabetes dataset
    """
    np.random.seed(42)
    n_samples = 768

    # Generate realistic medical features
    data = {
        "pregnancies": np.random.poisson(3, n_samples).clip(0, 17),
        "glucose": np.random.normal(120, 30, n_samples).clip(0, 200),
        "blood_pressure": np.random.normal(80, 15, n_samples).clip(0, 122),
        "skin_thickness": np.random.exponential(20, n_samples).clip(0, 99),
        "insulin": np.random.exponential(80, n_samples).clip(0, 846),
        "bmi": np.random.normal(32, 8, n_samples).clip(15, 67),
        "diabetes_pedigree_function": np.random.exponential(0.5, n_samples).clip(
            0.08, 2.42
        ),
        "age": np.random.gamma(2, 15, n_samples).clip(21, 81).astype(int),
    }

    df = pd.DataFrame(data)

    # Create target based on realistic medical correlations
    diabetes_risk = (
        0.02 * df["glucose"]
        + 0.01 * df["bmi"]
        + 0.005 * df["age"]
        + 2 * df["diabetes_pedigree_function"]
        + np.random.normal(0, 5, n_samples)
    )

    df["target"] = (diabetes_risk > np.percentile(diabetes_risk, 65)).astype(int)

    return df


def save_dataset_splits(df):
    """
    Save dataset as train/test splits with proper directory structure.

    Args:
        df (pd.DataFrame): Complete dataset

    Returns:
        dict: Paths to saved files
    """
    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Save raw dataset
    raw_path = "data/raw/diabetes_dataset.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw dataset saved: {raw_path}")

    # Create stratified train/test split
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save training set
    train_df = pd.concat([X_train, y_train], axis=1)
    train_path = "data/processed/diabetes_train.csv"
    train_df.to_csv(train_path, index=False)

    # Save test set
    test_df = pd.concat([X_test, y_test], axis=1)
    test_path = "data/processed/diabetes_test.csv"
    test_df.to_csv(test_path, index=False)

    logger.info(f"Training set saved: {train_path} ({len(train_df)} samples)")
    logger.info(f"Test set saved: {test_path} ({len(test_df)} samples)")

    return {"raw": raw_path, "train": train_path, "test": test_path}


def generate_data_report(df):
    """Generate comprehensive data quality report."""
    logger.info("\n=== DATA QUALITY REPORT ===")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Duplicate rows: {df.duplicated().sum()}")

    logger.info("\n=== TARGET DISTRIBUTION ===")
    target_counts = df["target"].value_counts()
    for class_label, count in target_counts.items():
        percentage = (count / len(df)) * 100
        label = "Diabetic" if class_label == 1 else "Non-diabetic"
        logger.info(f"{label}: {count} ({percentage:.1f}%)")

    logger.info("\n=== FEATURE STATISTICS ===")
    for col in df.columns[:-1]:
        stats = df[col].describe()
        logger.info(
            f"{col}: min={stats['min']:.1f}, max={stats['max']:.1f}, mean={stats['mean']:.1f}"
        )


def main():
    """Main execution function."""
    logger.info("Starting diabetes dataset loading pipeline...")

    # Load dataset
    df = load_pima_diabetes_dataset()

    # Generate data quality report
    generate_data_report(df)

    # Save dataset splits
    file_paths = save_dataset_splits(df)

    logger.info("\n✅ Diabetes dataset pipeline completed successfully!")
    logger.info("📁 Files created:")
    for key, path in file_paths.items():
        logger.info(f"  {key}: {path}")

    return file_paths


if __name__ == "__main__":
    main()
