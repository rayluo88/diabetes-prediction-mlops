"""
Diabetes Dataset Preprocessing Pipeline

Comprehensive data preprocessing for diabetes prediction including
feature engineering, scaling, and validation.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DiabetesPreprocessor:
    """Diabetes dataset preprocessing pipeline."""

    def __init__(self):
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_names = None
        self.is_fitted = False

    def clean_data(self, df):
        """Clean and validate diabetes dataset."""
        logger.info("Cleaning diabetes dataset...")

        df_clean = df.copy()

        # Replace 0 values with NaN for medical features where 0 is impossible
        zero_invalid_features = [
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
        ]
        for feature in zero_invalid_features:
            if feature in df_clean.columns:
                df_clean[feature] = df_clean[feature].replace(0, np.nan)

        # Log missing values
        missing_counts = df_clean.isnull().sum()
        for feature, count in missing_counts.items():
            if count > 0:
                logger.info(
                    f"Missing values in {feature}: {count} ({count / len(df_clean) * 100:.1f}%)"
                )

        return df_clean

    def engineer_features(self, df):
        """Create new features for diabetes prediction."""
        logger.info("Engineering features...")

        df_engineered = df.copy()

        # BMI categories
        df_engineered["bmi_category"] = pd.cut(
            df_engineered["mass"],
            bins=[0, 18.5, 25, 30, 100],
            labels=["underweight", "normal", "overweight", "obese"],
            include_lowest=True,
        )

        # Age groups
        df_engineered["age_group"] = pd.cut(
            df_engineered["age"],
            bins=[0, 30, 50, 100],
            labels=["young", "middle", "older"],
            include_lowest=True,
        )

        # Glucose categories
        df_engineered["glucose_category"] = pd.cut(
            df_engineered["plas"],
            bins=[0, 100, 125, 200],
            labels=["normal", "prediabetic", "diabetic"],
            include_lowest=True,
        )

        # Interaction features
        df_engineered["bmi_age_interaction"] = (
            df_engineered["bmi"] * df_engineered["age"]
        )
        df_engineered["glucose_bmi_ratio"] = df_engineered["glucose"] / (
            df_engineered["bmi"] + 1e-8
        )

        # Pregnancy risk (for females)
        df_engineered["high_pregnancy_risk"] = (
            df_engineered["pregnancies"] > 5
        ).astype(int)

        # Convert categorical features to dummy variables
        categorical_features = ["bmi_category", "age_group", "glucose_category"]
        df_engineered = pd.get_dummies(
            df_engineered, columns=categorical_features, prefix=categorical_features
        )

        logger.info(f"Features after engineering: {df_engineered.shape[1]}")

        return df_engineered

    def fit(self, X, y=None):
        """Fit preprocessing pipeline."""
        logger.info("Fitting preprocessing pipeline...")

        # Clean data
        X_clean = self.clean_data(X)

        # Engineer features
        X_engineered = self.engineer_features(X_clean)

        # Separate numerical features for scaling
        numerical_features = X_engineered.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if "target" in numerical_features:
            numerical_features.remove("target")

        # Fit imputer and scaler on numerical features
        X_numerical = X_engineered[numerical_features]
        X_imputed = self.imputer.fit_transform(X_numerical)
        self.scaler.fit(X_imputed)

        self.feature_names = X_engineered.columns.tolist()
        if "target" in self.feature_names:
            self.feature_names.remove("target")

        self.is_fitted = True
        logger.info("Preprocessing pipeline fitted successfully")

        return self

    def transform(self, X):
        """Transform data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        logger.info("Transforming data...")

        # Clean data
        X_clean = self.clean_data(X)

        # Engineer features
        X_engineered = self.engineer_features(X_clean)

        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in X_engineered.columns:
                X_engineered[feature] = 0

        # Select and order features consistently
        X_engineered = X_engineered[self.feature_names]

        # Separate numerical features
        numerical_features = X_engineered.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        # Transform numerical features
        X_numerical = X_engineered[numerical_features]
        X_imputed = self.imputer.transform(X_numerical)
        X_scaled = self.scaler.transform(X_imputed)

        # Create final dataframe
        X_final = X_engineered.copy()
        X_final[numerical_features] = X_scaled

        logger.info(f"Data transformed: {X_final.shape}")

        return X_final

    def fit_transform(self, X, y=None):
        """Fit and transform data in one step."""
        return self.fit(X, y).transform(X)

    def save(self, filepath):
        """Save fitted preprocessor."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to: {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load fitted preprocessor."""
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from: {filepath}")
        return preprocessor


def preprocess_diabetes_data(train_path, test_path):
    """
    Main preprocessing function for diabetes dataset.

    Args:
        train_path (str): Path to training dataset
        test_path (str): Path to test dataset

    Returns:
        tuple: Preprocessed train and test data
    """
    logger.info("Starting diabetes data preprocessing...")

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Training data: {train_df.shape}")
    logger.info(f"Test data: {test_df.shape}")

    # Separate features and target
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    # Initialize and fit preprocessor
    preprocessor = DiabetesPreprocessor()
    preprocessor.fit(X_train)

    # Transform datasets
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save preprocessor
    preprocessor_path = "models/artifacts/diabetes_preprocessor.pkl"
    preprocessor.save(preprocessor_path)

    # Save processed datasets
    os.makedirs("data/processed", exist_ok=True)

    train_processed = pd.concat(
        [X_train_processed, y_train.reset_index(drop=True)], axis=1
    )
    test_processed = pd.concat(
        [X_test_processed, y_test.reset_index(drop=True)], axis=1
    )

    train_processed_path = "data/processed/diabetes_train_processed.csv"
    test_processed_path = "data/processed/diabetes_test_processed.csv"

    train_processed.to_csv(train_processed_path, index=False)
    test_processed.to_csv(test_processed_path, index=False)

    logger.info(f"Processed training data saved: {train_processed_path}")
    logger.info(f"Processed test data saved: {test_processed_path}")

    return train_processed, test_processed, preprocessor


if __name__ == "__main__":
    # Process diabetes data
    train_path = "data/processed/diabetes_train.csv"
    test_path = "data/processed/diabetes_test.csv"

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_processed, test_processed, preprocessor = preprocess_diabetes_data(
            train_path, test_path
        )
        print("\n✅ Diabetes data preprocessing completed!")
        print(f"📊 Processed training data shape: {train_processed.shape}")
        print(f"📊 Processed test data shape: {test_processed.shape}")
    else:
        print("❌ Please run load_diabetes_data.py first to generate the datasets")
