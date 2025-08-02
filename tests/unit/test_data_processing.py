"""
Unit Tests for Diabetes Data Processing Components

Comprehensive test suite for data loading, preprocessing, and validation.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from data.load_diabetes_data import (
    create_synthetic_diabetes_dataset,
    generate_data_report,
    save_dataset_splits,
)
from data.preprocess import DiabetesPreprocessor


class TestDiabetesDataLoading:
    """Test diabetes data loading functionality."""

    def test_create_synthetic_diabetes_dataset(self):
        """Test synthetic dataset creation."""
        df = create_synthetic_diabetes_dataset()

        # Check dataset structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 768  # Expected sample size

        # Check required columns
        expected_columns = [
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "diabetes_pedigree_function",
            "age",
            "target",
        ]
        assert all(col in df.columns for col in expected_columns)

        # Check data types
        assert df["pregnancies"].dtype in [np.int64, np.float64]
        assert df["glucose"].dtype in [np.float64]
        assert df["target"].dtype in [np.int64, np.bool_]

        # Check value ranges
        assert df["pregnancies"].min() >= 0
        assert df["pregnancies"].max() <= 17
        assert df["glucose"].min() >= 0
        assert df["glucose"].max() <= 200
        assert df["bmi"].min() >= 15
        assert df["bmi"].max() <= 67
        assert df["age"].min() >= 21
        assert df["age"].max() <= 81

        # Check target distribution
        assert df["target"].nunique() == 2
        assert set(df["target"].unique()) == {0, 1}

    def test_save_dataset_splits(self):
        """Test dataset splitting and saving."""
        # Create test dataset
        df = create_synthetic_diabetes_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the data directories
            os.makedirs(os.path.join(temp_dir, "data", "raw"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "data", "processed"), exist_ok=True)

            with (
                patch("data.load_diabetes_data.os.makedirs"),
                patch("data.load_diabetes_data.pd.DataFrame.to_csv") as mock_to_csv,
            ):

                file_paths = save_dataset_splits(df)

                # Check return structure
                assert isinstance(file_paths, dict)
                assert "raw" in file_paths
                assert "train" in file_paths
                assert "test" in file_paths

                # Check that to_csv was called
                assert mock_to_csv.call_count == 3

    def test_generate_data_report(self):
        """Test data quality report generation."""
        df = create_synthetic_diabetes_dataset()

        # Should not raise any exceptions
        generate_data_report(df)

        # Test with missing values
        df_with_missing = df.copy()
        df_with_missing.loc[0:10, "glucose"] = np.nan

        generate_data_report(df_with_missing)


class TestDiabetesPreprocessor:
    """Test diabetes preprocessing functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample diabetes data for testing."""
        np.random.seed(42)
        data = {
            "pregnancies": np.random.randint(0, 10, 100),
            "glucose": np.random.normal(120, 30, 100),
            "blood_pressure": np.random.normal(80, 15, 100),
            "skin_thickness": np.random.exponential(20, 100),
            "insulin": np.random.exponential(80, 100),
            "bmi": np.random.normal(32, 8, 100),
            "diabetes_pedigree_function": np.random.exponential(0.5, 100),
            "age": np.random.randint(21, 81, 100),
            "target": np.random.randint(0, 2, 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def preprocessor(self):
        """Create a fitted preprocessor for testing."""
        return DiabetesPreprocessor()

    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert hasattr(preprocessor, "scaler")
        assert hasattr(preprocessor, "imputer")
        assert hasattr(preprocessor, "feature_names")
        assert preprocessor.is_fitted is False

    def test_clean_data(self, preprocessor, sample_data):
        """Test data cleaning functionality."""
        # Add some zero values that should be converted to NaN
        sample_data.loc[0:5, "glucose"] = 0
        sample_data.loc[10:15, "blood_pressure"] = 0

        cleaned_data = preprocessor.clean_data(sample_data)

        # Check that zeros were converted to NaN for medical features
        assert cleaned_data.loc[0:5, "glucose"].isna().all()
        assert cleaned_data.loc[10:15, "blood_pressure"].isna().all()

        # Check that pregnancies zeros were not converted (valid)
        assert not cleaned_data["pregnancies"].isna().any()

    def test_engineer_features(self, preprocessor, sample_data):
        """Test feature engineering."""
        engineered_data = preprocessor.engineer_features(sample_data)

        # Check that new features were created
        assert "bmi_age_interaction" in engineered_data.columns
        assert "glucose_bmi_ratio" in engineered_data.columns
        assert "high_pregnancy_risk" in engineered_data.columns

        # Check categorical features were encoded
        bmi_category_cols = [
            col for col in engineered_data.columns if "bmi_category" in col
        ]
        age_group_cols = [col for col in engineered_data.columns if "age_group" in col]
        glucose_category_cols = [
            col for col in engineered_data.columns if "glucose_category" in col
        ]

        assert len(bmi_category_cols) > 0
        assert len(age_group_cols) > 0
        assert len(glucose_category_cols) > 0

        # Check interaction features
        assert engineered_data["bmi_age_interaction"].min() >= 0
        assert engineered_data["glucose_bmi_ratio"].min() >= 0

        # Check binary features
        assert set(engineered_data["high_pregnancy_risk"].unique()).issubset({0, 1})

    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit and transform functionality."""
        X = sample_data.drop("target", axis=1)

        # Test fit_transform
        X_transformed = preprocessor.fit_transform(X)

        assert preprocessor.is_fitted
        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == len(X)
        assert X_transformed.shape[1] > X.shape[1]  # More features after engineering

        # Test separate transform
        X_new = X.sample(10, random_state=42)
        X_new_transformed = preprocessor.transform(X_new)

        assert isinstance(X_new_transformed, pd.DataFrame)
        assert len(X_new_transformed) == 10
        assert X_new_transformed.shape[1] == X_transformed.shape[1]

    def test_transform_without_fit_raises_error(self, preprocessor, sample_data):
        """Test that transform without fit raises error."""
        X = sample_data.drop("target", axis=1)

        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(X)

    def test_save_load_preprocessor(self, preprocessor, sample_data):
        """Test saving and loading preprocessor."""
        X = sample_data.drop("target", axis=1)
        preprocessor.fit(X)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            try:
                # Save preprocessor
                preprocessor.save(tmp_file.name)
                assert os.path.exists(tmp_file.name)

                # Load preprocessor
                loaded_preprocessor = DiabetesPreprocessor.load(tmp_file.name)
                assert loaded_preprocessor.is_fitted
                assert loaded_preprocessor.feature_names == preprocessor.feature_names

                # Test that loaded preprocessor works
                X_test = X.sample(5, random_state=42)
                X_transformed_original = preprocessor.transform(X_test)
                X_transformed_loaded = loaded_preprocessor.transform(X_test)

                pd.testing.assert_frame_equal(
                    X_transformed_original, X_transformed_loaded
                )

            finally:
                # Clean up
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)

    def test_missing_values_handling(self, preprocessor, sample_data):
        """Test missing values handling."""
        # Add missing values
        sample_data_missing = sample_data.copy()
        sample_data_missing.loc[0:10, "glucose"] = np.nan
        sample_data_missing.loc[5:15, "bmi"] = np.nan

        X = sample_data_missing.drop("target", axis=1)
        X_transformed = preprocessor.fit_transform(X)

        # Check that missing values were handled
        assert not X_transformed.isna().any().any()

    def test_edge_cases(self, preprocessor):
        """Test edge cases and boundary conditions."""
        # Test with minimum valid data
        edge_data = pd.DataFrame(
            {
                "pregnancies": [0, 1],
                "glucose": [50, 200],
                "blood_pressure": [40, 120],
                "skin_thickness": [0, 50],
                "insulin": [0, 100],
                "bmi": [15, 50],
                "diabetes_pedigree_function": [0.1, 1.0],
                "age": [21, 80],
            }
        )

        X_transformed = preprocessor.fit_transform(edge_data)
        assert len(X_transformed) == 2
        assert not X_transformed.isna().any().any()

        # Test single row
        single_row = edge_data.iloc[[0]]
        X_single = preprocessor.transform(single_row)
        assert len(X_single) == 1


class TestDataValidation:
    """Test data validation and quality checks."""

    def test_feature_ranges(self):
        """Test that generated data meets medical constraints."""
        df = create_synthetic_diabetes_dataset()

        # Medical constraints
        assert df["glucose"].min() >= 0
        assert df["glucose"].max() <= 300
        assert df["blood_pressure"].min() >= 0
        assert df["blood_pressure"].max() <= 200
        assert df["bmi"].min() >= 10
        assert df["bmi"].max() <= 70
        assert df["age"].min() >= 18
        assert df["age"].max() <= 120

    def test_data_consistency(self):
        """Test data consistency across runs."""
        # With same seed, should get consistent results
        df1 = create_synthetic_diabetes_dataset()
        df2 = create_synthetic_diabetes_dataset()

        pd.testing.assert_frame_equal(df1, df2)

    def test_target_distribution(self):
        """Test target class distribution."""
        df = create_synthetic_diabetes_dataset()

        target_counts = df["target"].value_counts()
        total_samples = len(df)

        # Check that both classes are present
        assert len(target_counts) == 2

        # Check that distribution is reasonable (not too imbalanced)
        minority_class_ratio = target_counts.min() / total_samples
        assert minority_class_ratio > 0.2  # At least 20% for minority class


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
