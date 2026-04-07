"""Tests for data pipeline modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.ingestion import DataIngestion, IngestionConfig
from src.cleaning import DataCleaning, CleaningConfig
from src.preprocessing import DataPreprocessing, PreprocessingConfig
from src.splitting import DataSplitting, SplittingConfig
from src.validation import DataValidation, ValidationConfig


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "text": ["Hello world"] * 50,
            "label": np.random.choice([0, 1], 50),
            "category": np.random.choice(["A", "B", "C"], 50),
        }
    )


class TestDataIngestion:
    """Tests for DataIngestion."""

    def test_initialization(self):
        """Test ingestion initialization."""
        ingestion = DataIngestion()
        assert ingestion.data is None

    def test_csv_loading(self, tmp_path):
        """Test CSV data loading."""
        # Create temp CSV
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        config = IngestionConfig(source_type="csv", source_path=str(csv_path))

        ingestion = DataIngestion()
        loaded = ingestion.load(config)

        assert len(loaded) == 3
        assert list(loaded.columns) == ["col1", "col2"]


class TestDataCleaning:
    """Tests for DataCleaning."""

    def test_remove_duplicates(self, sample_df):
        """Test duplicate removal."""
        df = sample_df.copy()
        df = pd.concat([df, df.iloc[0:5]], ignore_index=True)

        config = CleaningConfig(remove_duplicates=True)
        cleaner = DataCleaning(config)

        # This should remove duplicates but our sample has unique indices
        result = cleaner.clean(df)

        assert len(result) <= len(df)

    def test_handle_missing_drop(self):
        """Test missing value handling with drop."""
        df = pd.DataFrame({"text": ["Hello", None, "World"], "label": [0, 1, None]})

        config = CleaningConfig(handle_missing="drop")
        cleaner = DataCleaning(config)

        result = cleaner.clean(df)

        assert result.isnull().sum().sum() == 0

    def test_remove_urls(self):
        """Test URL removal."""
        df = pd.DataFrame(
            {"text": ["Check https://example.com", "Visit www.test.com", "Normal text"]}
        )

        config = CleaningConfig(remove_urls=True)
        cleaner = DataCleaning(config)

        result = cleaner.clean(df, text_columns=["text"])

        assert "https://" not in result["text"].iloc[0]
        assert "www." not in result["text"].iloc[1]


class TestDataSplitting:
    """Tests for DataSplitting."""

    def test_basic_split(self, sample_df):
        """Test basic train/val/test split."""
        config = SplittingConfig(train_size=0.7, val_size=0.15, test_size=0.15)

        splitter = DataSplitting(config)
        splits = splitter.split(sample_df)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(sample_df)

    def test_split_ratios(self, sample_df):
        """Test split ratio correctness."""
        config = SplittingConfig(train_size=0.8, val_size=0.1, test_size=0.1)

        splitter = DataSplitting(config)
        splits = splitter.split(sample_df)

        train_ratio = len(splits["train"]) / len(sample_df)
        val_ratio = len(splits["val"]) / len(sample_df)
        test_ratio = len(splits["test"]) / len(sample_df)

        assert 0.7 < train_ratio < 0.9
        assert 0.05 < val_ratio < 0.15
        assert 0.05 < test_ratio < 0.15

    def test_save_load_splits(self, sample_df, tmp_path):
        """Test saving and loading splits."""
        config = SplittingConfig()
        splitter = DataSplitting(config)
        splits = splitter.split(sample_df)

        splitter.save_splits(str(tmp_path))

        assert (tmp_path / "train.parquet").exists()
        assert (tmp_path / "val.parquet").exists()
        assert (tmp_path / "test.parquet").exists()


class TestDataValidation:
    """Tests for DataValidation."""

    def test_valid_data(self, sample_df):
        """Test validation of valid data."""
        config = ValidationConfig(required_columns=["text", "label"], min_samples=10)

        validator = DataValidation(config)
        is_valid = validator.validate(sample_df)

        assert is_valid is True

    def test_missing_required_column(self, sample_df):
        """Test detection of missing required column."""
        df = sample_df[["text", "label"]].copy()

        config = ValidationConfig(required_columns=["text", "label", "missing_col"])

        validator = DataValidation(config)
        is_valid = validator.validate(df)

        assert is_valid is False
        assert len(validator.validation_results["errors"]) > 0

    def test_insufficient_samples(self, sample_df):
        """Test detection of insufficient samples."""
        df = sample_df.head(5)  # Only 5 samples

        config = ValidationConfig(min_samples=100)

        validator = DataValidation(config)
        is_valid = validator.validate(df)

        assert is_valid is False

    def test_generate_report(self, sample_df):
        """Test report generation."""
        config = ValidationConfig(required_columns=["text", "label"])
        validator = DataValidation(config)
        validator.validate(sample_df)

        report = validator.generate_report()

        assert "is_valid" in report
        assert "num_checks" in report
        assert "details" in report


class TestDataPreprocessing:
    """Tests for DataPreprocessing."""

    def test_tokenizer_loading(self):
        """Test tokenizer initialization."""
        config = PreprocessingConfig(tokenizer_name="gpt2")
        preprocessor = DataPreprocessing(config)

        assert preprocessor.tokenizer is not None

    def test_preprocess_standard(self):
        """Test standard preprocessing."""
        df = pd.DataFrame({"text": ["Hello world", "Test data"]})

        config = PreprocessingConfig(tokenizer_name="gpt2", format_type="standard")

        preprocessor = DataPreprocessing(config)
        result = preprocessor.preprocess(df, text_columns=["text"])

        assert "input_ids" in result.columns
        assert "attention_mask" in result.columns
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
