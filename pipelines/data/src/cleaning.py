"""Data cleaning module for text validation and sanitization."""

import logging
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning."""

    remove_duplicates: bool = True
    duplicate_subset: Optional[List[str]] = None
    handle_missing: str = "drop"  # "drop", "fill"
    missing_fill_value: Optional[str] = None
    lowercase: bool = False
    remove_special_chars: bool = False
    remove_extra_whitespace: bool = True
    min_length: int = 0
    max_length: Optional[int] = None
    detect_language: bool = False
    target_languages: Optional[List[str]] = None
    remove_urls: bool = False
    remove_html: bool = False


class DataCleaning:
    """Data cleaning and validation."""

    def __init__(self, config: CleaningConfig):
        """Initialize data cleaning.

        Args:
            config: Cleaning configuration
        """
        self.config = config
        self.cleaning_stats = {}

    def clean(
        self, data: pd.DataFrame, text_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Clean data.

        Args:
            data: Input DataFrame
            text_columns: List of columns to clean as text

        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        initial_rows = len(df)

        # Remove duplicates
        if self.config.remove_duplicates:
            df = self._remove_duplicates(df)

        # Handle missing values
        if self.config.handle_missing == "drop":
            df = df.dropna()
        elif self.config.handle_missing == "fill":
            df = df.fillna(self.config.missing_fill_value or "")

        # Clean text columns
        if text_columns:
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self._clean_text)

        # Filter by length
        if text_columns:
            df = self._filter_by_length(df, text_columns)

        # Detect language if needed
        if self.config.detect_language and text_columns:
            df = self._filter_by_language(df, text_columns)

        final_rows = len(df)
        self.cleaning_stats = {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "removed_rows": initial_rows - final_rows,
            "removal_percentage": (initial_rows - final_rows) / initial_rows * 100
            if initial_rows > 0
            else 0,
        }

        logger.info(f"Cleaning complete: {initial_rows} → {final_rows} rows")
        logger.info(
            f"Removal percentage: {self.cleaning_stats['removal_percentage']:.2f}%"
        )

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial = len(df)

        if self.config.duplicate_subset:
            df = df.drop_duplicates(subset=self.config.duplicate_subset, keep="first")
        else:
            df = df.drop_duplicates(keep="first")

        removed = initial - len(df)
        logger.info(f"Removed {removed} duplicate rows")

        return df

    def _clean_text(self, text: str) -> str:
        """Clean individual text string."""
        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        if self.config.remove_html:
            text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r"http\S+|www\S+", "", text)

        # Remove special characters
        if self.config.remove_special_chars:
            text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-\']", "", text)

        # Remove extra whitespace
        if self.config.remove_extra_whitespace:
            text = " ".join(text.split())

        # Lowercase
        if self.config.lowercase:
            text = text.lower()

        return text

    def _filter_by_length(
        self, df: pd.DataFrame, text_columns: List[str]
    ) -> pd.DataFrame:
        """Filter rows by text length."""
        initial = len(df)

        for col in text_columns:
            if col in df.columns:
                df["_length"] = df[col].astype(str).str.len()

                if self.config.min_length > 0:
                    df = df[df["_length"] >= self.config.min_length]

                if self.config.max_length:
                    df = df[df["_length"] <= self.config.max_length]

                df = df.drop("_length", axis=1)

        removed = initial - len(df)
        if removed > 0:
            logger.info(f"Filtered {removed} rows by length constraints")

        return df

    def _filter_by_language(
        self, df: pd.DataFrame, text_columns: List[str]
    ) -> pd.DataFrame:
        """Filter rows by detected language."""
        try:
            from langdetect import detect, LangDetectException
        except ImportError:
            logger.warning("langdetect not installed, skipping language detection")
            return df

        if not self.config.target_languages:
            return df

        initial = len(df)

        def detect_language(text: str) -> Optional[str]:
            try:
                return detect(str(text))
            except LangDetectException:
                return None

        # Use first text column for language detection
        primary_col = text_columns[0]
        df["_language"] = df[primary_col].apply(detect_language)

        df = df[df["_language"].isin(self.config.target_languages)]
        df = df.drop("_language", axis=1)

        removed = initial - len(df)
        logger.info(f"Filtered {removed} rows by language")

        return df

    def get_statistics(self) -> Dict[str, Any]:
        """Get cleaning statistics."""
        return self.cleaning_stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    config = CleaningConfig(
        remove_duplicates=True,
        handle_missing="drop",
        remove_urls=True,
        remove_html=True,
        min_length=10,
    )

    cleaner = DataCleaning(config)

    # Create sample data
    df = pd.DataFrame(
        {
            "text": [
                "Hello world",
                "Hello world",
                "Short",
                "<html>Test</html>",
                "Visit https://example.com",
                np.nan,
            ]
        }
    )

    cleaned = cleaner.clean(df, text_columns=["text"])
    print(cleaner.get_statistics())
    print(cleaned)
