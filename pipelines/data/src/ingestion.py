"""Data ingestion module for loading raw data from multiple sources."""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""

    source_type: str  # "csv", "json", "jsonl", "parquet", "huggingface"
    source_path: str
    max_samples: Optional[int] = None
    sample_fraction: Optional[float] = None
    encoding: str = "utf-8"
    cache_dir: Optional[str] = None


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def load(self, config: IngestionConfig) -> pd.DataFrame:
        """Load data from source."""
        pass


class CSVDataSource(DataSource):
    """Load data from CSV files."""

    def load(self, config: IngestionConfig) -> pd.DataFrame:
        """Load CSV data."""
        logger.info(f"Loading CSV from {config.source_path}")

        df = pd.read_csv(
            config.source_path, encoding=config.encoding, nrows=config.max_samples
        )

        if config.sample_fraction:
            df = df.sample(frac=config.sample_fraction, random_state=42)

        logger.info(f"Loaded {len(df)} rows from CSV")
        return df


class JSONDataSource(DataSource):
    """Load data from JSON files."""

    def load(self, config: IngestionConfig) -> pd.DataFrame:
        """Load JSON data."""
        logger.info(f"Loading JSON from {config.source_path}")

        with open(config.source_path, "r", encoding=config.encoding) as f:
            data = json.load(f)

        # Handle list of dicts
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])

        if config.max_samples:
            df = df.head(config.max_samples)

        if config.sample_fraction:
            df = df.sample(frac=config.sample_fraction, random_state=42)

        logger.info(f"Loaded {len(df)} rows from JSON")
        return df


class JSONLDataSource(DataSource):
    """Load data from JSONL (JSON Lines) files."""

    def load(self, config: IngestionConfig) -> pd.DataFrame:
        """Load JSONL data."""
        logger.info(f"Loading JSONL from {config.source_path}")

        data = []
        with open(config.source_path, "r", encoding=config.encoding) as f:
            for i, line in enumerate(f):
                if config.max_samples and i >= config.max_samples:
                    break
                data.append(json.loads(line))

        df = pd.DataFrame(data)

        if config.sample_fraction:
            df = df.sample(frac=config.sample_fraction, random_state=42)

        logger.info(f"Loaded {len(df)} rows from JSONL")
        return df


class ParquetDataSource(DataSource):
    """Load data from Parquet files."""

    def load(self, config: IngestionConfig) -> pd.DataFrame:
        """Load Parquet data."""
        logger.info(f"Loading Parquet from {config.source_path}")

        df = pd.read_parquet(config.source_path)

        if config.max_samples:
            df = df.head(config.max_samples)

        if config.sample_fraction:
            df = df.sample(frac=config.sample_fraction, random_state=42)

        logger.info(f"Loaded {len(df)} rows from Parquet")
        return df


class HuggingFaceDataSource(DataSource):
    """Load data from HuggingFace datasets."""

    def load(self, config: IngestionConfig) -> pd.DataFrame:
        """Load data from HuggingFace Hub."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for HuggingFace source")

        logger.info(f"Loading HuggingFace dataset: {config.source_path}")

        dataset = load_dataset(
            config.source_path, cache_dir=config.cache_dir, split="train"
        )

        df = dataset.to_pandas()

        if config.max_samples:
            df = df.head(config.max_samples)

        if config.sample_fraction:
            df = df.sample(frac=config.sample_fraction, random_state=42)

        logger.info(f"Loaded {len(df)} rows from HuggingFace")
        return df


class DataIngestion:
    """Main data ingestion orchestrator."""

    sources = {
        "csv": CSVDataSource,
        "json": JSONDataSource,
        "jsonl": JSONLDataSource,
        "parquet": ParquetDataSource,
        "huggingface": HuggingFaceDataSource,
    }

    def __init__(self):
        """Initialize data ingestion."""
        self.data: Optional[pd.DataFrame] = None

    def load(self, config: IngestionConfig) -> pd.DataFrame:
        """Load data from specified source.

        Args:
            config: Ingestion configuration

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If source type is not supported
        """
        if config.source_type not in self.sources:
            raise ValueError(
                f"Unknown source type: {config.source_type}. "
                f"Supported: {list(self.sources.keys())}"
            )

        source_class = self.sources[config.source_type]
        source = source_class()

        self.data = source.load(config)
        logger.info(f"Data shape: {self.data.shape}")

        return self.data

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        stats = {
            "num_rows": len(self.data),
            "num_columns": len(self.data.columns),
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
            "missing_values": self.data.isnull().sum().to_dict(),
        }

        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    config = IngestionConfig(
        source_type="csv", source_path="data/raw/sample.csv", max_samples=1000
    )

    ingestion = DataIngestion()
    data = ingestion.load(config)
    print(ingestion.get_statistics())
