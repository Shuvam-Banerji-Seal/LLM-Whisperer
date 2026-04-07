"""Data splitting module for train/val/test splits."""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class SplittingConfig:
    """Configuration for data splitting."""

    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42
    stratify_column: Optional[str] = None
    shuffle: bool = True
    cross_validation_folds: Optional[int] = None


class DataSplitting:
    """Data splitting orchestrator."""

    def __init__(self, config: SplittingConfig):
        """Initialize data splitting.

        Args:
            config: Splitting configuration
        """
        self.config = config
        self._validate_config()
        self.splits = {}

    def _validate_config(self):
        """Validate splitting configuration."""
        total = self.config.train_size + self.config.val_size + self.config.test_size

        if not (0.99 <= total <= 1.01):  # Allow small floating point differences
            raise ValueError(
                f"Split sizes must sum to 1.0, got {total}. "
                f"train_size={self.config.train_size}, "
                f"val_size={self.config.val_size}, "
                f"test_size={self.config.test_size}"
            )

    def split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into train/val/test sets.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        logger.info(
            f"Splitting data: {self.config.train_size * 100:.1f}% train, "
            f"{self.config.val_size * 100:.1f}% val, {self.config.test_size * 100:.1f}% test"
        )

        if self.config.cross_validation_folds:
            return self._split_cross_validation(data)

        # Determine stratification
        stratify_array = None
        if self.config.stratify_column and self.config.stratify_column in data.columns:
            stratify_array = data[self.config.stratify_column]

        # First split: separate test set
        if self.config.test_size > 0:
            temp_ratio = self.config.test_size / (1 - 0.0)  # Ratio of test to remaining

            train_val, test = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                shuffle=self.config.shuffle,
                stratify=stratify_array,
            )
        else:
            train_val = data
            test = pd.DataFrame()

        # Second split: separate val from train
        if self.config.val_size > 0 and len(train_val) > 0:
            val_ratio = self.config.val_size / (
                self.config.train_size + self.config.val_size
            )

            if (
                self.config.stratify_column
                and self.config.stratify_column in train_val.columns
            ):
                stratify_array = train_val[self.config.stratify_column]
            else:
                stratify_array = None

            train, val = train_test_split(
                train_val,
                test_size=val_ratio,
                random_state=self.config.random_state,
                shuffle=self.config.shuffle,
                stratify=stratify_array,
            )
        else:
            train = train_val
            val = pd.DataFrame()

        self.splits = {
            "train": train,
            "val": val,
            "test": test,
        }

        self._log_split_stats()

        return self.splits

    def _split_cross_validation(self, data: pd.DataFrame) -> Dict[str, list]:
        """Split data for cross-validation.

        Returns:
            Dictionary with 'folds' containing list of (train, val) tuples
        """
        n_splits = self.config.cross_validation_folds or 5

        if self.config.stratify_column and self.config.stratify_column in data.columns:
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
            stratify_array = data[self.config.stratify_column]
        else:
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state,
            )
            stratify_array = None

        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(data, stratify_array)
            if stratify_array is not None
            else skf.split(data, np.zeros(len(data)))
        ):
            train_fold = data.iloc[train_idx]
            val_fold = data.iloc[val_idx]
            folds.append((train_fold, val_fold))
            logger.info(
                f"Fold {fold_idx + 1}: {len(train_fold)} train, {len(val_fold)} val"
            )

        self.splits = {"folds": folds}
        return self.splits

    def _log_split_stats(self):
        """Log statistics about splits."""
        total = sum(
            len(split)
            for split in self.splits.values()
            if isinstance(split, pd.DataFrame)
        )

        for split_name, split_data in self.splits.items():
            if isinstance(split_data, pd.DataFrame) and len(split_data) > 0:
                percentage = len(split_data) / total * 100 if total > 0 else 0
                logger.info(
                    f"{split_name.upper()}: {len(split_data)} samples ({percentage:.1f}%)"
                )

    def get_split_info(self) -> Dict[str, int]:
        """Get information about current splits."""
        info = {}
        for split_name, split_data in self.splits.items():
            if isinstance(split_data, pd.DataFrame):
                info[split_name] = len(split_data)
            elif split_name == "folds" and isinstance(split_data, list):
                info["num_folds"] = len(split_data)
                for i, (train, val) in enumerate(split_data):
                    info[f"fold_{i}_train_size"] = len(train)
                    info[f"fold_{i}_val_size"] = len(val)

        return info

    def save_splits(self, output_dir: str):
        """Save splits to disk.

        Args:
            output_dir: Directory to save splits
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        for split_name, split_data in self.splits.items():
            if isinstance(split_data, pd.DataFrame) and len(split_data) > 0:
                path = os.path.join(output_dir, f"{split_name}.parquet")
                split_data.to_parquet(path)
                logger.info(f"Saved {split_name} to {path}")

    def load_splits(self, input_dir: str):
        """Load splits from disk.

        Args:
            input_dir: Directory containing split files
        """
        import os

        self.splits = {}
        for split_name in ["train", "val", "test"]:
            path = os.path.join(input_dir, f"{split_name}.parquet")
            if os.path.exists(path):
                self.splits[split_name] = pd.read_parquet(path)
                logger.info(f"Loaded {split_name} from {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    config = SplittingConfig(
        train_size=0.8, val_size=0.1, test_size=0.1, random_state=42
    )

    splitter = DataSplitting(config)

    # Create sample data
    df = pd.DataFrame(
        {
            "text": [f"Sample {i}" for i in range(100)],
            "label": np.random.choice([0, 1], 100),
        }
    )

    splits = splitter.split(df)
    print(splitter.get_split_info())
